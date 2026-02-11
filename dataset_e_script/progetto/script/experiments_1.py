import json
import torch
import os
import torch.nn.functional as F # per softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISULTATI_DIR = os.path.join(BASE_DIR, "risultati")

MODEL_NAME = "EleutherAI/pythia-160m"
INPUT_FILE = os.path.join(RISULTATI_DIR, "operational.json")
MAX_NEW_TOKENS = 50 # Numero massimo di token generati

def load_model(model_name=MODEL_NAME):
    """Carica modello e tokenizer gestendo il dispositivo hardware."""
    print(f"Caricamento modello: {model_name}...")
    
    # Rilevamento automatico device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Device rilevato: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Imposta il pad_token se manca (Pythia ne ha bisogno)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer, device

def format_prompt(experiment_data, question_text):
    """
    Costruisce il prompt in formato Q&A.
    Gestisce la minaccia (usa threaten_question se presente) e l'ordine e la duplicazione delle opzioni.
    """
    # 1. Determina il testo della domanda (normale o minacciata)
    if experiment_data.get('is_threatened') and experiment_data.get('threaten_question'):
        q_text = experiment_data['threaten_question']
    else:
        q_text = question_text

    # 2. Costruisce il blocco opzioni in base all'esperimento. Prendiamo SEMPRE la lista 'options_order' specifica di questo esperimento, perchè è già stato gestito in precedenza in helpful_scripts.py
    current_options = experiment_data['options_order']
    
    # Mappiamo le opzioni a lettere: A, B, C, D...
    options_block = ""
    labels = []
    # Generiamo etichette dinamiche (A, B, C...) in base al numero di opzioni → utile per quando si hanno le duplicazioni
    for idx, opt in enumerate(current_options):
        label = chr(65 + idx) # 65 è 'A' ASCII
        options_block += f"{label}. {opt}\n"
        labels.append(label)

    # 3. Assemblaggio del prompt finale
    prompt = f"Question: {q_text}\nOptions:\n{options_block}Answer:"
    
    return prompt, labels

def get_choice_logprobs(model, tokenizer, device, prompt, valid_labels):
    """
    Calcola la probabilità del prossimo token ristretta alle opzioni valide (A, B, C...).
    """
    # 1. Tokenizziamo il prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 2. Forward pass (senza generazione) per ottenere i logits dell'ultimo token
    with torch.no_grad():
        outputs = model(**inputs)
        # Prendiamo i logits dell'ultimo token della sequenza input
        # Shape: [batch_size, vocab_size]
        last_token_logits = outputs.logits[0, -1, :]

    # 3. Identifichiamo i token ID per le etichette valide (" A", " B", ecc.)
    # Nota: Pythia/GPT spesso aggiungono uno spazio prima del token (es. " A")
    # Proviamo sia con spazio che senza per sicurezza
    choice_stats = {}
    
    for label in valid_labels:
        # Cerchiamo l'ID del token. 
        # ATTENZIONE: Tokenizer diversi gestiscono gli spazi diversamente.
        # Per sicurezza proviamo " A" (con spazio) che è tipico dopo "Answer:"
        token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if not token_ids:
             # Fallback senza spazio
             token_ids = tokenizer.encode(label, add_special_tokens=False)
        
        # Prendiamo l'ultimo ID (in caso il tokenizer rompa la parola, ma per lettere singole è ok)
        target_id = token_ids[-1]
        
        # Estraiamo il logit grezzo per questo token
        logit = last_token_logits[target_id].item()
        choice_stats[label] = logit

    # 4. Normalizzazione (Softmax sui soli token di interesse)
    # P(A) = exp(logit_A) / sum(exp(logit_X) for X in valid_labels)
    logits_tensor = torch.tensor(list(choice_stats.values()))
    probs_tensor = F.softmax(logits_tensor, dim=0)
    
    # Creiamo dizionario finale { "A": 0.45, "B": 0.12 ... }
    probs_map = {label: round(prob.item(), 4) for label, prob in zip(choice_stats.keys(), probs_tensor)}
    
    # Restituiamo anche la label con probabilità massima (la "scelta" del modello)
    best_choice = max(probs_map, key=probs_map.get)
    
    return best_choice, probs_map


def run_inference(model_name=MODEL_NAME):
    # 1. Carica il dataset operativo
    if not os.path.exists(INPUT_FILE):
        print(f"[ERRORE] File non trovato in: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)

    # 2. Carica il modello
    model, tokenizer, device = load_model(model_name)
    model.eval() 

    # --- GESTIONE CARTELLE DINAMICA ---
    # 1. Puliamo il nome: da "EleutherAI/pythia-160m" a "pythia_160m"
    # Sostituiamo anche i trattini con underscore
    pure_name = model_name.split("/")[-1].replace("-", "_")
    
    # 2. Creiamo il percorso per la sottocartella specifica
    # Es: .../risultati/pythia_160m
    model_output_dir = os.path.join(RISULTATI_DIR, pure_name)
    os.makedirs(model_output_dir, exist_ok=True) # Crea la cartella se non esiste

    # 3. Definiamo il file di output dentro quella cartella
    output_filename = f"results_{pure_name}.json"
    output_path = os.path.join(model_output_dir, output_filename)
    # ---------------------------------------------

    print(f"Avvio esperimenti per {pure_name}, su {len(dataset)} domande...")
    print(f"I risultati verranno salvati in: {output_path}")

    # 3. Ciclo principale (con barra di caricamento)
    for entry in tqdm(dataset, desc="Processing questions"):
        neutral_question = entry['question']
        
        # Iteriamo su tutti gli esperimenti della domanda
        for experiment in entry['experiments']:
            
            # A. Costruiamo il prompt specifico per questo trial
            prompt, valid_labels = format_prompt(experiment, neutral_question)

            # B.1 Calcoliamo la scelta del modello e la distribuzione di probabilità
            top_choice_char, distribution_map = get_choice_logprobs(
                model, tokenizer, device, prompt, valid_labels
            )
            
            # Salviamo l'intenzione matematica del modello
            experiment['llm_choice_probs'] = distribution_map
            experiment['llm_top_choice'] = top_choice_char # Es. "A"
            
            # B.2 Tokenizzazione. Lasciamo che il modello scriva per vedere se segue le istruzioni
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # C. Generazione
            # Usiamo torch.no_grad() per risparmiare memoria
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, 
                    output_scores=True, # per i logprobs
                    # quando poi si deve aggiungere la temperatura si deve mettere qua
                )

            # D. Decoding risposta
            # Estraiamo solo i nuovi token generati
            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # E. Calcolo dei logprobs (score di confidenza nella generazione)
            # Sommiamo i logprobs dei token generati per avere uno score della frase
            # Utile per vedere se il modello è "sicuro" nel caso in cui delira o rifiuta
            try:
                # transition_scores calcola i logprobs dei token generati
                transition_scores = model.compute_transition_scores(
                    outputs.sequences, 
                    outputs.scores, 
                    normalize_logits=True
                )
                # Somma dei logprobs (più è alto/meno negativo, più è sicuro)
                generated_confidence = torch.sum(transition_scores).item()
            except:
                generated_confidence = 0.0 # Fallback

            # F. Salvataggio nel dizionario dei risultati
            experiment['llm_generated_text'] = answer_text
            experiment['llm_generated_confidence'] = round(generated_confidence, 4)
            
            # Debug rapido (opzionale)
            # print(f"\nPrompt: {prompt}\nAns: {answer_text}")

    # 4. Salvataggio finale su file
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n[FINE] Esperimenti completati! Risultati salvati in: {output_path}")

    # Pulizia memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    run_inference(model_name=MODEL_NAME)