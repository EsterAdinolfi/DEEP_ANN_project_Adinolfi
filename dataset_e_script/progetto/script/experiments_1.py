import json
import torch
import os
import torch.nn.functional as F # per softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

# --- CONFIGURAZIONE ---
RANDOM_SEED = 42

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

def set_seed(seed=RANDOM_SEED):
    """Funzione che fissa il seed per garantire la riproducibilità totale degli esperimenti hardware."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    
    # 2. Forward pass (senza generazione, non deve allenare il modello) per ottenere i logits dell'ultimo token
    with torch.no_grad():
        outputs = model(**inputs)
        # Prendiamo i logits dell'ultimo token della sequenza input
        # logits = [elemento del batch, sequenza, vocab_size]
        #  Batch: numero di frasi che diamo al modello contemporaneamente. 0 = prende solo la prima frase del gruppo grande 1
        #  Sequenza: posizione del token nella frase. -1 = ultimo token
        #  Vocab_size: dimensione del vocabolario del modello
        last_token_logits = outputs.logits[0, -1, :]

    # 3. Identifichiamo i token ID per le etichette valide (" A", " B", ecc.)
    # Nota: Pythia/GPT spesso aggiungono uno spazio prima del token (es. " A")
    # Proviamo sia con spazio che senza per sicurezza
    choice_stats = {}
    
    for label in valid_labels:
        # Cerchiamo l'ID del token. 
        # ATTENZIONE: Tokenizer diversi gestiscono gli spazi diversamente.
        # add_special_tokens=False evita di aggiungere token speciali come [CLS], [SEP], ecc. che potrebbero restituire più numeri di token invece di uno solo. Es: [ BOS, " A" ] -> [1, 32] anzichè solo 32 corrispondente ad " A"
        # Per sicurezza proviamo " A" (con spazio) che è tipico dopo "Answer:"
        token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if not token_ids:
             # Fallback senza spazio
             token_ids = tokenizer.encode(label, add_special_tokens=False)
        
        # Prendiamo l'ultimo ID
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


def run_inference(model_name=MODEL_NAME, random_seed=RANDOM_SEED):
    # Fissa il seed 
    set_seed(random_seed)
    
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
            
            # C. Generazione con beam search
            # Usiamo torch.no_grad() per risparmiare memoria
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, 
                    output_scores=True, # per i logprobs
                    # quando poi si deve aggiungere la temperatura si deve mettere qua
                    num_beams=5,             # Attiva il Beam Search mantenendo le 5 migliori strade
                    length_penalty=1.0,      # Così non si penalizzano frasi lunghe ma sicure → 1.0 = normalizzazione lineare
                    early_stopping=True      
                )

            # D. Decoding risposta
            # Il Beam Search metterà automaticamente in outputs.sequences[0] la frase vincente
            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # E. Calcolo dei logprobs (score di confidenza nella generazione)
            # Utile per vedere se il modello è "sicuro" 
            # Con il beam search, Hugging Face calcola già lo score finale applicando la formula:
            # Somma dei LogProbs / (Lunghezza_Sequenza ** length_penalty)
            # Avendo messo length_penalty=1.0, questo valore è già la media perfetta
            try:
                generated_confidence = outputs.sequences_scores[0].item()
            except Exception as e:
                print(f"Errore nell'estrazione dello score: {e}")
                generated_confidence = 0.0 # Fallback

            # F. Salvataggio nel dizionario dei risultati
            experiment['llm_generated_text'] = answer_text
            experiment['llm_generated_confidence'] = round(generated_confidence, 4)

            # 4. Salvataggio incrementale su file → aggiornato dopo ogni domanda per evitare perdite di dati in caso di crash
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
    
    print(f"\n[FINE] Esperimenti completati! Risultati salvati in: {output_path}")

    # Pulizia memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Esegue esperimenti con un LLM")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f"Nome del modello da utilizzare (default: {MODEL_NAME})")
    args = parser.parse_args()
    
    run_inference(model_name=args.model_name)