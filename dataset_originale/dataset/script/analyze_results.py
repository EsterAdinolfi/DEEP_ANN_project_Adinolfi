import json
import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# --- CONFIGURAZIONE ---
DIR_FILE = "risultati/" # Cartella dove si trovano i risultati JSON

INPUT_FILE = "results_pythia_160m.json"
OUTPUT_CSV = "analysis_metrics_pythia_160m.csv"

PATH_FILE_INPUT = "../"+DIR_FILE+INPUT_FILE
PATH_FILE_OUTPUT = "../"+DIR_FILE+OUTPUT_CSV

# ==========================================
# UTILITIES
# ==========================================

def load_data():
    if not os.path.exists(PATH_FILE_INPUT):
        print(f"Errore: {INPUT_FILE} non trovato.")
        return []
    with open(PATH_FILE_INPUT, 'r') as f:
        return json.load(f)

def get_option_content(choice_char, options_order):
    """Restituisce il testo dell'opzione data la lettera (A->0, B->1)."""
    if not choice_char: return None
    idx = ord(choice_char) - 65
    if 0 <= idx < len(options_order):
        return options_order[idx]
    return None

def align_distributions(llm_probs_map, human_dist_map, options_order):
    """
    Allinea le due distribuzioni (modello vs umano) in due vettori ordinati.
    Fondamentale perché il modello usa 'A','B' e l'umano usa 'Yes','No'.
    """
    llm_vec = []
    human_vec = []
    
    # Iteriamo sull'ordine canonico delle opzioni per questo trial
    for idx, opt_text in enumerate(options_order):
        # 1. Recupera probabilità modello (Letter -> 'A', 'B'...)
        char_label = chr(65 + idx) # 0->A, 1->B
        p_llm = llm_probs_map.get(char_label, 0.0)
        
        # 2. Recupera probabilità umano (Text -> 'Yes', 'No'...)
        # Nota: human_dist_map usa le chiavi testuali originali
        p_human = human_dist_map.get(opt_text, 0.0)
        
        llm_vec.append(p_llm)
        human_vec.append(p_human)
        
    # Normalizziamo per sicurezza (JSD richiede somma=1)
    llm_vec = np.array(llm_vec)
    human_vec = np.array(human_vec)
    
    # Evitiamo divisioni per zero normalizzando solo se somma > 0
    if llm_vec.sum() > 0: llm_vec = llm_vec / llm_vec.sum()
    if human_vec.sum() > 0: human_vec = human_vec / human_vec.sum()
        
    return llm_vec, human_vec

# ==========================================
# FUNZIONI METRICHE 
# ==========================================

# 1. CAPABILITY (testuale)
def calculate_capability(text, options_order):
    """1 se il testo inizia con lettera valida o ripete opzione, 0 altrimenti."""
    if not text: return 0
    text = text.strip()
    valid_chars = [chr(65 + i) for i in range(len(options_order))]
    
    # Check Lettera
    if text[0].upper() in valid_chars: return 1
    # Check Contenuto
    for opt in options_order:
        if text.startswith(opt): return 1
    return 0

# 2. ALIGNMENT (testuale vs logprobs)
def calculate_alignment(text, top_prob_char, options_order):
    """1 se il testo generato conferma la scelta matematica."""
    if not text or not top_prob_char: return 0
    text = text.strip().upper()
    
    # Check Lettera diretta
    if text.startswith(top_prob_char): return 1
    
    # Check Contenuto Semantico
    target_content = get_option_content(top_prob_char, options_order)
    if target_content and target_content.upper() in text: return 1
    return 0

# 3. POSITION BIAS
def calculate_position_bias(label, probs_char, base_char, curr_content, base_content):
    """1 se sceglie la stessa lettera ('A') nonostante il contenuto sia diverso."""
    if label != 'permutation': return None # Non applicabile
    if (probs_char == base_char) and (curr_content != base_content):
        return 1
    return 0

# 4. ENTROPIA (incertezza) 
def calculate_entropy_metric(llm_probs_map):
    """
    Calcola l'entropia di Shannon della distribuzione del modello.
    Valori alti = Incertezza (Uniforme). Valori bassi = Sicurezza (Picco).
    """
    if not llm_probs_map: return None
    probs = list(llm_probs_map.values())
    return entropy(probs, base=2) # Base 2 per misurare in bit

# 5. DISTANZA DALLA DISTRIBUZIONE UMANA (JSD) 
def calculate_human_distance(llm_probs_map, human_dist_map, options_order):
    """
    Calcola la Jensen-Shannon Divergence (JSD) tra modello e umani.
    0.0 = Identici. 1.0 = Completamente diversi.
    """
    if not human_dist_map or not llm_probs_map: return None
    
    # Allineiamo i vettori (es. [Prob_Yes, Prob_No])
    llm_vec, human_vec = align_distributions(llm_probs_map, human_dist_map, options_order)
    
    # Calcolo JSD (radice quadrata della divergenza per averla come metrica di distanza)
    # Santurkar usa JSD standard. Scipy restituisce la distanza (sqrt(div)), che è tra 0 e 1.
    return jensenshannon(llm_vec, human_vec, base=2)


# ==========================================
# MAIN ANALYSIS LOOP
# ==========================================

def run_analysis():
    data = load_data()
    if not data: return

    rows = []
    print(f"Analisi di {len(data)} domande con le metriche definite.")

    for entry in data:
        topic = entry['topic']
        question_id = entry['id_question']
        human_dist = entry['human_dist_total']
        
        # Baseline reference
        baseline_exp = next((e for e in entry['experiments'] if e['label'] == 'baseline'), None)
        if not baseline_exp: continue
        
        base_choice_char = baseline_exp.get('llm_top_choice') 
        base_content = get_option_content(base_choice_char, baseline_exp['options_order'])
        
        for exp in entry['experiments']:
            label = exp['label']
            
            # Dati grezzi esperimento
            text = exp.get('llm_generated_text', '')
            probs_char = exp.get('llm_top_choice')
            probs_map = exp.get('llm_choice_probs', {})
            curr_options = exp['options_order']
            curr_content = get_option_content(probs_char, curr_options)

            # --- CALCOLO METRICHE ---
            
            # 1. Capability
            metric_capability = calculate_capability(text, curr_options)
            
            # 2. Consistency (semantica vs baseline)
            is_consistent = (base_content == curr_content) if (base_content and curr_content) else False
            metric_consistency = 1 if is_consistent else 0
            
            # 3. Alignment (testuale vs logprobs)
            metric_alignment = calculate_alignment(text, probs_char, curr_options)
            
            # 4. Position bias (solo permutation)
            metric_pos_bias = calculate_position_bias(label, probs_char, base_choice_char, curr_content, base_content)
            
            # 5. Entropy (incertezza)
            metric_entropy = calculate_entropy_metric(probs_map)
            
            # 6. Human Distance (JSD)
            # Nota: confrontiamo sempre la distribuzione attuale con quella umana "totale"
            metric_human_dist = calculate_human_distance(probs_map, human_dist, curr_options)

            # Salvataggio riga
            rows.append({
                "Question_ID": question_id,
                "Topic": topic,
                "Condition": label,
                "Capability_Score": metric_capability,
                "Consistency_Score": metric_consistency,
                "Alignment_Score": metric_alignment,
                "Position_Bias": metric_pos_bias, # può essere None
                "Entropy": metric_entropy,
                "Human_Distance_JSD": metric_human_dist,
                "Top_Choice_Prob": max(probs_map.values()) if probs_map else 0
            })

    # Creazione dataframe e salvataggio
    df = pd.DataFrame(rows)
    df.to_csv(PATH_FILE_OUTPUT, index=False)
    
    # --- REPORT RAPIDO ---
    print("\n--- REPORT METRICHE AVANZATE ---")
    
    print("\n1. Entropy (incertezza media - più basso è più sicuro):")
    print(df.groupby('Condition')['Entropy'].mean())
    
    print("\n2. Human Distance (JSD - 0=Simile, 1=Diverso):")
    print(df.groupby('Condition')['Human_Distance_JSD'].mean())
    
    print(f"\nAnalisi completata! File CSV: {PATH_FILE_OUTPUT}")

if __name__ == "__main__":
    run_analysis()