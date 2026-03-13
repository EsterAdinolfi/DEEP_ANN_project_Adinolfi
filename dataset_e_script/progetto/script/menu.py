"""
menu.py — Sistema di gestione ed esecuzione del progetto

Modalità d'uso:
    1. Esecuzione sequenziale automatica:
       python menu.py [--update]
       
    2. Menu interattivo:
       python menu.py --menu

Flags:
    --menu    : Attiva il menu interattivo (la modalità update viene scelta dentro il menu)
    --update  : Forza il ricalcolo e la sovrascrittura dei file esistenti (solo per modalità automatica)
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE GLOBALE
# ══════════════════════════════════════════════════════════════════════

# Directory base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(BASE_DIR, "script")
RISULTATI_DIR = os.path.join(BASE_DIR, "risultati")
HUMAN_RESP_DIR = os.path.join(os.path.dirname(BASE_DIR), "human_resp")

# Modelli di linguaggio disponibili
AVAILABLE_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]

# Percorsi degli script
SCRIPTS = {
    'generate_mapping': os.path.join(SCRIPT_DIR, 'generate_mapping.py'),
    'initialization': os.path.join(SCRIPT_DIR, 'initialization.py'),
    'experiments': os.path.join(SCRIPT_DIR, 'experiments_1.py'),
    'analyze': os.path.join(SCRIPT_DIR, 'analyze.py'),
    'visualize': os.path.join(SCRIPT_DIR, 'visualize.py'),
}

# File di output previsti
EXPECTED_FILES = {
    'mapping': os.path.join(HUMAN_RESP_DIR, 'question_mapping.json'),
    'operational': os.path.join(RISULTATI_DIR, 'operational.json'),
    'human_truth': os.path.join(RISULTATI_DIR, 'human_source_of_truth.csv'),
}

# File requirements.txt
REQUIREMENTS_FILE = os.path.join(BASE_DIR, 'requirements.txt')

# Tutte le figure attese per modello (23 in totale)
ALL_MODEL_FIGURES = [
    "fig0_summary_table.png",
    "fig1a_validity.png",
    "fig1b_log_coherence.png",
    "fig1c_jsd.png",
    "fig2a_perm_stability.png",
    "fig2b_position_bias.png",
    "fig2c_primacy_recency.png",
    "fig2d_cognitive_map.png",
    "fig3a_dup_stability.png",
    "fig3b_dup_cognitive_map.png",
    "fig4a_threat_resistance.png",
    "fig4b_threat_validity.png",
    "fig4c_threat_log_coherence.png",
    "fig4d_threat_jsd.png",
    "fig4e_most_disruptive_threat.png",
    "fig4f_most_effective_validity.png",
    "fig4g_most_effective_consistency.png",
    "fig5a_alignment_human.png",
    "fig5b_alignment_human_by_area.png",
    "fig6a_validity_by_area.png",
    "fig6b_log_consistency_by_area.png",
    "fig6c_political_compass.png",
    "fig6d_wd_heatmap.png",
]

# Mappa ogni figura al gruppo che la genera (per la rigenerazione selettiva)
FIGURE_TO_GROUP = {
    "fig0_summary_table.png":           "summary",
    "fig1a_validity.png":               "validity",
    "fig1b_log_coherence.png":          "coherence",
    "fig1c_jsd.png":                    "robustness",
    "fig2a_perm_stability.png":         "permutation",
    "fig2b_position_bias.png":          "permutation",
    "fig2c_primacy_recency.png":        "permutation",
    "fig2d_cognitive_map.png":          "cognitive_map",
    "fig3a_dup_stability.png":          "duplication",
    "fig3b_dup_cognitive_map.png":      "duplication",
    "fig4a_threat_resistance.png":      "threat",
    "fig4b_threat_validity.png":        "threat",
    "fig4c_threat_log_coherence.png":   "threat",
    "fig4d_threat_jsd.png":             "threat",
    "fig4e_most_disruptive_threat.png": "threat",
    "fig4f_most_effective_validity.png":"threat",
    "fig4g_most_effective_consistency.png": "threat",
    "fig5a_alignment_human.png":        "alignment",
    "fig5b_alignment_human_by_area.png":"alignment",
    "fig6a_validity_by_area.png":       "political",
    "fig6b_log_consistency_by_area.png":"political",
    "fig6c_political_compass.png":      "political",
    "fig6d_wd_heatmap.png":             "political",
}

# Tutte le figure comparative attese
ALL_COMPARATIVE_FIGURES = [
    "comp_jsd_violins.png",
    "comp_perm_cognitive_maps.png",
    "comp_dup_cognitive_maps.png",
    "comp_position_bias.png",
    "comp_primacy_recency.png",
    "comp_alignment.png",
    "comp_political_compass.png",
    "comp_wd_heatmaps.png",
    "comp_threats_stacked.png",
    "comp_leadership_alignment.png",
    "comp_areas_alignment.png",
]


# ══════════════════════════════════════════════════════════════════════
#  GESTIONE DIPENDENZE
# ══════════════════════════════════════════════════════════════════════

def install_dependencies():
    """
    Installa le dipendenze dal requirements.txt.
    """
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"⚠ File requirements.txt non trovato in: {REQUIREMENTS_FILE}")
        print("  Impossibile procedere con l'installazione.\n")
        return False
    
    print(f"\n📦 Installazione dipendenze da: {os.path.basename(REQUIREMENTS_FILE)}")
    print("   Questo potrebbe richiedere alcuni minuti...\n")
    
    # Installa le dipendenze
    print("⏳ Installazione in corso...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS_FILE, '--upgrade'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minuti max
        )
        
        if result.returncode == 0:
            print("✓ Dipendenze installate con successo!\n")
            print("⚠ IMPORTANTE: Riavvia il menu per caricare le nuove dipendenze.\n")
            return True
        else:
            print("✗ Errore durante l'installazione delle dipendenze:")
            if result.stderr:
                # Mostra solo le ultime righe dell'errore
                error_lines = result.stderr.strip().split('\n')
                print('\n'.join(error_lines[-10:]))
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout durante l'installazione (>5 minuti). Riprova manualmente con:")
        print(f"   pip install -r {REQUIREMENTS_FILE}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Installazione interrotta dall'utente.")
        return False
    except Exception as e:
        print(f"✗ Errore durante l'installazione: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
#  FUNZIONI HELPER
# ══════════════════════════════════════════════════════════════════════

def print_header(text):
    """Stampa un'intestazione formattata."""
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70 + "\n")


def print_step(step_num, total_steps, description):
    """Stampa il progresso di uno step."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 70)


def check_file_exists(filepath):
    """Controlla se un file esiste."""
    return os.path.exists(filepath)


def get_model_name_clean(model_path):
    """Estrae il nome pulito del modello dal path (es. 'EleutherAI/pythia-160m' -> 'pythia_160m')."""
    return model_path.split("/")[-1].replace("-", "_")


def get_results_file(model_name):
    """Restituisce il percorso del file results per un dato modello."""
    clean_name = get_model_name_clean(model_name)
    # Il file è salvato in risultati/{clean_name}/results_{clean_name}.json
    return os.path.join(RISULTATI_DIR, clean_name, f"results_{clean_name}.json")


def get_analysis_files(model_name):
    """Restituisce i percorsi dei file di analisi per un dato modello."""
    clean_name = get_model_name_clean(model_name)
    model_dir = os.path.join(RISULTATI_DIR, clean_name)
    metrics = os.path.join(model_dir, f"analysis_metrics_{clean_name}.csv")
    report = os.path.join(model_dir, f"report_topic_{clean_name}.csv")
    return metrics, report


def run_command(command, description, show_live_output=False):
    """
    Esegue un comando nel terminale e gestisce eventuali errori.
    
    Args:
        command: Lista dei comandi da eseguire
        description: Descrizione dell'operazione
        show_live_output: Se True, mostra l'output in tempo reale (per script lunghi)
    """
    print(f"\n▸ {description}")
    print(f"  Comando: {' '.join(command)}")
    
    if show_live_output:
        print(f"\n{'─' * 70}")
        print("OUTPUT IN TEMPO REALE:")
        print(f"{'─' * 70}\n")
    
    try:
        if show_live_output:
            # Mostra output in tempo reale (per script lunghi)
            result = subprocess.run(command, check=True, text=True)
            print(f"\n{'─' * 70}")
            print(f"✓ {description} completato con successo")
            print(f"{'─' * 70}")
            return True
        else:
            # Cattura output e mostra alla fine (per script veloci)
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            print(f"✓ {description} completato con successo")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ERRORE durante: {description}")
        print(f"  Codice uscita: {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  Errore: {e.stderr}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠ Operazione interrotta dall'utente: {description}")
        return False


# ══════════════════════════════════════════════════════════════════════
#  ESECUZIONE SINGOLI SCRIPT
# ══════════════════════════════════════════════════════════════════════

def run_generate_mapping(force_update=False):
    """Esegue generate_mapping.py se necessario."""
    mapping_file = EXPECTED_FILES['mapping']
    
    if check_file_exists(mapping_file) and not force_update:
        print(f"✓ File mapping già presente: {mapping_file}")
        print("  Salto generate_mapping.py")
        return True
    
    print(f"▸ Generazione mapping...")
    return run_command(
        [sys.executable, SCRIPTS['generate_mapping']],
        "Generazione question_mapping.json"
    )


def run_initialization(force_update=False):
    """Esegue initialization.py se necessario."""
    operational = EXPECTED_FILES['operational']
    human_truth = EXPECTED_FILES['human_truth']
    
    if all(check_file_exists(f) for f in [operational, human_truth]) and not force_update:
        print(f"✓ File di inizializzazione già presenti:")
        print(f"  • {operational}")
        print(f"  • {human_truth}")
        print("  Salto initialization.py")
        return True
    
    print(f"▸ Inizializzazione dataset...")
    return run_command(
        [sys.executable, SCRIPTS['initialization']],
        "Inizializzazione dataset"
    )


def run_experiments(model_name, force_update=False):
    """Esegue experiments_1.py per un modello specifico."""
    results_file = get_results_file(model_name)
    
    if check_file_exists(results_file) and not force_update:
        print(f"✓ Risultati già presenti per {model_name}: {results_file}")
        print("  Salto experiments_1.py")
        return True
    
    print(f"▸ Esecuzione esperimenti per {model_name}...")
    
    # Passa il model name come argomento - mostra output in tempo reale
    return run_command(
        [sys.executable, SCRIPTS['experiments'], '--model_name', model_name],
        f"Esperimenti con {model_name}",
        show_live_output=True  # Script lungo, mostra progresso
    )


def run_analyze(model_name, force_update=False):
    """
    Esegue analyze.py per un modello specifico.
    
    Args:
        model_name: Nome del modello (es. 'EleutherAI/pythia-160m')
        force_update: Se True, ricalcola anche se i file esistono
    """
    results_file = get_results_file(model_name)
    
    if not check_file_exists(results_file):
        print(f"✗ File risultati non trovato per {model_name}: {results_file}")
        return False
    
    metrics_file, report_file = get_analysis_files(model_name)
    
    # Controlla se i file di analisi esistono già
    if all(check_file_exists(f) for f in [metrics_file, report_file]) and not force_update:
        print(f"✓ File di analisi già presenti per {model_name}:")
        print(f"  • {metrics_file}")
        print(f"  • {report_file}")
        print("  Salto analyze.py")
        return True
    
    print(f"▸ Analisi risultati per {model_name}...")
    return run_command(
        [sys.executable, SCRIPTS['analyze'], '--input_file', results_file],
        f"Analisi {model_name}",
        show_live_output=True
    )


def run_visualize(model_name, force_update=False):
    """Esegue visualize.py per un modello specifico."""
    metrics_file, report_file = get_analysis_files(model_name)
    
    if not all(check_file_exists(f) for f in [metrics_file, report_file]):
        print(f"✗ File di analisi non trovati per {model_name}")
        return False
    
    # Le figure vanno in risultati/nome_modello/figure/
    outdir = os.path.join(RISULTATI_DIR, get_model_name_clean(model_name), "figure")
    os.makedirs(outdir, exist_ok=True)
    
    # Controlla quali figure mancano (verifica tutte le 23 figure attese)
    if not force_update:
        missing = [f for f in ALL_MODEL_FIGURES
                   if not check_file_exists(os.path.join(outdir, f))]
        if not missing:
            print(f"✓ Figure già presenti per {model_name}: {outdir}")
            print("  Salto visualize.py")
            return True
        # Determina i gruppi da rigenerare (solo quelli con figure mancanti)
        groups_needed = list(dict.fromkeys(FIGURE_TO_GROUP[f] for f in missing))
        print(f"  Figure mancanti ({len(missing)}): {', '.join(missing)}")
        print(f"  Gruppi da rigenerare: {', '.join(groups_needed)}")
        extra_args = ['--groups'] + groups_needed
    else:
        extra_args = []
    
    print(f"▸ Visualizzazione per {model_name}...")
    return run_command(
        [sys.executable, SCRIPTS['visualize'], 
         '--metrics', metrics_file,
         '--report', report_file,
         '--outdir', outdir] + extra_args,
        f"Visualizzazione {model_name}",
        show_live_output=True
    )


def run_comparative(force_update=False):
    """Esegue i grafici comparativi multi-modello, ma solo se necessario."""
    try:
        from visualize import generate_all_comparative, COMPARATIVE_OUT
    except Exception as e:
        print(f"✗ Impossibile importare i grafici comparativi: {e}")
        return False

    outdir = COMPARATIVE_OUT
    if not force_update:
        missing = [f for f in ALL_COMPARATIVE_FIGURES
                   if not check_file_exists(os.path.join(outdir, f))]
        if not missing:
            print(f"✓ Grafici comparativi già presenti: {outdir}")
            return True
        print(f"  Figure comparative mancanti ({len(missing)}): {', '.join(missing)}")

    return generate_all_comparative(outdir=outdir)

def run_only_experiments_pipeline(models=None, force_update=False):
    """
    Esegue SOLO la generazione degli esperimenti (per l'esecuzione sul server).
    """
    if models is None:
        models = AVAILABLE_MODELS
    
    print_header("AVVIO ESECUZIONE SERVER - SOLO ESPERIMENTI")
    print(f"Modelli selezionati: {len(models)}")
    
    total_steps = len(models)
    
    for idx, model in enumerate(models):
        print_header(f"MODELLO {idx + 1}/{len(models)}: {model}")
        print_step(idx + 1, total_steps, f"Esperimenti - {model}")
        
        if not run_experiments(model, force_update):
            print(f"\n✗ Errore negli esperimenti per {model}")
            # Sul server non chiediamo input all'utente se c'è un errore, passiamo al prossimo
            continue
            
    print_header("ESPERIMENTI COMPLETATI ✓")
    return True


# ══════════════════════════════════════════════════════════════════════
#  SEQUENZA OPERATIVA COMPLETA
# ══════════════════════════════════════════════════════════════════════

def run_full_pipeline(models=None, force_update=False):
    """
    Esegue la sequenza operativa completa del progetto.
    
    Args:
        models: Lista di modelli su cui eseguire gli esperimenti (None = tutti)
        force_update: Se True, ricalcola tutti i file anche se esistono
    """
    if models is None:
        models = AVAILABLE_MODELS
    
    print_header("AVVIO PIPELINE COMPLETA")
    print(f"Modelli selezionati: {len(models)}")
    for m in models:
        print(f"  • {m}")
    print(f"Force update: {'SÌ' if force_update else 'NO'}")
    
    total_steps = 2 + len(models) * 3  # mapping + init + (experiments + analyze + visualize) per modello
    current_step = 0
    
    # Step 1: Generate mapping
    current_step += 1
    print_step(current_step, total_steps, "Generazione mapping")
    if not run_generate_mapping(force_update):
        print("\n✗ Pipeline interrotta: errore in generate_mapping")
        return False
    
    # Step 2: Initialization
    current_step += 1
    print_step(current_step, total_steps, "Inizializzazione dataset")
    if not run_initialization(force_update):
        print("\n✗ Pipeline interrotta: errore in initialization")
        return False
    
    # Step 3-5 per ogni modello: Experiments, Analyze, Visualize
    for idx, model in enumerate(models):
        print_header(f"MODELLO {idx + 1}/{len(models)}: {model}")
        
        # Experiments
        current_step += 1
        print_step(current_step, total_steps, f"Esperimenti - {model}")
        if not run_experiments(model, force_update):
            print(f"\n✗ Errore negli esperimenti per {model}")
            if len(models) > 1 and idx < len(models) - 1:
                response = input("\n▸ Continuare con il prossimo modello? [S/N]: ").strip().upper()
                if response != 'S':
                    print("\n⚠ Pipeline interrotta dall'utente")
                    return False
            continue
        
        # Analyze
        current_step += 1
        print_step(current_step, total_steps, f"Analisi - {model}")
        if not run_analyze(model, force_update=force_update):
            print(f"\n✗ Errore nell'analisi per {model}")
            if len(models) > 1 and idx < len(models) - 1:
                response = input("\n▸ Continuare con il prossimo modello? [S/N]: ").strip().upper()
                if response != 'S':
                    print("\n⚠ Pipeline interrotta dall'utente")
                    return False
            continue
        
        # Visualize
        current_step += 1
        print_step(current_step, total_steps, f"Visualizzazione - {model}")
        if not run_visualize(model, force_update=force_update):
            print(f"\n✗ Errore nella visualizzazione per {model}")
            if len(models) > 1 and idx < len(models) - 1:
                response = input("\n▸ Continuare con il prossimo modello? [S/N]: ").strip().upper()
                if response != 'S':
                    print("\n⚠ Pipeline interrotta dall'utente")
                    return False
            continue
    
    print_header("PIPELINE COMPLETATA ✓")
    return True


# ══════════════════════════════════════════════════════════════════════
#  MENU INTERATTIVO
# ══════════════════════════════════════════════════════════════════════

def display_menu():
    """Mostra il menu principale."""
    print("\n" + "═" * 70)
    print("  MENU PRINCIPALE - Gestione Progetto")
    print("═" * 70)
    print("\n  [1] Esegui pipeline completa")
    print("  [2] Genera mapping (generate_mapping.py)")
    print("  [3] Inizializza dataset (initialization.py)")
    print("  [4] Esegui esperimenti (experiments_1.py)")
    print("  [5] Analizza risultati (analyze.py)")
    print("  [6] Crea grafici (visualize.py)")
    print("  [7] Genera grafici comparativi multi-modello")
    print("  [8] Installa/aggiorna dipendenze (requirements.txt)")
    print("  [9] Esegui SOLO gli esperimenti (modalità server)")
    print(" [10] Analisi risultati + creazione grafici (anche i comparativi)")
    print("\n  [0] Esci")
    print("-" * 70)


def select_models():
    """Interfaccia per selezionare i modelli su cui operare."""
    print("\n" + "─" * 70)
    print("  SELEZIONE MODELLI")
    print("─" * 70)
    print("\n  [0] Tutti i modelli")
    
    for idx, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  [{idx}] {model}")
    
    print("-" * 70)
    
    while True:
        choice = input("\n▸ Seleziona modello (0 per tutti): ").strip()
        
        if choice == '0':
            return AVAILABLE_MODELS
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return [AVAILABLE_MODELS[idx - 1]]
            else:
                print("✗ Scelta non valida. Riprova.")
        except ValueError:
            print("✗ Inserisci un numero valido.")


def ask_update_mode():
    """Chiede se attivare la modalità update."""
    while True:
        choice = input("\n▸ Modalità UPDATE (ricalcola file esistenti) [default: N]? [S/N]: ").strip().lower()
        if choice in ['s', 'si', 'sì', 'y', 'yes']:
            return True
        elif choice in ['n', 'no', '']:
            return False
        else:
            print("✗ Risposta non valida. Rispondi S/N.")


def interactive_menu():
    """Menu interattivo principale."""
    print_header("MENU INTERATTIVO - Progetto Deep Learning")
    
    # Chiedi sempre la modalità update nel menu interattivo
    update_mode = ask_update_mode()
    
    try:
        while True:
            display_menu()
            choice = input("\n▸ Scelta: ").strip()
            
            if choice == '0':
                print("\nUscita dal menu. Arrivederci!\n")
                break
            
            elif choice == '1':
                # Pipeline completa
                models = select_models()
                print_header("ESECUZIONE PIPELINE COMPLETA")
                run_full_pipeline(models, update_mode)
            
            elif choice == '2':
                # Generate mapping
                print_header("GENERAZIONE MAPPING")
                run_generate_mapping(update_mode)
            
            elif choice == '3':
                # Initialization
                print_header("INIZIALIZZAZIONE DATASET")
                run_initialization(update_mode)
            
            elif choice == '4':
                # Experiments
                models = select_models()
                print_header("ESECUZIONE ESPERIMENTI")
                for model in models:
                    print(f"\n▸ Modello: {model}")
                    run_experiments(model, update_mode)
            
            elif choice == '5':
                # Analyze
                models = select_models()
                print_header("ANALISI RISULTATI")
                for model in models:
                    print(f"\n▸ Modello: {model}")
                    run_analyze(model, force_update=update_mode)
            
            elif choice == '6':
                # Visualize
                models = select_models()
                print_header("VISUALIZZAZIONE GRAFICI")
                for model in models:
                    print(f"\n▸ Modello: {model}")
                    run_visualize(model, force_update=update_mode)
                if len(models) == len(AVAILABLE_MODELS):
                    run_comparative(force_update=update_mode)
            
            elif choice == '7':
                # Grafici comparativi multi-modello
                run_comparative(force_update=update_mode)

            elif choice == '8':
                # Installa/aggiorna dipendenze
                print_header("INSTALLAZIONE DIPENDENZE")
                install_dependencies()

            elif choice == '9':
                # Solo Esperimenti (Server)
                models = select_models()
                print_header("ESECUZIONE SERVER - SOLO ESPERIMENTI")
                run_only_experiments_pipeline(models, update_mode)

            elif choice == '10':
                # Solo Analisi + Visualizzazione + Grafici comparativi
                models = select_models()
                print_header("ANALISI RISULTATI + CREAZIONE GRAFICI + GRAFICI COMPARATIVI")
                any_success = False
                for model in models:
                    print(f"\n\u25b8 Modello: {model}")
                    ok = run_analyze(model, force_update=update_mode)
                    if ok:
                        any_success = True
                        run_visualize(model, force_update=update_mode)
                    else:
                        print(f"\u2717 Analisi fallita per {model}, visualizzazione saltata.")
                if any_success:
                    run_comparative(force_update=update_mode)

            else:
                print("\n✗ Scelta non valida. Riprova.")
            
            input("\n[Premi INVIO per continuare]")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interruzione rilevata. Uscita dal menu.\n")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    """Punto di ingresso principale del programma."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sistema di gestione ed esecuzione del progetto',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  python menu.py                    # Esegue pipeline completa automatica
  python menu.py --update           # Esegue pipeline forzando ricalcolo
  python menu.py --menu             # Apre il menu interattivo
        """
    )
    
    parser.add_argument(
        '--menu',
        action='store_true',
        help='Attiva il menu interattivo'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Forza il ricalcolo di tutti i file (sovrascrive esistenti)'
    )
    
    args = parser.parse_args()
    
    # Verifica che le directory esistano
    os.makedirs(RISULTATI_DIR, exist_ok=True)
    
    if args.menu:
        # Modalità menu interattivo
        interactive_menu()
    else:
        # Modalità pipeline automatica
        print_header("MODALITÀ AUTOMATICA - Pipeline Completa")
        if args.update:
            print("⚠ Modalità UPDATE attiva: i file esistenti verranno sovrascritti\n")
        run_full_pipeline(models=AVAILABLE_MODELS, force_update=args.update)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interruzione rilevata. Uscita dal programma.\n")
