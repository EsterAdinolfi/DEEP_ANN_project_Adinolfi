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


# ══════════════════════════════════════════════════════════════════════
#  GESTIONE DIPENDENZE
# ══════════════════════════════════════════════════════════════════════

def check_and_install_dependencies(force_install=False):
    """
    Verifica e installa le dipendenze necessarie dal requirements.txt.
    
    Args:
        force_install: Se True, reinstalla tutte le dipendenze
    """
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"⚠ File requirements.txt non trovato in: {REQUIREMENTS_FILE}")
        print("  Salto il controllo delle dipendenze.\n")
        return True
    
    print("Verifica dipendenze...")
    
    # Lista delle librerie critiche da verificare
    critical_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
    }
    
    missing_packages = []
    
    if not force_install:
        # Verifica se le librerie critiche sono installate
        for package, name in critical_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(name)
        
        if not missing_packages:
            print("✓ Tutte le dipendenze critiche sono installate.\n")
            return True
        
        print(f"⚠ Dipendenze mancanti: {', '.join(missing_packages)}")
    
    # Chiedi conferma prima di installare
    print(f"\nInstallazione dipendenze da: {os.path.basename(REQUIREMENTS_FILE)}")
    
    if not force_install:
        while True:
            choice = input("▸ Procedere con l'installazione [default: N]? [S/N]: ").strip().lower()
            if choice in ['s', 'si', 'sì', 'y', 'yes', '']:
                break
            elif choice in ['n', 'no']:
                print("⚠ Installazione saltata. Alcuni script potrebbero non funzionare.\n")
                return False
            else:
                print("✗ Risposta non valida. Rispondi S/N.")
    
    # Installa le dipendenze
    print("\n⏳ Installazione in corso...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS_FILE, '--upgrade'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minuti max
        )
        
        if result.returncode == 0:
            print("✓ Dipendenze installate con successo!\n")
            return True
        else:
            print("✗ Errore durante l'installazione delle dipendenze:")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout durante l'installazione (>5 minuti). Riprova manualmente con:")
        print(f"   pip install -r {REQUIREMENTS_FILE}")
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
    return os.path.join(RISULTATI_DIR, f"results_{clean_name}.json")


def get_analysis_files(model_name, mode='weighted'):
    """Restituisce i percorsi dei file di analisi per un dato modello."""
    clean_name = get_model_name_clean(model_name)
    metrics = os.path.join(RISULTATI_DIR, f"analysis_metrics_{mode}_{clean_name}.csv")
    report = os.path.join(RISULTATI_DIR, f"report_topic_{mode}_{clean_name}.csv")
    return metrics, report


def run_command(command, description):
    """Esegue un comando nel terminale e gestisce eventuali errori."""
    print(f"\n▸ {description}")
    print(f"  Comando: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✓ {description} completato con successo")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERRORE durante: {description}")
        print(f"  Codice uscita: {e.returncode}")
        if e.stderr:
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
    
    # Passa il model name come argomento
    return run_command(
        [sys.executable, SCRIPTS['experiments'], '--model_name', model_name],
        f"Esperimenti con {model_name}"
    )


def run_analyze(model_name, mode='weighted'):
    """Esegue analyze.py per un modello specifico."""
    results_file = get_results_file(model_name)
    metrics_file, report_file = get_analysis_files(model_name, mode)
    
    if not check_file_exists(results_file):
        print(f"✗ File risultati non trovato per {model_name}: {results_file}")
        return False
    
    print(f"▸ Analisi risultati per {model_name} (mode={mode})...")
    return run_command(
        [sys.executable, SCRIPTS['analyze'], '--input_file', results_file, '--mode', mode],
        f"Analisi {model_name}"
    )


def run_visualize(model_name, mode='weighted'):
    """Esegue visualize.py per un modello specifico."""
    metrics_file, report_file = get_analysis_files(model_name, mode)
    
    if not all(check_file_exists(f) for f in [metrics_file, report_file]):
        print(f"✗ File di analisi non trovati per {model_name}")
        return False
    
    outdir = os.path.join(RISULTATI_DIR, "figure", get_model_name_clean(model_name))
    os.makedirs(outdir, exist_ok=True)
    
    print(f"▸ Visualizzazione per {model_name}...")
    return run_command(
        [sys.executable, SCRIPTS['visualize'], 
         '--metrics', metrics_file,
         '--report', report_file,
         '--outdir', outdir],
        f"Visualizzazione {model_name}"
    )


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
            continue
        
        # Analyze
        current_step += 1
        print_step(current_step, total_steps, f"Analisi - {model}")
        if not run_analyze(model):
            print(f"\n✗ Errore nell'analisi per {model}")
            continue
        
        # Visualize
        current_step += 1
        print_step(current_step, total_steps, f"Visualizzazione - {model}")
        if not run_visualize(model):
            print(f"\n✗ Errore nella visualizzazione per {model}")
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
    print("  [6] Visualizza grafici (visualize.py)")
    print("  [7] Reinstalla dipendenze (requirements.txt)")
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
                    run_analyze(model)
            
            elif choice == '6':
                # Visualize
                models = select_models()
                print_header("VISUALIZZAZIONE GRAFICI")
                for model in models:
                    print(f"\n▸ Modello: {model}")
                    run_visualize(model)
            
            elif choice == '7':
                # Reinstalla dipendenze
                print_header("REINSTALLAZIONE DIPENDENZE")
                check_and_install_dependencies(force_install=True)
            
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
    
    # Verifica e installa dipendenze
    print_header("CONTROLLO DIPENDENZE")
    if not check_and_install_dependencies():
        print("⚠ Alcune dipendenze potrebbero mancare. Gli script potrebbero fallire.\n")
        while True:
            choice = input("▸ Continuare comunque [default: N]? [S/N]: ").strip().lower()
            if choice in ['s', 'si', 'sì', 'y', 'yes']:
                break
            elif choice in ['n', 'no', '']:
                print("\nUscita dal programma.\n")
                return
            else:
                print("✗ Risposta non valida. Rispondi S/N.")
    
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
