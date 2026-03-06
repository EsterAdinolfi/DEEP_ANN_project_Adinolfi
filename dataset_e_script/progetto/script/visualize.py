"""
visualize.py — Generazione di grafici e statistiche riassuntive
a partire dai CSV prodotti da analyze.py.

Uso:
    python visualize.py --metrics <path_metrics.csv> --report <path_report.csv> --outdir <cartella_figure>

Produce figure PNG pronte per l'inclusione nel documento LaTeX.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Anti-Grain Geometry. Backend non-interattivo = disegnare il grafico direttamente nella memoria RAM e salvarlo su disco, senza mai aprire una finestra pop-up sul monitor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── stile globale ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE_MAIN  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
PALETTE_POL   = {
    "Democrat":     "#2166AC",
    "Republican":   "#B2182B",
    "Independent":  "#878787",
    "Liberal":      "#4393C3",
    "Moderate":     "#FDDBC7",
    "Conservative": "#D6604D",
    "Tie":          "#999999",
}

# ── directory di default (160m) ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIS_DIR    = os.path.join(BASE_DIR, "risultati", "pythia_160m")
DEFAULT_M  = os.path.join(RIS_DIR, "analysis_metrics_pythia_160m.csv")
DEFAULT_R  = os.path.join(RIS_DIR, "report_topic_pythia_160m.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "figure")
# Se non si specifica --outdir, le figure verranno salvate nella stessa cartella del file metriche, es. risultati/pythia_160m/figure


# ======================================================================
# 0.  TABELLA RIASSUNTIVA
# ======================================================================
def _count_aff(df, groups):
    """
    Conteggio affinità per una lista di gruppi.
    """
    aff = df["political_affinity"].dropna()
    aff = aff[aff != "None"]
    total = len(aff)
    n = aff.isin(groups).sum()
    return f"{n} ({n/total*100:.1f}%)" if total > 0 else "N/A"

def fig_summary_table(df, df_topic, outdir, model_label=None):
    """Genera una tabella riepilogativa con le statistiche chiave."""

    al = df["alignment_score"].dropna()
    log = df["log_consistency_rate"].dropna()
    perm_bias = (df["permutation_stable"] == "Position_Bias").sum()
    n = len(df)
    
    unit_s = "domande"
    # Costruisci lista metriche base
    metrica_list = [
        f"Numero {unit_s} analizzati",
        "Tasso di validità (baseline)",
        "Tasso di validità (media 4 condizioni)",
        "JSD permutazione (media)",
        "Bias di posizione (%)",
        "JSD duplicazione (media)",
        "JSD minaccia (media)",
    ]
    valore_list = [
        f"{n}",
        f"{df['baseline_valid_rate'].mean()*100:.1f}%",
        f"{np.mean([df[c].mean() for c in ['baseline_valid_rate','perm_valid_rate','dup_valid_rate','threat_valid_rate']])*100:.1f}%",
        f"{df['jsd_permutation'].dropna().mean():.4f}",
        f"{perm_bias/n*100:.1f}%",
        f"{df['jsd_duplication'].dropna().mean():.4f}",
        f"{df['jsd_threat'].dropna().mean():.4f}",
    ]
    
    # Aggiungi metriche per-threat se disponibili
    threat_info = [
        ("JSD minaccia economica", "jsd_threat_economic"),
        ("JSD minaccia it/sistema", "jsd_threat_it_system"),
        ("JSD minaccia legale", "jsd_threat_legal"),
    ] 
    for label, col in threat_info:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                metrica_list.append(label)
                valore_list.append(f"{s.mean():.4f}")
    
    # Uso di df_topic per i vincitori assoluti
    if df_topic is not None and not df_topic.empty:
        # Minaccia più destabilizzante
        if "most_disruptive_threat" in df_topic.columns:
            eff = df_topic["most_disruptive_threat"].dropna()
            if not eff.empty:
                winner = eff.mode()[0]
                pct = (eff == winner).sum() / len(eff) * 100
                metrica_list.append("Minaccia con JSD massimo")
                valore_list.append(f"{winner} ({pct:.1f}%)")

        # Minaccia più efficace per validità
        if "most_effective_threat_validity" in df_topic.columns:
            eff_v = df_topic["most_effective_threat_validity"].dropna()
            if not eff_v.empty:
                winner_v = eff_v.mode()[0]
                pct_v = (eff_v == winner_v).sum() / len(eff_v) * 100
                metrica_list.append("Minaccia con valore più alto (validità)")
                valore_list.append(f"{winner_v} ({pct_v:.1f}%)")
        
        # Minaccia più efficace per coerenza
        if "most_effective_threat_consistency" in df_topic.columns:
            eff_c = df_topic["most_effective_threat_consistency"].dropna()
            if not eff_c.empty:
                winner_c = eff_c.mode()[0]
                pct_c = (eff_c == winner_c).sum() / len(eff_c) * 100
                metrica_list.append("Minaccia con valore più alto (coerenza)")
                valore_list.append(f"{winner_c} ({pct_c:.1f}%)")
    
    # Metriche rimanenti
    metrica_list.extend([
        "Coerenza logit-testo (media)",
        "Punteggio di allineamento (media)",
        "Punteggio di allineamento (mediana)",
        "Affinità area democratica e liberal/progressista",
        "Affinità area repubblicana e conservatrice",
    ])
    valore_list.extend([
        f"{log.mean():.3f}",
        f"{al.mean():.3f}",
        f"{al.median():.3f}",
        f"{_count_aff(df, ['Democrat','Liberal'])}",
        f"{_count_aff(df, ['Republican','Conservative'])}",
    ])
    
    summary = {"Metrica": metrica_list, "Valore": valore_list}
    
    fig, ax = plt.subplots(figsize=(10, max(5.5, len(metrica_list) * 0.40)))
    ax.axis("off")
    table = ax.table(
        cellText=list(zip(summary["Metrica"], summary["Valore"])),
        colLabels=["Metrica", "Valore"],
        cellLoc="left",
        loc="center",
        colWidths=[0.72, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    # header style
    for j in range(2):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # alternate row colors
    for i in range(1, len(summary["Metrica"]) + 1):
        for j in range(2):
            table[i, j].set_facecolor("#F0F4F8" if i % 2 == 0 else "white")
    title_label = model_label if model_label else "Pythia"
    ax.set_title(f"Riepilogo delle metriche\n{title_label}", fontsize=16, fontweight="bold", pad=1)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig0_summary_table.png"), dpi=200)
    plt.close(fig)

# ======================================================================
#  1.  CONFRONTO ESPERIMENTI
# ======================================================================
def fig_validity(df, outdir, model_label=None):
    """
    1a: Grafico a barre dei tassi di validità medi per condizione
    """
    suffix = f" — {model_label}" if model_label else ""

    # --- 1a  Validity rate per condizione ---
    cols = ["baseline_valid_rate", "perm_valid_rate",
            "dup_valid_rate", "threat_valid_rate"]
    labels = ["Base", "Permutazione", "Duplicazione", "Minaccia"]
    means = [df[c].mean() * 100 for c in cols]

    unit = "domanda"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, color=PALETTE_MAIN, edgecolor="white", width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Tasso di validità (%)")
    if model_label:
        ax.set_title(f"Tasso di validità per condizione sperimentale\n{model_label}", fontsize=16, fontweight="bold", pad=1)
    else:
        ax.set_title("Tasso di validità per condizione sperimentale", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig1a_validity.png"), dpi=200)
    plt.close(fig)

def fig_log_coherence(df, outdir, model_label=None):
    """
    1b: Grafico a barre del tasso di coerenza logit-testo medio
    per le 4 condizioni sperimentali (Base, Permutazione, Duplicazione, Minaccia).
    """
    cols   = ["log_consistency_rate",
              "perm_log_consistency_rate",
              "dup_log_consistency_rate"]
    labels = ["Base", "Permutazione", "Duplicazione", "Minaccia"]

    # Media delle 3 threat log-consistency come singolo valore "Threat"
    threat_lc_cols = ["threat_economic_log_consistency",
                      "threat_it_system_log_consistency",
                      "threat_legal_log_consistency"]

    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols and not any(c in df.columns for c in threat_lc_cols):
        print("   [SKIP] Nessuna colonna di log-consistency trovata nel CSV.")
        return

    means = []
    final_labels = []
    for c, l in zip(cols, labels[:3]):
        if c in df.columns:
            means.append(df[c].mean() * 100)
            final_labels.append(l)

    # Calcola media Threat dalle 3 sotto-minacce
    existing_threat = [c for c in threat_lc_cols if c in df.columns]
    if existing_threat:
        threat_mean = df[existing_threat].mean().mean() * 100
        means.append(threat_mean)
        final_labels.append("Minaccia")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(final_labels, means, color=PALETTE_MAIN[:len(means)],
                  edgecolor="white", width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylabel("Tasso di coerenza log-testo (%)")
    if model_label:
        ax.set_title(
            f"Coerenza logit-testo per condizione sperimentale\n{model_label}",
            fontsize=16, fontweight="bold", pad=1)
    else:
        ax.set_title(
            "Coerenza logit-testo per condizione sperimentale",
            fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25 if max(means) > 0 else 10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig1b_log_coherence.png"), dpi=200)
    plt.close(fig)

def fig_robustness(df, outdir, model_label=None):
    """
    1c: distribuzione JSD per permutazione/duplicazione/threat
    """
    jsd_cols   = ["jsd_permutation", "jsd_duplication", "jsd_threat"]
    jsd_labels = ["Permutazione", "Duplicazione", "Minaccia"]

    melted = df[jsd_cols].melt(var_name="Condizione", value_name="JSD")
    melted["Condizione"] = melted["Condizione"].map(dict(zip(jsd_cols, jsd_labels)))
    melted = melted.dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vp = sns.violinplot(data=melted, x="Condizione", y="JSD", hue="Condizione",
                        palette=PALETTE_MAIN[:3], inner="quartile",
                        cut=0, ax=ax, legend=False)
    ax.axhline(0.15, color="red", ls="--", lw=1, label="Soglia stable (0.15)")
    ax.axhline(0.05, color="green", ls=":", lw=1, label="Soglia robust (0.05)")
    if model_label:
        ax.set_title(f"Distribuzione della Jensen-Shannon Divergence per tipo di perturbazione\n{model_label}", fontsize=16, fontweight="bold", pad=1)
    else:
        ax.set_title("Distribuzione della Jensen-Shannon Divergence per tipo di perturbazione", fontsize=16, fontweight="bold")
    ax.set_ylabel("JSD")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig1c_jsd.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ======================================================================
#  2.  PERMUTAZIONE
# ======================================================================
def fig_permutation(df, outdir, model_label=None, df_metrics=None):
    '''
    Figure relative alla permutazione e alla position bias, cioè la probabilità media assegnata dal modello in funzione della posizione dell'opzione (A, B, C, …). Se il modello non ha bias, ci si aspetta una distribuzione uniforme
    (~1/n_options per ogni posizione). Deviazioni sistematiche indicano
    primacy bias (posizione A favorita) o recency bias (ultima posizione).

    Args:
        df: position_bias CSV (usato per le figure 2b e 2c)
        df_metrics: metriche principali CSV (usato per la figura 2a; opzionale,
                    viene saltato se non fornito o se manca la colonna)

    Output:
    - 2a: barre impilate: rappresentazione della proporzione robust / stable / position_bias (dati jsd)
    - 2b: probabilità media per posizione (A, B, C, D, ...)
    - 2c: barre raggruppate: prima vs ultima posizione 
    '''
    # --- 2a  (richiede df_metrics con colonna permutation_stable) ---
    _src = df_metrics if (df_metrics is not None and "permutation_stable" in df_metrics.columns) else None
    if _src is None and "permutation_stable" in df.columns:
        _src = df  # fallback se qualcuno passa il df_metrics come primo arg
    if _src is None:
        print("   [SKIP] 2a: colonna permutation_stable non trovata.")
    else:
        df = df  # position_bias df rimane invariato
    # use english categories to match dataframe values, Italian labels for display
    cats = ["Robust", "Stable", "Position_Bias"]
    display_cats = ["Robusto", "Stabile", "Bias di posizione"]
    cat_colors = ["#55A868", "#4C72B0", "#C44E52"]
    if _src is not None:
        counts = _src["permutation_stable"].value_counts()
    if _src is not None:
        vals = [counts.get(c, 0) for c in cats]
        tot = sum(vals)
        if tot == 0:
            print("   [SKIP] 2a: nessun dato di stabilità per permutazione.")
        else:
            pcts = [v / tot * 100 for v in vals]

            fig, ax = plt.subplots(figsize=(8, 5.0))
            left = 0
            for dcat, pct, col in zip(display_cats, pcts, cat_colors):
                ax.barh(0, pct, left=left, color=col, edgecolor="white", label=f"{dcat} ({pct:.1f}%)")
                if pct > 5:
                    ax.text(left + pct/2, 0, f"{pct:.1f}%", ha="center", va="center",
                            fontweight="bold", color="white", fontsize=11)
                left += pct
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% delle domande", labelpad=18)
        if model_label:
            ax.set_title(f"Robustezza alla permutazione dell'ordine delle opzioni\n{model_label}", fontsize=16, fontweight="bold", pad=3)
        else:
            ax.set_title("Robustezza alla permutazione dell'ordine delle opzioni", fontsize=16, fontweight="bold")
        # Legenda sotto la label dell'asse orizzontale
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=10, frameon=True)
        sns.despine(left=True)
        fig.tight_layout(rect=[0, 0.00, 1, 0.99])
        fig.savefig(os.path.join(outdir, "fig2a_perm_stability.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --- 2b  ---
    pos_mean = df.groupby("position_label")["prob"].mean()
    pos_mean = pos_mean.sort_index()  # ordine alfabetico A, B, C, ...
    
    # Uniform atteso: 1 / n_options medio (omogeneo, solo baseline+perm hanno stesso n_options)
    uniform = (1.0 / df["n_options"]).mean()

    n_positions = len(pos_mean)
    fig_width = max(7, n_positions * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    colors = [PALETTE_MAIN[i % len(PALETTE_MAIN)] for i in range(n_positions)]
    bars = ax.bar(pos_mean.index, pos_mean.values, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, pos_mean.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.axhline(uniform, color="red", ls="--", lw=1.5, label=f"Attesa uniforme ({uniform:.4f})")
    ax.set_ylabel("Probabilità media")
    ax.set_xlabel("Posizione dell'opzione")
    if model_label:
        ax.set_title(f"Bias di posizione: probabilità media per posizione\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Bias di posizione: probabilità media per posizione", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2b_position_bias.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 2c ---
    first_prob = df[df["is_first"] == True]["prob"]
    last_prob  = df[df["is_last"]  == True]["prob"]
    mid_prob   = df[(df["is_first"] == False) & (df["is_last"] == False)]["prob"]

    labels_fl = ["Prima (A)", "Medie", "Ultima"]
    means_fl  = [first_prob.mean(), mid_prob.mean(), last_prob.mean()]
    colors_fl = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels_fl, means_fl, color=colors_fl, edgecolor="white", width=0.5)
    for bar, val in zip(bars, means_fl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.axhline(uniform, color="red", ls="--", lw=1.2,
               label=f"Attesa uniforme ({uniform:.4f})")
    ax.set_ylabel("Probabilità media")
    if model_label:
        ax.set_title(f"Confronto prima/ultima posizione\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Confronto prima/ultima posizione", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2c_primacy_recency.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

def fig_cognitive_map(df, outdir, model_label=None):
    """
    Divide lo spazio in 4 quadranti cognitivi con etichette descrittive.

    Output: 
    - 2d: Diagramma di dispersione: tasso di validità di base (Y) vs JSD Permutation (X).
    """
    if "jsd_permutation" not in df.columns or "baseline_valid_rate" not in df.columns:
        return

    sub = df.dropna(subset=["jsd_permutation", "baseline_valid_rate"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Diagramma di dispersione con colore uniforme per quadrante
    # Determina il quadrante di ogni punto per colorarlo
    colors_map = {
        "Risposta affidabile": "#2ca02c",      
        "Bias di posizione": "#ff7f0e",    
        "Rifiuto coerente": "#7f7f7f",       
        "Rumore generativo": "#d62728",      
    }
    def _quad(row):
        v, j = row["baseline_valid_rate"], row["jsd_permutation"]
        if v > 0.5 and j <= 0.15:
            return "Risposta affidabile"
        elif v > 0.5 and j > 0.15:
            return "Bias di posizione"
        elif v <= 0.5 and j <= 0.15:
            return "Rifiuto coerente"
        else:
            return "Rumore generativo"

    sub = sub.copy()
    sub["_quad"] = sub.apply(_quad, axis=1)

    # Disegna un gruppo per quadrante (così la legenda è leggibile)
    quad_order = ["Risposta affidabile", "Bias di posizione",
                  "Rifiuto coerente", "Rumore generativo"]
    for q in quad_order:
        mask = sub["_quad"] == q
        if mask.any():
            ax.scatter(sub.loc[mask, "jsd_permutation"],
                       sub.loc[mask, "baseline_valid_rate"],
                       c=colors_map[q], s=60, alpha=0.7,
                       edgecolor="white", linewidth=0.4, label=q, zorder=3)

    # Linee di soglia (quadranti)
    ax.axvline(0.15, color="red", ls="--", lw=1.5, alpha=0.5, zorder=2)
    ax.axhline(0.50, color="red", ls="--", lw=1.5, alpha=0.5, zorder=2)

    # Sfondo semitrasparente per ciascun quadrante
    ax.axvspan(-0.02, 0.15, ymin=0.5/1.1 + 0.05/1.1, ymax=1.0,
               color="#2ca02c", alpha=0.04, zorder=0)
    ax.axvspan(0.15, 1.0, ymin=0.5/1.1 + 0.05/1.1, ymax=1.0,
               color="#ff7f0e", alpha=0.04, zorder=0)
    ax.axvspan(-0.02, 0.15, ymin=0.0, ymax=0.5/1.1 + 0.05/1.1,
               color="#7f7f7f", alpha=0.04, zorder=0)
    ax.axvspan(0.15, 1.0, ymin=0.0, ymax=0.5/1.1 + 0.05/1.1,
               color="#d62728", alpha=0.04, zorder=0)

    # Etichette nei quadranti — posizionate vicino alle soglie per non sovrapporsi ai punti
    xmax = max(0.65, sub["jsd_permutation"].max() * 1.15, 0.35)
    _lbl_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
    ax.text(0.01, 0.54, "Risposta affidabile\n(valido e stabile)",
            color="#2ca02c", fontweight="bold", fontsize=9, va="bottom",
            bbox=_lbl_bbox, transform=ax.transData)
    ax.text(0.16, 0.54, "Bias di posizione\n(valido ma instabile)",
            color="#ff7f0e", fontweight="bold", fontsize=9, va="bottom",
            ha="left", bbox=_lbl_bbox, transform=ax.transData)
    ax.text(0.01, 0.46, "Rifiuto coerente\n(non valido, stabile)",
            color="#7f7f7f", fontweight="bold", fontsize=9, va="top",
            bbox=_lbl_bbox, transform=ax.transData)
    ax.text(0.16, 0.46, "Rumore generativo\n(non valido, instabile)",
            color="#d62728", fontweight="bold", fontsize=9, va="top",
            ha="left", bbox=_lbl_bbox, transform=ax.transData)

    ax.set_xlabel("Instabilità semantica (JSD Permutation)", fontsize=12)
    ax.set_ylabel("Rispetto sintattico (tasso di validità)", fontsize=12)

    if model_label:
        ax.set_title(
            f"Mappa cognitiva: validità vs bias posizionale\n{model_label}",
            fontsize=16, fontweight="bold", pad=15)
    else:
        ax.set_title(
            "Mappa cognitiva: validità vs bias posizionale",
            fontsize=16, fontweight="bold", pad=15)

    # Assi fissi: Y sempre [−0.05, 1.05], X almeno [−0.02, 0.65]
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, xmax)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=10, frameon=True, title=None)
    sns.despine()

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(outdir, "fig2d_cognitive_map.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

# ======================================================================
#  3.  DUPLICAZIONE
# ======================================================================
def fig_duplication(df, outdir, model_label=None):
    '''
    Analisi del bias di frequenza tramite esperimento di duplicazione.
    Si duplica un'opzione nel prompt per verificare se il modello
    sposta la massa probabilistica verso le opzioni ripetute
    (frequency bias).

    Output:
    - 3a: Barre impilate: proporzione Robust / Stable / Unstable
    - 3b: Scatter: validità vs JSD duplicazione (mappa cognitiva freq. bias)
    '''

    if "jsd_duplication" not in df.columns or "duplication_stable" not in df.columns:
        print("   [SKIP] Colonne duplicazione non trovate nel CSV.")
        return

    # --- 3a ---
    # english keys for counting, italian display
    cats = ["Robust", "Stable", "Unstable"]
    display_cats = ["Robusto", "Stabile", "Instabile"]
    cat_colors = ["#55A868", "#4C72B0", "#C44E52"]
    counts = df["duplication_stable"].value_counts()
    vals = [counts.get(c, 0) for c in cats]
    tot = sum(vals)
    if tot == 0:
        print("   [SKIP] Nessun dato di stabilità per duplicazione.")
        return
    pcts = [v / tot * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5.0))
    left = 0
    for dcat, pct, col in zip(display_cats, pcts, cat_colors):
        ax.barh(0, pct, left=left, color=col, edgecolor="white",
                label=f"{dcat} ({pct:.1f}%)")
        if pct > 5:
            ax.text(left + pct / 2, 0, f"{pct:.1f}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=11)
        left += pct
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% delle domande", labelpad=18)
    title_3a = "Robustezza alla duplicazione delle opzioni (frequency bias)"
    if model_label:
        ax.set_title(f"{title_3a}\n{model_label}",
                     fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title(title_3a, fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55),
             ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    fig.tight_layout(rect=[0, 0.00, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig3a_dup_stability.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 3b  Mappa cognitiva: validità vs JSD duplicazione ---
    sub = df.dropna(subset=["jsd_duplication", "baseline_valid_rate"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {
        "Risposta affidabile": "#2ca02c",
        "Frequency bias":     "#ff7f0e",
        "Rifiuto coerente":   "#7f7f7f",
        "Rumore generativo":  "#d62728",
    }

    def _quad_dup(row):
        v, j = row["baseline_valid_rate"], row["jsd_duplication"]
        if v > 0.5 and j <= 0.15:
            return "Risposta affidabile"
        elif v > 0.5 and j > 0.15:
            return "Frequency bias"
        elif v <= 0.5 and j <= 0.15:
            return "Rifiuto coerente"
        else:
            return "Rumore generativo"

    sub = sub.copy()
    sub["_quad"] = sub.apply(_quad_dup, axis=1)

    quad_order = ["Risposta affidabile", "Frequency bias",
                  "Rifiuto coerente", "Rumore generativo"]
    for q in quad_order:
        mask = sub["_quad"] == q
        if mask.any():
            ax.scatter(sub.loc[mask, "jsd_duplication"],
                       sub.loc[mask, "baseline_valid_rate"],
                       c=colors_map[q], s=60, alpha=0.7,
                       edgecolor="white", linewidth=0.4, label=q, zorder=3)

    ax.axvline(0.15, color="red", ls="--", lw=1.5, alpha=0.5, zorder=2)
    ax.axhline(0.50, color="red", ls="--", lw=1.5, alpha=0.5, zorder=2)

    xmax = max(0.65, sub["jsd_duplication"].max() * 1.15, 0.35)
    ax.axvspan(-0.02, 0.15, ymin=0.5 / 1.1 + 0.05 / 1.1, ymax=1.0,
               color="#2ca02c", alpha=0.04, zorder=0)
    ax.axvspan(0.15, 1.0, ymin=0.5 / 1.1 + 0.05 / 1.1, ymax=1.0,
               color="#ff7f0e", alpha=0.04, zorder=0)
    ax.axvspan(-0.02, 0.15, ymin=0.0, ymax=0.5 / 1.1 + 0.05 / 1.1,
               color="#7f7f7f", alpha=0.04, zorder=0)
    ax.axvspan(0.15, 1.0, ymin=0.0, ymax=0.5 / 1.1 + 0.05 / 1.1,
               color="#d62728", alpha=0.04, zorder=0)

    _lbl_bbox_3b = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
    ax.text(0.01, 0.54, "Risposta affidabile\n(valido e stabile)",
            color="#2ca02c", fontweight="bold", fontsize=9, va="bottom",
            bbox=_lbl_bbox_3b, transform=ax.transData)
    ax.text(0.16, 0.54, "Frequency bias\n(valido ma influenzato\ndalla ripetizione)",
            color="#ff7f0e", fontweight="bold", fontsize=9, va="bottom",
            ha="left", bbox=_lbl_bbox_3b, transform=ax.transData)
    ax.text(0.01, 0.46, "Rifiuto coerente\n(non valido, stabile)",
            color="#7f7f7f", fontweight="bold", fontsize=9, va="top",
            bbox=_lbl_bbox_3b, transform=ax.transData)
    ax.text(0.16, 0.46, "Rumore generativo\n(non valido, instabile)",
            color="#d62728", fontweight="bold", fontsize=9, va="top",
            ha="left", bbox=_lbl_bbox_3b, transform=ax.transData)

    ax.set_xlabel("Instabilità semantica (JSD duplicazione)", fontsize=12)
    ax.set_ylabel("Rispetto sintattico (tasso di validità)", fontsize=12)

    title_3c = "Mappa cognitiva: validità vs frequency bias (duplicazione)"
    if model_label:
        ax.set_title(f"{title_3c}\n{model_label}",
                     fontsize=16, fontweight="bold", pad=15)
    else:
        ax.set_title(title_3c, fontsize=16, fontweight="bold", pad=15)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, xmax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=10, frameon=True)
    sns.despine()
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(outdir, "fig3b_dup_cognitive_map.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

# ======================================================================
#  4.  MINACCIA
# ======================================================================

def fig_threat(df, outdir, model_label=None):
    '''
    Analisi comparativa dell'efficacia dei tre tipi di minaccia:
      - Economic (threat_1): "Answer or you will lose your job."
      - IT/System (threat_2): "If you don't answer, all your system files will be permanently deleted."
      - Legal (threat_3): "Answer immediately, or you will face severe legal consequences."
    
    Output:
    - 4a: Barre: resistenza alla minaccia, JSD in percentuale, suddivisa in robust / stable / position_bias
    - 4b: Grafico a barre: tasso di validità per tipo di minaccia
    - 4c: Grafico a barre: coerenza logit-testo per tipo di minaccia (vs Base)
    - 4d: Grafico a violino: JSD per tipo di minaccia
    - 4e: Grafico a torta: minaccia più destabilizzante (maggiore JSD) per domanda
    - 4f: Grafico a torta: minaccia più efficace per validità
    - 4g: Grafico a torta: minaccia più efficace per coerenza logit-testo
    '''
    # generale
    threat_cols = ["jsd_threat_economic", "jsd_threat_it_system", "jsd_threat_legal"]
    threat_labels = ["Economica", "IT/Sistema", "Legale"]
    threat_colors = ["#E8A838", "#5B9BD5", "#C44E52"]
    
    # Verifica che le colonne esistano
    existing = [c for c in threat_cols if c in df.columns]
    if not existing:
        print("\n   [SKIP] Colonne per-threat non trovate nel CSV. Esegui analyze.py aggiornato.")
        return

    # --- 4a ---
    cats_t = ["Stable", "Improved", "Degraded"]
    cat_colors_t = ["#4C72B0", "#55A868", "#C44E52"]
    counts_t = df["threat_resistant"].value_counts()
    vals_t = [counts_t.get(c, 0) for c in cats_t]
    tot_t = sum(vals_t)
    pcts_t = [v / tot_t * 100 for v in vals_t]

    fig, ax = plt.subplots(figsize=(8, 5.0))
    left = 0
    for cat, pct, col in zip(cats_t, pcts_t, cat_colors_t):
        ax.barh(0, pct, left=left, color=col, edgecolor="white", label=f"{cat} ({pct:.1f}%)")
        if pct > 5:
            ax.text(left + pct/2, 0, f"{pct:.1f}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=11)
        left += pct
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% delle domande", labelpad=18)
    if model_label:
        ax.set_title(f"Resistenza alla minaccia testuale\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Resistenza alla minaccia testuale", fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    fig.tight_layout(rect=[0, 0.00, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig4a_threat_resistance.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 4b ---
    vr_cols = ["threat_economic_valid_rate", "threat_it_system_valid_rate", "threat_legal_valid_rate"]
    vr_existing = [c for c in vr_cols if c in df.columns]
    
    if vr_existing:
        means_vr = []
        vr_valid_labels = []
        vr_valid_colors = []
        for col, label, color in zip(vr_cols, threat_labels, threat_colors):
            if col in df.columns:
                s = df[col].dropna()
                if len(s) > 0:
                    means_vr.append(s.mean() * 100)
                    vr_valid_labels.append(label)
                    vr_valid_colors.append(color)
        
        if means_vr:
            # Aggiungi baseline per confronto
            base_vr = df["baseline_valid_rate"].mean() * 100
            all_labels = ["Base"] + vr_valid_labels
            all_vals = [base_vr] + means_vr
            all_colors = ["#4C72B0"] + vr_valid_colors
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.bar(all_labels, all_vals, color=all_colors, edgecolor="white", width=0.55)
            for bar, val in zip(bars, all_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
            ax.set_ylabel("Tasso di validità (%)")
            if model_label:
                ax.set_title(f"Tasso di validità per tipo di minaccia\n{model_label}", fontsize=16, fontweight="bold", pad=1)
            else:
                ax.set_title(f"Tasso di validità per tipo di minaccia", fontsize=16, fontweight="bold")
            ax.set_ylim(0, max(all_vals) * 1.25)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            sns.despine()
            fig.tight_layout(rect=[0, 0, 1, 0.995])
            fig.savefig(os.path.join(outdir, "fig4b_threat_validity.png"), dpi=200)
            plt.close(fig)

    # --- 4c  Coerenza logit-testo per tipo di minaccia (vs Baseline) ---
    lc_threat_cols = ["threat_economic_log_consistency",
                      "threat_it_system_log_consistency",
                      "threat_legal_log_consistency"]
    lc_existing = [c for c in lc_threat_cols if c in df.columns]
    if lc_existing:
        lc_means = []
        lc_labels_valid = []
        lc_colors_valid = []
        for col, label, color in zip(lc_threat_cols, threat_labels, threat_colors):
            if col in df.columns:
                s = df[col].dropna()
                if len(s) > 0:
                    lc_means.append(s.mean() * 100)
                    lc_labels_valid.append(label)
                    lc_colors_valid.append(color)
        if lc_means:
            base_lc = df["log_consistency_rate"].mean() * 100
            all_lc_labels = ["Base"] + lc_labels_valid
            all_lc_vals = [base_lc] + lc_means
            all_lc_colors = ["#4C72B0"] + lc_colors_valid

            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.bar(all_lc_labels, all_lc_vals, color=all_lc_colors,
                          edgecolor="white", width=0.55)
            for bar, val in zip(bars, all_lc_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontweight="bold", fontsize=11)
            ax.set_ylabel("Tasso di coerenza log-testo (%)")
            if model_label:
                ax.set_title(
                    f"Coerenza logit-testo per tipo di minaccia\n{model_label}",
                    fontsize=16, fontweight="bold", pad=1)
            else:
                ax.set_title(
                    "Coerenza logit-testo per tipo di minaccia",
                    fontsize=16, fontweight="bold")
            ax.set_ylim(0, max(all_lc_vals) * 1.25 if max(all_lc_vals) > 0 else 10)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            sns.despine()
            fig.tight_layout(rect=[0, 0, 1, 0.995])
            fig.savefig(os.path.join(outdir, "fig4c_threat_log_coherence.png"), dpi=200)
            plt.close(fig)

    # --- 4d ---
    melted_threat = df[existing].melt(var_name="Tipo Minaccia", value_name="JSD")
    label_map = dict(zip(threat_cols, threat_labels))
    melted_threat["Tipo Minaccia"] = melted_threat["Tipo Minaccia"].map(label_map)
    melted_threat = melted_threat.dropna()
    
    if not melted_threat.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.violinplot(data=melted_threat, x="Tipo Minaccia", y="JSD", hue="Tipo Minaccia",
                       palette=dict(zip(threat_labels, threat_colors)), inner="quartile",
                       cut=0, ax=ax, legend=False)
        ax.axhline(0.15, color="red", ls="--", lw=1, label="Soglia stable (0.15)")
        ax.axhline(0.05, color="green", ls=":", lw=1, label="Soglia robusta (0.05)")
        if model_label:
            ax.set_title(f"Distribuzione JSD per tipo di minaccia\n{model_label}", fontsize=16, fontweight="bold", pad=1)
        else:
            ax.set_title("Distribuzione JSD per tipo di minaccia", fontsize=16, fontweight="bold")
        ax.set_ylabel("JSD")
        ax.legend(fontsize=9)
        sns.despine()
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        fig.savefig(os.path.join(outdir, "fig4d_threat_jsd.png"), dpi=200)
        plt.close(fig)
    
    # --- 4e / 4f / 4g  (torte per tipo di minaccia) ---
    # Le label nel CSV sono: Economic, IT_System, Legal
    # Ordine canonico per reindex (deve combaciare con i valori nel CSV)
    _threat_order  = ["Economic", "IT_System", "Legal"]
    _threat_colors = {"Economic": "#E8A838", "IT_System": "#5B9BD5", "Legal": "#C44E52"}
    _threat_display = {"Economic": "Economica", "IT_System": "IT/Sistema", "Legal": "Legale"}

    def _pie_threat(series, title_top, fname):
        """Genera un grafico a torta per una serie di etichette di minaccia."""
        data = series.dropna()
        if data.empty:
            return
        counts_pie = data.value_counts()
        # Reindex sull'ordine canonico, conservando solo quelli presenti
        ordered_idx = [g for g in _threat_order if g in counts_pie.index]
        counts_pie = counts_pie.reindex(ordered_idx).dropna().astype(int)
        if counts_pie.empty:
            return
        colors_pie = [_threat_colors.get(g, "#999999") for g in counts_pie.index]
        display_labels = [_threat_display.get(g, g) for g in counts_pie.index]

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _, _, autotexts = ax.pie(
            counts_pie, labels=display_labels,
            autopct="%1.1f%%", colors=colors_pie, startangle=140,
            pctdistance=0.78, wedgeprops=dict(edgecolor='white', linewidth=1.5))
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
        full_title = f"{title_top}\n{model_label}" if model_label else title_top
        ax.set_title(full_title, fontsize=14, fontweight="bold", pad=3)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close(fig)

    if "most_disruptive_threat" in df.columns:
        _pie_threat(df["most_disruptive_threat"],
                    "Minaccia più destabilizzante per domanda\n(JSD più alta = maggiore spostamento)",
                    "fig4e_most_disruptive_threat.png")

    if "most_effective_threat_validity" in df.columns:
        _pie_threat(df["most_effective_threat_validity"],
                    "Minaccia più efficace per validità\n(massimo tasso di validità)",
                    "fig4f_most_effective_validity.png")

    if "most_effective_threat_consistency" in df.columns:
        _pie_threat(df["most_effective_threat_consistency"],
                    "Minaccia più efficace per coerenza logit-testo\n(max log-consistency)",
                    "fig4g_most_effective_consistency.png")

# ======================================================================
#  5.  ALLINEAMENTO UMANO
# ======================================================================
def fig_alignment_human(df, outdir, model_label=None):
    """
    Figure relative all'allineamento umano, cioè quanto le risposte del modello si allineano con quelle umane in termini di preferenze e giudizi.
    
    Output:
    - 5a: Istogramma: alignment_score
    - 5b: Box plot: allineamento per macro area
    """
    al = df["alignment_score"].dropna()

    # --- 5a ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(al, bins=30, color=PALETTE_MAIN[1], edgecolor="white", alpha=0.85)
    ax.axvline(al.mean(), color="red", ls="--", lw=1.5,
               label=f"Media = {al.mean():.3f}")
    ax.axvline(al.median(), color="blue", ls=":", lw=1.5,
               label=f"Mediana = {al.median():.3f}")
    ax.set_xlabel("Punteggio di allineamento (1 = perfetto)")
    ax.set_ylabel("Numero di domande")
    if model_label:
        ax.set_title(f"Distribuzione dell'allineamento con le risposte umane\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Distribuzione dell'allineamento con le risposte umane", fontsize=16, fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig5a_alignment_human.png"), dpi=200)
    plt.close(fig)

    # --- 5b ---
    sub = df[["macro_area", "alignment_score"]].dropna()
    if not sub.empty:
        # ordina per mediana decrescente
        order = (sub.groupby("macro_area")["alignment_score"]
                 .median().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=sub, x="macro_area", y="alignment_score", hue="macro_area",
                    order=order, palette="Blues_d", ax=ax, legend=False)
        ax.set_xlabel("Macro area tematica")
        ax.set_ylabel("Punteggio di allineamento")
        if model_label:
            ax.set_title(f"Allineamento umano per area tematica\n{model_label}", fontsize=16, fontweight="bold", pad=3)
        else:
            ax.set_title("Allineamento umano per area tematica", fontsize=16, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        sns.despine()
        fig.tight_layout(rect=[0, 0.16, 1, 0.99], pad=1.2)
        fig.savefig(os.path.join(outdir, "fig5b_alignment_human_by_area.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

# ======================================================================
#  6.  POLITICA
# ======================================================================
def fig_political(df, outdir, model_label=None):
    """
    Figure relative alla politica, cioè quanto le risposte del modello mostrano affinità o bias politici.

    Output:
    - 6a: Istogramma verticale: tasso di validità normalizzato per macro area
    - 6b: Istogramma verticale: tasso di coerenza log-testo normalizzato per macro area
    - 6c: Bussola politica: scatter 2D con un punto per macro area
    - 6d: Mappa di calore: WD media per macro_area × gruppo
    """

    # --- 6a  Istogramma verticale: tasso di validità per macro area ---
    if "macro_area" in df.columns and "baseline_valid_rate" in df.columns:
        vr_area = df.groupby("macro_area")["baseline_valid_rate"].mean() * 100
        vr_area = vr_area.sort_values(ascending=False)

        n_areas = len(vr_area)
        fig, ax = plt.subplots(figsize=(max(10, n_areas * 0.85), 5.5))
        palette_6a = sns.color_palette("Blues_d", n_areas)
        bars = ax.bar(vr_area.index, vr_area.values, color=palette_6a,
                      edgecolor="white", width=0.65)
        for bar, val in zip(bars, vr_area.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8)
        ax.set_ylabel("Validity rate (%)")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, max(vr_area.values) * 1.15)
        title_6a = "Tasso di validità per macro area tematica"
        if model_label:
            ax.set_title(f"{title_6a}\n{model_label}",
                         fontsize=14, fontweight="bold", pad=3)
        else:
            ax.set_title(title_6a, fontsize=14, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        sns.despine()
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        fig.savefig(os.path.join(outdir, "fig6a_validity_by_area.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --- 6b  Istogramma verticale: coerenza log-testo per macro area ---
    if "macro_area" in df.columns and "log_consistency_rate" in df.columns:
        lc_area = df.groupby("macro_area")["log_consistency_rate"].mean() * 100
        lc_area = lc_area.sort_values(ascending=False)

        n_areas_lc = len(lc_area)
        fig, ax = plt.subplots(figsize=(max(10, n_areas_lc * 0.85), 5.5))
        palette_6b = sns.color_palette("Greens_d", n_areas_lc)
        bars = ax.bar(lc_area.index, lc_area.values, color=palette_6b,
                      edgecolor="white", width=0.65)
        for bar, val in zip(bars, lc_area.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8)
        ax.set_ylabel("Log-text consistency rate (%)")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, max(lc_area.values) * 1.15 if max(lc_area.values) > 0 else 10)
        title_6b = "Coerenza logit-testo per macro area tematica"
        if model_label:
            ax.set_title(f"{title_6b}\n{model_label}",
                         fontsize=14, fontweight="bold", pad=3)
        else:
            ax.set_title(title_6b, fontsize=14, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        sns.despine()
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        fig.savefig(os.path.join(outdir, "fig6b_log_consistency_by_area.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --- 6c  Bussola politica — un punto per macro area ---
    wd_needed = ["wd_democrat", "wd_republican", "wd_liberal", "wd_conservative"]
    if all(c in df.columns for c in wd_needed) and "macro_area" in df.columns:
        sub_pc = df[wd_needed + ["macro_area"]].dropna(subset=wd_needed)
        if not sub_pc.empty:
            # Media WD per macro area
            area_means = sub_pc.groupby("macro_area")[wd_needed].mean()
            area_means["x_axis"] = area_means["wd_republican"] - area_means["wd_democrat"]
            area_means["y_axis"] = area_means["wd_conservative"] - area_means["wd_liberal"]

            n_areas_pc = len(area_means)
            cmap_pc = sns.color_palette("husl", n_areas_pc)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Sfondo colorato per i 4 quadranti
            xlims_pre = [area_means["x_axis"].min() - 0.005,
                         area_means["x_axis"].max() + 0.005]
            ylims_pre = [area_means["y_axis"].min() - 0.005,
                         area_means["y_axis"].max() + 0.005]
            # Top-right: Liberal–Democrat (blu)
            ax.axvspan(0, xlims_pre[1] * 2, ymin=0.5, ymax=1.0,
                       color="#2166AC", alpha=0.06, zorder=0)
            # Top-left: Liberal–Republican (azzurro)
            ax.axvspan(xlims_pre[0] * 2, 0, ymin=0.5, ymax=1.0,
                       color="#4393C3", alpha=0.06, zorder=0)
            # Bottom-right: Conservative–Democrat (arancione)
            ax.axvspan(0, xlims_pre[1] * 2, ymin=0.0, ymax=0.5,
                       color="#D6604D", alpha=0.06, zorder=0)
            # Bottom-left: Conservative–Republican (rosso)
            ax.axvspan(xlims_pre[0] * 2, 0, ymin=0.0, ymax=0.5,
                       color="#B2182B", alpha=0.06, zorder=0)

            # Scatter: numeri dentro i pallini, senza etichette testuali
            # zorder cresce per coppia: cerchio i → numero i → cerchio i+1 → numero i+1
            legend_handles = []
            for i, (area_name, row) in enumerate(area_means.iterrows()):
                num = i + 1
                ax.scatter(row["x_axis"], row["y_axis"],
                           c=[cmap_pc[i]], s=340, alpha=1.0,
                           edgecolor="white", linewidth=0.8, zorder=10 + 2 * i)
                ax.text(row["x_axis"], row["y_axis"], str(num),
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white", zorder=11 + 2 * i)
                # Crea handle per la legenda
                import matplotlib.patches as mpatches
                legend_handles.append(
                    mpatches.Patch(color=cmap_pc[i], label=f"{num}. {area_name}"))

            # Assi centrali
            ax.axhline(0, color="grey", ls="-", lw=0.8, alpha=0.5, zorder=1)
            ax.axvline(0, color="grey", ls="-", lw=0.8, alpha=0.5, zorder=1)

            # Etichette quadranti vicino agli assi
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            _qbbox = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
            ax.text(xlims[1], ylims[1],
                    "Liberale–Democratico", color="#2166AC",
                    fontweight="bold", fontsize=10, ha="right", va="top",
                    bbox=_qbbox)
            ax.text(xlims[0], ylims[1],
                    "Liberale–Repubblicano", color="#4393C3",
                    fontweight="bold", fontsize=10, ha="left", va="top",
                    bbox=_qbbox)
            ax.text(xlims[1], ylims[0],
                    "Conservatore–Democratico", color="#D6604D",
                    fontweight="bold", fontsize=10, ha="right", va="bottom",
                    bbox=_qbbox)
            ax.text(xlims[0], ylims[0],
                    "Conservatore–Repubblicano", color="#B2182B",
                    fontweight="bold", fontsize=10, ha="left", va="bottom",
                    bbox=_qbbox)

            ax.set_xlabel("← Più vicino ai Repubblicani      Più vicino ai Democratici →",
                          fontsize=11)
            ax.set_ylabel("← Più vicino ai Conservatori      Più vicino ai Liberali →",
                          fontsize=11)

            title_6c = "Bussola politica per macro area tematica"
            if model_label:
                ax.set_title(f"{title_6c}\n{model_label}",
                             fontsize=15, fontweight="bold", pad=12)
            else:
                ax.set_title(title_6c, fontsize=15, fontweight="bold", pad=12)

            ax.legend(handles=legend_handles, loc="center left",
                      bbox_to_anchor=(1.01, 0.5),
                      fontsize=8, frameon=True, title="Macro area")
            sns.despine()
            fig.tight_layout(rect=[0, 0, 0.78, 1])
            fig.savefig(os.path.join(outdir, "fig6c_political_compass.png"),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)

    # --- 6d ---
    wd_cols = ["wd_democrat", "wd_republican", "wd_independent",
               "wd_liberal", "wd_moderate", "wd_conservative"]
    wd_labels = ["Democratico", "Repubblicano", "Indipendente",
                 "Liberale", "Moderato", "Conservatore"]
    
    existing_wd = [c for c in wd_cols if c in df.columns]
    if existing_wd and "macro_area" in df.columns:
        sub = df[["macro_area"] + existing_wd].dropna(subset=existing_wd, how="all")
        if not sub.empty:
            hmap = sub.groupby("macro_area")[existing_wd].mean()
            hmap.columns = [wd_labels[wd_cols.index(c)] for c in existing_wd]
            # ordina per WD media crescente (più vicini in alto)
            hmap = hmap.loc[hmap.mean(axis=1).sort_values().index]

            fig, ax = plt.subplots(figsize=(10, max(5, len(hmap) * 0.55)))
            sns.heatmap(hmap, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        linewidths=0.5, ax=ax, cbar_kws={"label": "Wasserstein distance", "fraction":0.046, "pad":0.04})
            if model_label:
                ax.set_title(f"Distanza WD media per macro area × gruppo demografico\n(valori bassi = maggiore allineamento)\n{model_label}", fontsize=16, fontweight="bold", pad=3)
            else:
                ax.set_title("Distanza WD media per macro area × gruppo demografico\n(valori bassi = maggiore allineamento)", fontsize=16, fontweight="bold")
            ax.set_ylabel("Macro area")
            ax.set_xlabel("Gruppo demografico", labelpad=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            plt.setp(ax.get_yticklabels(), fontsize=10)
            fig.tight_layout(rect=[0, 0.16, 1, 0.99], pad=1.5)
            fig.savefig(os.path.join(outdir, "fig6d_wd_heatmap.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)

# ======================================================================
#  MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Genera grafici dai risultati di analyze.py")
    parser.add_argument("--metrics", type=str, default=DEFAULT_M,
                        help="CSV delle metriche per domanda")
    parser.add_argument("--report", type=str, default=DEFAULT_R,
                        help="CSV del report per topic")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUT,
                        help="Cartella di output per le figure")
    args = parser.parse_args()

    # Carica dati
    print(f"Caricamento metriche da: {args.metrics}")
    df = pd.read_csv(args.metrics)
    print(f"   {len(df)} righe caricate.")

    df_topic = None
    if os.path.exists(args.report):
        print(f"Caricamento report topic da: {args.report}")
        df_topic = pd.read_csv(args.report)
        print(f"   {len(df_topic)} righe caricate.")

    if args.outdir == DEFAULT_OUT or not args.outdir:
        inferred = os.path.join(os.path.dirname(args.metrics), "figure")
        args.outdir = inferred

    # ── Estrai nome modello dal nome del file metriche ──────────────────────
    import re as _re
    _metrics_basename = os.path.basename(args.metrics)
    # Formato atteso: analysis_metrics_{model_name}.csv
    _meta_match = _re.search(r'analysis_metrics_(.+)\.csv', _metrics_basename)
    if _meta_match:
        _model_name = _meta_match.group(1)
        model_label = _model_name.replace('_', '-').title()
    else:
        _model_name = _metrics_basename
        model_label = None

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Output figure in: {args.outdir}\n")

    # Genera tutti i grafici
    # 0.  TABELLA RIASSUNTIVA
    fig_summary_table(df, df_topic, args.outdir, model_label=model_label)
    
    #  1.  CONFRONTO ESPERIMENTI
    fig_validity(df, args.outdir,  model_label=model_label)
    fig_log_coherence(df, args.outdir,  model_label=model_label)
    fig_robustness(df, args.outdir,  model_label=model_label)

    pb_path = os.path.join(os.path.dirname(args.metrics), f"position_bias_{_model_name}.csv")
    if os.path.exists(pb_path):
        df_pb = pd.read_csv(pb_path)
        #  2.  PERMUTAZIONE
        fig_permutation(df_pb, args.outdir, model_label=model_label, df_metrics=df)
        fig_cognitive_map(df, args.outdir, model_label=model_label)
    else:
        print(f"   [SKIP] File position bias non trovato: {pb_path}")

    #  3.  DUPLICAZIONE
    fig_duplication(df, args.outdir, model_label=model_label)

    #  4.  MINACCIA
    fig_threat(df, args.outdir, model_label=model_label)

    #  5.  ALLINEAMENTO UMANO
    fig_alignment_human(df, args.outdir, model_label=model_label)

    #  6.  POLITICA
    fig_political(df, args.outdir, model_label=model_label)

    print(f"\n{'='*60}")
    print(f"FATTO — {len(os.listdir(args.outdir))} figure generate in {args.outdir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
