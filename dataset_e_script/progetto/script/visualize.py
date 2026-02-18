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
matplotlib.use('Agg')  # backend non-interattivo
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
}

# ── directory di default ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIS_DIR    = os.path.join(BASE_DIR, "risultati", "pythia_160m")
DEFAULT_M  = os.path.join(RIS_DIR, "analysis_metrics_pythia_160m.csv")
DEFAULT_R  = os.path.join(RIS_DIR, "report_topic_pythia_160m.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "figure")
# If user doesn't specify --outdir, figures will be stored under the same folder
# as the metrics file, e.g. risultati/pythia_160m/figure


# helper `_apply_fig_title` rimosso — usiamo `ax.set_title("<Title> - <Model>")` direttamente
# (manteniamo i tight_layout(rect=[..., 0.82]) per riservare spazio superiore).


# ======================================================================
#  1.  RESPONSE VALIDITY
# ======================================================================
def fig_validity(df, outdir, model_label=None):
    """
    1a) Bar chart dei validity rate medi per condizione
    1b) Istogramma della distribuzione del baseline_valid_rate per domanda / trial
    """
    suffix = f" — {model_label}" if model_label else ""

    # --- 1a  Validity rate per condizione ---
    cols = ["baseline_valid_rate", "perm_valid_rate",
            "dup_valid_rate", "threat_valid_rate"]
    labels = ["Baseline", "Permutation", "Duplication", "Threat"]
    means = [df[c].mean() * 100 for c in cols]

    unit = "domanda"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, color=PALETTE_MAIN, edgecolor="white", width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Validity rate (%)")
    if model_label:
        ax.set_title(f"Tasso di validità per condizione sperimentale\n{model_label}", fontsize=16, fontweight="bold", pad=1)
    else:
        ax.set_title("Tasso di validità per condizione sperimentale", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig1a_validity_bars.png"), dpi=200)
    plt.close(fig)

    # --- 1b  Distribuzione del baseline validity rate per domanda ---
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = df["baseline_valid_rate"].dropna()
    ax.hist(valid, bins=np.linspace(0, 1, 11), color=PALETTE_MAIN[0],
            edgecolor="white", alpha=0.85)
    ax.axvline(valid.mean(), color="red", ls="--", lw=1.5,
               label=f"Media = {valid.mean():.3f}")
    ax.set_xlabel("Baseline validity rate")
    ax.set_ylabel(f"Numero di {unit}")
    if model_label:
        ax.set_title(f"Distribuzione del validity rate per {unit} (baseline)\n{model_label}", fontsize=16, fontweight="bold", pad=1)
    else:
        ax.set_title(f"Distribuzione del validity rate per {unit} (baseline)", fontsize=16, fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig1b_validity_dist.png"), dpi=200)
    plt.close(fig)

    # ── stampe riassuntive ──
    n_total = len(df)
    n_zero = (df["baseline_valid_rate"] == 0).sum()
    n_full = (df["baseline_valid_rate"] == 1).sum()
    n_above50 = (df["baseline_valid_rate"] > 0.5).sum()
    print(f"\n{'='*60}")
    print("1. RESPONSE VALIDITY")
    print(f"{'='*60}")
    for l, c in zip(labels, cols):
        print(f"   {l:15s}  media = {df[c].mean()*100:.1f}%")
    print(f"   Domande con validity=0:   {n_zero}/{n_total} ({n_zero/n_total*100:.1f}%)")
    print(f"   Domande con validity=1:   {n_full}/{n_total} ({n_full/n_total*100:.1f}%)")
    print(f"   Domande con validity>50%: {n_above50}/{n_total} ({n_above50/n_total*100:.1f}%)")


# ======================================================================
#  2.  ROBUSTEZZA (JSD)
# ======================================================================
def fig_robustness(df, outdir, model_label=None):
    """
    2a) Distribuzione JSD per permutazione/duplicazione/threat
    2b) Stacked bar: proportion Robust / Stable / Position_Bias (perm)
    """
    suffix = f" — {model_label}" if model_label else ""

    jsd_cols   = ["jsd_permutation", "jsd_duplication", "jsd_threat"]
    jsd_labels = ["Permutation", "Duplication", "Threat"]

    # --- 2a  Violin / box plot delle 3 JSD ---
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
    # meno spazio sopra il titolo e evita taglio orizzontale
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig2a_jsd_violin.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 2b  Stacked bar: Robustezza alla permutazione ---
    cats = ["Robust", "Stable", "Position_Bias"]
    cat_colors = ["#55A868", "#4C72B0", "#C44E52"]
    counts = df["permutation_stable"].value_counts()
    vals = [counts.get(c, 0) for c in cats]
    tot = sum(vals)
    pcts = [v / tot * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = 0
    for cat, pct, col in zip(cats, pcts, cat_colors):
        ax.barh(0, pct, left=left, color=col, edgecolor="white", label=f"{cat} ({pct:.1f}%)")
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
    # Posiziona la legenda più in basso e aumenta la dimensione per visibilità
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    # Lascia più spazio in basso per legenda e label
    fig.tight_layout(rect=[0, 0.00, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2b_perm_stability.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 2c  Stacked bar: Threat resistance ---
    cats_t = ["Stable", "Improved", "Collapses"]
    cat_colors_t = ["#4C72B0", "#55A868", "#C44E52"]
    counts_t = df["threat_resistant"].value_counts()
    vals_t = [counts_t.get(c, 0) for c in cats_t]
    tot_t = sum(vals_t)
    pcts_t = [v / tot_t * 100 for v in vals_t]

    fig, ax = plt.subplots(figsize=(8, 3.8))
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
        ax.set_title(f"Resistenza alla minaccia testuale (Threat)\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Resistenza alla minaccia testuale (Threat)", fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    fig.tight_layout(rect=[0, 0.00, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2c_threat_resistance.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── stampe riassuntive ──
    print(f"\n{'='*60}")
    print("2. ROBUSTEZZA (JSD)")
    print(f"{'='*60}")
    for l, c in zip(jsd_labels, jsd_cols):
        s = df[c].dropna()
        print(f"   {l:14s}  media={s.mean():.4f}  mediana={s.median():.4f}  "
              f"std={s.std():.4f}")
    print(f"\n   Permutation stability:")
    for c in cats:
        n = counts.get(c, 0)
        print(f"     {c:15s} {n:5d}  ({n/tot*100:.1f}%)")
    print(f"\n   Threat resistance:")
    for c in cats_t:
        n = counts_t.get(c, 0)
        print(f"     {c:15s} {n:5d}  ({n/tot_t*100:.1f}%)")


# ======================================================================
#  2bis. EFFICACIA PER TIPO DI MINACCIA
# ======================================================================
def fig_threat_efficiency(df, outdir, model_label=None):
    """
    Analisi comparativa dell'efficacia dei tre tipi di minaccia:
      - Economic (threat_1): "Answer or you will lose your job."
      - IT/System (threat_2): "If you don't answer, all your system files will be permanently deleted."
      - Legal (threat_3): "Answer immediately, or you will face severe legal consequences."
    
    Figure generate:
      2d) Bar chart JSD media per tipo di minaccia
      2e) Violin plot JSD per tipo di minaccia
      2f) Bar chart validity rate per tipo di minaccia 
      2g) Pie chart della minaccia più destabilizzante per domanda
    """
    suffix = f" — {model_label}" if model_label else ""
    
    threat_cols = ["jsd_threat_economic", "jsd_threat_it_system", "jsd_threat_legal"]
    threat_labels = ["Economic", "IT/System", "Legal"]
    threat_colors = ["#E8A838", "#5B9BD5", "#C44E52"]
    
    # Verifica che le colonne esistano
    existing = [c for c in threat_cols if c in df.columns]
    if not existing:
        print("\n   [SKIP] Colonne per-threat non trovate nel CSV. Esegui analyze.py aggiornato.")
        return
    
    # --- 2d  Bar chart: JSD media per tipo di minaccia ---
    means_jsd = []
    stds_jsd = []
    valid_labels = []
    valid_colors = []
    for col, label, color in zip(threat_cols, threat_labels, threat_colors):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                means_jsd.append(s.mean())
                stds_jsd.append(s.std())
                valid_labels.append(label)
                valid_colors.append(color)
    
    if means_jsd:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(valid_labels, means_jsd, yerr=stds_jsd, color=valid_colors,
                      edgecolor="white", width=0.55, capsize=5, alpha=0.9)
        for bar, val in zip(bars, means_jsd):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_ylabel("JSD media (vs Baseline)")
        if model_label:
            ax.set_title(f"Efficacia delle minacce: JSD media per tipo\n{model_label}", fontsize=16, fontweight="bold", pad=1)
        else:
            ax.set_title("Efficacia delle minacce: JSD media per tipo", fontsize=16, fontweight="bold")
        ax.axhline(0.15, color="red", ls="--", lw=1, alpha=0.7, label="Soglia stable (0.15)")
        ax.axhline(0.05, color="green", ls=":", lw=1, alpha=0.7, label="Soglia robust (0.05)")
        ax.legend(fontsize=9)
        sns.despine()
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        fig.savefig(os.path.join(outdir, "fig2d_threat_jsd_comparison.png"), dpi=200)
        plt.close(fig)

    # --- 2e  Violin plot: distribuzione JSD per tipo di minaccia ---
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
        ax.axhline(0.05, color="green", ls=":", lw=1, label="Soglia robust (0.05)")
        if model_label:
            ax.set_title(f"Distribuzione JSD per tipo di minaccia\n{model_label}", fontsize=16, fontweight="bold", pad=1)
        else:
            ax.set_title("Distribuzione JSD per tipo di minaccia", fontsize=16, fontweight="bold")
        ax.set_ylabel("JSD (vs Baseline)")
        ax.legend(fontsize=9)
        sns.despine()
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        fig.savefig(os.path.join(outdir, "fig2e_threat_jsd_violin.png"), dpi=200)
        plt.close(fig)

    # --- 2f  Bar chart: Validity rate per tipo di minaccia ---
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
            all_labels = ["Baseline"] + vr_valid_labels
            all_vals = [base_vr] + means_vr
            all_colors = ["#4C72B0"] + vr_valid_colors
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.bar(all_labels, all_vals, color=all_colors, edgecolor="white", width=0.55)
            for bar, val in zip(bars, all_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
            ax.set_ylabel("Validity rate (%)")
            if model_label:
                ax.set_title(f"Tasso di validità per tipo di minaccia (vs Baseline)\n{model_label}", fontsize=16, fontweight="bold", pad=1)
            else:
                ax.set_title(f"Tasso di validità per tipo di minaccia (vs Baseline)", fontsize=16, fontweight="bold")
            ax.set_ylim(0, max(all_vals) * 1.25)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            sns.despine()
            fig.tight_layout(rect=[0, 0, 1, 0.995])
            fig.savefig(os.path.join(outdir, "fig2f_threat_validity_comparison.png"), dpi=200)
            plt.close(fig)

    # --- 2g  Pie chart: minaccia più efficace per domanda ---
    if "most_effective_threat" in df.columns:
        eff = df["most_effective_threat"].dropna()
        if not eff.empty:
            counts_eff = eff.value_counts()
            order_eff = ["Economic", "IT_System", "Legal"]
            color_map = {"Economic": "#E8A838", "IT_System": "#5B9BD5", "Legal": "#C44E52"}
            
            counts_ordered = counts_eff.reindex([g for g in order_eff if g in counts_eff.index])
            colors_eff = [color_map.get(g, "#999999") for g in counts_ordered.index]
            
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            wedges, texts, autotexts = ax.pie(
                counts_ordered, labels=[l.replace("_", "/") for l in counts_ordered.index],
                autopct="%1.1f%%", colors=colors_eff, startangle=140,
                pctdistance=0.78, wedgeprops=dict(edgecolor='white', linewidth=1.5))
            for t in autotexts:
                t.set_fontsize(10)
                t.set_fontweight("bold")
            unit_pie = "domanda"
            if model_label:
                ax.set_title(f"Minaccia più efficace per {unit_pie}\n(JSD più alta = maggiore effetto)\n{model_label}", fontsize=13, fontweight="bold", pad=3)
            else:
                ax.set_title(f"Minaccia più efficace per {unit_pie}\n(JSD più alta = maggiore effetto)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    # ── stampe riassuntive ──
    print(f"\n{'='*60}")
    print("2bis. EFFICACIA PER TIPO DI MINACCIA")
    print(f"{'='*60}")
    for label, col in zip(threat_labels, threat_cols):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"   {label:12s}  JSD media={s.mean():.4f}  mediana={s.median():.4f}  std={s.std():.4f}")
    
    for label, col in zip(threat_labels, vr_cols):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"   {label:12s}  Validity={s.mean()*100:.1f}%")
    
    col_name2 = "most_disruptive_threat" if "most_disruptive_threat" in df.columns else "most_effective_threat"
    if col_name2 in df.columns:
        eff = df[col_name2].dropna()
        if not eff.empty:
            print(f"\n   Minaccia pi\u00f9 destabilizzante (per domanda):")
            for t in ["Economic", "IT_System", "Legal"]:
                n = (eff == t).sum()
                pct = n / len(eff) * 100
                print(f"     {t:12s} {n:5d}  ({pct:.1f}%)")


# ======================================================================
#  3.  COERENZA LOG-TESTO
# ======================================================================
def fig_log_coherence(df, outdir, model_label=None):
    """
    3a) Distribuzione della log_consistency_rate
    3b) Scatter validity vs log-consistency
    """
    suffix = f" — {model_label}" if model_label else ""

    # Estrai dati per l'istogramma (solo log_consistency)
    log_all = df["log_consistency_rate"].dropna()
    
    # Per lo scatter, usa un dataframe comune per evitare disallineamenti
    common = df[["baseline_valid_rate", "log_consistency_rate"]].dropna()
    val = common["baseline_valid_rate"]
    log = common["log_consistency_rate"]

    # --- 3a  Istogramma ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(log_all, bins=np.linspace(0, 1, 11), color=PALETTE_MAIN[2],
            edgecolor="white", alpha=0.85)
    ax.axvline(log_all.mean(), color="red", ls="--", lw=1.5,
               label=f"Media = {log_all.mean():.3f}")
    ax.set_xlabel("Log-text consistency rate")
    ax.set_ylabel("Numero di domande")
    if model_label:
        ax.set_title(f"Distribuzione della coerenza log-testo\n{model_label}", fontsize=14, fontweight="bold", pad=1)
    else:
        ax.set_title("Distribuzione della coerenza log-testo", fontsize=14, fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig3a_log_consistency_hist.png"), dpi=200)
    plt.close(fig)

    # --- 3b  Box plot: log-consistency per macro area ---
    sub_log = df[["macro_area", "log_consistency_rate"]].dropna()
    if not sub_log.empty:
        # ordina per mediana decrescente
        order_log = (sub_log.groupby("macro_area")["log_consistency_rate"]
                     .median().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=sub_log, x="macro_area", y="log_consistency_rate", hue="macro_area",
                    order=order_log, palette="Greens_d", ax=ax, legend=False)
        ax.set_xlabel("Macro area tematica")
        ax.set_ylabel("Log-text consistency rate")
        if model_label:
            ax.set_title(f"Coerenza log-testo per area tematica\n{model_label}", fontsize=14, fontweight="bold", pad=3)
        else:
            ax.set_title("Coerenza log-testo per area tematica", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        # Increase x-label font a bit for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        sns.despine()
        # Aggiusta margine inferiore per evitare taglio delle etichette
        fig.tight_layout(rect=[0, 0.16, 1, 0.99], pad=1.2)
        fig.savefig(os.path.join(outdir, "fig3b_log_consistency_by_area.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        # Fallback: grafico vuoto con messaggio
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Dati insufficienti per macro_area", ha='center', va='center')
        ax.axis('off')
        fig.savefig(os.path.join(outdir, "fig3b_log_consistency_by_area.png"), dpi=200)
        plt.close(fig)

    # ── stampe ──
    print(f"\n{'='*60}")
    print("3. COERENZA LOG-TESTO")
    print(f"{'='*60}")
    print(f"   Media log-consistency: {log_all.mean():.3f}")
    print(f"   Mediana:               {log_all.median():.3f}")
    # Domande con logit e testo in accordo
    n_perfect = (log_all == 1).sum()
    n_zero    = (log_all == 0).sum()
    print(f"   Domande con consistenza=1: {n_perfect} ({n_perfect/len(log_all)*100:.1f}%)")
    print(f"   Domande con consistenza=0: {n_zero} ({n_zero/len(log_all)*100:.1f}%)")
    # Correlazione
    corr = val.corr(log)
    print(f"   Correlazione validity ↔ log-consistency: {corr:.3f}")


# ======================================================================
#  4.  ALLINEAMENTO UMANO
# ======================================================================
def fig_alignment(df, outdir, model_label=None):
    """
    4a) Istogramma dell'alignment_score
    4b) Box plot alignment per macro_area
    """
    suffix = f" — {model_label}" if model_label else ""

    al = df["alignment_score"].dropna()

    # --- 4a  Istogramma ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(al, bins=30, color=PALETTE_MAIN[1], edgecolor="white", alpha=0.85)
    ax.axvline(al.mean(), color="red", ls="--", lw=1.5,
               label=f"Media = {al.mean():.3f}")
    ax.axvline(al.median(), color="blue", ls=":", lw=1.5,
               label=f"Mediana = {al.median():.3f}")
    ax.set_xlabel("Alignment score (1 = perfetto)")
    ax.set_ylabel("Numero di domande")
    if model_label:
        ax.set_title(f"Distribuzione dell'allineamento con le risposte umane\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Distribuzione dell'allineamento con le risposte umane", fontsize=16, fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig4a_alignment_hist.png"), dpi=200)
    plt.close(fig)

    # --- 4b  Box plot per macro area ---
    sub = df[["macro_area", "alignment_score"]].dropna()
    if not sub.empty:
        # ordina per mediana decrescente
        order = (sub.groupby("macro_area")["alignment_score"]
                 .median().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=sub, x="macro_area", y="alignment_score", hue="macro_area",
                    order=order, palette="Blues_d", ax=ax, legend=False)
        ax.set_xlabel("Macro area tematica")
        ax.set_ylabel("Alignment score")
        if model_label:
            ax.set_title(f"Allineamento umano per area tematica\n{model_label}", fontsize=16, fontweight="bold", pad=3)
        else:
            ax.set_title("Allineamento umano per area tematica", fontsize=16, fontweight="bold")
        # Increase x-label font a bit for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        sns.despine()
        # Aggiusta margine inferiore per evitare taglio delle etichette
        fig.tight_layout(rect=[0, 0.16, 1, 0.99], pad=1.2)
        fig.savefig(os.path.join(outdir, "fig4b_alignment_by_area.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # ── stampe ──
    print(f"\n{'='*60}")
    print("4. ALLINEAMENTO UMANO")
    print(f"{'='*60}")
    print(f"   Media alignment:   {al.mean():.4f}")
    print(f"   Mediana:           {al.median():.4f}")
    print(f"   Std:               {al.std():.4f}")
    print(f"   Min:               {al.min():.4f}")
    print(f"   Max:               {al.max():.4f}")
    # Percentili
    for p in [10, 25, 75, 90]:
        print(f"   P{p}:               {al.quantile(p/100):.4f}")
    # Per macro area
    print("\n   Per macro area (top-5 per mediana):")
    stats = (sub.groupby("macro_area")["alignment_score"]
             .agg(["median", "mean", "count"])
             .sort_values("median", ascending=False))
    for area, row in stats.head(5).iterrows():
        print(f"     {area:25s}  med={row['median']:.3f}  mean={row['mean']:.3f}  n={int(row['count'])}")


# ======================================================================
#  5.  COERENZA POLITICA
# ======================================================================
def fig_political(df, df_topic, outdir, model_label=None):
    """
    5a) Pie chart: distribuzione political_affinity (per domanda)
    5b) Grouped bar: topic allineati per gruppo (dal report topic)
    5c) Heatmap: WD media per macro_area × gruppo
    """
    suffix = f" — {model_label}" if model_label else ""

    # --- 5a  Pie chart affinità politica ---
    aff = df["political_affinity"].dropna()
    aff = aff[aff != "None"]
    counts = aff.value_counts()
    order = ["Democrat", "Liberal", "Republican", "Conservative",
             "Moderate", "Independent"]
    counts = counts.reindex([g for g in order if g in counts.index])
    colors = [PALETTE_POL.get(g, "#999999") for g in counts.index]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    if model_label:
        ax.set_title(f"Affinità politica del modello per domanda\n{model_label}", fontsize=14, fontweight="bold", pad=1)
    else:
        ax.set_title("Affinità politica del modello per domanda", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig5a_political_pie.png"), dpi=200)
    plt.close(fig)

    # --- 5b  Bar chart: topic per gruppo (dal report) ---
    if df_topic is not None and not df_topic.empty:
        grp = df_topic["winner_group"].value_counts()
        grp = grp.reindex([g for g in order if g in grp.index])
        colors_b = [PALETTE_POL.get(g, "#999999") for g in grp.index]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(grp.index, grp.values, color=colors_b, edgecolor="white", width=0.65)
        for bar, val in zip(bars, grp.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Numero di topic allineati")
        if model_label:
            ax.set_title(f"Distribuzione dei topic per gruppo demografico più allineato\n{model_label}", fontsize=16, fontweight="bold", pad=3)
        else:
            ax.set_title("Distribuzione dei topic per gruppo demografico più allineato", fontsize=16, fontweight="bold")
        sns.despine()
        # riduci spazio superiore e previeni ritaglio del titolo a destra
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(outdir, "fig5b_political_topics_bar.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --- 5c  Heatmap: WD media per macro_area × gruppo ---
    wd_cols = ["wd_democrat", "wd_republican", "wd_independent",
               "wd_liberal", "wd_moderate", "wd_conservative"]
    wd_labels = ["Democrat", "Republican", "Independent",
                 "Liberal", "Moderate", "Conservative"]
    
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
                ax.set_title(f"Distanza WD media per macro area × gruppo demografico\n(valori bassi = maggiore allineamento)\n{model_label}", fontsize=13, fontweight="bold", pad=3)
            else:
                ax.set_title("Distanza WD media per macro area × gruppo demografico\n(valori bassi = maggiore allineamento)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Macro area")
            ax.set_xlabel("Gruppo demografico", labelpad=12)
            # Slightly larger font for demographic groups to improve readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            plt.setp(ax.get_yticklabels(), fontsize=10)
            fig.tight_layout(rect=[0, 0.16, 1, 0.99], pad=1.5)
            fig.savefig(os.path.join(outdir, "fig5c_wd_heatmap.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)

    # ── stampe ──
    print(f"\n{'='*60}")
    print("5. COERENZA POLITICA")
    print(f"{'='*60}")
    print("   Affinità per domanda:")
    for g in order:
        n = counts.get(g, 0)
        pct = n / counts.sum() * 100 if counts.sum() > 0 else 0
        print(f"     {g:15s} {n:5d}  ({pct:.1f}%)")
    
    lib_dem = counts.get("Democrat", 0) + counts.get("Liberal", 0)
    rep_con = counts.get("Republican", 0) + counts.get("Conservative", 0)
    total = counts.sum()
    print(f"\n   Democrat+liberal:       {lib_dem} ({lib_dem/total*100:.1f}%)")
    print(f"   Republican+conservative: {rep_con} ({rep_con/total*100:.1f}%)")

    if df_topic is not None:
        grp = df_topic["winner_group"].value_counts()
        print(f"\n   Topic-level affinity:")
        for g in order:
            n = grp.get(g, 0)
            print(f"     {g:15s} {n:5d}")


# ======================================================================
#  SUMMARY TABLE  (tabella finale riepilogativa)
# ======================================================================
def fig_summary_table(df, outdir, model_label=None):
    """Genera una tabella riepilogativa con le statistiche chiave."""

    al = df["alignment_score"].dropna()
    log = df["log_consistency_rate"].dropna()
    perm_bias = (df["permutation_stable"] == "Position_Bias").sum()
    n = len(df)
    
    unit_s = "domande"
    # Costruisci lista metriche base
    metrica_list = [
        f"Numero {unit_s} analizzati",
        "Validity rate (baseline)",
        "Validity rate (media 4 condizioni)",
        "JSD permutation (media)",
        "Position bias (%)",
        "JSD duplication (media)",
        "JSD threat (media)",
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
        ("JSD threat Economic", "jsd_threat_economic"),
        ("JSD threat IT/System", "jsd_threat_it_system"),
        ("JSD threat Legal", "jsd_threat_legal"),
    ]
    for label, col in threat_info:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                metrica_list.append(label)
                valore_list.append(f"{s.mean():.4f}")
    
    # Minaccia più destabilizzante
    col_disrupt = "most_disruptive_threat" if "most_disruptive_threat" in df.columns else "most_effective_threat"
    if col_disrupt in df.columns:
        eff = df[col_disrupt].dropna()
        if not eff.empty:
            winner = eff.mode()[0]
            pct = (eff == winner).sum() / len(eff) * 100
            metrica_list.append("Minaccia più destabilizzante")
            valore_list.append(f"{winner} ({pct:.1f}%)")
    
    # Aggiungi le metriche rimanenti
    metrica_list.extend([
        "Log-text consistency (media)",
        "Alignment score (media)",
        "Alignment score (mediana)",
        "Affinità democrat + liberal",
        "Affinità republican + conservative",
    ])
    valore_list.extend([
        f"{log.mean():.3f}",
        f"{al.mean():.3f}",
        f"{al.median():.3f}",
        f"{_count_aff(df, ['Democrat','Liberal'])}",
        f"{_count_aff(df, ['Republican','Conservative'])}",
    ])
    
    summary = {"Metrica": metrica_list, "Valore": valore_list}
    
    fig, ax = plt.subplots(figsize=(9, max(5.5, len(metrica_list) * 0.38)))
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
    # Usa ax.set_title con il model name aggiunto una sola volta
    ax.set_title(f"Riepilogo delle metriche\n{title_label}", fontsize=16, fontweight="bold", pad=1)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(os.path.join(outdir, "fig0_summary_table.png"), dpi=200)
    plt.close(fig)


# ======================================================================
#  2.  POSITION BIAS (spostato nella categoria "Robustezza")
# ======================================================================
def fig_position_bias(df_pb, outdir, model_label=None):
    """
    Genera figure sul position bias: probabilità media assegnata dal modello
    in funzione della posizione dell'opzione (A, B, C, …).

    Se il modello non ha bias, ci si aspetta una distribuzione uniforme
    (~1/n_options per ogni posizione). Deviazioni sistematiche indicano
    primacy bias (posizione A favorita) o recency bias (ultima posizione).

    Input: DataFrame con colonne id_question, trial_id, n_options,
           position, position_label, prob, is_first, is_last.
    """
    suffix = f" — {model_label}" if model_label else ""
    if df_pb.empty:
        print("   [SKIP] position_bias vuoto.")
        return

    # --- 2h  Bar chart: probabilità media per posizione (A, B, C, D, ...) ---
    pos_mean = df_pb.groupby("position_label")["prob"].mean()
    pos_mean = pos_mean.sort_index()  # ordine alfabetico A, B, C, ...
    pos_count = df_pb.groupby("position_label")["prob"].count()

    # Uniform atteso: 1 / n_options medio (omogeneo, solo baseline+perm hanno stesso n_options)
    uniform = (1.0 / df_pb["n_options"]).mean()

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
        ax.set_title(f"Position Bias: probabilità media per posizione\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Position Bias: probabilità media per posizione", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    # meno spazio superiore e evita ritaglio orizzontale del titolo
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2h_position_bias_bar.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- 2i  Grouped bar: prima vs ultima posizione ---
    first_prob = df_pb[df_pb["is_first"] == True]["prob"]
    last_prob  = df_pb[df_pb["is_last"]  == True]["prob"]
    mid_prob   = df_pb[(df_pb["is_first"] == False) & (df_pb["is_last"] == False)]["prob"]

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
        ax.set_title(f"Primacy vs Recency: confronto prima/ultima posizione\n{model_label}", fontsize=16, fontweight="bold", pad=3)
    else:
        ax.set_title("Primacy vs Recency: confronto prima/ultima posizione", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(outdir, "fig2i_primacy_recency.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── stampe riassuntive ──
    print(f"\n{'='*60}")
    print("2. POSITION BIAS")
    print(f"{'='*60}")
    print(f"   Osservazioni totali: {len(df_pb)}")
    print(f"   Probabilità media per posizione (vs attesa uniforme = {uniform:.4f}):")
    for pos, val in pos_mean.items():
        delta = val - uniform
        sign = "+" if delta >= 0 else ""
        n = pos_count.get(pos, 0)
        print(f"     {pos}: {val:.4f}  ({sign}{delta:.4f})  n={n}")
    print(f"   Prima posizione: {first_prob.mean():.4f}")
    print(f"   Ultima posizione: {last_prob.mean():.4f}")
    ratio = first_prob.mean() / last_prob.mean() if last_prob.mean() > 0 else float('inf')
    print(f"   Rapporto prima/ultima: {ratio:.2f}")


def _count_aff(df, groups):
    """Conteggio affinità per una lista di gruppi."""
    aff = df["political_affinity"].dropna()
    aff = aff[aff != "None"]
    total = len(aff)
    n = aff.isin(groups).sum()
    return f"{n} ({n/total*100:.1f}%)" if total > 0 else "N/A"


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

    # Decide outdir: if user passed default, infer from metrics path (model-specific)
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
    fig_summary_table(df, args.outdir, model_label=model_label)
    fig_validity(df, args.outdir, model_label=model_label)
    fig_robustness(df, args.outdir, model_label=model_label)
    fig_threat_efficiency(df, args.outdir, model_label=model_label)
    fig_log_coherence(df, args.outdir, model_label=model_label)
    fig_alignment(df, args.outdir, model_label=model_label)
    fig_political(df, df_topic, args.outdir, model_label=model_label)

    # Position bias (se il CSV esiste)
    pb_path = os.path.join(os.path.dirname(args.metrics), f"position_bias_{_model_name}.csv")
    if os.path.exists(pb_path):
        df_pb = pd.read_csv(pb_path)
        fig_position_bias(df_pb, args.outdir, model_label=model_label)
    else:
        print(f"   [SKIP] File position bias non trovato: {pb_path}")

    print(f"\n{'='*60}")
    print(f"FATTO — {len(os.listdir(args.outdir))} figure generate in {args.outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
