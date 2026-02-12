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
DEFAULT_M  = os.path.join(RIS_DIR, "analysis_metrics_weighted_pythia_160m.csv")
DEFAULT_R  = os.path.join(RIS_DIR, "report_topic_weighted_pythia_160m.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "figure")
# If user doesn't specify --outdir, figures will be stored under the same folder
# as the metrics file, e.g. risultati/pythia_160m/figure


# ======================================================================
#  1.  RESPONSE VALIDITY
# ======================================================================
def fig_validity(df, outdir):
    """
    1a) Bar chart dei validity rate medi per condizione
    1b) Istogramma della distribuzione del baseline_valid_rate per domanda
    """

    # --- 1a  Validity rate per condizione ---
    cols = ["baseline_valid_rate", "perm_valid_rate",
            "dup_valid_rate", "threat_valid_rate"]
    labels = ["Baseline", "Permutation", "Duplication", "Threat"]
    means = [df[c].mean() * 100 for c in cols]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, color=PALETTE_MAIN, edgecolor="white", width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Validity rate (%)")
    ax.set_title("Tasso di validità per condizione sperimentale", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    sns.despine()
    fig.tight_layout()
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
    ax.set_ylabel("Numero di domande")
    ax.set_title("Distribuzione del validity rate per domanda (baseline)", fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout()
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
def fig_robustness(df, outdir):
    """
    2a) Distribuzione JSD per permutazione/duplicazione/threat
    2b) Stacked bar: proportion Robust / Stable / Position_Bias (perm)
    """

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
    ax.set_title("Distribuzione della Jensen-Shannon Divergence per tipo di perturbazione", fontweight="bold")
    ax.set_ylabel("JSD")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2a_jsd_violin.png"), dpi=200)
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
    ax.set_title("Robustezza alla permutazione dell'ordine delle opzioni", fontweight="bold")
    # Posiziona la legenda più in basso e aumenta la dimensione per visibilità
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    # Lascia più spazio in basso per legenda e label
    fig.tight_layout(rect=[0, 0.00, 1, 1])
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
    ax.set_title("Resistenza alla minaccia testuale (Threat)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=10, frameon=True)
    sns.despine(left=True)
    fig.tight_layout(rect=[0, 0.00, 1, 1])
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
#  3.  COERENZA LOG-TESTO
# ======================================================================
def fig_log_coherence(df, outdir):
    """
    3a) Distribuzione della log_consistency_rate
    3b) Scatter validity vs log-consistency
    """

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
    ax.set_title("Distribuzione della coerenza log-testo", fontweight="bold", fontsize=16)
    ax.legend()
    sns.despine()
    fig.tight_layout()
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
        ax.set_title("Coerenza log-testo per area tematica", fontweight="bold", fontsize=16)
        ax.set_ylim(-0.05, 1.05)
        # Increase x-label font a bit for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        sns.despine()
        # Aggiusta margine inferiore per evitare taglio delle etichette
        fig.tight_layout(rect=[0, 0.16, 1, 1], pad=1.2)
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
def fig_alignment(df, outdir):
    """
    4a) Istogramma dell'alignment_score
    4b) Box plot alignment per macro_area
    """

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
    ax.set_title("Distribuzione dell'allineamento con le risposte umane", fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout()
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
        ax.set_title("Allineamento umano per area tematica", fontweight="bold")
        # Increase x-label font a bit for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        sns.despine()
        # Aggiusta margine inferiore per evitare taglio delle etichette
        fig.tight_layout(rect=[0, 0.16, 1, 1], pad=1.2)
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
def fig_political(df, df_topic, outdir):
    """
    5a) Pie chart: distribuzione political_affinity (per domanda)
    5b) Grouped bar: topic allineati per gruppo (dal report topic)
    5c) Heatmap: WD media per macro_area × gruppo
    """

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
    ax.set_title("Affinità politica del modello per domanda", fontsize=16, fontweight="bold")
    fig.tight_layout()
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
        ax.set_title("Distribuzione dei topic per gruppo demografico più allineato", fontweight="bold")
        sns.despine()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig5b_political_topics_bar.png"), dpi=200)
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
            ax.set_title("Distanza WD media per macro area × gruppo demografico\n(valori bassi = maggiore allineamento)", fontsize=16, fontweight="bold")
            ax.set_ylabel("Macro area")
            ax.set_xlabel("Gruppo demografico", labelpad=12)
            # Slightly larger font for demographic groups to improve readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            plt.setp(ax.get_yticklabels(), fontsize=10)
            fig.tight_layout(rect=[0, 0.16, 1, 1], pad=1.5)
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
def fig_summary_table(df, outdir):
    """Genera una tabella riepilogativa con le statistiche chiave."""

    al = df["alignment_score"].dropna()
    log = df["log_consistency_rate"].dropna()
    perm_bias = (df["permutation_stable"] == "Position_Bias").sum()
    n = len(df)
    
    summary = {
        "Metrica": [
            "Numero domande analizzate",
            "Validity rate (baseline)",
            "Validity rate (media 4 condizioni)",
            "JSD permutation (media)",
            "Position bias (%)",
            "JSD duplication (media)",
            "JSD threat (media)",
            "Log-text consistency (media)",
            "Alignment score (media)",
            "Alignment score (mediana)",
            "Affinità democrat + liberal",
            "Affinità republican + conservative",
        ],
        "Valore": [
            f"{n}",
            f"{df['baseline_valid_rate'].mean()*100:.1f}%",
            f"{np.mean([df[c].mean() for c in ['baseline_valid_rate','perm_valid_rate','dup_valid_rate','threat_valid_rate']])*100:.1f}%",
            f"{df['jsd_permutation'].dropna().mean():.4f}",
            f"{perm_bias/n*100:.1f}%",
            f"{df['jsd_duplication'].dropna().mean():.4f}",
            f"{df['jsd_threat'].dropna().mean():.4f}",
            f"{log.mean():.3f}",
            f"{al.mean():.3f}",
            f"{al.median():.3f}",
            f"{_count_aff(df, ['Democrat','Liberal'])}",
            f"{_count_aff(df, ['Republican','Conservative'])}",
        ],
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.set_title("Riepilogo delle metriche — Pythia-160M", fontsize=16,
                 fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig0_summary_table.png"), dpi=200)
    plt.close(fig)


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
    # Crea cartella output (model-specific)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Output figure in: {args.outdir}\n")

    # Genera tutti i grafici
    fig_summary_table(df, args.outdir)
    fig_validity(df, args.outdir)
    fig_robustness(df, args.outdir)
    fig_log_coherence(df, args.outdir)
    fig_alignment(df, args.outdir)
    fig_political(df, df_topic, args.outdir)

    print(f"\n{'='*60}")
    print(f"FATTO — {len(os.listdir(args.outdir))} figure generate in {args.outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
