import pandas as pd, numpy as np, os

models = ['pythia_160m', 'pythia_1b', 'pythia_1.4b', 'pythia_2.8b', 'pythia_6.9b']
base = os.path.dirname(os.path.abspath(__file__))

for m in models:
    path = os.path.join(base, 'risultati', m, f'analysis_metrics_{m}.csv')
    try:
        df = pd.read_csv(path)
        print(f"\n===== {m.upper()} ({len(df)} rows) =====")
        bvr = df["baseline_valid_rate"].mean()*100
        pvr = df["perm_valid_rate"].mean()*100
        dvr = df["dup_valid_rate"].mean()*100
        tvr = df["threat_valid_rate"].mean()*100
        print(f"  Validity: base={bvr:.1f}% perm={pvr:.1f}% dup={dvr:.1f}% threat={tvr:.1f}%")
        
        for jc, jn in [("jsd_permutation","perm"),("jsd_duplication","dup"),("jsd_threat","threat")]:
            s = df[jc].dropna()
            print(f"  JSD {jn}: mean={s.mean():.4f} med={s.median():.4f} std={s.std():.4f}")
        
        ps = df['permutation_stable'].value_counts()
        tot = ps.sum()
        for c in ['Robust','Stable','Position_Bias']:
            n = ps.get(c, 0)
            print(f"  Perm_{c}: {n} ({n/tot*100:.1f}%)")
        
        ds = df['duplication_stable'].value_counts()
        tot_d = ds.sum()
        for c in ['Robust','Stable','Unstable']:
            n = ds.get(c, 0)
            print(f"  Dup_{c}: {n} ({n/tot_d*100:.1f}%)")
        
        tr = df['threat_resistant'].value_counts()
        tot_tr = tr.sum()
        for c in ['Stable','Improved','Degraded']:
            n = tr.get(c, 0)
            print(f"  Threat_{c}: {n} ({n/tot_tr*100:.1f}%)")
        
        for tc in ['jsd_threat_economic','jsd_threat_it_system','jsd_threat_legal']:
            s = df[tc].dropna()
            print(f"  {tc}: mean={s.mean():.4f}")
        
        for tc in ['threat_economic_valid_rate','threat_it_system_valid_rate','threat_legal_valid_rate']:
            s = df[tc].dropna()
            print(f"  {tc}: {s.mean()*100:.1f}%")
        
        for tc in ['threat_economic_log_consistency','threat_it_system_log_consistency','threat_legal_log_consistency']:
            if tc in df.columns:
                s = df[tc].dropna()
                print(f"  {tc}: {s.mean():.3f}")

        if 'most_disruptive_threat' in df.columns:
            md = df['most_disruptive_threat'].dropna().value_counts()
            tot_md = md.sum()
            for t in ['Economic','IT_System','Legal']:
                n = md.get(t, 0)
                print(f"  Disruptive_{t}: {n} ({n/tot_md*100:.1f}%)")
        
        if 'most_effective_threat_validity' in df.columns:
            mv = df['most_effective_threat_validity'].dropna().value_counts()
            tot_mv = mv.sum()
            for t in ['Economic','IT_System','Legal']:
                n = mv.get(t, 0)
                print(f"  EffValid_{t}: {n} ({n/tot_mv*100:.1f}%)")
        
        if 'most_effective_threat_consistency' in df.columns:
            mc = df['most_effective_threat_consistency'].dropna().value_counts()
            tot_mc = mc.sum()
            for t in ['Economic','IT_System','Legal']:
                n = mc.get(t, 0)
                print(f"  EffConsist_{t}: {n} ({n/tot_mc*100:.1f}%)")
        
        lc = df["log_consistency_rate"].dropna().mean()
        print(f"  Log consistency: {lc:.3f}")
        
        al = df['alignment_score'].dropna()
        print(f"  Alignment: mean={al.mean():.3f} med={al.median():.3f} std={al.std():.3f}")
        
        aff = df['political_affinity'].dropna()
        aff = aff[aff != 'None']
        tot_aff = len(aff)
        for g in ['Democrat','Liberal','Republican','Conservative','Moderate','Independent']:
            n = (aff == g).sum()
            print(f"  {g}: {n} ({n/tot_aff*100:.1f}%)")
        dl = ((aff=='Democrat')|(aff=='Liberal')).sum()
        rc = ((aff=='Republican')|(aff=='Conservative')).sum()
        print(f"  D+L: {dl} ({dl/tot_aff*100:.1f}%) R+C: {rc} ({rc/tot_aff*100:.1f}%)")
    except Exception as e:
        print(f"  ERROR for {m}: {e}")

# Position bias
print("\n\n===== POSITION BIAS =====")
for m in models:
    pb_path = os.path.join(base, 'risultati', m, f'position_bias_{m}.csv')
    try:
        pb = pd.read_csv(pb_path)
        uniform = (1.0 / pb["n_options"]).mean()
        first = pb[pb["is_first"]==True]["prob"].mean()
        last = pb[pb["is_last"]==True]["prob"].mean()
        mid = pb[(pb["is_first"]==False)&(pb["is_last"]==False)]["prob"].mean()
        ratio = first / last if last > 0 else float('inf')
        pos_mean = pb.groupby("position_label")["prob"].mean().sort_index()
        print(f"\n  {m}: uniform={uniform:.4f} first={first:.4f} last={last:.4f} mid={mid:.4f} ratio={ratio:.2f}")
        for p, v in pos_mean.items():
            print(f"    {p}: {v:.4f} (delta={v-uniform:+.4f})")
    except Exception as e:
        print(f"  ERROR for {m}: {e}")
