#!/usr/bin/env python3
"""
COMNET-D-26-06148 -- Revision 2 verification script.

Regenerates the quantities introduced in the R2 revision, from the archived
revision-run artifacts only. Nothing is retrained and no experiment is re-run.

Inputs  : <run-dir>/R9_prevalence_budget/prevalence_budget_grid.csv
          <run-dir>/E1_detector_family/label_free_by_family.csv
          <run-dir>/E2_metrics/true_metrics_by_family.csv
          <run-dir>/predictions/lightgbm/UNR_*_0.0144/*.parquet
Outputs : <out-dir>/*.csv  plus r2_all_intervals.csv (every interval quoted in the paper)

Assertions test published point estimates and structural invariants, not
bit-exact floats.
"""
import argparse, sys
from math import comb, sqrt
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RATE, TAU, N_POOL = 0.0144, 0.75, 20000
N_BUILD, K_BUILD = 80000, 1152   # construction pool: int(round(80000*0.0144)) attacks, fixed by construction
BUDGETS = [10, 20, 50, 100]
KEY = ['config', 'source', 'model_seed', 'pool_seed', 'canary_seed']
SEEDS = dict(cluster=101, paired=202, lodo=303, auprc=404)

def eq1(r, B):           return 1 - (1 - r) ** B - r ** B
def hyper(N, K, B):      return 1 - comb(N - K, B) / comb(N, B) - (comb(K, B) / comb(N, B) if K >= B else 0.0)
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)

def load_selection_module(path):
    """Load sel_strat/sel_fp2 from the master run script so the decomposition identity
    can be proved by execution rather than assumed from the sign of a difference."""
    import types
    try:
        src = Path(path).read_text()
        blk = src[src.index('def bins_of'):src.index('def canary_metrics')]
        from sklearn.cluster import MiniBatchKMeans
        mod = types.ModuleType('sel'); mod.__dict__.update(np=np, MiniBatchKMeans=MiniBatchKMeans)
        exec(blk, mod.__dict__); return mod
    except Exception as e:
        print(f'  [warn] selection module not loaded ({e}); index-identity proof skipped'); return None

def main(run_dir, out_dir, reps_cluster, reps_paired, reps_lodo, reps_auprc, master=None):
    run, out = Path(run_dir), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sel_mod = load_selection_module(master) if master else None
    g  = pd.read_csv(run / 'R9_prevalence_budget/prevalence_budget_grid.csv')
    e1 = pd.read_csv(run / 'E1_detector_family/label_free_by_family.csv')
    e2 = pd.read_csv(run / 'E2_metrics/true_metrics_by_family.csv')
    pt = g[(g.alpha == 0) & (np.isclose(g.rate, RATE))]
    rows_all = []

    # ---------- 1. random computability: Eq.(1), Wilson, cluster bootstrap, hypergeometric
    rnd = pt[pt.strategy == 'random']
    rng = np.random.default_rng(SEEDS['cluster']); rec = []
    for B in BUDGETS:
        s = rnd[rnd.budget == B]
        grp = s.groupby(['pool_seed', 'canary_seed']).computable
        assert (grp.nunique() == 1).all(), "random computability varies within (pool_seed,canary_seed)"
        u  = grp.first().reset_index()
        k, n = int(u.computable.sum()), len(u)
        wl, wh = wilson(k, n)
        ps = sorted(u.pool_seed.unique())
        clus = [u[u.pool_seed == q].computable.values for q in ps]      # equal-sized clusters
        M = np.array(clus); nc, sz = M.shape
        ci = rng.integers(0, nc, size=(reps_cluster, nc))
        wi = rng.integers(0, sz, size=(reps_cluster, nc, sz))
        bs = M[ci[:, :, None], wi].reshape(reps_cluster, -1).mean(axis=1)
        cl, ch = np.percentile(bs, [2.5, 97.5])
        K = int(round(RATE * N_POOL))
        # Sampling is nested and uniform at every stage (80k construction pool -> 20k evaluation
        # pool -> B-flow canary), so a uniform canary is marginally B flows drawn without
        # replacement from the construction pool, whose attack count is fixed at K_BUILD.
        marg = hyper(N_BUILD, K_BUILD, B)
        rec.append(dict(budget=B, distinct_outcomes=n, computable=k, estimate=k / n,
                        wilson_lo=wl, wilson_hi=wh, cluster_lo=cl, cluster_hi=ch,
                        eq1=eq1(RATE, B),
                        marginal_hyper_build_pool=marg, marginal_minus_eq1=marg - eq1(RATE, B),
                        hypergeometric=hyper(N_POOL, K, B),
                        hyper_minus_eq1=hyper(N_POOL, K, B) - eq1(RATE, B),
                        eq1_in_wilson=bool(wl <= eq1(RATE, B) <= wh),
                        eq1_in_cluster=bool(cl <= eq1(RATE, B) <= ch)))
    d1 = pd.DataFrame(rec); d1.to_csv(out / 'r2_random_computability_intervals.csv', index=False)
    assert d1.eq1_in_wilson.all() and d1.eq1_in_cluster.all(), "Eq.(1) outside an interval"
    assert d1.marginal_minus_eq1.abs().max() < 0.001, \
        "marginal finite-population baseline no longer within 0.001 of Eq. (1)"
    for _, r in d1.iterrows():
        rows_all.append(dict(quantity=f"random computability B={r.budget}", estimate=r.estimate,
                             lo=r.wilson_lo, hi=r.wilson_hi, method="Wilson (15 outcomes)"))
        rows_all.append(dict(quantity=f"random computability B={r.budget}", estimate=r.estimate,
                             lo=r.cluster_lo, hi=r.cluster_hi, method="cluster bootstrap (pool_seed)"))

    # ---------- 2. paired strat_fp2 - random, hierarchical bootstrap
    rng = np.random.default_rng(SEEDS['paired']); rec = []
    for B in BUDGETS:
        a = pt[(pt.strategy == 'strat_fp2') & (pt.budget == B)][KEY + ['computable']].rename(columns={'computable': 'fp2'})
        b = pt[(pt.strategy == 'random')    & (pt.budget == B)][KEY + ['computable']].rename(columns={'computable': 'rnd'})
        m = a.merge(b, on=KEY); assert len(m) == len(a) == len(b), "pairing lost rows"
        m['d'] = m.fp2 - m.rnd
        # The R9 target pool is seeded by pool_seed and rate only (see run_r9 in the master
        # script), so all sources and model seeds at a given pool_seed share one target pool.
        # There are three independent target pools, not 180 independent rows; the outer
        # bootstrap therefore resamples pool_seed and carries every row with it.
        pools = sorted(m.pool_seed.unique())
        blocks = [m[m.pool_seed == q].d.values for q in pools]
        shp = {len(x) for x in blocks}
        assert len(shp) == 1, f"ragged pool blocks: {shp}"
        M = np.array(blocks); nc, nn = M.shape
        ci = rng.integers(0, nc, size=(reps_paired, nc))
        wi = rng.integers(0, nn, size=(reps_paired, nc, nn))
        bs = M[ci[:, :, None], wi].reshape(reps_paired, -1).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec.append(dict(budget=B, n_pairs=len(m), n_independent_pools=len(pools),
                        mean_diff=m.d.mean(), ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0)))
    d2 = pd.DataFrame(rec); d2.to_csv(out / 'r2_paired_fp2_vs_random.csv', index=False)
    assert d2.excludes_zero.sum() >= 3, "fewer than three paired intervals exclude zero"
    assert (d2.n_independent_pools == 3).all(), "unexpected number of independent target pools"
    for _, r in d2.iterrows():
        rows_all.append(dict(quantity=f"strat_fp2 - random computability B={int(r.budget)}",
                             estimate=r.mean_diff, lo=r.ci_lo, hi=r.ci_hi, method="hierarchical bootstrap"))

    # ---------- 3. component decomposition of strat_fp2 at B=20
    # sel_fp2(B=20, rho=0.5) calls sel_strat(B=10, same seed); strat_score_W at B=10 is that same
    # selection (the `weighted` flag changes the returned weights, not the chosen indices).
    def g2(s, b, pre):
        return pt[(pt.strategy == s) & (pt.budget == b)][KEY + ['n_attack', 'computable', 'true_auroc']] \
                 .rename(columns={'n_attack': 'a_' + pre, 'computable': 'c_' + pre})
    m = g2('strat_fp2', 20, 'fp2').merge(g2('strat_score_W', 10, 's10').drop(columns='true_auroc'), on=KEY) \
                                  .merge(g2('random', 20, 'rnd').drop(columns='true_auroc'), on=KEY)
    m['a_fp'] = m.a_fp2 - m.a_s10
    if sel_mod is not None:                       # exact index-identity proof, not just a sign check
        gg = np.random.default_rng(0); okid = 0; ntr = 0
        for t in range(60):
            nn = int(gg.integers(500, 3000)); dd = int(gg.integers(5, 14))
            Xs = gg.normal(size=(nn, dd)); ps_ = gg.random(nn)
            if t % 3 == 0: ps_ = np.round(ps_, 2)        # force score ties
            for cs in range(5):
                i2, _ = sel_mod.sel_fp2(Xs, ps_, 20, cs, rho=.5)
                i1, _ = sel_mod.sel_strat(Xs, ps_, 10, cs, True)
                ntr += 1; okid += int(set(i1.tolist()) == set(i2[:len(i1)].tolist()))
        assert okid == ntr, f'sel_strat(B=10) is not the first half of sel_fp2(B=20): {okid}/{ntr}'
        print(f'  index-identity proof: {okid}/{ntr} trials (ties included)')
    assert (m.a_fp >= 0).all(), "farthest-point contribution negative: decomposition identity broken"
    inv = m[m.true_auroc < TAU - 0.25]   # true AUROC < 0.5
    rec = [dict(component='stratified half (10 flows)', mean_attacks=m.a_s10.mean(),
                p_any_attack=(m.a_s10 > 0).mean(), mean_attacks_inverted=inv.a_s10.mean()),
           dict(component='farthest-point half (10 flows)', mean_attacks=m.a_fp.mean(),
                p_any_attack=(m.a_fp > 0).mean(), mean_attacks_inverted=inv.a_fp.mean()),
           dict(component='strat_fp2 total (20 flows)', mean_attacks=m.a_fp2.mean(),
                p_any_attack=m.c_fp2.mean(), mean_attacks_inverted=inv.a_fp2.mean()),
           dict(component='random (20 flows)', mean_attacks=m.a_rnd.mean(),
                p_any_attack=m.c_rnd.mean(), mean_attacks_inverted=inv.a_rnd.mean()),
           dict(component='uniform expectation (20 flows)', mean_attacks=20 * RATE, p_any_attack=eq1(RATE, 20),
                mean_attacks_inverted=20 * RATE)]
    d3 = pd.DataFrame(rec)
    d3['sole_source_share'] = [((m.a_s10 > 0) & (m.a_fp == 0)).mean(), ((m.a_s10 == 0) & (m.a_fp > 0)).mean(),
                               np.nan, np.nan, np.nan]
    d3.to_csv(out / 'r2_fp2_component_decomposition.csv', index=False)
    assert m.a_fp.mean() > 5 * m.a_s10.mean(), "farthest-point no longer dominates: mechanism claim invalid"
    assert m.a_s10.mean() < 1.5 * (10 * RATE), "stratified half above chance: mechanism claim invalid"

    # ---------- 4. attack concentration by detector-score decile
    # Shares are counted, not averaged: share[k] = attacks in decile k / all attacks.
    # Score ties make np.quantile bins unequal, so bin sizes are reported alongside.
    fs = sorted((run / 'predictions/lightgbm').glob('UNR_*0.0144/*.parquet'))
    assert fs, "no lightgbm UNR@1.44% prediction files found"
    cnt = np.zeros(10); size = np.zeros(10); tot = 0; nflow = 0; aucs = []; poolK = {}
    for f in fs:
        d = pd.read_parquet(f); y = d.y_true.values.astype(int); p = d.score.values
        e = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.digitize(p, e[1:-1]), 0, 9)
        for k in range(10):
            cnt[k] += y[b == k].sum(); size[k] += (b == k).sum()
        tot += y.sum(); nflow += len(y); aucs.append(roc_auc_score(y, p))
        poolK.setdefault(int(d.pool_seed.iloc[0]), int(y.sum()))
    d4 = pd.DataFrame(dict(decile=range(1, 11),
                           attacks=cnt.astype(int), flows=size.astype(int),
                           attack_rate=cnt / size,
                           share_of_all_attacks_pct=100 * cnt / tot,
                           share_of_all_flows_pct=100 * size / nflow))
    d4['lift_vs_flow_share'] = d4.share_of_all_attacks_pct / d4.share_of_all_flows_pct
    d4.to_csv(out / 'r2_score_decile_attack_rate.csv', index=False)
    assert abs(d4.share_of_all_attacks_pct.sum() - 100) < 1e-6, "decile shares do not sum to 100%"
    assert d4.loc[9, 'lift_vs_flow_share'] < 1.0, \
        "top score decile now enriched in attacks: the 'no score concentration' claim must be re-examined"
    n_pools, inverted = len(fs), int(sum(a < 0.5 for a in aucs))

    # ---------- 4b. finite-population baselines: fixed-K correction vs between-pool variation
    rec = []
    for B in BUDGETS:
        r = dict(budget=B, eq1=eq1(RATE, B), marginal_hyper_build_pool=hyper(N_BUILD, K_BUILD, B))
        r['marginal_minus_eq1'] = r['marginal_hyper_build_pool'] - r['eq1']
        for ps_, K_ in sorted(poolK.items()):
            r[f'hyper_pool{ps_}_K{K_}'] = hyper(N_POOL, K_, B)
        rec.append(r)
    d4b = pd.DataFrame(rec); d4b.to_csv(out / 'r2_finite_population_baselines.csv', index=False)
    kcols = [c for c in d4b.columns if c.startswith('hyper_pool')]
    spread = (d4b[kcols].max(axis=1) - d4b[kcols].min(axis=1)).max()
    assert spread > 0.01, "between-pool spread vanished: the two finite-population effects are no longer distinct"
    assert d4b.marginal_minus_eq1.abs().max() < 0.001, "marginal baseline drifted from Eq. (1)"
    assert len(poolK) == 3, f"expected 3 distinct target pools, found {len(poolK)}"
    print(f"  realized attack counts per target pool: {dict(sorted(poolK.items()))}")

    # ---------- 5. low-FPR Wilson intervals + median AUPRC bootstrap  (Table 6)
    op = e2[(e2.alpha == 0) & (e2.auroc >= TAU)]
    rng = np.random.default_rng(SEEDS['auprc']); rec = []
    for name, gg in list(op.groupby('model_family')) + [('All', op)]:
        r = dict(family=name, n=len(gg), median_auprc=gg.auprc.median())
        for t in (0.01, 0.05, 0.10):
            k = int((gg.tpr_fpr_001 < t).sum()); lo, hi = wilson(k, len(gg))
            r[f'n_below_{t}'] = k; r[f'lo_{t}'] = lo; r[f'hi_{t}'] = hi
        v = gg.auprc.values
        bs = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(reps_auprc)]
        r['auprc_lo'], r['auprc_hi'] = np.percentile(bs, [2.5, 97.5]); rec.append(r)
    d5 = pd.DataFrame(rec); d5.to_csv(out / 'r2_lowfpr_intervals.csv', index=False)
    pub = {'lightgbm': (14, 3, 9, 11), 'logreg': (6, 6, 6, 6), 'mlp': (22, 14, 17, 19), 'All': (42, 23, 32, 36)}
    for _, r in d5.iterrows():
        assert (r.n, r['n_below_0.01'], r['n_below_0.05'], r['n_below_0.1']) == pub[r.family], \
            f"Table 6 counts no longer reproduce for {r.family}"
        rows_all.append(dict(quantity=f"median AUPRC {r.family}", estimate=r.median_auprc,
                             lo=r.auprc_lo, hi=r.auprc_hi, method="bootstrap over realizations"))
        for t in (0.01, 0.05, 0.10):
            rows_all.append(dict(quantity=f"share TPR@1%FPR<{t} ({r.family})", estimate=r[f'n_below_{t}'] / r.n,
                                 lo=r[f'lo_{t}'], hi=r[f'hi_{t}'], method="Wilson"))

    # ---------- 6. LODO label-free warning AUROC, cluster bootstrap  (Table 4)
    lf = e1[e1.alpha == 0]
    def lodo_auc(df, sig, cfgs=None):
        S, Y = [], []
        for c in (df.config.unique() if cfgs is None else cfgs):
            tr, te = df[df.config != c], df[df.config == c]
            if tr.collapsed.nunique() < 2 or len(te) == 0: continue
            w = tr.groupby('config').config.transform(lambda s: 1.0 / len(s))
            try: a = roc_auc_score(tr.collapsed, tr[sig], sample_weight=w)
            except ValueError: continue
            o = 1 if a >= 0.5 else -1
            S += list(o * te[sig].values); Y += list(te.collapsed.values)
        Y = np.array(Y)
        return roc_auc_score(Y, np.array(S)) if len(np.unique(Y)) == 2 else np.nan
    rng = np.random.default_rng(SEEDS['lodo']); rec = []
    for fam, sig in [('lightgbm', 'pred_pos_rate'), ('logreg', 'mmd_lin'), ('mlp', 'mean_msp')]:
        gg = lf[lf.model_family == fam]; pt_est = lodo_auc(gg, sig)
        cfgs = gg.config.unique(); bs = []
        for _ in range(reps_lodo):
            a = lodo_auc(gg, sig, cfgs=rng.choice(cfgs, len(cfgs), replace=True))
            if not np.isnan(a): bs.append(a)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec.append(dict(family=fam, signal=sig, warning_auroc=pt_est, ci_lo=lo, ci_hi=hi, n_boot=len(bs)))
        rows_all.append(dict(quantity=f"LODO warning AUROC {fam} ({sig})", estimate=pt_est,
                             lo=lo, hi=hi, method="held-out deployment resampling, conditional on the fitted LODO calibration"))
    d6 = pd.DataFrame(rec); d6.to_csv(out / 'r2_lodo_label_free_intervals.csv', index=False)
    for fam, want in [('lightgbm', 0.852), ('logreg', 0.800), ('mlp', 0.574)]:
        got = float(d6[d6.family == fam].warning_auroc.iloc[0])
        assert abs(got - want) < 0.002, f"Table 4 warning AUROC no longer reproduces for {fam}: {got:.3f} vs {want}"

    pd.DataFrame(rows_all).to_csv(out / 'r2_all_intervals.csv', index=False)
    print("All invariant selftests PASSED.")
    print(f"Wrote 7 result files + r2_all_intervals.csv to {out}/")
    for f in sorted(out.glob('*.csv')): print("   ", f.name)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='COMNET_REVISION_RUN')
    ap.add_argument('--out-dir', default='r2_verification_output')
    # Defaults are the resample counts used for the published values.
    ap.add_argument('--reps-cluster', type=int, default=10000)
    ap.add_argument('--reps-paired',  type=int, default=10000)
    ap.add_argument('--reps-lodo',    type=int, default=1500)
    ap.add_argument('--reps-auprc',   type=int, default=10000)
    ap.add_argument('--master', default=None, help='path to comnet_revision_master.py (enables the index-identity proof)')
    a = ap.parse_args()
    sys.exit(main(a.run_dir, a.out_dir, a.reps_cluster, a.reps_paired, a.reps_lodo, a.reps_auprc, a.master))
