# Verification — COMNET-D-26-06148

Regenerates every quantity added in the Revision 2 round from the archived run.

## Run

    python3 r2_verification.py \
        --run-dir ../revision_run/COMNET_REVISION_RUN \
        --master  ../run_script/comnet_revision_master.py \
        --out-dir out

numpy, pandas, scikit-learn, pyarrow. About five minutes. Seeds are fixed (cluster 101,
paired 202, LODO 303, AUPRC 404) and the default resample counts are those used for the
published values, so the command reproduces `r2_verification_output/` byte for byte.
Overriding `--reps-*` shifts the interval endpoints in the third decimal; point
estimates and selftests are unaffected.

`--master` is optional to the code but required for the index-identity proof, which is
otherwise skipped with a warning.

Inputs read: `R9_prevalence_budget/`, `E1_detector_family/`, `E2_metrics/`,
`predictions/lightgbm/UNR_*0.0144/`.

## Outputs

| file | contents |
|---|---|
| `r2_random_computability_intervals.csv` | random computability at r=1.44%: Wilson and pool-seed cluster intervals, Eq. (1), both finite-population baselines |
| `r2_finite_population_baselines.csv` | marginal construction-pool baseline against the per-pool baselines at the realized attack counts |
| `r2_paired_fp2_vs_random.csv` | paired strat_fp2 - random differences, clustered on the shared target pool |
| `r2_fp2_component_decomposition.csv` | stratified vs farthest-point contribution to strat_fp2 at B=20 |
| `r2_score_decile_attack_rate.csv` | attacks, flows, attack share and flow share per detector-score decile |
| `r2_lowfpr_intervals.csv` | Table 6 counts with Wilson intervals, median AUPRC with bootstrap interval |
| `r2_lodo_label_free_intervals.csv` | LODO warning AUROC, held-out deployments resampled |
| `r2_all_intervals.csv` | every interval quoted in the article |

## Scope of the intervals

Random computability is reported two ways. The Wilson interval treats the 15 distinct
outcomes as independent; they are not, since they share three pool realizations, so the
clustered bootstrap is the defensible one and it is wide.

The R9 target pool is seeded by pool seed and prevalence only, so all sources and
detector seeds at a given pool seed share one target realization. The grid holds three
independent target pools rather than 180 independent draws, and the outer bootstrap
resamples pool seeds. With three clusters the paired intervals are coarse and are
reported as exploratory.

LODO warning AUROC resamples held-out deployments conditional on the fitted calibration
and the already-chosen signal; it is not an end-to-end bootstrap of signal selection and
threshold fitting. Low-FPR shares are marginal Wilson intervals, with shared pools and
configurations unmodelled.

Two finite-population effects are kept separate. Conditioning on a pool's realized
attack count moves the both-class probability by roughly 0.02 at B=50 (the pools hold
284, 308 and 271 attacks), and that variation is part of the Monte Carlo spread. The
marginal baseline is a different quantity: sampling is nested and uniform at every
stage, so a uniform canary is marginally B flows drawn without replacement from the
80,000-flow construction pool, whose attack count is fixed at 1,152; that baseline sits
within 0.001 of Eq. (1) at every budget. The fixed-K correction does not bound the
between-pool variation.

Decile shares are counted, not averaged: `share[k] = attacks in decile k / all attacks`.
Score ties make the quantile bins unequal, so flow counts per bin are reported alongside
and lift is taken against flow share rather than against 10%.

## Invariant selftests

The run aborts on any failure.

- random computability constant within each (pool_seed, canary_seed) group
- Eq. (1) inside both the Wilson and the clustered interval at every budget
- marginal construction-pool baseline within 0.001 of Eq. (1) at every budget
- between-pool spread at the realized attack counts above 0.01, keeping the two
  finite-population effects distinguishable
- exactly three distinct target pools in the saved predictions
- `sel_strat(B=10, seed)` equals the first half of `sel_fp2(B=20, seed)`, asserted by
  executing both selectors over randomised inputs with forced score ties (`--master`)
- farthest-point contribution non-negative in all 180 matched draws
- farthest-point half above five times the stratified half; stratified half below 1.5x
  the uniform expectation
- decile shares summing to 100%, top score decile not enriched against its flow share
- at least three of the four paired intervals excluding zero
- Table 6 counts exact (14/6/22/42 and 23/32/36)
- Table 4 warning AUROC within 0.002 for all three families

Criteria test impact, not bit-exact floats.
