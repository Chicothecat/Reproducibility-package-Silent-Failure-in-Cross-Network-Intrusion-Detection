# Reproducibility package — COMNET-D-26-06148

Regenerates the quantities added in the Revision 2 round. Nothing is retrained and no
experiment is re-run; the verifier reads the archived run.

## Contents

    revision_run/COMNET_REVISION_RUN/    per-draw grids, result tables, saved per-flow
                                         predictions, environment record
    run_script/comnet_revision_master.py the script that produced the run
    verification/r2_verification.py      the verifier
    verification/r2_verification_output/ its CSV outputs, as quoted in the article
    verification/README.md               outputs, caveats, selftests
    MANIFEST.sha256                      checksums for every file above
    LICENSE                              MIT

## Reproduce

    cd verification
    python3 r2_verification.py \
        --run-dir ../revision_run/COMNET_REVISION_RUN \
        --master  ../run_script/comnet_revision_master.py \
        --out-dir out

numpy, pandas, scikit-learn, pyarrow. About five minutes. Seeds and default resample
counts are those used for the published values, so `out/` matches
`r2_verification_output/` file for file. The run exits non-zero if any invariant
selftest fails.

To check the contents after extraction:

    sha256sum -c MANIFEST.sha256

## Notes

Per-flow predictions are included for all thirty transfer-and-detector combinations.
The copy attached to the journal submission carries only the four LightGBM Unraveled
1.44% transfers the verifier reads, to stay within a file-size limit; the reported
quantities are identical from either copy.

The benchmark datasets (NetFlow-v2 suite, Unraveled) are not redistributed and are
available from their original sources, cited in the article. Re-running
`comnet_revision_master.py` requires them; the verifier does not.
