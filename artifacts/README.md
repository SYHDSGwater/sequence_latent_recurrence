# Archived T²MLR results

The Phase-A impulse archives are `impulse-results.tar.gz` and `impulse-precision-results.tar.gz`. The first contains six checkpoints × two domains, six intervention arms plus a paired baseline, frozen inputs, per-token state/KL/NLL traces, actual injection sizes, and manifests. The second contains the fixed 16-example-per-domain BF16/FP32 numerical audit. It does not contain training runs. Extract to separate directories:

```sh
mkdir -p artifacts/impulse artifacts/impulse_precision
tar -xzf artifacts/impulse-results.tar.gz -C artifacts/impulse
tar -xzf artifacts/impulse-precision-results.tar.gz -C artifacts/impulse_precision
python experiments/t2mlr/impulse_analyze.py
python experiments/t2mlr/impulse_precision_analyze.py
```

The report and plotting commands are documented in `experiments/t2mlr/README.md`. The prior expanded/matrix archives are additionally required for the batch-sensitivity comparison. Numerical audits use the same frozen data and are not independent task replications.

The initial four tracked archives contain the original results for the initial study and its expansion. Check SHA256 and sizes against `04_evidence/t2mlr_raw_archives.json` before extraction. Model weights and runtime environments are not included.

The additional `matrix-results.tar.gz` contains the four remaining public checkpoint evaluations. It uses the same frozen input selection. Extract it to `artifacts/matrix`, alongside the existing `artifacts/expanded` extraction, then run `python experiments/t2mlr/matrix_analyze.py` from the repository root to recompute all six checkpoints and within-scale 50B-minus-10B paired differences.

From the repository root:

```sh
mkdir -p artifacts/5090 artifacts/expanded
tar -xzf artifacts/primary-results.tar.gz -C artifacts/5090
tar -xzf artifacts/confirmation-results.tar.gz -C artifacts/5090
tar -xzf artifacts/diagnostics-results.tar.gz -C artifacts/5090
tar -xzf artifacts/expanded-results.tar.gz -C artifacts/expanded
python experiments/t2mlr/expanded_analyze.py --root artifacts/expanded --output artifacts/expanded/recomputed_results.json
```

The expanded archive includes an incomplete `135m_wiki_batch16_pilot` for provenance. Only the four complete `{135m,982m}_{wiki,math}` directories belong to the final analysis. Each contains paired token losses, input token IDs, intervention positions and execution manifests. The frozen selected data are under `data/expanded` within that archive.

See `experiments/t2mlr/README.md` for dependencies and `06_reports/t2mlr_expanded_results.md` for scope and interpretation. Python and shell source files use LF line endings to preserve recorded execution-code hashes across platforms.
