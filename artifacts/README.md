# Archived T²MLR results

These four tracked archives contain the original results for the initial study and its expansion. Check SHA256 and sizes against `04_evidence/t2mlr_raw_archives.json` before extraction. Model weights and runtime environments are not included.

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
