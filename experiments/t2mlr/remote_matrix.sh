#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
for model in 135m_50b 362m_10b 362m_50b 982m_10b; do
  $PY experiments/t2mlr/matrix_prepare.py --model "$model"
  for domain in wiki math; do
    $PY experiments/t2mlr/expanded_run.py --model "$model" --data "data/expanded/$domain.jsonl" --output "artifacts/matrix/${model}_${domain}" --batch-size 64 --max-seconds 3600
  done
done
tar -czf matrix-results.tar.gz artifacts/matrix data/expanded
