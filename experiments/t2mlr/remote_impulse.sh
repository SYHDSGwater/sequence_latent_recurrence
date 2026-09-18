#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
$PY -m pytest experiments/t2mlr/test_impulse.py -q
for model in 135m 135m_50b 362m_10b 362m_50b 982m_10b 982m; do
  for domain in wiki math; do
    $PY experiments/t2mlr/impulse_run.py --model "$model" --domain "$domain" --output "artifacts/impulse/${model}_${domain}" --batch-size 32 --max-seconds 3600
  done
done
tar -czf impulse-results.tar.gz artifacts/impulse data/expanded
