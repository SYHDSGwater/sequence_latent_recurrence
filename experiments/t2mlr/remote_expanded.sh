#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
source /etc/network_turbo > /dev/null 2>&1
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
$PY experiments/t2mlr/expanded_prepare.py
$PY -m pytest experiments/t2mlr/test_mechanism.py -q
for model in 135m 982m; do
  for domain in wiki math; do
    $PY experiments/t2mlr/expanded_run.py --model "$model" --data "data/expanded/$domain.jsonl" --output "artifacts/expanded/${model}_${domain}" --batch-size 64 --max-seconds 5400
  done
done
tar -czf expanded-results.tar.gz artifacts/expanded data/expanded
