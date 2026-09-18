#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
for attempt in $(seq 1 90); do
    if grep -q 'STAGE complete' suite.log; then break; fi
    sleep 10
done
grep -q 'STAGE complete' suite.log
$PY experiments/t2mlr/diagnostics.py
