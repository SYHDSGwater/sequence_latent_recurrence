#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
$PY -m pytest experiments/t2mlr/test_mechanism.py -q
$PY experiments/t2mlr/run.py --data data/validation.jsonl --output artifacts/pilot --documents 4 --batch-size 4 --arms normal reset_once --max-seconds 1800
