#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
for model in 362m_10b 362m_50b 982m_10b; do
 /root/autodl-tmp/ddm-smoke-env/bin/python experiments/t2mlr/matrix_prepare.py --model "$model"
done
