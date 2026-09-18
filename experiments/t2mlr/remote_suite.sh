#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-fs/t2mlr-20260918/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/t2mlr-runtime/deps
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/ddm-smoke-env/bin/python
echo 'STAGE download completion'
blob="$HF_HOME/hub/models--JupiterZhu--T2MLR_982M_lstart9_lend24_50B_FineWebEdu/blobs/76b5ff218df444c6849cafbb9415710290127a6c3fbaac0592cdfc94872dd545"
for attempt in $(seq 1 120); do
    if test -f "$blob"; then break; fi
    sleep 10
done
test -f "$blob"
echo 'STAGE pilot'
$PY experiments/t2mlr/run.py --data data/validation.jsonl --output artifacts/pilot_verified --documents 4 --batch-size 4 --arms normal reset_once --max-seconds 900
echo 'STAGE full-weight BF16 validation'
$PY experiments/t2mlr/gpu_validate.py
echo 'STAGE primary'
$PY experiments/t2mlr/run.py --data data/validation.jsonl --output artifacts/primary --documents 32 --batch-size 4 --arms normal reset_once donor_once reset8 reset32 reset128 zero_every carry_off fusion_off fusion_and_carry_off reset_once_clean_kv --max-seconds 7200
echo 'STAGE confirmation'
$PY experiments/t2mlr/run.py --data data/test.jsonl --output artifacts/confirmation --documents 32 --batch-size 4 --reset-at 256 --arms normal reset_once donor_once carry_off reset_once_clean_kv --max-seconds 5400
echo 'STAGE complete'
