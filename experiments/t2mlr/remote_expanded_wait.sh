#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
blob=hf/hub/models--JupiterZhu--T2MLR_135M_lstart13_lend18_10B_FineWebEdu/blobs/2d378d0f4f92a2957e0099d9bf1c77a5d0999956c96540aa84c494d4717c480d
while ! test -f "$blob"; do sleep 5; done
exec bash experiments/t2mlr/remote_expanded.sh
