#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/t2mlr-20260918
while ! test -f impulse-results.tar.gz; do sleep 5; done
exec timeout 7200 bash experiments/t2mlr/remote_impulse_precision.sh
