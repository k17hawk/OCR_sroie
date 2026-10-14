#!/usr/bin/env bash
set -eo pipefail
DATA=/media/lang-chain/b5ee0e21-c51a-4025-949a-a7e888058222/nepali_ocr/data_v5

RUNS=/media/lang-chain/b5ee0e21-c51a-4025-949a-a7e888058222/runs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUNS"

# extra options per model (empty = defaults)
declare -A ARGS=(
  [layout]="--run b8 --batch 8 --lr 2e-4 --steps 30000"
  [det]=""
  [table]=""
  [rec]="--keep 1"
)

for m in layout det table rec; do
  echo "===== $m ====="
  python train.py --model "$m" --data "$DATA" --out "$RUNS" ${ARGS[$m]} 2>&1 | tee -a "$RUNS/train_$m.log"
done
echo "All four models finished."