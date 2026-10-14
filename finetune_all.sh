#!/usr/bin/env bash
# Fine-tune all four OCR models on a new dataset.
#   Weights in : the original trained runs (SRC_RUNS)
#   Weights out: a fresh tree (OUT_RUNS), so originals are never touched
# Usage:  bash finetune_all.sh /path/to/new_data [run_suffix]
set -eo pipefail

DATA="${1:?usage: bash finetune_all.sh /path/to/new_data [run_suffix]}"
SUFFIX="${2:-finetune}"

ROOT=/media/lang-chain/b5ee0e21-c51a-4025-949a-a7e888058222
SRC_RUNS="$ROOT/runs"       # <- pretrained weights live here (read-only for us)
OUT_RUNS="$ROOT/runs_2"     # <- everything this script writes goes here

# Where each pretrained model lives (checkpoints/best_ema_weights.pt inside)
INIT_LAYOUT="$SRC_RUNS/layout/b8/checkpoints/best_ema_weights.pt"
INIT_DET="$SRC_RUNS/det/main/checkpoints/best_ema_weights.pt"
INIT_TABLE="$SRC_RUNS/table/main/checkpoints/best_ema_weights.pt"
INIT_REC="$SRC_RUNS/rec/main/checkpoints/best_ema_weights.pt"

for f in "$INIT_LAYOUT" "$INIT_DET" "$INIT_TABLE" "$INIT_REC"; do
  [[ -f "$f" ]] || { echo "missing checkpoint: $f" >&2; exit 1; }
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_RUNS"

declare -A INIT ARGS
INIT[layout]="$INIT_LAYOUT"
INIT[det]="$INIT_DET"
INIT[table]="$INIT_TABLE"
INIT[rec]="$INIT_REC"

ARGS[layout]="--run ${SUFFIX}_layout --lr 1e-5 --warmup 500 --steps 20000 --batch 12"
ARGS[det]="   --run ${SUFFIX}_det    --lr 1e-4  --warmup 500 --steps 30000"
ARGS[table]=" --run ${SUFFIX}_table  --lr 3e-5  --warmup 500 --steps 30000"
ARGS[rec]="   --run ${SUFFIX}_rec    --lr 5e-5  --warmup 500 --steps 50000"

for m in layout det table rec; do
  echo "===== $m (finetune) ====="
  python train.py --model "$m" \
      --data "$DATA" \
      --out  "$OUT_RUNS" \
      --init "${INIT[$m]}" \
      ${ARGS[$m]} \
      2>&1 | tee -a "$OUT_RUNS/train_${SUFFIX}_$m.log"
done
echo "All four fine-tunes finished. Run dirs:"
for m in layout det table rec; do echo "  $OUT_RUNS/$m/${SUFFIX}_$m"; done