#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
MODES="${MODES:-core hybrid:-0.1 hybrid:-0.2}"
SAVE_DIR="${SAVE_DIR:-./results_negative_batch}"
DDIM_STEPS="${DDIM_STEPS:-500}"

if [[ -z "${IMAGE_IDS:-}" ]]; then
  if [[ ! -d samples ]]; then
    echo "samples/ not found. Set IMAGE_IDS manually." >&2
    exit 1
  fi

  if (( LIMIT > 0 )); then
    IMAGE_IDS="$(find samples -maxdepth 1 -name '*.png' -printf '%f\n' \
      | sed -E 's/^([0-9]+)\.png$/\1/' \
      | sort \
      | tail -n +"$((OFFSET + 1))" \
      | head -n "$LIMIT" \
      | xargs)"
  else
    IMAGE_IDS="$(find samples -maxdepth 1 -name '*.png' -printf '%f\n' \
      | sed -E 's/^([0-9]+)\.png$/\1/' \
      | sort \
      | tail -n +"$((OFFSET + 1))" \
      | xargs)"
  fi
fi

if [[ -z "$IMAGE_IDS" ]]; then
  echo "No images selected. Check samples/, LIMIT, OFFSET, or IMAGE_IDS." >&2
  exit 1
fi

image_count="$(wc -w <<< "$IMAGE_IDS" | xargs)"
mode_count="$(wc -w <<< "$MODES" | xargs)"
task_count=$((image_count * mode_count))

echo "===== Negative Alpha Batch ====="
echo "GPUs: $GPU_IDS"
echo "Images (${image_count}): $IMAGE_IDS"
echo "Modes: $MODES"
echo "DDIM steps: $DDIM_STEPS"
echo "Total tasks: $task_count"
echo "Save dir: $SAVE_DIR"
echo

GPU_IDS="$GPU_IDS" \
IMAGE_IDS="$IMAGE_IDS" \
MODES="$MODES" \
DDIM_STEPS="$DDIM_STEPS" \
SAVE_DIR="$SAVE_DIR" \
./scripts/run_projection_multigpu.sh

echo
./scripts/summarize_projection_results.sh "$SAVE_DIR"
