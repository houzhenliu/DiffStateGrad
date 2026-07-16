#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-diffusion}"
IMAGE_ID="${IMAGE_ID:-60004}"
TASK_CONFIG="${TASK_CONFIG:-configs/tasks/box_inpainting_config.yaml}"
CKPT="${CKPT:-models/ldm/ffhq/model.ckpt}"
DDIM_STEPS="${DDIM_STEPS:-20}"
SAVE_DIR="${SAVE_DIR:-./results_smoke}"
VAR_CUTOFF="${VAR_CUTOFF:-0.99}"
PIXEL_LR="${PIXEL_LR:-1e-2}"
LATENT_LR="${LATENT_LR:-5e-3}"
PIXEL_MAX_ITERS="${PIXEL_MAX_ITERS:-2000}"
LATENT_MAX_ITERS="${LATENT_MAX_ITERS:-500}"
SEED="${SEED:-42}"

export PYTHONPATH="${PYTHONPATH:-src/taming-transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torch_cache}"

mkdir -p "$SAVE_DIR"

run_mode() {
  local mode="$1"
  local period="$2"

  echo
  echo "===== Running projection_mode=${mode}, period=${period} ====="

  conda run -n "$CONDA_ENV" python diffstategrad_sample_condition.py \
    --task_config "$TASK_CONFIG" \
    --diffusion_config "$CKPT" \
    --image_id "$IMAGE_ID" \
    --ddim_steps "$DDIM_STEPS" \
    --period "$period" \
    --projection_mode "$mode" \
    --var_cutoff "$VAR_CUTOFF" \
    --pixel_lr "$PIXEL_LR" \
    --latent_lr "$LATENT_LR" \
    --pixel_max_iters "$PIXEL_MAX_ITERS" \
    --latent_max_iters "$LATENT_MAX_ITERS" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR"
}

latest_log_for_mode() {
  local mode="$1"
  find "$SAVE_DIR" -name log_stats.txt \
    -path "*file_id=(${IMAGE_ID})*" \
    -path "*projection=(${mode})*" \
    -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

metric_value() {
  local log_file="$1"
  local metric="$2"
  awk -F': ' -v metric="measurement ${metric}" '$1 == metric {print $2}' "$log_file" | xargs
}

run_mode none 0
run_mode core 5
run_mode tangent 5

echo
echo "===== Projection Smoke Summary ====="
printf "%-10s %10s %10s %10s %10s  %s\n" "mode" "PSNR" "NMSE" "SSIM" "LPIPS" "log"

for mode in none core tangent; do
  log_file="$(latest_log_for_mode "$mode")"
  if [[ -z "$log_file" ]]; then
    printf "%-10s %10s %10s %10s %10s  %s\n" "$mode" "NA" "NA" "NA" "NA" "missing log"
    continue
  fi

  psnr="$(metric_value "$log_file" psnr)"
  nmse="$(metric_value "$log_file" nmse)"
  ssim="$(metric_value "$log_file" ssim)"
  lpips="$(metric_value "$log_file" lpips)"
  printf "%-10s %10.4f %10.6f %10.6f %10.6f  %s\n" "$mode" "$psnr" "$nmse" "$ssim" "$lpips" "$log_file"
done
