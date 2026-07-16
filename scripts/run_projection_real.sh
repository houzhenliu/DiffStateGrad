#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-diffusion}"
IMAGE_IDS="${IMAGE_IDS:-60004}"
TASK_CONFIG="${TASK_CONFIG:-configs/tasks/box_inpainting_config.yaml}"
CKPT="${CKPT:-models/ldm/ffhq/model.ckpt}"
DDIM_STEPS="${DDIM_STEPS:-500}"
SAVE_DIR="${SAVE_DIR:-./results_real}"
VAR_CUTOFF="${VAR_CUTOFF:-0.99}"
PIXEL_LR="${PIXEL_LR:-1e-2}"
LATENT_LR="${LATENT_LR:-5e-3}"
PIXEL_MAX_ITERS="${PIXEL_MAX_ITERS:-2000}"
LATENT_MAX_ITERS="${LATENT_MAX_ITERS:-500}"
SEED="${SEED:-42}"
MODES="${MODES:-none core hybrid:0.05 hybrid:0.1 hybrid:0.25 tangent}"

export PYTHONPATH="${PYTHONPATH:-src/taming-transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torch_cache}"

mkdir -p "$SAVE_DIR"

RUN_RECORDS=()
IMAGE_COUNT="$(wc -w <<< "$IMAGE_IDS" | xargs)"
MODE_COUNT="$(wc -w <<< "$MODES" | xargs)"
TOTAL_RUNS=$((IMAGE_COUNT * MODE_COUNT))
CURRENT_RUN=0

print_overall_progress() {
  local current="$1"
  local total="$2"
  local label="$3"
  local elapsed="$4"
  local width=30
  local filled=$((current * width / total))
  local empty=$((width - filled))
  local bar

  bar="$(printf "%${filled}s" "" | tr ' ' '#')"
  bar="${bar}$(printf "%${empty}s" "" | tr ' ' '-')"
  printf "\rOverall [%s] %d/%d | elapsed %s | %s" "$bar" "$current" "$total" "$(format_seconds "$elapsed")" "$label"
}

mode_name() {
  local token="$1"
  case "$token" in
    hybrid:*) echo "hybrid" ;;
    hybrid_*) echo "hybrid" ;;
    *) echo "$token" ;;
  esac
}

format_alpha() {
  printf "%g\n" "$1"
}

mode_alpha() {
  local token="$1"
  local raw
  case "$token" in
    hybrid:*) raw="${token#hybrid:}"; format_alpha "$raw" ;;
    hybrid_*) raw="${token#hybrid_}"; format_alpha "$raw" ;;
    *) echo "1" ;;
  esac
}

mode_label() {
  local token="$1"
  local mode
  local alpha
  mode="$(mode_name "$token")"
  alpha="$(mode_alpha "$token")"
  if [[ "$mode" == "hybrid" ]]; then
    echo "hybrid:${alpha}"
  else
    echo "$mode"
  fi
}

run_one() {
  local image_id="$1"
  local mode_token="$2"
  local period="$3"
  local mode
  local alpha
  local label_mode
  local start_ts
  local end_ts
  local elapsed
  local global_start_ts
  local label

  mode="$(mode_name "$mode_token")"
  alpha="$(mode_alpha "$mode_token")"
  label_mode="$(mode_label "$mode_token")"
  CURRENT_RUN=$((CURRENT_RUN + 1))
  global_start_ts="${GLOBAL_START_TS:-$(date +%s)}"
  label="image=${image_id} mode=${label_mode}"
  echo
  echo "===== [${CURRENT_RUN}/${TOTAL_RUNS}] Running image=${image_id}, projection_mode=${mode}, alpha=${alpha}, period=${period}, ddim_steps=${DDIM_STEPS} ====="
  print_overall_progress "$((CURRENT_RUN - 1))" "$TOTAL_RUNS" "starting ${label}" "$(($(date +%s) - global_start_ts))"
  echo
  start_ts="$(date +%s)"

  conda run -n "$CONDA_ENV" python diffstategrad_sample_condition.py \
    --task_config "$TASK_CONFIG" \
    --diffusion_config "$CKPT" \
    --image_id "$image_id" \
    --ddim_steps "$DDIM_STEPS" \
    --period "$period" \
    --projection_mode "$mode" \
    --projection_alpha "$alpha" \
    --var_cutoff "$VAR_CUTOFF" \
    --pixel_lr "$PIXEL_LR" \
    --latent_lr "$LATENT_LR" \
    --pixel_max_iters "$PIXEL_MAX_ITERS" \
    --latent_max_iters "$LATENT_MAX_ITERS" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  RUN_RECORDS+=("${image_id}|${mode}|${alpha}|${label_mode}|${elapsed}")
  print_overall_progress "$CURRENT_RUN" "$TOTAL_RUNS" "finished ${label}" "$((end_ts - global_start_ts))"
  echo
}

period_for_mode() {
  case "$(mode_name "$1")" in
    none) echo 0 ;;
    core|fixed|tangent|normal_removed|hybrid) echo 5 ;;
    *) echo "Unknown mode '$1'" >&2; exit 1 ;;
  esac
}

latest_log() {
  local image_id="$1"
  local mode="$2"
  local alpha="${3:-}"
  local command=(find "$SAVE_DIR" -name log_stats.txt
    -path "*file_id=(${image_id})*" \
    -path "*projection=(${mode})*")
  if [[ "$mode" == "hybrid" && -n "$alpha" ]]; then
    command+=(-path "*alpha=(${alpha})*")
  fi
  "${command[@]}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

metric_value() {
  local log_file="$1"
  local metric="$2"
  awk -F': ' -v metric="measurement ${metric}" '$1 == metric {print $2}' "$log_file" | xargs
}

format_seconds() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  if (( h > 0 )); then
    printf "%dh%02dm%02ds" "$h" "$m" "$s"
  else
    printf "%dm%02ds" "$m" "$s"
  fi
}

GLOBAL_START_TS="$(date +%s)"

for image_id in $IMAGE_IDS; do
  for mode_token in $MODES; do
    run_one "$image_id" "$mode_token" "$(period_for_mode "$mode_token")"
  done
done

echo
echo "===== Projection Real Run Summary ====="
printf "%-8s %-12s %10s %10s %10s %10s %10s  %s\n" \
  "image" "mode" "time" "PSNR" "NMSE" "SSIM" "LPIPS" "log"

total_seconds=0
run_count=0

for record in "${RUN_RECORDS[@]}"; do
  IFS='|' read -r image_id mode alpha label_mode elapsed <<< "$record"
  log_file="$(latest_log "$image_id" "$mode" "$alpha")"
  total_seconds=$((total_seconds + elapsed))
  run_count=$((run_count + 1))

  if [[ -z "$log_file" ]]; then
    printf "%-8s %-12s %10s %10s %10s %10s %10s  %s\n" \
      "$image_id" "$label_mode" "$(format_seconds "$elapsed")" "NA" "NA" "NA" "NA" "missing log"
    continue
  fi

  psnr="$(metric_value "$log_file" psnr)"
  nmse="$(metric_value "$log_file" nmse)"
  ssim="$(metric_value "$log_file" ssim)"
  lpips="$(metric_value "$log_file" lpips)"
  printf "%-8s %-12s %10s %10.4f %10.6f %10.6f %10.6f  %s\n" \
    "$image_id" "$label_mode" "$(format_seconds "$elapsed")" "$psnr" "$nmse" "$ssim" "$lpips" "$log_file"
done

if (( run_count > 0 )); then
  avg_seconds=$((total_seconds / run_count))
  image_count="$(wc -w <<< "$IMAGE_IDS" | xargs)"
  mode_count="$(wc -w <<< "$MODES" | xargs)"
  avg_image_seconds=$((avg_seconds * mode_count))

  echo
  echo "===== Single-GPU Time Estimate ====="
  echo "Average time per reconstruction: $(format_seconds "$avg_seconds")"
  echo "Average time per image for ${mode_count} mode(s): $(format_seconds "$avg_image_seconds")"
  echo "Estimated time for 20 images x ${mode_count} mode(s): $(format_seconds $((avg_seconds * 20 * mode_count)))"
  echo "Estimated time for 100 images x ${mode_count} mode(s): $(format_seconds $((avg_seconds * 100 * mode_count)))"
fi
