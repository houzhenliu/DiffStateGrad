#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-diffusion}"
GPU_IDS="${GPU_IDS:-0}"
IMAGE_IDS="${IMAGE_IDS:-60004}"
TASK_CONFIG="${TASK_CONFIG:-configs/tasks/box_inpainting_config.yaml}"
CKPT="${CKPT:-models/ldm/ffhq/model.ckpt}"
DDIM_STEPS="${DDIM_STEPS:-500}"
SAVE_DIR="${SAVE_DIR:-./results_real_multigpu}"
VAR_CUTOFF="${VAR_CUTOFF:-0.99}"
PIXEL_LR="${PIXEL_LR:-1e-2}"
LATENT_LR="${LATENT_LR:-5e-3}"
PIXEL_MAX_ITERS="${PIXEL_MAX_ITERS:-2000}"
LATENT_MAX_ITERS="${LATENT_MAX_ITERS:-500}"
SEED="${SEED:-42}"
MODES="${MODES:-none core hybrid:0.05 hybrid:0.1 hybrid:0.25 tangent}"
POLL_SECONDS="${POLL_SECONDS:-5}"

export PYTHONPATH="${PYTHONPATH:-src/taming-transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torch_cache}"

mkdir -p "$SAVE_DIR"
RUN_LOG_DIR="$SAVE_DIR/multigpu_logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_LOG_DIR"

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

period_for_mode() {
  case "$(mode_name "$1")" in
    none) echo 0 ;;
    core|fixed|tangent|normal_removed|hybrid) echo 5 ;;
    *) echo "Unknown mode '$1'" >&2; exit 1 ;;
  esac
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

print_progress() {
  local done_count="$1"
  local total="$2"
  local failed_count="$3"
  local elapsed="$4"
  local width=30
  local filled=$((done_count * width / total))
  local empty=$((width - filled))
  local bar

  bar="$(printf "%${filled}s" "" | tr ' ' '#')"
  bar="${bar}$(printf "%${empty}s" "" | tr ' ' '-')"
  printf "\rOverall [%s] %d/%d | failed %d | elapsed %s" \
    "$bar" "$done_count" "$total" "$failed_count" "$(format_seconds "$elapsed")"
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

TASKS=()
for image_id in $IMAGE_IDS; do
  for mode_token in $MODES; do
    TASKS+=("${image_id}|$(mode_name "$mode_token")|$(mode_alpha "$mode_token")|$(mode_label "$mode_token")|$(period_for_mode "$mode_token")")
  done
done

GPUS=()
for gpu in $GPU_IDS; do
  GPUS+=("$gpu")
done

if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

if (( ${#TASKS[@]} == 0 )); then
  echo "No tasks to run. Check IMAGE_IDS and MODES." >&2
  exit 1
fi

echo "===== Projection Multi-GPU Run ====="
echo "GPUs: ${GPU_IDS}"
echo "Images: ${IMAGE_IDS}"
echo "Modes: ${MODES}"
echo "DDIM steps: ${DDIM_STEPS}"
echo "Save dir: ${SAVE_DIR}"
echo "Worker logs: ${RUN_LOG_DIR}"
echo

PIDS=()
SLOT_TASKS=()
SLOT_STARTS=()
SLOT_LOGS=()
SLOT_LABELS=()
for ((slot = 0; slot < ${#GPUS[@]}; slot++)); do
  PIDS[$slot]=""
  SLOT_TASKS[$slot]=""
  SLOT_STARTS[$slot]=""
  SLOT_LOGS[$slot]=""
  SLOT_LABELS[$slot]=""
done

RUN_RECORDS=()
next_task=0
done_count=0
failed_count=0
total_tasks="${#TASKS[@]}"
global_start="$(date +%s)"

start_task() {
  local slot="$1"
  local task_index="$2"
  local gpu="${GPUS[$slot]}"
  local task="${TASKS[$task_index]}"
  local image_id mode alpha label_mode period label worker_log

  IFS='|' read -r image_id mode alpha label_mode period <<< "$task"
  label="image=${image_id} mode=${label_mode} gpu=${gpu}"
  worker_log="$RUN_LOG_DIR/task_$((task_index + 1))_${image_id}_${label_mode//:/_}_gpu${gpu}.log"

  echo "Starting [$((task_index + 1))/${total_tasks}] ${label}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
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
  ) > "$worker_log" 2>&1 &

  PIDS[$slot]="$!"
  SLOT_TASKS[$slot]="$task"
  SLOT_STARTS[$slot]="$(date +%s)"
  SLOT_LOGS[$slot]="$worker_log"
  SLOT_LABELS[$slot]="$label"
}

while (( done_count < total_tasks )); do
  for ((slot = 0; slot < ${#GPUS[@]}; slot++)); do
    if [[ -z "${PIDS[$slot]}" && "$next_task" -lt "$total_tasks" ]]; then
      start_task "$slot" "$next_task"
      next_task=$((next_task + 1))
    fi
  done

  sleep "$POLL_SECONDS"

  for ((slot = 0; slot < ${#GPUS[@]}; slot++)); do
    pid="${PIDS[$slot]}"
    if [[ -z "$pid" ]]; then
      continue
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      task="${SLOT_TASKS[$slot]}"
      worker_log="${SLOT_LOGS[$slot]}"
      label="${SLOT_LABELS[$slot]}"
      start_ts="${SLOT_STARTS[$slot]}"
      end_ts="$(date +%s)"
      elapsed=$((end_ts - start_ts))

      if wait "$pid"; then
        status="ok"
        echo
        echo "Finished ${label} in $(format_seconds "$elapsed")"
      else
        status="failed"
        failed_count=$((failed_count + 1))
        echo
        echo "Failed ${label} after $(format_seconds "$elapsed"). See ${worker_log}"
      fi

      RUN_RECORDS+=("${task}|${elapsed}|${status}|${worker_log}")
      PIDS[$slot]=""
      SLOT_TASKS[$slot]=""
      SLOT_STARTS[$slot]=""
      SLOT_LOGS[$slot]=""
      SLOT_LABELS[$slot]=""
      done_count=$((done_count + 1))
    fi
  done

  print_progress "$done_count" "$total_tasks" "$failed_count" "$(($(date +%s) - global_start))"
  echo
done

echo
echo "===== Projection Multi-GPU Summary ====="
printf "%-8s %-12s %10s %8s %10s %10s %10s %10s  %s\n" \
  "image" "mode" "time" "status" "PSNR" "NMSE" "SSIM" "LPIPS" "worker_log"

total_seconds=0
ok_count=0

for record in "${RUN_RECORDS[@]}"; do
  IFS='|' read -r image_id mode alpha label_mode period elapsed status worker_log <<< "$record"
  total_seconds=$((total_seconds + elapsed))

  if [[ "$status" != "ok" ]]; then
    printf "%-8s %-12s %10s %8s %10s %10s %10s %10s  %s\n" \
      "$image_id" "$label_mode" "$(format_seconds "$elapsed")" "$status" "NA" "NA" "NA" "NA" "$worker_log"
    continue
  fi

  ok_count=$((ok_count + 1))
  log_file="$(latest_log "$image_id" "$mode" "$alpha")"
  if [[ -z "$log_file" ]]; then
    printf "%-8s %-12s %10s %8s %10s %10s %10s %10s  %s\n" \
      "$image_id" "$label_mode" "$(format_seconds "$elapsed")" "$status" "NA" "NA" "NA" "NA" "$worker_log"
    continue
  fi

  psnr="$(metric_value "$log_file" psnr)"
  nmse="$(metric_value "$log_file" nmse)"
  ssim="$(metric_value "$log_file" ssim)"
  lpips="$(metric_value "$log_file" lpips)"
  printf "%-8s %-12s %10s %8s %10.4f %10.6f %10.6f %10.6f  %s\n" \
    "$image_id" "$label_mode" "$(format_seconds "$elapsed")" "$status" "$psnr" "$nmse" "$ssim" "$lpips" "$worker_log"
done

echo
echo "Completed ${ok_count}/${total_tasks} task(s). Worker logs are in ${RUN_LOG_DIR}"

if (( failed_count > 0 )); then
  exit 1
fi
