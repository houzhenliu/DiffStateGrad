#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:-${SAVE_DIR:-./results_negative_batch}}"

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "Result directory not found: $RESULT_DIR" >&2
  exit 1
fi

TMP_CSV="$(mktemp /tmp/projection_metrics.XXXXXX.csv)"
trap 'rm -f "$TMP_CSV"' EXIT

metric_value() {
  local log_file="$1"
  local metric="$2"
  awk -F': ' -v metric="measurement ${metric}" '$1 == metric {print $2}' "$log_file" | xargs
}

mode_label_from_path() {
  local log_file="$1"
  local mode
  local alpha
  mode="$(sed -n 's/.*projection=(\([^)]*\)).*/\1/p' <<< "$log_file")"
  alpha="$(sed -n 's/.*alpha=(\([^)]*\)).*/\1/p' <<< "$log_file")"
  if [[ "$mode" == "hybrid" ]]; then
    echo "hybrid:${alpha}"
  else
    echo "$mode"
  fi
}

image_from_path() {
  local log_file="$1"
  sed -n 's/.*file_id=(\([0-9]*\)).*/\1/p' <<< "$log_file"
}

echo "image,mode,psnr,nmse,ssim,lpips,log" > "$TMP_CSV"

while IFS= read -r log_file; do
  image="$(image_from_path "$log_file")"
  mode="$(mode_label_from_path "$log_file")"
  psnr="$(metric_value "$log_file" psnr)"
  nmse="$(metric_value "$log_file" nmse)"
  ssim="$(metric_value "$log_file" ssim)"
  lpips="$(metric_value "$log_file" lpips)"
  printf "%s,%s,%s,%s,%s,%s,%s\n" "$image" "$mode" "$psnr" "$nmse" "$ssim" "$lpips" "$log_file" >> "$TMP_CSV"
done < <(find "$RESULT_DIR" -name log_stats.txt | sort)

if [[ "$(wc -l < "$TMP_CSV" | xargs)" -le 1 ]]; then
  echo "No log_stats.txt found under $RESULT_DIR" >&2
  exit 1
fi

echo "===== Per-Run Metrics ====="
LC_NUMERIC=C awk -F',' '
NR == 1 {
  printf "%-8s %-12s %10s %10s %10s %10s\n", "image", "mode", "PSNR", "NMSE", "SSIM", "LPIPS"
  next
}
{
  printf "%-8s %-12s %10.4f %10.6f %10.6f %10.6f\n", $1, $2, $3, $4, $5, $6
}
' "$TMP_CSV"

echo
echo "===== Mode Averages ====="
LC_NUMERIC=C awk -F',' '
NR == 1 { next }
{
  mode = $2
  n[mode] += 1
  psnr[mode] += $3
  nmse[mode] += $4
  ssim[mode] += $5
  lpips[mode] += $6
}
END {
  printf "%-12s %6s %10s %10s %10s %10s\n", "mode", "n", "PSNR", "NMSE", "SSIM", "LPIPS"
  for (mode in n) {
    printf "%-12s %6d %10.4f %10.6f %10.6f %10.6f\n",
      mode, n[mode], psnr[mode] / n[mode], nmse[mode] / n[mode],
      ssim[mode] / n[mode], lpips[mode] / n[mode]
  }
}
' "$TMP_CSV"

echo
echo "===== Best Mode Per Image ====="
LC_NUMERIC=C awk -F',' '
NR == 1 { next }
{
  image = $1
  mode = $2
  if (!(image in psnr_best) || $3 > psnr_best[image]) {
    psnr_best[image] = $3
    psnr_mode[image] = mode
  }
  if (!(image in nmse_best) || $4 < nmse_best[image]) {
    nmse_best[image] = $4
    nmse_mode[image] = mode
  }
  if (!(image in ssim_best) || $5 > ssim_best[image]) {
    ssim_best[image] = $5
    ssim_mode[image] = mode
  }
  if (!(image in lpips_best) || $6 < lpips_best[image]) {
    lpips_best[image] = $6
    lpips_mode[image] = mode
  }
}
END {
  printf "%-8s %-12s %-12s %-12s %-12s\n", "image", "PSNR", "NMSE", "SSIM", "LPIPS"
  for (image in psnr_best) {
    printf "%-8s %-12s %-12s %-12s %-12s\n",
      image, psnr_mode[image], nmse_mode[image], ssim_mode[image], lpips_mode[image]
  }
}
' "$TMP_CSV"

SUMMARY_CSV="$RESULT_DIR/metrics_summary.csv"
cp "$TMP_CSV" "$SUMMARY_CSV"
echo
echo "Saved CSV: $SUMMARY_CSV"
