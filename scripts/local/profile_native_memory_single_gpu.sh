#!/usr/bin/env bash
# CUDA allocation snapshots for the shared native profiling workload.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/profile_native_common.sh"
profile_init torch_memory "$@"
profile_build_command
profile_prepare_run
profile_run_training

# Preserve completed snapshots and the training exit code if a later step fails.
report_rc=0
"${python_bin}" "${repo_root}/scripts/local/analyze_torch_memory.py" \
  "${run_dir}/snapshots" --output-dir "${run_dir}/memory_report" \
  --max-entries "${max_entries}" --device-csv "${run_dir}/device_memory.csv" || report_rc=$?
for step in "${profile_steps[@]}"; do
  found=false
  for snapshot in "${run_dir}/snapshots/step${step}"/torch_memory*.pickle; do
    if [[ -s "${snapshot}" ]]; then found=true; break; fi
  done
  if [[ "${found}" != true ]]; then
    printf 'Missing memory snapshot for step %s; see %s/run.log\n' "${step}" "${run_dir}" >&2
    report_rc=1
  fi
done
printf 'Peak report: %s/memory_report/summary.md\n' "${run_dir}"
printf 'Viewer: https://pytorch.org/memory_viz (drag in a snapshots/step*/torch_memory*.pickle file)\n'
((pipeline_status[0] == 0)) || exit "${pipeline_status[0]}"
((pipeline_status[1] == 0)) || exit "${pipeline_status[1]}"
exit "${report_rc}"
