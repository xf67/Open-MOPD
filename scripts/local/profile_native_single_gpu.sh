#!/usr/bin/env bash
# Nsight timing capture for the shared native profiling workload.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/profile_native_common.sh"
profile_init nsys "$@"
profile_build_command
profile_prepare_run
profile_run_training

if ((train_rc != 0 || tee_rc != 0)); then
  printf 'Training exit=%s, tee exit=%s; collecting any completed reports. See %s/run.log\n' \
    "${train_rc}" "${tee_rc}" "${run_dir}" >&2
fi

# Absolute Nsight -o paths isolate reports from other jobs changing Ray's
# session_latest symlink. repeat-shutdown:1 finalizes our single continuous
# capture at the final selected step, without waiting for worker teardown.
# A newly-created .nsys-rep can
# remain at zero bytes for several polls, so file-size stability alone is not a
# completion signal.  Wait until nsys can read a non-empty report containing
# every requested step marker.
trace_files=()
worker_source=""
declare -A checked_trace_states=()
deadline=$((SECONDS + trace_wait_seconds))
collector_log="${run_dir}/collector.log"
printf 'Waiting up to %s seconds for steps [%s] in %s\n' \
  "${trace_wait_seconds}" "${profile_steps_csv}" "${raw_dir}" | tee "${collector_log}"
while ((SECONDS <= deadline)); do
  mapfile -d '' trace_files < <(
    find "${raw_dir}" -maxdepth 1 -type f \
      \( -name 'worker_*.nsys-rep' -o -name 'worker_*.qdrep' \) -newer "${profile_marker}" -print0 | sort -z
  )
  for trace_file in "${trace_files[@]}"; do
    [[ -s "${trace_file}" ]] || continue
    trace_state="$(stat -c '%i:%s:%y' "${trace_file}")"
    if [[ "${checked_trace_states[${trace_file}]:-}" == "${trace_state}" ]]; then
      continue
    fi
    remaining=$((deadline - SECONDS))
    ((remaining > 0)) || remaining=1
    printf 'Checking %s\n' "${trace_file}" >>"${collector_log}"
    if ! nvtx_summary="$(
      timeout --kill-after=5s "${remaining}s" \
        "${nsys_bin}" stats --force-export=true --report nvtx_pushpop_sum \
        --format csv "${trace_file}" 2>>"${collector_log}"
    )"; then
      # Retry transient export failures even if size/mtime did not change.
      continue
    fi
    # Reject a report that changed while nsys was reading it; it is still being
    # finalized and will be retried after its size/mtime changes.
    if [[ "$(stat -c '%i:%s:%y' "${trace_file}")" != "${trace_state}" ]]; then
      continue
    fi
    checked_trace_states["${trace_file}"]="${trace_state}"
    # Match complete CSV range names so step 30 cannot satisfy step 3.
    if "${python_bin}" -c '
import csv, sys
ranges = {row[-1].lstrip(":") for row in csv.reader(sys.stdin) if row}
expected = {"openmopd::step::" + step for step in sys.argv[1].split(",")}
sys.exit(0 if expected <= ranges else 1)
' "${profile_steps_csv}" <<<"${nvtx_summary}"; then
      worker_source="${trace_file}"
      break
    fi
  done
  if [[ -n "${worker_source}" ]]; then
    break
  fi
  sleep 2
done

if [[ -z "${worker_source}" ]]; then
  printf 'No finalized Nsight report containing steps [%s] appeared within %s seconds in %s\n' \
    "${profile_steps_csv}" "${trace_wait_seconds}" "${raw_dir}" >&2
  printf 'See %s/run.log and %s/collector.log\n' "${run_dir}" "${run_dir}" >&2
  ((train_rc == 0)) || exit "${train_rc}"
  ((tee_rc == 0)) || exit "${tee_rc}"
  exit 1
fi

worker_report="${run_dir}/traces/$(basename "${worker_source}")"
cp -f "${worker_source}" "${worker_report}"

"${nsys_bin}" export --type sqlite --force-overwrite=true --quiet=true \
  --output "${run_dir}/traces/worker.sqlite" "${worker_report}"
"${python_bin}" "${repo_root}/scripts/local/analyze_nsys_memory.py" \
  "${run_dir}/traces/worker.sqlite" >"${run_dir}/memory_analysis.md"
"${python_bin}" "${repo_root}/scripts/local/analyze_teacher_forward_overlap.py" \
  "${run_dir}" >"${run_dir}/teacher_overlap.json"
"${nsys_bin}" stats \
  --report nvtx_pushpop_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_sum,cuda_api_sum \
  --format csv --output "${run_dir}/traces/worker_stats" "${worker_report}"

printf 'Selected worker report: %s\n' "${worker_report}"
printf 'Profiled continuous steps: %s\n' "${profile_steps_csv}"
printf 'Profile output: %s\n' "${run_dir}"
((train_rc == 0)) || exit "${train_rc}"
((tee_rc == 0)) || exit "${tee_rc}"
