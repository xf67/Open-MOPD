"""Summarize every run into reports/summary.csv + summary.json (run on A6000j).

Steady-state window = after the warm-up steps: from the end of step WARMUP to
the end of the last recorded step.
  tokens/s      = sum(prompt+response tokens of measured steps) / window seconds
  resp tokens/s = generated tokens only
  peak GPU mem  = max nvidia-smi memory.used over the whole run (process-agnostic,
                  includes CUDA context, allocator cache, vLLM pools)
  avg GPU util  = mean nvidia-smi utilization.gpu over the steady window (500 ms samples)
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
OOM = re.compile(r'out of memory|OutOfMemoryError|CUDA error: out of memory', re.I)


def parse_samples(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        try:
            ts = datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            rows.append((ts, float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    return rows


def method_of(name):
    return name.split('-')[0]


def batching_of(name):
    rest = name.split('-', 1)[1]
    return rest


def summarize(run_dir):
    name = run_dir.name
    res = json.loads((run_dir / 'result.json').read_text()) if (run_dir / 'result.json').exists() else {}
    env = res.get('run', {}).get('env', {})
    warmup = int(env.get('WARMUP_STEPS', 2))
    steps = []
    if (run_dir / 'steps.jsonl').exists():
        steps = [json.loads(l) for l in (run_dir / 'steps.jsonl').read_text().splitlines() if l.strip()]
    log = (run_dir / 'train.log').read_text(errors='replace') if (run_dir / 'train.log').exists() else ''
    oom = bool(OOM.search(log))
    samples = parse_samples(run_dir / 'gpu-samples.csv')
    out = dict(run=name, method=method_of(name), batching=batching_of(name),
               exit_code=res.get('exit_code'), timed_out=res.get('timed_out'), oom=oom,
               steps_recorded=len(steps), warmup_steps=warmup,
               run_wall_s=round(res['finished_at'] - res['started_at'], 1) if res else None)
    if samples:
        out['peak_gpu_mem_gib'] = round(max(s[1] for s in samples) / 1024, 2)
        out['run_avg_gpu_util_pct'] = round(statistics.mean(s[2] for s in samples), 1)
    measured = [s for s in steps if s['step'] > warmup]
    if len(measured) >= 2 or (len(measured) >= 1 and any(s['step'] == warmup for s in steps)):
        start_row = [s for s in steps if s['step'] == warmup]
        t0 = start_row[0]['wall_time'] if start_row else None
        if t0 is None:
            t0 = measured[0]['wall_time']
            measured = measured[1:]
        t1 = measured[-1]['wall_time']
        window = t1 - t0
        tok = sum(s['metrics']['suite/batch_tokens'] for s in measured)
        resp = sum(s['metrics']['suite/batch_response_tokens'] for s in measured)
        seqs = sum(s['metrics']['suite/batch_sequences'] for s in measured)
        step_times = []
        prev = t0
        for s in measured:
            step_times.append(s['wall_time'] - prev)
            prev = s['wall_time']
        win = [x for x in samples if t0 <= x[0] <= t1]
        out.update(
            measured_steps=len(measured), window_s=round(window, 1),
            tokens_per_s=round(tok / window, 2), response_tokens_per_s=round(resp / window, 2),
            sequences_per_s=round(seqs / window, 4),
            mean_step_s=round(statistics.mean(step_times), 2),
            median_step_s=round(statistics.median(step_times), 2),
            mean_tokens_per_step=round(tok / len(measured), 1),
            mean_seqs_per_step=round(seqs / len(measured), 2),
            measured_tokens=int(tok),
            budget_reached=bool(measured[-1]['metrics'].get('suite/budget_reached', 0)),
        )
        if win:
            out['avg_gpu_util_pct'] = round(statistics.mean(x[2] for x in win), 1)
            out['steady_peak_gpu_mem_gib'] = round(max(x[1] for x in win) / 1024, 2)
            out['avg_gpu_power_w'] = round(statistics.mean(x[3] for x in win), 1)
        alloc = [s['metrics'].get('torch/max_allocated_gib') or s['metrics'].get('perf/max_memory_allocated_gb')
                 for s in steps]
        alloc = [a for a in alloc if a]
        if alloc:
            out['peak_torch_allocated_gib'] = round(max(alloc), 2)
        # phase timings when the framework reports them (verl)
        for key in ('timing_s/gen', 'timing_s/reward', 'timing_s/update_actor', 'timing_s/old_log_prob',
                    'timing_s/update_weights', 'timing_s/step'):
            vals = [s['metrics'][key] for s in measured if key in s['metrics']]
            if vals:
                out['mean_' + key.replace('/', '_')] = round(statistics.mean(vals), 3)
    if oom and not out.get('budget_reached'):
        out['status'] = 'OOM'
    elif out.get('budget_reached'):
        out['status'] = 'ok'
    elif out.get('timed_out'):
        out['status'] = 'timeout(partial)' if out.get('tokens_per_s') else 'timeout'
    elif res and res.get('exit_code') not in (0, None) and not out.get('tokens_per_s'):
        out['status'] = 'failed'
    else:
        out['status'] = 'partial' if out.get('tokens_per_s') else ('running' if not res else 'failed')
    return out


def main():
    rows = [summarize(d) for d in sorted((ROOT / 'runs').iterdir()) if d.is_dir()]
    kd = ROOT.parent / 'kdflow-mopd-a6000-20261118' / 'summary.json'
    if kd.exists():  # KDFlow sweep (separate harness, same workload/measurement rules); see its README
        for k in json.loads(kd.read_text()):
            rows.append(dict(run=k['run'], method='kdflow', batching=f"bs{k['bs']}", status=k['status'],
                             measured_steps=k['measured_steps'], warmup_steps=2,
                             tokens_per_s=round(k['tok_s'], 2), response_tokens_per_s=round(k['resp_tok_s'], 2),
                             peak_gpu_mem_gib=round(k['peak_gib'], 2),
                             avg_gpu_util_pct=round(k['avg_util'], 1) if k.get('avg_util') else None,
                             mean_step_s=round(k['s_per_step'], 2),
                             mean_tokens_per_step=round(k['tokens'] / k['measured_steps'], 1),
                             measured_tokens=k['tokens'], budget_reached=k['status'] == 'ok',
                             source='kdflow-mopd-a6000-20261118/summary.json'))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    (ROOT / 'reports').mkdir(exist_ok=True)
    with (ROOT / 'reports' / 'summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (ROOT / 'reports' / 'summary.json').write_text(json.dumps(rows, indent=2))
    cols = ['run', 'status', 'tokens_per_s', 'response_tokens_per_s', 'peak_gpu_mem_gib', 'avg_gpu_util_pct',
            'mean_step_s', 'mean_tokens_per_step', 'measured_steps']
    print('\t'.join(cols))
    for r in rows:
        print('\t'.join(str(r.get(c, '')) for c in cols))


if __name__ == '__main__':
    main()
