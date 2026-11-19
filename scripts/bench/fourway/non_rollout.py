"""Per-step time excluding rollout (generation), for every run (run on A6000j or locally).

For each measured step (step > WARMUP): wall = time between consecutive step ends
(steps.jsonl), rollout = the framework's own rollout timer for that step:
  verl (orig/zero/ours): metrics['timing_s/gen']
  Unsloth: OPD_STEP_METRICS.rollout_seconds (train.log)
  TRL: out/*/steps.json 'rollout'
non-rollout = wall - rollout. Writes reports/non_rollout.csv/.json.
"""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent


def rollout_times(run):
    name = run.name
    if name.startswith(('orig', 'zero', 'ours')):
        rows = [json.loads(l) for l in (run / 'steps.jsonl').read_text().splitlines() if l.strip()]
        return {r['step']: r['metrics'].get('timing_s/gen') for r in rows}
    if name.startswith('unsloth'):
        out = {}
        for line in (run / 'train.log').read_text(errors='replace').splitlines():
            if line.startswith('OPD_STEP_METRICS '):
                r = json.loads(line.split(' ', 1)[1])
                out[r['step']] = r['rollout_seconds']
        return out
    if name.startswith('trlvllm'):
        rows = [json.loads(l) for l in (run / 'steps.jsonl').read_text().splitlines() if l.strip()]
        return {r['step']: r['metrics'].get('suite/rollout_s') for r in rows}
    if name.startswith('trl'):
        files = list((run / 'out').glob('*/steps.json'))
        if not files:
            return {}
        return {r['step']: r['rollout'] for r in json.loads(files[0].read_text())}
    return {}


def main():
    result = []
    for run in sorted((ROOT / 'runs').iterdir()):
        steps_file = run / 'steps.jsonl'
        res_file = run / 'result.json'
        if not steps_file.exists() or not res_file.exists():
            continue
        warmup = int(json.loads(res_file.read_text())['run']['env'].get('WARMUP_STEPS', 2))
        steps = [json.loads(l) for l in steps_file.read_text().splitlines() if l.strip()]
        by_step = {s['step']: s for s in steps}
        roll = rollout_times(run)
        walls, rolls, toks = [], [], []
        for s in steps:
            k = s['step']
            if k <= warmup or (k - 1) not in by_step or roll.get(k) is None:
                continue
            walls.append(s['wall_time'] - by_step[k - 1]['wall_time'])
            rolls.append(roll[k])
            toks.append(s['metrics']['suite/batch_tokens'])
        if len(walls) < 2:
            continue
        non = [w - r for w, r in zip(walls, rolls)]
        result.append(dict(
            run=run.name, measured_steps=len(walls),
            mean_step_s=round(statistics.mean(walls), 2),
            mean_rollout_s=round(statistics.mean(rolls), 2),
            mean_non_rollout_s=round(statistics.mean(non), 2),
            rollout_share_pct=round(100 * sum(rolls) / sum(walls), 1),
            mean_tokens_per_step=round(statistics.mean(toks), 1),
            non_rollout_tokens_per_s=round(sum(toks) / sum(non), 1),
        ))
    kd = ROOT.parent / 'kdflow-mopd-a6000-20261118' / 'summary.json'
    if kd.exists():
        for k in json.loads(kd.read_text()):
            roll = k['rollout'] + k['sync']  # weight push into the engine counted as rollout, as for verl/TRL
            non = k['s_per_step'] - roll
            result.append(dict(run=k['run'], measured_steps=k['measured_steps'], mean_step_s=round(k['s_per_step'], 2),
                               mean_rollout_s=round(roll, 2), mean_non_rollout_s=round(non, 2),
                               rollout_share_pct=round(100 * roll / k['s_per_step'], 1),
                               mean_tokens_per_step=round(k['tokens'] / k['measured_steps'], 1),
                               non_rollout_tokens_per_s=round(k['tokens'] / (non * k['measured_steps']), 1)))
    keys = list(result[0])
    with (ROOT / 'reports' / 'non_rollout.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(result)
    (ROOT / 'reports' / 'non_rollout.json').write_text(json.dumps(result, indent=2))
    for r in result:
        print(r)


if __name__ == '__main__':
    main()
