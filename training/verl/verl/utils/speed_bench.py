"""Speed-benchmark instrumentation, enabled by setting ++trainer.suite_token_budget=<tokens>.

Environment: RUN_DIR, WARMUP_STEPS, TOKEN_BUDGET, MIN_MEASURED_STEPS, MAX_STEPS.
Pass verl.utils.speed_bench.seed_worker as ray_kwargs.ray_init.runtime_env.worker_process_setup_hook.
Origin: harness/long_observer.py of ai_gen/fourway-speed-a6000-20261118.

Records one JSON line per optimizer step with wall time and token counts, and
stops training once the measured window (after warm-up) has processed the
token budget. No model, optimizer or data changes.
"""
import json
import os
from pathlib import Path
import random
import time


def seed_worker():
    import numpy as np
    import torch
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_num_threads(8)


def _cfg():
    return dict(
        warmup=int(os.environ['WARMUP_STEPS']),
        budget=int(os.environ['TOKEN_BUDGET']),
        min_steps=int(os.environ['MIN_MEASURED_STEPS']),
        max_steps=int(os.environ.get('MAX_STEPS', '100000')),
    )


def batch_metrics(batch, counts, config, step):
    cfg = _cfg()
    total = int(batch.batch['attention_mask'].sum().item())
    if 'response_mask' in batch.batch.keys():
        response = int(batch.batch['response_mask'].sum().item())
    else:
        resp_len = batch.batch['responses'].shape[-1]
        response = int(batch.batch['attention_mask'][:, -resp_len:].sum().item())
    sequences = len(batch)
    counts['tokens'] += total
    counts['response_tokens'] += response
    counts['sequences'] += sequences
    if step > cfg['warmup']:
        counts.setdefault('measured_tokens', 0)
        counts.setdefault('measured_steps', 0)
        counts['measured_tokens'] += total
        counts['measured_steps'] += 1
    m_tok = counts.get('measured_tokens', 0)
    m_steps = counts.get('measured_steps', 0)
    last = (m_tok >= cfg['budget'] and m_steps >= cfg['min_steps']) or step >= cfg['max_steps']
    result = {
        'suite/batch_tokens': total, 'suite/batch_prompt_tokens': total - response,
        'suite/batch_response_tokens': response, 'suite/batch_sequences': sequences,
        'suite/cumulative_tokens': counts['tokens'],
        'suite/measured_tokens': m_tok, 'suite/measured_steps': m_steps,
        'suite/budget_reached': int(last),
    }
    return result, last


def record_step(metrics, step):
    numeric = {}
    for key, value in metrics.items():
        try:
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, (int, float)):
                numeric[key] = float(value)
        except Exception:
            pass
    out = Path(os.environ['RUN_DIR'])
    row = {'step': int(step), 'wall_time': time.time(), 'metrics': numeric}
    with (out / 'steps.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
    tmp = out / 'progress.tmp'
    tmp.write_text(json.dumps({'step': int(step), 'wall_time': row['wall_time'],
                               'budget_reached': numeric.get('suite/budget_reached', 0)}))
    tmp.replace(out / 'progress.json')


def finish_run():
    out = Path(os.environ['RUN_DIR'])
    (out / 'completed.json').write_text(json.dumps({'finished_at': time.time()}))
