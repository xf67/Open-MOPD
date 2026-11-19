"""Batch-size x method LaTeX tables with training/rollout framework in the header.

figures/table_throughput_by_bs.tex  : end-to-end training tokens/s
figures/table_non_rollout_by_bs.tex : time excluding rollout (s/step and tokens/s)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
summary = {r['run']: r for r in json.loads((ROOT / 'reports/summary.json').read_text())}
non = {r['run']: r for r in json.loads((ROOT / 'reports/non_rollout.json').read_text())}
BS = [1, 2, 4, 8, 16]
# key, name, training framework, rollout engine
METHODS = [
    ('orig', 'Open-MOPD', 'verl FSDP2 + CPU offload', 'vLLM (sep.\\ copy)'),
    ('zero', 'Open-MOPD+ZeRO-Off.', 'verl FSDP2 + DS CPUAdam', 'vLLM (sep.\\ copy)'),
    ('trl', 'TRL GKD-MOPD', 'TRL / HF Trainer', 'HF generate'),
    ('trlvllm', 'TRL+vLLM', 'TRL Distillation', 'vLLM colocate (sep.\\ copy)'),
    ('unsloth', 'Unsloth', 'Unsloth + TRL GKD', 'HF generate'),
    ('unslothvllm', 'Unsloth+vLLM', 'Unsloth + TRL GKD', 'vLLM colocate (sep.\\ copy)'),
    ('kdflow', 'KDFlow', 'FSDP2 + CPU offload', 'SGLang (sep.\\ copy)'),
    ('ours', '\\textbf{Ours}', 'verl FSDP + ours', 'vLLM (shared wts)'),
]


def header(first):
    cols = 'l' + 'r' * len(METHODS)
    return [r'\begin{tabular}{' + cols + '}', r'\toprule',
            first + ' & ' + ' & '.join(m[1] for m in METHODS) + r' \\',
            r'{\scriptsize Train} & ' + ' & '.join(r'{\scriptsize ' + m[2] + '}' for m in METHODS) + r' \\',
            r'{\scriptsize Rollout} & ' + ' & '.join(r'{\scriptsize ' + m[3] + '}' for m in METHODS) + r' \\',
            r'\midrule']


def table(path, caption, label, first, value):
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\caption{' + caption + '}', r'\label{' + label + '}']
    lines += header(first)
    for b in BS:
        cells = []
        for key, *_ in METHODS:
            cells.append(value(key, b))
        lines.append(f'{b} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    path.write_text('\n'.join(lines) + '\n')


def tput(key, b):
    r = summary.get(f'{key}-bs{b}')
    if not r or r.get('status') != 'ok':
        return 'OOM' if r and r.get('status') == 'OOM' else '--'
    s = f"{r['tokens_per_s']:.1f}"
    return r'\textbf{' + s + '}' if key == 'ours' else s


def nonroll(key, b):
    r = non.get(f'{key}-bs{b}')
    if not r:
        return '--'
    s = f"{r['mean_non_rollout_s']:.1f} / {r['non_rollout_tokens_per_s']:.0f}"
    return r'\textbf{' + s + '}' if key == 'ours' else s


table(ROOT / 'figures/table_throughput_by_bs.tex',
      'End-to-end training throughput (prompt+response tokens/s) on one RTX A6000.',
      'tab:throughput', 'Batch', tput)
table(ROOT / 'figures/table_non_rollout_by_bs.tex',
      'Per-step time excluding rollout (s/step) and tokens/s over that time. '
      'Rollout = each framework\'s own generation timer. Ours overlaps part of its CPU optimizer '
      'with the next rollout, which this split attributes to rollout.',
      'tab:nonrollout', 'Batch', nonroll)
print('ok')
