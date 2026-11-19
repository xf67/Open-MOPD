"""Paper figures for the single-A6000 MOPD speed study.

Usage: python3 harness/paper_figures.py [experiment_root]   ->  figures/paper/*.pdf|png
Data: reports/summary.json (throughput, peak memory, GPU util) and reports/non_rollout.json.

Encoding: hue = framework family (palette validated on adjacent pairs; >3 hues on one line chart are
not CVD-separable on all pairs, so every series also has its own marker and a solid/dashed style:
solid = framework as published, dashed = "+ faster engine/optimizer" variant). Ours is drawn last and thicker.
Sizes target USENIX: one column = 3.33 in, full width = 7.0 in. Fonts are embedded (Type 42).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures' / 'paper'
OUT.mkdir(parents=True, exist_ok=True)
summary = {r['run']: r for r in json.loads((ROOT / 'reports/summary.json').read_text())}
nonroll = {r['run']: r for r in json.loads((ROOT / 'reports/non_rollout.json').read_text())}

BS = [1, 2, 4, 8, 16]
CAP = 47.4  # GiB usable on RTX A6000
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e5e4e0'
# key, label, family colour, linestyle, marker, bar hatch
METHODS = [
    ('orig', 'Open-MOPD', '#eb6834', '-', 's', ''),
    ('zero', 'Open-MOPD + ZeRO-Offload', '#eb6834', '--', 'v', '///'),
    ('unsloth', 'Unsloth', '#1baf7a', '-', 'D', ''),
    ('unslothvllm', 'Unsloth + vLLM', '#1baf7a', '--', 'd', '///'),
    ('trl', 'TRL', '#eda100', '-', '^', ''),
    ('trlvllm', 'TRL + vLLM', '#eda100', '--', '<', '///'),
    ('kdflow', 'KDFlow', '#e87ba4', '-', 'P', ''),
    ('ours', 'Ours', '#2a78d6', '-', 'o', ''),
]
BASELINES = [m[0] for m in METHODS if m[0] != 'ours']
NAME = {m[0]: m[1] for m in METHODS}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'hatch.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def get(key, bs, field='tokens_per_s', table=summary):
    r = table.get(f'{key}-bs{bs}')
    if r is None or (table is summary and r.get('status') != 'ok'):
        return None
    return r.get(field)


def style_axis(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def batch_axis(ax):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(BS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(b) for b in BS])
    ax.set_xlim(0.8, 20)
    ax.set_xlabel('Batch size (prompts / step)')


def lines(ax, field, table=summary, keys=None):
    for key, label, color, ls, marker, _ in METHODS:
        if keys and key not in keys:
            continue
        pts = [(b, get(key, b, field, table)) for b in BS]
        pts = [(b, v) for b, v in pts if v is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        hero = key == 'ours'
        ax.plot(xs, ys, ls=ls, color=color, lw=1.8 if hero else 1.1, marker=marker, ms=4.2 if hero else 3.4,
                markeredgecolor='white', markeredgewidth=0.5, label=label, zorder=5 if hero else 3)


def legend_handles(keys=None):
    return [Line2D([], [], color=c, ls=ls, marker=mk, lw=1.8 if k == 'ours' else 1.1, ms=4,
                   markeredgecolor='white', markeredgewidth=0.5, label=l)
            for k, l, c, ls, mk, _ in METHODS if not keys or k in keys]


def top_legend(ax):
    ax.legend(handles=legend_handles(), ncol=2, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.01),
              handlelength=2.4, columnspacing=1.0, borderaxespad=0.0)


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=220)
    plt.close(fig)


def log_tokens_axis(ax, ticks):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(t) for t in ticks])


# 1. End-to-end throughput (single column)
fig, ax = plt.subplots(figsize=(3.33, 2.7))
lines(ax, 'tokens_per_s')
batch_axis(ax)
log_tokens_axis(ax, [25, 50, 100, 200, 400, 800])
ax.set_ylim(25, 700)
ax.set_ylabel('Training throughput (tokens/s, log)')
style_axis(ax)
top_legend(ax)
save(fig, 'fig_throughput')

# 2. Speedup of ours over each baseline (heatmap, single column)
fig, ax = plt.subplots(figsize=(3.33, 2.05))
import numpy as np
grid = np.full((len(BASELINES), len(BS)), np.nan)
for i, k in enumerate(BASELINES):
    for j, b in enumerate(BS):
        o, v = get('ours', b), get(k, b)
        if o and v:
            grid[i, j] = o / v
from matplotlib.colors import LinearSegmentedColormap, LogNorm
cmap = LinearSegmentedColormap.from_list('blue', ['#e7f0fb', '#9cc2ef', '#2a78d6', '#123f7a'])
im = ax.imshow(grid, cmap=cmap, norm=LogNorm(vmin=1.0, vmax=4.0), aspect='auto')
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}×', ha='center', va='center', fontsize=6.3,
                    color='white' if v >= 2.0 else INK)
best = [max((get(k, b) or 0) for k in BASELINES) for b in BS]
for j, b in enumerate(BS):  # outline the strongest baseline at each batch size
    i = max(range(len(BASELINES)), key=lambda i: get(BASELINES[i], b) or 0)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
ax.set_xticks(range(len(BS)), [str(b) for b in BS])
ax.set_yticks(range(len(BASELINES)), [NAME[k] for k in BASELINES])
ax.set_xlabel('Batch size (prompts / step)')
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('Ours / baseline throughput (box: strongest baseline)', loc='left', color=INK, fontsize=7)
save(fig, 'fig_speedup_heatmap')

# 3. Peak GPU memory (single column)
fig, ax = plt.subplots(figsize=(3.33, 2.7))
lines(ax, 'peak_gpu_mem_gib')
batch_axis(ax)
ax.axhline(CAP, color=INK2, lw=0.7, ls=':')
ax.text(0.85, CAP + 0.6, 'A6000 capacity (47.4 GiB)', fontsize=6, color=INK2, va='bottom')
ax.set_ylim(15, 52)
ax.set_ylabel('Peak GPU memory (GiB)')
style_axis(ax)
top_legend(ax)
save(fig, 'fig_peak_memory')

# 4. Average GPU utilisation (single column)
fig, ax = plt.subplots(figsize=(3.33, 2.7))
lines(ax, 'avg_gpu_util_pct')
batch_axis(ax)
ax.set_ylim(0, 105)
ax.set_ylabel('Avg. GPU utilization (%)')
style_axis(ax)
top_legend(ax)
save(fig, 'fig_gpu_util')

# 5. Throughput vs peak memory frontier (single column)
fig, ax = plt.subplots(figsize=(3.33, 2.8))
for key, label, color, ls, marker, _ in METHODS:
    pts = [(get(key, b, 'peak_gpu_mem_gib'), get(key, b), b) for b in BS]
    pts = [p for p in pts if p[0] and p[1]]
    if not pts:
        continue
    xs, ys, bs_ = zip(*pts)
    hero = key == 'ours'
    ax.plot(xs, ys, ls=ls, color=color, lw=1.8 if hero else 1.0, marker=marker, ms=4.2 if hero else 3.4,
            markeredgecolor='white', markeredgewidth=0.5, label=label, zorder=5 if hero else 3)
    if hero:
        for x, y, b in pts:
            ax.annotate(f'bs{b}', (x, y), textcoords='offset points', xytext=(-4, 5), fontsize=5.5,
                        color=INK2, ha='right')
ax.axvline(CAP, color=INK2, lw=0.7, ls=':')
ax.text(CAP - 0.4, 30, 'A6000\ncapacity', fontsize=5.5, color=INK2, ha='right', va='bottom')
log_tokens_axis(ax, [25, 50, 100, 200, 400, 800])
ax.set_ylim(25, 700)
ax.set_xlim(17, 49)
ax.set_xlabel('Peak GPU memory (GiB)')
ax.set_ylabel('Training throughput (tokens/s, log)')
style_axis(ax)
ax.xaxis.grid(True, color=GRID, lw=0.5)
top_legend(ax)
save(fig, 'fig_throughput_vs_memory')

# 6. Time excluding rollout: tokens/s over non-rollout time (single column)
fig, ax = plt.subplots(figsize=(3.33, 2.7))
lines(ax, 'non_rollout_tokens_per_s', table=nonroll)
batch_axis(ax)
log_tokens_axis(ax, [25, 50, 100, 200, 400, 800, 1600])
ax.set_ylim(30, 1500)
ax.set_ylabel('Tokens / s excluding rollout (log)')
style_axis(ax)
top_legend(ax)
save(fig, 'fig_non_rollout_throughput')

# 7. Step-time split, rollout vs everything else, at bs1 and bs8 (full width, two panels)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
order = [m[0] for m in METHODS]
for ax, b in zip(axes, (1, 8)):
    for i, k in enumerate(order):
        r = nonroll.get(f'{k}-bs{b}')
        if not r:
            continue
        roll, rest = r['mean_rollout_s'], r['mean_non_rollout_s']
        ax.barh(i, roll, 0.68, color='#c3c2b7', edgecolor='white', lw=1.0)
        ax.barh(i, rest, 0.68, left=roll, color='#2a78d6' if k == 'ours' else '#52514e',
                edgecolor='white', lw=1.0)
        ax.text(roll + rest + 0.5, i, f'{roll + rest:.1f} s', va='center', fontsize=6, color=INK)
    ax.set_title(f'Batch size {b}', loc='left', color=INK)
    ax.set_xlabel('Mean time per optimizer step (s)')
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(nonroll[f'{k}-bs{b}']['mean_step_s'] for k in order if f'{k}-bs{b}' in nonroll) * 1.18)
axes[0].set_yticks(range(len(order)), [NAME[k] for k in order])
axes[0].invert_yaxis()
fig.legend(handles=[Patch(color='#c3c2b7', label='Rollout (generation + weight push)'),
                    Patch(color='#52514e', label='Scoring + training + optimizer (baselines)'),
                    Patch(color='#2a78d6', label='Scoring + training + optimizer (ours)')],
           ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.55, 0.98), fontsize=6)
fig.tight_layout(w_pad=1.0)
save(fig, 'fig_step_breakdown')

# 8. Dynamic token batching (ours only, single column)
dyn = [r for r in summary.values() if r.get('method') == 'ours']
fixed = {int(r['batching'][2:]): r for r in dyn if r['batching'].startswith('bs') and r.get('status') == 'ok'}
dynr = [r for r in dyn if 'dyn' in r['batching']]
fig, ax = plt.subplots(figsize=(3.33, 2.0))
xs, ys = zip(*sorted((r['mean_tokens_per_step'], r['tokens_per_s']) for r in fixed.values()))
ax.plot(xs, ys, color='#2a78d6', lw=1.6, marker='o', ms=4, markeredgecolor='white', markeredgewidth=0.5,
        label='Fixed batch (bs 1–16)')
for b, r in fixed.items():
    ax.annotate(f'bs{b}', (r['mean_tokens_per_step'], r['tokens_per_s']), textcoords='offset points',
                xytext=(3, -8), fontsize=5.5, color=INK2)
ok = [r for r in dynr if r.get('status') == 'ok']
ax.plot([r['mean_tokens_per_step'] for r in ok], [r['tokens_per_s'] for r in ok], ls='none', marker='D', ms=4.2,
        color='#6250d6', markeredgecolor='white', markeredgewidth=0.5, label='Dynamic token batch')
for r in ok:
    lab = r['batching'].replace('dyn', '').replace('-chunk', 'k, chunk ').replace('kk', 'k')
    ax.annotate(lab, (r['mean_tokens_per_step'], r['tokens_per_s']), textcoords='offset points', xytext=(4, 3),
                fontsize=5.5, color='#6250d6')
oom = [r['batching'] for r in dynr if r.get('status') != 'ok']
if oom:
    ax.text(0.98, 0.04, 'OOM: ' + ', '.join(o.replace('dyn', '').replace('-chunk', '/c') for o in sorted(oom)),
            transform=ax.transAxes, ha='right', fontsize=5.5, color=INK2)
ax.set_xscale('log', base=2)
ax.xaxis.set_major_locator(FixedLocator([1024, 2048, 4096, 8192, 16384]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(['1k', '2k', '4k', '8k', '16k'])
ax.set_ylim(60, 640)
ax.set_xlabel('Trained tokens per optimizer step (log)')
ax.set_ylabel('Training throughput (tokens/s)')
style_axis(ax)
ax.legend(frameon=False, loc='upper left')
save(fig, 'fig_dynamic_batching')

# 9. Combined overview (full width): (a) throughput, (b) peak memory, (c) GPU util, (d) excl.-rollout throughput
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
panels = [('tokens_per_s', summary, '(a) Throughput (tok/s, log)', True),
          ('peak_gpu_mem_gib', summary, '(b) Peak GPU memory (GiB)', False),
          ('avg_gpu_util_pct', summary, '(c) Avg. GPU util. (%)', False),
          ('non_rollout_tokens_per_s', nonroll, '(d) Tok/s excl. rollout (log)', True)]
for ax, (field, table, title, logy) in zip(axes, panels):
    lines(ax, field, table)
    batch_axis(ax)
    ax.set_xlabel('Batch size')
    ax.set_title(title, loc='left', color=INK, fontsize=7)
    style_axis(ax)
    if logy:
        ticks = [25, 50, 100, 200, 400, 800] if field == 'tokens_per_s' else [50, 100, 200, 400, 800, 1600]
        log_tokens_axis(ax, ticks)
        ax.set_ylim(25 if field == 'tokens_per_s' else 30, 700 if field == 'tokens_per_s' else 1500)
axes[1].axhline(CAP, color=INK2, lw=0.7, ls=':')
axes[1].set_ylim(15, 50)
axes[2].set_ylim(0, 105)
fig.legend(handles=legend_handles(), ncol=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.99),
           handlelength=2.2, columnspacing=0.9, fontsize=5.8)
fig.tight_layout(w_pad=0.6)
save(fig, 'fig_overview')

print('wrote', sorted(p.name for p in OUT.glob('*.pdf')))
