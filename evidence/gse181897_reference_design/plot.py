"""Plot the verified external replay at its final 183 mm publication width."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RULES = ('shared_all_B', 'O_quarter', 'O_half', 'O_three_quarters', 'crossfit_2', 'crossfit_4')
LABELS = ('Shared 16', 'Separate 4 + 12', 'Separate 8 + 8',
          'Separate 12 + 4', 'Cross-fit 2', 'Cross-fit 4')
BLUE = '#286B8C'
GRAY = '#A4A8AC'


def draw(results_path, output):
    results_path, output = Path(results_path), Path(output)
    if output.exists():
        raise ValueError('Use a new figure output directory')
    result = json.loads(results_path.read_text())
    paired = result['paired_t']
    bootstrap = result['studentized_bootstrap']
    decisions = result['selection_decisions']['budgets']['16']
    families = tuple(decisions['shared_all_B']['family_risks'])
    # Canonical names and order are part of the frozen selection protocol.
    families = tuple(name for name in ('NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CellOT') if name in families)
    if len(families) != 7:
        raise ValueError('Expected all seven admitted model families')
    ranks = []
    for rule in RULES:
        scores = np.asarray([decisions[rule]['family_risks'][f] for f in families])
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite selection score')
        ranks.append(np.asarray([1 + np.sum(scores < value - 1e-12) for value in scores]))
    donors = result['donor_losses']
    shared = np.asarray([row['shared_loss'] for row in donors])
    selected = np.asarray([row['rule_loss'] for row in donors])
    if len(donors) != paired['independent_unit_count'] or not np.isfinite([shared, selected]).all():
        raise ValueError('Incomplete donor panel')
    recommended = result['recommended_rule']
    ci_t = np.asarray(paired['practical_contrast_interval'])
    ci_b = np.asarray(bootstrap['practical_contrast_interval'])
    if ci_t.shape != (2,) or ci_b.shape != (2,) or not np.isfinite([ci_t, ci_b]).all():
        raise ValueError('Both prespecified intervals are required')

    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.major.width': .6,
                         'ytick.major.width': .6, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'legend.frameon': False,
                         'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(7.2047244094, 3.3464566929))  # 183 by 85 mm.
    grid = fig.add_gridspec(1, 3, width_ratios=(1.4, 1, 1.18),
                           left=.16, right=.98, bottom=.28, top=.83, wspace=.62)
    a, b, c = (fig.add_subplot(grid[0, i]) for i in range(3))
    mesh = a.imshow(ranks, vmin=1, vmax=7, cmap='Blues_r', aspect='auto', interpolation='none')
    a.set_yticks(np.arange(6), LABELS)
    a.set_xticks(np.arange(7), families, rotation=45, ha='right', rotation_mode='anchor')
    for i, rule in enumerate(RULES):
        winner = families.index(decisions[rule]['family'])
        a.scatter(winner, i, s=24, facecolors='none', edgecolors='white', linewidths=.9)
    a.set_title('Selection rank\n16 selection donors', pad=9)
    a.set_xlabel('Model family')
    color_ax = fig.add_axes([.16, .10, .21, .025])
    bar = fig.colorbar(mesh, cax=color_ax, orientation='horizontal', ticks=[1, 4, 7])
    bar.set_label('Rank (1 = lowest loss)', labelpad=1, fontsize=6)
    bar.ax.tick_params(labelsize=5.5, length=2, pad=1)

    for i, (x, y) in enumerate(zip(shared, selected)):
        offset = (i / max(1, len(donors) - 1) - .5) * .14
        b.plot([offset, 1 + offset], [x, y], color=GRAY, alpha=.6, linewidth=.55, zorder=1)
        b.scatter([offset, 1 + offset], [x, y], s=8, c=[GRAY, BLUE], linewidths=0, zorder=2)
    b.set_xlim(-.25, 1.25)
    b.set_ylim(bottom=0)
    b.set_xticks([0, 1], ['Shared\n' + decisions['shared_all_B']['family'],
                         'Separate 8 + 8\n' + decisions[recommended]['family']])
    b.set_ylabel('Independent effect loss (standardized MSE)')
    b.set_title(f'Paired assessment\n{len(donors)} donors', pad=9)
    b.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=False)

    # Express all margin coordinates in milli-MSE units to avoid a floating
    # scientific-notation offset colliding with the axis label.
    estimate = 1000 * float(paired['practical_contrast'])
    ci_t, ci_b = 1000 * ci_t, 1000 * ci_b
    for y, interval in zip((1, 0), (ci_t, ci_b)):
        c.plot(interval, [y, y], color=BLUE, linewidth=1.2)
        c.vlines(interval, y - .04, y + .04, color=BLUE, linewidth=.8)
        c.plot([estimate], [y], marker='D', markersize=3.2, color=BLUE)
    c.axvline(0, color='#555555', linewidth=.65, linestyle=(0, (3, 3)))
    c.set_yticks([1, 0], ['Paired t', 'Studentized\nbootstrap'])
    c.set_ylim(-.7, 1.7)
    bounds = np.r_[ci_t, ci_b, 0]
    span = max(float(np.ptp(bounds)), abs(estimate) * .2, 1e-8)
    c.set_xlim(bounds.min() - .2 * span, bounds.max() + .2 * span)
    c.set_xlabel('Practical margin (10⁻³ MSE)\n0.95 × shared − separate')
    c.set_title('5% benefit criterion\nTwo-sided 95% intervals', pad=9)
    for label, axis in zip('abc', (a, b, c)):
        axis.text(-.14, 1.23, label, transform=axis.transAxes,
                  fontsize=8, fontweight='bold', va='top')
    output.mkdir(parents=True)
    fig.savefig(output / 'external_reference_design.pdf')
    fig.savefig(output / 'external_reference_design.svg')
    svg = output / 'external_reference_design.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(output / 'external_reference_design.png', dpi=300)
    fig.savefig(output / 'external_reference_design.tiff', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    (output / 'figure_receipt.json').write_text(json.dumps({
        'results_sha256': hashlib.sha256(results_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dimensions_mm': [183, 85], 'donors_shown': len(donors),
        'selection_designs': list(RULES), 'models_shown': list(families),
        'primary_donors_excluded': 0,
        'joint_practical_benefit': result['joint_practical_benefit'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    draw(args.results, args.output)
