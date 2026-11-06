"""Render the paired held-out interface and comparable validation losses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .compare import plot_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('results/supervised/world_t20'))
    parser.add_argument('--baseline', type=Path, default=Path('results/supervised/comparison/t20/nn'))
    args = parser.parse_args()
    root = args.folder
    summary = json.loads((root/'summary.json').read_text())
    chosen = summary['selected']
    if chosen == 'nn':
        chosen = summary['repeated_source']
    with np.load(root/'nn/test_predictions.npz') as base, np.load(root/chosen/'test_predictions.npz') as result:
        ids = base['ids']
        if not np.array_equal(result['ids'], ids):
            raise ValueError('The predictions must use the same test cases')
        baseline, target, repaired = base['z'][:, -1], base['target'][:, -1], result['z'][:, -1]
        limit = 1.08*max(abs(baseline).max(), abs(target).max(), abs(repaired).max())
        fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.8), layout='constrained')
        for i, ax in enumerate(axes.flat):
            plot_curve(ax, target[i], color='#31363b', linewidth=1.6, label='Simulation')
            plot_curve(ax, baseline[i], color='#d47245', linestyle='--', linewidth=1.2, label='CurveFNO')
            plot_curve(ax, repaired[i], color='#14768a', linewidth=1.2, label=chosen)
            b = summary['results']['nn']['test']['cases'][i]['final']
            r = summary['results'][chosen]['test']['cases'][i]['final']
            ax.set_title(f'Case {ids[i]}  |  {100*b:.1f}% → {100*r:.1f}% R', fontsize=10)
            ax.set_aspect('equal')
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.suptitle('Held-out interface contours at t = 20', fontsize=15)
        fig.legend(handles, names, ncols=3, loc='outside lower center', frameon=False)
        fig.savefig(root/'interfaces.png', dpi=175)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    base_history = np.load(args.baseline/'history.npy')
    baseline_mse = base_history[-1, 2]
    names = ['point', 'local', 'near', 'band', 'jepa', 'noise', 'oof']
    colors = ['#76757d', '#008f76', '#1b6ec2', '#a46416', '#b34092', '#5a55a3', '#b54332']
    if 'unroll' in summary['results']:
        names.append('unroll')
        colors.append('#242758')
    for ax in axes:
        ax.grid(alpha=.22)
        ax.set_xlabel('Repair optimizer step (frozen CurveFNO)')
    for name, color in zip(names, colors):
        history = np.load(root/name/'history.npy')
        axes[0].plot(history[:, 0], history[:, 2], '.-', color=color, label=name)
        axes[1].plot(history[:, 0], 100*history[:, 3], '.-', color=color, label=name)
    axes[0].axhline(baseline_mse, linestyle='--', color='#333333', linewidth=1, label='CurveFNO')
    axes[0].set_ylabel('Validation coordinate MSE')
    axes[1].set_ylabel('Validation contour selection score (% R)')
    axes[0].legend(ncols=2, frameon=False)
    fig.savefig(root/'loss.png', dpi=175)
    plt.close(fig)


if __name__ == '__main__':
    main()
