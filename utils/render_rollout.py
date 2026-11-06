"""Render held-out T=100 interfaces and the effect of free rollout on error."""
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
    parser.add_argument('--folder', type=Path, default=Path('results/supervised/rollout_t100'))
    args = parser.parse_args()
    root = args.folder
    summary = json.loads((root/'summary.json').read_text())
    summary['models'].update(json.loads((root/'refinement.json').read_text()))
    anchors = json.loads((root/'anchors.json').read_text())
    chosen = anchors['model']
    with np.load(root/chosen/'predictions.npz') as data, np.load(root/'curvefno/test_predictions.npz') as direct:
        ids, times = data['ids'], data['times']
        if not np.array_equal(ids, direct['ids']):
            raise ValueError('Expected paired test cases')
        target, predicted = (data[key][:, -1] for key in ('target', 'free'))
        z = lambda x: (x[:, 0]+1j*x[:, 1])*np.sqrt(1+2*times[-1])
        label, estimate = z(target), z(predicted)
        baseline = direct['z'][:, -1]
        limit = 1.08*max(abs(label).max(), abs(estimate).max(), abs(baseline).max())
        fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.8), layout='constrained')
        for i, ax in enumerate(axes.flat):
            plot_curve(ax, label[i], color='#22272d', linewidth=1.55, label='Simulation')
            plot_curve(ax, baseline[i], color='#ce7040', linestyle='--', linewidth=1.1, label='CurveFNO direct')
            plot_curve(ax, estimate[i], color='#13788a', linewidth=1.2, label=f'{chosen} free')
            metrics = summary['models'][chosen]['test']['free']['cases'][i]
            ax.set_title(f'Case {ids[i]}  |  {100*metrics["terminal"]:.1f}% R', fontsize=10)
            ax.set_aspect('equal')
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xticks=[], yticks=[])
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.suptitle('Held-out interface contours at t = 100', fontsize=14)
        fig.legend(handles, names, ncols=3, loc='outside lower center', frameon=False)
        fig.savefig(root/'interfaces.png', dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for name, color in [('teacher', '#777985'), ('scheduled', '#4578ac'), ('pushforward', '#c27137'),
                        ('self', '#00786d'), ('varnoise', '#904781'), ('event', '#953839'),
                        ('consistency', '#333533'), ('stream', '#967300'), ('early', '#25874b')]:
        history = np.load(root/name/'history.npy')
        axes[0].plot(history[:, 0], history[:, 3], '.-', color=color, label=name)
    axes[0].set(xlabel='Fine-tuning optimizer step', ylabel='Validation free-rollout coordinate MSE')
    axes[0].legend(ncols=2, frameon=False, fontsize=8)
    candidate = summary['models'][chosen]['test']
    for name, label, color in [('direct', 'Direct from initial', '#ce7040'),
                               ('free', 'Free rollout', '#13788a'),
                               ('teacher', 'True state at each block', '#575c69')]:
        axes[1].plot(times, 100*np.asarray(candidate[name]['mean_error_by_time']), '.-', color=color, label=label)
    reset = anchors['partitions']['test']['resets']['fixed_80']
    axes[1].plot(times, 100*np.asarray(reset['mean_error_by_time']), ':', color='#7a438b', label='Numerical reset at 80')
    axes[1].set(xlabel='Physical time', ylabel='Test mean contour distance (% R)')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(root/'loss.png', dpi=175)
    plt.close(fig)


if __name__ == '__main__':
    main()
