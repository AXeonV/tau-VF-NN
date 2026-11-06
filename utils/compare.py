"""Summarize six matched runs and render their interface and validation plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.trajectory_metrics import atomic_json, curve_difference, has_crossing


HORIZONS = (5, 20, 1000)
METHODS = ('nn', 'nn_ins')
COLORS = ('#287c9d', '#b65339')


def closed(curve):
    return np.r_[curve, curve[0]]


def plot_curve(ax, curve, *args, **kwargs):
    z = closed(curve)
    ax.plot(z.real, z.imag, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('results/supervised/comparison'))
    parser.add_argument('--case', type=int, default=7)
    args = parser.parse_args()
    root = args.folder
    summary = {'runs': [], 'paired_terminal_wins_for_nn_ins': {}}
    interface, geometry = plt.subplots(3, 2, figsize=(12, 16), layout='constrained')
    losses, validation = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    for row, horizon in enumerate(HORIZONS):
        tests = []
        label = None
        maximum = 0.
        for column, (method, color) in enumerate(zip(METHODS, COLORS)):
            folder = root/f't{horizon}'/method
            metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            test = metrics['test']
            assert metrics['horizon'] == horizon
            run = {name: metrics[name] for name in ('horizon', 'steps', 'seed', 'frames', 'best_step',
                                                     'validation_mse', 'elapsed')}
            run.update(method=method, **{name: value for name, value in test.items() if name != 'cases'})
            run['median_final_geometry_distance'] = float(np.median([case['final'] for case in test['cases']]))
            if 'teacher_forced' in metrics:
                run['teacher_forced'] = {name: value for name, value in metrics['teacher_forced'].items()
                                         if name != 'cases'}
            summary['runs'].append(run)
            tests.append(metrics)
            with np.load(folder/'test_predictions.npz') as archive:
                ids = archive['ids'].tolist()
                assert ids == metrics['cases']['test']
                assert archive['times'][0] == 0 and archive['times'][-1] == horizon
                times = archive['times']
                all_labels = archive['target']
                all_predictions = archive['z']
                if method == 'nn_ins':
                    radii = np.sqrt(1+2*times)
                    normalized = all_labels/radii[None, :, None]
                    differences = np.diff(normalized, axis=1)
                    held = normalized[:, -2]*radii[-1]
                    local_distances = [curve_difference(a[None], b[None], times[-1:], stride=1)['final']
                                       for a, b in zip(held, all_labels[:, -1])]
                    run['teacher_forced_homothetic_baseline'] = dict(
                        coordinate_mse=float(.5*np.mean(abs(differences)**2)*(len(times)-1)/len(times)),
                        mean_final_geometry_distance=float(np.mean(local_distances)))
                run['checkpoints'] = {}
                for t in HORIZONS:
                    if t > horizon:
                        continue
                    frame = int(np.argmin(abs(times-t)))
                    assert times[frame] == t
                    pairs = zip(all_predictions[:, frame], all_labels[:, frame])
                    distance = [curve_difference(x[None], y[None], times[frame:frame+1], stride=1)['final']
                                for x, y in pairs]
                    run['checkpoints'][str(t)] = dict(mean_geometry_distance=float(np.mean(distance)),
                                                     crossing_cases=sum(map(has_crossing, all_predictions[:, frame])))
                idx = ids.index(args.case)
                selected = all_labels[idx]
                if label is not None:
                    np.testing.assert_array_equal(selected, label)
                label = selected
                predicted = all_predictions[idx, -1]
            maximum = max(maximum, np.abs(label[-1]).max(), np.abs(predicted).max())
            ax = geometry[row, column]
            plot_curve(ax, label[0], color='#a0aab4', linewidth=.9, label='Initial')
            plot_curve(ax, label[-1], color='#44505d', linewidth=1.35, label='Time-stepped label')
            plot_curve(ax, predicted, '--', color=color, linewidth=1.5, label='Prediction')
            ax.set_aspect('equal')
            distance = next(case['final'] for case in test['cases'] if case['case'] == args.case)
            ax.set_title(f'{method} | T={horizon} | case {args.case:02}, distance {distance:.1%}')
            history = np.load(folder/'history.npy')
            validation[row].semilogy(history[:, 0], history[:, 2], color=color, label=method)
            best = int(np.argmin(history[:, 2]))
            validation[row].scatter(history[best, 0], history[best, 2], color=color, s=21, zorder=3)
        assert tests[0]['cases'] == tests[1]['cases']
        assert tests[0]['steps'] == tests[1]['steps'] and tests[0]['seed'] == tests[1]['seed']
        assert tests[0]['frames'] == tests[1]['frames']
        original, incremental = (result['test']['cases'] for result in tests)
        assert [c['case'] for c in original] == [c['case'] for c in incremental]
        summary['paired_terminal_wins_for_nn_ins'][str(horizon)] = sum(b['final'] < a['final'] for a, b in zip(original, incremental))
        for ax in geometry[row]:
            ax.set(xlim=(-1.07*maximum, 1.07*maximum), ylim=(-1.07*maximum, 1.07*maximum),
                   xlabel='x / R0', ylabel='y / R0')
        validation[row].set(title=f'Train to T={horizon}', xlabel='Optimizer step', ylabel='Validation coordinate MSE')
        validation[row].grid(alpha=.25)
        validation[row].legend(frameon=False)
    geometry[0, 0].legend(loc='upper right', frameon=False)
    interface.savefig(root/'distribution.png', dpi=155)
    losses.savefig(root/'loss.png', dpi=170)
    plt.close(interface)
    plt.close(losses)
    atomic_json(root/'summary.json', summary)
    for run in summary['runs']:
        print(run['horizon'], run['method'], f"endpoint distance {run['mean_final_geometry_distance']:.4%}",
              f"crossings {run['self_intersecting_cases']}/8", f"tip count match {run['final_tip_matches']}/8")


if __name__ == '__main__':
    main()
