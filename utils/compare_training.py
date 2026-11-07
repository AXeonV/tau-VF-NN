"""Compare original and expanded supervision at identical held-out time grids."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.predict import MODELS
from src.rollout import forecast
from src.trajectory_data import Trajectories
from src.trajectory_metrics import atomic_json, report_predictions


@torch.no_grad()
def predict(architecture, model, bank):
    rows = bank.splits['test']
    frames = torch.arange(len(bank.times), device=bank.times.device)
    if architecture != 'nn_direct':
        return forecast(model, bank, rows, frames).cpu().numpy()
    row = rows.repeat_interleave(len(frames))
    time = bank.times.repeat(len(rows))
    points = [model(bank.curves[i, 0], t) for i, t in zip(row.split(16), time.split(16))]
    return torch.cat(points).reshape(len(rows), len(frames), 2, -1).cpu().numpy()


def terminal_curves(x, time):
    return (x[:, -1, 0]+1j*x[:, -1, 1])*np.sqrt(1+2*time)


def plot_curve(axis, curve, **kwargs):
    closed = np.r_[curve, curve[0]]
    axis.plot(closed.real, closed.imag, **kwargs)


def render(root, result, outlines, truth, ids):
    colors = dict(original='#bf714c', dense='#117c8a')
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), layout='constrained')
    for name, axis in zip(MODELS, axes):
        for arm in ('original', 'dense'):
            history = np.load(root/arm/name/'history.npy')
            axis.semilogy(history[:, 0], history[:, 1], color=colors[arm], alpha=.38)
            axis.semilogy(history[:, 0], history[:, 2], color=colors[arm], label=f'{arm} validation')
        axis.set(title=name, xlabel='Optimizer step', ylabel='Loss')
        axis.grid(alpha=.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle('Training objective (faint) and validation coordinate MSE (solid)')
    fig.savefig(root/'loss.png', dpi=175)
    plt.close(fig)

    selected = [ids.index(case) for case in (7, 23, 39, 63)]
    limit = 1.07*max(abs(truth[selected]).max(), *(abs(z[selected]).max() for z in outlines.values()))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout='constrained')
    for row, name in enumerate(MODELS):
        for col, index in enumerate(selected):
            axis = axes[row, col]
            plot_curve(axis, truth[index], color='#282c30', linewidth=1.45, label='Simulation')
            errors = []
            for arm, style in [('original', '--'), ('dense', '-')]:
                plot_curve(axis, outlines[name, arm][index], color=colors[arm], linewidth=1.1,
                           linestyle=style, label=f'{arm} training')
                errors.append(100*result['models'][name][arm]['grids']['original']['cases'][index]['final'])
            axis.set_title(f'{name} · {ids[index]} · {errors[0]:.1f}% / {errors[1]:.1f}% R', fontsize=9)
            axis.set_aspect('equal')
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xticks=[], yticks=[])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, frameon=False, loc='outside lower center')
    fig.suptitle('Pure neural predictions at t = 100 on the common original time grid')
    fig.savefig(root/'interfaces.png', dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/supervised/runs/dense_t100'))
    parser.add_argument('--original', type=Path, default=Path('data/supervised/trajectories'))
    parser.add_argument('--dense', type=Path, default=Path('data/supervised/trajectories_t100_dense'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    banks = {name: Trajectories(folder, 1024, args.device, 100.)
             for name, folder in [('original', args.original), ('dense', args.dense)]}
    ids = {name: [bank.ids[i] for i in bank.splits['test'].tolist()] for name, bank in banks.items()}
    if ids['original'] != ids['dense']:
        raise ValueError('Expected identical held-out initial IDs')
    result = dict(test_ids=ids['original'], models={})
    outlines = {}
    for architecture, kind in MODELS.items():
        result['models'][architecture] = {}
        for arm in banks:
            folder = args.root/arm/architecture
            training = json.loads((folder/'metrics.json').read_text())
            checkpoint = torch.load(folder/'model.pt', map_location=args.device, weights_only=True)
            model = kind(**checkpoint['config']).to(args.device).eval()
            model.load_state_dict(checkpoint['model'])
            run = dict(step=int(checkpoint['step']), total_steps=training['steps'],
                       seed=training['seed'], train_cases=len(training['cases']['train']), grids={})
            for grid, bank in banks.items():
                prediction = predict(architecture, model, bank)
                run['grids'][grid] = report_predictions(bank, 'test', prediction)
                if grid == 'original':
                    outlines[architecture, arm] = terminal_curves(prediction, 100.)
            result['models'][architecture][arm] = run
            print(architecture, arm, 'terminal', {k: v['mean_final_geometry_distance']
                  for k, v in run['grids'].items()}, flush=True)
    atomic_json(args.root/'summary.json', result)
    original = banks['original']
    truth = terminal_curves(original.curves[original.splits['test']].cpu().numpy(), 100.)
    render(args.root, result, outlines, truth, ids['original'])


if __name__ == '__main__':
    main()
