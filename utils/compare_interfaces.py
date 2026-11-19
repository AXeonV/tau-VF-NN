"""Compare T=100 endpoints across independent initial shapes and render six distinct modes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.predict import load_model
from src.rollout import forecast
from src.trajectory_data import DEFAULT_DATA, Trajectories
from src.trajectory_metrics import has_crossing, terminal_distance
from .render import closed


MODELS = [('nn_direct', 'model.pt'), ('nn_geometry', 'finetuned.pt'), ('nn_boundary', 'model.pt')]
BANKS = ('drift_diagnostic_fresh', 'gradient_response_holdout', 'boundary_holdout')
STYLES = [
    ('Ground truth', '#222831', '-', 2.7),
    ('nn_direct', '#526eaa', (0, (1.1, 1.65)), 1.8),
    ('nn_geometry (fine-tuned)', '#cb632a', (0, (4.8, 2.4)), 1.85),
    ('nn_boundary', '#008774', '-', 1.7),
]


@torch.inference_mode()
def endpoints(device):
    models = []
    for name, filename in MODELS:
        path = Path('results/supervised/models')/name/filename
        model, nodes = load_model(name, path, device)
        models.append((name, model, nodes))
    if len({nodes for _, _, nodes in models}) != 1:
        raise ValueError('The comparison requires the same node count for each model')
    cases = []
    for name in BANKS:
        folder = DEFAULT_DATA.parent/name
        bank = Trajectories(folder, models[0][2], device, 100.)
        modes = {c['index']: c['dominant'] for c in json.loads((folder/'summary.json').read_text())['cases']}
        rows = bank.splits['test']
        frames = torch.arange(len(bank.times), device=device)
        truth = bank.curves[rows, -1]
        curves = [truth]
        errors = []
        for architecture, model, _ in models:
            if architecture == 'nn_direct':
                pred = model(bank.curves[rows, 0], bank.times[-1].expand(len(rows)))
            else:
                pred = forecast(model, bank, rows, frames)[:, -1]
            curves.append(pred)
            errors.append(terminal_distance(pred, truth))
        z = torch.stack(curves).cpu().numpy()
        z = z[:, :, 0]+1j*z[:, :, 1]
        for i, row in enumerate(rows.tolist()):
            case = bank.ids[row]
            cases.append(dict(case=case, mode=modes[case], curves=z[:, i], hd=np.array(errors)[:, i],
                              crossings=sum(has_crossing(v) for v in z[1:, i])))
    return cases


def select(cases):
    ranked = sorted(cases, key=lambda c: (max(c['hd']), c['crossings'], c['case']))
    chosen = {}
    for case in ranked:
        chosen.setdefault(case['mode'], case)
        if len(chosen) == 6:
            break
    if len(chosen) != 6:
        raise ValueError('Expected six different dominant modes')
    return [chosen[m] for m in sorted(chosen)]


def render(cases, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.8), sharex=True, sharey=True)
    limit = 1.06*max(np.abs(v).max() for c in cases for v in (c['curves'].real, c['curves'].imag))
    for case, ax in zip(cases, axes.flat):
        for z, (label, color, line, width) in zip(case['curves'], STYLES):
            z = closed(z)
            ax.plot(z.real, z.imag, label=label, color=color, linestyle=line, linewidth=width)
        ax.set(aspect='equal', xlim=(-limit, limit), ylim=(-limit, limit))
        ax.set_title(f"Case {case['case']}  ·  initial mode {case['mode']}", fontsize=12, loc='left')
        ax.tick_params(labelsize=9, length=3)
        for spine in ax.spines.values():
            spine.set_color('#c5cbd2')
    for ax in axes[-1]:
        ax.set_xlabel('x / R(100)')
    for ax in axes[:, 0]:
        ax.set_ylabel('y / R(100)')
    fig.suptitle('Viscous fingering interfaces at T = 100', y=.984, fontsize=18)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), ncols=4, loc='upper center',
               bbox_to_anchor=(.5, .947), frameon=False, handlelength=3, columnspacing=2.0)
    fig.subplots_adjust(left=.068, right=.985, bottom=.065, top=.855, wspace=.13, hspace=.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--out', type=Path, default=Path('docs/figures/direct_vs_finetuned_interfaces.png'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cases = endpoints(args.device)
    chosen = select(cases)
    print('Endpoint distances (%R), in direct / fine-tuned geometry / boundary order:')
    for case in chosen:
        print(case['case'], case['mode'], (100*case['hd']).round(2), 'crossing models:', case['crossings'])
    print(f'All {len(cases)} held-out cases:', (100*np.mean([c['hd'] for c in cases], axis=0)).round(2))
    render(chosen, args.out)


if __name__ == '__main__':
    main()
