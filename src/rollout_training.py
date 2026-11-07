"""Common supervised warm-up and Self Forcing training for the two rollout models."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .rollout import evaluation_grid, forecast
from .trajectory_data import DEFAULT_DATA, Trajectories
from .trajectory_metrics import atomic_json, report_predictions


def initialize(model_type, args):
    warm = torch.load(args.warm, map_location=args.device, weights_only=True) if args.warm else None
    config = dict(warm['config']) if warm else dict(horizon=100.)
    if args.architecture == 'nn_geometry':
        config['geometry'] = not args.no_geometry
    elif config.pop('geometry', False):
        raise ValueError('The four-frame model requires Fourier weights without a geometry branch')
    model = model_type(**config).to(args.device)
    if warm:
        weights = warm['model']
        if args.architecture == 'nn_geometry' and args.no_geometry:
            weights = {key: value for key, value in weights.items() if not key.startswith('geometry_branch.')}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected or any(not key.startswith('geometry_branch.') for key in missing):
            raise ValueError('Warm checkpoint does not match the shared Fourier architecture')
    return model, config


def main(architecture, model_type, loss_fn):
    parser = argparse.ArgumentParser(description=f'Train {architecture} with pure neural feedback')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/runs')/architecture)
    parser.add_argument('--warm', type=Path)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--report-every', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    if architecture == 'nn_geometry':
        parser.add_argument('--no-geometry', action='store_true', help='Pure Fourier one-frame ablation')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.report_every) < 1 or args.batch < 2:
        parser.error('Expected positive steps and a batch of at least two')
    args.architecture = architecture
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    bank = Trajectories(args.data, 1024, args.device, 100.)
    if any(len(rows) == 0 for rows in bank.splits.values()):
        raise ValueError('Training requires train, validation, and test initial conditions')
    model, config = initialize(model_type, args)
    grid = evaluation_grid(bank.times)
    validation = bank.splits['validation']
    expected = bank.curves[validation[:, None], grid]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00045 if args.warm else .0015, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00004)
    history, pending, best = [], [], float('inf')
    started = perf_counter()
    for step in range(1, args.steps+1):
        model.train()
        age = step+400 if args.warm and model.window_size == 4 else step
        loss = loss_fn(model, bank, grid, age, args.batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        pending.append(loss.detach())
        if step % args.report_every == 0 or step in (1, args.steps):
            prediction = forecast(model.eval(), bank, validation, grid)
            value = float((prediction-expected).square().mean())
            train = float(torch.stack(pending).mean())
            pending.clear()
            history.append([step, train, value])
            np.save(args.out/'history.npy', np.asarray(history))
            if value < best:
                best = value
                temporary = args.out/'model.tmp'
                torch.save(dict(model=model.state_dict(), config=config, nodes=1024, horizon=100.,
                                width=model.window_size, step=step, seed=args.seed, architecture=architecture),
                           temporary)
                temporary.replace(args.out/'model.pt')
            print(f'{architecture} step={step} loss={train:.5g} val={value:.5g} '
                  f'elapsed={perf_counter()-started:.1f}s', flush=True)
    checkpoint = torch.load(args.out/'model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    frames = torch.arange(len(bank.times), device=bank.times.device)
    predicted = forecast(model.eval(), bank, bank.splits['test'], frames).cpu().numpy()
    test = report_predictions(bank, 'test', predicted, export=args.out/'test_predictions.npz')
    atomic_json(args.out/'metrics.json', dict(architecture=architecture, steps=args.steps,
                best_step=checkpoint['step'], seed=args.seed, validation_mse=best, test=test,
                cases={name: [bank.ids[i] for i in rows.tolist()] for name, rows in bank.splits.items()}))
