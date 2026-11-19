"""Train the equivariant model and select weights by the native-grid endpoint error."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .model import BoundaryFNO
from ..rollout import evaluation_grid, forecast, single_loss
from ..trajectory_data import DEFAULT_DATA, Trajectories
from ..trajectory_metrics import atomic_json, terminal_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/runs/nn_boundary'))
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=541)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 2:
        parser.error('Use at least one update and two trajectories per batch')

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    bank = Trajectories(args.data, 1024, args.device, 100.)
    valid = bank.splits['validation']
    if not len(valid) or not len(bank.splits['train']):
        parser.error('Training requires separate training and validation cases')
    config = dict(modes=64, depth=4)
    model = BoundaryFNO(**config).to(args.device)
    torch.manual_seed(args.seed+1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00005)
    grid = evaluation_grid(bank.times)
    native = torch.arange(len(bank.times), device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    best_step = None
    losses, history = [], []
    for step in range(1, args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = single_loss(model, bank, grid, step, args.batch, initial_starts=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())
        if step % 400 and step != args.steps:
            continue
        endpoint = forecast(model.eval(), bank, valid, native)[:, -1]
        distance = float('inf')
        if endpoint.isfinite().all():
            distance = float(np.mean(terminal_distance(endpoint, bank.curves[valid, -1])))
        history.append([step, float(torch.stack(losses).mean())])
        losses.clear()
        np.save(args.out/'history.npy', np.asarray(history))
        if distance < best:
            best, best_step = distance, step
            temporary = args.out/'model.tmp'
            torch.save(dict(model=model.state_dict(), config=config, architecture='nn_boundary',
                            nodes=1024, step=step, seed=args.seed), temporary)
            temporary.replace(args.out/'model.pt')
        atomic_json(args.out/'metrics.json', dict(architecture='nn_boundary', seed=args.seed,
                    steps=step, best_step=best_step, validation_endpoint=best if best_step else None))
        print(f'step={step} validation endpoint={distance:.5g}', flush=True)
    if best_step is None:
        raise RuntimeError('No finite validation endpoint in this run')


if __name__ == '__main__':
    main()
