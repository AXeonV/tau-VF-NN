"""Supervised fitting and held-out geometry measurements for CurveFNO."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..tip_loss import tip_geometry_loss
from ..trajectory_data import DEFAULT_DATA, Trajectories, augment
from .model import CurveFNO, signed_area
from ..trajectory_metrics import atomic_json, probe_frames, report_predictions


@torch.no_grad()
def evaluate(model, bank, split, batch, *, geometry=False, export=None):
    model.eval()
    frames = (torch.arange(len(bank.times), device=bank.times.device)
              if geometry else probe_frames(bank.times))
    rows = bank.splits[split].repeat_interleave(len(frames))
    cols = frames.repeat(len(bank.splits[split]))
    batches = []
    for group in torch.arange(len(rows), device=rows.device).split(batch):
        initial = bank.curves[rows[group], 0]
        prediction = model(initial, bank.times[cols[group]])
        if geometry:
            batches.append(prediction.cpu().numpy())
        else:
            target = bank.curves[rows[group], cols[group]]
            batches.append((prediction-target).square().mean(dim=(1, 2)))
    if geometry:
        predicted = np.concatenate(batches).reshape(len(bank.splits[split]), len(frames), 2, -1)
        return report_predictions(bank, split, predicted, export=export)
    error = torch.cat(batches).reshape(-1, len(frames))
    return dict(coordinate_mse=float(error.mean()),
                final_coordinate_rms=float(error[:, -1].mean().sqrt()),
                worst_frame_coordinate_rms=float(error.max().sqrt()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/runs/nn_direct'))
    parser.add_argument('--horizon', type=float)
    parser.add_argument('--warm', type=Path, help='Start from a compatible CurveFNO checkpoint')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--tip-weight', type=float, default=0., help='Weight for tip and valley supervision')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--nodes', type=int, default=1024)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--modes', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--report-every', type=int, default=250)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.report_every, args.width, args.depth, args.modes) < 1 or args.nodes < 64:
        parser.error('Expected positive training sizes and at least 64 nodes')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate must be finite and positive')
    if not math.isfinite(args.tip_weight) or args.tip_weight < 0:
        parser.error('Tip weight must be finite and nonnegative')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    bank = Trajectories(args.data, args.nodes, args.device, args.horizon)
    if any(len(split) == 0 for split in bank.splits.values()):
        raise ValueError('Training requires the configured accepted trajectories and all three splits')
    config = dict(horizon=float(bank.times[-1]), width=args.width, modes=args.modes, depth=args.depth)
    if args.warm is not None:
        warm = torch.load(args.warm, map_location=args.device, weights_only=True)
        if warm['config']['horizon'] != config['horizon'] or warm.get('nodes', args.nodes) != args.nodes:
            parser.error('Warm checkpoint must match the selected horizon and node count')
        config = warm['config']
    model = CurveFNO(**config).to(args.device)
    if args.warm is not None:
        model.load_state_dict(warm['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.05*args.lr)
    train_rows = bank.splits['train']
    late = torch.nonzero(bank.times >= .30*bank.times[-1]).flatten()
    history = []
    best = float('inf')
    started = perf_counter()
    print('split counts', {key: len(value) for key, value in bank.splits.items()}, flush=True)
    for step in range(1, args.steps+1):
        model.train()
        rows = train_rows[torch.randint(len(train_rows), (args.batch,), device=args.device)]
        cols = torch.randint(len(bank.times), (args.batch,), device=args.device)
        late_count = max(1, args.batch//2)
        cols[:late_count] = late[torch.randint(len(late), (late_count,), device=args.device)]
        initial, target = augment(bank.curves[rows, 0], bank.curves[rows, cols])
        prediction = model(initial, bank.times[cols])
        residual = prediction-target
        tangent_error = (torch.roll(residual, -1, dims=-1)-residual)*(args.nodes/(2*np.pi))
        area_error = signed_area(prediction)-signed_area(initial)
        loss = residual.square().mean()+.005*tangent_error.square().mean()+.02*area_error.square().mean()
        if args.tip_weight:
            loss = loss+args.tip_weight*tip_geometry_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.report_every == 0 or step in (1, args.steps):
            validation = evaluate(model, bank, 'validation', args.batch)
            history.append([step, float(loss.detach()), validation['coordinate_mse']])
            np.save(args.out/'history.npy', np.asarray(history))
            if validation['coordinate_mse'] < best:
                best = validation['coordinate_mse']
                temporary = args.out/'model.tmp'
                torch.save(dict(model=model.state_dict(), config=config, nodes=args.nodes,
                                step=step, training_ids=[bank.ids[i] for i in train_rows.tolist()]), temporary)
                temporary.replace(args.out/'model.pt')
            print(f'step={step} training={history[-1][1]:.5e} '
                  f'validation={validation["coordinate_mse"]:.5e} elapsed={perf_counter()-started:.1f}s', flush=True)
    checkpoint = torch.load(args.out/'model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    export = args.out/'test_predictions.npz' if args.save_predictions else None
    test = evaluate(model, bank, 'test', args.batch, geometry=True, export=export)
    atomic_json(args.out/'metrics.json', dict(best_step=checkpoint['step'],
                validation_mse=best, test=test, elapsed=perf_counter()-started,
                horizon=config['horizon'], steps=args.steps, seed=args.seed, frames=len(bank.times),
                learning_rate=args.lr, tip_weight=args.tip_weight,
                warm_checkpoint=str(args.warm) if args.warm else None,
                cases={name: [bank.ids[i] for i in ids.tolist()] for name, ids in bank.splits.items()}))
    print('held-out', json.dumps({key: value for key, value in test.items() if key != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
