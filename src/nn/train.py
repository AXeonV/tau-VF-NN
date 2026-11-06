"""Supervised fitting and held-out geometry measurements for CurveFNO."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .data import Trajectories, augment
from .model import CurveFNO, signed_area
from ..traditional.generate import atomic_json, describe_geometry, curve_difference


def frame_indices(bank):
    horizon = bank.times[-1]
    probes = horizon*bank.times.new_tensor([.005, .02, .10, .30, .60, 1.])
    return (bank.times[:, None]-probes).abs().argmin(dim=0).unique()


@torch.no_grad()
def evaluate(model, bank, split, batch, *, geometry=False):
    model.eval()
    frames = (torch.arange(len(bank.times), device=bank.times.device)
              if geometry else frame_indices(bank))
    rows = bank.splits[split].repeat_interleave(len(frames))
    cols = frames.repeat(len(bank.splits[split]))
    losses = []
    if geometry:
        forecasts = []
    for group in torch.arange(len(rows), device=rows.device).split(batch):
        initial = bank.curves[rows[group], 0]
        target = bank.curves[rows[group], cols[group]]
        prediction = model(initial, bank.times[cols[group]])
        losses.append((prediction-target).square().mean(dim=(1, 2)))
        if geometry:
            forecasts.append(prediction.cpu().numpy())
    error = torch.cat(losses).reshape(-1, len(frames))
    result = dict(coordinate_mse=float(error.mean()),
                  final_coordinate_rms=float(error[:, -1].mean().sqrt()),
                  worst_frame_coordinate_rms=float(error.max().sqrt()))
    if geometry:
        predicted = np.concatenate(forecasts)
        target = bank.curves[rows, cols].cpu().numpy()
        times = bank.times[frames].cpu().numpy().astype(float)
        radii = np.sqrt(1+2*times)
        shape = (-1, len(frames), predicted.shape[-1])
        output = (predicted[:, 0]+1j*predicted[:, 1]).reshape(shape)*radii[None, :, None]
        labels = (target[:, 0]+1j*target[:, 1]).reshape(shape)*radii[None, :, None]
        result['cases'] = []
        for case_index, z, label in zip(bank.splits[split].tolist(), output, labels):
            contour = describe_geometry(z, times)
            truth = describe_geometry(label, times)
            result['cases'].append(dict(case=bank.ids[case_index],
                                        **curve_difference(z, label, times, stride=1),
                                        predicted_final_tips=contour['tips'][-1],
                                        label_final_tips=truth['tips'][-1],
                                        crossing_times=contour['crossing_times']))
        result['mean_final_geometry_distance'] = float(np.mean([case['final'] for case in result['cases']]))
        result['self_intersecting_cases'] = sum(bool(case['crossing_times']) for case in result['cases'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/model'))
    parser.add_argument('--expected-cases', type=int, default=64)
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
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    bank = Trajectories(args.data, args.nodes, args.device)
    if len(bank.ids) != args.expected_cases or any(len(split) == 0 for split in bank.splits.values()):
        raise ValueError('Training requires the configured accepted trajectories and all three splits')
    config = dict(horizon=float(bank.times[-1]), width=args.width, modes=args.modes, depth=args.depth)
    model = CurveFNO(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=5e-5)
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
    test = evaluate(model, bank, 'test', args.batch, geometry=True)
    atomic_json(args.out/'metrics.json', dict(best_step=checkpoint['step'],
                validation_mse=best, test=test, elapsed=perf_counter()-started,
                cases={name: [bank.ids[i] for i in ids.tolist()] for name, ids in bank.splits.items()}))
    print('held-out', json.dumps({key: value for key, value in test.items() if key != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
