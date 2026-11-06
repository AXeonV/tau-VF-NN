"""Train short-window updates and evaluate free rollouts from the initial contour."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .model import InterfaceStep
from ..trajectory_data import Trajectories, augment
from ..trajectory_metrics import atomic_json, probe_frames, report_predictions


@torch.no_grad()
def rollout(model, bank, split, batch):
    """Only frame zero is read from the labels during the rollout."""
    model.eval()
    trajectories = []
    for rows in bank.splits[split].split(batch):
        current = bank.curves[rows, 0]
        sequence = [current]
        for start, end in zip(bank.times[:-1], bank.times[1:]):
            current = model(current, start.expand(len(rows)), end.expand(len(rows)))
            sequence.append(current)
        trajectories.append(torch.stack(sequence, dim=1))
    return torch.cat(trajectories)


@torch.no_grad()
def local_predictions(model, bank, split, batch):
    model.eval()
    length = len(bank.times)-1
    rows = bank.splits[split].repeat_interleave(length)
    frames = torch.arange(length, device=rows.device).repeat(len(bank.splits[split]))
    outputs = []
    for group in torch.arange(len(rows), device=rows.device).split(batch):
        indices = frames[group]
        outputs.append(model(bank.curves[rows[group], indices], bank.times[indices], bank.times[indices+1]))
    prediction = torch.cat(outputs).reshape(len(bank.splits[split]), length, 2, -1)
    return torch.cat([bank.curves[bank.splits[split], :1], prediction], dim=1)


def training_window(bank, batch, length):
    train = bank.splits['train']
    device = bank.times.device
    rows = train[torch.randint(len(train), (batch,), device=device)]
    hop = torch.randint(1, min(3, (len(bank.times)-1)//length)+1, (batch,), device=device)
    upper = len(bank.times)-1-length*hop
    lower = torch.zeros_like(upper)
    late_count = max(1, batch//2)
    late = (bank.times >= .30*bank.times[-1]).long().argmax()
    lower[:late_count] = torch.minimum(upper[:late_count], late)
    start = lower+((upper-lower+1)*torch.rand(batch, device=device)).long()
    frames = start[:, None]+hop[:, None]*torch.arange(length+1, device=device)[None]
    curves = [bank.curves[rows, frames[:, j]] for j in range(length+1)]
    return augment(*curves), bank.times[frames]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/nn_ins'))
    parser.add_argument('--expected-cases', type=int, default=64)
    parser.add_argument('--horizon', type=float, default=1000.)
    parser.add_argument('--steps', type=int, default=4000)
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
    bank = Trajectories(args.data, args.nodes, args.device, args.horizon)
    if len(bank.ids) != args.expected_cases or any(len(split) == 0 for split in bank.splits.values()):
        raise ValueError('Training requires the configured accepted trajectories and all three splits')
    config = dict(horizon=float(bank.times[-1]), width=args.width, modes=args.modes,
                  depth=args.depth, sigma=bank.config['sigma'])
    model = InterfaceStep(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=5e-5)
    validation_frames = probe_frames(bank.times)
    validation_target = bank.curves[bank.splits['validation']][:, validation_frames]
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float('inf')
    started = perf_counter()
    print('split counts', {key: len(value) for key, value in bank.splits.items()}, flush=True)
    for step in range(1, args.steps+1):
        model.train()
        length = (1 if step <= args.steps/3 else 2 if step <= 2*args.steps/3 else 4)
        length = min(length, len(bank.times)-1)
        targets, times = training_window(bank, args.batch, length)
        current = targets[0]
        loss = 0.
        for frame in range(length):
            current = model(current, times[:, frame], times[:, frame+1])
            difference = current-targets[frame+1]
            tangent = (torch.roll(difference, -1, dims=-1)-difference)*(args.nodes/(2*np.pi))
            local = difference.square().mean(dim=(1, 2))+.005*tangent.square().mean(dim=(1, 2))
            duration = .5*torch.log((1+2*times[:, frame+1])/(1+2*times[:, 0]))
            weight = (.05/(duration+.01)).clamp(.4, 4.).square()
            loss = loss+(weight*local).mean()/length
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.report_every == 0 or step in (1, args.steps):
            validation = rollout(model, bank, 'validation', args.batch)
            score = float((validation[:, validation_frames]-validation_target).square().mean())
            history.append([step, float(loss.detach()), score])
            np.save(args.out/'history.npy', np.asarray(history))
            if score < best:
                best = score
                temporary = args.out/'model.tmp'
                torch.save(dict(model=model.state_dict(), config=config, nodes=args.nodes, step=step,
                                training_ids=[bank.ids[i] for i in bank.splits['train'].tolist()]), temporary)
                temporary.replace(args.out/'model.pt')
            print(f'step={step} unroll={length} training={history[-1][1]:.5e} '
                  f'validation={score:.5e} elapsed={perf_counter()-started:.1f}s', flush=True)
    checkpoint = torch.load(args.out/'model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    predictions = rollout(model, bank, 'test', args.batch).cpu().numpy()
    test = report_predictions(bank, 'test', predictions, export=args.out/'test_predictions.npz')
    single = local_predictions(model, bank, 'test', args.batch).cpu().numpy()
    local = report_predictions(bank, 'test', single)
    atomic_json(args.out/'metrics.json', dict(best_step=checkpoint['step'], validation_mse=best,
                test=test, teacher_forced=local, elapsed=perf_counter()-started,
                horizon=args.horizon, steps=args.steps, seed=args.seed, frames=len(bank.times),
                cases={name: [bank.ids[i] for i in ids.tolist()] for name, ids in bank.splits.items()}))
    print('held-out free rollout', json.dumps({key: value for key, value in test.items() if key != 'cases'}), flush=True)
    print('held-out teacher forced', json.dumps({key: value for key, value in local.items() if key != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
