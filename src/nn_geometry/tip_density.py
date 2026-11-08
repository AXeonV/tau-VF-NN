"""Paired fine-tuning and evaluation with sparse or dense late-tip supervision."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import GeometryFNO
from ..interface_events import match_births, tip_births
from ..rollout import evaluation_grid, forecast, single_loss, supervised_loss
from ..traditional.late_windows import WINDOWS
from ..trajectory_data import DEFAULT_DATA, Trajectories, augment
from ..trajectory_metrics import atomic_json, curve_difference, has_crossing, report_predictions


def windows(folder, bank, split):
    metadata = json.loads((folder/'summary.json').read_text())['windows']
    records = [record for record in metadata if record['split'] == split]
    ids = [bank.ids.index(record['case']) for record in records]
    if not records or not set(ids).issubset(bank.splits[split].tolist()):
        raise ValueError(f'Invalid {split} windows')
    times, curves = [], []
    for record in records:
        with np.load(folder/record['file']) as item:
            t = item['times']
            z = item['z']/np.sqrt(1+2*t[:, None])
        if len(t) != 31 or not np.all(np.diff(t) == .5):
            raise ValueError('Expected 31 half-unit frames')
        times.append(t)
        curves.append(np.stack([z.real, z.imag], axis=1))
    device = bank.times.device
    return (bank.curves[ids, 0], torch.tensor(np.stack(times), device=device, dtype=torch.float32),
            torch.tensor(np.stack(curves), device=device, dtype=torch.float32), records)


def local_loss(model, data, dense, batch, *, rate_balanced=False):
    initial, times, curves, _ = data
    device = times.device
    row = torch.randint(len(times), (batch,), device=device)
    sample = torch.rand(batch, device=device)
    if dense:
        current = 10+(20*sample).long()
        gap = 1
    else:
        current = 10+10*(2*sample).long()
        gap = 10
    previous, future = current-gap, current+gap
    first, state, old, label = augment(initial[row], curves[row, current],
                                        curves[row, previous], curves[row, future])
    prediction = model(first, state, old, times[row, previous], times[row, current],
                       times[row, future, None])
    loss = supervised_loss(prediction, label[:, None])
    return loss*(10/gap)**2 if rate_balanced else loss


@torch.inference_mode()
def predict_windows(model, data, gap, free):
    initial, times, truth, _ = data
    # Teacher-forced predictions share target times even when their lead differs.
    query = list(range(10+gap, 31, gap)) if free else list(range(11, 31))
    start = 10
    before = start-gap
    state, previous = truth[:, start], truth[:, before]
    output = []
    for future in query:
        if not free:
            start = future-gap
            before = start-1
            state, previous = truth[:, start], truth[:, before]
        predicted = model(initial, state, previous, times[:, before], times[:, start],
                          times[:, future, None])[:, 0]
        output.append(predicted)
        previous, state, before, start = state, predicted, start, future
    return query, torch.stack(output, dim=1).cpu().numpy()


def window_metrics(model, data, gap, free):
    query, predicted = predict_windows(model, data, gap, free)
    _, times, truth, records = data
    t = times[:, query].cpu().numpy()
    y = truth[:, query].cpu().numpy()
    if free:
        t = np.concatenate([times[:, 10:11].cpu().numpy(), t], axis=1)
        initial = truth[:, 10:11].cpu().numpy()
        y = np.concatenate([initial, y], axis=1)
        predicted = np.concatenate([initial, predicted], axis=1)
    cases = []
    for record, ts, z, target in zip(records, t, predicted, y):
        z = z[:, 0]+1j*z[:, 1]
        target = target[:, 0]+1j*target[:, 1]
        events = tip_births(z, ts)
        reference = [e for e in record['matched_events'] if ts[0] < e[0] < ts[-1]]
        matches = match_births(events, reference, time_limit=2.5)
        errors = [curve_difference(a[None], b[None], [0.])['maximum'] for a, b in zip(z, target)]
        cases.append(dict(window=record['file'], matches=matches, events=len(reference),
                          predicted_events=len(events), mean_distance=float(np.mean(errors[1:] if free else errors)),
                          final_distance=errors[-1], self_crossing=any(has_crossing(frame) for frame in z)))
    matches = [pair for case in cases for pair in case['matches']]
    return dict(mean_distance=float(np.mean([c['mean_distance'] for c in cases])),
                final_distance=float(np.mean([c['final_distance'] for c in cases])),
                matched=len(matches), events=sum(c['events'] for c in cases),
                self_crossing_windows=sum(c['self_crossing'] for c in cases),
                timing_mae=float(np.mean([p[0] for p in matches])) if matches else None,
                angle_mae=float(np.mean([p[1] for p in matches])) if matches else None, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', choices=['sparse', 'dense'], required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--windows', type=Path, default=WINDOWS)
    parser.add_argument('--paired-local', action='store_true',
                        help='Keep a shared coarse loss and balance auxiliary supervision by time gap')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    training, held_out = (windows(args.windows, bank, split) for split in ('train', 'test'))
    checkpoint = torch.load(args.warm, map_location=args.device, weights_only=True)
    model = GeometryFNO(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=3e-5)
    grid = evaluation_grid(bank.times)
    log = []
    for step in range(1, args.steps+1):
        if step % 2:
            age = 3000 + (step-1)//2
            loss = single_loss(model, bank, grid, age, 8, long_rollout=True, initial_starts=True)
        elif args.paired_local:
            coarse = local_loss(model, training, False, 8)
            auxiliary = local_loss(model, training, args.labels == 'dense', 8, rate_balanced=True)
            loss = (coarse+auxiliary)/2
        else:
            loss = local_loss(model, training, args.labels == 'dense', 8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        log.append(float(loss.detach()))
        if step in (1, args.steps) or step % 200 == 0:
            np.save(args.out/'history.npy', np.array(log))
            print(f'{args.labels} seed={args.seed} step={step} loss={np.mean(log[-200:]):.5g}', flush=True)
    model.eval()
    torch.save(dict(model=model.state_dict(), config=checkpoint['config'], nodes=1024, horizon=100.,
                    step=args.steps, seed=args.seed), args.out/'model.pt')
    results = {f'{"free" if free else "truth"}_{gap/2:g}': window_metrics(model, held_out, gap, free)
               for free in (False, True) for gap in (1, 10)}
    frames = torch.arange(len(bank.times), device=bank.times.device)
    predicted = forecast(model, bank, bank.splits['test'], frames).cpu().numpy()
    native = report_predictions(bank, 'test', predicted)
    atomic_json(args.out/'metrics.json', dict(seed=args.seed, steps=args.steps, labels=args.labels,
                paired_local=args.paired_local, windows=len(training[-1]), test_windows=len(held_out[-1]),
                local=results, native=native))
    print('Completed evaluation', flush=True)


if __name__ == '__main__':
    main()
