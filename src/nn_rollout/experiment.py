"""Matched T=100 experiments for temporal bundling and self-generated context."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.integrate import trapezoid

from .events import match_events, split_events
from .model import WindowFNO
from ..nn.model import signed_area
from ..nn_world.train import complex_curves, frame_errors
from ..trajectory_data import Trajectories, augment
from ..trajectory_metrics import atomic_json, has_crossing


ARCHITECTURES = {
    'direct': {},
    'pairs': {},
    'block': dict(coupled=True),
    'anchor': dict(coupled=True, anchor=True),
    'history': dict(coupled=True, anchor=True, history=True),
    'band_anchor': dict(coupled=True, anchor=True, history=True, history_bands=True),
}
FINETUNES = ('teacher', 'scheduled', 'pushforward', 'self', 'varnoise', 'event', 'consistency', 'stream', 'early')


def closest(times, query):
    right = torch.searchsorted(times, query.contiguous()).clamp_max(len(times)-1)
    left = (right-1).clamp_min(0)
    return torch.where((times[right]-query).abs() < (times[left]-query).abs(), right, left)


def choose(times, first, last, count=4):
    fraction = torch.arange(1, count+1, device=times.device)/count
    query = times[first, None]+fraction*(times[last, None]-times[first, None])
    return closest(times, query)


def evaluation_grid(times):
    if float(times[-1]) != 100.:
        raise ValueError('This comparison uses T=100')
    queries = [.25, 1., 2.4, 5., 8.35, 10., 15., 20., *range(25, 101, 5)]
    return torch.cat([torch.zeros(1, dtype=torch.long, device=times.device),
                      closest(times, times.new_tensor(queries))])


def previous_index(times, first):
    return closest(times, times[first]-.08*(1+times[first]))


def supervised_loss(prediction, target, event=False):
    residual = prediction-target
    tangent = (torch.roll(residual, -4, -1)-residual)*(residual.shape[-1]/(8*torch.pi))
    loss = residual.square().mean()+.004*tangent.square().mean()
    if event:
        # Local prominence and across-time changes near emerging valleys and tips.
        radius = target.norm(dim=-2)
        spacing = radius.shape[-1]//32
        curvature = (radius-.5*(radius.roll(spacing, -1)+radius.roll(-spacing, -1))).abs()
        change = torch.diff(radius, dim=1, prepend=radius[:, :1]).abs()
        weight = 1+6*(curvature+change).clamp_max(.5)
        radial_error = prediction.norm(dim=-2)-radius
        temporal = torch.diff(radial_error, dim=1)
        loss = loss+.25*(weight*radial_error.square()).mean()+.2*temporal.square().mean()
    return loss


def perturb(x, time, common=None):
    k = torch.arange(x.shape[-1]//2+1, device=x.device)
    coeff = torch.randn((len(x), 2, len(k)), dtype=torch.complex64, device=x.device)
    # Allow independent masking strength at each history position, with correlated shape errors.
    coeff *= torch.exp(-(k/22)**2)/(1+k/4).sqrt()
    field = torch.fft.irfft(coeff, n=x.shape[-1], norm='forward')
    field = field/field.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    if common is not None:
        field = .6*common+.8*field
    strength = .10*torch.sqrt(time/100)*torch.rand(len(x), device=x.device).square()
    corrupted = x+strength[:, None, None]*field
    target = signed_area(x)
    fixed = corrupted*torch.sqrt(target/torch.maximum(signed_area(corrupted), .1*target))[:, None, None]
    return fixed, field


def train_loss(model, bank, name, batch, stage, step, steps):
    device, times = bank.times.device, bank.times
    rows = bank.splits['train']
    row = rows[torch.randint(len(rows), (batch,), device=device)]
    last = torch.randint(1, len(times), (batch,), device=device)
    late = len(times)-19
    last[:3*batch//4] = torch.randint(late, len(times), (3*batch//4,), device=device)
    if name == 'early':
        targets = (.35+19.65*torch.rand(batch//2, device=device))
        last[-batch//2:] = closest(times, targets)
    distance = torch.randint(1, 12, (batch,), device=device)
    start = (last-distance).clamp_min(0)
    wide = closest(times, times[last]*torch.rand(batch, device=device))
    coin = torch.rand(batch, device=device)
    start = torch.where(coin < .45, wide, start)
    start = torch.where(coin > .75, 0, start)
    if name in ('direct', 'stream'):
        start = torch.zeros_like(start)
    # A random reset mix lets the same operator retain direct and local supervision.
    free = stage == 'finetune' and name in ('scheduled', 'pushforward', 'self', 'event', 'consistency', 'stream', 'early')
    count = 6 if name == 'stream' else 3
    endpoints = choose(times, start, last, count) if free else last[:, None]
    endpoints = torch.cat([start[:, None], endpoints], dim=1)
    before = previous_index(times, start)
    windows = [choose(times, a, b) for a, b in zip(endpoints.T, endpoints.T[1:])]
    source = bank.curves[row, start]
    previous = bank.curves[row, before]
    labels = [bank.curves[row[:, None], query] for query in windows]
    initial, state, previous, *targets = augment(bank.curves[row, 0], source, previous,
                                                 *(x for block in labels for x in block.unbind(1)))
    losses = []
    if name == 'varnoise':
        state, field = perturb(state, times[start])
        previous, _ = perturb(previous, times[before], field)
    for block, query in enumerate(windows):
        a = endpoints[:, block]
        burn_in = count-1 if name == 'pushforward' else count-2
        detach = free and block < burn_in
        with torch.set_grad_enabled(not detach):
            prediction = model(initial, state, previous, times[before], times[a], times[query])
            target = torch.stack(targets[4*block:4*block+4], dim=1)
            if not detach:
                losses.append(supervised_loss(prediction, target, event=name == 'event'))
            if block == len(windows)-1:
                if name == 'consistency':
                    zeros = torch.zeros_like(a)
                    direct_query = choose(times, zeros, last)
                    direct = model(initial, initial, initial, times[zeros], times[zeros], times[direct_query])[:, -1]
                    losses.append(supervised_loss(direct[:, None], target[:, -1:])+
                                  .5*(direct-prediction[:, -1]).square().mean())
                break
            previous, state = prediction[:, -2], prediction[:, -1]
            before = query[:, -2]
            if name == 'scheduled':
                chance = .15+.75*step/steps
                feed = torch.rand((batch, 1, 1), device=device) < chance
                state = torch.where(feed, state.detach(), target[:, -1])
                previous = torch.where(feed, previous.detach(), target[:, -2])
    # Keep a matched single-block objective in all self-fed regimes.
    if free:
        query = choose(times, start, last)
        clean = augment(bank.curves[row, 0], source, bank.curves[row, previous_index(times, start)],
                        *bank.curves[row[:, None], query].unbind(1))
        fit = model(*clean[:3], times[previous_index(times, start)], times[start], times[query])
        losses.append(supervised_loss(fit, torch.stack(clean[3:], dim=1), event=name == 'event'))
    return torch.stack(losses).mean()


def predict_block(model, initial, state, previous, before, start, queries, refiner=None):
    prediction = model(initial, state, previous, before, start, queries)
    if refiner is not None:
        prediction = refiner(initial, state, prediction, start, queries)
    return prediction


@torch.no_grad()
def forecast(model, bank, rows, frames, mode, refiner=None, window_size=4):
    times = bank.times
    initial = bank.curves[rows, 0]
    zeros = torch.zeros(len(rows), dtype=times.dtype, device=times.device)
    output = [initial[:, None]]
    if mode == 'direct':
        indices = frames[1:].repeat(len(rows))
        state = initial.repeat_interleave(len(frames)-1, dim=0)
        query = choose(times, torch.zeros_like(indices), indices)
        t0 = zeros.repeat_interleave(len(frames)-1)
        found = [predict_block(model, z, z, z, a, a, times[q], refiner)[:, -1]
                 for z, a, q in zip(state.split(12), t0.split(12), query.split(12))]
        output.append(torch.cat(found).reshape(len(rows), len(frames)-1, *initial.shape[1:]))
    else:
        previous = state = initial
        before = times[0]
        for start, query in zip(frames[::window_size], frames[1:].split(window_size)):
            t0 = times[start]
            if mode == 'teacher':
                state = bank.curves[rows, start]
                prior = previous_index(times, start)
                previous = bank.curves[rows, prior]
                before = times[prior]
            prediction = predict_block(model, initial, state, previous, before.expand_as(zeros),
                                       t0.expand_as(zeros), times[query].expand(len(rows), -1), refiner)
            output.append(prediction)
            previous = prediction[:, -2] if len(query) >= 2 else state
            before = times[query[-2]] if len(query) >= 2 else t0
            state = prediction[:, -1]
    return torch.cat(output, dim=1).cpu().numpy()


def summarize(prediction, target, times, ids, target_events=None):
    if not np.isfinite(prediction).all():
        raise ValueError('Expected finite interface predictions')
    errors = frame_errors(prediction, target)
    cases = []
    predictions = complex_curves(prediction)
    labels = complex_curves(target)
    for i, (case, curves, truth) in enumerate(zip(ids, predictions, labels)):
        cross = [float(t) for t, z in zip(times, curves) if has_crossing(z)]
        predicted = split_events(curves, times)
        events = split_events(truth, times) if target_events is None else target_events[i]
        matched = match_events(predicted, events)
        bad = (errors[i] > .10) | np.isin(times, cross)
        breach = np.flatnonzero(bad)
        cases.append(dict(case=case, terminal=float(errors[i, -1]), worst=float(errors[i].max()),
                          first_bad=float(times[breach[0]]) if len(breach) else None, crossings=cross,
                          predicted_events=predicted, label_events=events, matched=matched))
    matched = [match for case in cases for match in case['matched']]
    events = sum(len(x['label_events']) for x in cases)
    predicted = sum(len(x['predicted_events']) for x in cases)
    return dict(terminal=float(errors[:, -1].mean()), worst=float(errors.max(axis=1).mean()),
                time_mean=float(trapezoid(errors, times, axis=1).mean()/times[-1]),
                mean_error_by_time=errors.mean(axis=0).tolist(), coordinate_mse=float(np.mean((prediction-target)**2)),
                crossing_cases=sum(bool(x['crossings']) for x in cases), detected_events=len(matched),
                label_events=events, predicted_events=predicted, event_f1=2*len(matched)/max(1, events+predicted),
                matched_time_error=float(np.mean([x['time_error'] for x in matched])) if matched else None,
                cases=cases)


def fit(name, stage, bank, frames, out, steps, seed, base=None):
    torch.manual_seed(seed)
    started = perf_counter()
    config = ARCHITECTURES[name] if stage == 'structure' else ARCHITECTURES['history']
    model = WindowFNO(**config).cuda()
    if base is not None:
        model.load_state_dict(base['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015 if base is None else .00045, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00004)
    validation = bank.splits['validation']
    target = bank.curves[validation[:, None], frames].cpu().numpy()
    best = float('inf')
    records = []
    for step in range(1, steps+1):
        loss = train_loss(model, bank, name, 8, stage, step, steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 300 == 0 or step in (1, steps):
            direct = forecast(model.eval(), bank, validation, frames, 'direct')
            free = forecast(model, bank, validation, frames, 'free')
            endframes = np.arange(0, len(frames), 4)
            direct_err = frame_errors(direct[:, endframes], target[:, endframes])
            free_err = frame_errors(free[:, endframes], target[:, endframes])
            direct_score = float(.5*direct_err[:, -1].mean()+.5*direct_err.mean())
            free_score = float(.5*free_err[:, -1].mean()+.5*free_err.mean())
            score = direct_score if name == 'direct' else .4*direct_score+.6*free_score
            records.append([step, float(loss.detach()), float(np.mean((direct-target)**2)),
                            float(np.mean((free-target)**2)), direct_score, free_score])
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(), config=config, seed=seed, step=step), out/'model.pt')
            print(f'{name} step={step} loss={records[-1][1]:.5g} direct={direct_score:.4f} '
                  f'free={free_score:.4f} elapsed={perf_counter()-started:.1f}s', flush=True)
            model.train()
    np.save(out/'history.npy', np.asarray(records))
    checkpoint = torch.load(out/'model.pt', weights_only=True, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    result = dict(name=name, stage=stage, seed=seed, steps=steps, best_step=checkpoint['step'],
                  validation_score=best, parameters=sum(p.numel() for p in model.parameters()))
    times = bank.times[frames].cpu().numpy()
    for split in ('validation', 'test'):
        rows = bank.splits[split]
        ids = [bank.ids[i] for i in rows.tolist()]
        target = bank.curves[rows[:, None], frames].cpu().numpy()
        labels = [split_events(z, times) for z in complex_curves(target)]
        result[split] = {}
        export = dict(times=times, ids=np.asarray(ids), target=target)
        for mode in ('direct', 'free', 'teacher'):
            predictions = forecast(model.eval(), bank, rows, frames, mode)
            result[split][mode] = summarize(predictions, target, times, ids, labels)
            export[mode] = predictions
        if split == 'test':
            np.savez_compressed(out/'predictions.npz', **export)
    result['seconds'] = perf_counter()-started
    atomic_json(out/'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/rollout_t100'))
    parser.add_argument('--stage', choices=['structure', 'finetune', 'summarize'], required=True)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    bank = Trajectories(args.data, 1024, 'cuda', 100.)
    frames = evaluation_grid(bank.times)
    names = ARCHITECTURES if args.stage == 'structure' else FINETUNES if args.stage == 'finetune' else ()
    if args.only and not set(args.only) <= set(names):
        parser.error('Unknown model for this stage')
    base = None
    if args.stage == 'finetune':
        base = torch.load(args.out/'history/model.pt', weights_only=True, map_location='cuda')
    for name in args.only or names:
        out = args.out/name
        out.mkdir(exist_ok=True)
        result = fit(name, args.stage, bank, frames, out, args.steps or (2400 if base is None else 1500),
                     args.seed, base)
        print(name, {key: result['test'][key]['terminal'] for key in result['test']}, flush=True)
    completed = {name: json.loads((args.out/name/'metrics.json').read_text()) for name in (*ARCHITECTURES, *FINETUNES)
                 if (args.out/name/'metrics.json').is_file()}
    atomic_json(args.out/'summary.json', dict(times=bank.times[frames].cpu().tolist(), nodes=1024,
                splits={key: [bank.ids[i] for i in rows.tolist()] for key, rows in bank.splits.items()},
                models=completed))


if __name__ == '__main__':
    main()
