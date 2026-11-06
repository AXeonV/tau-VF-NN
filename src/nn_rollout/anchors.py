"""Independent numerical checkpoints and matched closed-loop rollout comparisons."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample
import torch

from .experiment import evaluation_grid, forecast, predict_block, previous_index, summarize
from .model import WindowFNO
from .refine import BlockRefiner
from .sensitivity import distances
from ..nn_world.train import complex_curves, frame_errors
from ..traditional.evolution import evolve
from ..trajectory_data import Trajectories
from ..trajectory_metrics import atomic_json, has_crossing


CHECKPOINTS = (20., 40., 60., 80.)


def scaled(curves, times):
    z = curves/np.sqrt(1+2*times[:, None])
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def prepare(bank, out):
    saved = np.asarray(bank.config['times'][:len(bank.times)])
    saved = saved[saved <= 80.]
    grid = np.unique(np.r_[saved, np.arange(1., 80.)])
    save_index = np.searchsorted(grid, saved)
    points = np.searchsorted(grid, CHECKPOINTS)
    restarts, trajectories = {}, {}
    cache = {}
    for split in ('validation', 'test'):
        rows = bank.splits[split].tolist()
        states, stats = [], []
        for row in rows:
            started = perf_counter()
            initial = complex_curves(bank.curves[row, 0].cpu().numpy())
            target = complex_curves(bank.curves[row, :len(saved)].cpu().numpy())
            for nodes in (1024, 1536):
                steps = np.zeros(len(grid), dtype=int)

                def progress(index, time, count):
                    steps[index] = count

                physical, info = evolve(resample(initial, nodes), grid, sigma=bank.config['sigma'],
                                        cfl=.16, step_fraction=.006, output_nodes=1024,
                                        pressure_cache=cache, progress=progress)
                normalized = scaled(physical, grid)
                deviation = distances(complex_curves(normalized[save_index]), target)
                if deviation.max() < .0035:
                    break
            if deviation.max() >= .0035:
                raise RuntimeError(f'Numerical checkpoint check failed for case {bank.ids[row]}')
            details = dict(case=bank.ids[row], nodes=nodes, steps=steps[points].tolist(),
                           elapsed=perf_counter()-started, saved_max_error=float(deviation.max()), **{
                               'flux_error': info['max_flux_relative_error']})
            states.append(normalized[save_index])
            stats.append(details)
            if split == 'test' and not restarts:
                start_index = int(np.searchsorted(grid, 40.))
                restarted, _ = evolve(resample(physical[start_index], nodes), grid[start_index:],
                                      sigma=bank.config['sigma'], cfl=.16, step_fraction=.006,
                                      output_nodes=1024, pressure_cache=cache)
                restart_error = distances(complex_curves(scaled(restarted, grid[start_index:])),
                                          complex_curves(normalized[start_index:]))
                restarts = dict(case=bank.ids[row], time=40., max_error=float(restart_error.max()),
                                terminal_error=float(restart_error[-1]))
            print(split, details, flush=True)
        trajectories[split] = np.stack(states)
        trajectories[f'{split}_stats'] = np.array(json.dumps(stats))
    np.savez_compressed(out/'physical_anchors.npz', times=saved, checkpoints=np.asarray(CHECKPOINTS),
                        restart=np.array(json.dumps(restarts)), **trajectories)


def load_candidate(folder, name):
    source = 'history' if name in ('block_repair', 'block_flow') else name
    check = torch.load(folder/source/'model.pt', weights_only=True, map_location='cuda')
    model = WindowFNO(**check['config']).cuda().eval()
    model.load_state_dict(check['model'])
    if name == source:
        return model, None
    check = torch.load(folder/name/'model.pt', weights_only=True, map_location='cuda')
    repair = BlockRefiner().cuda().eval()
    repair.load_state_dict(check['model'])

    def predict(initial, state, proposal, start, queries):
        return repair.predict(initial, state, proposal, start, queries, steps=4 if check['flow'] else 0)

    return model, predict


def geometry_flags(prediction):
    return np.array([[has_crossing(z) for z in path] for path in complex_curves(prediction)])


def stop_summary(times, disagreement, crossing, error, threshold):
    bad = (error > .10) | crossing
    alarms = crossing | (disagreement > threshold)
    n = len(times)
    stop = np.where(alarms.any(axis=1), alarms.argmax(axis=1), n)
    first_bad = np.where(bad.any(axis=1), bad.argmax(axis=1), n)
    missed = ((np.arange(n) < stop[:, None]) & bad).any(axis=1)
    warned = (stop <= first_bad) & (first_bad < n)
    return dict(retained_time=float(times[(stop-1).clip(0)].mean()), missed_cases=int(missed.sum()),
                cases_with_breach=int((first_bad < n).sum()), on_time=int(warned.sum()),
                leads=(times[first_bad[warned]]-times[stop[warned]]).tolist(),
                stop=[float(times[k]) if k < n else None for k in stop],
                first_bad=[float(times[k]) if k < n else None for k in first_bad])


@torch.no_grad()
def reset_forecast(model, refiner, bank, split, frames, dense, direct, schedule=(), threshold=None):
    rows = bank.splits[split]
    initial = bank.curves[rows, 0]
    times = bank.times
    state = previous = initial
    before = times[0].expand(len(rows))
    dense = torch.as_tensor(dense, device=rows.device)
    direct = torch.as_tensor(direct, device=rows.device)
    output = [initial[:, None]]
    requests = [[] for _ in rows]
    for i, (start, query) in enumerate(zip(frames[::4], frames[1:].split(4))):
        t0 = float(times[start])
        selected = torch.full((len(rows),), t0 in schedule, device=rows.device, dtype=torch.bool)
        if threshold is not None and t0 in CHECKPOINTS:
            gap = frame_errors(state[:, None].cpu().numpy(), direct[:, 4*i:4*i+1].cpu().numpy())[:, 0]
            crossings = geometry_flags(state[:, None].cpu().numpy())[:, 0]
            selected = torch.as_tensor((gap > threshold) | crossings, device=rows.device)
        if selected.any():
            index = previous_index(times, start)
            state = torch.where(selected[:, None, None], dense[:, start], state)
            previous = torch.where(selected[:, None, None], dense[:, index], previous)
            before = torch.where(selected, times[index], before)
            output[-1][:, -1] = state
            for j in torch.nonzero(selected).flatten().tolist():
                requests[j].append(t0)
        predicted = predict_block(model, initial, state, previous, before, times[start].expand(len(rows)),
                                  times[query].expand(len(rows), -1), refiner)
        output.append(predicted)
        previous, state = predicted[:, -2], predicted[:, -1]
        before = times[query[-2]].expand(len(rows))
    return torch.cat(output, dim=1).cpu().numpy(), requests


def costs(requests, statistics):
    return [stats['steps'][CHECKPOINTS.index(times[-1])] if times else 0
            for times, stats in zip(requests, statistics)]


def continuation_cost(bank, out):
    measurements = np.load(out/'physical_anchors.npz')
    cache = {}
    times = np.arange(80., 101.)
    result = {}
    for split in ('validation', 'test'):
        rows = bank.splits[split]
        prefix = json.loads(str(measurements[f'{split}_stats']))
        target = complex_curves(bank.curves[rows, -1].cpu().numpy())
        source = complex_curves(measurements[split][:, -1])*np.sqrt(161.)
        result[split] = []
        for z, expected, stats in zip(source, target, prefix):
            path, info = evolve(resample(z, stats['nodes']), times, sigma=bank.config['sigma'],
                                cfl=.16, step_fraction=.006, output_nodes=1024, pressure_cache=cache)
            final = path[-1]/np.sqrt(201.)
            terminal_error = distances([final], [expected])[0]
            total = stats['steps'][-1]+info['steps']
            result[split].append(dict(case=stats['case'], prefix_steps=stats['steps'][-1],
                                      suffix_steps=info['steps'], total_steps=total,
                                      terminal_error=float(terminal_error)))
    atomic_json(out/'numerical_cost.json', result)


def compare(bank, out, anchor_file=None):
    summary = json.loads((out/'summary.json').read_text())
    if (out/'refinement.json').is_file():
        summary['models'].update(json.loads((out/'refinement.json').read_text()))
    eligible = {n: x for n, x in summary['models'].items() if n != 'direct'}

    def score(item):
        free = item['validation']['free']
        return .5*(free['time_mean']+free['terminal'])

    name = min(eligible, key=lambda key: score(eligible[key]))
    model, refiner = load_candidate(out, name)
    frames = evaluation_grid(bank.times)
    times = bank.times[frames].cpu().numpy()
    measurements = np.load(anchor_file or out/'physical_anchors.npz')
    partitions = {}
    for split in ('validation', 'test'):
        rows = bank.splits[split]
        target = bank.curves[rows[:, None], frames].cpu().numpy()
        direct = forecast(model, bank, rows, frames, 'direct', refiner=refiner)
        free = forecast(model, bank, rows, frames, 'free', refiner=refiner)
        partitions[split] = dict(ids=[bank.ids[i] for i in rows.tolist()], target=target, direct=direct,
                                 free=free, crossings=geometry_flags(free), error=frame_errors(free, target),
                                 disagreement=frame_errors(free, direct), dense=measurements[split],
                                 costs=json.loads(str(measurements[f'{split}_stats'])))
    val = partitions['validation']
    candidates = np.unique(np.r_[0, np.quantile(val['disagreement'], np.linspace(0, 1, 81)), np.inf])

    def risk_choice(threshold):
        result = stop_summary(times, val['disagreement'], val['crossings'], val['error'], threshold)
        return result['missed_cases'], -result['retained_time']

    threshold = min(candidates, key=risk_choice)
    block_probes = np.arange(4, len(frames)-1, 4)[1:]
    reset_candidates = np.unique(np.r_[0, np.quantile(val['disagreement'][:, block_probes], [.2, .4, .6, .8]), np.inf])
    val_choices = []
    for gate in reset_candidates:
        trial, requests = reset_forecast(model, refiner, bank, 'validation', frames, val['dense'],
                                         val['direct'], threshold=gate)
        error = frame_errors(trial[:, -1:], val['target'][:, -1:]).mean()
        work = np.mean(costs(requests, val['costs']))/np.mean([c['steps'][-1] for c in val['costs']])
        val_choices.append((float(error+.03*work), float(error), float(work)))
    reset_gate = reset_candidates[min(range(len(val_choices)), key=val_choices.__getitem__)]
    result = dict(model=name, threshold=float(threshold) if np.isfinite(threshold) else None,
                  adaptive_threshold=float(reset_gate) if np.isfinite(reset_gate) else None,
                  restart_check=json.loads(str(measurements['restart'])), partitions={})
    for split, data in partitions.items():
        controls = dict(unchecked=stop_summary(times, data['disagreement'], data['crossings'], data['error'], np.inf),
                        path_agreement=stop_summary(times, data['disagreement'], data['crossings'], data['error'], threshold))
        repairs = {}
        for policy, schedule in [('fixed_40', (40.,)), ('fixed_80', (80.,)),
                                 ('fixed_40_80', (40., 80.)), ('adaptive', ())]:
            output, requests = reset_forecast(model, refiner, bank, split, frames, data['dense'],
                                               data['direct'], schedule,
                                               reset_gate if policy == 'adaptive' else None)
            metrics = summarize(output, data['target'], times, data['ids'])
            metrics.update(checkpoints=requests, numerical_steps=costs(requests, data['costs']))
            repairs[policy] = metrics
        full = torch.arange(len(bank.times), device=bank.times.device)
        synthetic = forecast(model, bank, bank.splits[split], full, 'direct', refiner=refiner)
        output, requests = reset_forecast(model, refiner, bank, split, frames, synthetic,
                                           data['direct'], schedule=(80.,))
        repairs['learned_80'] = summarize(output, data['target'], times, data['ids'])
        repairs['learned_80'].update(checkpoints=requests, numerical_steps=[0]*len(data['ids']))
        result['partitions'][split] = dict(safety=controls, resets=repairs, prefix_costs=data['costs'])
    # Reuse the same learned maps at different feedback cadences on the same output grid.
    pairs, _ = load_candidate(out, 'pairs')
    for candidate, operator, refine in [('pairs', pairs, None), (name, model, refiner)]:
        data = partitions['test']
        by_size = {}
        for width in (1, 2, 4, 8):
            prediction = forecast(operator, bank, bank.splits['test'], frames, 'free',
                                  refiner=refine, window_size=width)
            by_size[str(width)] = summarize(prediction, data['target'], times, data['ids'])
        result.setdefault('feedback_cadence', {})[candidate] = by_size
    atomic_json(out/'anchors.json', result)
    print('closed loop', name, {k: v['terminal'] for k, v in result['partitions']['test']['resets'].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/rollout_t100'))
    parser.add_argument('--stage', choices=['prepare', 'compare', 'cost'], required=True)
    parser.add_argument('--anchors', type=Path, help='Reuse independently generated physical checkpoints')
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(args.data, 1024, 'cuda', 100.)
    if args.stage == 'prepare':
        prepare(bank, args.out)
    elif args.stage == 'cost':
        continuation_cost(bank, args.out)
    else:
        compare(bank, args.out, args.anchors)


if __name__ == '__main__':
    main()
