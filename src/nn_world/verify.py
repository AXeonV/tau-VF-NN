"""Validate early stopping, repeated repair, and numerical fallback on held-out trajectories."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.integrate import cumulative_trapezoid
from scipy.signal import resample
from scipy.spatial.distance import cdist
import torch

from .audit import normalized
from .model import Repair
from .train import STRUCTURE, complex_curves, frame_errors
from ..nn.model import CurveFNO
from ..traditional.evolution import evolve, redistribute
from ..traditional.pressure import CudaBoundaryPressure, derivative
from ..trajectory_data import Trajectories
from ..trajectory_metrics import atomic_json, has_crossing


@torch.no_grad()
def predictions(bank, rows, times, base, folder, name, summary):
    device = bank.curves.device
    r = rows.repeat_interleave(len(times))
    t = torch.as_tensor(times, device=device, dtype=torch.float32).repeat(len(rows))
    base_prediction = torch.cat([base(bank.curves[i, 0], time) for i, time in zip(r.split(16), t.split(16))])
    base_prediction = base_prediction.reshape(len(rows), len(times), 2, -1)
    if name == 'nn':
        return base_prediction.cpu().numpy(), base_prediction.cpu().numpy()
    result = summary['results'][name]
    source = result.get('source', name)
    checkpoint = torch.load(folder/source/'model.pt', map_location=device, weights_only=True)
    local, near, _, _ = STRUCTURE[checkpoint['structure']]
    model = Repair(local=local, neighbors=near).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    outputs = []
    for i, time, curve in zip(r.split(16), t.split(16), base_prediction.flatten(0, 1).split(16)):
        for _ in range(result.get('iterations', 1)):
            curve = model(bank.curves[i, 0], curve, time)
        outputs.append(curve.cpu().numpy())
    return np.concatenate(outputs).reshape(base_prediction.shape), base_prediction.cpu().numpy()


def crossings(z):
    return np.array([[has_crossing(curve) for curve in path] for path in z])


def gap_scores(z):
    count = 256
    arc_offset = np.abs(np.arange(count)[:, None]-np.arange(count))
    near = np.minimum(arc_offset, count-arc_offset) <= count//16
    score = np.empty(z.shape[:2])
    for idx in np.ndindex(score.shape):
        xy = np.stack((z[idx].real, z[idx].imag), axis=-1)[::z.shape[-1]//count]
        distance = cdist(xy, xy)
        distance[near] = np.inf
        score[idx] = 1/max(distance.min(), .001)
    return score


def physics_scores(z, times, invalid, sigma):
    """R times the arclength-weighted RMS normal-velocity defect on filtered curves."""
    started = perf_counter()
    nodes = 384
    solver = CudaBoundaryPressure(nodes, sigma)
    radius = np.sqrt(1+2*times)
    k = abs(fftfreq(z.shape[-1], d=1/z.shape[-1]))
    filtered = ifft(fft(z, axis=-1)*np.exp(-(k/48)**8), axis=-1)
    physical = resample(filtered*radius[None, :, None], nodes, axis=-1)
    rate = np.gradient(physical, times, axis=1, edge_order=2)
    score = np.zeros(z.shape[:2])
    failures = 0
    for case, path in enumerate(physical):
        solver.reset()
        for j in range(1, len(times)):
            if invalid[case, j] or has_crossing(path[j]):
                score[case, j] = np.inf
                continue
            try:
                curve, velocity = redistribute(path[j], rate[case, j])
                _, vn, _, metric = solver.velocity(curve, True)
                outward = -1j*derivative(curve)/metric
                model_vn = (velocity*outward.conj()).real
                value = radius[j]*np.sqrt(np.average((model_vn-vn)**2, weights=metric))
                score[case, j] = value if np.isfinite(value) else np.inf
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                failures += 1
                score[case, j] = np.inf
    return score, dict(seconds=perf_counter()-started, failures=failures)


def stop_at(times, score, hard, threshold, cadence='every', cheap=None, cheap_limit=None):
    stop = np.full(len(hard), len(times))
    calls = np.zeros(len(hard), dtype=int)
    for case in range(len(hard)):
        next_check = .25
        for j, time in enumerate(times[1:], 1):
            if hard[case, j]:
                stop[case] = j
                break
            near_limit = cadence == 'adaptive' and time >= .25 and cheap[case, j] >= cheap_limit
            due = cadence == 'every' or time >= next_check-1e-9 or near_limit or j == len(times)-1
            if due:
                value = score[case, j]
                calls[case] += 1
                if value >= threshold:
                    stop[case] = j
                    break
                tight = cadence == 'adaptive' and (near_limit or value >= .65*threshold)
                next_check = time+(.25 if tight else 2.)
    return stop, calls


def outcomes(times, bad, stop, calls):
    dt = np.diff(times, prepend=0.)
    final = len(times)
    first = np.where(bad.any(axis=1), bad.argmax(axis=1), final)
    prefix = np.arange(final)[None, :] < stop[:, None]
    leaks = (prefix & bad)
    positive = first < final
    warned = positive & (stop <= first)
    leads = times[first[warned]]-times[stop[warned]]
    safe_time = (prefix & (np.arange(final)[None, :] < first[:, None]))@dt
    return dict(mean_retained_time=float(np.mean(times[stop-1])),
                mean_safe_prefix_time=float(safe_time.mean()),
                cases_with_violation=int(positive.sum()), early_or_on_time=int(warned.sum()),
                strictly_early=int(np.count_nonzero(leads > 0)),
                lead_times=leads.tolist(), unsafe_accepted_cases=int(leaks.any(axis=1).sum()),
                unsafe_accepted_time=float((leaks@dt).sum()),
                stopped_without_violation=int((~positive & (stop < final)).sum()),
                completed=int((stop == final).sum()), mean_score_calls=float(calls.mean()),
                stop=[None if i == final else float(times[i]) for i in stop],
                first_bad=[None if i == final else float(times[i]) for i in first])


def calibrate(times, split_data, signal, cadence, cheap_limit):
    val = split_data['validation']
    scores = val[signal][np.isfinite(val[signal])]
    thresholds = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 151)), np.inf])
    choices = []
    for threshold in thresholds:
        stop, calls = stop_at(times, val[signal], val['hard'], threshold, cadence,
                              val['cheap'], cheap_limit)
        metrics = outcomes(times, val['bad'], stop, calls)
        choices.append((metrics['unsafe_accepted_cases'], metrics['unsafe_accepted_time'],
                        -metrics['mean_retained_time'], metrics['mean_score_calls'], -threshold))
    index = min(range(len(choices)), key=choices.__getitem__)
    threshold = thresholds[index]
    result = dict(threshold=float(threshold) if np.isfinite(threshold) else None)
    for name, data in split_data.items():
        stop, calls = stop_at(times, data[signal], data['hard'], threshold, cadence,
                              data['cheap'], cheap_limit)
        result[name] = outcomes(times, data['bad'], stop, calls)
    return result


def fallback(bank, data, policy, times, tolerance):
    rows = bank.splits['test']
    labels = bank.curves[rows, -1].cpu().numpy()
    repaired = data['prediction'][:, -1].copy()
    cache = {}
    records = []
    for index, alarm in enumerate(policy['stop']):
        if alarm is None:
            continue
        start = perf_counter()
        row = int(rows[index])
        z0 = complex_curves(bank.curves[row, 0].cpu().numpy())
        z, info = evolve(z0, times, sigma=bank.config['sigma'], cfl=.15, step_fraction=.006,
                         output_nodes=1024, pressure_cache=cache)
        forecast = normalized(z, times)
        repaired[index] = forecast[-1]
        alarm_index = np.searchsorted(times, alarm)
        discrepancy = frame_errors(data['prediction'][index:index+1, alarm_index:alarm_index+1],
                                   forecast[None, alarm_index:alarm_index+1])[0, 0]
        records.append(dict(case=bank.ids[row], alarm=alarm, seconds=perf_counter()-start,
                            forecast_to_fallback_at_alarm=float(discrepancy), nodes=len(z0), steps=info['steps']))
        print('fallback', records[-1], flush=True)
    before = frame_errors(data['prediction'][:, -1:], labels[:, None])[:, 0]
    after = frame_errors(repaired[:, None], labels[:, None])[:, 0]
    return dict(cases=records, end_before=float(before.mean()), end_after=float(after.mean()),
                final_outside_tolerance=int((after > tolerance).sum()), seconds=sum(x['seconds'] for x in records),
                endpoint_errors=after.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--baseline', type=Path, default=Path('results/supervised/comparison/t20/nn/model.pt'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/world_t20'))
    parser.add_argument('--audit', type=Path, help='Reuse the same time-dense audit snapshots across seeds')
    parser.add_argument('--tolerance', type=float, default=.10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = args.out
    summary = json.loads((folder/'summary.json').read_text())
    checkpoint = torch.load(args.baseline, weights_only=True, map_location='cuda')
    if checkpoint['config']['horizon'] != 20 or checkpoint['nodes'] != 1024:
        raise ValueError('Expected the T=20 baseline with 1024 interface nodes')
    bank = Trajectories(args.data, checkpoint['nodes'], 'cuda', 20.)
    base = CurveFNO(**checkpoint['config']).cuda().eval()
    base.load_state_dict(checkpoint['model'])
    audit = np.load(args.audit or folder/'audit.npz')
    times = audit['times']
    selected = summary['selected']
    # Compare the chosen forecast to a different correction depth at the same time.
    other = f'{summary["repeated_source"]}_x3'
    if selected in (other, 'unroll'):
        other = summary['repeated_source']
    split_data, solve_cost, local_retry = {}, {}, {}
    for split in ('validation', 'test'):
        rows = bank.splits[split]
        forecast, coarse = predictions(bank, rows, times, base, folder, selected, summary)
        alternative, _ = predictions(bank, rows, times, base, folder, other, summary)
        z, z_alt = complex_curves(forecast), complex_curves(alternative)
        hard, other_hard = crossings(z), crossings(z_alt)
        error = frame_errors(forecast, audit[split])
        alt_error = frame_errors(alternative, audit[split])
        gap = gap_scores(z)
        disagreement = np.sqrt(np.mean(abs(z-complex_curves(coarse if selected != 'nn' else alternative))**2, axis=-1))
        physics, cost = physics_scores(z, times, hard, bank.config['sigma'])
        other_physics, other_cost = physics_scores(z_alt, times, other_hard, bank.config['sigma'])
        accumulated = cumulative_trapezoid(physics/(1+2*times), times, axis=1, initial=0.)
        solve_cost[split] = dict(selected=cost, alternative=other_cost)
        bad = (error > args.tolerance) | hard
        split_data[split] = dict(prediction=forecast, error=error, hard=hard, bad=bad, gap=gap,
                                 disagreement=disagreement, physics=physics, accumulated=accumulated,
                                 alternative_error=alt_error,
                                 alternative_hard=other_hard, alternative_physics=other_physics)
        print(split, 'physics', solve_cost[split], flush=True)
    val = split_data['validation']
    scale = {key: np.quantile(val[key][np.isfinite(val[key])], .9).clip(1e-5)
             for key in ('physics', 'gap', 'disagreement', 'accumulated')}
    for data in split_data.values():
        data['cheap'] = np.maximum(data['gap']/scale['gap'], data['disagreement']/scale['disagreement'])
        data['combined'] = np.maximum(data['cheap'], data['physics']/scale['physics'])
        data['accumulated_combined'] = np.maximum(data['cheap'], data['accumulated']/scale['accumulated'])
    cheap_limit = np.quantile(val['cheap'][:, times >= .25], .75)
    rules = dict()
    no_calls = np.zeros(len(val['hard']), dtype=int)
    for name in ('none', 'geometry_only', 'oracle'):
        result = {}
        for split, data in split_data.items():
            trigger = (data['bad'] if name == 'oracle' else data['hard']) if name != 'none' else np.zeros_like(data['hard'])
            stop = np.where(trigger.any(axis=1), trigger.argmax(axis=1), len(times))
            result[split] = outcomes(times, data['bad'], stop, no_calls)
        rules[name] = result
    for signal, cadence in [('gap', 'every'), ('disagreement', 'every'), ('physics', 'every'),
                            ('accumulated', 'every'), ('combined', 'every'), ('combined', 'fixed'),
                            ('combined', 'adaptive'), ('accumulated_combined', 'adaptive')]:
        rules[f'{signal}_{cadence}'] = calibrate(times, split_data, signal, cadence, cheap_limit)
    candidates = [name for name in rules if name not in ('none', 'oracle')]
    chosen = min(candidates, key=lambda name: (rules[name]['validation']['unsafe_accepted_cases'],
                 rules[name]['validation']['unsafe_accepted_time'], -rules[name]['validation']['mean_retained_time']))
    for split, data in split_data.items():
        # Once a predicted frame is rejected, try the alternate depth if its defect is materially smaller.
        alarms = rules[chosen][split]['stop']
        pairs = [(i, np.searchsorted(times, t)) for i, t in enumerate(alarms) if t is not None]
        attempted = [(i, j) for i, j in pairs if not data['alternative_hard'][i, j]
                     and np.isfinite(data['alternative_physics'][i, j])
                     and data['alternative_physics'][i, j] < .9*data['physics'][i, j]]
        accepted = [(i, j) for i, j in attempted if data['alternative_error'][i, j] <= args.tolerance]
        local_retry[split] = dict(alarms=len(pairs), defect_reduced=len(attempted),
                                  error_improved=sum(bool(data['alternative_error'][i, j] < data['error'][i, j])
                                                     for i, j in attempted), within_tolerance=len(accepted))
    recomputed = fallback(bank, split_data['test'], rules[chosen]['test'], times, args.tolerance)
    atomic_json(folder/'verification.json', dict(horizon=20, frames=len(times), tolerance=args.tolerance,
                candidate=selected, alternate_depth=other, policies=rules, selected_policy=chosen,
                local_retry=local_retry, fallback=recomputed, physics_cost=solve_cost,
                audit={name: json.loads(str(audit[f'{name}_statistics'])) for name in split_data}))
    print('selected monitor', chosen, rules[chosen]['test'], flush=True)


if __name__ == '__main__':
    main()
