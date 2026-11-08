"""Compare neural and boundary-integral forecasts from identical late-tip states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from scipy.spatial import cKDTree

from .model import GeometryFNO
from ..interface_events import angle_distance, match_births, tip_births
from ..traditional.evolution import evolve, redistribute
from ..trajectory_metrics import _distance_to_curve, area, atomic_json, curve_difference, has_crossing


DT = .5
OFFSETS = np.arange(11)*DT
DRIFT_STEPS = (4, 8)
NODES = 1024
SECTOR = .24


def radius(times):
    return np.sqrt(1+2*np.asarray(times))


def distance(a, b):
    return curve_difference(a[None], b[None], [0.])['maximum']


def distances(a, b, angle):
    global_errors, local_errors = [], []
    for x, y in zip(a, b):
        n = 2*max(len(x), len(y))
        x, y = resample(x, n), resample(y, n)
        dxy, dyx = _distance_to_curve(x, y), _distance_to_curve(y, x)
        global_errors.append(float(max(dxy.max(), dyx.max())))
        x_mask = angle_distance(np.angle(x), angle) <= SECTOR
        y_mask = angle_distance(np.angle(y), angle) <= SECTOR
        local_errors.append(float(max(dxy[x_mask].max(initial=0), dyx[y_mask].max(initial=0)))
                            if x_mask.any() and y_mask.any() else None)
    return global_errors, local_errors


def normalized(z, times):
    return z/radius(times)[..., None]


def physical_pair(state, previous):
    """Redistribute both shapes by the current arclength map before either forecast."""
    current, old = redistribute(state, previous)
    signed = area(current)
    if signed <= 0:
        raise ValueError('nonpositive initial area')
    return current*np.sqrt(np.pi/signed), old


def solve(state, times, device, cache, tolerance):
    """Certify a physical continuation. The input and outputs use normalized curves."""
    solutions = []
    for nodes, cfl, fraction in ((1024, .16, .005), (1536, .12, .0035)):
        curves, metrics = evolve(resample(state*radius(times[0]), nodes), times,
                                 device=device, cfl=cfl, step_fraction=fraction,
                                 output_nodes=NODES, pressure_cache=cache)
        solutions.append((normalized(curves, times), metrics))
    (low, low_stats), (high, high_stats) = solutions
    errors = [distance(a, b) for a, b in zip(low, high)]
    mismatches = [distance(state, solution[0]) for solution in (low, high)]
    crossings = [int(sum(has_crossing(z) for z in solution)) for solution in (low, high)]
    flux = [v['max_flux_relative_error'] for v in (low_stats, high_stats)]
    record = dict(refinement_errors=errors, max_refinement=max(errors),
                  input_to_solver_t0=mismatches, flux_error=flux, crossings=crossings,
                  steps=[low_stats['steps'], high_stats['steps']], tolerance=tolerance)
    reasons = []
    if max(errors) > tolerance:
        reasons.append('refinement')
    if max(mismatches) > .1*tolerance:
        reasons.append('input_to_solver_t0')
    if any(crossings):
        reasons.append('solver_crossing')
    if max(flux) > 5e-5:
        reasons.append('flux')
    record['reasons'] = reasons
    return high, record


@torch.inference_mode()
def network_step(model, initial, previous, state, old_time, time, future):
    def tensor(z):
        array = np.stack([z.real, z.imag], axis=0)
        return torch.as_tensor(array[None], dtype=torch.float32, device=next(model.parameters()).device)

    first, current, old = (tensor(z) for z in (initial, state, previous))
    output = model(first, current, old, first.new_tensor([old_time]), first.new_tensor([time]),
                   first.new_tensor([[future]]))[0, 0].cpu().numpy()
    estimate = (output[0]+1j*output[1]).astype(np.complex128)
    if not np.isfinite(estimate).all():
        raise RuntimeError('nonfinite network prediction')
    return estimate


def network_paths(model, initial, previous, state, physical, times):
    free, teacher = [state], [state]
    for j, t in enumerate(times[1:]):
        old = previous if j == 0 else free[j-1]
        old_time = times[j]-DT
        free.append(network_step(model, initial, old, free[j], old_time, times[j], t))
        if j == 0:
            teacher.append(free[-1])
        else:
            old_physical = state if j == 1 else physical[j-1]
            teacher.append(network_step(model, initial, old_physical, physical[j], old_time, times[j], t))
    direct = network_step(model, initial, previous, state, times[0]-DT, times[0], times[-1])
    return np.stack(free), np.stack(teacher), direct


def normal_rms(predict, target, current, angle):
    tangent = np.roll(current, -4)-np.roll(current, 4)
    unit_normal = -1j*tangent/np.maximum(abs(tangent), 1e-12)
    projected = ((predict-target)*unit_normal.conj()).real/DT
    mask = angle_distance(np.angle(current), angle) <= SECTOR
    return float(np.sqrt(np.mean(projected[mask]**2))) if mask.any() else None


def event_matches(path, times, event):
    found = tip_births(path, times)
    return dict(found=found, matched=match_births(found, [event], time_limit=1.5, angle_limit=SECTOR))


def certify_window(times, curves, coarse_event, device, cache):
    center, direction = coarse_event
    fine_times = np.arange(center-15, center+5+DT, DT)
    first = int(np.searchsorted(times, fine_times[0]))
    coarse = np.flatnonzero((times >= fine_times[0]) & (times <= fine_times[-1]))
    dense = np.searchsorted(fine_times, times[coarse])
    aligned = dense < len(fine_times)
    coarse, dense = coarse[aligned], dense[aligned]
    aligned = np.isclose(times[coarse], fine_times[dense], rtol=0, atol=1e-9)
    coarse, dense = coarse[aligned], dense[aligned]
    fine, audit = solve(normalized(curves[first], fine_times[0]), fine_times, device, cache, .001)
    discrepancies = [distance(a, b) for a, b in zip(fine[dense], normalized(curves[coarse], times[coarse]))]
    audit['saved_label_max'] = max(discrepancies)
    if audit['reasons'] or max(discrepancies) > .0035:
        return None, audit
    candidates = [event for event in tip_births(fine, fine_times)
                  if match_births([event], [coarse_event], time_limit=5)]
    candidates = [event for event in candidates if np.searchsorted(fine_times, event[0])-2-max(DRIFT_STEPS) >= 1]
    if not candidates:
        audit['reasons'].append('no_matching_fine_event')
        return None, audit
    event = min(candidates, key=lambda e: abs(e[0]-center)/5+angle_distance(e[1], direction)/SECTOR)
    anchor = int(np.searchsorted(fine_times, event[0]))-2
    return (fine_times, fine, event, anchor), audit


def drifts(model, initial, fine_times, fine, anchor):
    for length in DRIFT_STEPS:
        start = anchor-length
        previous, state = fine[start-1:start+1]
        for j in range(start, anchor):
            predicted = network_step(model, initial, previous, state, *fine_times[j-1:j+2])
            previous, state = state, predicted
        yield length, previous, state


def pair(model, initial, raw_old, raw, times, clean, clean_paths, clean_audit, event, device, cache):
    if has_crossing(raw):
        return None, dict(reason='drift_input_crossing')
    raw_error = distance(raw, clean[0])
    if not .003 <= raw_error <= .05:
        return None, dict(reason='outside_mild_drift_range', raw_initial_error=raw_error)
    current, previous = physical_pair(raw, raw_old)
    initial_error = distance(current, clean[0])
    raw_to_shared = distance(raw, current)
    audit = dict(raw_initial_error=raw_error, raw_to_shared=raw_to_shared,
                 raw_area_error=float(abs(area(raw)/np.pi-1)), initial_error=initial_error)
    if raw_to_shared > .1*initial_error:
        return None, dict(audit, reason='preprocessing_comparable_to_drift')
    tolerance = min(.001, .2*initial_error)
    phys, audit['solver'] = solve(current, times, device, cache, tolerance)
    audit['initial_error'] = distance(phys[0], clean[0])
    if audit['solver']['reasons'] or clean_audit['max_refinement'] > tolerance:
        return None, dict(audit, reason='uncertified_paired_continuation')
    free, teacher, direct = network_paths(model, initial, previous, current, phys, times)
    free_clean, teacher_clean, direct_clean = clean_paths
    comparisons = dict(physical=(phys, clean), free_same=(free, phys), free_nominal=(free, clean),
                       teacher_same=(teacher, phys), clean_free=(free_clean, clean),
                       clean_teacher=(teacher_clean, clean))
    both = {key: distances(a, b, event[1]) for key, (a, b) in comparisons.items()}
    audit['errors'] = {key: values[0] for key, values in both.items()}
    audit['local_errors'] = {key: values[1] for key, values in both.items()}
    motion = {key: distances(path[:-1], path[1:], event[1])
              for key, path in (('drift', phys), ('clean', clean))}
    audit['teacher_motion'] = {key: [0.]+values[0] for key, values in motion.items()}
    audit['local_teacher_motion'] = {key: [0.]+values[1] for key, values in motion.items()}
    audit['direct5'] = dict(same=distance(direct, phys[-1]), nominal=distance(direct, clean[-1]),
                            clean=distance(direct_clean, clean[-1]),
                            drift_crossing=has_crossing(direct), clean_crossing=has_crossing(direct_clean))
    audit['normal_velocity_rms'] = dict(
        drift=[normal_rms(teacher[j], phys[j], phys[j-1], event[1]) for j in range(1, len(times))],
        clean=[normal_rms(teacher_clean[j], clean[j], clean[j-1], event[1]) for j in range(1, len(times))])
    states = dict(physical_drift=phys, physical_clean=clean, free=free, teacher=teacher,
                  free_clean=free_clean, teacher_clean=teacher_clean)
    audit['events'] = {key: event_matches(z, times, event) for key, z in states.items()}
    audit['events']['drift_network_vs_drift_physical'] = match_births(
        audit['events']['free']['found'], audit['events']['physical_drift']['found'],
        time_limit=1.5, angle_limit=SECTOR)
    audit['crossings'] = {key: [float(t) for t, zt in zip(times, z) if has_crossing(zt)]
                          for key, z in states.items()}
    return states, audit


def inspect_case(path, model, out, device, cache, report, max_windows):
    with np.load(path) as item:
        times, curves = item['times'], item['z']
        metadata = json.loads(str(item['metadata']))
    info = metadata['case']
    index = info['index']
    if index < 128 or info['split'] != 'test' or not metadata['accepted']:
        raise ValueError(f'Expected a fresh, certified, test-only shape: {path}')
    initial = normalized(resample(curves[0], NODES), times[0])
    candidates = [event for event in tip_births(normalized(curves, times), times)
                  if 50 <= event[0] <= 95 and all(event[0]+s in times for s in (-15, -10, -5, 0, 5))]
    used = set()
    completed = []
    if not candidates:
        report['excluded'].append(dict(case=index, reason='no_late_coarse_event'))
        return
    for target in (60, 80)[:max_windows]:
        for event in sorted(candidates, key=lambda e: (abs(e[0]-target), e[0], e[1])):
            center = event[0]
            if center in used or any(abs(center-c) < 10 for c in completed):
                continue
            used.add(center)
            name = f'case_{index:03}_t{int(center):03}'
            context = dict(case=index, mode=info['dominant'], window=name, coarse_event=event)
            try:
                window, validation = certify_window(times, curves, event, device, cache)
                if window is None:
                    report['excluded'].append(dict(context, reason='window_certification', audit=validation))
                    continue
                fine_times, fine, fine_event, anchor = window
                t = fine_times[anchor]+OFFSETS
                current, previous = physical_pair(fine[anchor], fine[anchor-1])
                clean, clean_audit = solve(current, t, device, cache, .001)
                clean_audit['raw_to_shared'] = distance(fine[anchor], current)
                clean_audit['event_found'] = event_matches(clean, t, fine_event)
                if clean_audit['reasons'] or not clean_audit['event_found']['matched']:
                    report['excluded'].append(dict(context, reason='clean_restart_certification',
                                                   audit=clean_audit))
                    continue
                clean_paths = network_paths(model, initial, previous, current, clean, t)
            except (RuntimeError, ValueError) as error:
                report['excluded'].append(dict(context, reason='window_exception', detail=str(error)))
                continue
            context.update(event_time=fine_event[0], event_angle=fine_event[1], times=t.tolist())
            report['windows'].append(dict(context, fine_certification=validation, clean_solver=clean_audit))
            completed.append(center)
            try:
                histories = list(drifts(model, initial, fine_times, fine, anchor))
            except (RuntimeError, ValueError) as error:
                report['excluded'].append(dict(context, reason='drift_generation_exception', detail=str(error)))
                continue
            for length, old, state in histories:
                try:
                    states, record = pair(model, initial, old, state, t, clean, clean_paths, clean_audit,
                                          fine_event, device, cache)
                except (RuntimeError, ValueError) as error:
                    states, record = None, dict(reason='drift_exception', detail=str(error))
                if states is None:
                    report['excluded'].append(dict(context, drift_steps=length, **record))
                    continue
                filename = f'{name}_drift{length}.npz'
                np.savez_compressed(out/filename, times=t, raw_start=state, raw_previous=old, **states)
                report['records'].append(dict(context, drift_steps=length, filename=filename, **record))
                print(name, f'drift={length}', f'initial={record["initial_error"]:.4g}',
                      f'physical={record["errors"]["physical"][-1]:.4g}',
                      f'free_same={record["errors"]["free_same"][-1]:.4g}', flush=True)
            atomic_json(out/'report.json', report)
            break


def closest_segment(source, curve):
    curve = resample(curve, 2*len(curve))
    _, near = cKDTree(np.column_stack([curve.real, curve.imag])).query(
        np.column_stack([source.real, source.imag]))
    start = curve[near, None]
    edge = curve[(near[:, None]+[-1, 1]) % len(curve)]-start
    offset = source[:, None]-start
    fraction = np.clip((offset*edge.conj()).real/np.maximum(abs(edge)**2, 1e-30), 0., 1.)
    foot = start+fraction*edge
    index = np.argmin(abs(source[:, None]-foot), axis=1)
    return foot[np.arange(len(source)), index]


def normal_coherence(physical, free, teacher, angle, segment):
    """Project signed temporal errors at the current solver normals."""
    current, target = physical[:-1], physical[1:]
    tangent = np.roll(current, -4, axis=-1)-np.roll(current, 4, axis=-1)
    normal = -1j*tangent/np.maximum(abs(tangent), 1e-12)

    def position(z):
        return np.stack([closest_segment(src, dst) for src, dst in zip(current, z)]) if segment else z

    reference = position(target)
    predicted = position(teacher[1:])
    residual = ((predicted-reference)*normal.conj()).real
    final = closest_segment(current[-1], free[-1]) if segment else free[-1]
    terminal = ((final-reference[-1])*normal[-1].conj()).real
    physical_motion = ((reference-current)*normal.conj()).real
    neural_motion = ((predicted-current)*normal.conj()).real
    local = angle_distance(np.angle(physical[0]), angle) <= SECTOR
    result = {}
    for name, mask in (('global', np.ones(len(local), dtype=bool)), ('local', local)):
        e, v = residual[:, mask], terminal[mask]
        u = np.sum(e, axis=0)
        norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
        sum_norms = np.linalg.norm(e, axis=1).sum()
        motion, forecast_motion = physical_motion[:, mask], neural_motion[:, mask]
        motion_sq = np.sum(motion**2)
        result[name] = dict(
            coherence=float(norm_u/sum_norms) if sum_norms else None,
            final_cosine=float(np.dot(u, v)/(norm_u*norm_v)) if norm_u and norm_v else None,
            cumulative_to_final=float(norm_u/norm_v) if norm_v else None,
            motion_gain=float(np.sum(motion*forecast_motion)/motion_sq) if motion_sq else None)
    return result


def bias_report(sources, out):
    """Reuse saved trajectories without running the solver or network again."""
    records, seen_clean = [], set()
    for folder in sources:
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        for sample in report['records']:
            context = {key: sample[key] for key in ('case', 'window', 'drift_steps', 'event_angle')}
            with np.load(folder/sample['filename']) as data:
                for arm in ('drift', 'clean'):
                    if arm == 'clean':
                        key = (folder, sample['window'])
                        if key in seen_clean:
                            continue
                        seen_clean.add(key)
                    suffix = '_clean' if arm == 'clean' else ''
                    args = (data[f'physical_{arm}'], data[f'free{suffix}'], data[f'teacher{suffix}'],
                            sample['event_angle'])
                    records.append(dict(context, folder=str(folder), arm=arm,
                                        indexed=normal_coherence(*args, segment=False),
                                        closest_segment=normal_coherence(*args, segment=True)))
    summary = {}
    for arm in ('clean', 'drift'):
        summary[arm] = {}
        for method in ('indexed', 'closest_segment'):
            summary[arm][method] = {}
            for sector in ('global', 'local'):
                subset = [r[method][sector] for r in records if r['arm'] == arm]
                if not subset:
                    continue
                values = {}
                for key in subset[0]:
                    finite = [r[key] for r in subset if r[key] is not None]
                    values[key] = dict(mean=float(np.mean(finite)), median=float(np.median(finite)),
                                       count=len(finite)) if finite else None
                summary[arm][method][sector] = values
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'normal_bias.json', dict(sources=[str(path) for path in sources], records=records,
                summary=summary,
                limitation=('Time aggregation follows the solver arclength index. '
                            'Closest-segment projection removes cross-curve point correspondence, '
                            'but evolving normals and the temporal arclength gauge still affect the proxy.')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bias-from', type=Path, nargs='+', help='Summarize signed normal errors from saved runs')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-windows', type=int, choices=[1, 2], default=2)
    parser.add_argument('--case', type=int, nargs='+')
    args = parser.parse_args()
    if args.bias_from:
        bias_report(args.bias_from, args.out)
        return
    if args.data is None or args.model is None:
        parser.error('--data and --model are required for a new diagnostic run')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    model = GeometryFNO(**checkpoint['config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(model=str(args.model), data=str(args.data), time_offsets=OFFSETS.tolist(),
                  angle_sector=SECTOR, records=[], windows=[], excluded=[])
    cache = {}
    for path in sorted(args.data.glob('case_*.npz')):
        if args.case is None or int(path.stem[5:]) in args.case:
            inspect_case(path, model, args.out, args.device, cache, report, args.max_windows)
    atomic_json(args.out/'report.json', report)
    print(f'{len(report["records"])} certified pairs, {len(report["excluded"])} exclusions', flush=True)


if __name__ == '__main__':
    main()
