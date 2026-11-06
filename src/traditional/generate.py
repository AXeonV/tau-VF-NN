"""Generate and certify independent time-stepped interface trajectories."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample, find_peaks
from scipy.spatial import cKDTree

from .cases import CASE_COUNT, case_description, initial_curve, output_times
from .evolution import area, evolve


def _points(z):
    return np.column_stack([z.real, z.imag])


def _distance_to_curve(source, curve):
    points = _points(source)
    target = _points(curve)
    _, nearest = cKDTree(target).query(points)
    next_index = (nearest+1) % len(curve)
    prev_index = (nearest-1) % len(curve)
    start = target[nearest]
    sides = target[np.column_stack([prev_index, next_index])]-start[:, None, :]
    difference = points-start
    along = np.einsum('nki,ni->nk', sides, difference)/np.sum(sides*sides, axis=-1).clip(1e-30)
    distances = np.linalg.norm(difference[:, None]-np.clip(along, 0., 1.)[..., None]*sides, axis=-1)
    return distances.min(axis=1)


def curve_difference(first, second, times, *, stride=3):
    count = 2*max(first.shape[-1], second.shape[-1])
    indices = np.unique(np.r_[np.arange(0, len(times), stride), len(times)-1])
    maxima = []
    for index in indices:
        x = resample(first[index], count)
        y = resample(second[index], count)
        difference = max(_distance_to_curve(x, y).max(), _distance_to_curve(y, x).max())
        maxima.append(float(difference/np.sqrt(1+2*times[index])))
    worst = int(np.argmax(maxima))
    return dict(sampled_frames=len(indices), maximum=maxima[worst],
                worst_time=float(times[indices[worst]]), final=maxima[-1])


def has_crossing(z):
    edges = np.roll(z, -1)-z
    midpoints = z+.5*edges
    pairs = cKDTree(_points(midpoints)).query_pairs(float(abs(edges).max())*(1+1e-10), output_type='ndarray')
    i, j = pairs.T
    keep = (j-i > 1) & (j-i < len(z)-1)
    i, j = i[keep], j[keep]
    delta = z[j]-z[i]
    denominator = (edges[i].conj()*edges[j]).imag
    keep = abs(denominator) > 1e-14
    i, j, delta, denominator = i[keep], j[keep], delta[keep], denominator[keep]
    a = (delta.conj()*edges[j]).imag/denominator
    b = (delta.conj()*edges[i]).imag/denominator
    return bool(np.any((a > 1e-9) & (a < 1-1e-9) & (b > 1e-9) & (b < 1-1e-9)))


def tip_count(z, time):
    radius = abs(resample(z, 1024))/np.sqrt(1+2*time)
    indices = find_peaks(np.tile(radius, 3), prominence=.025, distance=18)[0]
    return int(((indices >= 1024) & (indices < 2048)).sum())


def describe_geometry(curves, times):
    crossings = [float(t) for t, z in zip(times, curves) if has_crossing(z)]
    area_error = max(abs(area(z)/(np.pi*(1+2*t))-1) for t, z in zip(times, curves))
    return dict(crossing_times=crossings, max_area_relative_error=float(area_error),
                tips=[tip_count(z, t) for z, t in zip(curves, times)])


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_case(path, times, curves, info):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, times=times, z=curves, metadata=json.dumps(info))
    temporary.replace(path)


def summarize(folder, *, compare_final=False):
    entries = []
    final_curves = []
    for path in sorted(folder.glob('case_*.npz')):
        with np.load(path) as item:
            info = json.loads(str(item['metadata']))
            times = item['times']
            if compare_final:
                final_curves.append(resample(item['z'][-1], 2048)/np.sqrt(1+2*times[-1]))
        level = info['levels'][-1]
        probes = times[-1]*np.array([.02, .05, .10, .20, .40, .60, .80, 1.])
        frames = abs(times[:, None]-probes).argmin(axis=0)
        tip_profile = [level['tips'][i] for i in frames]
        entries.append(dict(**info['case'], nodes=level['nodes'],
                            final_tips=level['tips'][-1],
                            tip_profile=tip_profile,
                            refinement=level['refinement_difference']['maximum']))
    summary = dict(accepted=len(entries), requested=CASE_COUNT,
                   splits={name: sum(e['split'] == name for e in entries)
                           for name in ('train', 'validation', 'test')},
                   distinct_tip_profiles=len({tuple(e['tip_profile']) for e in entries}),
                   rejected=[p.stem for p in sorted(folder.glob('rejected_*.json'))], cases=entries)
    if len(final_curves) > 1:
        curves = np.stack(final_curves)
        mirrored = curves[:, (-np.arange(curves.shape[1])) % curves.shape[1]].conj()
        frequency = np.fft.fft(curves, axis=1)
        mirror_frequency = np.fft.fft(mirrored, axis=1)
        powers = np.mean(abs(curves)**2, axis=1)
        nearest = (float('inf'), None, None)
        for index, spectrum in enumerate(frequency[:-1]):
            comparison = frequency[index+1:]
            correlations = np.fft.ifft(spectrum.conj()*comparison, axis=1)
            mirror_correlations = np.fft.ifft(spectrum.conj()*mirror_frequency[index+1:], axis=1)
            overlap = np.maximum(abs(correlations).max(axis=1), abs(mirror_correlations).max(axis=1))
            distance = np.sqrt(np.maximum(powers[index]+powers[index+1:]-2*overlap/curves.shape[1], 0.))
            peer = int(distance.argmin())
            candidate = (float(distance[peer]), entries[index]['index'], entries[index+1+peer]['index'])
            nearest = min(nearest, candidate)
        summary['minimum_final_pair_rms_after_symmetry_alignment'] = nearest[0]
        summary['closest_final_cases'] = list(nearest[1:])
    atomic_json(folder/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--T', type=float, default=1000.)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--count', type=int)
    parser.add_argument('--nodes', type=int, nargs='+', default=[768, 1024, 1536, 2048, 2560, 3072])
    parser.add_argument('--output-nodes', type=int, default=2048)
    parser.add_argument('--tolerance', type=float, default=.0035)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    count = CASE_COUNT-args.first if args.count is None else args.count
    if not 0 <= args.first < args.first+count <= CASE_COUNT or args.T <= 0:
        parser.error('Invalid case range or horizon')
    if len(args.nodes) < 2 or sorted(set(args.nodes)) != args.nodes or min(args.nodes) < 64:
        parser.error('Expected at least two increasing node counts of 64 or more')
    if args.output_nodes < 64 or args.tolerance <= 0:
        parser.error('Expected at least 64 output nodes and positive tolerance')

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    times = output_times(args.T)
    configuration = dict(version=2, horizon=args.T, sigma=.015, Q=2*np.pi,
                         nodes=args.nodes, output_nodes=args.output_nodes,
                         tolerance=args.tolerance, times=times.tolist())
    settings = out/'configuration.json'
    if settings.exists():
        existing = json.loads(settings.read_text(encoding='utf-8'))
        old_nodes = existing['nodes']
        if existing != configuration and (existing | {'nodes': args.nodes} != configuration
                                            or args.nodes[:len(old_nodes)] != old_nodes):
            raise ValueError(f'Dataset configuration differs in {out}')
    atomic_json(settings, configuration)
    pressure_cache = {}
    for index in range(args.first, args.first+count):
        path = out/f'case_{index:02}.npz'
        provisional = out/f'provisional_{index:02}.npz'
        rejection = out/f'rejected_{index:02}.json'
        if path.exists():
            with np.load(path) as existing:
                metadata = json.loads(str(existing['metadata']))
                if metadata['accepted'] and np.array_equal(times, existing['times']):
                    print(f'case {index:02} already accepted', flush=True)
                    continue
        started = perf_counter()
        previous = None
        levels = []
        accepted = False
        if provisional.exists():
            with np.load(provisional) as saved:
                info = json.loads(str(saved['metadata']))
                if np.array_equal(times, saved['times']):
                    levels = info['levels']
                    previous = saved['z']
        for level, nodes in enumerate(args.nodes):
            if level < len(levels):
                continue
            step_scale = .78**level
            start_nodes = max(64, (nodes//32)*16) if 20. in times[1:-1] else nodes
            refinements = {}
            if start_nodes != nodes:
                if nodes > 1024 and 150. in times[1:-1]:
                    refinements[20.] = 3*nodes//4
                    refinements[150.] = nodes
                else:
                    refinements[20.] = nodes
            try:
                curves, metrics = evolve(initial_curve(index, start_nodes), times, cfl=.20*step_scale,
                                         step_fraction=.012*step_scale, max_step=2*step_scale,
                                         device=args.device, output_nodes=max(nodes, args.output_nodes),
                                         refinements=refinements, pressure_cache=pressure_cache)
            except RuntimeError as error:
                levels.append(dict(nodes=nodes, error=str(error)))
                print(f'case {index:02} n={nodes} failed: {error}', flush=True)
                continue
            geometry = describe_geometry(curves, times)
            info = dict(nodes=nodes, start_nodes=start_nodes, refinements=refinements,
                        step_scale=step_scale, **metrics, **geometry)
            if previous is not None:
                check = curve_difference(previous, curves, times)
                info['refinement_difference'] = check
                accepted = (check['maximum'] <= args.tolerance and not geometry['crossing_times']
                            and metrics['max_flux_relative_error'] <= 5e-5)
            levels.append(info)
            previous = curves
            difference = info.get('refinement_difference', {}).get('maximum')
            print(f'case {index:02} n={nodes} steps={metrics["steps"]} '
                  f'fallbacks={metrics["dense_pressure_solves"]} delta={difference} '
                  f'tips={geometry["tips"][-1]} accepted={accepted} '
                  f'elapsed={perf_counter()-started:.1f}s', flush=True)
            if accepted:
                break
            save_case(provisional, times, curves, dict(case=case_description(index),
                      accepted=False, levels=levels, seconds=perf_counter()-started))
        info = dict(case=case_description(index), accepted=accepted, levels=levels,
                    seconds=perf_counter()-started)
        if accepted:
            save_case(path, times, curves, info)
            provisional.unlink(missing_ok=True)
            rejection.unlink(missing_ok=True)
        else:
            if previous is not None:
                save_case(provisional, times, previous, info)
            atomic_json(rejection, info)
        summarize(out)
    summary = summarize(out, compare_final=True)
    print(f'Accepted {summary["accepted"]}/{CASE_COUNT} trajectories', flush=True)
    accepted_ids = {entry['index'] for entry in summary['cases']}
    if not set(range(args.first, args.first+count)).issubset(accepted_ids):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
