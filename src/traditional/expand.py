"""Extend the T=100 supervision with dense branching windows and independent initial shapes."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample

from .evolution import evolve
from .generate import save_case
from ..trajectory_metrics import atomic_json, curve_difference, has_crossing, tip_count


def new_cases(first, per_family):
    """Stratify the dominant amplitude and vary secondary-mode amplitudes and phases."""
    random = np.random.default_rng(20261119)
    for mode in range(2, 10):
        amplitudes = .105+.16*(random.permutation(per_family)+random.random(per_family))/per_family
        for amplitude in amplitudes:
            pool = [k for k in range(2, 15) if k != mode]
            secondary, tertiary, fourth = map(int, random.choice(pool, 3, replace=False))
            terms = [[mode, float(amplitude), 0.]]
            for k, lower, upper in [(secondary, .035, .09), (tertiary, .01, .038), (fourth, .004, .02)]:
                terms.append([k, float(random.uniform(lower, upper)), float(random.uniform(-np.pi, np.pi))])
            yield dict(index=first, dominant=mode, terms=terms, split='train', source='new')
            first += 1


def initial_from_terms(terms, nodes=1024):
    alpha = np.linspace(0., 2*np.pi, nodes, endpoint=False)
    radius = 1+sum(amplitude*np.cos(mode*alpha+phase) for mode, amplitude, phase in terms)
    return radius*np.exp(1j*alpha)


def solve(initial, times, sigma, tolerance, cache, known=None):
    previous = None
    checks = []
    levels = ((1024, .17, .006), (1536, .15, .0045)) if known else (
        (768, .19, .008), (1024, .17, .006), (1536, .15, .0045))
    for nodes, cfl, fraction in levels:
        try:
            curves, info = evolve(resample(initial, nodes), times, sigma=sigma, cfl=cfl,
                                  step_fraction=fraction, output_nodes=1024, pressure_cache=cache)
        except RuntimeError as error:
            checks.append(dict(nodes=nodes, error=str(error)))
            continue
        if known:
            old_times, old_curves = known
            delta = curve_difference(curves[np.searchsorted(times, old_times)], old_curves, old_times, stride=1)
        elif previous is not None:
            delta = curve_difference(curves, previous, times, stride=1)
        else:
            delta = None
        previous = curves
        simple = not any(has_crossing(z) for z in curves)
        checks.append(dict(nodes=nodes, cfl=cfl, step_fraction=fraction, simple=simple,
                           refinement_difference=delta, **info))
        if delta and delta['maximum'] <= tolerance and simple and info['max_flux_relative_error'] <= 5e-5:
            return curves, checks
    return None, checks


def summary(out, expected, rejected):
    cases = []
    for path in sorted(out.glob('case_*.npz')):
        with np.load(path) as item:
            info = json.loads(str(item['metadata']))
        level = info['levels'][-1]
        cases.append(dict(info['case'], nodes=level['nodes'], error=level['refinement_difference']['maximum'],
                          flux_error=level['max_flux_relative_error'], final_tips=info['final_tips']))
    atomic_json(out/'summary.json', dict(requested=expected, accepted=len(cases), rejected=rejected,
        splits={name: sum(c['split'] == name for c in cases) for name in ('train', 'validation', 'test')}, cases=cases))
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/trajectories_t100_dense'))
    parser.add_argument('--new-per-family', type=int, default=4)
    args = parser.parse_args()
    if args.new_per_family < 1:
        parser.error('Expected at least one new shape per family')
    config = json.loads((args.source/'configuration.json').read_text())
    saved = np.asarray(config['times'])
    saved = saved[saved <= 100.]
    times = np.unique(np.r_[saved, np.arange(.25, 10., .25), np.arange(10., 40.5, .5)])
    old_cases = sorted(args.source.glob('case_*.npz'))
    if len(old_cases) != 64 or saved[-1] != 100:
        raise ValueError('Expected the original 64 trajectories with a t=100 snapshot')
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(version=1, horizon=100., sigma=config['sigma'], Q=config['Q'], output_nodes=1024,
                    tolerance=.0035, times=times.tolist(), new_per_family=args.new_per_family)
    settings_path = output/'configuration.json'
    if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Expansion configuration differs from the existing output')
    atomic_json(settings_path, settings)
    additions = list(new_cases(64, args.new_per_family))
    # Interleave new and existing cases so both parts of the dataset become available early.
    jobs = [job for i, path in enumerate(old_cases) for job in
            ([path, additions[i]] if i < len(additions) else [path])]
    jobs.extend(additions[len(old_cases):])
    rejected, cache = [], {}
    for job in jobs:
        started = perf_counter()
        known = None
        if isinstance(job, Path):
            with np.load(job) as item:
                metadata = json.loads(str(item['metadata']))
                if not metadata['accepted'] or not np.array_equal(item['times'][:len(saved)], saved):
                    raise ValueError(f'Invalid original label {job}')
                original = item['z'][:len(saved)]
            case = dict(metadata['case'], source='densified')
            initial = original[0]
            known = saved, original
        else:
            case = job
            initial = initial_from_terms(case['terms'])
        index = case['index']
        path = output/f'case_{index:03}.npz'
        if path.exists():
            with np.load(path) as item:
                metadata = json.loads(str(item['metadata']))
                if (not metadata['accepted'] or metadata['case'] != case
                        or not np.array_equal(item['times'], times)):
                    raise ValueError(f'Existing expanded label does not match the requested case: {path}')
            continue
        label, checks = solve(initial, times, settings['sigma'], settings['tolerance'], cache, known)
        if label is None:
            rejected.append(index)
            atomic_json(output/f'rejected_{index:03}.json', dict(case=case, levels=checks))
        else:
            save_case(path, times, label, dict(case=case, accepted=True, levels=checks,
                      final_tips=tip_count(label[-1], times[-1]), seconds=perf_counter()-started))
            (output/f'rejected_{index:03}.json').unlink(missing_ok=True)
        print(f'{index:03} {case["source"]} accepted={label is not None} '
              f'nodes={checks[-1]["nodes"]} elapsed={perf_counter()-started:.1f}s', flush=True)
        # Publish progress without re-reading every existing trajectory at each case.
        if index % 4 == 0:
            summary(output, len(jobs), rejected)
    total = summary(output, len(jobs), rejected)
    print(f'Accepted {total}/{len(jobs)} trajectories', flush=True)
    if total != len(jobs):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
