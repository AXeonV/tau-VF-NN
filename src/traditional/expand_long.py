"""Build 128 distinct trajectories with certified labels through t=1000."""
import argparse
import json
import os
from pathlib import Path
from shutil import copy2
from time import perf_counter

import numpy as np
from scipy.signal import resample

from .evolution import evolve
from .expand import initial_from_terms, solve
from .generate import save_case
from ..trajectory_data import case_paths
from ..trajectory_metrics import atomic_json, curve_difference, has_crossing, tip_count


DATA = Path('data/supervised')
LONG = 'trajectories_t1000_128'
DENSE = 'trajectories_t100_dense_128'


def read(path):
    with np.load(path) as item:
        return item['times'], item['z'], json.loads(str(item['metadata']))


def json_file(path):
    return json.loads(path.read_text(encoding='utf-8'))


def link(source, destination):
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        copy2(source, destination)


def diversity(profile, pool):
    """Minimum radial RMS distance to a pool after rotations and reflections."""
    spectrum = np.fft.rfft(profile)
    other = np.fft.rfft(pool, axis=1)
    dots = np.fft.irfft(spectrum.conj()*other, n=len(profile), axis=1)
    flipped = np.fft.irfft(spectrum*other, n=len(profile), axis=1)
    return float(np.sqrt(np.maximum((profile**2).mean()+(pool**2).mean(1)
                     -2*np.maximum(dots.max(1), flipped.max(1))/len(profile), 0.)).min())


def choose_cases(dense):
    rng = np.random.default_rng(20261123)
    known = {k: [] for k in range(2, 10)}
    alpha = 2*np.pi*np.arange(256)/256

    def profile(terms):
        r = 1+sum(a*np.cos(k*alpha+p) for k, a, p in terms)
        return r/np.sqrt(np.mean(r**2))

    for record in json_file(dense/'summary.json')['cases']:
        if record['split'] == 'train':
            known[record['dominant']].append(profile(record['terms']))
    cases = []
    for dominant in known:
        candidates = []
        for index in range(64):
            modes = [k for k in range(2, 16) if k != dominant]
            extra = rng.choice(modes, 3, replace=False)
            amplitude = .10+.17*(index+rng.random())/64
            terms = [[dominant, float(amplitude), 0.]]
            for k, (lo, hi) in zip(extra, [(.040, .091), (.015, .045), (.006, .025)]):
                terms.append([int(k), float(rng.uniform(lo, hi)), float(rng.uniform(-np.pi, np.pi))])
            candidates.append((terms, profile(terms)))
        for _ in range(4):
            reference = np.stack(known[dominant])
            selection = int(np.argmax([diversity(r, reference) for _, r in candidates]))
            terms, r = candidates.pop(selection)
            known[dominant].append(r)
            cases.append(dict(index=96+len(cases), dominant=dominant, terms=terms,
                              split='train', source='maximin', separation=diversity(r, reference)))
    return cases


def prepare(data):
    original, dense = data/'trajectories', data/'trajectories_t100_dense'
    long, early = data/LONG, data/DENSE
    long.mkdir(parents=True, exist_ok=True)
    early.mkdir(parents=True, exist_ok=True)
    for source, target, expected in [(original, long, 64), (dense, early, 96)]:
        settings = dict(json_file(source/'configuration.json'), expected_cases=128)
        settings.pop('new_per_family', None)
        inputs = case_paths(source)
        if len(inputs) != expected:
            raise ValueError(f'Expected {expected} certified paths in {source}')
        config = target/'configuration.json'
        if config.exists() and json_file(config) != settings:
            raise ValueError(f'Existing configuration does not match {target}')
        atomic_json(config, settings)
        for path in inputs:
            link(path, target/f'case_{int(path.stem[5:]):03}.npz')
    plan = long/'new_initials.json'
    if not plan.exists():
        atomic_json(plan, choose_cases(dense))


def continuation(initial, times, mode, sigma, tolerance, cache):
    # Higher dominant modes tend to need higher long-time interface resolution.
    levels = (1024, 1536, 2048, 2560, 3072, 3584) if mode < 7 else (1536, 2048, 2560, 3072, 3584)
    previous, records = None, []
    for n in levels:
        cfl = .17*(1024/n)**.65
        fraction = .007*(1024/n)**.7
        start = perf_counter()
        try:
            z, info = evolve(resample(initial, n), times, sigma=sigma, cfl=cfl, step_fraction=fraction,
                             output_nodes=2048, pressure_cache=cache)
        except RuntimeError as error:
            records.append(dict(nodes=n, error=str(error)))
            previous = None
            continue
        difference = None if previous is None else curve_difference(previous, z, times, stride=1)
        simple = not any(has_crossing(value) for value in z)
        records.append(dict(nodes=n, cfl=cfl, step_fraction=fraction, simple=simple,
                            refinement_difference=difference, seconds=perf_counter()-start, **info))
        error = difference['maximum'] if difference else None
        print(f'  n={n} steps={info["steps"]} difference={error} elapsed={perf_counter()-start:.1f}s', flush=True)
        if difference and error <= tolerance and simple and info['max_flux_relative_error'] <= 5e-5:
            return z, records
        previous = z if simple and info['max_flux_relative_error'] <= 5e-5 else None
    return None, records


def run(data, shard, shards, only):
    long, early = data/LONG, data/DENSE
    config = json_file(long/'configuration.json')
    times = np.asarray(config['times'])
    early_times = np.asarray(json_file(early/'configuration.json')['times'])
    cut = int(np.flatnonzero(times == 100.).item())
    prefix_indices = np.searchsorted(early_times, times[:cut+1])
    if not np.array_equal(early_times[prefix_indices], times[:cut+1]):
        raise ValueError('Original timestamps are missing from the dense prefix')
    new = {case['index']: case for case in json_file(long/'new_initials.json')}
    # Complete a representative mix of families early while each worker owns disjoint IDs.
    order = [i for variant in range(4) for family in range(8)
             for i in (64+4*family+variant, 96+4*family+variant)]
    cache = {}
    for index in only or order[shard::shards]:
        destination = long/f'case_{index:03}.npz'
        prefix = early/f'case_{index:03}.npz'
        if destination.exists():
            continue
        print(f'Case {index:03}', flush=True)
        started = perf_counter()
        if prefix.is_file():
            _, history, metadata = read(prefix)
            case = metadata['case']
        else:
            case = new[index]
            initial = initial_from_terms(case['terms'])
            rejection = early/f'rejected_{index:03}.json'
            history, levels = None, []
            if rejection.exists():
                levels = json_file(rejection)['levels']
            else:
                history, levels = solve(initial, early_times, config['sigma'], config['tolerance'], cache)
            if history is None:
                fine_levels = [(1536, .15, .0045), (2048, .12, .0035),
                               (2560, .10, .0028), (3072, .09, .0024)]
                history, checks = solve(initial, early_times, config['sigma'],
                                        config['tolerance'], cache, levels=fine_levels)
                levels.extend(checks)
            if history is None:
                atomic_json(rejection, dict(case=case, levels=levels))
                print(f'Case {index:03} needs further early refinement', flush=True)
                continue
            metadata = dict(case=case, accepted=True, levels=levels,
                            final_tips=tip_count(history[-1], 100.), seconds=perf_counter()-started)
            save_case(prefix, early_times, history, metadata)
            rejection.unlink(missing_ok=True)
        tail, levels = continuation(history[-1], times[cut:], case['dominant'], config['sigma'],
                                    config['tolerance'], cache)
        if tail is None:
            atomic_json(long/f'rejected_{index:03}.json', dict(case=case, levels=levels))
            print(f'Case {index:03} needs further refinement', flush=True)
            continue
        head = resample(history[prefix_indices], tail.shape[1], axis=1)
        label = np.concatenate([head, tail[1:]])
        save_case(destination, times, label, dict(case=case, accepted=True, levels=levels,
                  prefix_levels=metadata['levels'], final_tips=tip_count(label[-1], 1000.),
                  seconds=perf_counter()-started))
        (long/f'rejected_{index:03}.json').unlink(missing_ok=True)
        print(f'Case {index:03} accepted in {perf_counter()-started:.1f}s', flush=True)


def report(data):
    for name in (LONG, DENSE):
        folder = data/name
        cases = []
        for path in sorted(folder.glob('case_*.npz')):
            with np.load(path) as item:
                record = json.loads(str(item['metadata']))
            level = record['levels'][-1]
            cases.append(dict(record['case'], nodes=level['nodes'],
                              error=level['refinement_difference']['maximum'],
                              flux_error=level['max_flux_relative_error'],
                              final_tips=record.get('final_tips', level.get('tips', [None])[-1])))
        train_ids = [c['index'] for c in cases if c['split'] == 'train']
        views = [dict(rotation_degrees=45*i, reflected=reflected) for reflected in (False, True)
                 for i in range(8)]
        atomic_json(folder/'augmentation.json', dict(train_ids=train_ids, views=views,
                    nominal_train_views=len(views)*len(train_ids),
                    online=['continuous rotation', 'reflection', 'cyclic arclength shift']))
        atomic_json(folder/'summary.json', dict(accepted=len(cases), requested=128, cases=cases,
            splits={key: sum(c['split'] == key for c in cases) for key in ('train', 'validation', 'test')},
            rejected=[p.stem for p in sorted(folder.glob('rejected_*.json'))],
            nominal_train_views=len(views)*len(train_ids)))
        print(name, len(cases), '/ 128', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'report'], required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case', type=int, nargs='+', choices=range(64, 128))
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Expected 0 <= shard < shards')
    if args.stage == 'run':
        run(args.data, args.shard, args.shards, args.case)
    elif args.stage == 'prepare':
        prepare(args.data)
    else:
        report(args.data)


if __name__ == '__main__':
    main()
