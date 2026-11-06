"""Measure numerical resolution and physical sensitivity to controlled initial perturbations."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample

from .events import match_events, split_events
from ..traditional.evolution import area, evolve
from ..traditional.pressure import derivative
from ..trajectory_metrics import _distance_to_curve, atomic_json, has_crossing


def distances(a, b):
    return np.asarray([max(_distance_to_curve(x, y).max(), _distance_to_curve(y, x).max())
                       for x, y in zip(a, b)])


def run(initial, times, nodes, sigma, cache):
    start = perf_counter()
    path, info = evolve(resample(initial, nodes), times, sigma=sigma, cfl=.16, step_fraction=.006,
                        output_nodes=2048, pressure_cache=cache)
    scaled = path/np.sqrt(1+2*times[:, None])
    return scaled, dict(seconds=perf_counter()-start, nodes=nodes, **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/rollout_t100'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = (7, 23, 39, 63)
    config = json.loads((args.data/'configuration.json').read_text())
    saved = np.asarray(config['times'])
    saved = saved[saved <= 100]
    times = np.unique(np.r_[saved, np.arange(1., 100.)])
    archive = {}
    cache = {}
    result = dict(times=times.tolist(), levels=[1e-5, 1e-3, 1e-2], cases=[])
    for case in cases:
        with np.load(args.data/f'case_{case:02d}.npz') as file:
            initial = file['z'][0]
            original = file['z'][:len(saved)]/np.sqrt(1+2*saved[:, None])
            mode = json.loads(str(file['metadata']))['case']['dominant']
        coarse, coarse_info = run(initial, times, 1024, config['sigma'], cache)
        fine, fine_info = run(initial, times, 1536, config['sigma'], cache)
        numerical = distances(coarse, fine)
        original_error = distances(fine[np.searchsorted(times, saved)], original)
        archived = split_events(fine, times)
        normal = -1j*derivative(initial)/abs(derivative(initial))
        alpha = 2*np.pi*np.arange(len(initial))/len(initial)
        direction = np.cos((mode+1)*alpha+.37)+.45*np.sin(2*mode*alpha-.81)
        direction /= abs(direction).max()
        record = dict(case=case, coarse=coarse_info, fine=fine_info, numerical=numerical.tolist(),
                      original_max=float(original_error.max()), events=archived, perturbations=[])
        archive[f'base_{case}'] = fine
        for level in result['levels']:
            for sign in (-1, 1):
                perturbed = initial+sign*level*direction*normal
                perturbed *= np.sqrt(np.pi/area(perturbed))
                path, info = run(perturbed, times, 1536, config['sigma'], cache)
                separation = distances(path, fine)
                events = split_events(path, times)
                # A geometry deviation must also clear the observed mesh-disagreement floor.
                detectable = separation > np.maximum(3*numerical, .01)
                thresholds = {}
                for threshold in (.01, .05, .10):
                    observed = np.flatnonzero(detectable & (separation >= threshold))
                    thresholds[str(threshold)] = float(times[observed[0]]) if len(observed) else None
                record['perturbations'].append(dict(level=level, sign=sign, actual_initial=float(separation[0]),
                    final=float(separation[-1]), maximum=float(separation.max()), thresholds=thresholds,
                    distance=separation.tolist(), events=events, matches=match_events(events, archived, time_limit=3.),
                    crossings=sum(has_crossing(z) for z in path), **info))
                archive[f'{case}_{level:g}_{sign:+d}'] = path
                print('perturbation', case, level, sign, f'end={separation[-1]:.5g}', thresholds, flush=True)
        result['cases'].append(record)
        atomic_json(args.out/'sensitivity.json', result)
        np.savez_compressed(args.out/'sensitivity_curves.npz', times=times, **archive)
        print('finished case', case, 'resolution', float(numerical.max()),
              'dataset', float(original_error.max()), flush=True)


if __name__ == '__main__':
    main()
