"""Certify denser late tip windows on disjoint training and test shapes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .evolution import evolve
from .generate import save_case
from ..interface_events import match_births, tip_births
from ..trajectory_metrics import atomic_json, curve_difference, has_crossing


WINDOWS = Path('data/supervised/late_tip_density')


def generate(source, out, device):
    out.mkdir(parents=True, exist_ok=True)
    candidates = {'train': {}, 'test': {}}
    for path in sorted(source.glob('case_*.npz')):
        with np.load(path) as sample:
            info = json.loads(str(sample['metadata']))['case']
            if info['split'] not in candidates:
                continue
            times, curves = sample['times'], sample['z']
        births = tip_births(curves/np.sqrt(1+2*times[:, None]), times)
        for t, angle in births:
            if 50 <= t <= 95 and all(t+shift in times for shift in (-10, -5, 5)):
                candidates[info['split']].setdefault(info['dominant'], []).append((info['index'], t, angle, path))
    rng = np.random.default_rng(20261107)
    records = []
    cache = {}
    for split, by_mode in candidates.items():
        for mode, pool in sorted(by_mode.items()):
            limit = 6 if split == 'train' else 3
            seen = set()
            used = set()
            count = 0
            for i in rng.permutation(len(pool)):
                index, middle, angle, path = pool[i]
                if (index, middle) in seen or (split == 'train' and index in used):
                    continue
                seen.add((index, middle))
                name = f'case_{index:03}_t{int(middle):03}.npz'
                destination = out/name
                if destination.exists():
                    with np.load(destination) as item:
                        record = json.loads(str(item['metadata']))
                else:
                    with np.load(path) as sample:
                        saved = sample['times']
                        anchors = middle+np.array([-10., -5., 0., 5.])
                        coarse = sample['z'][np.searchsorted(saved, anchors)]
                    times = np.linspace(anchors[0], anchors[-1], 31)
                    try:
                        low, _ = evolve(resample(coarse[0], 768), times, device=device, cfl=.16,
                                        step_fraction=.005, output_nodes=1024, pressure_cache=cache)
                        high, metrics = evolve(resample(coarse[0], 1024), times, device=device, cfl=.12,
                                               step_fraction=.0035, output_nodes=1024, pressure_cache=cache)
                    except RuntimeError as error:
                        print(name, error, flush=True)
                        continue
                    refinement = curve_difference(low, high, times, stride=1)['maximum']
                    anchor = curve_difference(coarse, high[::10], anchors, stride=1)['maximum']
                    events = tip_births(high/np.sqrt(1+2*times[:, None]), times)
                    matching = [event for event in events if match_births([event], [(middle, angle)], time_limit=5)]
                    flux = metrics['max_flux_relative_error']
                    if (max(refinement, anchor) > .0035 or not matching or flux > 5e-5
                            or any(has_crossing(frame) for frame in high)):
                        print(name, 'rejected', refinement, anchor, flush=True)
                        continue
                    record = dict(case=index, mode=mode, split=split, file=name, coarse_time=middle,
                                  fine_events=events, matched_events=matching, refinement=refinement,
                                  anchor=anchor, max_flux_relative_error=flux)
                    save_case(destination, times, high, record)
                used.add(index)
                count += 1
                records.append(record)
                print(split, name, 'accepted', flush=True)
                if count == limit:
                    break
    atomic_json(out/'summary.json', dict(windows=records, dt=.5))
    print({split: sum(r['split'] == split for r in records) for split in candidates}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/supervised/trajectories_t100_dense_128'))
    parser.add_argument('--out', type=Path, default=WINDOWS)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    generate(args.source, args.out, args.device)
