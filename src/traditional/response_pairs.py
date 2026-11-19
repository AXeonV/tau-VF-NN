"""Refinement-checked paired continuations around persistent tip births."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .evolution import evolve, redistribute
from ..interface_events import angle_distance, tip_births
from ..trajectory_data import DEFAULT_DATA
from ..trajectory_metrics import atomic_json, curve_difference, has_crossing


PAIR_DATA = Path('data/supervised/branch_response')
DIAGNOSTIC = DEFAULT_DATA.parent/'drift_diagnostic_fresh'


def pair(case_path, cache, *, ordinary=False):
    with np.load(case_path) as item:
        times, curves = item['times'], item['z']
    events = tip_births(curves/np.sqrt(1+2*times[:, None]), times)
    eligible = [(t, a) for t, a in events if 18 <= t <= 90]
    if not eligible:
        return None
    event, angle = min(eligible, key=lambda p: abs(p[0]-45))
    index = np.abs(times-(event-3)).argmin()
    grid = times[index]+np.arange(10)*.5
    z = redistribute(curves[index])[0]
    tangent = np.roll(z, -1)-np.roll(z, 1)
    normal = -1j*tangent/abs(tangent)
    # Local bump centred on the outward-most point near the future birth.
    sector = angle_distance(np.angle(z), angle) <= .25
    if not sector.any():
        return None
    center = np.where(sector, abs(z), -np.inf).argmax()
    arc = np.angle(np.exp(2j*np.pi*(np.arange(len(z))-center)/len(z)))
    bump = .008*np.sqrt(1+2*grid[0])*np.exp(-.5*(arc/.12)**2)*normal
    fine, coarse, errors = [], [], []
    for sign in ((0,) if ordinary else (-1, 1)):
        levels = []
        for nodes, cfl, fraction in [(768, .19, .008), (1024, .17, .006)]:
            answer, check = evolve(resample(z+sign*bump, nodes), grid, cfl=cfl,
                                   step_fraction=fraction, output_nodes=1024, pressure_cache=cache)
            if check['max_flux_relative_error'] > .00005 or any(map(has_crossing, answer)):
                return None
            levels.append(answer)
        error = curve_difference(*levels, grid, stride=1)['maximum']
        if error > .0005:
            return None
        errors.append(error)
        for dest, values in zip((coarse, fine), levels):
            dest.append((values/np.sqrt(1+2*grid[:, None])).astype(np.complex64))
    return dict(initial=curves[0].astype(np.complex64), times=grid, angle=np.angle(z[center]),
                event_angle=angle, event_time=event, fine=np.stack(fine), coarse=np.stack(coarse),
                error=np.array(errors))


def main(group):
    PAIR_DATA.mkdir(exist_ok=True, parents=True)
    cache, records = {}, []
    groups = {'pairs': [('train', DEFAULT_DATA, 3), ('diagnostic', DIAGNOSTIC, 1)],
              'ordinary': [('ordinary', DEFAULT_DATA, 6)],
              'sealed': [('sealed', DEFAULT_DATA.parent/'gradient_response_holdout', 1)]}[group]
    for split, folder, limit in groups:
        cases = json.loads((folder/'summary.json').read_text())['cases']
        for mode in range(2, 10):
            accepted = 0
            for case in cases:
                if case['dominant'] != mode or case['split'] != ('test' if split in ('diagnostic','sealed') else 'train'):
                    continue
                idx = case['index']
                if split == 'ordinary' and (PAIR_DATA/f'train_{idx:03}.npz').exists():
                    continue
                path = PAIR_DATA/f'{split}_{idx:03}.npz'
                if not path.exists():
                    data = pair(folder/f'case_{idx:03}.npz', cache, ordinary=split == 'ordinary')
                    if data is None:
                        continue
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as f:
                        np.savez_compressed(f, **data)
                    tmp.replace(path)
                with np.load(path) as saved:
                    records.append(dict(case=idx, split=split, mode=mode, event=float(saved['event_time']),
                                        refinement=float(saved['error'].max())))
                accepted += 1
                print(split, idx, records[-1]['refinement'], flush=True)
                if accepted == limit:
                    break
    atomic_json(PAIR_DATA/('summary.json' if group == 'pairs' else f'{group}.json'), dict(cases=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['pairs','ordinary','sealed'], default='pairs')
    main(parser.parse_args().group)
