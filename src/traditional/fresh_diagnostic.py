"""Generate independent, refinement-checked trajectories for drift diagnostics."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .expand import initial_from_terms, solve
from .expand_long import diversity
from .generate import save_case
from ..interface_events import tip_births
from ..trajectory_metrics import atomic_json, tip_count


DATA = Path('data/supervised')
SOURCE = DATA/'trajectories_t100_dense_128'
OUTPUT = DATA/'drift_diagnostic_fresh'


def radial_profile(terms):
    radius = abs(initial_from_terms(terms, nodes=256))
    return radius/np.sqrt(np.mean(radius**2))


def candidates(index, mode, reference):
    rng = np.random.default_rng(102195 + index)
    others = [k for k in range(2, 15) if k != mode]
    for _ in range(16):
        terms = [[mode, float(rng.uniform(.115, .245)), 0.]]
        for frequency, (lower, upper) in zip(rng.choice(others, 3, replace=False),
                                             [(.035, .09), (.012, .043), (.005, .023)]):
            terms.append([int(frequency), float(rng.uniform(lower, upper)),
                          float(rng.uniform(-np.pi, np.pi))])
        separation = diversity(radial_profile(terms), reference)
        if separation >= .03:
            yield dict(index=index, dominant=mode, terms=terms, split='test',
                       source='fresh_drift_diagnostic', separation=separation)


def generate(source, output):
    config = json.loads((source/'configuration.json').read_text())
    existing = json.loads((source/'summary.json').read_text())['cases']
    times = np.asarray(config['times'])
    if times[0] != 0 or times[-1] != 100:
        raise ValueError('Expected the dense T=100 time grid')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config, expected_cases=8)
    config_path = output/'configuration.json'
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'Dataset configuration does not match {output}')
    atomic_json(config_path, config)
    reference = np.stack([radial_profile(case['terms']) for case in existing])
    first_id = max(case['index'] for case in existing)+1
    cache, records, failures = {}, [], []
    for mode in range(2, 10):
        index = first_id+mode-2
        destination = output/f'case_{index:03d}.npz'
        if destination.exists():
            with np.load(destination) as saved:
                record = json.loads(str(saved['metadata']))
                if (not record['accepted'] or record['case']['index'] != index
                        or record['case']['split'] != 'test'
                        or not np.array_equal(saved['times'], times)):
                    raise ValueError(f'Invalid cached trajectory: {destination}')
        else:
            for case in candidates(index, mode, reference):
                started = perf_counter()
                label, checks = solve(initial_from_terms(case['terms']), times, config['sigma'],
                                      config['tolerance'], cache, levels=(
                                          (768, .19, .008), (1024, .17, .006),
                                          (1536, .15, .0045), (2048, .12, .0035)))
                stable = label is not None and all(
                    level.get('simple') and level['max_flux_relative_error'] <= 5e-5
                    for level in checks[-2:])
                if not stable:
                    failures.append(dict(case=case, levels=checks))
                    print(f'{index} refinement rejected, trying another initial shape', flush=True)
                    continue
                births = tip_births(label/np.sqrt(1+2*times[:, None]), times)
                record = dict(case=case, accepted=True, levels=checks,
                              final_tips=tip_count(label[-1], times[-1]),
                              events=births, seconds=perf_counter()-started)
                save_case(destination, times, label, record)
                break
            else:
                atomic_json(output/'summary.json', dict(cases=records, rejected=failures))
                raise RuntimeError(f'No certified candidate for mode {mode}')
        late = sum(50 <= time <= 95 for time, _ in record['events'])
        error = record['levels'][-1]['refinement_difference']['maximum']
        records.append(dict(record['case'], error=error, late_events=late, events=record['events']))
        print(f'{index} mode={mode} error={error:.6g} late_events={late} '
              f'seconds={record["seconds"]:.1f}', flush=True)
    atomic_json(output/'summary.json', dict(cases=records, rejected=failures, accepted=len(records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.source, args.out)
