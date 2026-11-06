"""Compare detected persistent tip births without assuming branch parentage."""
import argparse
import json
from pathlib import Path

import numpy as np

from .events import match_births, tip_births
from ..nn_world.train import complex_curves
from ..trajectory_metrics import atomic_json


def evaluate(path, times, labels, ids):
    predicted = [tip_births(z, times) for z in path]
    associations = [match_births(a, b) for a, b in zip(predicted, labels)]
    matched = [pair for case in associations for pair in case]
    n_predicted = sum(map(len, predicted))
    n_true = sum(map(len, labels))
    return dict(predicted=n_predicted, target=n_true, matched=len(matched),
                f1=2*len(matched)/max(1, n_predicted+n_true),
                mean_time_error=float(np.mean([x[0] for x in matched])) if matched else None,
                mean_angle_error=float(np.mean([x[1] for x in matched])) if matched else None,
                cases=[dict(case=int(case), predicted=a, label=b, matched=c)
                       for case, a, b, c in zip(ids, predicted, labels, associations)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/rollout_t100'))
    args = parser.parse_args()
    root = args.out
    summary = json.loads((root/'summary.json').read_text())
    if (root/'refinement.json').is_file():
        summary['models'].update(json.loads((root/'refinement.json').read_text()))
    result = {}
    for name in summary['models']:
        with np.load(root/name/'predictions.npz') as data:
            times, ids = data['times'], data['ids']
            labels = [tip_births(z, times) for z in complex_curves(data['target'])]
            result[name] = {mode: evaluate(complex_curves(data[mode]), times, labels, ids)
                            for mode in ('direct', 'free', 'teacher')}
    baseline_path = root/'curvefno/test_predictions.npz'
    if baseline_path.is_file():
        with np.load(baseline_path) as data:
            frames = abs(data['times'][:, None]-times).argmin(axis=0)
            if not np.array_equal(data['ids'], ids):
                raise ValueError('Baseline test cases do not match the experimental split')
            z = data['z'][:, frames]/np.sqrt(1+2*times[None, :, None])
            result['curvefno'] = dict(direct=evaluate(z, times, labels, ids))
    atomic_json(root/'tip_births.json', result)


if __name__ == '__main__':
    main()
