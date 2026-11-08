"""Aggregate matched drift histories across seeds, with equal weight per initial shape."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.trajectory_metrics import atomic_json


def records(report):
    indexed = {(r['case'], r['window'], r['drift_steps']): r for r in report['records']}
    if len(indexed) != len(report['records']):
        raise ValueError('Report contains duplicated histories')
    return indexed


def mean_curves(selected, field):
    by_case = {}
    for key, pair in selected.items():
        by_case.setdefault(key[0], []).extend(pair)
    names = sorted(set.intersection(*(set(r[field]) for pair in selected.values() for r in pair)))
    return {name: np.mean([np.mean([r[field][name] for r in group], axis=0)
                          for group in by_case.values()], axis=0).tolist() for name in names}


def compare(reports):
    all_records = [records(report) for report in reports]
    common = sorted(set.intersection(*(set(r) for r in all_records)))
    if not common:
        raise ValueError('No matched certified histories across reports')
    selected = {key: [r[key] for r in all_records] for key in common}
    time = np.asarray(reports[0]['time_offsets'], dtype=float)
    if any(not np.array_equal(time, r['time_offsets']) for r in reports[1:]):
        raise ValueError('Reports use different evaluation times')
    errors = mean_curves(selected, 'errors')
    motion = mean_curves(selected, 'teacher_motion')
    local_motion = mean_curves(selected, 'local_teacher_motion')
    if len(time) != len(errors['physical']):
        raise ValueError('Error series do not match evaluation times')
    by_case = {}
    for key, pair in selected.items():
        by_case.setdefault(key[0], []).extend(pair)
    sampled = {}
    for value in (.5, 2., 4.):
        i = int(np.flatnonzero(time == value).item())
        gains = [np.mean([r['errors']['physical'][i]/r['initial_error'] for r in group])
                 for group in by_case.values()]
        sampled[str(value)] = dict(physical_amplification_median=float(np.median(gains)),
            physical_amplification_by_shape={case: float(gain) for case, gain in zip(by_case, gains)},
            physical_greater_than_initial_shapes=int(sum(g > 1 for g in gains)),
            free_exceeds_physical_separation_shapes=int(sum(
                np.mean([r['errors']['free_same'][i]-r['errors']['physical'][i] for r in group]) > 0
                for group in by_case.values())),
            teacher_beats_persistence_shapes=int(sum(
                np.mean([r['errors']['teacher_same'][i]-r['teacher_motion']['drift'][i] for r in group]) < 0
                for group in by_case.values())))
    paired_records = [r for group in by_case.values() for r in group]
    events = {label: sum(bool(r['events'][label]['matched']) for r in paired_records)
              for label in ('physical_clean', 'physical_drift', 'free', 'teacher', 'free_clean')}
    events['neural_matches_same_state_physical'] = sum(
        bool(r['events']['drift_network_vs_drift_physical']) for r in paired_records)
    return dict(time_offsets=time.tolist(), matched_histories=len(common), shapes=sorted(by_case),
                input_records=[len(r) for r in all_records], errors=errors,
                local_errors=mean_curves(selected, 'local_errors'), sampled=sampled,
                teacher_motion=motion, local_teacher_motion=local_motion, event_matches=events,
                max_refinement=max(r['solver']['max_refinement'] for r in paired_records),
                max_raw_to_shared=max(r['raw_to_shared'] for r in paired_records))


def plot(summary, destination):
    time = summary['time_offsets']
    errors = summary['errors']
    fig, axis = plt.subplots(figsize=(8, 4.6), layout='constrained')
    lines = [('physical', 'Physical separation of clean and drifted states', '#46536a', '-'),
             ('free_same', 'NN free rollout error from the same drifted state', '#c45e37', '-'),
             ('teacher_same', 'NN one-step error on numerical states', '#16847a', '--'),
             ('clean_free', 'NN free rollout error from the clean state', '#8d6bb0', ':')]
    for key, name, color, style in lines:
        axis.plot(time, 100*np.array(errors[key]), label=name, color=color, ls=style, lw=2)
    axis.set(xlabel='Time after the restart', ylabel='Contour distance (% of equivalent radius)')
    axis.grid(alpha=.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = compare([json.loads(path.read_text(encoding='utf-8')) for path in args.reports])
    atomic_json(args.out/'comparison.json', summary)
    plot(summary, args.out/'errors.png')
