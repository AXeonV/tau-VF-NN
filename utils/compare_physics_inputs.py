"""Summarize the matched physical-input ablations without pooling repeated shapes."""
import argparse
import json
from pathlib import Path
from statistics import mean

from src.trajectory_metrics import atomic_json


VARIANTS = ('baseline', 'geometry_control', 'neighbors', 'history', 'combined')
FUSIONS = ('fusion_trunk', 'fusion_geometry', 'fusion_joint')


def metrics(run):
    velocities = run['velocity']
    all_shapes = velocities['all']['mean_by_shape']
    drift = run['drift']
    errors = drift['mean_errors']
    native = run['native']
    return dict(speed_rmse=all_shapes['velocity_rmse'],
                sector_speed_rmse=all_shapes['sector_velocity_rmse'],
                active_sign=all_shapes['active_sign_accuracy'],
                half_step=all_shapes['target_contour_distance'],
                drift_speed_rmse=velocities['drift']['mean_by_shape']['velocity_rmse'],
                drift_free_4=errors['free'][8],
                drift_sector_4=errors['free_sector'][8],
                drift_births=drift['branch_matches'], drift_birth_labels=drift['physical_branches'],
                native_terminal=native['mean_final_geometry_distance'],
                native_births=native['tip_births_matched'], native_birth_labels=native['tip_births_label'],
                native_crossings=native['self_intersecting_cases'])


def compare(report):
    variants = VARIANTS+('history_replace',) if any(
        run.get('variant') == 'history_replace' for run in report['runs']) else VARIANTS
    if any(run.get('variant') in FUSIONS for run in report['runs']):
        variants += FUSIONS
    by_variant = {key: {} for key in variants}
    for run in report['runs']:
        variant = run.get('variant') or 'baseline'
        seed = str(run['seed'])
        if seed in by_variant[variant] or run['steps'] != 1200:
            raise ValueError('Expected one 1200-step run per variant and seed')
        by_variant[variant][seed] = metrics(run)
    seeds = sorted(by_variant['baseline'])
    if len(seeds) < 2 or any(sorted(runs) != seeds for runs in by_variant.values()):
        raise ValueError('Expected the same paired seeds in every arm')
    count_keys = {'drift_births', 'drift_birth_labels', 'native_births', 'native_birth_labels',
                  'native_crossings'}
    aggregate = {}
    for name, runs in by_variant.items():
        keys = runs[seeds[0]]
        aggregate[name] = {key: (sum if key in count_keys else mean)(v[key] for v in runs.values())
                           for key in keys}
        print(name, {key: round(value, 6) for key, value in aggregate[name].items()}, flush=True)
    changes = {}
    for name in ('neighbors', 'history', 'combined'):
        changes[name] = {key: aggregate[name][key]-aggregate['geometry_control'][key]
                         for key in aggregate[name]}
    result = dict(seeds=seeds, native_cases=report['cases'], aggregate=aggregate,
                  per_seed=by_variant, change_vs_matched_head=changes)
    if 'history_replace' in aggregate:
        result['history_replacement_vs_baseline'] = {
            key: aggregate['history_replace'][key]-aggregate['baseline'][key]
            for key in aggregate['baseline']}
    if 'fusion_trunk' in aggregate:
        result['fusion_vs_baseline'] = {
            name: {key: aggregate[name][key]-aggregate['baseline'][key]
                   for key in aggregate['baseline']} for name in FUSIONS}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report, *additional = [json.loads(path.read_text(encoding='utf-8')) for path in args.evaluations]
    for result in additional:
        if any(report[key] != result[key] for key in ('cases', 'labels', 'active_threshold', 'sector_width')):
            raise ValueError('Evaluation settings differ across reports')
        report['runs'].extend(result['runs'])
    atomic_json(args.out, compare(report))
