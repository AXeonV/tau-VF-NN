"""Score held-out velocity labels, same-state restarts, and native rollouts."""
import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from src.nn_geometry import checkpoint_model
from src.nn_geometry.normal_speed_probe import FRESH, RUNS, score_drift
from src.rollout import forecast
from src.trajectory_data import Trajectories
from src.trajectory_metrics import atomic_json, curve_difference, report_predictions


LABELS = Path('data/supervised/normal_velocity_labels/evaluation.npz')
DIAGNOSTIC = RUNS/'drift_diagnostic'
CURVES = ('initial', 'state', 'previous', 'normal')
TIMES = ('before', 'start', 'future')


def load_labels(path):
    with np.load(path) as item:
        labels = {key: item[key] for key in item.files}
    count = len(labels['case'])
    if not count:
        raise ValueError('Expected held-out velocity samples')
    curve_shape = (count, 2, 1024)
    for key in (*CURVES, 'target', 'short_target'):
        if key not in CURVES and key not in labels:
            continue
        if labels[key].shape != curve_shape or not np.isfinite(labels[key]).all():
            raise ValueError(f'Invalid {key} curves in {path}')
    for key in (*TIMES, 'case', 'drift', 'seed'):
        if labels[key].shape != (count,) or not np.isfinite(labels[key]).all():
            raise ValueError(f'Invalid {key} array in {path}')
    if labels['speed'].shape != (count, 1024) or not np.isfinite(labels['speed']).all():
        raise ValueError('Invalid speed labels')
    if np.any(labels['before'] < 0) or np.any(labels['before'] > labels['start']):
        raise ValueError('Expected nonnegative history no later than the current time')
    if not np.allclose(labels['future'] - labels['start'], .5, rtol=0, atol=1e-6):
        raise ValueError('Expected half-step future queries')
    if 'velocity_future' in labels:
        future = labels['velocity_future']
        if (future.shape != (count,) or not np.isfinite(future).all()
                or not np.allclose(future - labels['start'], .05, rtol=0, atol=1e-5)):
            raise ValueError('Expected a short-step velocity query per sample')
    elif 'short_target' in labels:
        raise ValueError('Short-step targets require velocity_future')
    if not np.allclose(np.linalg.norm(labels['normal'], axis=1), 1., rtol=0, atol=1e-3):
        raise ValueError('Expected unit normals')
    if 'event_angle' in labels and labels['event_angle'].shape != (count,):
        raise ValueError('Expected an event angle per sample')
    return labels


def as_complex(points):
    return points[:, 0] + 1j * points[:, 1]


def contour_errors(prediction, target):
    return np.array([curve_difference(a[None], b[None], [0.])['maximum']
                     for a, b in zip(as_complex(prediction), as_complex(target))])


def projected_speed(next_shape, labels, future_key='future'):
    """Use the current outward normals to project the queried displacement."""
    start, future = (labels[key].astype(np.float64) for key in ('start', future_key))
    dt = .5 * np.log1p(2 * (future - start) / (1 + 2 * start))
    displacement = ((next_shape - labels['state']) * labels['normal']).sum(axis=1)
    return displacement / dt[:, None]


def masked_mean(values, mask):
    return float(np.mean(values[mask])) if mask.any() else None


def masked_rmse(errors, mask):
    squared = masked_mean(errors**2, mask)
    return None if squared is None else float(np.sqrt(squared))


def summarize_velocity(prediction, labels, threshold, sector_width, velocity_prediction=None):
    truth = labels['speed']
    half_speed = projected_speed(prediction, labels)
    estimate = (half_speed if velocity_prediction is None else
                projected_speed(velocity_prediction, labels, 'velocity_future'))
    active = np.abs(truth) >= threshold
    correct_sign = np.sign(estimate) == np.sign(truth)
    errors = estimate - truth
    sector = None
    if 'event_angle' in labels:
        angles = np.angle(as_complex(labels['state']))
        event = labels['event_angle'][:, None]
        known = np.isfinite(event)
        offset = angles - np.where(known, event, 0.)
        sector = known & (np.abs(np.angle(np.exp(1j * offset))) <= sector_width)
    if 'target' in labels:
        finite_speed = projected_speed(labels['target'], labels)
        geometric = contour_errors(prediction, labels['target'])
    if 'short_target' in labels:
        short_speed = projected_speed(labels['short_target'], labels, 'velocity_future')

    def measure(rows):
        here = np.broadcast_to(rows[:, None], truth.shape)
        active_here = active & here
        result = dict(velocity_rmse=masked_rmse(errors, here),
                      active_sign_accuracy=masked_mean(correct_sign, active_here),
                      half_step_velocity_rmse=masked_rmse(half_speed - truth, here),
                      active_fraction=masked_mean(active, here))
        if sector is not None:
            local = here & sector
            result.update(sector_velocity_rmse=masked_rmse(errors, local),
                          sector_active_sign_accuracy=masked_mean(correct_sign, local & active))
        if 'target' in labels:
            result.update(finite_projection_rmse=masked_rmse(half_speed - finite_speed, here),
                          finite_vs_instantaneous_rmse=masked_rmse(finite_speed - truth, here),
                          target_contour_distance=float(np.mean(geometric[rows])))
            if sector is not None:
                result.update(sector_finite_projection_rmse=masked_rmse(half_speed - finite_speed, local),
                              sector_finite_vs_instantaneous_rmse=masked_rmse(finite_speed - truth, local))
        if 'short_target' in labels:
            result.update(short_projection_rmse=masked_rmse(estimate - short_speed, here),
                          short_vs_instantaneous_rmse=masked_rmse(short_speed - truth, here))
            if sector is not None:
                result.update(sector_short_projection_rmse=masked_rmse(estimate - short_speed, local),
                              sector_short_vs_instantaneous_rmse=masked_rmse(short_speed - truth, local))
        return result

    groups = dict(all=np.ones(len(truth), dtype=bool),
                  clean=labels['drift'] == 0, drift=labels['drift'] > 0)
    summary = {}
    for name, rows in groups.items():
        per_shape = []
        for case in np.unique(labels['case'][rows]):
            subset = rows & (labels['case'] == case)
            per_shape.append(dict(case=int(case), samples=int(subset.sum()), **measure(subset)))
        average = {}
        if per_shape:
            for key in sorted(per_shape[0].keys() - {'case', 'samples'}):
                values = [item[key] for item in per_shape if item[key] is not None]
                average[key] = mean(values) if values else None
        summary[name] = dict(samples=int(rows.sum()), shapes=len(per_shape),
                             mean_by_shape=average, per_shape=per_shape)
    return summary


@torch.inference_mode()
def velocity_scores(model, archive, seed, device, batch, threshold, sector_width):
    rows = archive['seed'] == seed
    if not rows.any():
        raise ValueError(f'No velocity labels for checkpoint seed {seed}')
    labels = {key: value[rows] for key, value in archive.items()
              if key in (*CURVES, *TIMES, 'speed', 'target', 'case', 'drift', 'event_angle',
                         'short_target', 'velocity_future')}
    queries = ['future', 'velocity_future'] if 'velocity_future' in labels else ['future']
    predictions = {key: [] for key in queries}
    for i in range(0, int(rows.sum()), batch):
        span = slice(i, i + batch)
        args = [torch.as_tensor(labels[key][span], dtype=torch.float32, device=device)
                for key in ('initial', 'state', 'previous', 'before', 'start')]
        for key in queries:
            future = torch.as_tensor(labels[key][span, None], dtype=torch.float32, device=device)
            predictions[key].append(model(*args, future)[:, 0].cpu().numpy())
    joined = {key: np.concatenate(value) for key, value in predictions.items()}
    if not all(np.isfinite(value).all() for value in joined.values()):
        raise RuntimeError('Nonfinite velocity prediction')
    return summarize_velocity(joined['future'], labels, threshold, sector_width,
                              joined.get('velocity_future'))


def evaluate(args):
    archive = load_labels(args.labels)
    bank = Trajectories(args.data, 1024, args.device, 100.)
    rows = bank.splits['test']
    ids = [bank.ids[i] for i in rows.tolist()]
    if not ids or not set(np.unique(archive['case'])).issubset(ids):
        raise ValueError('Velocity labels must come from the fresh test split')
    frames = torch.arange(len(bank.times), device=bank.times.device)
    results = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=args.device, weights_only=True)
        architecture = checkpoint.get('architecture', 'baseline')
        kind = checkpoint_model(architecture)
        model = kind(**checkpoint['config']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'])
        seed = checkpoint['seed']
        velocity = velocity_scores(model, archive, seed, args.device, args.batch,
                                   args.active_threshold, args.sector_width)
        drift = score_drift(model, bank, args.diagnostic/f'seed_{seed}')
        native = report_predictions(bank, 'test', forecast(model, bank, rows, frames).cpu().numpy())
        result = dict(checkpoint=str(path), architecture=architecture, seed=seed,
                      method=checkpoint.get('method'), arm=checkpoint.get('arm', 'warm'),
                      steps=checkpoint.get('steps'), velocity_query_dt=.05 if 'velocity_future' in archive else .5,
                      velocity=velocity, drift=drift, native=native)
        fallback_variant = ('history_replace' if architecture == 'exact_history'
                            else checkpoint['config'].get('physics_inputs'))
        variant = checkpoint.get('variant', fallback_variant)
        if variant is not None:
            result['variant'] = variant
        results.append(result)
        whole = velocity['all']['mean_by_shape']
        print(f'{path} speed_RMSE={whole["velocity_rmse"]:.6g} '
              f'active_sign={whole["active_sign_accuracy"]} '
              f'drift_4={drift["mean_errors"]["free"][8]:.6g} '
              f'T100={native["mean_final_geometry_distance"]:.6g}', flush=True)
        atomic_json(args.out, dict(labels=str(args.labels), cases=ids,
                                   active_threshold=args.active_threshold,
                                   sector_width=args.sector_width, runs=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--labels', type=Path, default=LABELS)
    parser.add_argument('--data', type=Path, default=FRESH)
    parser.add_argument('--diagnostic', type=Path, default=DIAGNOSTIC)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--active-threshold', type=float, default=.05)
    parser.add_argument('--sector-width', type=float, default=.24)
    args = parser.parse_args()
    if (args.batch < 1 or not np.isfinite(args.active_threshold) or args.active_threshold <= 0
            or not 0 < args.sector_width <= np.pi):
        parser.error('Expected positive batch size, active threshold and angular sector width')
    torch.set_num_threads(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    evaluate(args)
