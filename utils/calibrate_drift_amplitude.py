"""Fit one short-step displacement gain on old training windows, then score fresh diagnostic paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.nn_geometry.drift_diagnostic import distances
from src.nn_geometry.model import GeometryFNO
from src.nn_geometry.tip_density import predict_windows, windows
from src.traditional.late_windows import WINDOWS
from src.trajectory_data import DEFAULT_DATA, Trajectories
from src.trajectory_metrics import area, atomic_json, has_crossing


def complex_curves(x):
    return x[..., 0, :] + 1j * x[..., 1, :]


def normal_displacements(current, prediction, target):
    tangent = np.roll(current, -4, axis=-1) - np.roll(current, 4, axis=-1)
    normal = -1j * tangent / np.maximum(np.abs(tangent), 1e-12)
    predicted = ((prediction-current) * normal.conj()).real
    truth = ((target-current) * normal.conj()).real
    return predicted, truth


def fit_gain(model, training):
    indexes, prediction = predict_windows(model, training, 1, False)
    truth = training[2].cpu().numpy()
    current = complex_curves(truth[:, np.asarray(indexes)-1])
    predicted, target = normal_displacements(
        current, complex_curves(prediction), complex_curves(truth[:, indexes]))
    predicted, target = predicted.astype(np.float64), target.astype(np.float64)
    numerator = np.sum(predicted*target)
    predicted_norm = np.linalg.norm(predicted)
    target_norm = np.linalg.norm(target)
    if min(predicted_norm, target_norm) <= 0:
        raise ValueError('Calibration requires nonzero normal motion')
    gain = numerator / predicted_norm**2
    return dict(gain=float(gain), windows=len(truth), steps=len(indexes),
                normal_amplitude_ratio=float(predicted_norm/target_norm),
                normal_motion_cosine=float(numerator/(predicted_norm*target_norm)),
                normal_mse_raw=float(np.mean((predicted-target)**2)),
                normal_mse_calibrated=float(np.mean((gain*predicted-target)**2)))


def measure(record, archive, arm, gain):
    state = archive[f'physical_{arm}']
    path = archive['teacher_clean' if arm == 'clean' else 'teacher']
    # The first archived teacher step used the preprocessed shared contour.
    current = np.concatenate([path[:1], state[1:-1]])
    prediction = path[1:]
    calibrated = current + gain*(prediction-current)
    calibrated = np.array([z*np.sqrt(area(a)/area(z)) for a, z in zip(current, calibrated)])
    global_before, local_before = distances(prediction, state[1:], record['event_angle'])
    global_after, local_after = distances(calibrated, state[1:], record['event_angle'])
    normal_before, target = normal_displacements(current, prediction, state[1:])
    normal_after, _ = normal_displacements(current, calibrated, state[1:])
    return dict(case=record['case'], window=record['window'], drift_steps=record['drift_steps'],
                global_raw=global_before, global_calibrated=global_after,
                sector_raw=local_before, sector_calibrated=local_after,
                normal_rms_raw=np.sqrt(np.mean((normal_before-target)**2, axis=-1)).tolist(),
                normal_rms_calibrated=np.sqrt(np.mean((normal_after-target)**2, axis=-1)).tolist(),
                raw_crossings=sum(has_crossing(z) for z in prediction),
                calibrated_crossings=sum(has_crossing(z) for z in calibrated))


def summarize(records):
    cases = sorted({r['case'] for r in records})
    result = dict(cases=len(cases), paths=len(records))
    fields = ('global_raw', 'global_calibrated', 'sector_raw', 'sector_calibrated',
              'normal_rms_raw', 'normal_rms_calibrated')
    for field in fields:
        grouped = [[v for r in records if r['case'] == case for v in r[field] if v is not None]
                   for case in cases]
        result[field] = float(np.mean([np.mean(values) for values in grouped if values]))
    return result


def audit(folder, training):
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(report['model'], map_location='cpu', weights_only=True)
    model = GeometryFNO(**checkpoint['config']).to(training[0].device).eval()
    model.load_state_dict(checkpoint['model'])
    fit = fit_gain(model, training)
    clean, drift = [], []
    seen = set()
    for record in report['records']:
        with np.load(folder/record['filename']) as archive:
            drift.append(measure(record, archive, 'drift', fit['gain']))
            if record['window'] not in seen:
                clean.append(measure(record, archive, 'clean', fit['gain']))
                seen.add(record['window'])
    return dict(fit=fit, clean=summarize(clean), drift=summarize(drift),
                clean_paths=clean, drift_paths=drift)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/supervised/runs/drift_diagnostic'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    training = windows(WINDOWS, bank, 'train')
    runs = {}
    for folder in sorted(args.out.glob('seed_*')):
        if not (folder/'report.json').is_file():
            continue
        run = audit(folder, training)
        runs[folder.name] = run
        print(folder.name, run['fit'], run['clean'], run['drift'], flush=True)
    if not runs:
        parser.error('No drift diagnostic seed directories found')
    atomic_json(args.out/'step_amplitude_calibration.json',
                dict(fit_split='old late_tip_density train', aggregation='equal independent shape weight',
                     horizon=.5, seeds=runs))


if __name__ == '__main__':
    main()
