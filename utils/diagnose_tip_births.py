"""Check whether a model predicts persistent tip births from exact recent interfaces."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.interface_events import angle_distance, match_births, tip_births, tips
from src.nn_geometry.model import GeometryFNO
from src.trajectory_data import DEFAULT_DATA, Trajectories
from src.trajectory_metrics import atomic_json


@torch.inference_mode()
def teacher_forced(model, bank, rows):
    predictions = []
    frames = torch.arange(1, len(bank.times), device=bank.times.device)
    for row in rows:
        truth = bank.curves[row]
        parts = [truth[:1]]
        for query in frames.split(16):
            current = query-1
            previous = (query-2).clamp_min(0)
            predicted = model(truth[0].expand(len(query), -1, -1), truth[current], truth[previous],
                              bank.times[previous], bank.times[current], bank.times[query, None])
            parts.append(predicted[:, 0])
        predictions.append(torch.cat(parts))
    return torch.stack(predictions).cpu().numpy()


def audit(bank, rows, runs):
    times = bank.times.cpu().numpy()
    target = bank.curves[rows].cpu().numpy()
    events = []
    cases = []
    for i, row in enumerate(rows.tolist()):
        case = bank.ids[row]
        actual = target[i, :, 0]+1j*target[i, :, 1]
        births = tip_births(actual, times)
        cases.append(dict(case=case, events=len(births), runs={
            name: len(match_births(tip_births(values[i, :, 0]+1j*values[i, :, 1], times), births))
            for name, values in runs.items()}))
        for time, angle in births:
            frame = int(np.searchsorted(times, time))
            previous = actual[frame-1]
            local = angle_distance(np.angle(actual[frame]), angle) <= .15
            persistence = np.mean(abs(previous[local]-actual[frame, local])**2)
            scores = {}
            for name, values in runs.items():
                predicted = values[i, frame, 0]+1j*values[i, frame, 1]
                error = np.mean(abs(predicted[local]-actual[frame, local])**2)
                scores[name] = dict(local_mse=float(error), beats_persistence=bool(error < persistence),
                                    has_tip_at_birth=bool(np.any(
                                        angle_distance(np.angle(tips(predicted)), angle) <= .12)))
            events.append(dict(case=case, time=time, angle=angle,
                               dt=float(times[frame]-times[frame-1]), persistence_mse=float(persistence),
                               runs=scores))
    return dict(cases=cases, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    rows = bank.splits['validation']
    predictions = {}
    for folder in args.runs:
        checkpoint = torch.load(folder/'model.pt', map_location=args.device, weights_only=True)
        model = GeometryFNO(**checkpoint['config']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'])
        predictions[folder.name] = teacher_forced(model, bank, rows)
        print(folder.name, 'validation predictions complete', flush=True)
    result = audit(bank, rows, predictions)
    atomic_json(args.output, result)
    print(len(result['events']), 'validation tip births', flush=True)


if __name__ == '__main__':
    main()
