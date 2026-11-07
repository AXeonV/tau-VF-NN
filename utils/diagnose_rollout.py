"""Measure one-step skill and error growth from exact saved interfaces."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.nn_geometry.model import GeometryFNO
from src.rollout import forecast
from src.trajectory_data import DEFAULT_DATA, Trajectories
from src.trajectory_metrics import atomic_json, curve_difference, has_crossing


def diagnose(model, bank):
    rows = bank.splits['test']
    frames = torch.arange(len(bank.times), device=bank.times.device)
    times = bank.times.cpu().numpy().astype(float)
    target = bank.curves[rows].cpu().numpy()
    native = forecast(model, bank, rows, frames).cpu().numpy()
    ids = [bank.ids[i] for i in rows.tolist()]
    tests = []
    for start_time in (5., 10., 20.):
        start = int(np.flatnonzero(times == start_time).item())
        reset = forecast(model, bank, rows, frames[start:], previous_frame=frames[start-1]).cpu().numpy()
        requested = [times[start+1], start_time+5, start_time+10, start_time+20, times[-1]]
        probes = np.unique([np.abs(times-time).argmin() for time in requested])
        cases = []
        for row, case in enumerate(ids):
            reference = target[row]
            estimates = reset[row]
            reset_error = np.mean((estimates-reference[start:])**2, axis=(1, 2))
            native_error = np.mean((native[row]-reference)**2, axis=(1, 2))
            one_step_baseline = np.mean((reference[start]-reference[start+1])**2)
            radius = np.sqrt(1+2*times[probes])

            def physical(curves):
                return (curves[:, 0]+1j*curves[:, 1])*radius[:, None]

            exact = physical(reference[probes])
            predicted = physical(estimates[probes-start])
            distances = [curve_difference(p[None], z[None], [t])['maximum']
                         for p, z, t in zip(predicted, exact, times[probes])]
            first_crossing = next((float(t) for t, z in zip(times[start+1:], estimates[1:])
                                   if has_crossing(z[0]+1j*z[1])), None)
            cases.append(dict(case=case, persistence_mse=float(one_step_baseline),
                              first_self_crossing=first_crossing,
                              probes=[dict(time=float(times[i]), reset_mse=float(reset_error[i-start]),
                                           native_mse=float(native_error[i]), geometry_distance=distance)
                                      for i, distance in zip(probes, distances)]))
        averaged = []
        for index, frame in enumerate(probes):
            averaged.append(dict(time=float(times[frame]),
                                 **{metric: float(np.mean([c['probes'][index][metric] for c in cases]))
                                    for metric in ('reset_mse', 'native_mse', 'geometry_distance')}))
        baseline = float(np.mean([c['persistence_mse'] for c in cases]))
        tests.append(dict(start=start_time, next_step=float(times[start+1]-start_time),
                          persistence_mse=baseline, one_step_skill=1-averaged[0]['reset_mse']/baseline,
                          self_crossing_cases=sum(c['first_self_crossing'] is not None for c in cases),
                          probes=averaged, cases=cases))
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(args.data, 1024, args.device, 100.)
    result = {}
    for run in args.runs:
        checkpoint = torch.load(run/'model.pt', map_location=args.device, weights_only=True)
        model = GeometryFNO(**checkpoint['config']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'])
        result[run.name] = dict(step=checkpoint['step'], resets=diagnose(model, bank))
        print(run.name, [(r['start'], round(r['one_step_skill'], 3))
                         for r in result[run.name]['resets']], flush=True)
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
