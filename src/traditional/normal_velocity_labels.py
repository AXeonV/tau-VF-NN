"""Generate FP64 half-step interface targets and signed normal-speed labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample

from .evolution import evolve
from .late_windows import WINDOWS
from .pressure import BoundaryPressure, CudaBoundaryPressure, derivative
from ..nn_geometry.drift_diagnostic import certify_window, distance, network_step, physical_pair
from ..nn_geometry.model import GeometryFNO
from ..trajectory_data import DEFAULT_DATA
from ..trajectory_metrics import atomic_json, has_crossing


NODES = 1024
PATH_LEVELS = (1024, 1536)
SPEED_LEVELS = (1024, 1536, 2048, 3072)
DT = .5
SHORT_DT = .05
RUNS = Path('results/supervised/runs')
FRESH = Path('data/supervised/drift_diagnostic_fresh')
OUTPUT = Path('data/supervised/normal_velocity_labels')


def load_initials(folder):
    initials = {}
    for path in sorted(folder.glob('case_*.npz')):
        with np.load(path) as item:
            metadata = json.loads(str(item['metadata']))
            case = metadata['case']
            if not metadata['accepted']:
                raise ValueError(f'Uncertified initial case: {path}')
            z = item['z'][0]
            initials[case['index']] = (resample(z, NODES), case['split'])
    return initials


def training_states(model, source, initials):
    windows = json.loads((source/'summary.json').read_text(encoding='utf-8'))['windows']
    for window in windows:
        case = window['case']
        if window['split'] != 'train':
            continue
        initial, split = initials[case]
        if split != 'train' or case >= 128:
            raise ValueError(f'Window contains a nontraining case: {case}')
        with np.load(source/window['file']) as item:
            times = item['times']
            truth = item['z']/np.sqrt(1+2*times[:, None])
        birth, angle = min(window['matched_events'], key=lambda event: abs(event[0]-window['coarse_time']))
        for i, offset in enumerate((-2., -1., 1.)):
            index = int(np.flatnonzero(times == birth+offset).item())
            raw, old = truth[index], truth[index-1]
            context = dict(case=case, window=window['file'], event_angle=angle, event_time=birth,
                           start=float(times[index]))
            yield dict(context, drift=0), initial, raw, old
            steps = 8 if (case+i) % 2 else 4
            if index <= steps:
                steps = 4
            begin = index-steps
            if begin < 1:
                raise ValueError('Window lacks drift history')
            raw, old = truth[begin], truth[begin-1]
            for j in range(begin, index):
                next_state = network_step(model, initial, old, raw, *times[j-1:j+2])
                raw, old = next_state, raw
            yield dict(context, drift=steps), initial, raw, old


def evaluation_states(diagnostics, fresh, device, cache):
    initials = load_initials(fresh)
    clean = {}
    for seed in (20261106, 20261107):
        folder = diagnostics/f'seed_{seed}'
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        for record in report['records']:
            case, name = record['case'], record['window']
            initial, split = initials[case]
            if split != 'test' or case < 128:
                raise ValueError(f'Evaluation case is not a fresh holdout: {case}')
            context = dict(case=case, window=name, seed=seed, event_angle=record['event_angle'],
                           event_time=record['event_time'], start=record['times'][0])
            if name not in clean:
                with np.load(fresh/f'case_{case:03}.npz') as item:
                    result, audit = certify_window(item['times'], item['z'], record['coarse_event'], device, cache)
                if result is None:
                    raise ValueError(f'Cannot reconstruct clean window {name}: {audit}')
                times, curves, event, anchor = result
                if (float(times[anchor]) != context['start']
                        or not np.allclose(event, (context['event_time'], context['event_angle']),
                                           rtol=0, atol=1e-7)):
                    raise ValueError(f'Clean history does not match archived input: {name}')
                clean[name] = curves[anchor], curves[anchor-1]
            if record['drift_steps'] == 4:
                yield dict(context, drift=0), initial, *clean[name]
            with np.load(folder/record['filename']) as item:
                yield dict(context, drift=record['drift_steps']), initial, item['raw_start'], item['raw_previous']


def speed_at(pressure, state, radius):
    tangent = derivative(state)
    normal = -1j*tangent/abs(tangent)
    _, velocity, _, metric = pressure.velocity(state*radius, True)
    speed = radius*velocity-(state*normal.conj()).real
    flux_error = float(abs(np.mean(velocity*metric)-1.))
    return speed, normal, flux_error


def refine_speed(state, radius, pressure):
    prior = None
    flux_error = 0.
    for nodes, solver in pressure.items():
        fine_speed, _, flux = speed_at(solver, resample(state, nodes), radius)
        speed = resample(fine_speed, NODES)
        flux_error = max(flux_error, flux)
        if prior is not None:
            speed_rms = np.sqrt(np.mean(speed**2))
            error = float(np.sqrt(np.mean((speed-prior)**2))/max(speed_rms, 1e-8))
            if error <= .025:
                return speed, speed_rms, error, flux_error, nodes
        prior = speed
    raise ValueError(f'velocity_refinement {error:.4g}')


def finite_step_audit(path, normal, speed, speed_rms, radius):
    projected = ((path[-1]-path[0])*normal.conj()).real/np.log(radius[1]/radius[0])
    error = float(np.sqrt(np.mean((projected-speed)**2))/max(speed_rms, 1e-8))
    denom = np.linalg.norm(projected)*np.linalg.norm(speed)
    cosine = float(np.dot(projected, speed)/denom) if denom > 0 else None
    return error, cosine


def label_state(raw, old, start, pressure, device, cache):
    if has_crossing(raw):
        raise ValueError('input_crossing')
    state, previous = physical_pair(raw, old)
    prep_error = distance(raw, state)
    if prep_error > .001:
        raise ValueError(f'input_preparation {prep_error:.4g}')
    times = np.array([start, start+SHORT_DT, start+DT])
    radius = np.sqrt(1+2*times)
    speed, speed_rms, velocity_error, flux_error, speed_nodes = refine_speed(state, radius[0], pressure)
    levels = []
    for nodes in PATH_LEVELS:
        contour = resample(state, nodes)
        cfl, fraction = (.16, .005) if nodes == PATH_LEVELS[0] else (.12, .0035)
        path, info = evolve(contour*radius[0], times, device=device, cfl=cfl,
                            step_fraction=fraction, output_nodes=NODES, pressure_cache=cache)
        levels.append(path/radius[:, None])
        flux_error = max(flux_error, info['max_flux_relative_error'])
    low, path = levels
    tangent = derivative(state)
    normal = -1j*tangent/abs(tangent)
    if not np.all(np.isfinite(np.r_[speed, path.ravel(), normal])):
        raise ValueError('nonfinite_label')
    target_error = distance(low[-1], path[-1])
    restart_error = max(distance(state, row[0]) for row in levels)
    if max(target_error, restart_error) > .001 or flux_error > 5e-5:
        raise ValueError(f'refinement speed={velocity_error:.4g} target={target_error:.4g} '
                         f'restart={restart_error:.4g} flux={flux_error:.4g}')
    if any(has_crossing(frame) for curves in levels for frame in curves):
        raise ValueError('target_crossing')
    finite_error, motion_cosine = finite_step_audit(path, normal, speed, speed_rms, radius[[0, -1]])
    short_target = path[1]
    short_error, _ = finite_step_audit(path[:2], normal, speed, speed_rms, radius[:2])
    audit_dt = SHORT_DT
    # Sharp model-induced ripples can decay even within .05. Check the
    # instantaneous label with progressively shorter integrations in that case.
    for candidate_dt in (.0025, .00025):
        if np.isfinite(short_error) and short_error <= .12:
            break
        audit_dt = candidate_dt
        short_times = np.array([start, start+audit_dt])
        short_radius = np.sqrt(1+2*short_times)
        short_path, info = evolve(resample(state, speed_nodes)*radius[0], short_times, device=device,
                                  cfl=.12, step_fraction=.0035, max_step=audit_dt/4,
                                  output_nodes=NODES, pressure_cache=cache)
        short_error, _ = finite_step_audit(short_path/short_radius[:, None], normal,
                                           speed, speed_rms, short_radius)
        flux_error = max(flux_error, info['max_flux_relative_error'])
    if not np.isfinite(short_error) or short_error > .12 or flux_error > 5e-5:
        raise ValueError(f'short_step_scale_check {short_error:.4g}')
    audit = dict(velocity_refinement_rms_relative=velocity_error, target_refinement=target_error,
                 input_preparation=prep_error, solver_restart=restart_error, flux_error=flux_error,
                 finite_step_rms_relative=finite_error, finite_step_cosine=motion_cosine,
                 short_step_rms_relative=short_error, speed_audit_dt=audit_dt,
                 speed_nodes=speed_nodes, speed_rms=float(speed_rms))
    return state, previous, path[-1], short_target, speed, normal, audit


def xy(z):
    return np.stack([z.real, z.imag]).astype(np.float32)


def generate(rows, destination, pressure, device, cache):
    accepted, rejected, data = [], [], {}
    for context, initial, raw, old in rows:
        try:
            state, previous, target, short_target, speed, normal, audit = label_state(
                raw, old, context['start'], pressure, device, cache)
        except (RuntimeError, ValueError) as error:
            rejected.append(dict(context, reason=str(error)))
            print(destination.stem, context['case'], context['start'], context['drift'], error, flush=True)
            continue
        before, start, future = context['start']-DT, context['start'], context['start']+DT
        arrays = dict(initial=xy(initial), state=xy(state), previous=xy(previous), target=xy(target),
                      normal=xy(normal), speed=speed.astype(np.float32), short_target=xy(short_target),
                      before=np.float32(before), start=np.float32(start), future=np.float32(future),
                      velocity_future=np.float32(start+SHORT_DT),
                      event_angle=np.float32(context['event_angle']), event_time=np.float32(context['event_time']),
                      case=context['case'], drift=context['drift'])
        if 'seed' in context:
            arrays['seed'] = context['seed']
        for key, value in arrays.items():
            data.setdefault(key, []).append(value)
        accepted.append(dict(context, **audit))
        if len(accepted) % 24 == 0:
            print(destination.stem, len(accepted), 'accepted', len(rejected), 'rejected', flush=True)
    if not accepted:
        raise ValueError(f'No accepted velocity labels for {destination}')
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in data.items()})
    temporary.replace(destination)
    return dict(count=len(accepted), records=accepted, rejected=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--windows', type=Path, default=WINDOWS)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--fresh', type=Path, default=FRESH)
    parser.add_argument('--diagnostics', type=Path, default=RUNS/'drift_diagnostic')
    parser.add_argument('--warm', type=Path, default=RUNS/'tip_density/paired_dense_20261106/model.pt')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--stage', choices=('all', 'training', 'evaluation'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    pressure_type = CudaBoundaryPressure if args.device == 'cuda' else BoundaryPressure
    pressure = {n: pressure_type(n) for n in SPEED_LEVELS}
    cache = {}
    circle = {}
    for nodes, solver in pressure.items():
        state = np.exp(2j*np.pi*np.arange(nodes)/nodes)
        speed, _, flux = speed_at(solver, state, np.sqrt(121.))
        circle[nodes] = dict(max_speed=float(abs(speed).max()), flux_error=flux)
        if circle[nodes]['max_speed'] > 1e-7 or flux > 5e-5:
            raise ValueError('Normalized circle velocity sanity check failed')
    path = args.out/'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    summary.update(path_levels=PATH_LEVELS, speed_levels=SPEED_LEVELS, dt=DT, velocity_dt=SHORT_DT, circle=circle,
                   time_coordinate='s=0.5 log(1+2t)',
                   target='normalized signed normal speed R*v_n - Re(z_normalized*conj(n))')
    if args.stage in ('all', 'training'):
        checkpoint = torch.load(args.warm, map_location=args.device, weights_only=True)
        model = GeometryFNO(**checkpoint['config']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'])
        rows = training_states(model, args.windows, load_initials(args.data))
        summary['training'] = generate(rows, args.out/'training.npz', pressure, args.device, cache)
        atomic_json(path, summary)
        print('Training labels ready:', summary['training']['count'], flush=True)
    if args.stage in ('all', 'evaluation'):
        rows = evaluation_states(args.diagnostics, args.fresh, args.device, cache)
        summary['evaluation'] = generate(rows, args.out/'evaluation.npz', pressure, args.device, cache)
        atomic_json(path, summary)
        print('Evaluation labels ready:', summary['evaluation']['count'], flush=True)


if __name__ == '__main__':
    main()
