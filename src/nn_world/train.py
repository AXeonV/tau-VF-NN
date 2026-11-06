"""Train paired T=20 ablations with the same frozen global proposal."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.signal import resample

from .model import FutureTokens, Repair
from .proposals import sample, shape_loss
from ..nn.model import signed_area
from ..trajectory_data import Trajectories, augment
from ..trajectory_metrics import _distance_to_curve, atomic_json, probe_frames, report_predictions


STRUCTURE = {
    'point': (False, False, False, False),
    'local': (True, False, False, False),
    'near': (True, True, False, False),
    'band': (True, True, True, False),
    'jepa': (True, True, True, True),
}
BANDS = (slice(1, 9), slice(9, 33), slice(33, 97))


def complex_curves(x):
    return x[..., 0, :]+1j*x[..., 1, :]


def frame_errors(predictions, targets):
    forecasts = resample(complex_curves(predictions), 2*predictions.shape[-1], axis=-1)
    labels = resample(complex_curves(targets), forecasts.shape[-1], axis=-1)
    distances = [max(_distance_to_curve(x, y).max(), _distance_to_curve(y, x).max())
                 for x, y in zip(forecasts.reshape(-1, forecasts.shape[-1]), labels.reshape(-1, labels.shape[-1]))]
    return np.asarray(distances).reshape(forecasts.shape[:2])


@torch.no_grad()
def forecast(model, bank, proposals, split, frames=None, iterations=1):
    rows = bank.splits[split]
    if frames is None:
        frames = torch.arange(len(bank.times), device=rows.device)
    if model is None:
        return proposals[rows[:, None], frames].cpu().numpy()
    output = []
    for r, c in zip(rows.repeat_interleave(len(frames)).split(16), frames.repeat(len(rows)).split(16)):
        proposal = proposals[r, c]
        for _ in range(iterations):
            proposal = model(bank.curves[r, 0], proposal, bank.times[c])
        output.append(proposal.cpu().numpy())
    return np.concatenate(output).reshape(len(rows), len(frames), 2, -1)


def spectral_loss(residual, scales):
    spectrum = torch.fft.rfft(residual, norm='forward').abs().square()
    return sum(spectrum[..., band].sum(dim=-1).mean()/scale for band, scale in zip(BANDS, scales))


def summarize(path, model, bank, full, iterations=1):
    validation = forecast(model, bank, full, 'validation', iterations=iterations)
    test = forecast(model, bank, full, 'test', iterations=iterations)
    val = report_predictions(bank, 'validation', validation)
    test_metrics = report_predictions(bank, 'test', test, export=path/'test_predictions.npz')
    for name, predicted, metrics in [('validation', validation, val), ('test', test, test_metrics)]:
        target = bank.curves[bank.splits[name]].cpu().numpy()
        spectrum = np.abs(np.fft.rfft(predicted[:, -1]-target[:, -1], axis=-1, norm='forward'))**2
        metrics['final_band_rmse'] = [float(np.sqrt(2*spectrum[..., band].sum(axis=-1).mean())) for band in BANDS]
    return dict(validation=val, test=test_metrics)


def fit(name, structure, bank, full, path, steps, seed, oof=None, noise_power=None, *,
        iterations=1, input_mode=None):
    started = perf_counter()
    local, near, bands, tokens = STRUCTURE[structure]
    torch.manual_seed(seed)
    model = Repair(local=local, neighbors=near).cuda()
    auxiliary = FutureTokens().cuda() if tokens else None
    params = list(model.parameters())+(list(auxiliary.parameters()) if tokens else [])
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=5e-5)
    train = bank.splits['train']
    target_spectrum = torch.fft.rfft(bank.curves[train], norm='forward').abs().square()
    scales = [(target_spectrum[..., band].sum(dim=-1).mean()/.01).clamp_min(.02) for band in BANDS]
    future_cols = torch.searchsorted(bank.times, bank.times+.20*(1+bank.times)).clamp_max(len(bank.times)-1)
    probes = probe_frames(bank.times)
    val_target = bank.curves[bank.splits['validation'][:, None], probes].cpu().numpy()
    best, history = float('inf'), []
    noise_rng = torch.Generator(device='cuda').manual_seed(seed+1)
    for step in range(1, steps+1):
        row, col = sample(bank, train, 12)
        target = bank.curves[row, col]
        proposal = full[row, col]
        mode = input_mode or name
        if mode in ('noise', 'oof'):
            shape = noise_power[col].shape
            noise = torch.randn(shape, dtype=torch.complex64, device=proposal.device,
                                generator=noise_rng)*noise_power[col]
            synthetic = target+torch.fft.irfft(noise, n=target.shape[-1], norm='forward')
            area = signed_area(target)
            synthetic = synthetic*torch.sqrt(area/torch.maximum(signed_area(synthetic), .05*area))[:, None, None]
            proposal = synthetic if mode == 'noise' else torch.where(
                torch.rand(len(row), 1, 1, device=row.device, generator=noise_rng) < .25, synthetic, oof[row, col])
        initial, proposal, target, future = augment(bank.curves[row, 0], proposal, target,
                                                   bank.curves[row, future_cols[col]])
        time = bank.times[col]
        losses = []
        for _ in range(iterations):
            prediction, features = model(initial, proposal, time, return_features=True)
            residual = prediction-target
            geometric_loss = shape_loss(residual)
            if bands:
                geometric_loss = geometric_loss+.3*spectral_loss(residual, scales)
            losses.append(geometric_loss)
            proposal = prediction
        auxiliary_loss = (auxiliary(features, future, (bank.times[future_cols[col]]-time)/20)
                          if tokens else time.new_zeros(()))
        loss = losses[-1]+.02*auxiliary_loss
        if iterations > 1:
            loss = loss+.3*torch.stack(losses[:-1]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if tokens:
            auxiliary.update()
        if step % 200 == 0 or step in (1, steps):
            forecast_val = forecast(model, bank, full, 'validation', probes, iterations)
            errors = frame_errors(forecast_val, val_target)
            # Equal weight to the terminal error and the average probe error.
            score = float(.5*errors.mean()+.5*errors[:, -1].mean())
            history.append([step, float(loss.detach()), float(np.mean((forecast_val-val_target)**2)),
                            score, float(auxiliary_loss.detach())])
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(), structure=structure, step=step, seed=seed),
                           path/'model.pt')
            np.save(path/'history.npy', np.asarray(history))
            print(f'{name} step={step} val={score:.5f} loss={history[-1][1]:.5g} '
                  f't={perf_counter()-started:.1f}s', flush=True)
    checkpoint = torch.load(path/'model.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    metrics = summarize(path, model, bank, full, iterations)
    metrics.update(best_step=checkpoint['step'], steps=steps, validation_selection=best, structure=structure,
                   seed=seed, repair_parameters=sum(p.numel() for p in model.parameters()),
                   seconds=perf_counter()-started)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/world_t20'))
    parser.add_argument('--cache', type=Path, help='Reuse proposal caches from a previous run')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--stage', choices=['structure', 'repair', 'unroll'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache or args.out
    cache = torch.load(cache_dir/'full.pt', weights_only=True)
    bank = Trajectories(args.data, cache['full'].shape[-1], 'cuda', 20.)
    if cache['ids'] != bank.ids or not torch.equal(cache['times'], bank.times.cpu()):
        raise ValueError('Proposal cache does not match the dataset')
    full = cache['full'].cuda()
    if args.stage == 'structure':
        for name in ('nn', *STRUCTURE):
            path = args.out/name
            path.mkdir(exist_ok=True)
            result = summarize(path, None, bank, full) if name == 'nn' else fit(
                name, name, bank, full, path, args.steps, args.seed)
            atomic_json(path/'metrics.json', result)
            print(name, 'final', result['test']['mean_final_geometry_distance'], flush=True)
        return

    folded = torch.load(cache_dir/'oof.pt', weights_only=True)
    oof = folded['oof'].cuda()
    train = bank.splits['train']
    train_ids = {bank.ids[i] for i in train.tolist()}
    predicted_ids = set()
    for fold in folded['folds']:
        fit_ids, held_ids = set(fold['fit']), set(fold['predicted'])
        if fit_ids & held_ids or (fit_ids | held_ids) != train_ids or held_ids & predicted_ids:
            raise ValueError('Out-of-fold predictions must exclude the predicted and held-out cases')
        predicted_ids.update(held_ids)
    if predicted_ids != train_ids or not oof[train].isfinite().all():
        raise ValueError('Missing out-of-fold predictions')
    noise_power = torch.fft.rfft(oof[train]-bank.curves[train], norm='forward').abs().square().mean(dim=0).sqrt()
    if args.stage == 'unroll':
        summary_path = args.out/'summary.json'
        summary = json.loads(summary_path.read_text())
        name = 'unroll'
        path = args.out/name
        path.mkdir(exist_ok=True)
        source = summary['repeated_source']
        result = fit(name, summary['repair_structure'], bank, full, path, args.steps, args.seed, oof,
                     noise_power, iterations=3, input_mode=source)
        result.update(iterations=3, input_source=source)
        atomic_json(path/'metrics.json', result)
        summary['results'][name] = result
        chosen = summary['selected']
        if result['validation']['mean_final_geometry_distance'] < summary['results'][chosen]['validation']['mean_final_geometry_distance']:
            summary['selected'] = name
        atomic_json(summary_path, summary)
        return

    results = {name: json.loads((args.out/name/'metrics.json').read_text()) for name in ('nn', *STRUCTURE)}
    # Hold architecture fixed while varying the repair-input distribution.
    structure = min(STRUCTURE, key=lambda name: results[name]['validation_selection'])
    for name in ('noise', 'oof'):
        path = args.out/name
        path.mkdir(exist_ok=True)
        results[name] = fit(name, structure, bank, full, path, args.steps, args.seed, oof, noise_power)
        atomic_json(path/'metrics.json', results[name])
    contenders = [structure, 'noise', 'oof']
    source = min(contenders, key=lambda name: results[name]['validation']['mean_final_geometry_distance'])
    local, near, _, _ = STRUCTURE[structure]
    model = Repair(local=local, neighbors=near).cuda().eval()
    model.load_state_dict(torch.load(args.out/source/'model.pt', map_location='cuda', weights_only=True)['model'])
    for count in (2, 3):
        name = f'{source}_x{count}'
        path = args.out/name
        path.mkdir(exist_ok=True)
        results[name] = summarize(path, model, bank, full, count)
        results[name].update(source=source, iterations=count)
        atomic_json(path/'metrics.json', results[name])
        contenders.append(name)
    winner = min(['nn', *contenders], key=lambda name: results[name]['validation']['mean_final_geometry_distance'])
    atomic_json(args.out/'summary.json', dict(horizon=20, frames=len(bank.times),
                splits={key: [bank.ids[i] for i in rows.tolist()] for key, rows in bank.splits.items()},
                fold_steps=folded['steps'], folds=folded['folds'], repair_structure=structure,
                repeated_source=source, selected=winner, results=results))
    print('selected', winner, flush=True)


if __name__ == '__main__':
    main()
