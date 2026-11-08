"""Paired fine-tuning of GeometryFNO with or without numerical normal-speed supervision."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .model import GeometryFNO
from ..rollout import evaluation_grid, single_loss, supervised_loss
from ..trajectory_data import DEFAULT_DATA, Trajectories, augment
from ..trajectory_metrics import atomic_json


DEFAULT_LABELS = Path('data/supervised/normal_velocity_labels/training.npz')
RUNS = Path('results/supervised/runs')
REFERENCE_INTERVAL = .025


def load_labels(path, bank):
    """Load certified local updates and reject any overlap with the held-out cases."""
    curves = ('initial', 'state', 'previous', 'target', 'short_target', 'normal')
    times = ('before', 'start', 'future', 'velocity_future')
    fields = (*curves, 'speed', *times, 'case', 'drift')
    with np.load(path, allow_pickle=False) as saved:
        labels = {field: saved[field] for field in fields}
    count = len(labels['case'])
    nodes = bank.curves.shape[-1]
    expected = {field: (count, 2, nodes) for field in curves}
    expected['speed'] = (count, nodes)
    expected.update({field: (count,) for field in (*times, 'case', 'drift')})
    for field, value in labels.items():
        if value.shape != expected[field] or not np.isfinite(value).all():
            raise ValueError(f'Invalid velocity label field {field}: expected finite {expected[field]}')
    allowed = {bank.ids[i] for i in bank.splits['train'].tolist()}
    cases = set(labels['case'].tolist())
    if not cases or not cases.issubset(allowed):
        raise ValueError('Velocity annotations must come from training initial conditions')
    if not np.allclose(np.linalg.norm(labels['normal'], axis=1), 1., atol=1e-3, rtol=0.):
        raise ValueError('Velocity annotations must include unit normal vectors')
    before, start, future, velocity_future = (labels[field] for field in times)
    if np.any((before < 0) | (before > start) | (future > float(bank.times[-1]))):
        raise ValueError('Velocity label time is outside the trajectory window')
    if not np.allclose(future-start, .5, atol=1e-5, rtol=0.):
        raise ValueError('Expected a half-unit prediction interval')
    if not np.allclose(velocity_future-start, .05, atol=1e-5, rtol=0.):
        raise ValueError('Expected a 0.05-unit velocity prediction interval')
    clean = np.flatnonzero(labels['drift'] == 0)
    drift = np.flatnonzero(labels['drift'] > 0)
    if len(clean) == 0 or len(drift) == 0 or len(clean)+len(drift) != count:
        raise ValueError('Expected annotations for clean and model-drifted contours')
    device = bank.times.device
    tensors = {field: torch.as_tensor(labels[field], dtype=torch.float32, device=device)
               for field in fields[:-2]}
    groups = tuple(torch.as_tensor(group, device=device) for group in (clean, drift))
    metadata = dict(samples=count, clean=len(clean), drift=len(drift), case_ids=sorted(cases))
    return tensors, groups, metadata


def local_step_loss(model, labels, groups, batch):
    """Use numerical position labels at both leads and velocity supervision at the shorter lead."""
    clean, drift = groups
    device = labels['start'].device
    n_clean = batch//2
    rows = torch.cat([clean[torch.randint(len(clean), (n_clean,), device=device)],
                      drift[torch.randint(len(drift), (batch-n_clean,), device=device)]])
    fields = ('initial', 'state', 'previous', 'target', 'short_target', 'normal')
    initial, state, previous, target, short_target, normal = (labels[field][rows] for field in fields)
    speed_vector = labels['speed'][rows, None]*normal
    initial, state, previous, target, short_target, normal, speed_vector = augment(
        initial, state, previous, target, short_target, normal, speed_vector)
    before, start, future, velocity_future = (labels[field][rows] for field in
                                             ('before', 'start', 'future', 'velocity_future'))
    # Keep one query per row so multi-query mixing cannot couple the two leads.
    inputs = [torch.cat((value, value)) for value in (initial, state, previous, before)]
    starts = torch.cat((start, start))
    leads = torch.cat((velocity_future, future))
    prediction = model(*inputs, starts, leads[:, None])[:, 0]
    interval = .5*torch.log1p(2*(leads-starts)/(1+2*starts))
    residual = (prediction-torch.cat((short_target, target)))*(REFERENCE_INTERVAL/interval[:, None, None])
    position_loss = supervised_loss(residual, torch.zeros_like(residual))
    # In normalized coordinates the label is the instantaneous derivative in s = log(R).
    # Approximate it using the 0.05-unit displacement, rather than the half-unit displacement.
    predicted_speed = ((prediction[:batch]-state)*normal).sum(dim=1)/interval[:batch, None]
    true_speed = (speed_vector*normal).sum(dim=1)
    speed_mse = (predicted_speed-true_speed).square().mean()
    return position_loss, speed_mse


def fine_tune(model, bank, labels, groups, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=3e-5)
    grid = evaluation_grid(bank.times)
    history = []
    model.train()
    for step in range(1, args.steps+1):
        position_loss = speed_mse = None
        if step % 2:
            loss = single_loss(model, bank, grid, 3000+(step-1)//2, args.batch,
                               long_rollout=True, initial_starts=True)
        else:
            position_loss, speed_mse = local_step_loss(model, labels, groups, args.batch)
            loss = position_loss
            if args.arm == 'velocity':
                loss = loss+args.velocity_weight*REFERENCE_INTERVAL**2*speed_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        history.append([float(loss.detach()),
                        np.nan if position_loss is None else float(position_loss.detach()),
                        np.nan if speed_mse is None else float(speed_mse.detach())])
        if step in (1, args.steps) or step % 200 == 0:
            np.save(args.out/'history.npy', np.asarray(history))
            recent = np.asarray(history[-200:])
            message = f'{args.arm} seed={args.seed} step={step} loss={recent[:, 0].mean():.6g}'
            if step > 1:
                message += (f' local_position={np.nanmean(recent[:, 1]):.6g}'
                            f' diagnostic_speed_mse={np.nanmean(recent[:, 2]):.6g}')
            print(message, flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['control', 'velocity'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--warm', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--velocity-weight', type=float, default=1.)
    args = parser.parse_args()
    if args.steps < 2 or args.batch < 2 or not np.isfinite(args.velocity_weight) or args.velocity_weight <= 0:
        parser.error('Expected at least two steps and samples per batch and a positive finite velocity weight')
    warm = args.warm or RUNS/'tip_density'/f'paired_dense_{args.seed}'/'model.pt'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(warm, map_location=args.device, weights_only=True)
    model = GeometryFNO(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    labels, groups, counts = load_labels(args.labels, bank)
    torch.manual_seed(args.seed)
    history = fine_tune(model, bank, labels, groups, args)
    model.eval()
    metadata = dict(architecture='baseline', method='velocity_labels', arm=args.arm, seed=args.seed,
                    steps=args.steps, velocity_weight=args.velocity_weight if args.arm == 'velocity' else 0.)
    temporary = args.out/'model.tmp'
    torch.save(dict(model=model.state_dict(), config=checkpoint['config'], **metadata), temporary)
    temporary.replace(args.out/'model.pt')
    recent = np.asarray(history[-200:])
    atomic_json(args.out/'training.json', dict(
        **metadata, batch=args.batch, warm=str(warm), labels=str(args.labels), samples=counts,
        reference_interval=REFERENCE_INTERVAL, speed_target='instantaneous normal derivative in log(R)',
        position_leads=[.05, .5], velocity_lead=.05,
        history_columns=['objective', 'local_position_loss', 'local_normal_speed_mse_diagnostic'],
        final_objective=float(recent[:, 0].mean()), final_local_position=float(np.nanmean(recent[:, 1])),
        final_local_speed_mse=float(np.nanmean(recent[:, 2]))))
    print(f'Finished {args.arm} seed={args.seed} with {counts["samples"]} annotated samples', flush=True)


if __name__ == '__main__':
    main()
