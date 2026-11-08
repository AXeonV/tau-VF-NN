"""Pair a normal-speed residual head with the same GeometryFNO fine-tuning control."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn

from .drift_diagnostic import SECTOR, distances, network_paths, physical_pair
from .model import GeometryFNO, frame, smooth
from .tip_density import local_loss, windows
from ..fourier import signed_area
from ..interface_events import match_births, tip_births
from ..rollout import evaluation_grid, forecast, single_loss
from ..traditional.late_windows import WINDOWS
from ..trajectory_data import DEFAULT_DATA, Trajectories
from ..trajectory_metrics import atomic_json, has_crossing, report_predictions


RUNS = Path('results/supervised/runs')
FRESH = Path('data/supervised/drift_diagnostic_fresh')


class NormalSpeedFNO(GeometryFNO):
    """Add an area-preserving normal residual in the scale-normalized time coordinate."""

    def __init__(self, **config):
        super().__init__(**config)
        self.normal_speed = nn.Sequential(
            nn.Conv1d(6, 24, 1), nn.GELU(),
            nn.Conv1d(24, 24, 9, padding=4, padding_mode='circular'), nn.GELU(),
            nn.Conv1d(24, 1, 1),
        )
        nn.init.zeros_(self.normal_speed[-1].weight)
        nn.init.zeros_(self.normal_speed[-1].bias)

    def forward(self, initial, state, previous, before, start, queries):
        prediction = super().forward(initial, state, previous, before, start, queries)
        _, normal = frame(smooth(state, 64.))
        time = .5*torch.log1p(2*start)
        elapsed = time-.5*torch.log1p(2*before)
        movement = ((state-previous)*normal).sum(dim=1, keepdim=True)
        elapsed = elapsed[:, None, None]
        history = torch.where(elapsed > 1e-5, movement/elapsed.clamp_min(1e-5), 0.).clamp(-2., 2.)
        if not self.history:
            history = torch.zeros_like(history)
        age = (start/self.horizon)[:, None, None].expand(-1, 1, state.shape[-1])
        anchor = initial if self.anchor else torch.zeros_like(initial)
        speed = self.normal_speed(torch.cat([state, anchor, history, age], dim=1))
        duration = .5*torch.log1p(2*queries)-time[:, None]
        corrected = prediction+duration[:, :, None, None]*(speed*normal)[:, None]
        base_area = signed_area(prediction.flatten(0, 1))
        corrected_area = signed_area(corrected.flatten(0, 1))
        factor = torch.sqrt(base_area/torch.maximum(corrected_area, .1*base_area))
        corrected = corrected*factor.reshape(*duration.shape, 1, 1)
        return torch.where((duration == 0)[:, :, None, None], state[:, None], corrected)


def fine_tune(model, bank, training, steps, out):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=3e-5)
    grid = evaluation_grid(bank.times)
    losses = []
    model.train()
    for step in range(1, steps+1):
        if step % 2:
            loss = single_loss(model, bank, grid, 3000+(step-1)//2, 8,
                               long_rollout=True, initial_starts=True)
        else:
            coarse = local_loss(model, training, False, 8)
            short = local_loss(model, training, True, 8, rate_balanced=True)
            loss = (coarse+short)/2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if step in (1, steps) or step % 200 == 0:
            np.save(out/'history.npy', np.array(losses))
            print(f'step={step} loss={np.mean(losses[-200:]):.6g}', flush=True)


def score_drift(model, bank, diagnostic):
    report = json.loads((diagnostic/'report.json').read_text(encoding='utf-8'))
    records = []
    for record in report['records']:
        with np.load(diagnostic/record['filename']) as saved:
            times, physical, nominal = (saved[key] for key in ('times', 'physical_drift', 'physical_clean'))
            state, previous = physical_pair(saved['raw_start'], saved['raw_previous'])
        initial = bank.curves[bank.ids.index(record['case']), 0].cpu().numpy()
        initial = initial[0]+1j*initial[1]
        free, teacher, _ = network_paths(model, initial, previous, state, physical, times)
        angle = record['event_angle']
        global_free, sector_free = distances(free, physical, angle)
        global_teacher, sector_teacher = distances(teacher, physical, angle)
        physical_events = record['events']['physical_drift']['found']
        free_events = tip_births(free, times)
        records.append(dict(case=record['case'], window=record['window'], drift_steps=record['drift_steps'],
                            time_offsets=(times-times[0]).tolist(),
                            teacher=global_teacher, teacher_sector=sector_teacher,
                            free=global_free, free_sector=sector_free,
                            free_nominal=distances(free, nominal, angle)[0],
                            branch_matches=len(match_births(free_events, physical_events,
                                                             time_limit=1.5, angle_limit=SECTOR)),
                            physical_branches=len(physical_events),
                            crossing_frames=sum(has_crossing(z) for z in free)))
    if not records:
        raise ValueError('Expected certified archived drift inputs')
    by_case = [[r for r in records if r['case'] == case] for case in sorted({r['case'] for r in records})]
    summary = {}
    for key in ('teacher', 'teacher_sector', 'free', 'free_sector', 'free_nominal'):
        summary[key] = []
        for i in range(len(records[0][key])):
            valid = []
            for group in by_case:
                values = [r[key][i] for r in group if r[key][i] is not None]
                if values:
                    valid.append(mean(values))
            summary[key].append(mean(valid) if valid else None)
    return dict(cases=len(by_case), paths=len(records), mean_errors=summary,
                branch_matches=sum(r['branch_matches'] for r in records),
                physical_branches=sum(r['physical_branches'] for r in records),
                crossing_paths=sum(r['crossing_frames'] > 0 for r in records), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['baseline', 'normal_speed'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--warm', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one training step')
    warm = args.warm or RUNS/'tip_density'/f'paired_dense_{args.seed}'/'model.pt'
    diagnostic = args.diagnostic or RUNS/'drift_diagnostic'/f'seed_{args.seed}'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(warm, map_location=args.device, weights_only=True)
    config = checkpoint['config']
    kind = GeometryFNO if args.architecture == 'baseline' else NormalSpeedFNO
    torch.manual_seed(args.seed)
    model = kind(**config).to(args.device)
    missing, extra = model.load_state_dict(checkpoint['model'], strict=False)
    expected = {name for name, _ in model.named_parameters() if name.startswith('normal_speed.')}
    if extra or set(missing) != expected:
        raise ValueError('Incompatible warm-start checkpoint')
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    training = windows(WINDOWS, bank, 'train')
    if len(training[-1]) != 42:
        raise ValueError('Expected the 42 training windows in the paired experiment')
    # Reset after constructing the larger head so both arms sample the same training clips.
    torch.manual_seed(args.seed)
    fine_tune(model, bank, training, args.steps, args.out)
    model.eval()
    torch.save(dict(model=model.state_dict(), config=config, architecture=args.architecture,
                    seed=args.seed, steps=args.steps), args.out/'model.pt')
    fresh = Trajectories(FRESH, 1024, args.device, 100.)
    drift = score_drift(model, fresh, diagnostic)
    frames = torch.arange(len(fresh.times), device=fresh.times.device)
    predicted = forecast(model, fresh, fresh.splits['test'], frames).cpu().numpy()
    native = report_predictions(fresh, 'test', predicted)
    result = dict(architecture=args.architecture, seed=args.seed, steps=args.steps, warm=str(warm),
                  diagnostic=str(diagnostic), training_windows=len(training[-1]),
                  parameters=sum(p.numel() for p in model.parameters()), drift=drift, native=native)
    atomic_json(args.out/'metrics.json', result)
    print(f'Finished {args.architecture} seed={args.seed}: '
          f'teacher(.5)={drift["mean_errors"]["teacher"][1]:.5g} '
          f'free(4)={drift["mean_errors"]["free"][8]:.5g} '
          f'native(100)={native["mean_final_geometry_distance"]:.5g}', flush=True)


if __name__ == '__main__':
    main()
