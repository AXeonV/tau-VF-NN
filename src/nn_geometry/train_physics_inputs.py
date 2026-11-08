"""Compare matched geometry-head and physics-input variants under velocity supervision."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .model import GeometryFNO
from .train_velocity_labels import DEFAULT_LABELS, REFERENCE_INTERVAL, RUNS, fine_tune, load_labels
from ..trajectory_data import DEFAULT_DATA, Trajectories
from ..trajectory_metrics import atomic_json


FUSIONS = ('fusion_trunk', 'fusion_geometry', 'fusion_joint')
VARIANTS = ('baseline', 'geometry_control', 'neighbors', 'history', 'combined', 'history_replace', *FUSIONS)


def initialize(checkpoint, variant):
    config = dict(checkpoint['config'])
    if variant == 'history_replace':
        from .physics_inputs import ExactHistoryFNO

        model = ExactHistoryFNO(**config)
        model.load_state_dict(checkpoint['model'])
        return model, config

    baseline = GeometryFNO(**config)
    baseline.load_state_dict(checkpoint['model'])
    if variant == 'baseline':
        return baseline, config
    if variant in FUSIONS:
        from .physics_fusion import PhysicsFusionFNO

        config['fusion'] = variant.removeprefix('fusion_')
        model = PhysicsFusionFNO(**config)
    else:
        from .physics_inputs import PhysicsInputFNO

        config['physics_inputs'] = variant
        model = PhysicsInputFNO(**config)
    additions = set(model.state_dict()) - set(baseline.state_dict())
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    if not additions or set(missing) != additions or unexpected:
        raise ValueError(f'{variant} does not share the warm-start GeometryFNO parameters')
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
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
    torch.manual_seed(args.seed)
    checkpoint = torch.load(warm, map_location='cpu', weights_only=True)
    model, config = initialize(checkpoint, args.variant)
    model = model.to(args.device)
    bank = Trajectories(DEFAULT_DATA, 1024, args.device, 100.)
    labels, groups, counts = load_labels(args.labels, bank)

    # Preserve the paired sampling stream regardless of how many parameters were initialized.
    torch.manual_seed(args.seed)
    args.arm = 'velocity'
    history = fine_tune(model, bank, labels, groups, args)
    model.eval()
    architecture = {'baseline': 'baseline', 'history_replace': 'exact_history'}.get(
        args.variant, 'physics_inputs')
    if args.variant in FUSIONS:
        architecture = 'physics_fusion'
    metadata = dict(architecture=architecture, method='physics_inputs', variant=args.variant,
                    arm=args.arm, seed=args.seed, steps=args.steps, velocity_weight=args.velocity_weight)
    temporary = args.out/'model.tmp'
    torch.save(dict(model=model.state_dict(), config=config, **metadata), temporary)
    temporary.replace(args.out/'model.pt')

    recent = np.asarray(history[-200:])
    atomic_json(args.out/'training.json', dict(
        **metadata, batch=args.batch, warm=str(warm), labels=str(args.labels), samples=counts,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        reference_interval=REFERENCE_INTERVAL, position_leads=[.05, .5], velocity_lead=.05,
        history_columns=['objective', 'local_position_loss', 'local_normal_speed_mse_diagnostic'],
        final_objective=float(recent[:, 0].mean()), final_local_position=float(np.nanmean(recent[:, 1])),
        final_local_speed_mse=float(np.nanmean(recent[:, 2]))))
    print(f'Finished {args.variant} seed={args.seed} with {counts["samples"]} annotated samples', flush=True)


if __name__ == '__main__':
    main()
