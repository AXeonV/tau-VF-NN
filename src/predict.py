"""Predict a held-out interface with a trained interface model."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .nn_block.model import BlockFNO
from .nn_boundary.model import BoundaryFNO
from .nn_direct.model import CurveFNO
from .nn_geometry.model import GeometryFNO
from .rollout import evaluation_grid, forecast
from .trajectory_data import DEFAULT_DATA, Trajectories


MODELS = dict(nn_direct=CurveFNO, nn_block=BlockFNO, nn_geometry=GeometryFNO, nn_boundary=BoundaryFNO)


def load_model(architecture, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('initial_frame'):
        raise ValueError(f'{path} requires the retired initial-frame inference mode')
    tag = checkpoint.get('architecture')
    if tag == 'baseline' and architecture == 'nn_geometry':
        tag = 'nn_geometry'
    if tag is not None and tag != architecture:
        raise ValueError(f'Checkpoint {path} declares {tag!r}. Select a checkpoint for {architecture}.')
    config = checkpoint['config']
    if architecture == 'nn_boundary' and config.get('equivariant') is True and config.get('integral') is False:
        config = {key: value for key, value in config.items() if key not in ('equivariant', 'integral')}
    model = MODELS[architecture](**config).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint.get('nodes', 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('architecture', choices=MODELS)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--case', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--coarse', action='store_true', help='Use the 25-frame T=100 rollout schedule')
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = args.model or Path('results/supervised/models')/args.architecture/'model.pt'
    model, nodes = load_model(args.architecture, path, args.device)
    bank = Trajectories(args.data, nodes, args.device, model.horizon)
    index = int(bank.splits['test'][0]) if args.case is None else bank.ids.index(args.case)
    frames = evaluation_grid(bank.times) if args.coarse else torch.arange(len(bank.times), device=args.device)
    times = bank.times[frames]
    with torch.no_grad():
        if args.architecture == 'nn_direct':
            initial = bank.curves[index, 0]
            predicted = torch.cat([model(initial.expand(len(batch), -1, -1), batch) for batch in times.split(8)])
        else:
            rows = torch.tensor([index], device=args.device)
            predicted = forecast(model, bank, rows, frames)[0]
    radius = torch.sqrt(1+2*times)[:, None]
    label = bank.curves[index, frames]
    output = args.out or path.parent/'prediction.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, times=times.cpu().numpy(), case=bank.ids[index],
                        z=(torch.complex(predicted[:, 0], predicted[:, 1])*radius).cpu().numpy(),
                        target=(torch.complex(label[:, 0], label[:, 1])*radius).cpu().numpy())
    print(f'Saved case {bank.ids[index]} to {output}')


if __name__ == '__main__':
    main()
