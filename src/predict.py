"""Predict a held-out interface with one of the three retained models."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .nn_block.model import BlockFNO
from .nn_direct.model import CurveFNO
from .nn_geometry.model import GeometryFNO
from .rollout import evaluation_grid, forecast
from .trajectory_data import DEFAULT_DATA, Trajectories


MODELS = dict(nn_direct=CurveFNO, nn_block=BlockFNO, nn_geometry=GeometryFNO)


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
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    tag = checkpoint.get('architecture')
    if tag == 'baseline' and args.architecture == 'nn_geometry':
        tag = 'nn_geometry'
    if tag is not None and tag != args.architecture:
        parser.error(f'Checkpoint {path} declares {tag!r}. Select a checkpoint for {args.architecture}.')
    model = MODELS[args.architecture](**checkpoint['config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    bank = Trajectories(args.data, checkpoint.get('nodes', 1024), args.device, model.horizon)
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
