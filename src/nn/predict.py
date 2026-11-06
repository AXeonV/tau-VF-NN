"""Query a supervised model at requested times without rolling out predictions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import Trajectories
from .model import CurveFNO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--model', type=Path, default=Path('results/supervised/model/model.pt'))
    parser.add_argument('--case', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    bank = Trajectories(args.data, checkpoint['nodes'], args.device)
    if args.case is None:
        index = int(bank.splits['test'][0])
    else:
        index = bank.ids.index(args.case)
    model = CurveFNO(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    times = bank.times
    with torch.no_grad():
        estimates = torch.cat([model(bank.curves[index, 0].expand(len(batch), -1, -1), batch)
                               for batch in times.split(8)])
    label = bank.curves[index]
    radius = torch.sqrt(1+2*times)[:, None]
    z = torch.complex(estimates[:, 0], estimates[:, 1])*radius
    target = torch.complex(label[:, 0], label[:, 1])*radius
    output = args.model.parent/'prediction.npz'
    np.savez_compressed(output, times=times.cpu().numpy(), z=z.cpu().numpy(),
                        target=target.cpu().numpy(), case=bank.ids[index])
    print(f'Saved case {bank.ids[index]:02} predictions to {output}')


if __name__ == '__main__':
    main()
