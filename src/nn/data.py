"""Load accepted trajectories and apply synchronous rigid symmetries."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample
import torch


class Trajectories:
    def __init__(self, folder, nodes, device):
        folder = Path(folder)
        self.config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
        times = np.asarray(self.config['times'])
        self.times = torch.as_tensor(times, dtype=torch.float32, device=device)
        radius = np.sqrt(1+2*times)
        self.ids = []
        self.splits = {split: [] for split in ('train', 'validation', 'test')}
        curves = []
        for path in sorted(folder.glob('case_*.npz')):
            with np.load(path) as item:
                metadata = json.loads(str(item['metadata']))
                if not metadata['accepted'] or not np.array_equal(times, item['times']):
                    raise ValueError(f'Trajectory {path} did not pass the dataset checks')
                case = metadata['case']
                self.splits[case['split']].append(len(curves))
                self.ids.append(case['index'])
                z = resample(item['z'], nodes, axis=-1)/radius[:, None]
                curves.append(np.stack([z.real, z.imag], axis=1).astype(np.float32))
        if not curves:
            raise ValueError(f'No accepted trajectories in {folder}')
        self.curves = torch.from_numpy(np.stack(curves)).to(device)
        self.splits = {name: torch.tensor(indices, device=device, dtype=torch.long)
                       for name, indices in self.splits.items()}


def transform(curve, angle, mirror, shift):
    size = curve.shape[-1]
    indices = torch.arange(size, device=curve.device)[None, :]+shift[:, None]
    direction = torch.where(mirror, -1, 1)
    indices = (direction[:, None]*indices).remainder(size)
    x, y = curve.gather(-1, indices[:, None, :].expand(-1, 2, -1)).unbind(dim=1)
    y = y*direction[:, None]
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    return torch.stack([c*x-s*y, s*x+c*y], dim=1)


def augment(initial, target):
    batch, _, size = initial.shape
    device = initial.device
    angle = 2*torch.pi*torch.rand(batch, device=device)
    mirror = torch.rand(batch, device=device) < .5
    shift = torch.randint(size, (batch,), device=device)
    return transform(initial, angle, mirror, shift), transform(target, angle, mirror, shift)
