"""Load accepted trajectories and apply synchronous rigid symmetries."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample
import torch


DEFAULT_DATA = Path('data/supervised/trajectories_t100_dense_128')
DATASET_PREFIXES = {
    'trajectories': ('trajectories_t1000_128', 64),
    'trajectories_t100_dense': ('trajectories_t100_dense_128', 96),
}


def case_paths(folder):
    """Reuse promoted cases when a staging dataset has no local copy."""
    paths = sorted(folder.glob('case_*.npz'))
    if folder.name not in DATASET_PREFIXES:
        return paths
    canonical, count = DATASET_PREFIXES[folder.name]
    local = {int(path.stem[5:]) for path in paths}
    for index in range(count):
        candidate = folder.parent/canonical/f'case_{index:03}.npz'
        if index not in local and candidate.is_file():
            paths.append(candidate)
    return sorted(paths, key=lambda path: int(path.stem[5:]))


class Trajectories:
    def __init__(self, folder, nodes, device, horizon=None):
        folder = Path(folder)
        self.config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
        full_times = np.asarray(self.config['times'])
        if horizon is None:
            times = full_times
        else:
            index = np.flatnonzero(np.isclose(full_times, horizon, rtol=0., atol=1e-9))
            if horizon <= 0 or len(index) != 1:
                raise ValueError('Horizon must be a positive saved label time')
            times = full_times[:index[0]+1]
        self.times = torch.as_tensor(times, dtype=torch.float32, device=device)
        radius = np.sqrt(1+2*times)
        self.ids = []
        self.splits = {split: [] for split in ('train', 'validation', 'test')}
        curves = []
        for path in case_paths(folder):
            with np.load(path) as item:
                metadata = json.loads(str(item['metadata']))
                if not metadata['accepted'] or not np.array_equal(full_times, item['times']):
                    raise ValueError(f'Trajectory {path} did not pass the dataset checks')
                case = metadata['case']
                self.splits[case['split']].append(len(curves))
                self.ids.append(case['index'])
                z = resample(item['z'][:len(times)], nodes, axis=-1)/radius[:, None]
                curves.append(np.stack([z.real, z.imag], axis=1).astype(np.float32))
        if not curves:
            raise ValueError(f'No accepted trajectories in {folder}')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f'Duplicate initial conditions in {folder}')
        expected = self.config.get('expected_cases')
        if 'new_per_family' in self.config and expected is None:
            expected = 64+8*self.config['new_per_family']
        if expected is not None and len(curves) != expected:
            raise ValueError(f'The expanded dataset is incomplete in {folder}')
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


def augment(initial, *targets):
    batch, _, size = initial.shape
    device = initial.device
    angle = 2*torch.pi*torch.rand(batch, device=device)
    mirror = torch.rand(batch, device=device) < .5
    shift = torch.randint(size, (batch,), device=device)
    return tuple(transform(curve, angle, mirror, shift) for curve in (initial, *targets))
