"""Generate independent time-dense validation and test snapshots and check their resolution."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample
import torch

from .train import complex_curves, frame_errors
from ..traditional.evolution import evolve
from ..trajectory_data import Trajectories


def normalized(z, times):
    z = z/np.sqrt(1+2*times)[..., None]
    return np.stack([z.real, z.imag], axis=-2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/supervised/trajectories'))
    parser.add_argument('--out', type=Path, default=Path('results/supervised/world_t20'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = Trajectories(args.data, 1024, 'cpu', 20.)
    saved = np.asarray(bank.config['times'][:len(bank.times)])
    times = np.unique(np.r_[saved, np.arange(.25, 20., .25)])
    original = np.searchsorted(times, saved)
    options = dict(sigma=bank.config['sigma'], cfl=.17, step_fraction=.008, output_nodes=1024)
    pressure_cache = {}
    splits = {}
    for split in ('validation', 'test'):
        rows = bank.splits[split].tolist()
        dense, statistics = [], []
        labels = bank.curves[rows].numpy()
        for case, target in zip([bank.ids[row] for row in rows], labels):
            start = perf_counter()
            z0 = complex_curves(target[0])
            for nodes in (768, 1024, 1536):
                curves, info = evolve(resample(z0, nodes), times, pressure_cache=pressure_cache, **options)
                scaled = normalized(curves, times)
                error = frame_errors(scaled[None, original], target[None])[0]
                if error.max() <= .0035:
                    break
            if error.max() > .0035:
                raise RuntimeError(f'Audit failed the trajectory tolerance on case {case}')
            dense.append(scaled.astype(np.float32))
            statistics.append(dict(case=case, nodes=nodes, seconds=perf_counter()-start,
                                   saved_max_error=float(error.max()), final_error=float(error[-1]), **info))
            print(split, statistics[-1], flush=True)
        splits[split] = np.stack(dense)
        splits[f'{split}_statistics'] = np.array(json.dumps(statistics))
    np.savez_compressed(args.out/'audit.npz', times=times, original=original, **splits)


if __name__ == '__main__':
    main()
