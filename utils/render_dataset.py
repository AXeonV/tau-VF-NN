"""Overlay eight interface times on one axis per certified long trajectory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TIMES = np.array([0., 10., 20., 50., 100., 200., 500., 1000.])
COLORS = plt.colormaps['viridis'](np.linspace(.06, .93, len(TIMES)))[:, :3]


def render(source, out, *, overwrite=False):
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('case_*.npz')):
        destination = out/f'{path.stem}.png'
        if destination.exists() and not overwrite:
            continue
        with np.load(path) as item:
            meta = json.loads(str(item['metadata']))
            times = item['times']
            indices = np.searchsorted(times, TIMES)
            if not meta['accepted'] or indices[-1] >= len(times) or not np.array_equal(times[indices], TIMES):
                raise ValueError(f'Uncertified trajectory or missing times in {path}')
            contours = item['z'][indices]
        case = meta['case']
        limit = 1.07*np.max(np.abs(contours))
        fig, axis = plt.subplots(figsize=(7.7, 6.2), layout='constrained')
        # Paint the earliest, smallest interiors last so their colors stay visible.
        for z, color in zip(contours[::-1], COLORS[::-1]):
            axis.fill(z.real, z.imag, color=.82+.18*color, linewidth=0, zorder=1)
        for t, z, color in zip(TIMES, contours, COLORS):
            closed = np.r_[z, z[0]]
            axis.plot(closed.real, closed.imag, color=color,
                      linewidth=2.0, label=f'T = {t:g}', zorder=3)
        axis.set_aspect('equal')
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='x', ylabel='y')
        axis.set_title(f'Case {case["index"]:03} · mode {case["dominant"]} · {case["split"]}')
        axis.grid(alpha=.16)
        axis.legend(title='Time', loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
        temporary = destination.with_suffix('.tmp')
        fig.savefig(temporary, format='png', dpi=160)
        plt.close(fig)
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/supervised/trajectories_t1000_128'))
    parser.add_argument('--out', type=Path, default=Path('data/img'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    render(args.source, args.out, overwrite=args.overwrite)
    print(f'{len(list(args.out.glob("case_*.png")))} images', flush=True)


if __name__ == '__main__':
    main()
