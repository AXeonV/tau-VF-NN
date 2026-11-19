"""Render the held-out interface comparison and the supervised loss."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def closed(curve):
    return np.r_[curve, curve[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path('results/supervised/models/nn_direct'))
    folder = parser.parse_args().case
    with np.load(folder/'prediction.npz') as predictions:
        times = predictions['times']
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        for field, style, color, label in [('target', '-', '#6b8195', 'Time-stepped label'),
                                            ('z', '--', '#bc5d37', 'NN prediction')]:
            z = closed(predictions[field][-1])
            ax.plot(z.real, z.imag, style, color=color, linewidth=1.5, label=label)
        initial = closed(predictions['target'][0])
        ax.plot(initial.real, initial.imag, '-', color='#343c45', linewidth=1, label='Initial interface')
        ax.set(title=f'Held-out case {int(predictions["case"]):02} at t = {times[-1]:g}', xlabel='x / R0', ylabel='y / R0', aspect='equal')
        ax.legend(frameon=False)
        fig.savefig(folder/'distribution.png', dpi=170)
        plt.close(fig)
    history = np.load(folder/'history.npy')
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    ax.semilogy(history[:, 0], history[:, 1], label='Training loss', color='#347d9c', alpha=.75)
    if history.shape[1] > 2:
        ax.semilogy(history[:, 0], history[:, 2], label='Validation coordinate MSE', color='#bc5d37')
    ax.set(xlabel='Training step', ylabel='Loss', title=folder.name)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(folder/'loss.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
