"""Deterministic initial shapes, grouped by dominant radial perturbation."""
import numpy as np


CASE_COUNT = 64


def case_description(index):
    if not 0 <= index < CASE_COUNT:
        raise ValueError(f'Case index must be between 0 and {CASE_COUNT-1}')
    family, variant = divmod(index, 8)
    dominant = 2+family
    rng = np.random.default_rng(72001+index)
    available = [mode for mode in range(2, 13) if mode != dominant]
    secondary = int(available[(3*variant+family) % len(available)])
    available.remove(secondary)
    tertiary = int(rng.choice(available))
    amplitude = (.14+.018*variant) if dominant == 2 else (.12+.012*variant)
    terms = [(dominant, amplitude, 0.),
             (secondary, float(rng.uniform(.032, .075)), float(rng.uniform(-np.pi, np.pi))),
             (tertiary, float(rng.uniform(.008, .028)), float(rng.uniform(-np.pi, np.pi)))]
    return dict(index=index, dominant=dominant, terms=terms,
                split=('train' if variant < 6 else 'validation' if variant == 6 else 'test'))


def initial_curve(index, nodes):
    description = case_description(index)
    theta = np.arange(nodes)*2*np.pi/nodes
    radius = np.ones(nodes)
    for mode, amplitude, phase in description['terms']:
        radius += amplitude*np.cos(mode*theta+phase)
    return radius*np.exp(1j*theta)


def output_times(horizon=1000.):
    count = max(2, int(np.ceil(horizon/5))+1)
    early = np.geomspace(1e-4, min(horizon, 10.), 65)
    return np.unique(np.r_[np.linspace(0, horizon, count), early])
