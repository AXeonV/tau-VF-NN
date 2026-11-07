"""Track persistent tip-splitting candidates in normalized interface trajectories."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, resample

from .trajectory_metrics import has_crossing


def angle_distance(a, b):
    return np.abs(np.angle(np.exp(1j*(a-b))))


def tips(curve):
    n = 1024
    z = resample(curve, n)
    radii = np.abs(z)
    peaks = find_peaks(np.tile(radii, 3), prominence=.035, distance=18)[0]
    return z[peaks[(peaks >= n) & (peaks < 2*n)]-n]


def link(earlier, later, limit=.25):
    if not len(earlier) or not len(later):
        return {}
    angle = angle_distance(np.angle(earlier[:, None]), np.angle(later[None, :]))
    radius = np.abs(np.abs(earlier[:, None])-np.abs(later[None, :]))
    cost = np.where(angle <= limit, angle+.25*radius, 1e3)
    row, col = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(row, col) if cost[i, j] < 1e3}


def tip_births(curves, times):
    """Detect new persistent radial peaks, including those with ambiguous parentage."""
    valid = [not has_crossing(z) for z in curves]
    peaks = [tips(z) if simple else np.empty(0, complex) for z, simple in zip(curves, valid)]
    links = [link(a, b) for a, b in zip(peaks, peaks[1:])]
    births = []
    for j in range(1, len(times)-1):
        if all(valid[j-1:j+2]):
            for k in sorted(set(links[j])-set(links[j-1].values())):
                births.append((float(times[j]), float(np.angle(peaks[j][k]))))
    return births


def match_births(first, second, time_limit=7.5, angle_limit=.24):
    if not first or not second:
        return []
    first, second = np.asarray(first), np.asarray(second)
    dt = np.abs(first[:, None, 0]-second[None, :, 0])
    da = angle_distance(first[:, None, 1], second[None, :, 1])
    admissible = (dt <= time_limit) & (da <= angle_limit)
    rows, cols = linear_sum_assignment(np.where(admissible, dt/time_limit+da/angle_limit, 1e3))
    return [(float(dt[i, j]), float(da[i, j])) for i, j in zip(rows, cols) if admissible[i, j]]
