"""Track persistent tip-splitting candidates in normalized interface trajectories."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, resample

from ..trajectory_metrics import has_crossing


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


def split_events(curves, times):
    """A parent keeps one peak and gains a nearby peak that persists to the next saved frame."""
    peaks = [tips(curve) if not has_crossing(curve) else np.empty(0, complex) for curve in curves]
    links = [link(a, b) for a, b in zip(peaks, peaks[1:])]
    events = []
    for j in range(1, len(times)-1):
        # A crossing frame breaks the association rather than creating an apparent split.
        if any(not len(peaks[k]) for k in (j-1, j, j+1)):
            continue
        before, present = peaks[j-1:j+1]
        retained, continuation = links[j-1:j+1]
        children = set(continuation)-set(retained.values())
        claimed = set()
        for new in sorted(children):
            parents = [i for i, k in retained.items() if k in continuation and i not in claimed]
            if not parents:
                break
            angular = angle_distance(np.angle(before[parents]), np.angle(present[new]))
            index = int(np.argmin(angular))
            parent = parents[index]
            sister = retained[parent]
            spread = angle_distance(np.angle(present[new]), np.angle(present[sister]))
            if angular[index] <= .29 and spread <= .34 and abs(abs(present[sister])-abs(present[new])) < .32:
                claimed.add(parent)
                events.append(dict(time=float(times[j]), parent=float(np.angle(before[parent])),
                                   daughter=float(np.angle(present[new])), spread=float(spread)))
    return events


def match_events(predicted, target, time_limit=7.5, angle_limit=.22):
    if not predicted or not target:
        return []
    a = np.array([[x['time'], x['parent'], x['daughter']] for x in predicted])
    b = np.array([[x['time'], x['parent'], x['daughter']] for x in target])
    timing = np.abs(a[:, None, 0]-b[None, :, 0])
    parent = angle_distance(a[:, None, 1], b[None, :, 1])
    daughter = angle_distance(a[:, None, 2], b[None, :, 2])
    # The two daughters can swap identities across two discretizations.
    daughter = np.minimum(daughter, angle_distance(a[:, None, 1], b[None, :, 2])+parent)
    admissible = (timing <= time_limit) & (parent <= angle_limit) & (daughter <= .34)
    cost = np.where(admissible, timing/time_limit+parent/angle_limit+.25*daughter, 1e3)
    row, col = linear_sum_assignment(cost)
    return [dict(predicted=int(i), target=int(j), time_error=float(timing[i, j]), angle_error=float(parent[i, j]))
            for i, j in zip(row, col) if admissible[i, j]]


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
