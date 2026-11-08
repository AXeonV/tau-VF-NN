"""Model-independent comparisons against the supervised interface snapshots."""
import json

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import resample, find_peaks
from scipy.spatial import cKDTree


def area(z):
    k = fftfreq(len(z), d=1/len(z))
    derivative = ifft(1j*k*fft(z))
    return np.pi*np.mean((z.conj()*derivative).imag)


def _points(z):
    return np.column_stack([z.real, z.imag])


def _distance_to_curve(source, curve):
    points = _points(source)
    target = _points(curve)
    _, nearest = cKDTree(target).query(points)
    next_index = (nearest+1) % len(curve)
    prev_index = (nearest-1) % len(curve)
    start = target[nearest]
    sides = target[np.column_stack([prev_index, next_index])]-start[:, None, :]
    difference = points-start
    along = np.einsum('nki,ni->nk', sides, difference)/np.sum(sides*sides, axis=-1).clip(1e-30)
    distances = np.linalg.norm(difference[:, None]-np.clip(along, 0., 1.)[..., None]*sides, axis=-1)
    return distances.min(axis=1)


def curve_difference(first, second, times, *, stride=3, threshold=None):
    count = 2*max(first.shape[-1], second.shape[-1])
    indices = np.unique(np.r_[np.arange(0, len(times), stride), len(times)-1])
    maxima = []
    for index in indices:
        x = resample(first[index], count)
        y = resample(second[index], count)
        difference = max(_distance_to_curve(x, y).max(), _distance_to_curve(y, x).max())
        maxima.append(float(difference/np.sqrt(1+2*times[index])))
    worst = int(np.argmax(maxima))
    result = dict(sampled_frames=len(indices), maximum=maxima[worst],
                  worst_time=float(times[indices[worst]]), final=maxima[-1])
    if threshold is not None:
        result['first_exceed'] = next((float(times[i]) for i, value in zip(indices, maxima) if value > threshold), None)
    return result


def has_crossing(z):
    edges = np.roll(z, -1)-z
    midpoints = z+.5*edges
    pairs = cKDTree(_points(midpoints)).query_pairs(float(abs(edges).max())*(1+1e-10), output_type='ndarray')
    i, j = pairs.T
    keep = (j-i > 1) & (j-i < len(z)-1)
    i, j = i[keep], j[keep]
    delta = z[j]-z[i]
    denominator = (edges[i].conj()*edges[j]).imag
    keep = abs(denominator) > 1e-14
    i, j, delta, denominator = i[keep], j[keep], delta[keep], denominator[keep]
    a = (delta.conj()*edges[j]).imag/denominator
    b = (delta.conj()*edges[i]).imag/denominator
    return bool(np.any((a > 1e-9) & (a < 1-1e-9) & (b > 1e-9) & (b < 1-1e-9)))


def tip_count(z, time):
    radius = abs(resample(z, 1024))/np.sqrt(1+2*time)
    indices = find_peaks(np.tile(radius, 3), prominence=.025, distance=18)[0]
    return int(((indices >= 1024) & (indices < 2048)).sum())


def describe_geometry(curves, times):
    crossings = [float(t) for t, z in zip(times, curves) if has_crossing(z)]
    area_error = max(abs(area(z)/(np.pi*(1+2*t))-1) for t, z in zip(times, curves))
    return dict(crossing_times=crossings, max_area_relative_error=float(area_error),
                tips=[tip_count(z, t) for z, t in zip(curves, times)])


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def probe_frames(times):
    probes = times[-1]*times.new_tensor([.005, .02, .10, .30, .60, 1.])
    return (times[:, None]-probes).abs().argmin(dim=0).unique()


def report_predictions(bank, split, predictions, *, export=None):
    """Compare normalized [case, time, xy, node] predictions at every saved time."""
    from .interface_events import match_births, tip_births

    rows = bank.splits[split].tolist()
    target = bank.curves[rows].cpu().numpy()
    times = bank.times.cpu().numpy().astype(float)
    if predictions.shape != target.shape or not np.isfinite(predictions).all():
        raise ValueError('Expected finite predictions on all saved frames')
    coordinate = np.mean((predictions-target)**2, axis=(2, 3))
    radius = np.sqrt(1+2*times)[None, :, None]
    curves = (predictions[:, :, 0]+1j*predictions[:, :, 1])*radius
    labels = (target[:, :, 0]+1j*target[:, :, 1])*radius
    cases = []
    event_errors = []
    for row, z, label in zip(rows, curves, labels):
        geometry = describe_geometry(z, times)
        difference = curve_difference(z, label, times, stride=1, threshold=.10)
        bad = [t for t in [difference.pop('first_exceed'), *geometry['crossing_times']] if t is not None]
        expected = tip_births(label/radius[0], times)
        found = tip_births(z/radius[0], times)
        matched = match_births(found, expected)
        event_errors.extend(matched)
        cases.append(dict(case=bank.ids[row], **difference, first_breach=min(bad, default=None),
                          predicted_final_tips=geometry['tips'][-1], label_final_tips=tip_count(label[-1], times[-1]),
                          crossing_times=geometry['crossing_times'], tip_births_matched=len(matched),
                          tip_birth_time_mae=float(np.mean([dt for dt, _ in matched])) if matched else None,
                          tip_birth_angle_mae=float(np.mean([da for _, da in matched])) if matched else None,
                          tip_births_predicted=len(found), tip_births_label=len(expected)))
    if export is not None:
        np.savez_compressed(export, times=times, z=curves, target=labels,
                            ids=np.array([bank.ids[row] for row in rows]))
    return dict(coordinate_mse=float(coordinate.mean()),
                final_coordinate_rms=float(np.sqrt(coordinate[:, -1].mean())),
                worst_frame_coordinate_rms=float(np.sqrt(coordinate.max())),
                mean_final_geometry_distance=float(np.mean([item['final'] for item in cases])),
                mean_worst_geometry_distance=float(np.mean([item['maximum'] for item in cases])),
                self_intersecting_cases=sum(bool(item['crossing_times']) for item in cases),
                final_tip_matches=sum(item['predicted_final_tips'] == item['label_final_tips'] for item in cases),
                first_breach_median=float(np.median([item['first_breach'] if item['first_breach'] is not None
                                                    else times[-1] for item in cases])),
                tip_births_matched=sum(item['tip_births_matched'] for item in cases),
                tip_births_label=sum(item['tip_births_label'] for item in cases),
                tip_birth_time_mae=float(np.mean([dt for dt, _ in event_errors])) if event_errors else None,
                tip_birth_angle_mae=float(np.mean([da for _, da in event_errors])) if event_errors else None,
                cases=cases)
