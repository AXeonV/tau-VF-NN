"""FP64 arclength boundary tracking for supervised Hele-Shaw trajectories."""
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import CubicSpline
from scipy.signal import resample

from .pressure import BoundaryPressure, CudaBoundaryPressure, derivative


def area(z):
    return np.pi*np.mean((z.conj()*derivative(z)).imag)


def redistribute(z, *history):
    n = len(z)
    count = 8*n
    angle = np.linspace(0, 2*np.pi, count+1)
    dense = resample(z, count)
    metric = abs(derivative(dense))
    k = fftfreq(count, d=1/count)
    speed_hat = fft(metric)
    speed_hat[1:] /= 1j*k[1:]
    speed_hat[0] = 0.
    periodic_integral = ifft(speed_hat).real
    length = metric.mean()*2*np.pi
    arc = metric.mean()*angle+np.r_[periodic_integral, periodic_integral[0]]-periodic_integral[0]
    if np.min(np.diff(arc)) <= 0:
        raise RuntimeError('Cannot reparameterize a degenerate interface')
    inverse = CubicSpline(arc, angle, bc_type='not-a-knot')
    query = inverse(np.arange(n)*length/n)
    curves = []
    for state in (z, *history):
        dense_state = dense if state is z else resample(state, count)
        spline = CubicSpline(angle, np.r_[dense_state, dense_state[0]], bc_type='periodic')
        curves.append(spline(query))
    return curves


def evolve(initial, times, *, sigma=.015, cfl=.22, step_fraction=.012,
           max_step=2., device='cuda', progress=None, refinements=None,
           output_nodes=None, pressure_cache=None):
    """Advance an oriented simple closed curve, with exact prescribed output times."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Expected increasing output times starting at zero')
    initial = np.asarray(initial, dtype=np.complex128)
    n = len(initial)
    output_nodes = output_nodes or n
    refinements = refinements or {}
    if any(end not in times[1:-1] for end in refinements):
        raise ValueError('Refinement times must be interior output times')
    pressure_cache = {} if pressure_cache is None else pressure_cache
    pressure_class = CudaBoundaryPressure if device == 'cuda' else BoundaryPressure

    def backend(nodes):
        key = (device, nodes, sigma)
        if key not in pressure_cache:
            pressure_cache[key] = pressure_class(nodes, sigma)
        solver = pressure_cache[key]
        if device == 'cuda':
            solver.reset()
        else:
            solver.guess = None
        return solver

    def spectrum(nodes):
        k = abs(fftfreq(nodes, d=1/nodes))
        return k, np.exp(-24*(k/(nodes/2))**24)

    z = redistribute(initial)[0]
    a0 = area(z)
    if a0 <= 0:
        raise ValueError('Expected counterclockwise initial interface')
    z *= np.sqrt(np.pi/a0)
    pressure = backend(n)
    k, spectrum_filter = spectrum(n)
    output = np.empty((len(times), output_nodes), dtype=np.complex128)
    output[0] = resample(z, output_nodes)
    t = 0.
    previous = previous_velocity = previous_step = None
    steps = fallbacks = 0
    max_flux_error = max_area_correction = max_metric_variation = 0.
    for index, end in enumerate(times[1:], 1):
        while end-t > 1e-10:
            velocity, vn, _, metric = pressure.velocity(z, True)
            if not np.isfinite(velocity).all() or metric.min() <= 0:
                raise RuntimeError(f'Degenerate velocity or geometry at t={t:.6g}')
            max_flux_error = max(max_flux_error, abs(np.mean(vn*metric)-1.))
            variation = float(np.max(abs(metric/metric.mean()-1)))
            max_metric_variation = max(max_metric_variation, variation)
            dt = min(max_step, step_fraction*(1+t), cfl*(2*np.pi/n)*metric.min()/max(abs(velocity).max(), 1e-12), end-t)
            if previous_step is not None:
                dt = min(dt, 1.4*previous_step)
            if dt < 1e-10:
                raise RuntimeError(f'Time step underflow at t={t:.6g}')
            stiff = 2*sigma*(k/metric.mean())**3
            zh, vh = fft(z), fft(velocity)
            if previous is None:
                new_hat = (zh+dt*(vh+stiff*zh))/(1+dt*stiff)
            else:
                ratio = dt/previous_step
                old_hat = fft(previous)
                rhs = (1+ratio)*zh-ratio**2/(1+ratio)*old_hat
                rhs += dt*((1+ratio)*vh-ratio*fft(previous_velocity)
                           + stiff*((1+ratio)*zh-ratio*old_hat))
                new_hat = rhs/((1+2*ratio)/(1+ratio)+dt*stiff)
            new = ifft(new_hat*spectrum_filter)
            t += dt
            target_area = np.pi*(1+2*t)
            raw_area = area(new)
            if not np.isfinite(raw_area) or raw_area <= 0:
                raise RuntimeError(f'Nonpositive signed area at t={t:.6g}')
            max_area_correction = max(max_area_correction, abs(raw_area/target_area-1))
            new *= np.sqrt(target_area/raw_area)
            previous, previous_velocity, previous_step = z, velocity, dt
            z = new
            steps += 1
            if variation > .025:
                z, previous, previous_velocity = redistribute(z, previous, previous_velocity)
                z *= np.sqrt(target_area/area(z))
        if end in refinements:
            fallbacks += getattr(pressure, 'fallbacks', 0)
            n = refinements[end]
            states = [resample(value, n) for value in (z, previous, previous_velocity)]
            z, previous, previous_velocity = redistribute(*states)
            z *= np.sqrt(np.pi*(1+2*t)/area(z))
            previous *= np.sqrt(np.pi*(1+2*(t-previous_step))/area(previous))
            pressure = backend(n)
            k, spectrum_filter = spectrum(n)
        output[index] = resample(redistribute(z)[0], output_nodes)
        output[index] *= np.sqrt(np.pi*(1+2*end)/area(output[index]))
        if progress is not None:
            progress(index, end, steps)
    fallbacks += getattr(pressure, 'fallbacks', 0)
    return output, dict(steps=steps, dense_pressure_solves=fallbacks,
                        max_flux_relative_error=float(max_flux_error),
                        max_area_relative_correction=float(max_area_correction),
                        max_metric_variation=max_metric_variation)
