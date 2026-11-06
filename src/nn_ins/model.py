"""A short-time interface map with semi-implicit damping of capillary modes."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def area_factor(curve):
    z = torch.complex(curve[:, 0], curve[:, 1])
    spectrum = torch.fft.fft(z, norm='forward')
    k = torch.fft.fftfreq(z.shape[-1], d=1/z.shape[-1], device=z.device, dtype=z.real.dtype)
    return (k*spectrum.abs().square()).sum(dim=-1)


def redistribute(curve):
    """Piecewise linear equal-arclength interpolation, preserving the first vertex."""
    wrapped = torch.cat([curve, curve[..., :1]], dim=-1)
    lengths = (wrapped[..., 1:]-wrapped[..., :-1]).norm(dim=1).clamp_min(1e-8)
    arc = F.pad(lengths.cumsum(-1), (1, 0))
    alpha = torch.arange(curve.shape[-1], device=curve.device, dtype=curve.dtype)/curve.shape[-1]
    desired = (arc[:, -1:]*alpha).contiguous()
    left = (torch.searchsorted(arc, desired, right=True)-1).clamp(0, curve.shape[-1]-1)
    fraction = (desired-arc.gather(1, left))/lengths.gather(1, left)
    indices = left[:, None, :].expand(-1, 2, -1)
    first = wrapped.gather(-1, indices)
    second = wrapped.gather(-1, indices+1)
    return first+(second-first)*fraction[:, None, :]


class FourierBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = nn.Parameter(torch.randn(modes, width, width, 2)/width)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, features, scale, bias):
        spectrum = torch.fft.rfft(features)
        count = min(len(self.spectral), spectrum.shape[-1])
        result = torch.bmm(spectrum[..., :count].permute(2, 0, 1),
                           torch.view_as_complex(self.spectral[:count])).permute(1, 2, 0)
        update = torch.fft.irfft(F.pad(result, (0, spectrum.shape[-1]-count)), n=features.shape[-1])
        return features+F.gelu((1+scale[..., None])*(update+self.local(features))+bias[..., None])


class InterfaceStep(nn.Module):
    def __init__(self, horizon=1000., width=64, modes=128, depth=6, sigma=.015):
        super().__init__()
        self.horizon = float(horizon)
        self.sigma = float(sigma)
        self.register_buffer('frequencies', torch.tensor([1., 2., 4., 8.]))
        self.lift = nn.Conv1d(2, width, 1)
        self.condition = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, 2*depth*width))
        self.blocks = nn.ModuleList(FourierBlock(width, modes) for _ in range(depth))
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 2, 1))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, curve, start, end):
        dt = end-start
        ds = .5*torch.log1p(2*dt/(1+2*start))
        log_time = torch.log1p(start/.01)/math.log1p(self.horizon/.01)
        phase = 2*torch.pi*log_time[:, None]*self.frequencies[None]
        condition = torch.cat([start[:, None]/self.horizon, log_time[:, None],
                               phase.sin(), phase.cos(), ds[:, None], torch.log1p(dt)[:, None]], dim=1)
        modulation = self.condition(condition).reshape(len(start), len(self.blocks), 2, -1)
        features = self.lift(curve)
        for index, block in enumerate(self.blocks):
            scale, bias = modulation[:, index].unbind(1)
            features = block(features, scale, bias)
        derivative = self.project(features)
        proposal = curve+ds[:, None, None]*derivative
        circumference = (torch.roll(curve, -1, dims=-1)-curve).norm(dim=1).sum(dim=-1)
        metric = circumference*torch.sqrt(1+2*start)/(2*torch.pi)
        k = torch.arange(curve.shape[-1]//2+1, device=curve.device, dtype=curve.dtype)
        damping = 1+dt[:, None]*self.sigma*(k/metric[:, None].clamp_min(1e-4))**3
        # Ignore the leading rigid/growth modes in the small-scale damping.
        damping = torch.where(k <= 1, 1., damping)
        smoothed = torch.fft.irfft(torch.fft.rfft(proposal)/damping[:, None], n=curve.shape[-1])
        regridded = redistribute(smoothed)
        previous_area = area_factor(curve)
        candidate_area = area_factor(regridded)
        projected = regridded*torch.sqrt(previous_area/torch.maximum(candidate_area, .1*previous_area))[:, None, None]
        return torch.where(((candidate_area > .1*previous_area) & (dt > 0))[:, None, None], projected, curve)
