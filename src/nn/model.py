"""Time-conditioned periodic Fourier operator for normalized interface coordinates."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def signed_area(curves):
    z = torch.complex(curves[:, 0], curves[:, 1])
    coefficients = torch.fft.fft(z, norm='forward')
    k = torch.fft.fftfreq(z.shape[-1], d=1/z.shape[-1], device=z.device, dtype=z.real.dtype)
    return torch.pi*(k*coefficients.abs().square()).sum(dim=-1)


class SpectralConv(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(modes, width, width, 2)/width)

    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        modes = min(self.weights.shape[0], spectrum.shape[-1])
        filtered = torch.bmm(spectrum[..., :modes].permute(2, 0, 1),
                             torch.view_as_complex(self.weights[:modes])).permute(1, 2, 0)
        return torch.fft.irfft(F.pad(filtered, (0, spectrum.shape[-1]-modes)), n=x.shape[-1])


class CurveFNO(nn.Module):
    def __init__(self, horizon=1000., width=64, modes=128, depth=6):
        super().__init__()
        self.horizon = float(horizon)
        self.register_buffer('frequencies', torch.tensor([1., 2., 4., 8.]))
        self.lift = nn.Conv1d(2, width, 1)
        self.time_encoder = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, 2*depth*width))
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(depth))
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(depth))
        self.output = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 2, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, initial, time):
        log_time = torch.log1p(time/.01)/math.log1p(self.horizon/.01)
        phase = 2*torch.pi*log_time[:, None]*self.frequencies[None]
        features = torch.cat([time[:, None]/self.horizon, log_time[:, None], phase.sin(), phase.cos()], dim=1)
        modulation = self.time_encoder(features).reshape(len(time), len(self.local), 2, -1)
        x = self.lift(initial)
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            scale, shift = modulation[:, index].unbind(dim=1)
            update = spectral(x)+local(x)
            x = x+F.gelu((1+scale[..., None])*update+shift[..., None])
        gate = (time/(1+time))[:, None, None]
        raw = initial+gate*self.output(x)
        initial_area = signed_area(initial)
        raw_area = signed_area(raw)
        safe_area = torch.maximum(raw_area, .05*initial_area)
        normalized = raw*torch.sqrt(initial_area/safe_area)[:, None, None]
        return normalized
