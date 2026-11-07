"""Shared periodic spectral layers and spectral area."""
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
