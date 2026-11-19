"""Finite interface displacements with a rotation- and reflection-equivariant Fourier operator."""
import torch
from torch import nn
from torch.nn import functional as F

from ..rollout_model import RolloutFNO


def geometry(state, normal):
    radius = state.norm(dim=1, keepdim=True).clamp_min(1e-6)
    radial = state/radius
    # Signed bending relative to the outward normal is unchanged by reflection.
    spacing = max(1, state.shape[-1]//128)
    left, right = state.roll(spacing, -1)-state, state.roll(-spacing, -1)-state
    bend = ((F.normalize(left, dim=1)+F.normalize(right, dim=1))*normal).sum(1, keepdim=True)
    alignment = (radial*normal).sum(1, keepdim=True)
    return radius, radial, bend, alignment


class EvenSpectralConv(nn.Module):
    """A real, even Fourier multiplier commutes with periodic reversal and reindexing."""
    def __init__(self, width, modes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(modes, width, width)/width)

    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        m = min(spectrum.shape[-1], len(self.weights))
        y = torch.bmm(spectrum[..., :m].permute(2, 0, 1), self.weights[:m].to(spectrum.dtype))
        return torch.fft.irfft(F.pad(y.permute(1, 2, 0), (0, spectrum.shape[-1]-m)), n=x.shape[-1])


class BoundaryFNO(RolloutFNO):
    window_size = 1

    def __init__(self, *, modes=64, depth=4):
        width = 56
        super().__init__(width=width, modes=modes, depth=depth)
        self.lift = nn.Conv1d(7, width, 1)
        self.spectral = nn.ModuleList(EvenSpectralConv(width, modes) for _ in range(depth))
        self.output[-1] = nn.Linear(width, 3)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def encode_curve(self, anchor, state, speed, normal):
        radius, radial, bend, alignment = geometry(state, normal)
        features = (radius, anchor.norm(dim=1, keepdim=True), (anchor*radial).sum(1, keepdim=True),
                    (anchor*normal).sum(1, keepdim=True), alignment, bend, speed)
        return self.lift(torch.cat(features, dim=1))

    def decode_update(self, features, state, normal):
        outward, radial, potential = self.output(features).transpose(2, 3).split(1, dim=2)
        # On reversing node order both this difference and the tangent change sign.
        d = max(1, state.shape[-1]//128)
        along = (potential.roll(-d, -1)-potential.roll(d, -1))*4
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
        return outward*normal[:, None]+radial*F.normalize(state, dim=1)[:, None]+along*tangent[:, None]
