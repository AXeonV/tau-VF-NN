"""Local geometry and spatially close, arclength-separated interface context."""
import torch
from torch import nn
from torch.nn import functional as F

from ..rollout_model import RolloutFNO


def smooth(curve, scale):
    k = torch.arange(curve.shape[-1]//2+1, device=curve.device)
    return torch.fft.irfft(torch.fft.rfft(curve)*torch.exp(-(k/scale)**4), n=curve.shape[-1])


def frame(curve):
    tangent = torch.roll(curve, -1, -1)-torch.roll(curve, 1, -1)
    tangent = F.normalize(tangent, dim=1)
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)
    return tangent, normal


def _coarse_geometry(curve):
    filtered = smooth(curve, 32.)
    tangent, normal = frame(filtered)
    stride = max(1, curve.shape[-1]//128)
    turn = torch.roll(tangent, -stride, -1)-torch.roll(tangent, stride, -1)
    arc = (torch.roll(filtered, -stride, -1)-torch.roll(filtered, stride, -1)).norm(dim=1, keepdim=True)
    curvature = (-.2*(turn*normal).sum(dim=1, keepdim=True)/arc.clamp_min(.01)).tanh()
    prominence = (4*(filtered-smooth(filtered, 8.))*normal).sum(dim=1, keepdim=True)
    return torch.cat([curvature, prominence], dim=1)


@torch.no_grad()
def nonlocal_neighbors(curve):
    # Decimate the search, excluding points on the same short arclength patch.
    step = max(1, curve.shape[-1]//256)
    sampled = curve[..., ::step].transpose(1, 2).contiguous()
    index = torch.arange(sampled.shape[1], device=curve.device)
    offset = (index[:, None]-index).abs()
    mask = torch.minimum(offset, len(index)-offset) <= len(index)//16
    distance = torch.cdist(sampled, sampled).masked_fill(mask, torch.inf)
    nearest = distance.argmin(dim=-1)
    vector = sampled.gather(1, nearest[..., None].expand(-1, -1, 2))-sampled
    vector = vector.transpose(1, 2)
    features = torch.cat([vector, vector.norm(dim=1, keepdim=True)], dim=1)
    padded = torch.cat([features, features[..., :1]], dim=-1)
    return F.interpolate(padded, size=curve.shape[-1]+1, mode='linear', align_corners=True)[..., :-1]


class InterfaceGeometry(nn.Module):
    """Predict a normal displacement jointly from Fourier features and interface geometry."""
    def __init__(self, global_width, width=24):
        super().__init__()
        self.lift = nn.Conv1d(7, width, 1)
        self.local = nn.ModuleList(nn.Conv1d(width, width, 3, dilation=d, padding=d,
                                            padding_mode='circular') for d in (1, 8, 24))
        self.output = nn.Sequential(nn.Linear(global_width+width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state, speed, features):
        filtered = smooth(state, 64.)
        tangent, normal = frame(filtered)
        spacing = max(1, state.shape[-1]//256)
        turn = tangent.roll(-spacing, -1)-tangent.roll(spacing, -1)
        arc = (filtered.roll(-spacing, -1)-filtered.roll(spacing, -1)).norm(dim=1, keepdim=True)
        curvature = (-.12*(turn*normal).sum(1, keepdim=True)/arc.clamp_min(.006)).tanh()
        vector = nonlocal_neighbors(filtered)
        distance = vector[:, 2:3]
        weight = torch.exp(-distance/.25)
        direction = vector[:, :2]/distance.clamp_min(.015)
        normal_gap = (direction*normal).sum(1, keepdim=True)*weight
        tangent_gap = (direction*tangent).sum(1, keepdim=True).abs()*weight
        coarse_shape = _coarse_geometry(state)
        geometry = self.lift(torch.cat([curvature, coarse_shape, speed, weight, normal_gap, tangent_gap], dim=1))
        for layer in self.local:
            geometry = geometry+.5*F.gelu(layer(geometry))
        geometry = geometry.transpose(1, 2)[:, None].expand(-1, features.shape[1], -1, -1)
        increment = self.output(torch.cat([features, geometry], dim=-1)).transpose(2, 3)
        return increment*normal[:, None]


class GeometryFNO(RolloutFNO):
    window_size = 1

    def __init__(self, *args, geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        if geometry:
            self.geometry_branch = InterfaceGeometry(self.lift.out_channels)
