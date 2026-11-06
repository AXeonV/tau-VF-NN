"""Local repair of a direct, frozen proposal at the requested time."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..nn.model import signed_area


def smooth(curve, scale):
    k = torch.arange(curve.shape[-1]//2+1, device=curve.device)
    return torch.fft.irfft(torch.fft.rfft(curve)*torch.exp(-(k/scale)**4), n=curve.shape[-1])


def frame(curve):
    tangent = torch.roll(curve, -1, -1)-torch.roll(curve, 1, -1)
    tangent = F.normalize(tangent, dim=1)
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)
    return tangent, normal


def descriptors(curve):
    filtered = smooth(curve, 32.)
    tangent, normal = frame(filtered)
    stride = max(1, curve.shape[-1]//128)
    turn = torch.roll(tangent, -stride, -1)-torch.roll(tangent, stride, -1)
    arc = (torch.roll(filtered, -stride, -1)-torch.roll(filtered, stride, -1)).norm(dim=1, keepdim=True)
    curvature = (-.2*(turn*normal).sum(dim=1, keepdim=True)/arc.clamp_min(.01)).tanh()
    prominence = (4*(filtered-smooth(filtered, 8.))*normal).sum(dim=1, keepdim=True)
    return torch.cat([filtered, curvature, prominence], dim=1)


@torch.no_grad()
def nonlocal_neighbors(curve):
    # Decimate the search, excluding points on the same short arclength patch.
    step = curve.shape[-1]//256
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


class Repair(nn.Module):
    def __init__(self, *, local=True, neighbors=False):
        super().__init__()
        self.neighbors = neighbors
        width = 40 if local else 66
        self.lift = nn.Conv1d(13, width, 1)
        self.blocks = nn.ModuleList()
        for dilation in (1, 4, 16, 64, 16):
            kernel = 3 if local else 1
            self.blocks.append(nn.Conv1d(width, width, kernel, dilation=dilation,
                                         padding=dilation if local else 0, padding_mode='circular'))
        self.output = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, initial, proposal, time, *, return_features=False):
        tangent, normal = frame(smooth(proposal, 48.))
        scale = torch.stack([time/20, torch.log1p(time)/3], dim=1)
        context = nonlocal_neighbors(proposal) if self.neighbors else torch.zeros_like(proposal[:, :1]).expand(-1, 3, -1)
        x = self.lift(torch.cat([initial, proposal, normal, descriptors(proposal)[:, 2:],
                                 scale[:, :, None].expand(-1, -1, proposal.shape[-1]), context], dim=1))
        for layer in self.blocks:
            x = x+.5*F.gelu(layer(x))
        move = smooth(self.output(x), proposal.shape[-1]/10)
        raw = proposal+(time/(1+time))[:, None, None]*(move[:, :1]*normal+move[:, 1:]*tangent)
        a0 = signed_area(initial)
        fixed = raw*torch.sqrt(a0/torch.maximum(signed_area(raw), .05*a0))[:, None, None]
        return (fixed, x) if return_features else fixed


class FutureTokens(nn.Module):
    """Predict EMA target tokens of future geometry, anchored by descriptor reconstruction."""
    def __init__(self, width=40, size=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(4, 32, 1), nn.GELU(), nn.Conv1d(32, size, 1))
        self.teacher = deepcopy(self.encoder).requires_grad_(False)
        self.decode = nn.Sequential(nn.Conv1d(size, 32, 1), nn.GELU(), nn.Conv1d(32, 4, 1))
        self.predict = nn.Sequential(nn.Conv1d(width+1, 32, 1), nn.GELU(), nn.Conv1d(32, size, 1))

    def forward(self, features, future, interval):
        target = F.avg_pool1d(descriptors(future), 8)
        features = F.avg_pool1d(features, 8)
        interval = interval[:, None, None].expand(-1, 1, target.shape[-1])
        prediction = self.predict(torch.cat([features, interval], dim=1))
        with torch.no_grad():
            teacher = self.teacher(target)
        token_error = (prediction-teacher).square().mean(dim=(1, 2))
        active = interval[:, 0, 0] > 0
        consistency = (token_error*active).sum()/active.sum().clamp_min(1)
        reconstruction = (self.decode(self.encoder(target))-target).square().mean()
        return consistency+reconstruction

    @torch.no_grad()
    def update(self):
        for teacher, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.lerp_(online, .02)
