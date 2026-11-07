"""One spatial operator with a switch between independent and coupled time queries."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .fourier import SpectralConv, signed_area


class RolloutFNO(nn.Module):
    def __init__(self, horizon=100., width=40, modes=64, depth=4, *,
                 coupled=True, anchor=True, history=True):
        super().__init__()
        self.horizon = float(horizon)
        self.coupled = coupled
        self.anchor = anchor
        self.history = history
        self.lift = nn.Conv1d(5, width, 1)
        self.start = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, 2*depth*width))
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(depth))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(depth))
        self.query = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, 2*width))
        self.query_mix = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.geometry_branch = None

    def time_features(self, time):
        phase = torch.log1p(2*time)/math.log1p(2*self.horizon)
        return torch.stack([time/self.horizon, phase, (2*torch.pi*phase).sin(),
                            (2*torch.pi*phase).cos()], dim=-1)

    def forward(self, initial, state, previous, before, start, queries):
        gap = (start-before)/(1+2*start)
        tangent = F.normalize(torch.roll(state, -4, -1)-torch.roll(state, 4, -1), dim=1)
        normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)
        speed = ((state-previous)*normal).sum(dim=1, keepdim=True)
        speed = (speed/gap.clamp_min(.015)[:, None, None]).clamp(-2., 2.)
        if not self.history:
            speed = torch.zeros_like(speed)
        anchor = initial if self.anchor else torch.zeros_like(initial)
        x = self.lift(torch.cat([state, anchor, speed], dim=1))
        modulation = self.start(self.time_features(start)).reshape(len(start), len(self.spectral), 2, -1)
        for index, (spectral, local) in enumerate(zip(self.spectral, self.pointwise)):
            scale, shift = modulation[:, index].unbind(dim=1)
            x = x+F.gelu((1+scale[..., None])*(spectral(x)+local(x))+shift[..., None])
        lead = (queries-start[:, None])/(1+2*start[:, None])
        timing = torch.cat([self.time_features(queries), self.time_features(lead)], dim=-1)
        scale, shift = self.query(timing).chunk(2, dim=-1)
        features = x.transpose(1, 2)[:, None]*(1+scale[:, :, None])+shift[:, :, None]
        if self.coupled:
            left = torch.cat([features[:, :1], features[:, :-1]], dim=1)
            right = torch.cat([features[:, 1:], features[:, -1:]], dim=1)
            context = .25*left+.50*features+.25*right
        else:
            context = features
        features = features+F.gelu(self.query_mix(context))
        displacement = self.output(features).transpose(2, 3)
        if self.geometry_branch is not None:
            displacement = displacement+self.geometry_branch(state, speed, features)
        gate = lead/(.25+lead)
        raw = state[:, None]+gate[:, :, None, None]*displacement
        flat = raw.flatten(0, 1)
        target = signed_area(state)[:, None].expand_as(lead).reshape(-1)
        normalized = flat*torch.sqrt(target/torch.maximum(signed_area(flat), .10*target))[:, None, None]
        normalized = normalized.reshape_as(raw)
        return torch.where((lead == 0)[:, :, None, None], state[:, None], normalized)
