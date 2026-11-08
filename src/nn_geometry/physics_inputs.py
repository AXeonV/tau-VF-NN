"""Matched residual heads with additional geometric and motion input channels."""
import torch
from torch import nn

from .model import GeometryFNO, descriptors, frame, smooth
from ..fourier import signed_area


VARIANTS = ('geometry_control', 'neighbors', 'history', 'combined')
NEIGHBORS = 3
NEIGHBOR_CHANNELS = 4*NEIGHBORS
HISTORY_CHANNELS = 2


def neighbor_features(curve, tangent, normal):
    """Read the closest distinct nonlocal patches in the local interface frame."""
    nodes = curve.shape[-1]
    stride = (nodes+255)//256
    sampled, tangents, normals = (x[..., ::stride] for x in (curve, tangent, normal))
    count = sampled.shape[-1]
    index = torch.arange(count, device=curve.device)
    separation = (index[:, None]-index).abs()
    separation = torch.minimum(separation, count-separation)
    with torch.no_grad():
        points = sampled.transpose(1, 2).contiguous()
        distance = torch.cdist(points, points).masked_fill(separation <= count//16, torch.inf)
        choices = []
        for _ in range(NEIGHBORS):
            nearest = distance.argmin(dim=-1)
            valid = distance.gather(-1, nearest[..., None]).isfinite()
            choices.append((nearest, valid.transpose(1, 2)))
            # A second point from the same arclength patch adds little information.
            distance = distance.masked_fill(separation[nearest] <= count//12, torch.inf)
    features = []
    for nearest, valid in choices:
        gather = nearest[:, None].expand(-1, 2, -1)
        offset = sampled.gather(-1, gather)-sampled
        length = offset.norm(dim=1, keepdim=True)
        direction = offset/length.clamp_min(.015)
        proximity = (-length/.25).exp()*valid
        along_normal = (direction*normals).sum(1, keepdim=True)
        along_tangent = (direction*tangents).sum(1, keepdim=True).abs()
        opposing_normal = -(normals.gather(-1, gather)*normals).sum(1, keepdim=True)
        features.extend((proximity, proximity*along_normal, proximity*along_tangent,
                         proximity*opposing_normal))
    values = torch.cat(features, dim=1)
    # Interpolate periodically, including a potentially shorter final segment.
    grid = torch.arange(nodes, device=curve.device)
    left = grid//stride
    right = (left+1) % count
    last_span = nodes-(count-1)*stride
    span = torch.where(right == 0, last_span, stride)
    weight = ((grid-left*stride)/span).to(curve.dtype)
    return values.index_select(-1, left)*(1-weight)+values.index_select(-1, right)*weight


def motion_features(state, previous, normal, before, start):
    """Estimate normalized normal motion per unit log-radius time at two scales."""
    gap = .5*torch.log1p(2*(start-before).clamp_min(0)/(1+2*before))
    gap = gap[:, None, None]
    difference = state-previous
    speeds = []
    for motion in (difference, smooth(difference, 24.)):
        speed = (motion*normal).sum(1, keepdim=True)/gap.clamp_min(1e-6)
        speeds.append(torch.where(gap > 1e-6, speed.clamp(-2., 2.), 0.))
    return torch.cat(speeds, dim=1)


class PhysicsInputFNO(GeometryFNO):
    """Add an equal-size normal residual head with selected physics inputs."""

    def __init__(self, *, physics_inputs='combined', **config):
        super().__init__(**config)
        if physics_inputs not in VARIANTS:
            raise ValueError(f'Expected one of {VARIANTS}')
        self.physics_inputs = physics_inputs
        self.physics_head = nn.Sequential(
            nn.Conv1d(7+NEIGHBOR_CHANNELS+HISTORY_CHANNELS, 24, 1), nn.GELU(),
            nn.Conv1d(24, 24, 9, padding=4, padding_mode='circular'), nn.GELU(),
            nn.Conv1d(24, 1, 1),
        )
        nn.init.zeros_(self.physics_head[-1].weight)
        nn.init.zeros_(self.physics_head[-1].bias)

    def forward(self, initial, state, previous, before, start, queries):
        prediction = super().forward(initial, state, previous, before, start, queries)
        filtered = smooth(state, 64.)
        tangent, normal = frame(filtered)
        batch, _, nodes = state.shape
        neighbors = state.new_zeros(batch, NEIGHBOR_CHANNELS, nodes)
        if self.physics_inputs in ('neighbors', 'combined'):
            neighbors = neighbor_features(filtered, tangent, normal)
        history = state.new_zeros(batch, HISTORY_CHANNELS, nodes)
        if self.history and self.physics_inputs in ('history', 'combined'):
            history = motion_features(state, previous, normal, before, start)
        radial = (filtered*normal).sum(1, keepdim=True)
        anchor = ((filtered-initial)*normal).sum(1, keepdim=True) if self.anchor else torch.zeros_like(radial)
        age = (start/self.horizon)[:, None, None].expand_as(radial)
        local = torch.cat([descriptors(state)[:, 2:], radial, anchor, age, neighbors, history], dim=1)
        local = local[:, None].expand(-1, queries.shape[1], -1, -1)
        lead = (queries-start[:, None])/(1+2*start[:, None])
        gate = lead/(.25+lead)
        gate = gate[:, :, None, None]
        base_motion = ((prediction-state[:, None])*normal[:, None]).sum(2, keepdim=True)
        base_motion = torch.where(gate > 0, base_motion/gate.clamp_min(1e-8), 0.)
        query_time = (queries/self.horizon)[:, :, None, None].expand_as(base_motion)
        features = torch.cat([local, base_motion, query_time], dim=2)
        correction = self.physics_head(features.flatten(0, 1)).reshape(batch, queries.shape[1], 1, nodes)
        candidate = prediction+gate*correction*normal[:, None]
        base_area = signed_area(prediction.flatten(0, 1))
        candidate_area = signed_area(candidate.flatten(0, 1))
        valid = (base_area > 0) & (candidate_area >= .1*base_area)
        safe_area = torch.where(valid, candidate_area, 1.)
        factor = torch.sqrt(torch.where(valid, base_area/safe_area, 1.)).reshape(batch, -1, 1, 1)
        return torch.where((valid.reshape(batch, -1, 1, 1) & (gate > 0)),
                           candidate*factor, prediction)


class ExactHistoryFNO(GeometryFNO):
    """Replace the history time scale while preserving every trainable parameter.

    Warm weights load exactly. Their predictions change because the existing
    history channel now estimates the derivative in log-radius time.
    """

    def forward(self, initial, state, previous, before, start, queries):
        interval = start-before
        old_gap = (interval/(1+2*start)).clamp_min(.015)
        elapsed = .5*torch.log1p(2*interval.clamp_min(0)/(1+2*before))
        scale = torch.where(elapsed > 1e-6, old_gap/elapsed.clamp_min(1e-6), 0.)
        scaled_previous = state-(state-previous)*scale[:, None, None]
        return super().forward(initial, state, scaled_previous, before, start, queries)
