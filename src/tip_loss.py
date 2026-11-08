"""Spatial supervision for fingertip protrusions and their intervening valleys."""

import math

import torch
from torch.nn import functional as F


def _radial_features(radius):
    nodes = radius.shape[-1]
    features = []
    for fraction in (1/48, 1/20):
        stride = max(1, round(nodes*fraction))
        left = radius-radius.roll(stride, dims=-1)
        right = radius-radius.roll(-stride, dims=-1)
        difference = ((left-right).square()+.002**2).sqrt()
        protrusion = .5*(left+right-difference)
        depression = .5*(-left-right-difference)
        features.extend((protrusion, depression))
    # A smooth threshold leaves gradients on weak, as-yet unsplit fingers.
    return .015*F.softplus((torch.stack(features, dim=1)-.025)/.015)


def _feature_maps(curves, grid, bandwidth):
    radius = curves.norm(dim=1)
    edges = (curves.roll(-1, dims=-1)-curves).norm(dim=1)
    arc = .5*(edges+edges.roll(1, dims=-1))
    arc = arc/arc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mass = _radial_features(radius)*arc[:, None]

    x, y = curves.unbind(dim=1)
    x_kernel = torch.exp(-.5*((grid[:, :, None]-x[:, None])/bandwidth[:, None, None]).square())
    y_kernel = torch.exp(-.5*((grid[:, None, :]-y[:, :, None])/bandwidth[:, None, None]).square())
    maps = (x_kernel[:, None]*mass[:, :, None]) @ y_kernel[:, None]
    return maps*(math.sqrt(2*math.pi)/bandwidth[:, None, None, None])


def tip_geometry_loss(prediction, target, *, reduction='mean'):
    """Compare spatial maps of multiscale radial tip and valley prominence.

    Inputs are normalized curves of shape [..., 2, nodes] sampled along the
    closed interface. Cyclic reindexing leaves the maps unchanged. Each feature
    is splatted in Cartesian space, so multiple points along a ray are supported.
    """
    if prediction.shape[:-1] != target.shape[:-1] or prediction.shape[-2] != 2:
        raise ValueError('Expected curves with matching leading axes and two coordinates')
    if min(prediction.shape[-1], target.shape[-1]) < 3:
        raise ValueError('Expected at least three nodes in each closed curve')
    predicted = prediction.reshape(-1, 2, prediction.shape[-1])
    expected = target.reshape(-1, 2, target.shape[-1])
    with torch.no_grad():
        radius = torch.maximum(predicted.norm(dim=1).amax(dim=-1), expected.norm(dim=1).amax(dim=-1))
        extent = (radius+.30).clamp_min(1.7)
        grid = extent[:, None]*torch.linspace(-1., 1., 49, device=prediction.device, dtype=prediction.dtype)
        bandwidth = .045*extent
    difference = _feature_maps(predicted, grid, bandwidth)-_feature_maps(expected, grid, bandwidth)
    # Account for the small fraction of the raster covered by the interface.
    loss = 32.*difference.square().mean(dim=(1, 2, 3))
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'none':
        return loss.reshape(prediction.shape[:-2])
    raise ValueError(f'Unknown reduction {reduction!r}')
