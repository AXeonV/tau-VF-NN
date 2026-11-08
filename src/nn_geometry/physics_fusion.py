"""Fuse predicted-interface neighborhoods into the pretrained feature lifts."""
from torch import nn

from .model import GeometryFNO, frame, smooth
from .physics_inputs import NEIGHBOR_CHANNELS, neighbor_features


FUSIONS = ('trunk', 'geometry', 'joint')


class PhysicsFusionFNO(GeometryFNO):
    """Inject zero-initialized neighborhood adapters before pretrained blocks."""

    def __init__(self, *, fusion='joint', **config):
        super().__init__(**config)
        if fusion not in FUSIONS:
            raise ValueError(f'Expected one of {FUSIONS}')
        widths = {}
        if fusion in ('trunk', 'joint'):
            widths['trunk'] = self.lift.out_channels
        if fusion in ('geometry', 'joint'):
            if self.geometry_branch is None:
                raise ValueError('Geometry fusion requires an interface geometry branch')
            widths['geometry'] = self.geometry_branch.lift.out_channels
        self.fusion = fusion
        self.fusion_adapters = nn.ModuleDict({
            name: nn.Conv1d(NEIGHBOR_CHANNELS, width, 1, bias=False)
            for name, width in widths.items()
        })
        for adapter in self.fusion_adapters.values():
            nn.init.zeros_(adapter.weight)

    def _lift_additions(self, state):
        filtered = smooth(state, 64.)
        tangent, normal = frame(filtered)
        neighbors = neighbor_features(filtered, tangent, normal)
        trunk = self.fusion_adapters['trunk'](neighbors) if 'trunk' in self.fusion_adapters else None
        geometry = self.fusion_adapters['geometry'](neighbors) if 'geometry' in self.fusion_adapters else None
        return trunk, geometry
