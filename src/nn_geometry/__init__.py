"""Geometry-enhanced single-frame rollout."""


def checkpoint_model(architecture):
    if architecture in ('baseline', 'nn_geometry'):
        from .model import GeometryFNO
        return GeometryFNO
    if architecture == 'normal_speed':
        from .normal_speed_probe import NormalSpeedFNO
        return NormalSpeedFNO
    if architecture in ('physics_inputs', 'exact_history'):
        from .physics_inputs import ExactHistoryFNO, PhysicsInputFNO
        return ExactHistoryFNO if architecture == 'exact_history' else PhysicsInputFNO
    if architecture == 'physics_fusion':
        from .physics_fusion import PhysicsFusionFNO
        return PhysicsFusionFNO
    raise ValueError(f'Unsupported geometry checkpoint architecture {architecture}')
