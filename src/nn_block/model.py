"""Advance four time queries together before feeding back the endpoint."""
from ..rollout_model import RolloutFNO


class BlockFNO(RolloutFNO):
    window_size = 4
