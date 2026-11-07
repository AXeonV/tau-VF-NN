"""Train the single-frame Fourier rollout with a jointly trained geometry branch."""
from .model import GeometryFNO
from ..rollout import single_loss
from ..rollout_training import main


if __name__ == '__main__':
    main('nn_geometry', GeometryFNO, single_loss)
