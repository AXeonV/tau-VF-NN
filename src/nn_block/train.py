"""Train the four-frame Fourier rollout."""
from .model import BlockFNO
from ..rollout import block_loss
from ..rollout_training import main


if __name__ == '__main__':
    main('nn_block', BlockFNO, block_loss)
