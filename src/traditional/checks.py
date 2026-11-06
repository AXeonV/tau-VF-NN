"""Check circular pressure, a linear mode, and CPU/GPU pressure agreement."""
import numpy as np

from .pressure import BoundaryPressure, CudaBoundaryPressure


def main():
    angle = np.arange(256)*2*np.pi/256
    physics = BoundaryPressure(256)
    circle = np.exp(1j*angle)
    vn = physics.velocity(circle, True)[1]
    assert np.max(abs(vn-1)) < 1e-8
    eps = 1e-6
    curve = (1+eps*np.cos(5*angle))*circle
    vn = physics.velocity(curve, True)[1]
    growth = 2*np.mean(vn*np.cos(5*angle))/eps
    assert abs(growth-2.2) < 1e-5
    print(f'Circular pressure and mode-five growth passed ({growth:.8f})')

    import torch

    if torch.cuda.is_available():
        gpu = CudaBoundaryPressure(256)
        for amplitude in [.1, .1001, .101, .11]:
            curve = (1+amplitude*np.cos(3*angle+.2)+.04*np.sin(5*angle))*circle
            expected = physics.velocity(curve, True)[1]
            actual = gpu.velocity(curve, True)[1]
            assert np.max(abs(actual-expected)) < 1e-8
        print('FP64 CPU/CUDA pressure agreement passed across moving shapes')


if __name__ == '__main__':
    main()
