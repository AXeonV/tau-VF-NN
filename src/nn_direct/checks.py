"""Check initial conditions, area, reindexing, and supervised gradients."""
import torch

from ..trajectory_data import transform
from .model import CurveFNO, signed_area


def main():
    torch.manual_seed(15)
    model = CurveFNO(horizon=1000., width=16, modes=16, depth=2).double()
    with torch.no_grad():
        model.output[-1].weight.normal_(std=.01)
    angle = torch.arange(128, dtype=torch.float64)*2*torch.pi/128
    initial = torch.stack([angle.cos(), angle.sin()])[None]
    t = initial.new_tensor([700.])
    prediction = model(initial, t)
    assert torch.equal(model(initial, t*0), initial)
    torch.testing.assert_close(signed_area(prediction), signed_area(initial), atol=1e-11, rtol=1e-11)
    shifted = model(initial.roll(31, dims=-1), t)
    torch.testing.assert_close(shifted, prediction.roll(31, dims=-1), atol=1e-11, rtol=1e-11)
    augmented = transform(initial, t/1400, torch.tensor([True]), torch.tensor([23]))
    torch.testing.assert_close(signed_area(augmented), signed_area(initial), atol=1e-11, rtol=1e-11)
    (prediction-initial).square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    print('Initial state, area, cyclic phase, reflection, and parameter gradients passed')


if __name__ == '__main__':
    main()
