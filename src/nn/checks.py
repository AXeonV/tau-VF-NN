"""Check exact circular motion, differentiable loss, and FNO output shape."""
import torch
from .model import TauFNO
from .conditions import features,sample_conditions
from .physics import TauPhysicsLoss


def main():
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    physics=TauPhysicsLoss(48).to(device)
    p=torch.zeros((1,12),device=device,dtype=torch.float64);p[:,10]=.015;p[:,11]=1.
    field=torch.zeros((1,1,32,32),device=device,dtype=torch.float64)
    t=torch.tensor([.5],device=device,dtype=torch.float64)
    circle,_=physics(field,p,t)
    assert float(circle)<1e-7
    p=sample_conditions(1,device)
    torch.manual_seed(123)
    field=(torch.randn((1,1,16,16),device=device,dtype=torch.float64)*.001).requires_grad_()
    direction=torch.randn_like(field);direction/=direction.norm()
    loss,_=physics(field,p,t);loss.backward();analytic=float((field.grad*direction).sum())
    with torch.no_grad():
        plus,_=physics(field+1e-6*direction,p,t);minus,_=physics(field-1e-6*direction,p,t)
    finite=float((plus-minus)/2e-6)
    assert abs(analytic-finite)<1e-5*max(1,abs(finite))
    inputs,_,_=features(p)
    model=TauFNO().to(device)
    assert model(inputs).shape==(1,1,32,32)
    print(f'{device}: circle loss={float(circle):.3e}, directional gradient error={abs(analytic-finite):.3e}; FNO shape OK')


if __name__=='__main__':main()
