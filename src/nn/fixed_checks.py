"""Analytic precision checks for the single-case neural boundary solver."""
import torch
from .fixed_model import NeuralInterface,FreeBoundaryGeometry,impose_initial_velocity
from .physics import BoundaryPressure


def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    modes=64;u0=torch.zeros(modes,dtype=torch.float64);u0[0]=u0[modes//2]=1.
    model=NeuralInterface(u0,torch.ones_like(u0)*.001,1000,width=32,depth=2).to(device)
    with torch.no_grad():model.output.weight.normal_(std=.01)
    t=torch.tensor([.3,20.,800.],device=device,dtype=torch.float64,requires_grad=True)
    u,ud=model(t)
    _,automatic=torch.autograd.functional.jvp(lambda tt:model(tt)[0],t,torch.ones_like(t))
    derivative_error=float((ud-automatic).abs().max())
    assert derivative_error<1e-12
    # Exact zero-tension homothetic ellipse, including the nonuniform metric.
    initial=torch.zeros((3,modes),device=device,dtype=torch.float64)
    initial[:,0]=3.**.5;initial[:,modes//2]=3.**-.5
    geometry=FreeBoundaryGeometry(256,modes).to(device)
    z,metric,normal,kappa,vg=geometry(initial,t.detach(),torch.zeros_like(initial))
    physical=BoundaryPressure(256).to(device)(z,metric,normal,kappa,t.detach().new_zeros(3))
    velocity_error=float((physical-vg).abs().max())
    assert velocity_error<1e-8
    # Explicit geometry time derivative must match a centered difference.
    z,metric,normal,kappa,vg=geometry(u,t,ud)
    step=1e-5
    plus,dp=model(t+step);minus,dm=model(t-step)
    zp=geometry(plus,t+step,dp)[0];zm=geometry(minus,t-step,dm)[0]
    finite_normal=(((zp-zm)/(2*step))*normal.conj()).real
    geometry_error=float((finite_normal-vg).abs().max())
    assert geometry_error<1e-7
    impose_initial_velocity(model,256,modes,.015)
    zero=t.detach().new_zeros(1);_,exact_derivative=model(zero)
    assert float((exact_derivative-model.initial_velocity[None]).abs().max())<1e-14
    hard_t=t.detach().new_tensor([1e-6,.001,.1,20.,800.]).requires_grad_()
    _,hard_derivative=model(hard_t)
    _,hard_automatic=torch.autograd.functional.jvp(lambda tt:model(tt)[0],hard_t,torch.ones_like(hard_t))
    hard_error=float((hard_derivative-hard_automatic).abs().max())
    assert hard_error<1e-12
    print(f'{device}: time derivative error={derivative_error:.3e}; initial-layer derivative error={hard_error:.3e}; exact ellipse velocity error={velocity_error:.3e}; geometric velocity FD error={geometry_error:.3e}')


if __name__=='__main__':main()
