"""Analytic pressure and automatic-gradient checks."""
import numpy as np
from .pressure import BoundaryPressure


def main():
    n=256;angle=np.arange(n)*2*np.pi/n
    physics=BoundaryPressure(n,.015)
    z=np.exp(1j*angle)
    _,vn,_,_=physics.velocity(z,True)
    circle=float(max(abs(vn-1)))
    eps=1e-6;z=(1+eps*np.cos(5*angle))*np.exp(1j*angle)
    _,vn,_,_=physics.velocity(z,True)
    growth=float(2*np.mean(vn*np.cos(5*angle))/eps)
    assert circle<1e-8
    assert abs(growth-2.2)<1e-5
    from .optimize import InterfaceModel
    import torch
    model=InterfaceModel(64,16,.015)
    u=torch.zeros((1,16));u[0,0]=u[0,8]=1.;u[0,2]=.002
    u.requires_grad_();t=torch.tensor([1.])
    z,s,normal,k,_=model.geometry(u,t)
    v=model.pressure_velocity(z,s,normal,k)
    loss=v.square().mean();loss.backward()
    analytic=float(u.grad[0,2]);step=1e-6
    def objective(delta):
        trial=u.detach().clone();trial[0,2]+=delta
        z,s,normal,k,_=model.geometry(trial,t)
        return float(model.pressure_velocity(z,s,normal,k).square().mean())
    finite=(objective(step)-objective(-step))/(2*step)
    assert abs(analytic-finite)<1e-5*max(1,abs(finite))
    print(f'circle speed error: {circle:.3e}; mode-5 growth: {growth:.8f}; gradient error: {abs(analytic-finite):.3e}')


if __name__=='__main__':main()
