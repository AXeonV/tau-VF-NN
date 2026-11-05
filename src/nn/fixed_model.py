"""FP64 neural free-boundary representation for one exact ellipse.

The time network outputs Fourier coefficients, not polar radii. Concave and
non-star-shaped contours and retreat are representable. The two reflections
are exact symmetries of this particular initial/boundary-value problem.
"""
import math
import torch
from torch import nn


class NeuralInterface(nn.Module):
    def __init__(self,u0,scale,T,width=192,depth=4,time_scale=.5):
        super().__init__()
        self.T=float(T);self.width=width;self.depth=depth
        self.register_buffer('u0',u0.double())
        self.register_buffer('scale',scale.double())
        self.register_buffer('initial_velocity',torch.zeros_like(u0,dtype=torch.float64))
        self.register_buffer('initial_decay',torch.ones_like(u0,dtype=torch.float64))
        self.register_buffer('hard_velocity',torch.tensor(False))
        self.register_buffer('time_scale',torch.tensor(time_scale,dtype=torch.float64))
        self.register_buffer('frequencies',torch.tensor([1,2,4,8,16,32],dtype=torch.float64))
        self.layers=nn.ModuleList([nn.Linear(15,width,dtype=torch.float64)]+[nn.Linear(width,width,dtype=torch.float64) for _ in range(depth-1)])
        self.output=nn.Linear(width,len(u0),dtype=torch.float64)
        for layer in self.layers:nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def network_values(self,t):
        denominator=torch.log1p(self.time_scale.new_tensor(self.T)/self.time_scale)
        s=torch.log1p(t/self.time_scale)/denominator
        ds=1/(self.time_scale+t)/denominator
        phase=2*torch.pi*s[:,None]*self.frequencies[None]
        dphase=2*torch.pi*ds[:,None]*self.frequencies[None]
        h=torch.cat([s[:,None],s[:,None]**2,s[:,None]**3,phase.sin(),phase.cos()],dim=1)
        dh=torch.cat([ds[:,None],(2*s*ds)[:,None],(3*s*s*ds)[:,None],phase.cos()*dphase,-phase.sin()*dphase],dim=1)
        for layer in self.layers:
            a=layer(h);da=dh@layer.weight.T
            h=a.tanh();dh=(1-h.square())*da
        value=self.output(h);derivative=dh@self.output.weight.T
        return value,derivative

    def unconstrained(self,t):
        value,derivative=self.network_values(t)
        gate=t/(1+t);dgate=1/(1+t).square()
        u=self.u0[None]+self.scale[None]*gate[:,None]*value
        ud=self.scale[None]*(gate[:,None]*derivative+dgate[:,None]*value)
        return u,ud

    def forward(self,t):
        if bool(self.hard_velocity):
            value,derivative=self.network_values(t)
            decay=torch.exp(-t[:,None]*self.initial_decay[None])
            response=-torch.expm1(-t[:,None]*self.initial_decay[None])/self.initial_decay[None]
            gate=-torch.expm1(-t[:,None]*self.initial_decay[None]);dgate=self.initial_decay[None]*decay
            stiffness=(self.initial_decay[None]-1)/(1+2*t[:,None]).pow(1.5)
            precondition=1/(1+stiffness)
            dprecondition=3*(self.initial_decay[None]-1)/(1+2*t[:,None]).pow(2.5)*precondition.square()
            amplitude=gate.square()*precondition
            damplitude=2*gate*dgate*precondition+gate.square()*dprecondition
            u=self.u0[None]+self.initial_velocity[None]*response+self.scale[None]*amplitude*value
            ud=self.initial_velocity[None]*decay+self.scale[None]*(amplitude*derivative+damplitude*value)
            return u,ud
        return self.unconstrained(t)


class FreeBoundaryGeometry(nn.Module):
    def __init__(self,n,modes):
        super().__init__();self.n=n;self.modes=modes
        if n<2*modes:raise ValueError('At least two quadrature nodes per spatial cutoff are required')
        k=torch.arange(1,modes+1,2,dtype=torch.float64)
        angle=torch.arange(n,dtype=torch.float64)*2*torch.pi/n
        self.register_buffer('k',k)
        self.register_buffer('cos',(angle[:,None]*k).cos())
        self.register_buffer('sin',(angle[:,None]*k).sin())

    def forward(self,u,t,ud):
        m=len(self.k);a,b=u[:,:m],u[:,m:];ad,bd=ud[:,:m],ud[:,m:]
        signed=(self.k*a*b).sum(dim=1)
        if bool((signed<=0).any()):raise ValueError('Invalid signed enclosed area')
        factor=torch.sqrt((1+2*t)/signed)
        raw=a@self.cos.T+1j*(b@self.sin.T)
        z=factor[:,None]*raw
        za=factor[:,None]*((-a*self.k)@self.sin.T+1j*((b*self.k)@self.cos.T))
        zaa=factor[:,None]*((-a*self.k.square())@self.cos.T+1j*((-b*self.k.square())@self.sin.T))
        metric=za.abs()
        if bool((metric<=1e-8).any()):raise ValueError('Degenerate interface parameterization')
        normal=-1j*za/metric
        curvature=(za.conj()*zaa).imag/metric.pow(3)
        dsigned=(self.k*(ad*b+a*bd)).sum(dim=1)
        zt=factor[:,None]*(ad@self.cos.T+1j*(bd@self.sin.T))+z*(1/(1+2*t)-dsigned/(2*signed))[:,None]
        geometric=(zt*normal.conj()).real
        return z,metric,normal,curvature,geometric


def impose_initial_velocity(model,n,modes,sigma):
    from .physics import BoundaryPressure
    geometry=FreeBoundaryGeometry(n,modes).to(model.u0.device)
    pressure=BoundaryPressure(n).to(model.u0.device)
    with torch.no_grad():
        t=model.u0.new_zeros(1)
        z,metric,normal,kappa,_=geometry(model.u0[None],t,torch.zeros_like(model.u0[None]))
        vp=pressure(z,metric,normal,kappa,t.new_full((1,),sigma))
        mean=(kappa*vp*metric).mean(dim=1,keepdim=True)/metric.mean(dim=1,keepdim=True)
        integral=torch.fft.fft(metric*(mean-kappa*vp));freq=pressure.k
        integral[:,freq!=0]/=1j*freq[freq!=0];integral[:,0]=0.
        vt=torch.fft.ifft(integral).real
        motion=vp*normal+vt*(1j*normal)
        velocity=torch.cat([2/n*motion.real@geometry.cos,2/n*motion.imag@geometry.sin],dim=1)-model.u0[None]
        model.initial_velocity.copy_(velocity[0])
        decay=1+sigma*geometry.k.pow(3)/metric.mean().pow(3)
        model.initial_decay.copy_(torch.cat([decay,decay]));model.hard_velocity.fill_(True)
