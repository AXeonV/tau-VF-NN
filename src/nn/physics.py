"""Differentiable exterior pressure and the early star-shaped FNO loss adapter.

BoundaryPressure also serves the free closed-interface NN in fixed_case.py.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .conditions import radius


class BoundaryPressure(nn.Module):
    def __init__(self,n):
        super().__init__();self.n=n
        angle=torch.arange(n,dtype=torch.float64)*2*torch.pi/n
        k=torch.fft.fftfreq(n,d=1/n,dtype=torch.float64)
        eye=torch.eye(n,dtype=torch.float64)
        self.register_buffer('k',k);self.register_buffer('eye',eye)
        self.register_buffer('log_chord',torch.log(2*torch.sin((angle[:,None]-angle[None,:])/2).abs()+eye))
        self.register_buffer('single',torch.where(k==0,0.,1/(2*k.abs().clamp_min(1))))

    def forward(self,z,metric,normal,curvature,sigma):
        B,N=z.shape;eye=self.eye
        delta=z[:,:,None]-z[:,None,:];d2=delta.abs().square()+eye[None]
        kernel=(delta*normal.conj()[:,None,:]).real/d2*metric[:,None,:]/N
        kernel=kernel*(1-eye)[None]+torch.diag_embed(-curvature*metric/(2*N))
        top=torch.cat([kernel+.5*eye[None],z.real.new_ones(B,N,1)],dim=2)
        bottom=torch.cat([metric/metric.sum(dim=1,keepdim=True),z.real.new_zeros(B,1)],dim=1)[:,None]
        matrix=torch.cat([top,bottom],dim=1)
        rhs=torch.cat([torch.log(z.abs())-sigma[:,None]*curvature,z.real.new_zeros(B,1)],dim=1)
        # Classical LU handles large GPU systems; CPU MKL avoids batched LU.
        if z.is_cuda and N<2048:mu=torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)[:,:N]
        else:mu=torch.stack([torch.linalg.solve(a,b) for a,b in zip(matrix,rhs)])[:,:N]
        mua=torch.fft.ifft(1j*self.k*torch.fft.fft(mu)).real
        smooth=(.5*torch.log(d2)-self.log_chord[None])*(1-eye)[None]+torch.diag_embed(torch.log(metric))
        single=torch.fft.ifft(self.single*torch.fft.fft(mua)).real-torch.bmm(smooth,mua[...,None]).squeeze(-1)/N
        dn=torch.fft.ifft(1j*self.k*torch.fft.fft(single)).real/metric
        return (z.conj()*normal).real/z.abs().square()-dn


class TauPhysicsLoss(nn.Module):
    def __init__(self,n=48,extent=2.8,h=.01,cutoff=8):
        super().__init__();self.n,self.extent,self.h=n,extent,h
        angle=torch.arange(n,dtype=torch.float64)*2*torch.pi/n
        k=torch.fft.fftfreq(n,d=1/n,dtype=torch.float64)
        self.register_buffer('angle',angle);self.register_buffer('k',k)
        self.register_buffer('lowpass',(k.abs()<=cutoff).double())
        self.pressure=BoundaryPressure(n)

    def correction(self,field,r):
        x=r*self.angle.cos();y=r*self.angle.sin()
        points=torch.stack([x/self.extent,y/self.extent],dim=-1)
        shape=r.shape
        values=F.grid_sample(field.double(),points.reshape(len(r),-1,1,2),mode='bilinear',padding_mode='border',align_corners=True)
        return values.reshape(shape)

    def level_sets(self,field,p,t):
        r0=radius(p,self.angle[None].expand(len(p),-1))[:,None]
        query=t[:,None]+t.new_tensor([-self.h,0,self.h])[None]
        r=torch.sqrt(r0.square()+2*query[:,:,None])
        for _ in range(12):r=torch.sqrt(r0.square()+2*query[:,:,None]*torch.exp(-self.correction(field,r)))
        r=torch.fft.ifft(torch.fft.fft(r)*self.lowpass).real
        return r,r0

    def forward(self,field,p,t):
        r,r0=self.level_sets(field,p,t)
        center=r[:,1];rt=(r[:,2]-r[:,0])/(2*self.h)
        za=torch.fft.ifft(1j*self.k*torch.fft.fft(center*torch.exp(1j*self.angle)))
        zaa=torch.fft.ifft(-self.k.square()*torch.fft.fft(center*torch.exp(1j*self.angle)))
        z=center*torch.exp(1j*self.angle);metric=za.abs();normal=-1j*za/metric
        curvature=(za.conj()*zaa).imag/metric.pow(3)
        geometric=(rt*torch.exp(1j*self.angle)*normal.conj()).real
        physical=self.pressure(z,metric,normal,curvature,p[:,10])
        kinetic=((geometric-physical)*torch.sqrt(1+2*t[:,None])).square().mean()
        retreat=(torch.relu(-geometric)*torch.sqrt(1+2*t[:,None])).square().mean()
        area=((center.square().mean(dim=1)-(1+2*t))/(1+2*t)).square().mean()
        tau=.5*(center.square()-r0[:,0].square())*torch.exp(self.correction(field,center[:,None])[:,0])
        level=(tau-t[:,None]).square().mean()
        total=kinetic+10*retreat+area+level
        return total,{'kinetic':kinetic,'retreat':retreat,'area':area,'level':level}
