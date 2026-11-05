"""On-demand initial conditions; no evolution labels or solver trajectories."""
import torch


def sample_conditions(count,device,generator=None):
    # [log aspect, rotation, four cosine and four sine modes, sigma, radius normalization]
    p=torch.zeros(count,12,device=device,dtype=torch.float64)
    p[:,0]=torch.rand(count,device=device,generator=generator,dtype=torch.float64)*.47
    p[:,1]=torch.rand(count,device=device,generator=generator,dtype=torch.float64)*2*torch.pi
    p[:,2:10]=.012*torch.randn(count,8,device=device,generator=generator,dtype=torch.float64)
    p[:,10]=.01+.015*torch.rand(count,device=device,generator=generator,dtype=torch.float64)
    p[:,11]=1.
    theta=torch.arange(256,device=device,dtype=torch.float64)*2*torch.pi/256
    r=radius(p,theta[None].expand(count,-1))
    p[:,11]=r.square().mean(dim=1).sqrt()
    return p


def radius(p,theta):
    # theta has one independent coordinate array per condition.
    trailing=(1,)*(theta.ndim-1)
    def col(i):return p[:,i].reshape((-1,)+trailing)
    angle=theta-col(1)
    a=torch.exp(.5*col(0));b=1/a
    r=1/torch.sqrt(angle.cos().square()/a.square()+angle.sin().square()/b.square())
    k=torch.arange(2,6,device=p.device,dtype=p.dtype)
    c=p[:,2:6].reshape((len(p),)+trailing+(4,))
    s=p[:,6:10].reshape((len(p),)+trailing+(4,))
    modulation=(c*(theta[...,None]*k).cos()+s*(theta[...,None]*k).sin()).sum(dim=-1)
    return r*torch.exp(modulation)/col(11)


def features(p,size=32,extent=2.8):
    axis=torch.linspace(-extent,extent,size,device=p.device,dtype=p.dtype)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    rr=torch.sqrt(xx.square()+yy.square());theta=torch.atan2(yy,xx)
    r0=radius(p,theta[None].expand(len(p),-1,-1))
    sdf=rr[None]-r0
    base_tau=.5*(rr[None].square()-r0.square()).clamp_min(0)
    x=xx[None].expand_as(sdf)/extent;y=yy[None].expand_as(sdf)/extent
    sig=(p[:,10,None,None]/.015).log().expand_as(sdf)
    inputs=torch.stack([sdf/extent,torch.sigmoid(-sdf/.05),x,y,sig],dim=1).float()
    return inputs,base_tau,axis
