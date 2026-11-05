"""Optimize a full interface history using only pressure/kinematic residuals."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from scipy.interpolate import BSpline,CubicSpline

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)


class InterfaceModel:
    def __init__(self,n=384,modes=128,sigma=.015):
        self.n,self.sigma=n,sigma
        self.k=torch.arange(1,modes+1,2)
        self.freq=torch.tensor(np.fft.fftfreq(n,1/n))
        angle=torch.arange(n)*2*np.pi/n
        phase=angle[:,None]*self.k[None]
        self.cos,self.sin=phase.cos(),phase.sin()
        self.eye=torch.eye(n)
        chord=2*torch.sin((angle[:,None]-angle[None,:])/2).abs()
        self.log_chord=torch.log(chord+self.eye)
        self.single_multiplier=torch.where(self.freq==0,0.,1/(2*self.freq.abs().clamp_min(1)))

    def geometry(self,u,t,ud=None):
        m=len(self.k);a,b=u[:,:m],u[:,m:]
        area=(self.k*a*b).sum(dim=1).clamp_min(.05)
        scale=torch.sqrt((1+2*t)/area)
        raw=(a@self.cos.T).to(torch.complex128)+1j*(b@self.sin.T)
        za=scale[:,None]*((-a*self.k)@self.sin.T+1j*((b*self.k)@self.cos.T))
        zaa=scale[:,None]*((-a*self.k.square())@self.cos.T+1j*((-b*self.k.square())@self.sin.T))
        z=raw*scale[:,None]
        speed=za.abs().clamp_min(1e-8)
        normal=-1j*za/speed
        curvature=(za.conj()*zaa).imag/speed.pow(3)
        zt=None
        if ud is not None:
            ad,bd=ud[:,:m],ud[:,m:]
            darea=(self.k*(ad*b+a*bd)).sum(dim=1)
            zd=(ad@self.cos.T).to(torch.complex128)+1j*(bd@self.sin.T)
            zt=scale[:,None]*zd+z*(1/(1+2*t)-darea/(2*area))[:,None]
        return z,speed,normal,curvature,zt

    def pressure_velocity(self,z,speed,normal,curvature):
        B,N=z.shape
        delta=z[:,:,None]-z[:,None,:]
        d2=delta.abs().square()+self.eye[None]
        kernel=(delta*normal.conj()[:,None,:]).real/d2*speed[:,None,:]/N
        kernel=kernel*(1-self.eye)[None]+torch.diag_embed(-curvature*speed/(2*N))
        top=torch.cat([kernel+.5*self.eye[None],torch.ones((B,N,1))],dim=2)
        bottom=torch.cat([speed/speed.sum(dim=1,keepdim=True),torch.zeros((B,1))],dim=1)[:,None,:]
        matrix=torch.cat([top,bottom],dim=1)
        rhs=torch.cat([torch.log(z.abs())-self.sigma*curvature,torch.zeros((B,1))],dim=1)
        if not torch.isfinite(matrix).all() or not torch.isfinite(rhs).all():
            raise RuntimeError('Non-finite boundary pressure system')
        mu=torch.stack([torch.linalg.solve(matrix[j],rhs[j]) for j in range(B)])[:,:N]
        mua=torch.fft.ifft(1j*self.freq[None]*torch.fft.fft(mu)).real
        smooth=.5*torch.log(d2)-self.log_chord[None]
        smooth=smooth*(1-self.eye)[None]+torch.diag_embed(torch.log(speed))
        single=torch.fft.ifft(self.single_multiplier[None]*torch.fft.fft(mua)).real-torch.bmm(smooth,mua[:,:,None]).squeeze(-1)/N
        dn=torch.fft.ifft(1j*self.freq[None]*torch.fft.fft(single)).real/speed
        return (z.conj()*normal).real/z.abs().square()-dn


def time_basis(T,count):
    edges=T*np.linspace(0,1,count-2)**1.7
    knots=np.r_[np.zeros(4),edges[1:-1],np.full(4,T)]
    spline=BSpline(knots,np.eye(count),3)
    q,w=np.polynomial.legendre.leggauss(2)
    t=np.concatenate([(a+b)/2+(b-a)/2*q for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)/2/T*w for a,b in zip(edges[:-1],edges[1:])])
    greville=np.array([knots[j+1:j+4].mean() for j in range(count)])
    return spline,t,weights,greville


def optimize(folder,n=512,modes=192,count=112,iterations=100,retreat_weight=1.,chunk=8):
    folder=Path(folder);source=np.load(folder/'initialization.npz')
    settings=json.loads((folder/'settings.json').read_text())
    T=float(source['times'][-1]);model=InterfaceModel(n,modes,settings['sigma'])
    angles=source['theta'];ks=model.k.numpy()
    warm_z=source['z']/np.sqrt(1+2*source['times'][:,None])
    ca=2/len(angles)*warm_z.real@np.cos(angles[:,None]*ks)
    cb=2/len(angles)*warm_z.imag@np.sin(angles[:,None]*ks)
    warm=np.c_[ca,cb];u0=warm[0]
    spline,t,w,gr=time_basis(T,count)
    ft=np.unique(np.r_[t,gr,np.linspace(0,T,4*count)])
    desired=CubicSpline(source['times'],warm,axis=0)(ft)-u0
    coeff=np.linalg.lstsq(spline(ft)[:,1:],desired,rcond=None)[0]
    repeated=np.tile(ks,2)
    precondition=1/(1+settings['sigma']*repeated[None]**3/(1+2*gr[1:,None])**1.5)
    scale=torch.tensor(precondition);x=torch.nn.Parameter(torch.tensor(coeff)/scale)
    B=torch.tensor(spline(t)[:,1:]);D=torch.tensor(spline.derivative()(t)[:,1:])
    tt,ww,base_u=map(torch.tensor,(t,w,u0))
    opt=torch.optim.LBFGS([x],lr=.15,max_iter=iterations,max_eval=iterations*2,history_size=25,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    history=[];started=time.time()
    def closure():
        opt.zero_grad();total=0.;kin_total=0.;ret_total=0.
        for ids in np.array_split(np.arange(len(t)),max(1,(len(t)+chunk-1)//chunk)):
            u=base_u[None]+B[ids]@(x*scale);ud=D[ids]@(x*scale)
            z,metric,normal,curv,zt=model.geometry(u,tt[ids],ud)
            if not torch.isfinite(z).all():raise RuntimeError('Non-finite trial interface')
            vn=model.pressure_velocity(z,metric,normal,curv)
            motion=(zt*normal.conj()).real
            res=(motion-vn)*torch.sqrt(1+2*tt[ids,None])
            kin=(ww[ids]*res.square().mean(dim=1)).sum()
            retreat=(ww[ids]*(torch.relu(-motion)*torch.sqrt(1+2*tt[ids,None])).square().mean(dim=1)).sum()
            local=kin+retreat_weight*retreat
            local.backward()
            total+=float(local.detach());kin_total+=float(kin.detach());ret_total+=float(retreat.detach())
        if not torch.isfinite(x.grad).all():raise RuntimeError('Non-finite objective gradient')
        history.append([len(history),total,kin_total,ret_total,time.time()-started])
        if len(history)==1:print(f'initial loss={total:.6e} grad={float(x.grad.norm()):.4e}',flush=True)
        if len(history)%5==0:print(f'optimization step={len(history)} loss={total:.6e} elapsed={time.time()-started:.1f}s',flush=True)
        return torch.tensor(total)
    closure();opt.step(closure);closure()
    coeff=np.vstack([np.zeros((1,len(u0))),(x*scale).detach().numpy()])
    solution=BSpline(spline.t,coeff,3)
    output_times=np.unique(np.r_[source['times'],np.linspace(0,T,401)])
    zs=[];vmins=[]
    with torch.no_grad():
        for ids in np.array_split(np.arange(len(output_times)),32):
            ot=output_times[ids]
            z,metric,normal,curv,zt=model.geometry(torch.tensor(u0+solution(ot)),torch.tensor(ot),torch.tensor(solution.derivative()(ot)))
            zs.append(z.numpy());vmins.extend((zt*normal.conj()).real.min(dim=1).values.numpy())
    optimized=np.concatenate(zs)
    np.savez_compressed(folder/'optimized.npz',times=output_times,z=optimized,theta=np.arange(n)*2*np.pi/n,coefficients=coeff,knots=spline.t,u0=u0)
    np.save(folder/'history.npy',np.array(history))
    metrics=dict(T=T,n=n,modes=modes,temporal_basis=count,sigma=settings['sigma'],aspect=settings['aspect'],perturb=0.,initial_loss=history[0][1],final_loss=history[-1][1],kinematic_loss=history[-1][2],retreat_loss=history[-1][3],retreat_weight=retreat_weight,min_geometric_normal_speed=float(min(vmins)),evaluations=len(history),seconds=time.time()-started)
    (folder/'optimization_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--case',default='results/ellipse_long')
    p.add_argument('--n',type=int,default=512)
    p.add_argument('--modes',type=int,default=192)
    p.add_argument('--basis',type=int,default=112)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--retreat-weight',type=float,default=1.)
    a=p.parse_args();optimize(a.case,a.n,a.modes,a.basis,a.iterations,a.retreat_weight)


if __name__=='__main__':main()
