"""All-at-once conformal level-set optimization for the pure-ellipse test.

A coarse physical simulation supplies a warm start ONLY. Every pressure,
curvature and all spline coefficients are recomputed and optimized together.
The objective uses governing equations and a monotonicity penalty only.
The contour history represents an arrival field only if domains stay nested.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from scipy.interpolate import BSpline,CubicSpline

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)


def basis(T,count):
    degree=3
    edges=T*np.linspace(0,1,count-degree+1)**1.5
    knots=np.r_[np.zeros(4),edges[1:-1],np.full(4,T)]
    b=BSpline(knots,np.eye(count),3)
    q,w=np.polynomial.legendre.leggauss(3)
    times=np.concatenate([(a+c)/2+(c-a)*q/2 for a,c in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(c-a)*w/2/T for a,c in zip(edges[:-1],edges[1:])])
    greville=np.array([knots[j+1:j+4].mean() for j in range(count)])
    return b,times,weights,greville


class GlobalMap:
    def __init__(self,K,n,sigma,aspect):
        self.K,self.n,self.sigma,self.aspect=K,n,sigma,aspect
        self.k=torch.arange(1,K+1,2)
        self.freq=torch.tensor(np.fft.fftfreq(n,1/n))
        self.u0=torch.zeros(len(self.k));self.u0[0]=np.sinh(.5*np.log(aspect))

    def geometry(self,u,t,ud=None):
        base=torch.sqrt(1+2*t)
        ah=torch.sqrt(1+(u.square()*self.k).sum(dim=1))
        spec=torch.zeros((len(t),self.n),dtype=torch.complex128)
        spec[:,1]=(base*ah).to(torch.complex128)
        spec[:,-self.k]=(base[:,None]*u).to(torch.complex128)
        z=torch.fft.ifft(spec)*self.n
        zp=torch.fft.ifft(1j*self.freq*spec)*self.n
        zpp=torch.fft.ifft(-self.freq.square()*spec)*self.n
        H=-1j*zp
        curvature=(zp.conj()*zpp).imag/zp.abs().clamp_min(1e-9).pow(3)
        Dk=torch.fft.ifft(self.freq.abs()*torch.fft.fft(curvature)).real
        num=1-self.sigma*Dk
        zt=None
        if ud is not None:
            adh=(self.k*u*ud).sum(dim=1)/ah
            st=torch.zeros_like(spec)
            st[:,1]=(ah/base+base*adh).to(torch.complex128)
            st[:,-self.k]=(u/base[:,None]+base[:,None]*ud).to(torch.complex128)
            zt=torch.fft.ifft(st)*self.n
        return z,H,num,zt,curvature


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--coarse',default='results/ellipse_short')
    p.add_argument('--out',default='results/ellipse_short')
    p.add_argument('--modes',type=int,default=1024)
    p.add_argument('--n',type=int,default=4096)
    p.add_argument('--basis',type=int,default=80)
    p.add_argument('--iterations',type=int,default=1800)
    p.add_argument('--monotone-weight',type=float,default=1000.)
    a=p.parse_args()
    coarse=np.load(Path(a.coarse)/'initialization.npz')
    cm=json.loads((Path(a.coarse)/'initialization.json').read_text())
    T=cm['T'];K=a.modes
    model=GlobalMap(K,a.n,cm['sigma'],cm['aspect'])
    sp,t,w,gr=basis(T,a.basis)
    bt=torch.tensor(sp(t)[:,1:]);dt=torch.tensor(sp.derivative()(t)[:,1:])
    tt,wt=torch.tensor(t),torch.tensor(w)
    # Interpolate only the warm start; this is never used in the loss.
    warm=CubicSpline(coarse['times'],coarse['u'][:,1::2].real,axis=0)
    ft=np.linspace(0,T,6*a.basis)
    target=np.zeros((len(ft),len(model.k)))
    target[:,:warm(ft).shape[1]]=warm(ft)
    coef=np.linalg.lstsq(sp(ft)[:,1:],target-model.u0.numpy(),rcond=None)[0]
    scale=1/(1+cm['sigma']*model.k.numpy()[None]**3/(1+2*gr[1:,None])**1.5)
    scale=torch.tensor(scale)
    x=torch.nn.Parameter(torch.tensor(coef)/scale)
    optimizer=torch.optim.LBFGS([x],lr=.5,max_iter=a.iterations,max_eval=int(1.5*a.iterations),history_size=40,tolerance_change=1e-13,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
    history=[];start=time.time()
    def closure():
        optimizer.zero_grad()
        coeff=x*scale
        u=model.u0[None]+bt@coeff;ud=dt@coeff
        z,H,num,zt,curvature=model.geometry(u,tt,ud)
        motion=(zt*H.conj()).real
        residual=(motion-num)*torch.sqrt(1+2*tt[:,None])/H.abs()
        kin=(wt*residual.square().mean(dim=1)).sum()
        retreat=torch.relu(-motion)*torch.sqrt(1+2*tt[:,None])/H.abs()
        barrier=(wt*retreat.square().mean(dim=1)).sum()
        loss=kin+a.monotone_weight*barrier
        loss.backward()
        history.append([len(history),float(loss.detach()),float(kin.detach()),float(barrier.detach()),time.time()-start])
        if len(history)%50==0:print('global evaluation',len(history),'loss',history[-1][1],'elapsed',round(time.time()-start,1),flush=True)
        return loss
    closure();optimizer.step(closure);closure()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    coeff=np.vstack([np.zeros((1,len(model.k))), (x*scale).detach().numpy()])
    solution=BSpline(sp.t,coeff,3)
    times=np.linspace(0,T,401)
    us=model.u0.numpy()[None]+solution(times)
    dus=solution.derivative()(times)
    # Evaluate in chunks to keep high-resolution diagnostics bounded in memory.
    zs=[];vs=[];ps=[];res=[]
    with torch.no_grad():
        for ids in np.array_split(np.arange(len(times)),16):
            z,H,num,zt,curv=model.geometry(torch.tensor(us[ids]),torch.tensor(times[ids]),torch.tensor(dus[ids]))
            motion=(zt*H.conj()).real
            zs.append(z.numpy());vs.append((motion/H.abs()).numpy());ps.append((num/H.abs()).numpy())
            res.append(((motion-num)*torch.sqrt(1+2*torch.tensor(times[ids,None]))/H.abs()).numpy())
    z=np.concatenate(zs);vm=np.concatenate(vs);vp=np.concatenate(ps);rr=np.concatenate(res)
    ufull=np.zeros((len(times),K+1),complex);ufull[:,1::2]=us
    report=dict(T=T,sigma=cm['sigma'],aspect=cm['aspect'],perturb=0.,modes=K,n=a.n,temporal_basis=a.basis,
        initial_loss=history[0][1],final_loss=history[-1][1],global_velocity_residual_rms=float(np.sqrt(np.mean(rr**2))),
        min_geometric_normal_speed=float(vm.min()),min_pressure_normal_speed=float(vp.min()),seconds=time.time()-start,
        warm_start=str(a.coarse),objective='physical residual only; all spline coefficients optimized simultaneously',
        global_optimum_certified=False)
    np.savez_compressed(out/'optimized.npz',times=times,u=ufull,z=z,vn=vm,pressure_vn=vp,theta=np.arange(a.n)*2*np.pi/a.n,knots=sp.t,coefficients=coeff)
    np.save(out/'history.npy',np.array(history))
    (out/'optimization_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
