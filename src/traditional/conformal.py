"""Long-time nonlinear elliptic Hele-Shaw benchmark.

Exterior conformal map f(w)=a*w+c+sum b_k*w**(-k), |w|>=1.
Oil pressure: -log|w|-sigma*harmonic_extension(curvature).
Polubarinova-Galin condition:
  Re(f_t*conj(w*f_w)) = 1-sigma*|D_theta|*curvature.
This solves the nonlocal pressure exactly in conformal coordinates, up to
Fourier truncation; there are no interior pressure source points.
Initial area pi and areal injection rate 2*pi; a is eliminated using exact area.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.integrate import solve_ivp


class ConformalFlow:
    def __init__(self,modes=64,n=512,sigma=.015,aspect=1.6,perturb=0.):
        self.K,self.n,self.sigma,self.aspect,self.perturb=modes,n,sigma,aspect,perturb
        self.k=np.arange(1,modes+1)
        self.freq=np.fft.fftfreq(n,1/n)
        self.theta=np.arange(n)*2*np.pi/n
        self.u0=np.zeros(modes+1,dtype=complex)
        self.u0[1]=np.sinh(.5*np.log(aspect))
        self.u0[5]=perturb*(1+.15j)
        self.u0[8]=perturb*.18*np.exp(.6j)
        self.u0[11]=perturb*.07*np.exp(1.1j)

    def pack(self,u):return np.r_[u.real,u.imag]
    def unpack(self,y):return y[:self.K+1]+1j*y[self.K+1:]

    def geometry(self,t,u):
        base=np.sqrt(1+2*t)
        ahat=np.sqrt(1+np.sum(self.k*np.abs(u[1:])**2))
        spectrum=np.zeros(self.n,dtype=complex)
        spectrum[1]=base*ahat
        spectrum[0]=base*u[0]
        spectrum[-self.k]=base*u[1:]
        z=np.fft.ifft(spectrum)*self.n
        zp=np.fft.ifft(1j*self.freq*spectrum)*self.n
        zpp=np.fft.ifft(-self.freq**2*spectrum)*self.n
        H=-1j*zp
        curvature=np.imag(zp.conj()*zpp)/np.abs(zp)**3
        Dk=np.fft.ifft(np.abs(self.freq)*np.fft.fft(curvature)).real
        numerator=1-self.sigma*Dk
        vn=numerator/np.abs(H)
        return z,zp,H,curvature,numerator,vn

    def rhs(self,t,y):
        u=self.unpack(y)
        z,zp,H,curvature,num,vn=self.geometry(t,u)
        q=num/np.abs(H)**2
        qhat=np.fft.fft(q)/self.n
        ghat=np.zeros(self.n,dtype=complex)
        ghat[0]=qhat[0].real
        ghat[self.freq<0]=2*qhat[self.freq<0]
        g=np.fft.ifft(ghat)*self.n
        vhat=np.fft.fft(H*g)/self.n
        base=np.sqrt(1+2*t)
        udot=np.r_[vhat[0],vhat[-self.k]]/base-u/(1+2*t)
        if self.perturb==0:
            # Preserve both reflection symmetries of the EXACT ellipse.
            # A numerical antisymmetric seed must not select a finger pattern.
            udot=udot.real.astype(complex)
            udot[::2]=0
        return self.pack(udot)

    def symmetric_jacobian(self,t,active):
        """Exact Frechet derivative; avoids noisy finite differences of high modes."""
        u=np.zeros(self.K+1,complex);u[1::2]=active
        z,zp,H,curv,num,vn=self.geometry(t,u)
        zpp=np.fft.ifft(1j*self.freq*np.fft.fft(zp))
        speed=abs(zp);q=num/speed**2
        qhat=np.fft.fft(q)/self.n
        gh=np.zeros(self.n,complex);gh[0]=qhat[0].real;gh[self.freq<0]=2*qhat[self.freq<0]
        g=np.fft.ifft(gh)*self.n
        base=np.sqrt(1+2*t);ah=np.sqrt(1+np.sum(self.k*abs(u[1:])**2))
        ks=np.arange(1,self.K+1,2)
        J=np.zeros((len(ks),len(ks)))
        for ids in np.array_split(np.arange(len(ks)),max(1,(len(ks)+31)//32)):
            ds=np.zeros((len(ids),self.n),complex)
            ds[:,1]=base*ks[ids]*active[ids]/ah
            ds[np.arange(len(ids)),-ks[ids]]=base
            dp=np.fft.ifft(1j*self.freq[None]*ds,axis=1)*self.n
            dpp=np.fft.ifft(-self.freq[None]**2*ds,axis=1)*self.n
            dabs=(zp.conj()[None]*dp).real/speed[None]
            dk=(dp.conj()*zpp[None]+zp.conj()[None]*dpp).imag/speed[None]**3-3*curv[None]*dabs/speed[None]
            dn=-self.sigma*np.fft.ifft(abs(self.freq)[None]*np.fft.fft(dk,axis=1),axis=1).real
            dq=dn/speed[None]**2-2*q[None]*dabs/speed[None]
            dqh=np.fft.fft(dq,axis=1)/self.n
            dgh=np.zeros_like(dqh);dgh[:,0]=dqh[:,0].real;dgh[:,self.freq<0]=2*dqh[:,self.freq<0]
            dg=np.fft.ifft(dgh,axis=1)*self.n
            dv=np.fft.fft((-1j*dp)*g[None]+H[None]*dg,axis=1)/self.n
            J[:,ids]=(dv[:,-ks].real/base).T
        J-=np.eye(len(ks))/(1+2*t)
        return J


def initialize(flow,T,out,rtol=1e-7,method='DOP853'):
    start=time.time();calls=0
    symmetric=flow.perturb==0
    def decode(y):
        if not symmetric:return flow.unpack(y)
        u=np.zeros(flow.K+1,dtype=complex);u[1::2]=y
        return u
    def rhs(t,y):
        nonlocal calls
        calls+=1
        if calls%20000==0:print('initialization progress',calls,round(t,6),round(time.time()-start,1),flush=True)
        full=flow.rhs(t,flow.pack(decode(y)))
        return full[:flow.K+1][1::2] if symmetric else full
    def event(t,y):
        # Stop before loss of regularity / inadequate conformal resolution.
        return np.min(np.abs(flow.geometry(t,decode(y))[2]))/np.sqrt(1+2*t)-.012
    event.terminal=True;event.direction=-1
    y0=flow.u0[1::2].real if symmetric else flow.pack(flow.u0)
    kwargs={}
    if method in ('BDF','Radau') and symmetric:kwargs['jac']=flow.symmetric_jacobian
    ref=solve_ivp(rhs,(0,T),y0,method=method,rtol=rtol,atol=rtol*.02,dense_output=True,max_step=.10,events=event,**kwargs)
    times=np.linspace(0,ref.t[-1],401)
    u=np.array([decode(v) for v in ref.sol(times).T])
    geometry=[flow.geometry(t,v) for t,v in zip(times,u)]
    z=np.array([v[0] for v in geometry]);vn=np.array([v[-1] for v in geometry])
    H=np.array([v[2] for v in geometry]);curvature=np.array([v[3] for v in geometry])
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    metrics=dict(requested_T=T,T=float(times[-1]),modes=flow.K,n=flow.n,sigma=flow.sigma,aspect=flow.aspect,perturb=flow.perturb,
        nfev=ref.nfev,seconds=time.time()-start,success=bool(ref.success),min_vn=float(vn.min()),
        min_relative_map_derivative=float(np.min(np.abs(H)/np.sqrt(1+2*times[:,None]))),max_radius=float(np.abs(z).max()),
        high_mode_relative_amplitude=float(np.max(np.abs(u[:,-8:]))),message=ref.message,method=method)
    ref.decode=decode
    np.savez_compressed(out/'initialization.npz',times=times,u=u,z=z,vn=vn,curvature=curvature,theta=flow.theta)
    (out/'initialization.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)
    return ref,metrics


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--T',type=float,default=12.)
    p.add_argument('--modes',type=int,default=64)
    p.add_argument('--n',type=int,default=512)
    p.add_argument('--sigma',type=float,default=.015)
    p.add_argument('--aspect',type=float,default=3.)
    p.add_argument('--perturb',type=float,default=0.)
    p.add_argument('--out',default='results/ellipse_short')
    p.add_argument('--method',default='BDF')
    args=p.parse_args()
    flow=ConformalFlow(args.modes,args.n,args.sigma,args.aspect,args.perturb)
    initialize(flow,args.T,args.out,method=args.method)


if __name__=='__main__':main()
