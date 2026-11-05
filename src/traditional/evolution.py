"""Generate long-time initial guesses with arclength boundary tracking.

Surface-tension stiffness is handled by an implicit Fourier |D|^3 term.
Area and the two reflection symmetries of the initial ellipse are retained.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.fft import fft,ifft,fftfreq
from scipy.signal import resample
from .pressure import BoundaryPressure,derivative


def area(z):return np.pi*np.mean((z.conj()*derivative(z)).imag)


def symmetric(z):
    n=len(z);reverse=(-np.arange(n))%n
    z=(z+z[reverse].conj())/2
    return (z-np.roll(z,n//2))/2


def redistribute(z,*history):
    n=len(z);alpha=np.arange(n)*2*np.pi/n
    old=np.r_[alpha,2*np.pi]
    spline=CubicSpline(old,np.r_[z,z[0]],bc_type='periodic')
    dense=np.linspace(0,2*np.pi,8*n+1)
    speed=abs(spline(dense,1))
    ds=(speed[:-1]+speed[1:])/2*np.diff(dense)
    cumulative=np.r_[0,np.cumsum(ds)]
    query=PchipInterpolator(cumulative,dense)(np.arange(n)*cumulative[-1]/n)
    values=[]
    for value in (z,)+history:
        interp=CubicSpline(old,np.r_[value,value[0]],bc_type='periodic')
        values.append(symmetric(interp(query)))
    return values


def evolve(T=1000.,n=1024,sigma=.015,aspect=3.,dtmax=.15,out='results/ellipse_long',save_every=2.,restart=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    alpha=np.arange(n)*2*np.pi/n
    z=np.sqrt(aspect)*np.cos(alpha)+1j/np.sqrt(aspect)*np.sin(alpha)
    z=redistribute(z)[0];z*=np.sqrt(np.pi/area(z))
    physics=BoundaryPressure(n,sigma)
    k=fftfreq(n,1/n)
    filt=np.exp(-24*(abs(k)/(n/2))**20)
    t=0.;previous=None;fprev=None;dtprev=None
    times=[t];curves=[z.copy()];speeds=[];steps=0
    if restart:
        saved=np.load(restart)
        times=list(saved['times']);curves=[redistribute(resample(c,n))[0] for c in saved['z']]
        z=curves[-1].copy();t=float(times[-1])
    next_save=min(t+save_every,T)
    started=time.time()
    while t<T-1e-10:
        f,vn,curvature,metric=physics.velocity(z,True)
        dt=min(dtmax,.22*(2*np.pi*metric.min()/n)/max(abs(f).max(),1e-8),T-t,next_save-t)
        if dt<1e-7:raise RuntimeError(f'Interface underresolved at t={t:g}')
        stiff=2*sigma*abs(k)**3/metric.mean()**3
        zh=fft(z);fh=fft(f)
        if previous is None:
            new=ifft((zh+dt*(fh+stiff*zh))/(1+dt*stiff))
        else:
            ratio=dt/dtprev
            a0=(1+2*ratio)/(1+ratio)
            prevh=fft(previous)
            rhs=(1+ratio)*zh-ratio**2/(1+ratio)*prevh
            rhs+=dt*((1+ratio)*fh-ratio*fft(fprev)+stiff*((1+ratio)*zh-ratio*prevh))
            new=ifft(rhs/(a0+dt*stiff))
        new=symmetric(ifft(fft(new)*filt))
        new*=np.sqrt(np.pi*(1+2*(t+dt))/area(new))
        previous,fprev=z,f
        z=new;t+=dt;dtprev=dt;steps+=1
        if steps%10==0:
            z,previous,fprev=redistribute(z,previous,fprev)
            z*=np.sqrt(np.pi*(1+2*t)/area(z))
        if t>=next_save-1e-9 or t>=T-1e-9:
            times.append(t);curves.append(z.copy());speeds.append(float(vn.min()))
            next_save=min(next_save+save_every,T)
        if steps%100==0:print(f'initialization t={t:.3f}/{T:g} steps={steps} elapsed={time.time()-started:.1f}s',flush=True)
        if steps%500==0:
            np.savez_compressed(out/'initialization.npz',times=times,z=curves,theta=alpha)
        if not np.isfinite(z).all():raise RuntimeError('Non-finite interface')
    settings=dict(T=T,n=n,sigma=sigma,aspect=aspect,dtmax=dtmax,steps=steps,seconds=time.time()-started)
    np.savez_compressed(out/'initialization.npz',times=times,z=curves,theta=alpha)
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(json.dumps(settings,indent=2),flush=True)
    return np.array(times),np.array(curves)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--T',type=float,default=1000.)
    p.add_argument('--n',type=int,default=1024)
    p.add_argument('--sigma',type=float,default=.015)
    p.add_argument('--aspect',type=float,default=3.)
    p.add_argument('--dt',type=float,default=.15)
    p.add_argument('--out',default='results/ellipse_long')
    p.add_argument('--save-every',type=float,default=2.)
    p.add_argument('--restart')
    a=p.parse_args();evolve(a.T,a.n,a.sigma,a.aspect,a.dt,a.out,a.save_every,a.restart)


if __name__=='__main__':main()
