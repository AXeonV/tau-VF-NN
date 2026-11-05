"""Resolve the initial capillary transient before encoding NN weights."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from ..traditional.conformal import ConformalFlow
from ..traditional.evolution import redistribute


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='results/ellipse_long/initialization.npz')
    parser.add_argument('--out',default='results/nn/ellipse_long_accuracy/initialization.npz')
    a=parser.parse_args();coarse=np.load(a.source)
    flow=ConformalFlow(128,1024,.015,3.,0.)
    def decode(y):
        u=np.zeros(129,complex);u[1::2]=y;return u
    def rhs(t,y):return flow.rhs(t,flow.pack(decode(y)))[:129][1::2]
    sol=solve_ivp(rhs,(0,5),flow.u0[1::2].real,method='BDF',jac=flow.symmetric_jacobian,rtol=2e-11,atol=2e-13,max_step=.01,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    times=np.unique(np.r_[0,np.geomspace(1e-6,5,513),np.linspace(0,5,401),coarse['times'][coarse['times']>2]])
    old=CubicSpline(coarse['times'],coarse['z'],axis=0)
    curves=[]
    for t in times:
        if t<5:
            z=redistribute(flow.geometry(t,decode(sol.sol(t)))[0])[0]
            if t>2:
                s=(t-2)/3;w=s**3*(10-15*s+6*s*s)
                z=(1-w)*z+w*old(t)
        else:z=old(t)
        # Match the prescribed area, not a discretization-dependent area drift.
        from ..traditional.evolution import area
        z*=np.sqrt(np.pi*(1+2*t)/area(z));curves.append(z)
    out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out,times=times,z=curves,theta=flow.theta)
    print(f'Prepared initialization: {len(times)} times, initial transient resolved to 1e-6; no new trajectory term in physical loss.')


if __name__=='__main__':main()
