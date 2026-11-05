"""Exterior Laplace pressure on a general closed interface.

Second-kind double-layer equation and the periodic logarithmic single-layer
quadrature give the Dirichlet-to-Neumann map. The fixed -log|z| term injects
area at rate 2*pi. Normal points from air into oil.
"""
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import gmres
from scipy.fft import fft, ifft, fftfreq


def derivative(v,order=1):
    k=fftfreq(len(v),1/len(v))
    return ifft((1j*k)**order*fft(v))


class BoundaryPressure:
    def __init__(self,n,sigma=.015):
        self.n,self.sigma=n,sigma
        self.guess=None
        self.k=fftfreq(n,1/n)
        angle=np.arange(n)*2*np.pi/n
        chord=2*abs(np.sin((angle[:,None]-angle[None,:])/2))
        np.fill_diagonal(chord,1.)
        self.chord=chord
        self.singular_multiplier=np.zeros(n)
        self.singular_multiplier[self.k!=0]=1/(2*abs(self.k[self.k!=0]))

    def geometry(self,z):
        za=derivative(z);zaa=derivative(z,2)
        speed=abs(za)
        tangent=za/speed
        normal=-1j*tangent
        curvature=(za.conj()*zaa).imag/speed**3
        return speed,tangent,normal,curvature

    def velocity(self,z,details=False):
        n=self.n
        speed,tangent,normal,curvature=self.geometry(z)
        diff=z[:,None]-z[None,:]
        d2=abs(diff)**2
        np.fill_diagonal(d2,1.)
        kernel=(diff*normal.conj()[None,:]).real/d2*speed[None,:]/n
        np.fill_diagonal(kernel,-curvature*speed/(2*n))
        mat=np.empty((n+1,n+1))
        mat[:n,:n]=kernel+.5*np.eye(n)
        mat[:n,n]=1.;mat[n,:n]=speed/speed.sum();mat[n,n]=0.
        rhs=np.r_[np.log(abs(z))-self.sigma*curvature,0.]
        if n>=768:
            solution,info=gmres(mat,rhs,x0=self.guess,rtol=1e-10,atol=1e-12,restart=50,maxiter=20)
            if info:solution=solve(mat,rhs,check_finite=False)
            self.guess=solution
        else:solution=solve(mat,rhs,check_finite=False)
        mu=solution[:n]
        mua=derivative(mu).real
        smooth=.5*np.log(d2)-np.log(self.chord)
        np.fill_diagonal(smooth,np.log(speed))
        single=ifft(self.singular_multiplier*fft(mua)).real-smooth@mua/n
        normal_derivative=derivative(single).real/speed
        vn=(z.conj()*normal).real/abs(z)**2-normal_derivative
        # Tangential redistribution maintains uniform arclength spacing.
        mean=np.mean(curvature*vn*speed)/np.mean(speed)
        ta=speed*(mean-curvature*vn)
        th=fft(ta);th[self.k!=0]/=1j*self.k[self.k!=0];th[0]=0.
        tangential=ifft(th).real
        velocity=vn*normal+tangential*tangent
        if details:return velocity,vn,curvature,speed
        return velocity
