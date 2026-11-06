"""Exterior Laplace pressure on a general closed interface.

Second-kind double-layer equation and the periodic logarithmic single-layer
quadrature give the Dirichlet-to-Neumann map. The fixed -log|z| term injects
area at rate 2*pi. Normal points from air into oil.
"""
from functools import lru_cache
import warnings

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


class CudaBoundaryPressure:
    """FP64 GPU quadrature and residual-checked warm density solve."""

    def __init__(self, n, sigma=.015):
        import torch

        self.torch = torch
        self.n = n
        self.sigma = sigma
        self.k = torch.fft.fftfreq(n, d=1/n, device='cuda', dtype=torch.float64)
        angle = torch.arange(n, device='cuda', dtype=torch.float64)*(2*np.pi/n)
        chord = 2*torch.sin(angle/2).abs()
        chord[0] = 1.
        self.log_chord = chord.log()
        self.fused = _quadrature_kernel(torch)
        if self.fused is None:
            indices = torch.arange(n, device='cuda')
            self.full_log_chord = self.log_chord[(indices[:, None]-indices).remainder(n)]
        self.single = torch.where(self.k == 0, 0., .5/self.k.abs().clamp_min(1))
        self.tangent_integral = torch.zeros_like(self.k, dtype=torch.complex128)
        self.tangent_integral[1:] = 1/(1j*self.k[1:])
        self.z = torch.exp(1j*angle)
        self.guess = torch.zeros(n+1, device='cuda', dtype=torch.float64)
        self.inverse = torch.zeros((n+1, n+1), device='cuda', dtype=torch.float64)
        self.graph = None
        self.fallbacks = 0

    def reset(self):
        self.guess.zero_()
        self.inverse.zero_()
        self.fallbacks = 0

    def _system(self):
        torch = self.torch
        z = self.z
        spectrum = torch.fft.fft(z)
        za = torch.fft.ifft(1j*self.k*spectrum)
        zaa = torch.fft.ifft(-self.k.square()*spectrum)
        speed = za.abs()
        normal = -1j*za/speed
        curvature = (za.conj()*zaa).imag/speed.pow(3)
        if self.fused is not None:
            kernel, smooth = self.fused(z, normal, speed, curvature, self.log_chord)
        else:
            delta = z[:, None]-z[None, :]
            d2 = delta.real.square()+delta.imag.square()
            d2.diagonal().fill_(1.)
            kernel = ((delta.real*normal.real + delta.imag*normal.imag)
                      * (speed/self.n)/d2)
            kernel.diagonal().copy_(-curvature*speed/(2*self.n))
            smooth = .5*d2.log()-self.full_log_chord
            smooth.diagonal().copy_(speed.log())
        rhs = z.abs().log()-self.sigma*curvature
        weights = speed/speed.sum()
        return kernel, rhs, weights, smooth, speed, normal, curvature

    def _normal_and_tangent(self, system, density):
        torch = self.torch
        _, _, weights, smooth, speed, normal, curvature = system
        derivative = torch.fft.ifft(1j*self.k*torch.fft.fft(density)).real
        single = (torch.fft.ifft(self.single*torch.fft.fft(derivative)).real
                  - smooth@derivative/self.n)
        dn = torch.fft.ifft(1j*self.k*torch.fft.fft(single)).real/speed
        vn = (self.z.conj()*normal).real/self.z.abs().square()-dn
        mean = torch.dot(curvature*vn, weights)
        tangential_hat = torch.fft.fft(speed*(mean-curvature*vn))
        tangential = torch.fft.ifft(tangential_hat*self.tangent_integral).real
        velocity = vn*normal+1j*tangential*normal
        return torch.stack([velocity.real, velocity.imag, vn, curvature, speed])

    def _defect(self, system, solution):
        torch = self.torch
        kernel, rhs, weights = system[:3]
        density = solution[:-1]
        residual = rhs-(kernel@density+.5*density+solution[-1])
        return torch.cat([residual, -torch.dot(density, weights).reshape(1)])

    def _evaluate(self):
        system = self._system()
        solution = self.guess
        for _ in range(10):
            solution = solution+self.inverse@self._defect(system, solution)
        error = self._defect(system, solution).abs().max()
        return system, solution, error, self._normal_and_tangent(system, solution[:-1])

    def _capture(self):
        torch = self.torch
        # Warm up the cuFFT and cuBLAS plans on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._evaluate()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.captured = self._evaluate()

    def velocity(self, z, details=False):
        torch = self.torch
        self.z.copy_(torch.from_numpy(np.asarray(z, dtype=np.complex128)))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        system, solution, error, output = self.captured
        if not float(error) < 2e-11:
            self.fallbacks += 1
            kernel, rhs, weights = system[:3]
            matrix = torch.empty_like(self.inverse)
            matrix[:-1, :-1] = kernel
            matrix[:-1, :-1].diagonal().add_(.5)
            matrix[:-1, -1] = 1.
            matrix[-1, :-1] = weights
            matrix[-1, -1] = 0.
            self.inverse.copy_(torch.linalg.inv(matrix))
            augmented = torch.cat([rhs, rhs.new_zeros(1)])
            solution = self.inverse@augmented
            if not float(self._defect(system, solution).abs().max()) < 2e-11:
                solution = torch.linalg.solve(matrix, augmented)
                if not float(self._defect(system, solution).abs().max()) < 2e-11:
                    raise RuntimeError('Boundary pressure solve did not meet its residual tolerance')
            output = self._normal_and_tangent(system, solution[:-1])
        self.guess.copy_(solution)
        output = output.cpu().numpy()
        velocity = output[0]+1j*output[1]
        if details:
            return velocity, output[2], output[3], output[4]
        return velocity


@lru_cache(maxsize=1)
def _quadrature_kernel(torch):
    from ._cuda import QuadratureKernel

    try:
        return QuadratureKernel(torch)
    except (RuntimeError, OSError) as error:
        warnings.warn(f'Using PyTorch CUDA quadrature: {error}', RuntimeWarning)
        return None
