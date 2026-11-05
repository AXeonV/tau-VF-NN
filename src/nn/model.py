"""2-D Fourier neural operator predicting a spatial arrival-time correction."""
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv(nn.Module):
    def __init__(self,width,modes):
        super().__init__();self.modes=modes
        scale=1/width
        self.positive=nn.Parameter(scale*torch.randn(width,width,modes,modes,2))
        self.negative=nn.Parameter(scale*torch.randn(width,width,modes,modes,2))

    def forward(self,x):
        spectrum=torch.fft.rfft2(x)
        out=torch.zeros_like(spectrum);m=self.modes
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',spectrum[:,:,:m,:m],torch.view_as_complex(self.positive))
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',spectrum[:,:,-m:,:m],torch.view_as_complex(self.negative))
        return torch.fft.irfft2(out,s=x.shape[-2:])


class TauFNO(nn.Module):
    def __init__(self,width=24,modes=8,depth=4,padding=6):
        super().__init__();self.padding=padding
        self.lift=nn.Conv2d(5,width,1)
        self.spectral=nn.ModuleList([SpectralConv(width,modes) for _ in range(depth)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(depth)])
        self.project=nn.Sequential(nn.Conv2d(width,48,1),nn.GELU(),nn.Conv2d(48,1,1))
        nn.init.zeros_(self.project[-1].weight);nn.init.zeros_(self.project[-1].bias)
        self.float()

    def forward(self,features):
        x=F.pad(self.lift(features),(0,self.padding,0,self.padding))
        for spectral,local in zip(self.spectral,self.local):x=F.gelu(spectral(x)+local(x))
        if self.padding:x=x[:,:,:-self.padding,:-self.padding]
        return 1.5*torch.tanh(self.project(x))
