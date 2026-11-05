"""Predict an arrival field from an initial condition without optimization."""
from pathlib import Path
import argparse
import numpy as np
import torch
from .model import TauFNO
from .conditions import features,radius


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='results/nn/pino_pilot/model.pt')
    parser.add_argument('--aspect',type=float,default=1.4)
    parser.add_argument('--rotation',type=float,default=.2)
    parser.add_argument('--sigma',type=float,default=.015)
    parser.add_argument('--cos',type=float,nargs=4,default=[0.]*4)
    parser.add_argument('--sin',type=float,nargs=4,default=[0.]*4)
    parser.add_argument('--out',default='results/nn/prediction.npz')
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    if args.aspect<=0 or args.sigma<=0:parser.error('aspect and sigma must be positive')
    device=args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
    cfg=checkpoint['config'];model=TauFNO(cfg['width'],cfg['modes'],cfg['depth']).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    p=torch.zeros((1,12),device=device,dtype=torch.float64)
    p[0,0]=np.log(args.aspect);p[0,1]=args.rotation
    p[0,2:6]=p.new_tensor(args.cos);p[0,6:10]=p.new_tensor(args.sin)
    p[0,10]=args.sigma;p[0,11]=1.
    angle=torch.arange(256,device=device,dtype=torch.float64)*2*torch.pi/256
    p[0,11]=radius(p,angle[None]).square().mean().sqrt()
    with torch.no_grad():
        inputs,base,axis=features(p,cfg['grid'])
        tau=base*torch.exp(model(inputs)[:,0].double())
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,tau=tau[0].cpu().numpy(),x=axis.cpu().numpy(),y=axis.cpu().numpy(),condition=p.cpu().numpy(),T=cfg['T'])
    print(f'Saved {out} using {device}; no physics solve or optimization during prediction.')


if __name__=='__main__':main()
