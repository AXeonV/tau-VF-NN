"""Deterministic FP64 full-interval physical refinement of a fixed-case NN."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from .fixed_model import NeuralInterface,FreeBoundaryGeometry,impose_initial_velocity
from .physics import BoundaryPressure
from .fixed_case import physical_terms,evaluate


def load_model(path,device):
    saved=torch.load(path,map_location=device,weights_only=True)
    cfg=saved['config'];state=saved['state_dict']
    model=NeuralInterface(state['u0'],state['scale'],cfg.get('T',1000.),cfg['width'],cfg['depth'],float(state.get('time_scale',.5))).to(device)
    model.load_state_dict(state,strict=False)
    return model,cfg


def time_quadrature(T,eps,count,log_points=None,rule='gauss'):
    log_points=count//2 if log_points is None else log_points
    if not 0<log_points<count:raise ValueError('Time quadrature needs both linear and logarithmic points')
    arrays=[];weights=[]
    for logarithmic,n in [(False,count-log_points),(True,log_points)]:
        if rule=='gauss':
            q,w=np.polynomial.legendre.leggauss(n);s=(q+1)/2;w=w/4
        else:s=(np.arange(n)+.5)/n;w=np.full(n,.5/n)
        arrays.append(eps*np.expm1(s*np.log1p(T/eps)) if logarithmic else s*T)
        weights.append(w)
    t=np.concatenate(arrays+[np.array([0,T])]);w=np.concatenate(weights+[np.array([.01,.01])])
    return t,w/w.sum()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='results/nn/ellipse_long_accuracy/model.pt')
    p.add_argument('--out',default='results/nn/ellipse_long_accuracy')
    p.add_argument('--nodes',type=int,default=1024)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--time-points',type=int,default=160)
    p.add_argument('--log-points',type=int)
    p.add_argument('--quadrature',choices=['gauss','midpoint'],default='gauss')
    p.add_argument('--early-nodes',type=int,default=0)
    p.add_argument('--early-until',type=float,default=200.)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--early-batch',type=int,default=128)
    p.add_argument('--append-history',action='store_true')
    p.add_argument('--lr',type=float,default=.5)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model,cfg=load_model(a.checkpoint,a.device);T=model.T;modes=cfg['modes'];sigma=cfg['sigma']
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    impose_initial_velocity(model,a.nodes,modes,sigma)
    geometry=FreeBoundaryGeometry(a.nodes,modes).to(a.device);pressure=BoundaryPressure(a.nodes).to(a.device)
    time_scale=float(model.time_scale)
    times,weights=time_quadrature(T,time_scale,a.time_points,a.log_points,a.quadrature)
    if a.append_history:
        previous=json.loads((out/'refinement_config.json').read_text())
        for key in ['nodes','time_points','log_points','quadrature','early_nodes','early_until']:
            if previous.get(key)!=vars(a)[key]:raise ValueError(f'Cannot append loss histories with different {key}')
    (out/'refinement_config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    runs=out/'refinement_runs.json'
    stages=json.loads(runs.read_text()) if a.append_history and runs.exists() else []
    if a.append_history and not stages:stages=[previous]
    stages.append(vars(a));runs.write_text(json.dumps(stages,indent=2),encoding='utf-8')
    groups=[]
    if a.early_nodes:
        early_geometry=FreeBoundaryGeometry(a.early_nodes,modes).to(a.device);early_pressure=BoundaryPressure(a.early_nodes).to(a.device)
        groups.append((np.flatnonzero(times<=a.early_until),early_geometry,early_pressure))
        groups.append((np.flatnonzero(times>a.early_until),geometry,pressure))
    else:groups.append((np.arange(len(times)),geometry,pressure))
    batches=[]
    for ids,g,p in groups:
        chunk=a.early_batch if a.early_nodes and g.n==a.early_nodes else a.batch
        batches.extend((part,g,p) for part in np.array_split(ids,max(1,(len(ids)+chunk-1)//chunk)) if len(part))
    tt=torch.tensor(times,device=a.device);ww=torch.tensor(weights,device=a.device)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=a.lr,max_iter=a.steps,max_eval=a.steps*2,history_size=60,tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    history=np.load(out/'refinement_history.npy').tolist() if a.append_history else []
    elapsed_offset=history[-1][3] if history else 0.
    started=time.time();best=float('inf');best_state=None
    def closure():
        nonlocal best,best_state
        optimizer.zero_grad();total=0.;kin=0.
        for ids,g,p in batches:
            per,metric,*_=physical_terms(model,g,p,tt[ids],sigma)
            physics=(ww[ids]*per).sum();loss=physics+1e-6*ww[ids].sum()*metric
            loss.backward();total+=float(loss.detach());kin+=float(physics.detach())
        if not np.isfinite(total):raise RuntimeError('Non-finite physical objective')
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise RuntimeError('Non-finite physical gradient')
        history.append([len(history),total,kin,elapsed_offset+time.time()-started])
        if total<best:
            best=total;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if len(history)%10==0:
            print(f'full physical evaluation={len(history)} loss={total:.6e} best={best:.6e} seconds={time.time()-started:.1f}',flush=True)
            np.save(out/'refinement_history.npy',np.array(history))
            torch.save({'config':cfg,'state_dict':best_state},out/'refined.pt')
        return torch.tensor(total,dtype=torch.float64,device=a.device)
    closure();optimizer.step(closure);closure()
    model.load_state_dict(best_state)
    torch.save({'config':cfg,'state_dict':model.state_dict()},out/'model.pt')
    np.save(out/'refinement_history.npy',np.array(history))
    t=torch.tensor(np.unique(np.r_[np.linspace(0,T,65),time_scale*np.expm1(np.linspace(0,np.log1p(T/time_scale),257))]),device=a.device)
    audit=evaluate(model,a.nodes,modes,t,sigma)
    fine=evaluate(model,2*a.nodes,modes,t,sigma,chunk=1)
    (out/'refinement_metrics.json').write_text(json.dumps({'objective':best,'audit':audit,'refined_audit':fine,'seconds':time.time()-started},indent=2),encoding='utf-8')
    print(json.dumps({'objective':best,'audit':audit,'refined_audit':fine,'seconds':time.time()-started},indent=2),flush=True)


if __name__=='__main__':main()
