"""High-accuracy single-case NN initialization and physics-only refinement."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from .fixed_model import NeuralInterface,FreeBoundaryGeometry,impose_initial_velocity
from .physics import BoundaryPressure


def initial_data(path,modes):
    source=np.load(path);t=source['times'];z=source['z'];angle=source['theta']
    k=np.arange(1,modes+1,2)
    normalized=z/np.sqrt(1+2*t[:,None])
    c=np.c_[2/len(angle)*normalized.real@np.cos(angle[:,None]*k),2/len(angle)*normalized.imag@np.sin(angle[:,None]*k)]
    scale=np.maximum(np.max(abs(c-c[0]),axis=0),1e-6)
    return float(t[-1]),c[0],scale,CubicSpline(t,c,axis=0)


def time_samples(count,T,device,time_scale=.001):
    u=torch.rand(count,device=device,dtype=torch.float64)
    logarithmic=torch.rand(count,device=device)<.5
    return torch.where(logarithmic,time_scale*torch.expm1(u*np.log1p(T/time_scale)),u*T)


def physical_terms(model,geometry,pressure,t,sigma):
    u,ud=model(t);z,metric,normal,kappa,vg=geometry(u,t,ud)
    vp=pressure(z,metric,normal,kappa,t.new_full(t.shape,sigma))
    residual=(vg-vp)*torch.sqrt(1+2*t[:,None])
    weight=metric/metric.mean(dim=1,keepdim=True)
    per_time=(weight*residual.square()).mean(dim=1)
    metric_loss=(metric/metric.mean(dim=1,keepdim=True)-1).square().mean()
    return per_time,metric_loss,vg,vp,z,residual


def evaluate(model,n,modes,t,sigma,chunk=2):
    device=next(model.parameters()).device
    geometry=FreeBoundaryGeometry(n,modes).to(device);pressure=BoundaryPressure(n).to(device)
    losses=[];mins=[];maxima=[];speeds=[]
    with torch.no_grad():
        for batch in t.split(chunk):
            per,_,vg,vp,z,res=physical_terms(model,geometry,pressure,batch,sigma)
            losses.extend(per.cpu().tolist());mins.append(float(vg.min()));maxima.append(float(res.abs().max()));speeds.append(float(vp.min()))
    return dict(loss=float(np.mean(losses)),rms=float(np.sqrt(np.mean(losses))),worst_time_rms=float(np.sqrt(max(losses))),max_point_residual=max(maxima),min_geometric_speed=min(mins),min_pressure_speed=min(speeds),nodes=n,times=len(t))


def save_model(model,args,out,name):
    cfg=vars(args).copy();cfg['T']=model.T
    torch.save({'config':cfg,'state_dict':model.state_dict()},out/name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initialization',default='results/nn/ellipse_long_accuracy/initialization.npz')
    parser.add_argument('--out',default='results/nn/ellipse_long_accuracy')
    parser.add_argument('--modes',type=int,default=256)
    parser.add_argument('--nodes',type=int,default=1024)
    parser.add_argument('--fit-steps',type=int,default=12000)
    parser.add_argument('--physics-steps',type=int,default=2000)
    parser.add_argument('--fit-lr',type=float,default=5e-4)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--width',type=int,default=192)
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--sigma',type=float,default=.015)
    parser.add_argument('--seed',type=int,default=71)
    parser.add_argument('--resume')
    parser.add_argument('--hard-initial-velocity',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--time-scale',type=float,default=.001)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();device=torch.device(args.device)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    T,u0,scale,spline=initial_data(args.initialization,args.modes)
    model=NeuralInterface(torch.tensor(u0),torch.tensor(scale),T,args.width,args.depth,args.time_scale).to(device)
    if args.resume:model.load_state_dict(torch.load(args.resume,map_location=device,weights_only=True)['state_dict'],strict=False)
    if args.hard_initial_velocity:impose_initial_velocity(model,args.nodes,args.modes,args.sigma)
    cfg=vars(args).copy();cfg.update(T=T,precision='float64',representation='time Fourier-feature MLP -> free closed interface history',symmetry='exact ellipse reflection symmetries',warm_start='initialization fitting only; no trajectory term in physical refinement')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    started=time.time();fit_history=[]
    if args.fit_steps:
        bank_t=np.unique(np.r_[np.linspace(0,T,4097),args.time_scale*np.expm1(np.linspace(0,np.log1p(T/args.time_scale),4097))])[1:]
        target=torch.tensor(spline(bank_t),device=device)
        target_d=torch.tensor(spline(bank_t,1),device=device)
        bank_t=torch.tensor(bank_t,device=device)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.fit_lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.fit_steps,eta_min=args.fit_lr*.05)
        for step in range(1,args.fit_steps+1):
            ids=torch.randint(len(bank_t),(128,),device=device);t=bank_t[ids]
            u,ud=model(t)
            value=((u-target[ids])/model.scale).square().mean()
            derivative=(((ud-target_d[ids])*(args.time_scale+t[:,None]))/model.scale).square().mean()
            loss=value+.01*derivative
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();scheduler.step()
            if step%100==0:fit_history.append([step,float(loss.detach()),float(value.detach()),float(derivative.detach())])
            if step%1000==0:
                print(f'weight initialization {step} loss={float(loss):.4e} seconds={time.time()-started:.1f}',flush=True)
                save_model(model,args,out,'initialized.pt');np.save(out/'initialization_history.npy',np.array(fit_history))
        save_model(model,args,out,'initialized.pt');np.save(out/'initialization_history.npy',np.array(fit_history))
    geometry=FreeBoundaryGeometry(args.nodes,args.modes).to(device);pressure=BoundaryPressure(args.nodes).to(device)
    validation_t=torch.tensor(np.unique(np.r_[np.linspace(0,T,17),args.time_scale*np.expm1(np.linspace(0,np.log1p(T/args.time_scale),33))]),device=device)
    initial=evaluate(model,args.nodes,args.modes,validation_t,args.sigma)
    print('initial physical audit',json.dumps(initial),flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,args.physics_steps),eta_min=args.lr*.1)
    history=[];audits=[];best=initial['loss'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for step in range(1,args.physics_steps+1):
        t=time_samples(args.batch,T,device)
        if step%4==0:t[0]=0.
        elif step%4==1:t[0]=T
        per,metric,_,_,_,_=physical_terms(model,geometry,pressure,t,args.sigma)
        loss=per.mean()+1e-5*metric
        if not torch.isfinite(loss):raise RuntimeError('Non-finite physical loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Non-finite physical gradient')
        optimizer.step();scheduler.step();history.append([step,float(loss.detach()),float(per.mean().detach()),float(metric.detach())])
        if step%100==0 or step==args.physics_steps:
            audit=evaluate(model,args.nodes,args.modes,validation_t,args.sigma)
            audits.append([step,audit['loss'],audit['worst_time_rms']])
            if audit['loss']<best:best=audit['loss'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'physical step={step} train={float(loss):.4e} audit={audit["loss"]:.4e} worst={audit["worst_time_rms"]:.3e} seconds={time.time()-started:.1f}',flush=True)
            np.save(out/'history.npy',np.array(history));np.save(out/'audit_history.npy',np.array(audits));save_model(model,args,out,'last.pt')
    model.load_state_dict(best_state);save_model(model,args,out,'model.pt')
    dense_t=torch.tensor(np.unique(np.r_[np.linspace(0,T,49),np.expm1(np.linspace(0,np.log1p(T),49))]),device=device)
    audit=evaluate(model,args.nodes,args.modes,dense_t,args.sigma)
    refined=evaluate(model,2*args.nodes,args.modes,dense_t,args.sigma,chunk=1)
    times=np.unique(np.r_[np.linspace(0,T,401),np.expm1(np.linspace(0,np.log1p(T),401))])
    fine_geometry=FreeBoundaryGeometry(2*args.nodes,args.modes).to(device)
    curves=[]
    with torch.no_grad():
        for values in np.array_split(times,32):
            t=torch.tensor(values,device=device);u,ud=model(t);z,*_=fine_geometry(u,t,ud);curves.append(z.cpu().numpy())
    np.savez_compressed(out/'optimized.npz',times=times,z=np.concatenate(curves),theta=np.arange(2*args.nodes)*2*np.pi/(2*args.nodes))
    metrics=dict(T=T,aspect=3.,sigma=args.sigma,initial_physical_audit=initial,physical_audit=audit,refined_audit=refined,initial_loss=initial['loss'],final_loss=refined['loss'],seconds=time.time()-started,step_label='梯度更新 step（物理训练）',interface_times=[0,.1*T,.3*T,.5*T,.75*T,T],auxiliary_state='closed interface history; retreat allowed',fit_steps=args.fit_steps,physics_steps=args.physics_steps)
    (out/'optimization_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
