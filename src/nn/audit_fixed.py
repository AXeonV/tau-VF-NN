"""Independent FP64 quadrature/time and geometric checks; export NN states."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from .refine_fixed import load_model,time_quadrature
from .fixed_model import FreeBoundaryGeometry
from .physics import BoundaryPressure
from .fixed_case import physical_terms


def crossing_count(z):
    # Segment intersection on a polygon; adjacent edges share a vertex normally.
    r=torch.roll(z,-1)-z
    difference=z[None,:]-z[:,None]
    denominator=(r[:,None].conj()*r[None,:]).imag
    nonparallel=denominator.abs()>1e-12
    safe=torch.where(nonparallel,denominator,torch.ones_like(denominator))
    s=(difference.conj()*r[None,:]).imag/safe
    t=(difference.conj()*r[:,None]).imag/safe
    index=torch.arange(len(z),device=z.device)
    distance=(index[:,None]-index[None,:]).abs()
    nonlocal_edge=torch.minimum(distance,len(z)-distance)>1
    crossed=nonparallel&nonlocal_edge&(s>1e-8)&(s<1-1e-8)&(t>1e-8)&(t<1-1e-8)
    return int(crossed.sum())//2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',default='results/nn/ellipse_long_accuracy')
    p.add_argument('--nodes',type=int,default=1024)
    p.add_argument('--time-points',type=int,default=192)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();folder=Path(a.case)
    model,cfg=load_model(folder/'model.pt',a.device);T=model.T;modes=cfg['modes'];sigma=cfg['sigma'];eps=float(model.time_scale)
    ref_config=folder/'refinement_config.json'
    rc=json.loads(ref_config.read_text()) if ref_config.exists() else {}
    training_t,weights=time_quadrature(T,eps,rc.get('time_points',a.time_points),rc.get('log_points'),rc.get('quadrature','gauss'))
    independent_t=np.unique(np.r_[np.linspace(0,T,257),eps*np.expm1(np.linspace(0,np.log1p(T/eps),513)),np.geomspace(1e-7,eps,65)])
    report={'T':T,'precision':'float64','training_nodes':a.nodes,'early_training_nodes':rc.get('early_nodes',0) or a.nodes,'audit_nodes':2*a.nodes,'training_time_points':len(training_t),'time_quadrature':rc.get('quadrature','gauss'),'independent_times':len(independent_t),'retreat_allowed':True}
    rows={};coarse_speed=[];pressure_error=[];crossings=[];point_max=0.
    with torch.no_grad():
        for n in [a.nodes,2*a.nodes]:
            geometry=FreeBoundaryGeometry(n,modes).to(a.device);pressure=BoundaryPressure(n).to(a.device)
            per_list=[];objective=0.
            for ids in np.array_split(np.arange(len(training_t)),(len(training_t)+3)//4):
                t=torch.tensor(training_t[ids],device=a.device)
                per,metric,*_=physical_terms(model,geometry,pressure,t,sigma)
                objective+=float((per*torch.tensor(weights[ids],device=a.device)).sum()+1e-6*float(weights[ids].sum())*metric)
            for ids in np.array_split(np.arange(len(independent_t)),(len(independent_t)+1)//2):
                t=torch.tensor(independent_t[ids],device=a.device)
                per,_,vg,vp,z,residual=physical_terms(model,geometry,pressure,t,sigma);per_list.extend(per.cpu().tolist())
                if n==a.nodes:coarse_speed.extend(vp.cpu())
                else:
                    point_max=max(point_max,float(residual.abs().max()))
                    coarse=torch.stack([coarse_speed[j] for j in ids]).to(a.device)
                    error=(vp[:,::2]-coarse)*torch.sqrt(1+2*t[:,None])
                    pressure_error.extend(error.square().mean(dim=1).sqrt().cpu().tolist())
                    crossings.extend(crossing_count(curve[::2]) for curve in z)
            rows[n]=np.array(per_list)
            report[f'objective_{n}']=objective
            report[f'independent_rms_{n}']=float(np.sqrt(np.mean(per_list)))
            report[f'worst_time_rms_{n}']=float(np.sqrt(max(per_list)))
            print(f'audit nodes={n} objective={objective:.6e} independent RMS={report[f"independent_rms_{n}"]:.6e}',flush=True)
        ids=np.argsort(rows[2*a.nodes])[-16:]
        report['worst_times']=[{'t':float(independent_t[j]),'rms':float(np.sqrt(rows[2*a.nodes][j]))} for j in ids[::-1]]
        special=torch.tensor([0.,.0001,.01,.1,1,10,30,100,300,500,700,1000],device=a.device,dtype=torch.float64)
        g1=FreeBoundaryGeometry(a.nodes,modes).to(a.device);g2=FreeBoundaryGeometry(2*a.nodes,modes).to(a.device)
        p1=BoundaryPressure(a.nodes).to(a.device);p2=BoundaryPressure(2*a.nodes).to(a.device)
        areas=[];flux=[];initial_error=None
        for t in special:
            tt=t[None];u,ud=model(tt)
            z,s,n,k,vg=g1(u,tt,ud);v1=p1(z,s,n,k,tt.new_full((1,),sigma))
            zz,ss,nn,kk,vgg=g2(u,tt,ud);v2=p2(zz,ss,nn,kk,tt.new_full((1,),sigma))
            pressure_error.append(float(torch.sqrt((((v1-v2[:,::2])*torch.sqrt(1+2*tt[:,None]))**2).mean())))
            flux.append(float((v2*ss).mean()-1))
            alpha=torch.fft.fftfreq(zz.shape[1],d=1/zz.shape[1],device=a.device,dtype=torch.float64)
            za=torch.fft.ifft(1j*alpha*torch.fft.fft(zz))
            areas.append(float(((zz.conj()*za).imag.mean()/(1+2*t)-1).abs()))
            crossings.append(crossing_count(zz[0,::2]))
            if float(t)==0:initial_error=float((zz.real.square()/3+3*zz.imag.square()-1).abs().max())
        report.update(max_pressure_refinement_rms=max(pressure_error),max_area_relative_error=max(areas),max_flux_relative_error=max(abs(v) for v in flux),max_self_intersections=max(crossings),self_intersection_checked_times=len(crossings),initial_ellipse_equation_max_error=initial_error,max_point_normalized_residual=point_max)
        times=np.unique(np.r_[np.linspace(0,T,801),eps*np.expm1(np.linspace(0,np.log1p(T/eps),801))])
        curves=[]
        for values in np.array_split(times,32):
            tt=torch.tensor(values,device=a.device);u,ud=model(tt);curves.append(g2(u,tt,ud)[0].cpu().numpy())
        np.savez_compressed(folder/'optimized.npz',times=times,z=np.concatenate(curves),theta=np.arange(2*a.nodes)*2*np.pi/(2*a.nodes))
    report['accuracy_targets']={'independent_rms':.005,'worst_time_rms':.02,'pressure_refinement_rms':.001,'area_error':1e-10,'self_intersections':0}
    report['targets_met']=bool(report[f'independent_rms_{2*a.nodes}']<=.005 and report[f'worst_time_rms_{2*a.nodes}']<=.02 and max(pressure_error)<=.001 and max(areas)<=1e-10 and max(crossings)==0)
    (folder/'accuracy.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'audit_samples.npz',times=independent_t,coarse_loss=rows[a.nodes],fine_loss=rows[2*a.nodes])
    metrics=dict(T=T,aspect=3.,sigma=sigma,final_loss=report[f'objective_{2*a.nodes}'],initial_loss=float(np.load(folder/'refinement_history.npy')[0,1]),plot_history_file='refinement_history.npy',step_label='物理 loss 评估次数（含线搜索）',interface_times=[0,100,300,500,750,1000],auxiliary_state='NN closed interface history',accuracy_targets_met=report['targets_met'])
    (folder/'optimization_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
