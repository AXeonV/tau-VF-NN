"""Physics-only conditional FNO training. No saved trajectory is loaded."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from .model import TauFNO
from .conditions import sample_conditions,features
from .physics import TauPhysicsLoss


def train(args):
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    validation_generator=torch.Generator(device=device).manual_seed(args.seed+10000)
    holdout=sample_conditions(16,device,validation_generator)
    holdout_times=torch.linspace(.08,args.T-.02,len(holdout),device=device,dtype=torch.float64)
    model=TauFNO(args.width,args.modes,args.depth).to(device)
    physics=TauPhysicsLoss(args.angles).to(device)
    independent=TauPhysicsLoss(96,h=.005).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.05)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config.update(device=str(device),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        training='physics only; online initial shapes; no paired evolution labels',
        input='initial signed radial distance, soft air mask, x, y, log sigma',
        output='tau(x,y)=max((r^2-r0(theta)^2)/2,0)*exp(FNO correction)',
        adapter='monotone star-shaped level sets; angular Fourier cutoff 8',parameters=sum(p.numel() for p in model.parameters()))
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    def validate(backend):
        with torch.no_grad():
            inputs,_,_=features(holdout,args.grid)
            value,parts=backend(model(inputs),holdout,holdout_times)
        return float(value),{k:float(v) for k,v in parts.items()}
    initial,_=validate(physics)
    validation=[[0,initial]];history=[];started=time.time();best=initial
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for step in range(1,args.steps+1):
        model.train();p=sample_conditions(args.batch,device,generator)
        horizon=args.T*(.25+.75*min(1.,step/(args.steps*.4)))
        t=.02+(horizon-.04)*torch.rand(args.batch,device=device,generator=generator,dtype=torch.float64)
        inputs,_,_=features(p,args.grid)
        optimizer.zero_grad(set_to_none=True)
        loss,parts=physics(model(inputs),p,t)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at step {step}')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError(f'Non-finite gradient at step {step}')
        optimizer.step();scheduler.step()
        history.append([step,float(loss.detach()),*(float(parts[k].detach()) for k in ('kinetic','retreat','area','level')),horizon])
        if step%50==0 or step==args.steps:
            model.eval();value,_=validate(physics);validation.append([step,value])
            if value<best:
                best=value;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'step={step} train={float(loss):.5e} heldout={value:.5e} best={best:.5e} seconds={time.time()-started:.1f}',flush=True)
            torch.save({'state_dict':best_state,'config':config,'best_validation_loss':best},out/'model.pt')
            np.save(out/'history.npy',np.array(history));np.save(out/'validation.npy',np.array(validation))
    model.load_state_dict(best_state);model.eval()
    independent_loss,independent_parts=validate(independent)
    test_generator=torch.Generator(device=device).manual_seed(args.seed+20000)
    test=sample_conditions(32,device,test_generator)
    test_t=torch.linspace(.05,args.T-.02,len(test),device=device,dtype=torch.float64)
    with torch.no_grad():
        test_inputs,_,_=features(test,args.grid)
        test_field=model(test_inputs)
        test_loss,_=independent(test_field,test,test_t)
        test_initial_loss,_=independent(torch.zeros_like(test_field),test,test_t)
    with torch.no_grad():
        sample=holdout[0:1];inputs,base,axis=features(sample,args.grid)
        correction=model(inputs)
        tau=base*torch.exp(correction[:,0].double())
        times=torch.linspace(.02,args.T-.02,101,device=device,dtype=torch.float64)
        fields=correction.expand(len(times),-1,-1,-1);conditions=sample.expand(len(times),-1)
        r,_=independent.level_sets(fields,conditions,times)
        z=r[:,1]*torch.exp(1j*independent.angle)
    np.savez_compressed(out/'prediction.npz',tau=tau[0].cpu().numpy(),correction=correction[0,0].cpu().numpy(),x=axis.cpu().numpy(),y=axis.cpu().numpy(),times=times.cpu().numpy(),z=z.cpu().numpy(),condition=sample.cpu().numpy())
    report=dict(initial_heldout_loss=initial,best_heldout_loss=best,independent_loss=independent_loss,independent_parts=independent_parts,test_loss=float(test_loss),test_initial_loss=float(test_initial_loss),seconds=time.time()-started,steps=args.steps,initial_conditions_used=args.batch*args.steps,training_labels=0,device=str(device))
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--grid',type=int,default=32)
    parser.add_argument('--angles',type=int,default=48)
    parser.add_argument('--T',type=float,default=1.)
    parser.add_argument('--width',type=int,default=24)
    parser.add_argument('--modes',type=int,default=8)
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--out',default='results/nn/pino_pilot')
    args=parser.parse_args();train(args)


if __name__=='__main__':main()
