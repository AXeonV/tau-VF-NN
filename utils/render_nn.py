"""Plot a trained arrival field and physics-only training losses."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def render(folder):
    folder=Path(folder);p=np.load(folder/'prediction.npz')
    history=np.load(folder/'history.npy');validation=np.load(folder/'validation.npy')
    config=json.loads((folder/'config.json').read_text())
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fc','savefig.facecolor':'#f6f8fc'})
    fig,ax=plt.subplots(figsize=(7,6.5),layout='constrained')
    tau=p['tau'].copy()
    if 'correction' in p:
        import torch
        import torch.nn.functional as F
        from src.nn.conditions import features
        # Display the same bilinear correction used by the physics adapter.
        _,base,_=features(torch.tensor(p['condition'],dtype=torch.float64),256)
        correction=F.interpolate(torch.tensor(p['correction'])[None,None],size=(256,256),mode='bilinear',align_corners=True)
        tau=(base*torch.exp(correction[:,0].double()))[0].numpy()
    tau[tau>config['T']]=np.nan
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#e5ebf2')
    extent=[p['x'][0],p['x'][-1],p['y'][0],p['y'][-1]]
    im=ax.imshow(tau,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=config['T'])
    for i in [0,25,50,75,100]:
        z=p['z'][i];ax.plot(np.r_[z.real,z.real[0]],np.r_[z.imag,z.imag[0]],color='white',lw=.8,alpha=.85)
    ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y');ax.set_title('FNO 预测的 τ 场：未参与训练的初始条件')
    bound=np.abs(p['z']).max()*1.15;ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
    fig.colorbar(im,ax=ax,label='到达时间 τ');fig.savefig(folder/'distribution.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.semilogy(history[:,0],history[:,1],alpha=.35,lw=.7,label='训练：随机条件／递增时间窗')
    ax.semilogy(validation[:,0],validation[:,1],color='#d46a43',lw=2,label='固定留出条件物理 loss')
    ax.set_xlabel('梯度更新 step');ax.set_ylabel('Loss');ax.set_title('无轨迹标签训练');ax.grid(True,which='both',alpha=.2);ax.legend()
    fig.savefig(folder/'loss.png',dpi=170);plt.close(fig)
    print('Saved distribution.png and loss.png')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='results/nn/pino_pilot');render(parser.parse_args().case)
