"""Save only the optimized interface distribution and loss curve."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import CubicSpline


def render(case):
    folder=Path(case);data=np.load(folder/'optimized.npz')
    config=json.loads((folder/'optimization_metrics.json').read_text())
    history=np.load(folder/config.get('plot_history_file','history.npy'))
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.facecolor':'#f6f8fc','axes.facecolor':'#edf2f8','savefig.facecolor':'#f6f8fc','axes.titleweight':'bold','grid.alpha':.2})
    times=data['times'];z=data['z'];loss=float(config.get('final_loss',history[-1,1]))
    # Periodic interpolation displays the saved spatial representation smoothly.
    angle=np.linspace(0,2*np.pi,z.shape[1]+1)
    fine_angle=np.linspace(0,2*np.pi,max(4096,4*z.shape[1]))
    initial=CubicSpline(angle,np.r_[z[0],z[0,0]],bc_type='periodic')(fine_angle)
    final=CubicSpline(angle,np.r_[z[-1],z[-1,0]],bc_type='periodic')(fine_angle)
    fig,ax=plt.subplots(figsize=(9,9),layout='constrained')
    if 'interface_times' in config:
        requested=config['interface_times']
        palette=plt.get_cmap('Blues')(np.linspace(.35,.95,len(requested)-1))
        ax.plot(initial.real,initial.imag,color='#626c77',lw=1.,label='t = 0')
        for target,color in zip(requested[1:],palette):
            j=int(np.argmin(abs(times-target)))
            curve=CubicSpline(angle,np.r_[z[j],z[j,0]],bc_type='periodic')(fine_angle)
            ax.plot(curve.real,curve.imag,color=color,lw=1.35,label=f't = {times[j]:g}')
        ax.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='none',fontsize=10)
    else:
        ax.fill(final.real,final.imag,color='#56a4bc',alpha=.98)
        ax.plot(final.real,final.imag,color='#244c65',lw=1.15)
        ax.plot(initial.real,initial.imag,color='white',lw=1.1)
    bound=max(abs(final.real).max(),abs(final.imag).max())*1.055
    ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_aspect('equal')
    ax.set_xlabel('x / R0');ax.set_ylabel('y / R0')
    title='NN 界面状态曲线' if 'interface_times' in config else '优化后的界面分布'
    ax.set_title(f'{title}   T = {times[-1]:g}',fontsize=18,pad=15)
    ax.text(.02,.025,f'初始椭圆轴比 {config["aspect"]:g}:1 · 无额外微扰\nσ = {config["sigma"]:g}   loss = {loss:.4e}',transform=ax.transAxes,fontsize=10,bbox=dict(fc='white',ec='none',alpha=.92,pad=6))
    fig.savefig(folder/'distribution.png',dpi=190,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    ax.semilogy(history[:,0],history[:,1],color='#246c96',lw=1.8)
    ax.scatter(history[-1,0],history[-1,1],color='#dc7952',s=28,zorder=5)
    ax.set_xlabel(config.get('step_label','Step（loss 评估次数，含线搜索）'));ax.set_ylabel('Loss（对数坐标）')
    ax.set_title('Loss 随 step 的变化',fontsize=17,pad=12);ax.grid(True,which='both',alpha=.22)
    ax.text(.98,.95,f'初始 loss = {history[0,1]:.6e}\n最终 loss = {history[-1,1]:.6e}',transform=ax.transAxes,ha='right',va='top',fontsize=11,bbox=dict(fc='white',ec='none',pad=7))
    fig.savefig(folder/'loss.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(f'{folder}: loss={loss:.6e}; saved distribution.png and loss.png')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--case',default='results/ellipse_long')
    render(p.parse_args().case)


if __name__=='__main__':main()
