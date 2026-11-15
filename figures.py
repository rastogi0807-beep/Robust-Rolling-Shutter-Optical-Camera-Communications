"""High-resolution manuscript figures. Reads simulation outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from simulate import sim_frame, estimate, GRID, T, PILOT, S, N_P, N_D, N_R, r, overlap, P_R, D, Dpinv, Tcenter, TDEN
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'figures';P.mkdir(exist_ok=True)
R=ROOT/'results'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                     'figure.dpi':130,'savefig.dpi':600,'axes.spines.top':False,'axes.spines.right':False})
colors={'Raw midpoint threshold':'#83949d','Nominal model (fixed timing)':'#ba985a','Data-only gradient boosting':'#457ca0',
        'Calibrated physics inverse':'#3c9783','Proposed physics + residual ML':'#6b56a7'}
methods=list(colors)

def save(fig,name):
    fig.savefig(P/name,dpi=600,bbox_inches='tight',facecolor='white',pad_inches=.11)
    plt.close(fig)

# Figure 1 3 lane pictorial diagram
fig,ax=plt.subplots(figsize=(10.5,5.25)); ax.set_xlim(0,10.5);ax.set_ylim(0,5.2);ax.axis('off')
def block(x,y,w,h,label,fc='#f0f5f8',ec='#456b83',fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.055,rounding_size=0.10',linewidth=1.35,facecolor=fc,edgecolor=ec))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fontsize,linespacing=1.25)
def arrow(x1,y1,x2,y2,color='#4b5563'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.4,color=color))
ax.text(.1,4.83,'A. Optical transmission and rolling-shutter acquisition',weight='bold',fontsize=12)
block(.25,3.61,1.85,.9,'PRBS pilot + payload\nLED driver',fc='#f1f5fa')
ax.add_patch(Rectangle((2.48,3.77),.32,.58,color='#f9ca61',ec='#8b692c'))
ax.text(2.63,3.5,'LED / diffuser',ha='center',fontsize=9)
for dy in [-.22,0,.22]:arrow(2.85,4.05+dy,4.05,4.05+1.3*dy,color='#daa433')
block(4.12,3.6,2.2,.95,'Rolling CMOS\nrow-integrated samples',fc='#e9f8ef',ec='#31845b')
arrow(6.42,4.06,7.02,4.06)
block(7.05,3.60,3.05,.95,'Row signal  y[r]  + ambient\nunknown exposure and phase',fc='#f1f4f7')
ax.plot([0,10.3],[3.24,3.24],lw=1,color='#e0e4ea')
ax.text(.1,2.9,'B. Receiver-side estimation and decoding',weight='bold',fontsize=12)
for x,w,txt,fc in [(0.18,2.1,'Preamble\nphotometric fit','#edf6fa'),(2.74,2.1,'Exposure + phase\ngrid calibration','#edf6fa'),
                   (5.3,2.05,'Physics-guided\nregularized inverse','#eaf5ef'),(7.85,2.25,'Residual-gradient\nboosting classifier','#f5effb')]:
    block(x,1.70,w,.86,txt,fc=fc)
for a,b in [(2.30,2.7),(4.87,5.25),(7.4,7.79)]:arrow(a,2.13,b,2.13)
block(3.1,.4,4.4,.65,'Decoded binary payload + error accounting',fc='#fff5df',ec='#b38836')
arrow(8.96,1.63,8.96,.75);arrow(8.96,.75,7.53,.75)
ax.text(.12,.15,'Not to scale. All depicted analysis is simulated; no camera hardware or fiber testbed is claimed.',fontsize=8.5,color='#555')
save(fig,'fig01_system_flow.png')

# Figure 2 exposure integration effects
rng=np.random.default_rng(101)
frame, payload, meta=sim_frame(rng,'difficult')
w,ph=meta['w'],meta['phi']
full=np.r_[PILOT,payload]
times=np.arange(N_R*12)/12
idxt=np.floor((times-ph)/S).astype(int)
intensity=np.zeros_like(times); mask=(idxt>=0)&(idxt<len(full)); intensity[mask]=full[idxt[mask]]
conv=overlap(w,ph)@full
sel=(r<150)
fig,axs=plt.subplots(3,1,figsize=(10.3,6.7),sharex=True,gridspec_kw={'height_ratios':[1,1.35,1.3]})
axs[0].step(times[times<150],intensity[times<150],where='post',lw=1.15,color='#b17a19')
axs[0].set_ylabel('LED OOK\nlevel');axs[0].set_title('Finite exposure mixes adjacent symbols across successive CMOS rows')
axs[1].plot(r[sel],conv[sel],lw=2,color='#307d74',label='Integrated optical waveform')
axs[1].plot(r[sel],frame[sel],lw=.8,alpha=.62,color='#778d9b',label='Simulated sensor output')
axs[1].legend(loc='upper right',fontsize=8,frameon=False,ncol=2)
axs[1].set_ylabel('Normalized\nsensor response')
axs[2].imshow(np.tile(frame[sel],(24,1)).T,aspect='auto',origin='lower',extent=[0,24,0,150],cmap='gray')
axs[2].set_ylabel('Image rows');axs[2].set_xlabel('Illustrative image-column coordinate (constant along stripe)')
for ax in axs[:2]:ax.set_xlim(0,150);ax.grid(alpha=.12)
fig.tight_layout();save(fig,'fig02_exposure_stripes.png')

# Figure3 timing score map
yp=frame[P_R];yc=yp-D@(Dpinv@yp)
corr=Tcenter@yc;score=np.maximum(corr,0)**2/TDEN; score=score.reshape(len(np.unique(GRID[:,0])),len(np.unique(GRID[:,1])))
idx,*_=estimate(frame)
fig,ax=plt.subplots(figsize=(7.8,4.7))
I=ax.imshow(score,origin='lower',aspect='auto',extent=[0,3.84,1.35,7.95],cmap='viridis')
ax.plot([ph],[w],marker='x',color='#ea6060',markersize=11,mew=2,label='Generating parameter')
ax.plot([GRID[idx,1]],[GRID[idx,0]],marker='o',markersize=9,markerfacecolor='none',markeredgecolor='white',mew=1.7,label='Preamble estimate')
ax.set_xlabel('Unknown row/symbol phase  φ (row units)');ax.set_ylabel('Exposure width  w (row units)')
ax.set_title('Self-calibration: pilot-template score over candidate shutter parameters')
ax.legend(loc='lower right',fontsize=9,facecolor='white',framealpha=.95)
fig.colorbar(I,ax=ax,label='Detrended pilot-template correlation score',shrink=.9)
fig.tight_layout();save(fig,'fig03_timing_score.png')

# Figure4 grouped bars across regimes
summ=pd.read_csv(R/'method_summary.csv')
regs=['Nominal','Difficult','Shifted exposure + clipping'];fig,ax=plt.subplots(figsize=(10.6,5.4))
x=np.arange(len(regs));width=.15
for j,m in enumerate(methods):
    v=[summ[(summ.regime==re)&(summ.method==m)].BER.iloc[0] for re in regs]
    ax.bar(x+(j-2)*width,v,width*0.96,color=colors[m],label=m)
ax.set_yscale('log');ax.set_ylim(1e-4,.7);ax.set_xticks(x);ax.set_xticklabels(['Nominal','High exposure / noise','Shifted exposure + clipping'])
ax.set_ylabel('Test bit-error rate (logarithmic scale)');ax.set_title('Held-out test BER; each method evaluated on the same 14,400 bits per regime')
ax.grid(axis='y',alpha=.22);ax.legend(loc='upper left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=8.3)
fig.tight_layout();save(fig,'fig04_ber_comparison.png')

# Figure 5 exposure sensitivity
sen=pd.read_csv(R/'sensitivity_exposure.csv');fig,ax=plt.subplots(figsize=(9,4.8));groups=['0–0.75','0.75–1.00','1.00–1.30','1.30–1.70']
for meth in methods:
    d=sen[sen.method==meth].set_index('exposure_group').reindex(groups)
    ax.plot(range(len(groups)),d.BER.values,marker='o',lw=1.8,ms=5,label=meth,color=colors[meth])
ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(0,.7);ax.set_xticks(range(len(groups)));ax.set_xticklabels(groups)
ax.set_xlabel('Exposure-to-symbol width  w / s  (dimensionless)');ax.set_ylabel('Test BER')
ax.set_title('Sensitivity to exposure-induced intersymbol interference')
ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,fontsize=8,frameon=False)
fig.tight_layout();save(fig,'fig05_exposure_sensitivity.png')

# Figure 6 timing scatter
pf=pd.read_csv(R/'timing_estimates.csv'); fig,ax=plt.subplots(figsize=(7.7,5))
for reg,co in zip(regs,['#3c9783','#457ca0','#bb7757']):
    u=pf[pf.regime==reg]
    ax.scatter(u.w_true,u.w_est,s=15,alpha=.55,edgecolor='none',label=reg,color=co)
ax.plot([1,9],[1,9],color='black',lw=1,ls='--')
ax.set_xlim(1,9);ax.set_ylim(1,9)
ax.set_xlabel('Generating exposure width  w (rows)');ax.set_ylabel('Estimated exposure width  ŵ (rows)')
ax.set_title('Preamble-based exposure estimation across 450 unseen frames')
ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.17)
fig.tight_layout();save(fig,'fig06_exposure_calibration.png')

# Figure 7 frame-wise distribution
raw=pd.read_csv(R/'per_frame_method_metrics.csv'); fig,axs=plt.subplots(1,3,figsize=(10.9,4.3),sharey=True)
for i,reg in enumerate(regs):
    d=raw[raw.regime==reg]
    dat=[d[d.method==m].sort_values('frame').errors.values/N_D for m in methods]
    b=axs[i].boxplot(dat,positions=np.arange(5),patch_artist=True,showfliers=False,widths=.62,
                      medianprops={'color':'#222'},whiskerprops={'color':'#666'},capprops={'color':'#666'})
    for p,m in zip(b['boxes'],methods):p.set_facecolor(colors[m]);p.set_alpha(.7)
    axs[i].set_title(['Nominal','Difficult','Shifted + clipping'][i],fontsize=10.3)
    axs[i].set_xticks(range(5));axs[i].set_xticklabels(['Thr','Fix','ML','Phys','Hybrid'],fontsize=9)
    axs[i].grid(axis='y',alpha=.15)
axs[0].set_ylabel('Frame-level bit-error fraction');fig.suptitle('Dispersion across independent test frames, not just aggregate BER',fontsize=11)
fig.tight_layout();save(fig,'fig07_frame_robustness.png')

# Figure8 ablation one method no timing
ab=pd.read_csv(R/'ablation_summary.csv');fig,ax=plt.subplots(figsize=(7.7,4.7));wbar=.25
for j,(name,lookup) in enumerate([('Calibrated inverse',lambda reg:summ[(summ.regime==reg)&(summ.method=='Calibrated physics inverse')].BER.iloc[0]),
                                  ('Hybrid, no metadata',lambda reg:ab[ab.regime==reg].BER.iloc[0]),
                                  ('Full hybrid',lambda reg:summ[(summ.regime==reg)&(summ.method=='Proposed physics + residual ML')].BER.iloc[0])]):
    ax.bar(np.arange(3)+(j-1)*wbar,[lookup(reg) for reg in regs],wbar,color=['#3c9783','#9990b4','#6b56a7'][j],label=name)
ax.set_yscale('log');ax.set_ylim(1e-4,.5);ax.set_xticks(np.arange(3));ax.set_xticklabels(['Nominal','Difficult','Shifted'])
ax.set_ylabel('Test bit-error rate');ax.set_title('Ablation: most improvement comes from physics state and local residual learning')
ax.grid(axis='y',alpha=.17);ax.legend(frameon=False,fontsize=9)
fig.tight_layout();save(fig,'fig08_ablation.png')
print('figures:',*[x.name for x in sorted(P.glob('*.png'))],sep='\n')
