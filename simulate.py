"""Reproducible SYNTHETIC rolling-shutter optical-camera link experiment.
No hardware capture; all figures are outputs of this model. seed=20261103.
Requires numpy scipy scikit-learn matplotlib pandas.
"""
import os, csv, json, time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
S=4.0; N_P=24; N_D=96; N=N_P+N_D; N_R=int(N*S+9)
SEED=20261103
r=np.arange(N_R,dtype=float)
rnorm=(r-r.mean())/N_R
PILOT=np.random.default_rng(79).integers(0,2,size=N_P).astype(float)
assert np.unique(PILOT).size==2


def overlap(w,phi,start=0,stop=N):
    k=np.arange(start,stop,dtype=float)
    left=phi+k*S
    right=left+S
    overlap=np.maximum(0., np.minimum(r[:,None]+w,right[None,:])-np.maximum(r[:,None],left[None,:])) / w
    return overlap

# candidate timing parameters
WS=np.linspace(1.35,7.95,23)
PHIS=np.linspace(0,3.84,17)
GRID=np.array([(w,ph) for w in WS for ph in PHIS])
P_R=np.arange(int(N_P*S-9))
rr=(P_R-P_R.mean()) / len(P_R)
D=np.c_[np.ones(len(P_R)),rr]
Dpinv=np.linalg.pinv(D)
T=np.array([overlap(w,ph,0,N_P)[P_R]@PILOT for w,ph in GRID])
Tcenter=T-(T@Dpinv.T)@D.T
TDEN=np.sum(Tcenter**2,axis=1)+1e-12

def estimate(y,forced=None):
    yp=y[P_R]
    yc=yp-D@(Dpinv@yp)
    if forced is None:
        corr=Tcenter@yc
        idx=int(np.argmax((np.maximum(corr,0)**2)/TDEN))
    else:
        idx=int(np.argmin((GRID[:,0]-forced[0])**2+(GRID[:,1]-forced[1])**2))
    A=np.c_[np.ones(len(P_R)),T[idx],rnorm[P_R]]
    a0,g,sl=np.linalg.lstsq(A,yp,rcond=None)[0]
    g=max(0.10,float(g))
    err=np.sqrt(np.mean((A@np.array([a0,g,sl])-yp)**2))
    return idx,float(a0),g,float(sl),float(err)

def solve(y,idx,a,g,sl,reg=.11):
    w,ph=GRID[idx]
    mat=overlap(w,ph)
    known=(mat[:,:N_P]@PILOT)
    M=mat[:,N_P:]
    target=(y-a-sl*rnorm)/g-known
    # Normal equations are 96x96 positive definite after ridge.
    gram=M.T@M+reg*np.eye(N_D)
    rhs=M.T@target+reg*.5*np.ones(N_D)
    try:
        c=cho_factor(gram,check_finite=False)
        est=cho_solve(c,rhs,check_finite=False)
    except Exception:
        est=np.linalg.solve(gram,rhs)
    return np.clip(est,-0.5,1.5), M, target

def raw_feats(y, a, g, sl):
    center=(np.arange(N_P,N)*S+S/2+1.2).astype(int)
    # Features are from nominal (uncalibrated) row positions.
    yn=(y-a-sl*rnorm)/g
    A=[]
    for delta in [-6,-4,-2,0,2,4,6]:
        ix=np.clip(center+delta,0,N_R-1)
        A.append(yn[ix])
    return np.array(A).T

def sim_frame(rng,regime):
    if regime=='nominal':
        w=rng.uniform(1.5,4.7); sigma=rng.uniform(.045,.105)
        slope=rng.uniform(-.085,.085); drift=rng.uniform(.01,.055)
    elif regime=='difficult':
        w=rng.uniform(3.8,6.35); sigma=rng.uniform(.085,.155)
        slope=rng.uniform(-.15,.15); drift=rng.uniform(.03,.095)
    elif regime=='shifted':
        w=rng.uniform(6.35,8.3); sigma=rng.uniform(.105,.205)
        slope=rng.uniform(-.26,.26); drift=rng.uniform(.06,.155)
    else: raise ValueError(regime)
    ph=rng.uniform(0,S)
    payload=rng.integers(0,2,size=N_D).astype(float)
    bits=np.r_[PILOT,payload]
    src=overlap(w,ph)@bits
    a=rng.uniform(.2,.45);gain=rng.uniform(.65,1.22)
    # second, unmodeled low-frequency illumination source creates model mismatch
    cyc=drift*np.sin(2*np.pi*r/N_R*rng.uniform(.7,2.1)+rng.uniform(0,2*np.pi))
    noi=sigma*np.sqrt(.65+.35*src)*rng.normal(size=N_R)
    y=a+gain*src+slope*rnorm+cyc+noi
    if regime=='shifted':
        # camera clipping, not included in the inverse model
        y=np.minimum(y, rng.uniform(.91,1.35))
    return y,payload,{'w':w,'phi':ph,'sigma':sigma,'slope':slope,'gain':gain}

def derive(y,truth,meta):
    t1=time.perf_counter(); idx,a,g,sl,err=estimate(y); cal_ms=(time.perf_counter()-t1)*1000
    t2=time.perf_counter(); p,_,_=solve(y,idx,a,g,sl); phy_ms=(time.perf_counter()-t2)*1000
    idx_nom,a2,g2,sl2,err2=estimate(y,forced=(3.0,1.0))
    n,_,_=solve(y,idx_nom,a2,g2,sl2)
    R=raw_feats(y,a,g,sl)
    Rnom=raw_feats(y,a2,g2,sl2)
    # simple raw-threshold with fixed bit timing, same frame-specific pilot normalization
    raw=(Rnom[:,3]>.5).astype(float)
    # physical contextual features only depend on decoded values, not the ground truth
    context=np.stack([np.roll(p,d) for d in [-2,-1,0,1,2]],axis=1)
    
    for j,shift in enumerate([-2,-1,0,1,2]):
        if shift>0: context[:shift,j]=.5
        if shift<0: context[shift:,j]=.5
    physics_features=np.c_[context, R, GRID[idx,0]*np.ones(N_D),GRID[idx,1]*np.ones(N_D),err*np.ones(N_D)]
    # Learned baseline features: noisy row samples w/ nominal photometric normalization.
    raw_ml_features=Rnom
    return dict(true=truth,p=p,n=n,raw=raw,physical=physics_features,learned=raw_ml_features,
                w=meta['w'],we=GRID[idx,0],phi=meta['phi'],pe=GRID[idx,1],
                sigma=meta['sigma'],cal_ms=cal_ms,phy_ms=phy_ms,fit_rmse=err)

def make_set(name,n,seed,regime):
    rng=np.random.default_rng(seed)
    out=[]
    for i in range(n):
        yy,xx,meta=sim_frame(rng,regime)
        out.append(derive(yy,xx,meta))
    print(name,regime,n,'frames',flush=True)
    return out

def wilson(k,n,alpha=.05):
    z=norm.ppf(1-alpha/2);ph=k/n
    d=1+z*z/n
    cen=(ph+z*z/(2*n))/d
    rad=z/d*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))
    return float(max(0,cen-rad)),float(min(1,cen+rad))

def main():
    # fully disjoint frame seeds and payloads: training, validation and testing
    train=make_set('train_a',210,10001,'nominal')+make_set('train_b',170,10002,'difficult')
    valid=make_set('validation',80,20001,'difficult')
    test={
       'Nominal':make_set('test1',150,30001,'nominal'),
       'Difficult':make_set('test2',150,30002,'difficult'),
       'Shifted exposure + clipping':make_set('test3',150,30003,'shifted')}
    Xt=np.concatenate([i['physical'] for i in train]); Xtr=np.concatenate([i['learned'] for i in train]);
    ytr=np.concatenate([i['true'] for i in train])
    model=HistGradientBoostingClassifier(max_iter=115,max_leaf_nodes=13,max_depth=4,min_samples_leaf=60,learning_rate=.067,l2_regularization=3.0,random_state=79)
    learned=HistGradientBoostingClassifier(max_iter=115,max_leaf_nodes=13,max_depth=4,min_samples_leaf=60,learning_rate=.067,l2_regularization=3.0,random_state=81)
    model.fit(Xt,ytr);learned.fit(Xtr,ytr)
    # report validation, no hyperparam tuning against held-out test
    Vp=np.vstack([i['physical'] for i in valid]); Vy=np.concatenate([i['true'] for i in valid]);
    print('validation physics',np.mean((np.concatenate([i['p'] for i in valid])>.5) !=Vy),
          'model',np.mean(model.predict(Vp)!=Vy))
    methods=['Raw midpoint threshold','Nominal model (fixed timing)','Data-only gradient boosting','Calibrated physics inverse','Proposed physics + residual ML']
    rows=[]; per=[]; cal=[]
    for regime,frames in test.items():
        for fid,item in enumerate(frames):
            pred=[item['raw'],(item['n']>.5), learned.predict(item['learned']),
                  (item['p']>.5), model.predict(item['physical'])]
            rec={'regime':regime,'frame':fid,'w_true':item['w'],'w_est':item['we'],
                 'phi_true':item['phi'],'phi_est':item['pe'],'noise_sigma':item['sigma'],
                 'pilot_rmse':item['fit_rmse'],'cal_ms':item['cal_ms'],'phy_ms':item['phy_ms']}
            for met,val in zip(methods,pred):
                rec[met+'_errors']=int(np.sum(val!=item['true']))
            per.append(rec)
            for meth,val in zip(methods,pred):
                err=int(np.sum(val!=item['true']))
                rows.append({'regime':regime,'frame':fid,'method':meth,'errors':err,'bits':N_D,'exposure_ratio':item['w']/S,
                             'noise_sigma':item['sigma']})
    df=pd.DataFrame(rows);pf=pd.DataFrame(per)
    df.to_csv(RESULTS/'per_frame_method_metrics.csv',index=False)
    pf.to_csv(RESULTS/'timing_estimates.csv',index=False)
    tab=[]
    for (reg,meth),g in df.groupby(['regime','method'],sort=False):
        k=int(g.errors.sum()); n=int(g.bits.sum());lo,hi=wilson(k,n)
        tab.append({'regime':reg,'method':meth,'errors':k,'bits':n,'BER':k/n,'95CI_low_wilson':lo,'95CI_high_wilson':hi})
    summ=pd.DataFrame(tab);summ.to_csv(RESULTS/'method_summary.csv',index=False)
    # sensitivity: exposures by quantiles across nominal+difficult, report exact counts
    sb=df[df.regime!='Shifted exposure + clipping'].copy()
    sb['exposure_group']=pd.cut(sb.exposure_ratio,bins=[0,.75,1.0,1.3,1.7],labels=['0–0.75','0.75–1.00','1.00–1.30','1.30–1.70'],include_lowest=True)
    sens=sb.groupby(['exposure_group','method'],observed=True).agg(errors=('errors','sum'),bits=('bits','sum')).reset_index()
    sens['BER']=sens.errors/sens.bits
    sens.to_csv(RESULTS/'sensitivity_exposure.csv',index=False)
    # Ablation: physical posterior with threshold; model without timing features
    model_no_t=HistGradientBoostingClassifier(max_iter=115,max_leaf_nodes=13,max_depth=4,min_samples_leaf=60,learning_rate=.067,l2_regularization=3,random_state=79)
    model_no_t.fit(Xt[:,:12],ytr)
    ab=[]
    for regime,frames in test.items():
        pred=model_no_t.predict(np.vstack([i['physical'][:,:12] for i in frames]));yy=np.concatenate([i['true'] for i in frames]);
        ab.append({'regime':regime,'method':'Ablation: no timing metadata','errors':int(np.sum(pred!=yy)),'bits':len(yy),'BER':float(np.mean(pred!=yy))})
    abdf=pd.DataFrame(ab);abdf.to_csv(RESULTS/'ablation_summary.csv',index=False)
    out={'seed':SEED,'N_pilot':N_P,'N_payload':N_D,'symbol_width_rows':S,
         'n_train_frames':len(train),'n_validation_frames':len(valid),'n_test_frames_per_condition':150,
         'training_bits':len(ytr),'test_bits_each_condition':150*N_D,
         'reference_exposure_candidates':len(GRID),'methods':methods,
         'results':summ.to_dict(orient='records'),'ablation':ab,'calibration_median_ms':float(pf.cal_ms.median()),
         'inverse_median_ms':float(pf.phy_ms.median()),
         'exposure_median_abs_err_rows_by_regime':{k:float(np.median(abs(v.w_true-v.w_est))) for k,v in pf.groupby('regime')}}
    (RESULTS/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    import joblib
    joblib.dump({'residual':model,'data_only':learned,'no_timing':model_no_t},RESULTS/'models.joblib')
    print('\nSUMMARY\n',summ.to_string(index=False),flush=True)
    print('\nABLATION\n',abdf.to_string(index=False),flush=True)
    print('\nTIMING MAE\n',out['exposure_median_abs_err_rows_by_regime'],flush=True)

if __name__=='__main__':main()
