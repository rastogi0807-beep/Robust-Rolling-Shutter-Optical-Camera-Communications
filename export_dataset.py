"""Regenerate and save the raw synthetic pilot+payload image-row datasets as NPZ.
Each NPZ split records row intensity (one-dimensional), payload bits,
true exposure width and phase, noise scale, and numerical seed provenance.
"""
from pathlib import Path
import numpy as np
from simulate import sim_frame,PILOT,S,N_P,N_D,ROOT
out={}
for name,seed,n,reg in [('training_nominal',10001,210,'nominal'),('training_difficult',10002,170,'difficult'),
                        ('validation_difficult',20001,80,'difficult'),('test_nominal',30001,150,'nominal'),
                        ('test_difficult',30002,150,'difficult'),('test_shifted',30003,150,'shifted')]:
    rng=np.random.default_rng(seed)
    frames=[];truth=[];wid=[];phase=[];sig=[]
    for _ in range(n):
        y,b,m=sim_frame(rng,reg)
        frames.append(y.astype('float32'));truth.append(b.astype('uint8'))
        wid.append(m['w']);phase.append(m['phi']);sig.append(m['sigma'])
    out[name+'_rows']=np.array(frames);out[name+'_payload_bits']=np.array(truth)
    out[name+'_true_width_rows']=np.array(wid,dtype='float32')
    out[name+'_true_phase_rows']=np.array(phase,dtype='float32')
    out[name+'_noise_sigma']=np.array(sig,dtype='float32')
    print(name,len(frames))
out['pilot_bits']=PILOT.astype('uint8')
out['rows_per_symbol']=np.array([S],dtype='float32')
out['seed_table']=np.array([10001,10002,20001,30001,30002,30003])
p=ROOT/'results'/'synthetic_frames.npz'
np.savez_compressed(p,**out)
print(p,p.stat().st_size)
