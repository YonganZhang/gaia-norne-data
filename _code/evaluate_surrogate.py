"""Evaluate frozen released weights on held-out and confirmation data; no retraining."""
import json
import numpy as np
import torch
from fc_team_proxy import Proxy,WT
from fc_rank_fidelity import load_dir
from release_paths import data_path

torch.set_num_threads(2);p=Proxy();result={}
for name in ['norne_fc','norne_fc_confirm']:
    th,y=load_dir(data_path(name),8000 if name=='norne_fc' else 0)
    if name=='norne_fc':
        if len(th)!=8000: raise ValueError('Expected all 8000 main samples')
        idx=np.random.default_rng(0).permutation(8000)[:400];th,y=th[idx],y[idx]
    truth=np.einsum('bt,t->b',y[:,p.live,0,:].sum(1),WT)
    pred=np.concatenate([p.evaluate(th[i:i+256])['oil'] for i in range(0,len(th),256)])
    result[name]={'n':len(th),'mean_relative_cumulative_oil_error_percent':float(np.mean(np.abs(pred-truth)/truth)*100)}
print(json.dumps(result,indent=2))
