"""Predict oil/water trajectories from one complete 6-stage x 4-injector schedule."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fc_team_proxy import Proxy,DAYS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--theta',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('research-output/prediction.npz'))
p.add_argument('--device',default='cpu')
a=p.parse_args();torch.set_num_threads(1)
theta=np.asarray(json.loads(a.theta.read_text()),dtype=np.float32)
if theta.shape not in [(6,4),(24,)]: raise ValueError('Expected (6,4) stage-major or (24,) log10 control values')
if not np.isfinite(theta).all() or np.abs(theta).max()>1: raise ValueError('Controls must be finite and within [-1,1]')
proxy=Proxy(a.device); y=proxy.trajectories(theta.ravel())
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,theta=theta.reshape(6,4),prediction=y,grid_days=DAYS,live_mask=proxy.live)
print(json.dumps({'output':str(a.output),'shape':list(y.shape),'axes':['batch','active producer','oil/water','time node']}))
