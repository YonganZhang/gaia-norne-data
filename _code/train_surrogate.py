"""Retrain the deployed PosNet recipe without starting a simulator or language model."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from fc_rank_fidelity import load_dir
from fc_mech import PosNet
from release_paths import data_path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=600)
p.add_argument('--output',type=Path,default=Path('research-output/retrained'))
a=p.parse_args()
if (a.output/'surrogate.pt').exists(): raise FileExistsError('Choose a new output directory to preserve existing weights')
TH,Y=load_dir(data_path('norne_fc'),8000)
if len(TH)!=8000: raise ValueError('Training requires all 8000 main samples')
live=~(np.abs(Y).max(axis=(0,3))<1e-9).all(1);Y=Y[:,live];nw=int(live.sum())
idx=np.random.default_rng(0).permutation(8000);tr=idx[600:];Yf=Y.reshape(8000,-1)
xm,xs=TH[tr].mean(0),TH[tr].std(0)+1e-8;ym,ys=Yf[tr].mean(0),Yf[tr].std(0)+1e-8
X=((TH-xm)/xs).astype(np.float32);torch.set_num_threads(8);torch.manual_seed(0)
net=PosNet(24,nw*2,40,'fourier',n_bands=16).to(a.device)
opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
A=torch.tensor(X[tr],device=a.device)
B=torch.tensor(((Yf[tr]-ym)/ys).reshape(-1,nw*2,40).astype(np.float32),device=a.device)
for ep in range(a.epochs):
    pm=torch.randperm(len(A),device=a.device)
    for i in range(0,len(A),64):
        b=pm[i:i+64];opt.zero_grad();F.mse_loss(net(A[b]),B[b]).backward();opt.step()
    sch.step()
    if (ep+1)%25==0: print(f'epoch {ep+1}/{a.epochs}',flush=True)
a.output.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),a.output/'surrogate.pt')
np.savez_compressed(a.output/'norm.npz',live=live,xm=xm,xs=xs,ym=ym,ys=ys)
(a.output/'training.json').write_text(json.dumps({'seed':0,'epochs':a.epochs,'train':7400,'reserved_validation':200,'test':400,'device':a.device,'note':'Numerical equivalence is not guaranteed across hardware/PyTorch versions.'},indent=2))
print(a.output)
