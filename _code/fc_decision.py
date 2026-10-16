from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import forecast_gen as FG
import norne_bulk as NB
OUT = Path(__file__).resolve().parent.parent / '_pipelines' / 'fc_decision'
(N_STAGE, INJ_W) = (FG.N_STAGE, FG.INJ_W)
N_T = FG.FC_N

def load(max_n: int=0):
    sh = sorted((FG.OUT / 'shards').glob('*.npz'))
    if max_n:
        sh = sh[:max_n]
    (TH, Y, IA, FC) = ([], [], [], [])
    for p in sh:
        d = np.load(p)
        ob = d['obs']
        TH.append(d['theta'].ravel())
        IA.append(d['inj_actual'])
        FC.append(d['field_cum'])
        Y.append(np.stack([[ob[(i * 3 + k) * N_T:(i * 3 + k + 1) * N_T] for k in (0, 1)] for i in range(len(NB.PRODUCERS))]))
    return (np.stack(TH).astype(np.float32), np.stack(Y).astype(np.float32), np.stack(IA).astype(np.float32), np.stack(FC).astype(np.float32))

def build_net(d_in, d_out, hidden, depth):
    import torch.nn as nn
    (layers, d) = ([], d_in)
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))

def train(args) -> int:
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    (TH, Y, IA, FC) = load(args.max_n)
    n = len(TH)
    print(f'样本 {n}   θ {TH.shape[1]} 维 ({N_STAGE} 段 × {len(INJ_W)} 井)   输出 {Y.shape[1:]}')
    if n < 100:
        print('样本太少, 等数据')
        return 1
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_te = max(50, n // 10)
    (te, tr) = (idx[:n_te], idx[n_te:])
    Yf = Y.reshape(n, -1)
    (xm, xs) = (TH[tr].mean(0), TH[tr].std(0) + 1e-08)
    (ym, ys) = (Yf[tr].mean(0), Yf[tr].std(0) + 1e-08)
    (Xn, Yn) = ((TH - xm) / xs, (Yf - ym) / ys)
    torch.manual_seed(args.seed)
    net = build_net(TH.shape[1], Yf.shape[1], args.hidden, args.depth).to(dev)
    print(f'网络 {args.depth}×{args.hidden}   参数 {sum((p.numel() for p in net.parameters())):,}')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0001)
    Xtr = torch.tensor(Xn[tr], device=dev)
    Ytr = torch.tensor(Yn[tr], device=dev)
    Xte = torch.tensor(Xn[te], device=dev)
    Yte = torch.tensor(Yn[te], device=dev)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    (best, bad, t0) = (1e+30, 0, time.time())
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xtr[b]), Ytr[b]).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(net(Xte), Yte).item()
        if vl < best - 1e-05:
            (best, bad) = (vl, 0)
            torch.save({'state': net.state_dict(), 'xm': xm, 'xs': xs, 'ym': ym, 'ys': ys, 'shape': Y.shape[1:], 'hidden': args.hidden, 'depth': args.depth, 'd_in': TH.shape[1]}, OUT / 'net.pt')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'早停 ep{ep}')
                break
        if ep % 50 == 0:
            print(f'  ep {ep:4d} val {vl:.5f} best {best:.5f}', flush=True)
    ck = torch.load(OUT / 'net.pt', weights_only=False)
    net.load_state_dict(ck['state'])
    net.eval()
    with torch.no_grad():
        P = (net(Xte).cpu().numpy() * ys + ym).reshape(-1, *Y.shape[1:])
    days = FG.FC_GRID
    co_p = np.trapezoid(P[:, :, 0, :], days, axis=2).sum(1)
    co_t = np.trapezoid(Y[te][:, :, 0, :], days, axis=2).sum(1)
    err = np.abs(co_p - co_t) / np.maximum(np.abs(co_t), 1e-09)
    base = np.abs(np.trapezoid(Y[tr][:, :, 0, :], days, axis=2).sum(1).mean() - co_t) / np.abs(co_t)
    r2 = 1 - ((co_p - co_t) ** 2).sum() / ((co_t - co_t.mean()) ** 2).sum()
    print(f'\n留出集 · 预测段(13.1 年)累计产油:')
    print(f'  相对误差 中位 {np.median(err):.2%}  p90 {np.percentile(err, 90):.2%}   R² {r2:.4f}')
    print(f"  平凡基线(训练集均值) {np.median(base):.2%}   {('✅ 打赢' if np.median(err) < np.median(base) else '🔴 没打赢')}")
    print(f'  训练耗时 {(time.time() - t0) / 60:.1f} min')
    json.dump({'n': n, 'n_test': len(te), 'relerr_median': float(np.median(err)), 'relerr_p90': float(np.percentile(err, 90)), 'r2': float(r2), 'baseline_relerr': float(np.median(base)), 'beats_baseline': bool(np.median(err) < np.median(base)), 'test_idx': te.tolist()}, open(OUT / 'train.json', 'w'), indent=1)
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'optimize'])
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--max-n', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.cmd == 'train':
        return train(args)
    print('optimize 待数据齐后实现')
    return 1
if __name__ == '__main__':
    raise SystemExit(main())
