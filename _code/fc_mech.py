from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fc_decision as FD
N_PIN = 4000
import forecast_gen as FG
(NPH, NT) = (2, FG.FC_N)
DAYS = FG.FC_GRID
OUT = Path(__file__).resolve().parent.parent / '_pipelines' / 'fc_mech'

class PosNet(nn.Module):

    def __init__(self, d_in, n_ch, n_t, pos_kind, d=128, heads=4, layers=3, ff=256, n_bands=16, seed=0):
        super().__init__()
        self.cond = nn.Linear(d_in, d)
        self.kind = pos_kind
        g = torch.Generator().manual_seed(1234 + seed)
        t = torch.linspace(0, 1, n_t)
        if pos_kind == 'learnable':
            self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.02)
        elif pos_kind == 'learnable_matched':
            self.pos = nn.Parameter(torch.randn(1, n_t, d) * 0.42)
        elif pos_kind in ('fourier', 'fourier_nyquist'):
            f = 2.0 ** torch.arange(n_bands) if pos_kind == 'fourier' else torch.linspace(1, n_t // 2, n_bands)
            a = t[:, None] * f[None, :] * 2 * np.pi
            self.register_buffer('feat', torch.cat([torch.sin(a), torch.cos(a)], -1))
            self.proj = nn.Linear(2 * n_bands, d)
        elif pos_kind == 'randproj':
            self.register_buffer('feat', torch.randn(n_t, 2 * n_bands, generator=g))
            self.proj = nn.Linear(2 * n_bands, d)
        elif pos_kind == 'randfrozen':
            self.register_buffer('fpos', torch.randn(1, n_t, d, generator=g) * 0.02)
        else:
            raise ValueError(pos_kind)
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, dropout=0.0), layers)
        self.head = nn.Linear(d, n_ch)

    def _pos(self):
        if self.kind in ('learnable', 'learnable_matched'):
            return self.pos
        if self.kind == 'randfrozen':
            return self.fpos
        return self.proj(self.feat)[None]

    def forward(self, x):
        return self.head(self.tr(self.cond(x)[:, None, :] + self._pos())).transpose(1, 2)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    (TH, Y, IA, FC) = FD.load(N_PIN)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-09).all(1)
    Y = Y[:, live]
    nw = int(live.sum())
    n = len(TH)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    (va, tr) = (idx[400:600], idx[600:])
    Yf = Y.reshape(n, -1)
    NCH = nw * NPH
    (ym, ys) = (Yf[tr].mean(0), Yf[tr].std(0) + 1e-08)
    (xm, xs) = (TH[tr].mean(0), TH[tr].std(0) + 1e-08)
    X = ((TH - xm) / xs).astype(np.float32)
    Xtr = torch.tensor(X[tr], device=dev)
    Xva = torch.tensor(X[va], device=dev)
    Ytr = torch.tensor(((Yf[tr] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
    cum = lambda A_: np.trapezoid(A_.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    cva = cum(Yf[va])
    print(f'train {len(tr)} / val {len(va)}   test 400 **全程不碰**')
    print(f'{args.seeds} 种子 × {args.epochs} epoch\n')

    def trial(kind, seed):
        torch.manual_seed(seed)
        net = PosNet(TH.shape[1], NCH, NT, kind, seed=seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            pm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), 64):
                b = pm[i:i + 64]
                opt.zero_grad()
                F.mse_loss(net(Xtr[b]), Ytr[b]).backward()
                opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            P = net(Xva).cpu().numpy().reshape(len(va), -1) * ys + ym
        return float(np.mean(np.abs(cum(P) - cva) / np.abs(cva)))
    KINDS = [('A 可学习 pos(基线 std=0.02)', 'learnable'), ('A2 可学习 pos(幅度对齐 std=0.42)', 'learnable_matched'), ('B 傅里叶 f=2^k(现方案)', 'fourier'), ('C 冻结随机+可训练投影', 'randproj'), ('D 冻结随机 pos(无投影)', 'randfrozen'), ('E 傅里叶但不超 Nyquist', 'fourier_nyquist')]
    (R, t0) = ({}, time.time())
    for (tag, k) in KINDS:
        e = np.array([trial(k, s) for s in range(args.seeds)]) * 100
        R[tag] = {'mean': float(e.mean()), 'sd': float(e.std(ddof=1)), 'raw': e.tolist()}
        print(f'  {tag:26s} {e.mean():.3f}% ± {e.std(ddof=1):.3f}   ({(time.time() - t0) / 60:.0f}min)', flush=True)
    from scipy.stats import ttest_rel
    A = np.array(R['A 可学习 pos(基线 std=0.02)']['raw'])
    B = np.array(R['B 傅里叶 f=2^k(现方案)']['raw'])
    print(f'\n=== 判定(配对 t 检验, 同种子) ===')
    for tag in R:
        if tag.startswith('A 可'):
            continue
        v = np.array(R[tag]['raw'])
        pA = float(ttest_rel(v, A).pvalue)
        pB = float(ttest_rel(v, B).pvalue)
        print(f'  {tag:26s} vs A {v.mean() - A.mean():+7.3f}pp p={pA:.4f}   vs B {v.mean() - B.mean():+7.3f}pp p={pB:.4f}')
    A2 = np.array(R['A2 可学习 pos(幅度对齐 std=0.42)']['raw'])
    B_ = np.array(R['B 傅里叶 f=2^k(现方案)']['raw'])
    dA2 = A2.mean() - B_.mean()
    pA2 = float(ttest_rel(A2, B_).pvalue)
    print(f'\n🔴 幅度混杂判定(Codex 提出):')
    print(f'   A2(仅把基线 std 从 0.02 调到 0.42) − B(傅里叶) = {dA2:+.3f}pp  p={pA2:.4f}')
    if abs(dA2) < 0.02 and pA2 > 0.05:
        print(f"   → **傅里叶的收益全部可由初始化幅度解释**。整个'谱偏差'叙事作废,")
        print(f'      真实结论是:基线的 pos 初始化 *0.02 过小,是个 baseline bug。')
    elif dA2 > 0.02:
        print(f'   → 幅度不足以解释;傅里叶(或固定基)确有额外贡献。')
    C = np.array(R['C 冻结随机+可训练投影']['raw'])
    E = np.array(R['E 傅里叶但不超 Nyquist']['raw'])
    dCB = C.mean() - B.mean()
    pCB = float(ttest_rel(C, B).pvalue)
    print(f'\n🔴 机理判定:')
    if abs(dCB) < 0.02 and pCB > 0.05:
        verdict = "**谱解释被推翻** —— 冻结随机基与傅里叶无差异, 收益来自'固定基优于可学习 pos'(正则化/优化效应), 与频率结构无关。"
    elif dCB > 0 and pCB < 0.05:
        verdict = '**谱解释成立** —— 傅里叶显著优于同维度冻结随机基, 收益确实来自正弦基的频率结构。'
    else:
        verdict = f'**不确定** —— C−B={dCB:+.3f}pp p={pCB:.4f}, 需要更多种子。'
    print(f'   {verdict}')
    dEB = E.mean() - B.mean()
    pEB = float(ttest_rel(E, B).pvalue)
    print(f"   超 Nyquist 频带: E(不超)−B(超) = {dEB:+.3f}pp p={pEB:.4f}  → {('应改用不超 Nyquist 的频率' if dEB < -0.02 and pEB < 0.05 else '超采样频带无害')}")
    json.dump({'results': R, 'verdict': verdict, 'C_minus_B': dCB, 'p_CB': pCB, 'E_minus_B': dEB, 'p_EB': pEB, 'seeds': args.seeds, 'note': '在 val 上做;test 400 条全程未碰。判定规则在跑之前就写死在 docstring 里。'}, open(OUT / 'mech.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n已写入 {OUT}/mech.json')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
