from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import forecast_gen as FG
import norne_bulk as NB
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_decide'
(NPH, NT) = (2, FG.FC_N)
DAYS = FG.FC_GRID
(N_STAGE, INJ) = (FG.N_STAGE, FG.INJ_W)
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ])
SHORT_YEARS = 3.0

def horizon_mask(years):
    return DAYS - DAYS[0] <= years * 365.25

def plan(args) -> int:
    import torch
    import fc_decision as FD
    from fc_mech import PosNet
    import torch.nn.functional as F
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    (TH, Y, IA, FC) = FD.load(args.n_pin)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-09).all(1)
    Y = Y[:, live]
    nw = int(live.sum())
    n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    (te, va, pool) = (idx[:400], idx[400:600], idx[600:])
    tr = pool[:args.ntrain]
    Yf = Y.reshape(n, -1)
    NCH = nw * NPH
    (ym, ys) = (Yf[tr].mean(0), Yf[tr].std(0) + 1e-08)
    (xm, xs) = (TH[tr].mean(0), TH[tr].std(0) + 1e-08)
    if len(tr) < 50:
        raise SystemExit(f'🔴 训练集只有 {len(tr)} 条(n_pin={args.n_pin} 太小,前 600 条被 test/val 占满)。请增大 --n-pin。')
    print(f'训练代理模型:{len(tr)} 样本  生产井 {nw}  θ {TH.shape[1]} 维\n')
    X = ((TH - xm) / xs).astype(np.float32)
    torch.manual_seed(0)
    net = PosNet(TH.shape[1], NCH, NT, 'fourier', n_bands=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
    A = torch.tensor(X[tr], device=dev)
    B = torch.tensor(((Yf[tr] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        pm = torch.randperm(len(A), device=dev)
        for i in range(0, len(A), 64):
            b = pm[i:i + 64]
            opt.zero_grad()
            F.mse_loss(net(A[b]), B[b]).backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[te], device=dev)).cpu().numpy().reshape(len(te), -1) * ys + ym
    cum = lambda a: np.trapezoid(a.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1).sum(1)
    (ct, cp) = (cum(Yf[te]), cum(P))
    acc = float(np.mean(np.abs(cp - ct) / np.abs(ct)))
    print(f'代理模型封存 test 误差 {acc:.3%}   ← 后面所有决策都建立在这个精度上\n')
    ymT = torch.tensor(ym, device=dev, dtype=torch.float32)
    ysT = torch.tensor(ys, device=dev, dtype=torch.float32)
    xmT = torch.tensor(xm, device=dev, dtype=torch.float32)
    xsT = torch.tensor(xs, device=dev, dtype=torch.float32)
    BW_t = torch.tensor(BASE_W, device=dev, dtype=torch.float32)
    m_short = torch.tensor(horizon_mask(SHORT_YEARS).astype(np.float32), device=dev)
    m_long = torch.ones(NT, device=dev)
    dT = torch.tensor(DAYS, device=dev, dtype=torch.float32)
    sig = float(TH.std())

    def oil_curve(theta_flat):
        z = (theta_flat - xmT) / xsT
        o = net(z[None])[0].reshape(-1) * ysT + ymT
        return o.reshape(nw, NPH, NT)[:, 0, :].sum(0)

    def integ(curve, mask):
        c = curve * mask
        return torch.trapezoid(c, dT)

    def water_total(theta_flat):
        r = 10.0 ** theta_flat.reshape(N_STAGE, len(INJ)) * torch.tensor(BASE_W, device=dev, dtype=torch.float32)[None]
        return r.sum()
    w0 = float(water_total(torch.zeros(N_STAGE * len(INJ), device=dev)))
    base_curve = oil_curve(torch.zeros(N_STAGE * len(INJ), device=dev))
    base_long = float(integ(base_curve, m_long))
    base_short = float(integ(base_curve, m_short))
    print(f'基准(θ=0):  全期 {base_long:,.0f}   前 {SHORT_YEARS:.0f} 年 {base_short:,.0f}')
    print(f'基准总注水(相对量) {w0:,.1f}\n')

    def solve(mask, ref, tag, stage_mask=None):
        z = torch.zeros(N_STAGE * len(INJ), device=dev, requires_grad=True)
        o = torch.optim.Adam([z], lr=args.lr)
        (best, best_th) = (-1e+30, None)
        for it in range(args.steps):
            o.zero_grad()
            th = args.trust_r * sig * torch.tanh(z)
            if stage_mask is not None:
                th = th * torch.tensor(stage_mask, device=dev, dtype=torch.float32)
            frac = torch.softmax(th.reshape(N_STAGE, len(INJ)) * np.log(10.0), 1)
            rate = frac * BW_t.sum()
            th_eff = torch.log10(rate / BW_t[None] + 1e-12).reshape(-1)
            obj = integ(oil_curve(th_eff), mask)
            wr = (water_total(th_eff) - w0) / w0
            pen = args.water_pen * wr ** 2 * ref
            (-(obj - pen)).backward()
            o.step()
            with torch.no_grad():
                (v, wv) = (float(obj), float(abs(wr)))
                if wv <= args.water_tol and v > best:
                    (best, best_th) = (v, th_eff.detach().cpu().numpy().copy())
        if best_th is None:
            raise SystemExit(f'🔴 {tag}: 没有任何迭代满足注水量约束(tol={args.water_tol}),拒绝输出')
        wfin = float(abs((water_total(torch.tensor(best_th, device=dev)) - w0) / w0))
        print(f'  {tag:10s} 目标 {best:,.0f}  vs 基准 {(best / ref - 1) * 100:+.2f}%   注水量偏离 {wfin:.2%}')
        return (best_th, best, wfin)
    print(f'=== 求解(信赖域 {args.trust_r}σ, {args.steps} 步) ===')
    (th_short, v_short, w_short) = solve(m_short, base_short, '短期最优')
    (th_long, v_long, w_long) = solve(m_long, base_long, '长期最优')
    sw = th_short.reshape(N_STAGE, -1).copy()
    sw[3:] = th_long.reshape(N_STAGE, -1)[3:]
    th_switch = sw.reshape(-1)
    th_unif = np.zeros(N_STAGE * len(INJ), dtype=np.float32)
    plans = {'baseline': np.zeros(N_STAGE * len(INJ)).tolist(), 'short': th_short.tolist(), 'long': th_long.tolist(), 'switch': th_switch.tolist()}
    with torch.no_grad():
        pred = {k: {'long': float(integ(oil_curve(torch.tensor(v, device=dev, dtype=torch.float32)), m_long)), 'short': float(integ(oil_curve(torch.tensor(v, device=dev, dtype=torch.float32)), m_short))} for (k, v) in plans.items()}
    print(f'\n=== 代理模型的预测(还没有裁判) ===')
    for (k, v) in pred.items():
        print(f"  {k:9s} 全期 {v['long']:,.0f} ({v['long'] / base_long - 1:+.2%})   前{SHORT_YEARS:.0f}年 {v['short']:,.0f} ({v['short'] / base_short - 1:+.2%})")
    gap = pred['long']['long'] / pred['short']['long'] - 1
    print(f'\n  🔴 代理模型预测的「只顾眼前的代价」= {gap:+.2%}')
    print(f'     ⚠️ 这只是代理模型说的。**必须用 OPM Flow 真跑一遍才算数** → fc_decide.py verify')
    json.dump({'surrogate_test_err': acc, 'n_train': len(tr), 'short_years': SHORT_YEARS, 'plans': plans, 'surrogate_pred': pred, 'base_water': w0, 'water_dev': {'short': w_short, 'long': w_long}, 'caveats': ['地质维度失效:渗透率乘子对十年产油 R²=0.0008,多地质稳健性当前参数化下无法宣称', 'F-4H 是假旋钮:BHP 500 bar 截断,目标注水率只实现 27.6%', '预测段真实工况是 WAG(水气交替),本工作只优化水,是简化']}, open(OUT / 'plans.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n已写入 {OUT}/plans.json')
    return 0

def verify(args) -> int:
    src = 'plans_calibrated.json' if args.calibrated else 'plans.json'
    pj = json.loads((OUT / src).read_text())
    print(f'用方案文件: {src}')
    work_root = FG.OUT / '_decide'
    work_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FG.HIST / 'meta.json').read_text())
    hist_rel = str((FG.HIST / 'out' / 'NORNE_ATW2013').resolve())
    import hashlib

    def _h(tag):
        b = np.asarray(pj['plans'][tag], dtype=np.float64).tobytes()
        return hashlib.sha256(b).hexdigest()[:8]
    sfx = '_cal' if args.calibrated else ''

    def one(tag):
        sh = OUT / 'sim' / f'{tag}{sfx}_{_h(tag)}.npz'
        if sh.exists():
            return (tag, 'cached')
        th = np.asarray(pj['plans'][tag], dtype=np.float32).reshape(N_STAGE, len(INJ))
        w = work_root / f'{tag}{sfx}_{_h(tag)}'
        shutil.rmtree(w, ignore_errors=True)
        try:
            FG.build_restart_deck(w, th, hist_rel, meta['last_step'])
            cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{w}:/data', '-v', f"{FG.HIST / 'out'}:{FG.HIST / 'out'}:ro", '-w', '/data', NB.IMAGE, 'flow', NB.DECK, '--output-dir=/data/out', f'--threads-per-process={args.threads}', '--parsing-strictness=low']
            with open(w / 'run.log', 'w') as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200).returncode
            if rc != 0:
                return (tag, f'flow rc={rc}')
            g = FG.harvest_fc(w)
            if g is None:
                return (tag, 'harvest 失败')
            sh.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sh, theta=th, **g)
            return (tag, 'ok')
        finally:
            shutil.rmtree(w, ignore_errors=True)
    tags = list(pj['plans'])
    print(f'用 OPM Flow 真跑 {len(tags)} 个方案(纯 CPU,{args.jobs} 并发)')
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for f in as_completed({ex.submit(one, t): t for t in tags}):
            (tag, st) = f.result()
            print(f'  {tag:9s} {st}   ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return report(args)

def report(args) -> int:
    src = 'plans_calibrated.json' if args.calibrated else 'plans.json'
    pj = json.loads((OUT / src).read_text())
    ms = horizon_mask(SHORT_YEARS)
    R = {}
    for tag in pj['plans']:
        import hashlib
        h = hashlib.sha256(np.asarray(pj['plans'][tag], dtype=np.float64).tobytes()).hexdigest()[:8]
        f = OUT / 'sim' / f"{tag}{('_cal' if args.calibrated else '')}_{h}.npz"
        if not f.exists():
            print(f'  {tag}: 还没跑')
            continue
        d = np.load(f)
        ob = d['obs']
        n_prod = len(NB.PRODUCERS)
        oil = np.stack([ob[i * 3 * NT:(i * 3 + 1) * NT] for i in range(n_prod)])
        R[tag] = {'long': float(np.trapezoid(oil, DAYS, axis=-1).sum()), 'short': float(np.trapezoid(oil[:, ms], DAYS[ms], axis=-1).sum()), 'winj': float(d['field_cum'][1][-1] - d['field_cum'][1][0])}
    if 'baseline' not in R:
        print('缺基准算例')
        return 1
    b = R['baseline']
    print(f"\n{'=' * 74}\n=== 🔴 OPM Flow 裁判结果(不是代理模型自己说的) ===\n")
    print(f"  {'方案':10s}{'全期产油':>16s}{'vs基准':>9s}{'前3年产油':>15s}{'vs基准':>9s}{'注水量偏离':>11s}")
    for tag in ('baseline', 'short', 'long', 'switch'):
        if tag not in R:
            continue
        r = R[tag]
        print(f"  {tag:10s}{r['long']:>16,.0f}{(r['long'] / b['long'] - 1) * 100:>+8.2f}%{r['short']:>15,.0f}{(r['short'] / b['short'] - 1) * 100:>+8.2f}%{(r['winj'] / b['winj'] - 1) * 100:>+10.2f}%")
    if 'short' in R and 'long' in R:
        gap = R['long']['long'] / R['short']['long'] - 1
        print(f'\n  🔴 **只顾眼前的代价 = {gap:+.2%}**')
        print(f'     (一直按短期最优采 vs 一直按长期最优采,同样多的水)')
        sp = pj['surrogate_pred']
        gp = sp['long']['long'] / sp['short']['long'] - 1
        print(f'     代理模型预测 {gp:+.2%}  →  代理误差 {abs(gap - gp) * 100:.2f} 个百分点')
    print(f'\n  ⚠️ 必须与结果一起声明的限制:')
    for c in pj['caveats']:
        print(f'     · {c}')
    json.dump({'sim': R, 'surrogate_pred': pj['surrogate_pred'], 'caveats': pj['caveats']}, open(OUT / 'verify.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n已写入 {OUT}/verify.json')
    return 0

def calib(args) -> int:
    import copy
    pj = json.loads((OUT / 'plans.json').read_text())
    W0 = _sim_water('baseline', 0.0, args)
    print(f'基准实测总注水 {W0:,.0f}\n')
    out = {'baseline': {'s': 0.0, 'water_dev': 0.0}}
    for tag in ('short', 'long', 'switch'):
        th0 = np.asarray(pj['plans'][tag], dtype=np.float64)
        pts = []
        for it in range(args.max_iter):
            if it == 0:
                ls = 0.0
            elif it == 1:
                ls = -np.log10(1.0 + pts[0][1])
            else:
                ((x1, y1), (x2, y2)) = (pts[-2], pts[-1])
                ls = x2 - y2 * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-12 else x2
                ls = float(np.clip(ls, -0.3, 0.3))
            W = _sim_water(tag, ls, args, theta=th0 + ls)
            dev = W / W0 - 1.0
            pts.append((ls, dev))
            print(f'  {tag:8s} iter{it}  s=10^{ls:+.4f}={10 ** ls:.4f}  实测注水 {W:,.0f}  偏离 {dev:+.2%}', flush=True)
            if abs(dev) <= args.water_tol:
                break
        best = min(pts, key=lambda t: abs(t[1]))
        out[tag] = {'s': best[0], 'water_dev': best[1]}
        print(f'  → {tag} 选 s=10^{best[0]:+.4f},实测偏离 {best[1]:+.2%}\n')
    pj2 = copy.deepcopy(pj)
    pj2['plans'] = {k: (np.asarray(v, dtype=np.float64) + out[k]['s']).tolist() for (k, v) in pj['plans'].items()}
    pj2['calibration'] = out
    pj2['caveats'].append('方案经**实测注水量**标量校正(整体缩放,不改井间份额),因为 F-4H 被 BHP 截断导致目标守恒≠实测守恒')
    (OUT / 'plans_calibrated.json').write_text(json.dumps(pj2, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'已写入 {OUT}/plans_calibrated.json  →  下一步 verify --calibrated')
    return 0

def _sim_water(tag, ls, args, theta=None):
    key = f'{tag}_s{ls:+.4f}'
    sh = OUT / 'sim' / f'{key}.npz'
    if tag == 'baseline' and abs(ls) < 1e-12 and (OUT / 'sim' / 'baseline.npz').exists():
        sh = OUT / 'sim' / 'baseline.npz'
    if not sh.exists():
        pj = json.loads((OUT / 'plans.json').read_text())
        th = (np.asarray(pj['plans'][tag], dtype=np.float32) if theta is None else np.asarray(theta, dtype=np.float32)).reshape(N_STAGE, len(INJ))
        _run_sim(key, th, args)
        sh = OUT / 'sim' / f'{key}.npz'
    d = np.load(sh)
    return float(d['field_cum'][1][-1] - d['field_cum'][1][0])

def _run_sim(key, th, args):
    work_root = FG.OUT / '_decide'
    work_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FG.HIST / 'meta.json').read_text())
    hist_rel = str((FG.HIST / 'out' / 'NORNE_ATW2013').resolve())
    w = work_root / key
    shutil.rmtree(w, ignore_errors=True)
    try:
        FG.build_restart_deck(w, th, hist_rel, meta['last_step'])
        cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{w}:/data', '-v', f"{FG.HIST / 'out'}:{FG.HIST / 'out'}:ro", '-w', '/data', NB.IMAGE, 'flow', NB.DECK, '--output-dir=/data/out', f'--threads-per-process={args.threads}', '--parsing-strictness=low']
        with open(w / 'run.log', 'w') as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200).returncode
        if rc != 0:
            raise SystemExit(f'🔴 {key}: flow rc={rc}')
        g = FG.harvest_fc(w)
        if g is None:
            raise SystemExit(f'🔴 {key}: harvest 失败')
        (OUT / 'sim').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / 'sim' / f'{key}.npz', theta=th, **g)
    finally:
        shutil.rmtree(w, ignore_errors=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'calib', 'verify', 'report'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n-pin', type=int, default=8000)
    ap.add_argument('--ntrain', type=int, default=7400)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--trust-r', type=float, default=2.0)
    ap.add_argument('--water-pen', type=float, default=50.0)
    ap.add_argument('--water-tol', type=float, default=0.02)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-iter', type=int, default=4)
    ap.add_argument('--calibrated', action='store_true', help='verify/report 使用 plans_calibrated.json')
    a = ap.parse_args()
    return {'plan': plan, 'calib': calib, 'verify': verify, 'report': report}[a.cmd](a)
if __name__ == '__main__':
    raise SystemExit(main())
