from __future__ import annotations
import datetime as dt
import glob
import json
from pathlib import Path
import numpy as np
import forecast_gen as FG
ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / '_pipelines' / 'fc_decide' / 'sim'
OUT = ROOT / '_pipelines' / 'fc_fix'
DAYS = FG.FC_GRID
T_START = dt.date(1997, 11, 6)
BBL = 6.2898
BRENT = {2006: 65.16, 2007: 72.44, 2008: 96.94, 2009: 61.74, 2010: 79.61, 2011: 111.26, 2012: 111.63, 2013: 108.56, 2014: 98.97, 2015: 52.32, 2016: 43.64, 2017: 54.13, 2018: 71.34, 2019: 64.3, 2020: 41.96}

def load(key):
    d = np.load(SIM / f'{key}.npz')
    fc = d['field_cum']
    (fopt, fwit) = (fc[0], fc[1])
    oil_cum = fopt - fopt[0]
    wat_cum = fwit - fwit[0]
    yrs = np.array([(T_START + dt.timedelta(days=float(x))).year for x in DAYS])
    price = np.array([BRENT[y] for y in yrs])
    t = (DAYS - DAYS[0]) / 365.25
    dq = np.diff(oil_cum, prepend=oil_cum[0])
    rev = dq * BBL * price
    ms = t <= 3.0
    npv = {f'npv{int(r * 100)}': float((rev / (1 + r) ** t).sum()) for r in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15)}
    return {**npv, 'oil': float(oil_cum[-1]), 'short': float(oil_cum[ms][-1]), 'water': float(wat_cum[-1]), 'water_3y': float(wat_cum[ms][-1]), 'theta': d['theta']}

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b = load('ms_base')
    print(f"基准(模拟器自报口径): 产油 {b['oil']:,.0f}  注水 {b['water']:,.0f}  NPV@8% {b['npv8'] / 1000000.0:,.1f}M$\n")
    ms = json.loads((ROOT / '_pipelines' / 'fc_decide' / 'multistart.json').read_text())
    picks = {}
    for nm in ('short', 'long'):
        k = ms['results'][nm + '_pick']['k']
        tgt = ms['results'][nm + '_pick']['oil']
        (best, bd) = (None, 1e+30)
        for f in sorted(SIM.glob(f'ms_{nm}_{k}_c*.npz')):
            import norne_bulk as NB
            ob = np.load(f)['obs']
            n = len(NB.PRODUCERS)
            T = FG.FC_N
            o = float(np.trapezoid(np.stack([ob[i * 3 * T:(i * 3 + 1) * T] for i in range(n)]).sum(0), DAYS))
            if abs(o - tgt) < bd:
                (best, bd) = (f.stem, abs(o - tgt))
        picks[nm] = best
    print('=== 1. 主线三方案:梯形积分 vs 模拟器自报 ===')
    print(f"  {'方案':10s}{'产油(新口径)':>15s}{'vs基准':>9s}{'注水偏离(新)':>13s}{'注水偏离(旧)':>13s}{'NPV@8%':>12s}{'vs基准':>11s}")
    R = {'baseline': b}
    for (nm, key) in picks.items():
        m = load(key)
        R[nm] = m
        import norne_bulk as NB
        d = np.load(SIM / f'{key}.npz')
        old_w = float(d['field_cum'][1][-1] - d['field_cum'][1][0])
        old_wb = float((lambda z: z['field_cum'][1][-1] - z['field_cum'][1][0])(np.load(SIM / 'ms_base.npz')))
        print(f"  {nm:10s}{m['oil']:>15,.0f}{(m['oil'] / b['oil'] - 1) * 100:>+8.2f}%{(m['water'] / b['water'] - 1) * 100:>+12.2f}%{(old_w / old_wb - 1) * 100:>+12.2f}%{m['npv8'] / 1000000.0:>10,.1f}M${(m['npv8'] - b['npv8']) / 1000000.0:>+9.1f}M$")
    gap = R['long']['oil'] / R['short']['oil'] - 1
    print(f'\n  🔴 只顾眼前的代价(新口径) = {gap:+.2%}')
    print(f"     短期最优前 3 年 {(R['short']['short'] / b['short'] - 1) * 100:+.2f}%")
    print(f'\n  折现率敏感性(vs 基准, 百万美元):')
    print(f"  {'折现率':>8s}{'短期最优':>13s}{'长期最优':>13s}")
    prev = None
    for rr in (0, 2, 4, 6, 8, 10, 12, 15):
        k = f'npv{rr}'
        (ds, dl) = ((R['short'][k] - b[k]) / 1000000.0, (R['long'][k] - b[k]) / 1000000.0)
        mark = ''
        if prev is not None and (prev[0] < prev[1]) != (ds < dl):
            mark = '  ← 🔴 反转发生在这一档与上一档之间'
        prev = (ds, dl)
        print(f"  {str(rr) + '%':>8s}{ds:>+12,.1f}M{dl:>+12,.1f}M   {('短期赢' if ds > dl else '长期赢')}{mark}")
    print(f'\n=== 2. ΔNPV 到底由什么解释:时间前置 还是 井间重分配? ===')
    rows = []
    for f in sorted(SIM.glob('*.npz')):
        if f.stem in ('ms_base',):
            continue
        try:
            m = load(f.stem)
        except Exception:
            continue
        th = m['theta'].reshape(FG.N_STAGE, -1)
        if th.shape[1] != len(FG.INJ_W):
            continue
        rows.append({'key': f.stem, 'dnpv': (m['npv8'] - b['npv8']) / 1000000.0, 'front': m['water_3y'] / m['water'] - b['water_3y'] / b['water'], 'spread': float(np.std(10.0 ** th, axis=1).mean()) - 0.0, 'wdev': m['water'] / b['water'] - 1})
    if len(rows) < 8:
        print(f'  可用算例只有 {len(rows)} 个,样本太少,不做回归')
        return 0
    y = np.array([r['dnpv'] for r in rows])
    A = np.array([r['front'] for r in rows])
    B = np.array([r['spread'] for r in rows])
    W = np.array([r['wdev'] for r in rows])
    print(f'  样本 {len(rows)} 个算例')

    def r2(X, y):
        X = np.column_stack([np.ones(len(y))] + [x for x in X])
        (beta, *_) = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        return (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), beta)
    for (nm, X) in (('只用 时间前置', [A]), ('只用 井间离散度', [B]), ('只用 总注水偏离', [W]), ('前置+离散度', [A, B]), ('前置+离散度+水量', [A, B, W])):
        (v, _) = r2(X, y)
        print(f'    {nm:18s} R² = {v:.4f}')
    (ra, _) = r2([A], y)
    (rab, _) = r2([A, B], y)
    print(f'\n  🔴 控制「时间前置」后,「井间重分配」的增量 R² = {rab - ra:+.4f}')
    print(f"     {('→ 空间重分配几乎没有独立贡献,论文叙事需改' if rab - ra < 0.05 else '→ 空间重分配有独立贡献,原叙事成立')}")
    json.dump({'note': '全部改用模拟器自报累计量(FOPT/FWIT),不再用 40 点梯形积分', 'baseline': {k: v for (k, v) in b.items() if k != 'theta'}, 'results': {k: {kk: vv for (kk, vv) in v.items() if kk != 'theta'} for (k, v) in R.items()}, 'headline_gap': gap, 'picks': picks, 'causal': {'n': len(rows), 'r2_front': ra, 'r2_front_spread': rab, 'delta_r2_spread': rab - ra}}, open(OUT / 'fixed.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n已写入 {OUT}/fixed.json')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
