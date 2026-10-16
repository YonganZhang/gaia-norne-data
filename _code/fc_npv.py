from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import numpy as np
import forecast_gen as FG
import norne_bulk as NB
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_decide'
NT = FG.FC_N
DAYS = FG.FC_GRID
T_START = dt.date(1997, 11, 6)
BBL_PER_M3 = 6.2898
BRENT = {2006: 65.16, 2007: 72.44, 2008: 96.94, 2009: 61.74, 2010: 79.61, 2011: 111.26, 2012: 111.63, 2013: 108.56, 2014: 98.97, 2015: 52.32, 2016: 43.64, 2017: 54.13, 2018: 71.34, 2019: 64.3, 2020: 41.96}

def oil_rate(tag):
    d = np.load(OUT / 'sim' / f'{tag}.npz')
    ob = d['obs']
    n = len(NB.PRODUCERS)
    return np.stack([ob[i * 3 * NT:(i * 3 + 1) * NT] for i in range(n)]).sum(0)

def main() -> int:
    dates = [T_START + dt.timedelta(days=float(x)) for x in DAYS]
    years = np.array([d.year for d in dates])
    price = np.array([BRENT[y] for y in years])
    w = np.gradient(DAYS).astype(float)
    w[0] *= 0.5
    w[-1] *= 0.5
    t_yr = (DAYS - DAYS[0]) / 365.25
    print(f'预测段 {dates[0]} → {dates[-1]}   {len(DAYS)} 个时间格点')
    print(f'Brent 年均价范围 {price.min():.2f} ~ {price.max():.2f} 美元/桶 (波动 {price.max() / price.min():.1f}×)\n')
    print('  年份   Brent(美元/桶)')
    for y in sorted(set(years)):
        print(f'  {y}    {BRENT[y]:>6.2f}')
    ms = json.loads((OUT / 'multistart.json').read_text())

    def pick_calibrated(nm):
        k = ms['results'][nm + '_pick']['k']
        target = ms['results'][nm + '_pick']['oil']
        (best, bd) = (None, 1e+30)
        for f in sorted((OUT / 'sim').glob(f'ms_{nm}_{k}_c*.npz')):
            d = np.load(f)
            ob = d['obs']
            n = len(NB.PRODUCERS)
            o = float(np.trapezoid(np.stack([ob[i * 3 * NT:(i * 3 + 1) * NT] for i in range(n)]).sum(0), DAYS))
            if abs(o - target) < bd:
                (best, bd) = (f.stem, abs(o - target))
        if best is None or bd > 1.0:
            raise SystemExit(f'🔴 {nm}: 找不到与 multistart 记录(产油 {target:,.0f})匹配的算例')
        return best
    picks = {'基准': 'ms_base', '短期最优': pick_calibrated('short'), '长期最优': pick_calibrated('long')}
    print(f'\n用的算例: ' + '  '.join((f'{k}={v}' for (k, v) in picks.items())))
    R = {}
    for (name, key) in picks.items():
        q = oil_rate(key)
        rev_t = q * w * BBL_PER_M3 * price
        R[name] = {'oil_m3': float((q * w).sum()), 'rev_undisc': float(rev_t.sum()), 'npv': {}}
        for r in (0.0, 0.08, 0.1, 0.15):
            R[name]['npv'][f'{r:.0%}'] = float((rev_t / (1 + r) ** t_yr).sum())
    b = R['基准']
    print(f"\n{'=' * 86}\n=== 🔴 用真实历史 Brent 油价的经济评价 ===\n")
    print(f"  {'方案':10s}{'累计产油(m³)':>16s}{'未贴现收入':>16s}{'NPV@8%':>16s}{'vs基准@8%':>14s}")
    for name in ('基准', '短期最优', '长期最优'):
        d = R[name]
        n8 = d['npv']['8%']
        print(f"  {name:10s}{d['oil_m3']:>16,.0f}{d['rev_undisc'] / 1000000.0:>14,.1f}M${n8 / 1000000.0:>14,.1f}M${(n8 - b['npv']['8%']) / 1000000.0:>+12,.1f}M$")
    print(f'\n  === 折现率敏感性(vs 基准,单位:百万美元) ===')
    print(f"  {'折现率':>8s}{'短期最优':>14s}{'长期最优':>14s}{'两者之差':>14s}")
    for r in ('0%', '8%', '10%', '15%'):
        s = (R['短期最优']['npv'][r] - b['npv'][r]) / 1000000.0
        l = (R['长期最优']['npv'][r] - b['npv'][r]) / 1000000.0
        print(f'  {r:>8s}{s:>+13,.1f}M{l:>+13,.1f}M{s - l:>+13,.1f}M')
    print(f"\n  🔴 关键:折现率越高,'早采'越占便宜 —— 这正是短期策略的价值所在。")
    print(f'     而这段历史里 2008 年 Brent 年均 96.94、2011-2014 稳在 ~110,')
    print(f"     '把产量前移'恰好吃到高价期。")
    json.dump({'source': 'EIA RBRTE annual, https://www.eia.gov/dnav/pet/hist_xls/RBRTEa.xls', 'brent': BRENT, 'period': [str(dates[0]), str(dates[-1])], 'cases': picks, 'results': R, 'caveats': ['只算收入侧,不含 CAPEX/OPEX/税费(三方案井数设备注水量相同,成本近似抵消;计入注水 OPEX 则长期方案更有利,故本口径保守)', '油价用年均值,不用月度 —— 产量输出仅 40 个格点(约 120 天/点),月度价会制造虚假精度']}, open(OUT / 'npv.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n已写入 {OUT}/npv.json')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
