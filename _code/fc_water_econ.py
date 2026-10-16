import numpy as np, datetime as dt, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fc_fix import BRENT, T_START, BBL

def econ(npz, c_inj=0.0, c_prod=0.0, rates=(0.0, 0.02, 0.08, 0.15)):
    d = np.load(npz)
    (fc, days) = (d['field_cum'], d['grid_days'])
    (fopt, fwit, fwpt) = (fc[0] - fc[0][0], fc[1] - fc[1][0], fc[2] - fc[2][0])
    yrs = np.array([(T_START + dt.timedelta(days=float(x))).year for x in days])
    price = np.array([BRENT[y] for y in yrs])
    t = (days - days[0]) / 365.25
    d_oil = np.diff(fopt, prepend=fopt[0])
    d_winj = np.diff(fwit, prepend=fwit[0])
    d_wprd = np.diff(fwpt, prepend=fwpt[0])
    cash = d_oil * BBL * price - d_winj * BBL * c_inj - d_wprd * BBL * c_prod
    out = {f'npv{int(r * 100)}': float((cash / (1 + r) ** t).sum()) for r in rates}
    out.update(oil=float(fopt[-1]), oil_3y=float(np.interp(3.0, t, fopt)), winj=float(fwit[-1]), wprd=float(fwpt[-1]))
    return out

def main():
    sims = sorted(glob.glob('_pipelines/*/sim/*.npz'))
    base = [s for s in sims if Path(s).stem in ('baseline', 'ms_base', 'baseline_cal')]
    if not base:
        print('找不到基准算例')
        return
    print(f'算例总数 {len(sims)}，基准 {Path(base[0]).name}\n')
    grid = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 2.0), (4.0, 3.0)]
    print(f"{'c_inj/c_prod':>14s} {'基准NPV8(M$)':>13s} {'最佳算例':>26s} {'ΔNPV8(M$)':>11s} {'其注水量vs基准':>14s}")
    print('-' * 88)
    rows = []
    for (ci, cp) in grid:
        b = econ(base[0], ci, cp)
        (best, bv) = (None, -9e+18)
        for s in sims:
            if Path(s).stem in ('baseline', 'ms_base', 'baseline_cal'):
                continue
            try:
                e = econ(s, ci, cp)
            except Exception:
                continue
            if e['npv8'] > bv:
                (best, bv) = ((Path(s).stem, e), e['npv8'])
        if best is None:
            continue
        (nm, e) = best
        dw = 100 * (e['winj'] / b['winj'] - 1)
        print(f"{ci:5.1f}/{cp:<8.1f} {b['npv8'] / 1000000.0:13,.1f} {nm:>26s} {(bv - b['npv8']) / 1000000.0:+11.1f} {dw:+13.2f}%")
        rows.append(dict(c_inj=ci, c_prod=cp, base_npv8=b['npv8'], best=nm, delta_npv8=bv - b['npv8'], water_dev=dw / 100))
    Path('_pipelines/fc_water_econ').mkdir(parents=True, exist_ok=True)
    json.dump({'grid': rows, 'note': '含水成本重评价，复用已有模拟，未跑新算例'}, open('_pipelines/fc_water_econ/sweep.json', 'w'), indent=1)
    print('\n→ _pipelines/fc_water_econ/sweep.json')
if __name__ == '__main__':
    main()
