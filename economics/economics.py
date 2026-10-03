import numpy as np
import datetime as dt
import json
from pathlib import Path
data = json.loads((Path(__file__).parent / 'constants.json').read_text())
BRENT = {int(k): v for k, v in data['BRENT'].items()}
T_START = dt.datetime.fromisoformat(data['T_START'])
BBL = data['BBL']

def econ(npz, c_inj=0.0, c_prod=0.0, rates=(0.0,0.02,0.08,0.15)):
    """一条算例在给定水价下的 NPV。收入-成本都按格点差分后逐点折现。"""
    d = np.load(npz)
    fc, days = d["field_cum"], d["grid_days"]
    fopt, fwit, fwpt = fc[0]-fc[0][0], fc[1]-fc[1][0], fc[2]-fc[2][0]
    yrs = np.array([(T_START+dt.timedelta(days=float(x))).year for x in days])
    price = np.array([BRENT[y] for y in yrs])
    t = (days-days[0])/365.25
    d_oil  = np.diff(fopt, prepend=fopt[0])
    d_winj = np.diff(fwit, prepend=fwit[0])
    d_wprd = np.diff(fwpt, prepend=fwpt[0])
    cash = d_oil*BBL*price - d_winj*BBL*c_inj - d_wprd*BBL*c_prod
    out = {f"npv{int(r*100)}": float((cash/(1+r)**t).sum()) for r in rates}
    # 前 3 年用插值到精确 3.0 年，消除 fc_fix 的格点 off-by-one(约 155k Sm3)
    out.update(oil=float(fopt[-1]), oil_3y=float(np.interp(3.0, t, fopt)),
               winj=float(fwit[-1]), wprd=float(fwpt[-1]))
    return out
