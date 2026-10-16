from __future__ import annotations
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
SIMDIR = ROOT / '_pipelines' / 'fc_decide' / 'sim'

def _fc(npz) -> np.ndarray:
    d = npz if isinstance(npz, np.lib.npyio.NpzFile) else np.load(npz)
    return d['field_cum']

def measured_winj(npz) -> float:
    fc = _fc(npz)
    return float(fc[1][-1] - fc[1][0])

def measured_oil(npz) -> float:
    fc = _fc(npz)
    return float(fc[0][-1] - fc[0][0])

def measured_wprod(npz) -> float:
    fc = _fc(npz)
    return float(fc[2][-1] - fc[2][0])
BASELINE_NPZ = SIMDIR / 'baseline.npz'

def baseline() -> dict:
    return {'oil': measured_oil(BASELINE_NPZ), 'winj': measured_winj(BASELINE_NPZ), 'wprod': measured_wprod(BASELINE_NPZ)}
