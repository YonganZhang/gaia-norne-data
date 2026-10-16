"""Public-release paths. No private workstation paths or credentials."""
import os
from pathlib import Path
from release_paths import ROOT, data_path

def resolve(key, relative):
    return Path(os.environ.get(key,str(ROOT/relative))).expanduser()
NORNE_DECK=resolve('NORNE_DECK','vendor/norne')
NORNE_BULK=data_path('norne')
NORNE_BULK_V2=data_path('norne_v2')
NORNE_OBS_V2=data_path('norne_obs_v2')
NORNE_GRID_CACHE=data_path('norne_grid_cache.npz')
CONTROLS_JSON=ROOT/'_meta/norne_controls.json'
DECK_HOWTO='Obtain https://github.com/OPM/opm-data and set NORNE_DECK to its norne directory.'
def require(p,what,how):
    if not p.exists(): raise FileNotFoundError(f'{what}: {p}. {how}')
    return p
