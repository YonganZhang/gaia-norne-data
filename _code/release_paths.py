"""Portable data root; set GAIA_DATA_ROOT to the directory containing norne_fc/."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def data_path(name):
    return Path(os.environ.get('GAIA_DATA_ROOT',str(ROOT/'data'))).expanduser()/name
