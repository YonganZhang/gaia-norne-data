"""Verify release inputs without extracting files or trusting pickle objects."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True,help='Downloaded norne-forecast-data.zip')
    args=parser.parse_args()
    manifest=json.loads((Path(__file__).parent/'data_manifest.json').read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest()==manifest['archive']['sha256'],'archive digest mismatch'
    checked=0
    with zipfile.ZipFile(args.archive) as z:
        assert len(z.namelist())==len(set(z.namelist())),'duplicate archive entries'
        for name,expected in manifest['files'].items():
            raw=z.read(name)
            assert hashlib.sha256(raw).hexdigest()==expected,name
            if name.endswith('.npz'):
                with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                    assert int(d['rc'])==0 and all(np.isfinite(d[k]).all() for k in d.files),name
                checked+=1
    print(json.dumps({'passed':True,'samples_checked':checked,'input_files_checked':len(manifest['files'])}))

if __name__=='__main__':main()
