"""Offline checks of published weights, schema, team role slices and baseline economics."""
import json
import numpy as np
import torch
from release_paths import ROOT
from fc_team_proxy import Proxy
from fc_team_loop import slice_for,ROLES
from fc_water_econ import econ

torch.set_num_threads(1);p=Proxy();y=p.trajectories(np.zeros(24,dtype=np.float32))
assert y.shape==(1,11,2,40) and np.isfinite(y).all()
carto=json.loads((ROOT/'_pipelines/fc_team/cartography.json').read_text())
for role in ROLES:
    text,provenance=slice_for(role['id'],carto,None)
    assert text and provenance
base=econ(ROOT/'examples/baseline.npz',2.,1.)['npv8']/1e6
assert abs(base-1773.397307737011)<1e-6,base
print(json.dumps({'status':'passed','trajectory_shape':list(y.shape),'roles':len(ROLES),'baseline_npv_MUSD':base,'llm_calls':0,'simulator_calls':0}))
