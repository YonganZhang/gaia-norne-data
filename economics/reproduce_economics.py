"""Run inside the exported economics companion; requires NumPy only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from economics import econ

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
baseline = HERE / manifest['baseline']
reference = HERE / manifest['reference']
policies = [HERE / p for p in manifest['policies']]
expected = json.loads((HERE / 'expected.json').read_text())
assert len(policies) == 10 and len(set(policies)) == 10
grid = {(a, b, r, p) for a, b in ((0., 0.), (1., .5), (2., 1.), (3., 2.), (4., 3.))
        for r in (0, 2, 8, 15) for p in (.8, 1., 1.2)}
assert len(expected['economic_sensitivity']) == 60
assert {(r['c_inj'], r['c_prod'], r['discount_percent'], r['price_scale'])
        for r in expected['economic_sensitivity']} == grid
assert [r['water_limit'] for r in expected['water_limit_sensitivity']] == [.1, .15, .2, .25, .3]


def close(actual, target):
    np.testing.assert_allclose(actual, target, rtol=1e-10, atol=1e-7)

b = econ(baseline, 2., 1.)
gains = np.array([econ(p, 2., 1.)['npv8'] - b['npv8'] for p in policies]) / 1e6
ref = (econ(reference, 2., 1.)['npv8'] - b['npv8']) / 1e6
close(gains, [r['gain_USD']/1e6 for r in expected['primary']['runs']])
close(gains.mean(), expected['primary']['mean_MUSD'])
close(ref, expected['primary']['reference_MUSD'])
rows = []
for row in expected['economic_sensitivity']:
    rate = 'npv' + str(row['discount_percent'])
    def value(path):
        return econ(path, row['c_inj'], row['c_prod'])[rate] + (row['price_scale']-1)*econ(path, 0., 0.)[rate]
    base = value(baseline)
    values = np.array([value(p)-base for p in policies]) / 1e6
    reference_gain = (value(reference)-base) / 1e6
    measured = {'mean_gain_MUSD': float(values.mean()),
                'mean_margin_over_fixed_reference_MUSD': float(values.mean()-reference_gain),
                'runs_above_reference': int((values > reference_gain).sum()),
                'min_gain_MUSD': float(values.min())}
    for key, val in measured.items():
        close(val, row[key])
    rows.append({**row, **measured})
for row in expected['water_limit_sensitivity']:
    retained = [abs(econ(p, 2., 1.)['winj']/b['winj']-1) <= row['water_limit'] for p in policies]
    close(sum(retained), row['retained_policies'])
    close(np.mean(gains*np.array(retained)), row['mean_gain_MUSD'])
    feasible_ref = abs(econ(reference, 2., 1.)['winj']/b['winj']-1) <= row['water_limit']
    close(ref if feasible_ref else 0., row['reference_gain_MUSD'])

def components(p):
    oil = econ(p, 0., 0.)['npv8']
    return np.array([oil, econ(p, 2., 0.)['npv8']-oil, econ(p, 0., 1.)['npv8']-oil])/1e6

base = components(baseline)
team = np.mean([components(p)-base for p in policies], axis=0)
reference_parts = components(reference)-base
for i, row in enumerate(expected['economic_decomposition']):
    close([team[i], reference_parts[i], team[i]-reference_parts[i]],
          [row['team_gain_MUSD'], row['reference_gain_MUSD'], row['margin_MUSD']])
report = {'passed': True, 'manifest_sha256': hashlib.sha256((HERE / 'manifest.json').read_bytes()).hexdigest(), 'primary_runs': len(gains), 'mean_gain_MUSD': float(gains.mean()),
          'reference_gain_MUSD': float(ref), 'scenarios': len(rows),
          'positive_mean_margins': sum(r['mean_margin_over_fixed_reference_MUSD'] > 0 for r in rows),
          'water_limits_checked': 5, 'decomposition_components_checked': 3,
          'scope': 'Revalues already selected physical trajectories; does not regenerate agent decisions or verify F4 feedback.'}
(HERE / 'reproduced.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
