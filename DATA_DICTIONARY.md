# Data dictionary

Each NPZ is a simulated schedule; filenames are zero-based sample indices.

| Field | Shape | Meaning |
|---|---|---|
| theta | (6,4) | log10 injection-rate multipliers; actual targets multiply the baseline by 10**theta |
| obs | (2640,) | producer-major, variable-next, time-inner; reshape to (22,3,40) |
| inj_actual | (4,40) | simulator-reported actual water injection rate, sm3/day |
| field_cum | (4,40) | rows FOPT, FWIT, FWPT, FPR; first three are cumulative sm3, last is pressure in bar |
| grid_days | (40,) | summary interpolation grid in simulation days, measured from 1997-11-06 |
| snap_days | (8,) | requested optional field-snapshot times; presence does not prove field arrays exist |
| has_fields | scalar | whether optional spatial fields were harvested |
| rc | scalar | simulator process return code; released samples are zero |
| sim_days_end | scalar | last simulation day available |

`theta_all.npy` is (N,6,4), matching shards in index order. Stage years: 2007, 2008, 2010, 2012, 2014, 2017. Injector order: F-1H, F-2H, F-3H, F-4H. Exact producer and observation-variable order is in data_manifest.json. Observation variables are WOPR (oil rate, sm3/day), WWPR (water rate, sm3/day) and WBHP (bottom-hole pressure, bar). WBHP is not gas production.

`field_cum` contains three cumulative quantities and one pressure row; do not integrate all four as rates. Economic gains use differences in simulator-reported cumulative totals, not trapezoidal integration of the coarse rate grid. Gains in tables are M USD; injection deviation is explicitly percent or fraction as the column name indicates. Outside the final absolute 25% injection-deviation tolerance, F8 executable gain falls back to zero even when raw economic gain is positive.
