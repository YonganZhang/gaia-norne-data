# Gaia: research code and Norne simulation data

Companion to **Gaia: a large language model agent team for water-injection decision optimisation**. The manuscript is a research manuscript; this repository does not imply journal acceptance.

Version 0.2.0 adds the research implementation and frozen surrogate to the existing 0.1.0 data release. Data bytes and their licenses are unchanged.

## What is public

| Component | Location | Use |
|---|---|---|
| 8,000 main + 500 confirmation simulated schedules | [Data archive, v0.1.0](https://github.com/YonganZhang/gaia-norne-data/releases/download/v0.1.0/norne-forecast-data.zip) | Train/evaluate the surrogate |
| Transformer architecture, original training routine and a training-only entrypoint | `_code/fc_mech.py`, `_code/fc_rank_fidelity.py`, `_code/train_surrogate.py` | Inspect and retrain PosNet |
| Frozen trained weights, normalisation and injectivity helpers | `_pipelines/` | Inference without retraining |
| Specialist roles, prompts, collaboration, synthesis, physical adjudication | `_code/fc_team.py`, `_code/fc_team_loop.py` | Inspect or run a new decision experiment |
| Forecast schedule generation and OPM adapters | `_code/forecast_gen.py`, `_code/norne_bulk.py`, `_code/fc_decide.py` | Run new reservoir simulations with upstream decks |
| 12 fixed physical trajectories and economic reproduction | `economics/` | Reproduce primary mean gains and sensitivity grid |
| Data definitions, SHA-256 and licenses | `DATA_DICTIONARY.md`, `data_manifest.json`, `LICENSES/` | Interpret and verify inputs |

## Quick start: no account, simulator or GPU

Python 3.11 or newer is recommended. Install PyTorch appropriate for your CPU/GPU; the requirements allow compatible versions rather than promising bitwise equivalence.

```bash
python -m pip install -r requirements-research.txt
python _code/smoke_test.py
python _code/predict.py --theta examples/theta.json --output research-output/prediction.npz
python economics/reproduce_economics.py
```

The smoke test loads the released weights, checks `(1, 11, 2, 40)` output, creates every specialist's input slice, and independently checks the baseline economic calculation. It makes no language-model or simulator calls.

The input is a **complete injection schedule**, not a daily/monthly observation. `theta` has 6 stages × 4 injectors, in stage-major order. Stages begin in 2007, 2008, 2010, 2012, 2014 and 2017. Injector order is F-1H, F-2H, F-3H, F-4H. Values are log10 multipliers: `theta[0,0] = 0.1` requests about `1.259 × 8554.2 = 10769 sm3/day` for F-1H in the first stage. Actual injection can be clipped by simulator pressure constraints. The model predicts oil and water rates at 40 reporting nodes for 11 active producer outputs over the forecast horizon.

## Download, verify and evaluate all data

Download `norne-forecast-data.zip` from the v0.1.0 release. SHA-256:
`b3e376b9eee5b35da890616f4ee6d5955c6b7ba8ce70de7754e342ae3b3e9abe`.

```bash
python verify_data.py --archive norne-forecast-data.zip
python -m zipfile -e norne-forecast-data.zip .
python _code/evaluate_surrogate.py
python _code/train_surrogate.py --device cpu --epochs 600 --output research-output/retrained
```

Set `GAIA_DATA_ROOT` if the extracted `norne_fc` and `norne_fc_confirm` directories live elsewhere. Training reserves the first 400 indices of the seed-0 permutation for testing, the next 200 for validation, and trains on the remaining 7,400. The deployed routine uses a fixed 600-epoch budget; it does not select a checkpoint using those 200 validation samples. Retraining writes a new directory and does not overwrite the released model. GPU training is supported with `--device cuda:0`.

## Running a new agent experiment

The release-time team implementation includes specialist feedback corrections made after the original ten F2 observations. It must **not** be represented as an exact historical replay of F2. Model aliases, service behaviour and stochastic outputs can also change. The paper's primary results are independently reproduced from their fixed physical trajectories in `economics/`; the source release makes the architecture and current execution workflow inspectable and reusable.

For a new run, install the Claude CLI and authenticate your own account, install Docker/OPM Flow and `resdata`, obtain [OPM's Norne benchmark](https://github.com/OPM/opm-data/tree/master/norne), and set `NORNE_DECK`. The generator can first create a history restart with `python _code/forecast_gen.py history`. Review `python _code/fc_team_loop.py --help` and use a new unique run tag. Historical F2/G2 tags are rejected to prevent mixing experiments. New LLM or simulator runs use real compute and potentially a paid account; none is started by the quick-start commands.

The optional calibration-derived `tilt_bins.json` is intentionally absent from the default release. F4's later calibration prior, model and feedback configuration are a distinct treatment, described in the manuscript supplement. No paused project queue is distributed or started. An optional `knowledge/` directory is for a user's own licensed background material; no copyrighted paper full texts are redistributed. Executable team role prompts and the domain rules used by the current implementation are included in source.

## Research interpretation

The forecast inputs are simulation outputs from one fixed Norne geological model, not new field measurements. The primary cohort is ten historical team runs. The single-specialist comparator retains a separate synthesiser; the between-configuration mean difference was not statistically established. The uniform reference searches one dimension while Gaia adjusts 24 controls. The 60 economic scenarios revalue fixed selected schedules, not 60 newly optimised policies. Later Haiku and Luna cohorts remain separate.

Timing in the paper is batched surrogate inference, not whole-workflow time or an estimate of engineering labour savings. Final reported economics use simulator-reported cumulative quantities. The surrogate's screening integration must not be substituted for those final totals.

## Provenance and licensing

Code: MIT, see `LICENSE`. Norne-derived data: ODbL 1.0 database terms and DbCL 1.0 contents terms, see `DATA_LICENSE.md` and `LICENSES/`. Model weights and normalisation files are distributed with the Norne-derived-data attribution and terms in `DATA_LICENSE.md`.

`RESEARCH_CODE_MANIFEST.json` records the source and export hashes. Export changes remove internal development commentary and docstrings, adapt workstation paths to portable configuration, and add standalone entrypoints. Research prompts and mathematical routines are retained. The export is a source snapshot at release time, not an attestation of every historical runtime dependency. Neither private Git history, credentials nor raw language-model conversations are included.
