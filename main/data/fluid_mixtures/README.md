# Lab fluid mixtures

The task is a pool of the 1,300 heat-transfer fluid mixtures that sdagent's lab
measured in campaign 11 (`cleaned_processed_data.csv`). It covers 12 fluids: 12 pure
rows, 157 binary and 1,131 ternary. Every row has μ, c_p and ρ measured. Campaigns
1–10 are left out because sdagent keeps their pure-component basis separate from
campaign 11's. Campaign 12 is left out because it has no c_p.

- **Features:** the volume-fraction-weighted mean of the components' RDKit 2D
  descriptors. That is RDKit's full descriptor set minus those that are constant or
  non-finite across the 12 fluids, which leaves 109. The names are in
  `provenance.json`. The feature matrix has rank 11, because it is a linear image
  of the composition simplex.
- **Objective:** `y = log10(μ / (c_p·ρ)²)`, the laminar pumping-power figure of merit
  at fixed heat duty, in m³·s·K²/J. μ is in Pa·s (converted from cP), c_p in J/(kg·K)
  and ρ in kg/m³ (converted from g/mL). Lower is better.
- **Optimum:** `f_opt = −15.998`, a 25/75 binary. The best pure fluid ranks 6th.

The measurements are proprietary (sdagent `docs/data.md`) and this repository is
public. So the inputs and the built pool live only in git-ignored
`external/fluid_mixtures/`:

| File | Source |
|---|---|
| `measurements.csv` | sdagent `data/processed/measurements.csv` (on the work machine) |
| `fluid_library.csv` | sdagent `data/processed/fluid_library.csv`, used for the SMILES by CID |
| `data.csv` | the built pool, written by `build.py` |

Only `build.py` and `provenance.json` are tracked. `provenance.json` records the
input and output hashes and the RDKit version. RDKit isn't in this repository's
environments, so rebuild the pool with sdagent's:

```bash
~/Documents/Work/Grad/sdagent/.venv/bin/python main/data/fluid_mixtures/build.py
```

The task is registered as `fluid_mixtures`, built as `CSVPoolTask(..., maximize=False)`
in `main/src/tasks/csv_pool.py`. Each evaluation picks one untested mixture by
anonymous index. The agent sees only the feature matrix; CIDs, compositions and
labels stay private.

```bash
.venv/bin/python main/scripts/run_loop.py --task fluid_mixtures --budget 50 --check-only
.venv/bin/python main/scripts/run_loop.py --task fluid_mixtures --budget 50 --seed 0
.venv/bin/python main/scripts/run_sweep.py --task fluid_mixtures --budget 50 --seeds 10 --outdir runs/fluid-mixture-baselines
```
