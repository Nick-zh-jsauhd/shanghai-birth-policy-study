# FSE

`fse/` is the first stage of the research workflow and serves as the main experimental module for the full project.

## Main entry

- `main.py`
  Runs the core family-scenario experiment analysis and produces the first-stage analytical outputs used by later modules.

## Submodules

### `preprocessing/`

- Main script: `model3_stage0_data_processing.py`
- Role: cleans raw experimental inputs, prepares Zhejiang subsamples, and builds grouped scenario tables for downstream analysis.

### `estimation/`

- Main script: `model3_stage1_fe_estimation.py`
- Role: estimates scenario effects from the cleaned FSE data and produces effect tables for later modules.

### `cost_mapping/`

- Main script: `model3_stage2_cost_mapping.py`
- Role: maps first-stage experimental effects into monetary cost representations used by optimization.

## Workflow inside FSE

1. `main.py` provides the first-stage experimental core.
2. `preprocessing/` prepares cleaned analytical inputs.
3. `estimation/` converts cleaned inputs into scenario-level effects.
4. `cost_mapping/` converts those effects into optimization-ready cost tables.

## Data note

This repository does not version raw data, generated CSV files, figures, or exported result assets.

The default script paths now assume a repository-root layout with untracked outputs stored under `outputs/fse/`.
