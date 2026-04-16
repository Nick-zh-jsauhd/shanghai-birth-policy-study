# Repository Architecture

This repository is organized strictly by the three analytical stages of the project:

1. `fse/`
2. `fsqca/`
3. `optimization/`

The purpose of this document is to make file placement and future uploads consistent.

## Stage map

### Stage 1: `fse/`

Use `fse/` for all first-stage family-scenario experiment code.

Current tracked Python files:

- `fse/main.py`
  Core first-stage experimental analysis and first-stage figures.
- `fse/preprocessing/model3_stage0_data_processing.py`
  Raw-data cleaning, sample construction, and analytical preprocessing.
- `fse/estimation/model3_stage1_fe_estimation.py`
  First-stage fixed-effects estimation and policy-effect construction.
- `fse/cost_mapping/model3_stage2_cost_mapping.py`
  Cost conversion and optimization-facing cost scenario generation.

Put future code in `fse/` when it belongs to:

- the main first-stage experiment
- data preparation for the first-stage experiment
- first-stage estimation and robustness analysis
- cost-side transformation of first-stage outputs

### Stage 2: `fsqca/`

Use `fsqca/` for all second-stage configurational analysis code.

Current tracked Python files:

- `fsqca/main.py`
  Calibration, truth-table construction, minimization, sensitivity analysis, and fsQCA visualization.

Put future code in `fsqca/` when it belongs to:

- fuzzy-set calibration
- necessity/sufficiency analysis
- truth-table or minimization logic
- recipe interpretation or fsQCA figures

### Stage 3: `optimization/`

Use `optimization/` for all final-stage optimization and result-presentation code.

Current tracked Python files:

- `optimization/solver/model3_stage3_optimization.py`
  Policy frontier construction and budget-allocation optimization.
- `optimization/visualization/model3_visualization_suite.py`
  Baseline optimization result visualization suite.
- `optimization/visualization/model3_visualization_suite_v2.py`
  Second-generation visualization suite.
- `optimization/visualization/model3_visualization_suite_v3.py`
  Presentation-oriented visualization suite.

Put future code in `optimization/` when it belongs to:

- budget allocation
- policy optimization
- Pareto frontier construction
- optimization result packaging
- presentation figures derived from optimization outputs

## Placement rules

- If a script generates or analyzes first-stage experimental outputs, it belongs in `fse/`.
- If a script performs configurational inference from upstream analytical outputs, it belongs in `fsqca/`.
- If a script solves allocation problems or visualizes optimization outcomes, it belongs in `optimization/`.
- If a script is mainly for charts, place it under the stage-specific visualization folder rather than the stage root whenever possible.
- Keep old `stage0/1/2/3` naming inside file names only when preserving compatibility is useful. Do not reintroduce old top-level `stage*` directories.

## Output convention

Local generated artifacts should stay outside Git tracking and follow the repository-root output layout:

- `outputs/fse/...`
- `outputs/fsqca/...`
- `outputs/optimization/...`

The repository itself should remain code-only.
