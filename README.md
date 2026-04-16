# Shanghai Birth Policy Study

This repository is organized around the three-stage research workflow.

## Research workflow

1. `fse/`
   Runs the first-stage family-scenario experiment and its downstream preprocessing, estimation, and cost-mapping steps.
2. `fsqca/`
   Runs the second-stage fsQCA analysis on outputs from `fse/`.
3. `optimization/`
   Performs the final-stage policy optimization and visualization based on upstream analytical results.

## Repository structure

```text
.
|-- ARCHITECTURE.md
|-- fse/
|   |-- main.py
|   |-- preprocessing/
|   |-- estimation/
|   `-- cost_mapping/
|-- fsqca/
`-- optimization/
    |-- solver/
    `-- visualization/
```

## Code modules

### FSE

- `main.py`
  Runs the first-stage experimental core of the project.
- `preprocessing/model3_stage0_data_processing.py`
  Cleans and reshapes FSE data used by downstream modules.
- `estimation/model3_stage1_fe_estimation.py`
  Estimates scenario effects from the Zhejiang sample.
- `cost_mapping/model3_stage2_cost_mapping.py`
  Maps experimental policy effects to monetary costs under the project assumptions.

FSE itself is internally organized as:

- `fse/main.py` for the core experiment layer
- `fse/preprocessing/` for data preparation
- `fse/estimation/` for effect estimation
- `fse/cost_mapping/` for optimization-facing cost conversion

### fsQCA

- `fsqca/main.py`
  Runs calibration, truth-table construction, minimization, and fsQCA figure generation.

### Optimization

- `solver/model3_stage3_optimization.py`
  Builds budget allocation and optimization outputs from upstream scenario results.
- `visualization/model3_visualization_suite*.py`
  Produces presentation-oriented visual summaries of optimization results.

## Data policy

This repository is code-only by design.

- Raw data, intermediate tables, exported figures, and generated outputs are not versioned.
- Local execution still depends on project-specific input files that remain outside Git tracking.
- The scripts preserve their original logic and file naming, so some local path assumptions may still need adaptation before running in a fresh environment.

## Where New Code Should Go

Use [`ARCHITECTURE.md`](ARCHITECTURE.md) as the placement guide for future uploads. It maps every tracked Python file to its analytical stage and explains where new scripts should be added.

## Recommended reading order

1. Start from `fse/README.md`
2. Continue to `fsqca/README.md`
3. Finish with `optimization/README.md`
4. Use `ARCHITECTURE.md` when extending the repository
