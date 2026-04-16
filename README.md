# Shanghai Birth Policy Study

This repository is organized around the research workflow rather than the original internal stage numbering.

## Research workflow

1. `scenario_experiments/`
   Generates scenario-level inputs, fixed-effects estimates, and policy cost mappings.
2. `fsqca/`
   Runs the fsQCA analysis on scenario-based experimental outputs.
3. `optimization/`
   Performs policy optimization and visualization based on previous analytical results.

## Repository structure

```text
.
|-- scenario_experiments/
|   |-- preprocessing/
|   |-- estimation/
|   `-- cost_mapping/
|-- fsqca/
`-- optimization/
    |-- solver/
    `-- visualization/
```

## Code modules

### Scenario Experiments

- `preprocessing/model3_stage0_data_processing.py`
  Cleans and reshapes the scenario experiment data used by downstream modules.
- `estimation/model3_stage1_fe_estimation.py`
  Estimates scenario effects from the Zhejiang sample.
- `cost_mapping/model3_stage2_cost_mapping.py`
  Maps experimental policy effects to monetary costs under the project assumptions.

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

## Recommended reading order

1. Start from `scenario_experiments/README.md`
2. Continue to `fsqca/README.md`
3. Finish with `optimization/README.md`
