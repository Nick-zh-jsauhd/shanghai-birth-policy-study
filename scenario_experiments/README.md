# Scenario Experiments

This module collects the code used to construct the scenario-experiment pipeline that feeds the later fsQCA and optimization analyses.

## Submodules

### `preprocessing/`

- Main script: `model3_stage0_data_processing.py`
- Role: cleans the original survey/scenario data, builds the Zhejiang analytical sample, and prepares grouped scenario inputs.
- Typical outputs: cleaned long-format data, balanced analytical samples, group weights, and scenario grids.

### `estimation/`

- Main script: `model3_stage1_fe_estimation.py`
- Role: estimates scenario effects using the Zhejiang subsample and fixed-effects specifications.
- Typical outputs: fitted coefficients, model fit summaries, scenario-level effect tables, and policy-level effect summaries.

### `cost_mapping/`

- Main script: `model3_stage2_cost_mapping.py`
- Role: converts scenario effects into monetary cost representations under the study's policy assumptions.
- Typical outputs: policy mapping tables, cost inputs, scenario-level cost tables, and policy cost summaries.

## Dependency chain

1. `preprocessing/` prepares the cleaned analytical inputs.
2. `estimation/` consumes cleaned scenario data and produces effect estimates.
3. `cost_mapping/` combines policy assumptions with estimated effects to prepare optimization-ready inputs.

## Data note

No CSV or JSON files are versioned in this repository.

To run these scripts locally, keep the required experimental input tables outside Git tracking and adapt local paths if necessary.
