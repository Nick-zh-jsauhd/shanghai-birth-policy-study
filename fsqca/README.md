# fsQCA

This module contains the fsQCA analysis layer that sits between the scenario-experiment pipeline and the final optimization stage.

## Main entry

- `main.py`
  Runs calibration, necessary-condition checks, truth-table construction, logical minimization, sensitivity analysis, recipe interpretation, and fsQCA-oriented plotting.

## Inputs and outputs

- Expected inputs: scenario-level analytical results produced by the earlier experimental pipeline.
- Expected outputs: fsQCA solution summaries, recipe mappings, and explanatory figures used in interpretation.

## Position in the workflow

`fsqca/` should be read after `scenario_experiments/` and before `optimization/`.

The repository keeps only the analysis code. Any local configuration files, generated tables, or exported figures remain untracked.
