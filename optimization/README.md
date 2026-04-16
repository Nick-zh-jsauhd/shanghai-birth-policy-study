# Optimization

This module contains the final policy optimization stage of the study.

## Submodules

### `solver/`

- Main script: `model3_stage3_optimization.py`
- Role: combines upstream cost and effect inputs, builds policy frontiers, and solves uniform-versus-stratified allocation problems under multiple budget assumptions.

### `visualization/`

- Main scripts:
  - `model3_visualization_suite.py`
  - `model3_visualization_suite_v2.py`
  - `model3_visualization_suite_v3.py`
- Role: renders optimization outputs into interpretable presentation figures.

## Inputs and outputs

- Expected inputs: scenario-effect and cost outputs produced earlier in `fse/`.
- Expected outputs: optimization summaries, allocation comparisons, and figure-ready result artifacts.

## Workflow position

`optimization/` is the final analytical module and depends on outputs from `fse/`. In the project narrative it follows both the FSE stage and the fsQCA interpretation layer.
