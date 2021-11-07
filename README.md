Code for offline RL dataset experiments based on [D4RL](https://github.com/rail-berkeley/d4rl), [d4rl_evaluations](https://github.com/rail-berkeley/d4rl_evaluations), and [CQL](https://github.com/aviralkumar2907/CQL).

# Setup
Need to have MuJoCo 2.0 downloaded and `MUJOCO_PY_MUJOCO_PATH`, `MUJOCO_PY_MJKEY_PATH`, and `LD_LIBRARY_PATH` environment variables set (this was before the open-sourcing by DeepMind).

Easiest to use [conda](https://docs.conda.io/en/latest/) to create the environment:
`conda install --name off --file spec-file.txt`

`conda activate off`

The singularity `.def` file is also provided for containerization.

# Running experiments
Only works for Maze2D environments and selected algorithms (CQL, BEAR, …) so far.

Use `d4rl/scripts/generation/generate_maze2d_starts_and_targets.py` to generate a starts and targets file, which can then be used with `d4rl/scripts/generation/generate_maze2d_datasets.py` to generate the actual datasets for offline learning. Both of these can be passed to e.g. `CQL/d4rl/examples/cql_mujoco_new.py` to train on the given dataset (defined by the buffer file) with the given evaluation method (defined by the starts and targets file).

