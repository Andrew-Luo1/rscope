[![PyPI](https://img.shields.io/pypi/v/rscope.svg)](https://pypi.org/project/rscope/) [![Python Versions](https://img.shields.io/pypi/pyversions/rscope.svg)](https://pypi.org/project/rscope/) [![License: MIT](https://img.shields.io/pypi/l/rscope.svg)](https://opensource.org/licenses/MIT)

## Welcome to rscope!

A light-weight package to collect and interactively visualize trajectories while training Mujoco Playground environments. Rscope can visualize both local and remote (potentially headless) training runs.

![rscope_header](https://github.com/user-attachments/assets/225d0290-501d-4a2e-ace9-f2122786ffb6)

### Installation
> [!IMPORTANT]
> - Requires Python 3.10 or later.

`pip install rscope`

---

### Usage
> [!IMPORTANT]
> Mac users must run `mjpython` instead of python, ex. `mjpython -m rscope`

#### Local training runs
To visualize locally stored rollouts:

`python -m rscope`

#### Remote training runs
Below, **update user@remote_host**, for example alice@168.42.4.8.

First, set up password-free key-based SSH connection with the remote device:
```
ssh-keygen -t ed25519 -f ~/.ssh/rsync_key -N ""
ssh-copy-id -i ~/.ssh/rsync_key.pub user@remote_host
```
If this worked, you should be able to ssh in without using a password:
```
ssh -i ~/.ssh/rsync_key user@remote_host
echo hello
exit
```

To visualize rollouts stored on a remote server via SSH:

`python -m rscope --ssh_to user@remote_host[:port] --ssh_key ~/.ssh/rsync_key --polling_interval 5 # port defaults to 22`

---

### Features

1. Most features from [Mujoco viewer](https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate)
2. Browse through trajectories. Use left/right arrow keys to switch through parallel environments and up/down for recent/past trajectories.
3. Live Plotting. Use `SHIFT+M` to plot trajectory rewards and the contents of `state.metrics`, up to the first 11 keys.
4. Pixel Observations. Use `SHIFT+O` to overlay pixel observations if available. To use this feature, the observation must be a `dict` and the pixel keys must be prefixed with `pixels/`.

---
### Sharp bits

Some background on how rscope works: between policy updates, `rscope` unrolls multiple trajectories in parallel then visualizes them on CPU. While this is simpler to implement and less expensive than tracing *training* runs like in IsaacLab, this and other implementation details lead to some unexpected gotchas:
- Typically, stochastic policies are used for evaluating training progress while determinsitic ones are deployed. While you can use rscope on stochastic policies to get a feel for the agent's training exploration, we recommend [deterministic evals](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py#L232).
- Renders incorrectly for domain-randomized training because the loaded assets are from the nominal model definition.
- Plots only the first 14 keys in the metrics without filtering for shaping rewards.
- Visualizes only the first 14 pixel observations.
- Cannot capture curriculum progression during training, as curriculums depend on `state.info`, which is reset at the start of an evaluator run.
- Currently supports only PPO-based training.

---
#### Contribution Guidelines:

Please run the following before making a PR:
```
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```
