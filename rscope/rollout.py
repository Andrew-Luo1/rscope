"""Rollout utilities."""

from pathlib import Path
import pickle
from typing import Dict, List, NamedTuple, Union

from absl import logging
from numpy.typing import NDArray

MAX_VIEWPORTS = 12


class Rollout(NamedTuple):
  qpos: NDArray
  qvel: NDArray
  mocap_pos: NDArray
  mocap_quat: NDArray
  obs: NDArray
  reward: NDArray
  time: NDArray
  metrics: NDArray


# Global rollout state.
rollouts: List[Rollout] = []
num_evals = 0
num_envs = 0
env_ctrl_dt = 0.0
change_rollout = False


def get_num_evals():
  return num_evals


def append_unroll(fpath: Union[str, Path]):
  """Load an unroll file and append it to the list of rollouts."""
  logging.info(f'Loading unroll: {fpath}')
  global num_evals, num_envs, env_ctrl_dt, change_rollout
  with open(fpath, 'rb') as f:
    rollout = pickle.load(f)
  rollouts.append(rollout)

  assert rollout.qpos.shape[:2] == rollout.qvel.shape[:2], (
      'qpos and qvel non-matching time or envs dimension:'
      f' {rollout.qpos.shape} vs {rollout.qvel.shape}'
  )

  num_envs = rollout.qpos.shape[1]
  num_evals += 1
  # TODO: this is wrong when it resets on timestep 1.
  env_ctrl_dt = rollout.time[1, 0] - rollout.time[0, 0]
  if len(rollouts) == 1:
    change_rollout = True


def load_all_local_unrolls(base_path: Union[str, Path]) -> List[str]:
  """Load all existing unroll files from base_path into rollouts."""
  base = Path(base_path)
  unroll_files = [
      f.name for f in base.iterdir() if f.name.endswith('.mj_unroll')
  ]
  for fname in unroll_files:
    append_unroll(base / fname)
  return unroll_files


def dict_obs_pixels_env_select(obs: Dict, i_env: int) -> Dict:
  """
  Select the first MAX_VIEWPORTS keys from the observation dictionary
  that start with 'pixels/' (excluding ones with 'latent') and extract column i_env.
  """
  obs_pixels = {}
  num_shown = 0
  for key in obs.keys():
    if num_shown >= MAX_VIEWPORTS:
      break
    if key.startswith('pixels/') and 'latent' not in key:
      obs_pixels[key] = obs[key][:, i_env]
      num_shown += 1
  return obs_pixels


def dict_obs_t_select(obs: Dict, t: int) -> Dict:
  """
  Select the first MAX_VIEWPORTS keys from the observation dictionary
  that start with 'pixels/' (excluding ones with 'latent') and extract index t.
  """
  obs_t = {}
  num_shown = 0
  for key in obs.keys():
    if num_shown >= MAX_VIEWPORTS:
      break
    if key.startswith('pixels/') and 'latent' not in key:
      obs_t[key] = obs[key][t]
      num_shown += 1
  return obs_t


def metrics_env_select(metrics: Dict, i_env: int) -> Dict:
  """Select column i_env from each metric."""
  return {key: metrics[key][:, i_env] for key in metrics.keys()}
