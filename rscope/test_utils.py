"""Shared utilities for rscope tests."""

import numpy as np

from rscope.rollout import Rollout


def create_fake_unroll(
    timesteps=10,
    num_envs=2,
    qpos_dim=7,
    qvel_dim=6,
    mocap_pos_dim=3,
    mocap_quat_dim=4,
):
  """Create a fake rollout with random data.

  Args:
    timesteps: Number of timesteps in the rollout.
    num_envs: Number of environments.
    qpos_dim: Dimension of qpos.
    qvel_dim: Dimension of qvel.
    mocap_pos_dim: Dimension of mocap_pos.
    mocap_quat_dim: Dimension of mocap_quat.

  Returns:
    A Rollout instance with random data.
  """
  # Create rollout data
  qpos = np.random.rand(timesteps, num_envs, qpos_dim)
  qvel = np.random.rand(timesteps, num_envs, qvel_dim)
  mocap_pos = np.random.rand(timesteps, num_envs, mocap_pos_dim)
  mocap_quat = np.random.rand(timesteps, num_envs, mocap_quat_dim)
  time = (
      np.linspace(0, 1, timesteps)
      .reshape(timesteps, 1)
      .repeat(num_envs, axis=1)
  )

  # Create observation and reward
  obs = {
      'state': np.random.rand(timesteps, num_envs, 8),
      'pixels/view_0': np.random.randint(
          0, 255, (timesteps, num_envs, 64, 64, 3), dtype=np.uint8
      ),
  }
  reward = np.random.rand(timesteps, num_envs)

  # Create metrics
  metrics = {
      'metric1': np.random.rand(timesteps, num_envs),
      'metric2': np.random.rand(timesteps, num_envs),
      'metric3': np.random.rand(timesteps, num_envs),
  }

  # Return a Rollout directly
  return Rollout(
      qpos=qpos,
      qvel=qvel,
      mocap_pos=mocap_pos,
      mocap_quat=mocap_quat,
      obs=obs,
      reward=reward,
      time=time,
      metrics=metrics,
  )
