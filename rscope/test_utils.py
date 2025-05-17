"""Shared utilities for rscope tests."""

import numpy as np


class MockTransitions:
  """Mock class to mimic the transitions object expected by append_unroll."""

  def __init__(self, observation, reward, state_extras):
    self.observation = observation
    self.reward = reward
    self.extras = {'state_extras': {'rscope': state_extras}}


def create_fake_unroll(
    timesteps=10,
    num_envs=2,
    qpos_dim=7,
    qvel_dim=6,
    mocap_pos_dim=3,
    mocap_quat_dim=4,
):
  """Create a fake unroll file with random data."""
  # Create raw rollout data
  raw_rollout = {
      'qpos': np.random.rand(timesteps, num_envs, qpos_dim),
      'qvel': np.random.rand(timesteps, num_envs, qvel_dim),
      'mocap_pos': np.random.rand(timesteps, num_envs, mocap_pos_dim),
      'mocap_quat': np.random.rand(timesteps, num_envs, mocap_quat_dim),
      'time': (
          np.linspace(0, 1, timesteps)
          .reshape(timesteps, 1)
          .repeat(num_envs, axis=1)
      ),
      'metrics': {
          'metric1': np.random.rand(timesteps, num_envs),
          'metric2': np.random.rand(timesteps, num_envs),
          'metric3': np.random.rand(timesteps, num_envs),
      },
  }

  # Create observation and reward
  observation = {
      'state': np.random.rand(timesteps, num_envs, 8),
      'pixels/view_0': np.random.randint(
          0, 255, (timesteps, num_envs, 64, 64, 3), dtype=np.uint8
      ),
  }
  reward = np.random.rand(timesteps, num_envs)

  # Create mock transitions object
  return MockTransitions(observation, reward, raw_rollout)
