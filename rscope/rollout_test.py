"""Tests for rollout utilities."""

import os
from pathlib import Path
import pickle
import tempfile

from absl.testing import absltest
import numpy as np

from rscope import rollout
from rscope.test_utils import create_fake_unroll


class RolloutTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global state before each test
    rollout.rollouts = []
    rollout.num_evals = 0
    rollout.num_envs = 0
    rollout.env_ctrl_dt = 0.0
    rollout.change_rollout = False

  def test_append_unroll(self):
    # Create a mock rollout
    timesteps = 10
    num_envs = 2

    # Create a fake unroll using the shared utility
    transitions = create_fake_unroll(timesteps=timesteps, num_envs=num_envs)

    # Save to a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    unroll_path = Path(temp_dir.name) / 'test.mj_unroll'

    with open(unroll_path, 'wb') as f:
      pickle.dump(transitions, f)

    # Call append_unroll
    initial_length = len(rollout.rollouts)
    rollout.append_unroll(unroll_path)

    # Test that rollout was appended
    self.assertEqual(len(rollout.rollouts), initial_length + 1)
    self.assertEqual(rollout.num_evals, 1)
    self.assertEqual(rollout.num_envs, num_envs)
    self.assertTrue(rollout.change_rollout)

    # Test that the appended rollout has the expected data
    appended_rollout = rollout.rollouts[0]
    self.assertIsNotNone(appended_rollout.qpos)
    self.assertIsNotNone(appended_rollout.qvel)
    self.assertIsNotNone(appended_rollout.mocap_pos)
    self.assertIsNotNone(appended_rollout.mocap_quat)
    self.assertIsNotNone(appended_rollout.obs)
    self.assertIsNotNone(appended_rollout.reward)
    self.assertIsNotNone(appended_rollout.time)
    self.assertIsNotNone(appended_rollout.metrics)

    # Clean up
    temp_dir.cleanup()

  def test_dict_obs_pixels_env_select(self):
    # Create a test observation
    obs = {
        'state': np.random.rand(10, 2, 8),
        'pixels/view_0': np.random.rand(10, 2, 64, 64, 3),
        'pixels/view_1': np.random.rand(10, 2, 64, 64, 3),
        'pixels/depth': np.random.rand(10, 2, 64, 64, 1),
    }

    # Call the function for env 0
    env_idx = 0
    obs_pixels = rollout.dict_obs_pixels_env_select(obs, env_idx)

    # Check that only pixel keys (excluding latent) are included
    self.assertIn('pixels/view_0', obs_pixels)
    self.assertIn('pixels/view_1', obs_pixels)
    self.assertIn('pixels/depth', obs_pixels)
    self.assertNotIn('state', obs_pixels)

    # Check that the env dimension is selected correctly
    self.assertEqual(obs_pixels['pixels/view_0'].shape, (10, 64, 64, 3))
    self.assertEqual(obs_pixels['pixels/view_1'].shape, (10, 64, 64, 3))
    self.assertEqual(obs_pixels['pixels/depth'].shape, (10, 64, 64, 1))

    # Call the function for env 1
    env_idx = 1
    obs_pixels = rollout.dict_obs_pixels_env_select(obs, env_idx)

    # Make sure we're selecting the right environment
    np.testing.assert_array_equal(
        obs_pixels['pixels/view_0'], obs['pixels/view_0'][:, 1]
    )
    np.testing.assert_array_equal(
        obs_pixels['pixels/view_1'], obs['pixels/view_1'][:, 1]
    )
    np.testing.assert_array_equal(
        obs_pixels['pixels/depth'], obs['pixels/depth'][:, 1]
    )

  def test_dict_obs_t_select(self):
    # Create a test observation dictionary
    obs = {
        'state': np.random.rand(10, 2, 8),
        'pixels/view_0': np.random.rand(10, 2, 64, 64, 3),
        'pixels/view_1': np.random.rand(10, 2, 64, 64, 3),
        'pixels/depth': np.random.rand(10, 2, 64, 64, 1),
    }

    # Call the function for timestep 5
    t_idx = 5
    obs_t = rollout.dict_obs_t_select(obs, t_idx)

    # Check that only pixel keys (excluding latent) are included
    self.assertIn('pixels/view_0', obs_t)
    self.assertIn('pixels/view_1', obs_t)
    self.assertIn('pixels/depth', obs_t)
    self.assertNotIn('state', obs_t)

    # Check that the time dimension is selected correctly
    self.assertEqual(obs_t['pixels/view_0'].shape, (2, 64, 64, 3))
    self.assertEqual(obs_t['pixels/view_1'].shape, (2, 64, 64, 3))
    self.assertEqual(obs_t['pixels/depth'].shape, (2, 64, 64, 1))

    # Make sure we're selecting the right timestep
    np.testing.assert_array_equal(
        obs_t['pixels/view_0'], obs['pixels/view_0'][5]
    )
    np.testing.assert_array_equal(
        obs_t['pixels/view_1'], obs['pixels/view_1'][5]
    )
    np.testing.assert_array_equal(obs_t['pixels/depth'], obs['pixels/depth'][5])

  def test_metrics_env_select(self):
    # Create a test metrics dictionary
    metrics = {
        'metric1': np.random.rand(10, 3),  # timesteps, num_envs
        'metric2': np.random.rand(10, 3),
        'metric3': np.random.rand(10, 3),
    }

    # Call the function for env 1
    env_idx = 1
    metrics_env = rollout.metrics_env_select(metrics, env_idx)

    # Check that all metrics are included
    self.assertIn('metric1', metrics_env)
    self.assertIn('metric2', metrics_env)
    self.assertIn('metric3', metrics_env)

    # Check that the env dimension is selected correctly
    self.assertEqual(metrics_env['metric1'].shape, (10,))
    self.assertEqual(metrics_env['metric2'].shape, (10,))
    self.assertEqual(metrics_env['metric3'].shape, (10,))

    # Make sure we're selecting the right environment
    np.testing.assert_array_equal(
        metrics_env['metric1'], metrics['metric1'][:, 1]
    )
    np.testing.assert_array_equal(
        metrics_env['metric2'], metrics['metric2'][:, 1]
    )
    np.testing.assert_array_equal(
        metrics_env['metric3'], metrics['metric3'][:, 1]
    )


if __name__ == '__main__':
  absltest.main()
