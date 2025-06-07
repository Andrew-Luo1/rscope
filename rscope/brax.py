from typing import Callable

import jax
from ml_collections import config_dict

from rscope import rscope_utils


def make_raw_rollout(state):
  ret = {
      'qpos': state.data.qpos,
      'qvel': state.data.qvel,
      'time': state.data.time,
      'metrics': state.metrics,
  }
  if hasattr(state.data, 'mocap_pos') and hasattr(state.data, 'mocap_quat'):
    ret['mocap_pos'] = state.data.mocap_pos
    ret['mocap_quat'] = state.data.mocap_quat
  return ret


class BraxRolloutSaver:

  def __init__(
      self,
      trace_env,
      ppo_params: config_dict.ConfigDict,
      vision: bool,
      rscope_envs: int,
      determistic: bool,
      key: jax.random.PRNGKey = jax.random.PRNGKey(0),
      callback_fn: Callable = None,
  ):
    self.trace_env = trace_env
    self.ppo_params = ppo_params
    self.vision = vision
    self.rscope_envs = rscope_envs
    self.determistic = determistic
    self.make_policy = None
    self.key = key
    if not hasattr(self.trace_env, '_model_assets'):
      raise NotImplementedError(
          '_model_assets not defined for this environment'
          'see cartpole.py for an example.'
      )
    rscope_utils.rscope_init(
        self.trace_env.xml_path, self.trace_env._model_assets
    )
    self.callback_fn = callback_fn

  def set_make_policy(self, new_make_policy):
    if not self.make_policy:
      self.make_policy = new_make_policy

  def _rollout(self, params):
    key_unroll, key_reset = jax.random.split(self.key)
    key_reset = jax.random.split(
        key_reset,
        self.ppo_params.num_envs if self.vision else self.rscope_envs,
    )
    # Assumed make_policy doesn't change.
    policy = self.make_policy(params, deterministic=self.determistic)
    state = self.trace_env.reset(key_reset)

    # collect rollout. Return raw_rolout, obs, rew.
    def step_fn(c, _):
      state, key = c
      key, key_act = jax.random.split(key)
      act, _ = policy(state.obs, key_act)
      state = self.trace_env.step(state, act)
      full_ret = (make_raw_rollout(state), state.obs, state.reward, state.done)
      return (state, key), jax.tree.map(
          lambda x: x[: self.rscope_envs], full_ret
      )

    _, (trace, obs, rew, done) = jax.lax.scan(
        step_fn,
        (state, key_unroll),
        None,
        length=self.ppo_params.episode_length // self.ppo_params.action_repeat,
    )
    return trace, obs, rew, done

  def dump_rollout(self, params):
    trace, obs, rew, done = jax.jit(self._rollout)(params)
    if self.callback_fn:
      self.callback_fn(trace, obs, rew, done)
    rscope_utils.dump_eval(trace, obs, rew)
