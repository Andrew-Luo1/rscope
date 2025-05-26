"""Brax training rscope utils."""

import datetime
import os
import pathlib
from pathlib import PosixPath
import pickle
import shutil
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np

from rscope import config
from rscope import rollout


def clear_dir(path: pathlib.Path):
  for child in path.iterdir():
    if child.is_dir():
      shutil.rmtree(child)
    else:
      child.unlink()


def rscope_init(
    xml_path: Union[PosixPath, str],
    model_assets: Optional[Dict[str, Any]] = None,
):
  # clear the active run directory.
  if os.path.exists(config.BASE_PATH):
    clear_dir(config.BASE_PATH)
  else:
    os.makedirs(config.BASE_PATH)

  # save the xml into the assets for remote rscope usage.
  if model_assets is None:
    model_assets = {}
  model_assets[pathlib.Path(xml_path).name] = pathlib.Path(
      xml_path
  ).read_bytes()

  if not isinstance(xml_path, str):
    xml_path = xml_path.as_posix()

  rscope_meta = {"xml_path": xml_path, "model_assets": model_assets}
  # Make the base path and temp path if they don't exist.
  if not os.path.exists(config.BASE_PATH):
    os.makedirs(config.BASE_PATH)
  if not os.path.exists(config.TEMP_PATH):
    os.makedirs(config.TEMP_PATH)

  with open(os.path.join(config.BASE_PATH, "rscope_meta.pkl"), "wb") as f:
    pickle.dump(rscope_meta, f)


def dump_eval(trace: dict, obs: Union[jp.ndarray, dict], rew: jp.ndarray):
  # write to <datetime>.mj_unroll.
  now = datetime.datetime.now()
  now_str = now.strftime("%Y_%m_%d-%H_%M_%S")
  # ensure it's numpy.
  trace, obs, rew = jax.tree.map(lambda x: np.array(x), (trace, obs, rew))

  # save as dict rather than brax Transition.
  eval_rollout = rollout.Rollout(
      qpos=trace["qpos"],
      qvel=trace["qvel"],
      mocap_pos=trace["mocap_pos"],
      mocap_quat=trace["mocap_quat"],
      obs=obs,
      reward=rew,
      time=trace["time"],
      metrics=trace["metrics"],
  )

  # 2 stages to ensure atomicity.
  temp_path = os.path.join(config.TEMP_PATH, f"partial_transition.tmp")
  final_path = os.path.join(config.BASE_PATH, f"{now_str}.mj_unroll")
  with open(temp_path, "wb") as f:
    pickle.dump(eval_rollout, f)
  os.rename(temp_path, final_path)
