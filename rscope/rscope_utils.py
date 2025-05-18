"""Brax training rscope utils."""

import datetime
import os
import pathlib
from pathlib import PosixPath
import pickle
import shutil
from typing import Any, Dict, Optional, Union

import jax
import numpy as np

from rscope import config
from rscope import rollout


def rscope_init(
    xml_path: Union[PosixPath, str],
    model_assets: Optional[Dict[str, Any]] = None,
):
  # clear the active run directory.
  if os.path.exists(config.BASE_PATH):
    shutil.rmtree(config.BASE_PATH)
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


def dump_eval(eval: dict):
  # write to <datetime>.mj_unroll.
  now = datetime.datetime.now()
  now_str = now.strftime("%Y_%m_%d-%H_%M_%S")
  # ensure it's numpy.
  eval = jax.tree.map(lambda x: np.array(x), eval)

  # save as dict rather than brax Transition.
  raw_rollout = eval.extras["state_extras"]["rscope"]
  eval_rollout = rollout.Rollout(
      qpos=raw_rollout["qpos"],
      qvel=raw_rollout["qvel"],
      mocap_pos=raw_rollout["mocap_pos"],
      mocap_quat=raw_rollout["mocap_quat"],
      obs=eval.observation,
      reward=eval.reward,
      time=raw_rollout["time"],
      metrics=raw_rollout["metrics"],
  )

  # 2 stages to ensure atomicity.
  temp_path = os.path.join(config.TEMP_PATH, f"partial_transition.tmp")
  final_path = os.path.join(config.BASE_PATH, f"{now_str}.mj_unroll")
  with open(temp_path, "wb") as f:
    pickle.dump(eval_rollout, f)
  os.rename(temp_path, final_path)
