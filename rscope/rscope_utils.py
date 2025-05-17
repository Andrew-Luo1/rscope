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

from rscope.config import BASE_PATH
from rscope.config import TEMP_PATH


def rscope_init(
    xml_path: Union[PosixPath, str],
    model_assets: Optional[Dict[str, Any]] = None,
):
  # clear the active run directory.
  if os.path.exists(BASE_PATH):
    shutil.rmtree(BASE_PATH)
  os.makedirs(BASE_PATH)

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
  if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
  if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

  with open(os.path.join(BASE_PATH, "rscope_meta.pkl"), "wb") as f:
    pickle.dump(rscope_meta, f)


def dump_eval(eval: dict):
  # write to <datetime>.mj_unroll.
  now = datetime.datetime.now()
  now_str = now.strftime("%Y_%m_%d-%H_%M_%S")
  # ensure it's numpy.
  eval = jax.tree.map(lambda x: np.array(x), eval)
  # 2 stages to ensure atomicity.
  temp_path = os.path.join(TEMP_PATH, f"partial_transition.tmp")
  final_path = os.path.join(BASE_PATH, f"{now_str}.mj_unroll")
  with open(temp_path, "wb") as f:
    pickle.dump(eval, f)
  os.rename(temp_path, final_path)
