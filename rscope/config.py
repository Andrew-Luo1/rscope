"""Rscope configuration."""

import os
from pathlib import Path

_root = Path(os.environ.get("RSCOPE_PATH", "/tmp/rscope"))
BASE_PATH = _root / "active_run"
TEMP_PATH = _root / "temp"
META_PATH = BASE_PATH / "rscope_meta.pkl"


def set_path(path: str):
  """Override rscope data directory at runtime."""
  global BASE_PATH, TEMP_PATH, META_PATH
  root = Path(path)
  BASE_PATH = root / "active_run"
  TEMP_PATH = root / "temp"
  META_PATH = BASE_PATH / "rscope_meta.pkl"
