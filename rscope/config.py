"""Rscope configuration."""

import os
from pathlib import Path

_root = Path(os.environ.get("RSCOPE_PATH", "/tmp/rscope"))
BASE_PATH = _root / "active_run"
TEMP_PATH = _root / "temp"
META_PATH = BASE_PATH / "rscope_meta.pkl"
# REMOTE_PATH: used by SSH mode for SFTP. When None, BASE_PATH is used.
REMOTE_PATH = None


def set_path(path: str):
  """Override rscope data directory at runtime (writer side / local mode)."""
  global BASE_PATH, TEMP_PATH, META_PATH
  root = Path(path)
  BASE_PATH = root / "active_run"
  TEMP_PATH = root / "temp"
  META_PATH = BASE_PATH / "rscope_meta.pkl"


def set_remote_path(path: str):
  """Set the remote path for SSH mode (viewer side)."""
  global REMOTE_PATH
  REMOTE_PATH = Path(path)


def get_remote_base_path():
  """Get the path to use for SFTP operations."""
  if REMOTE_PATH is not None:
    return REMOTE_PATH / "active_run"
  return BASE_PATH
