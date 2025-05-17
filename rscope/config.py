"""Rscope configuration."""

from pathlib import Path

BASE_PATH = Path("/tmp/rscope/active_run")
TEMP_PATH = Path("/tmp/rscope/temp")
META_PATH = BASE_PATH / "rscope_meta.pkl"

from ml_collections import config_dict

CONFIG = config_dict.create(
    ssh_username="foo",  # ex: george
    ssh_host="bar",  # ex: 192.168.4.42
    ssh_port=22,
    ssh_key_path="~/.ssh/foobar",  # ex: ~/.ssh/rsync_key
)
