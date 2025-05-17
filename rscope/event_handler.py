"""Event handler for Rscope."""

from absl import logging
from watchdog.events import FileSystemEventHandler

from rscope.rollout import append_unroll


class MjUnrollHandler(FileSystemEventHandler):
  """Handles new .mj_unroll files appearing in the base directory."""

  def on_created(self, event):
    if not event.is_directory and event.src_path.endswith(".mj_unroll"):
      logging.info(f"Event: {event}")
      append_unroll(event.src_path)
