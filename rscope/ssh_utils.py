import os
from queue import Empty
from queue import Queue
import re
import threading
import time

from absl import flags
from absl import logging
import paramiko

import rscope.config as config

# Get access to flags
FLAGS = flags.FLAGS


def ssh_connect(ssh):
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  # Parse ssh_to string (username@host[:port])
  ssh_to = FLAGS.ssh_to
  if "@" in ssh_to:
    username, host_port = ssh_to.split("@", 1)
  else:
    username = None
    host_port = ssh_to

  # Parse optional port
  if ":" in host_port:
    host, port_str = host_port.split(":", 1)
    port = int(port_str)
  else:
    host = host_port
    port = 22  # Default SSH port

  # Connect using parsed values with timeout
  try:
    ssh.connect(
        host,
        port=port,
        username=username,
        key_filename=os.path.expanduser(FLAGS.ssh_key),
        timeout=10,
    )
  except paramiko.SSHException as e:
    logging.error(f"SSH connection failed: {e}")
    exit(1)
  except Exception as e:
    logging.error(f"Unexpected error during SSH connection: {e}")
    exit(1)


class SSHFileWatcher(threading.Thread):

  def __init__(
      self,
      file_queue: Queue,
      known_files: set,
      stop_event: threading.Event,
      polling_interval=10,
  ):
    super().__init__(daemon=True)
    self.file_queue = file_queue
    self.known_files = known_files
    self.stop_event = stop_event
    self.polling_interval = polling_interval
    self.poll_event = threading.Event()

  def _extract_timestamp(self, filename):
    # Extract timestamp from filename like "2025_05_16-21_15_33.mj_unroll"
    match = re.search(r"-(\d+_\d+_\d+)\.", filename)
    if match:
      return match.group(1)
    return filename  # Fallback to original filename if pattern not found

  def run(self):
    ssh = paramiko.SSHClient()
    ssh_connect(ssh)
    sftp = ssh.open_sftp()
    try:
      while not self.stop_event.is_set():
        remote_files = sorted(sftp.listdir(str(config.BASE_PATH)))
        new_files = [
            f
            for f in remote_files
            if (f.endswith(".mj_unroll") and f not in self.known_files)
        ]

        # Sort new files by their embedded timestamps
        new_files.sort(key=self._extract_timestamp)

        for fname in new_files:
          self.file_queue.put(fname)
          self.known_files.add(fname)
          logging.info(f"Found new file: {fname}")

        # Wait for either polling interval or poll event
        self.poll_event.wait(self.polling_interval)
        self.poll_event.clear()
    finally:
      sftp.close()
      ssh.close()


class SSHFileTransfer(threading.Thread):

  def __init__(
      self, file_queue: Queue, stop_event: threading.Event, viewer_state=None
  ):
    super().__init__(daemon=True)
    self.file_queue = file_queue
    self.stop_event = stop_event
    self.viewer_state = viewer_state
    config.TEMP_PATH.mkdir(parents=True, exist_ok=True)

  def run(self):
    ssh = paramiko.SSHClient()
    ssh_connect(ssh)
    sftp = ssh.open_sftp()

    try:
      while not self.stop_event.is_set() or not self.file_queue.empty():
        try:
          fname = self.file_queue.get(timeout=1)
          logging.info(f"Transferring eval: {fname}")

          remote_path = str(config.BASE_PATH / fname)
          tmp_local_path = str(config.TEMP_PATH / f".tmp_{fname}")
          final_local_path = str(config.BASE_PATH / fname)

          # Update transfer status in viewer state
          if self.viewer_state:
            self.viewer_state.transfer_status = f"transferring {remote_path}..."
            self.viewer_state.transfer_until = (
                None  # Active transfer has no expiration
            )

          # Transfer to temporary file first
          sftp.get(remote_path, tmp_local_path)

          # Atomic rename after successful transfer
          os.rename(tmp_local_path, final_local_path)

          logging.info(f"Transfer complete: {fname}")

          # Update status to show completion and set expiration time
          if self.viewer_state:
            self.viewer_state.transfer_status = f"transferred {remote_path}"
            self.viewer_state.transfer_until = time.time() + 5  # Display time

          self.file_queue.task_done()
        except Empty:
          continue
        except Exception as e:
          logging.error(f"Error transferring {fname}: {e}")
          if os.path.exists(tmp_local_path):
            os.remove(tmp_local_path)

          # Show error message with timeout
          if self.viewer_state:
            self.viewer_state.transfer_status = (
                f"error transferring {fname}: {str(e)[:50]}"
            )
            self.viewer_state.transfer_until = time.time() + 10
    finally:
      sftp.close()
      ssh.close()
