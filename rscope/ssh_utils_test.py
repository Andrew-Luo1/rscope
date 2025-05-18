import os
from pathlib import Path
from queue import Queue
import shutil
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest

import rscope.config as config
from rscope.ssh_utils import SSHFileTransfer
from rscope.ssh_utils import SSHFileWatcher


class MockSFTP:

  def __init__(self, remote_dir):
    self.remote_dir = remote_dir

  def listdir(self, path):
    return os.listdir(self.remote_dir)

  def get(self, remote_path, local_path):
    # Extract just the filename from the remote path
    filename = os.path.basename(remote_path)
    src_path = os.path.join(self.remote_dir, filename)
    shutil.copy(src_path, local_path)

  def close(self):
    pass


class MockSSHClient:

  def __init__(self, remote_dir):
    self.remote_dir = remote_dir

  def set_missing_host_key_policy(self, policy):
    pass

  def connect(self, hostname, port=None, username=None, key_filename=None):
    pass

  def open_sftp(self):
    return MockSFTP(self.remote_dir)

  def close(self):
    pass


class SSHUtilsTest(absltest.TestCase):

  def setUp(self):
    # Create temporary directories for test
    self.local_temp_dir = tempfile.mkdtemp()
    self.remote_temp_dir = tempfile.mkdtemp()

    # Store original config values
    self.orig_base_path = config.BASE_PATH
    self.orig_temp_path = config.TEMP_PATH

    # Set config to use temporary directories
    config.BASE_PATH = Path(self.local_temp_dir)
    config.TEMP_PATH = Path(self.local_temp_dir)

    # Set up the SSH client mock
    self.ssh_client_patcher = mock.patch(
        "paramiko.SSHClient", return_value=MockSSHClient(self.remote_temp_dir)
    )
    self.mock_ssh_client = self.ssh_client_patcher.start()

    # Mock the ssh_connect function to prevent it from being called
    self.ssh_connect_patcher = mock.patch("rscope.ssh_utils.ssh_connect")
    self.mock_ssh_connect = self.ssh_connect_patcher.start()

  def tearDown(self):
    # Stop the patchers
    self.ssh_client_patcher.stop()
    self.ssh_connect_patcher.stop()

    # Restore original config values
    config.BASE_PATH = self.orig_base_path
    config.TEMP_PATH = self.orig_temp_path

    # Clean up temp directories
    shutil.rmtree(self.local_temp_dir, ignore_errors=True)
    shutil.rmtree(self.remote_temp_dir, ignore_errors=True)

  def test_existing_files(self):
    """Test that existing files on remote are discovered and transferred."""
    # Create test files in remote directory
    test_files = []
    for i in range(3):  # Reduced from 5 to 3 files
      filename = f"test_{i}.mj_unroll"
      file_path = os.path.join(self.remote_temp_dir, filename)
      with open(file_path, "w") as f:
        f.write(f"Content for {filename}")
      test_files.append(filename)

    # Set up threads
    file_queue = Queue()
    known_files = set()
    stop_event = threading.Event()

    # Use shorter polling interval for testing
    watcher_thread = SSHFileWatcher(
        file_queue, known_files, stop_event, polling_interval=0.1
    )
    transfer_thread = SSHFileTransfer(file_queue, stop_event)

    # Start the threads
    watcher_thread.start()
    transfer_thread.start()

    # Give some time for the watcher to find files and transfer to process them
    time.sleep(0.5)  # Reduced from 3s to 0.5s

    # Stop the threads
    stop_event.set()
    watcher_thread.join(timeout=1)  # Reduced timeout from 5s to 1s
    transfer_thread.join(timeout=1)

    # Verify all files were discovered and transferred
    for filename in test_files:
      local_path = os.path.join(self.local_temp_dir, filename)
      self.assertTrue(
          os.path.exists(local_path), f"File {filename} was not transferred"
      )

      # Verify content matches
      with open(local_path, "r") as f:
        content = f.read()
      self.assertEqual(content, f"Content for {filename}")

    # Verify known_files set contains all test files
    self.assertEqual(known_files, set(test_files))

  def test_trickling_files(self):
    """Test that files added over time are discovered and transferred."""
    file_queue = Queue()
    known_files = set()
    stop_event = threading.Event()

    # Use shorter polling interval for testing
    watcher_thread = SSHFileWatcher(
        file_queue, known_files, stop_event, polling_interval=0.1
    )
    transfer_thread = SSHFileTransfer(file_queue, stop_event)

    # Start the threads
    watcher_thread.start()
    transfer_thread.start()

    # Create 3 files quickly
    test_files = []
    for i in range(3):  # Reduced from 5 to 3 files
      filename = f"trickle_{i}.mj_unroll"
      file_path = os.path.join(self.remote_temp_dir, filename)
      with open(file_path, "w") as f:
        f.write(f"Trickle content for {filename}")
      test_files.append(filename)

      # Wait a shorter time before adding the next file
      time.sleep(0.1)  # Reduced from 1s to 0.1s

    # Give some additional time for the last file to be processed
    time.sleep(0.5)  # Reduced from 3s to 0.5s

    # Stop the threads
    stop_event.set()
    watcher_thread.join(timeout=1)  # Reduced timeout from 5s to 1s
    transfer_thread.join(timeout=1)

    # Verify all files were discovered and transferred
    for filename in test_files:
      local_path = os.path.join(self.local_temp_dir, filename)
      self.assertTrue(
          os.path.exists(local_path), f"File {filename} was not transferred"
      )

      # Verify content matches
      with open(local_path, "r") as f:
        content = f.read()
      self.assertEqual(content, f"Trickle content for {filename}")

    # Verify known_files set contains all test files
    for filename in test_files:
      self.assertIn(filename, known_files)


if __name__ == "__main__":
  absltest.main()
