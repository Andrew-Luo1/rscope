"""Test for event handler functionality in rscope."""

from pathlib import Path
import pickle
import shutil
import tempfile
import threading
import time

from absl.testing import absltest
from watchdog.observers import Observer

from rscope import config
from rscope import event_handler
from rscope import rollout
from rscope.test_utils import create_fake_unroll


class UnrollGenerator(threading.Thread):
  """Thread that creates unroll files at regular intervals."""

  def __init__(self, target_dir, interval=0.5, num_unrolls=5):
    super().__init__()
    self.target_dir = Path(target_dir)
    self.interval = interval
    self.num_unrolls = num_unrolls
    self.stop_event = threading.Event()

  def run(self):
    count = 0
    while not self.stop_event.is_set() and count < self.num_unrolls:
      # Create a fake unroll
      fake_rollout = create_fake_unroll()

      # Save to a file
      filename = f"test_unroll_{count}.mj_unroll"
      filepath = self.target_dir / filename
      with open(filepath, "wb") as f:
        pickle.dump(fake_rollout, f)

      count += 1
      time.sleep(self.interval)


class EventHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Create a temporary directory for testing
    self.temp_dir = tempfile.mkdtemp()
    # Save the original BASE_PATH and temporarily change it
    self.original_base_path = config.BASE_PATH
    config.BASE_PATH = Path(self.temp_dir)
    # Reset global state before each test
    rollout.rollouts = []
    rollout.num_evals = 0
    rollout.num_envs = 0
    rollout.env_ctrl_dt = 0.0
    rollout.change_rollout = False

  def tearDown(self):
    # Restore the original BASE_PATH
    config.BASE_PATH = self.original_base_path
    # Clean up temp directory
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def test_watchdog_detection(self):
    """Test that the watchdog detects new unroll files and adds them to rollouts."""
    # Setup file system observer
    event_handler_instance = event_handler.MjUnrollHandler()
    observer = Observer()
    observer.schedule(event_handler_instance, self.temp_dir, recursive=False)
    observer.start()

    try:
      # Make sure rollouts is empty initially
      self.assertEqual(len(rollout.rollouts), 0)

      # Start a thread that will create unroll files
      num_unrolls = 3
      unroll_generator = UnrollGenerator(
          self.temp_dir, interval=0.5, num_unrolls=num_unrolls
      )
      unroll_generator.start()

      # Wait for the generator to finish plus a little extra time
      # for the watchdog to process all files
      max_wait_time = (num_unrolls + 1) * 0.5  # 1 extra interval
      start_time = time.time()

      # Wait until we have all unrolls or timeout
      while (
          len(rollout.rollouts) < num_unrolls
          and time.time() - start_time < max_wait_time
      ):
        time.sleep(0.5)

      # Verify that all unrolls were added
      self.assertEqual(len(rollout.rollouts), num_unrolls)
      self.assertEqual(rollout.num_evals, num_unrolls)

      # Check that the files were created in the directory
      unroll_files = list(Path(self.temp_dir).glob("*.mj_unroll"))
      self.assertEqual(len(unroll_files), num_unrolls)

    finally:
      # Clean up
      unroll_generator.stop_event.set()
      unroll_generator.join()
      observer.stop()
      observer.join()


if __name__ == "__main__":
  absltest.main()
