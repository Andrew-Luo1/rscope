"""Rscope main script."""

from queue import Queue
import threading
import time

from absl import app
from absl import flags
from absl import logging
import mujoco
import mujoco.viewer as mujoco_viewer
from watchdog.observers import Observer

import rscope.config as config
import rscope.event_handler as event_handler
import rscope.image_processing as image_processing
import rscope.model_loader as model_loader
import rscope.rollout as rollout
from rscope.ssh_utils import SSHFileTransfer
from rscope.ssh_utils import SSHFileWatcher
from rscope.state import ViewerState
import rscope.viewer_utils as vu


def main(ssh_enabled=False, polling_interval=10):

  # if BASE_PATH does not exist, make it
  if not config.BASE_PATH.exists():
    config.BASE_PATH.mkdir(parents=True, exist_ok=True)
  if not config.TEMP_PATH.exists():
    config.TEMP_PATH.mkdir(parents=True, exist_ok=True)

  # Create an instance of ViewerState to encapsulate state.
  viewer_state = ViewerState()

  if ssh_enabled:
    logging.info(
        f"SSH file watching enabled with polling interval: {polling_interval}s"
    )

    file_queue = Queue()
    known_files = set()
    stop_event = threading.Event()

    watcher_thread = SSHFileWatcher(
        file_queue, known_files, stop_event, polling_interval=polling_interval
    )
    transfer_thread = SSHFileTransfer(file_queue, stop_event, viewer_state)

    watcher_thread.start()
    transfer_thread.start()

    # Delete all existing files in the base path to prevent duplication.
    for file in config.BASE_PATH.glob("*"):
      try:
        file.unlink()
      except Exception as e:
        logging.error(f"Error deleting {file}: {e}")

  # Setup file system observer.
  event_handler_instance = event_handler.MjUnrollHandler()
  observer = Observer()
  observer.schedule(
      event_handler_instance, str(config.BASE_PATH), recursive=False
  )
  try:
    observer.start()
  except Exception as e:
    logging.error(f"Error starting observer: {e}")

  if not ssh_enabled:
    # Duplicates in the case of ssh_enabled.
    rollout.load_all_local_unrolls(config.BASE_PATH)

  # Wait for new rollouts to trickle in.
  print("Waiting for rollouts...")
  while not rollout.rollouts:
    time.sleep(3)
  print(
      f"Found {len(rollout.rollouts)} rollouts in {config.BASE_PATH}, "
      "starting viewer..."
  )

  # Initialize figures using metrics keys from the first rollout.
  metrics_keys = list(rollout.rollouts[0].metrics.keys())
  vu.reset_figures(metrics_keys)

  # Determine the initial replay length.
  replay_len = rollout.rollouts[0].qpos.shape[0]

  # Load the Mujoco model and data.
  mj_model, mj_data, meta = model_loader.load_model_and_data(ssh_enabled)

  with mujoco_viewer.launch_passive(
      mj_model,
      mj_data,
      show_left_ui=False,
      show_right_ui=False,
      key_callback=viewer_state.key_callback,
  ) as viewer:

    while viewer.is_running():
      step_start = time.time()

      # Trajectory selection: if a new rollout is requested.
      if viewer_state.change_rollout:
        vu.reset_figures(metrics_keys)
        viewer_state.change_rollout = False
        full_rollout = rollout.rollouts[viewer_state.cur_eval]
        if isinstance(full_rollout.obs, dict):
          obs = rollout.dict_obs_pixels_env_select(
              full_rollout.obs, viewer_state.cur_env
          )
        else:
          obs = full_rollout.obs[:, viewer_state.cur_env]
        cur_rollout = full_rollout._replace(
            qpos=full_rollout.qpos[:, viewer_state.cur_env],
            qvel=full_rollout.qvel[:, viewer_state.cur_env],
            mocap_pos=full_rollout.mocap_pos[:, viewer_state.cur_env],
            mocap_quat=full_rollout.mocap_quat[:, viewer_state.cur_env],
            obs=obs,
            reward=full_rollout.reward[:, viewer_state.cur_env],
            time=full_rollout.time[:, viewer_state.cur_env],
            metrics=rollout.metrics_env_select(
                full_rollout.metrics, viewer_state.cur_env
            ),
        )
        replay_index = 0
        replay_len = cur_rollout.qpos.shape[0]

      with viewer.lock():
        # Check if the transfer status message has expired
        if (
            ssh_enabled
            and viewer_state.transfer_status
            and viewer_state.transfer_until
        ):
          current_time = time.time()
          if current_time > viewer_state.transfer_until:
            # Clear the message if it's time
            viewer_state.transfer_status = None
            viewer_state.transfer_until = None

        # Overlay text.
        text_1 = "Eval\nEnv\nStep\nStatus\nSpeed"
        text_2 = (
            f"{viewer_state.cur_eval+1}/{len(rollout.rollouts)}\n"
            f"{viewer_state.cur_env+1}/{rollout.num_envs}\n"
            f"{replay_index}\n"
        )
        text_2 += "Pause" if viewer_state.pause else "Play"
        text_2 += f"\n{viewer_state.playback_speed * 100:.1f}%"
        overlays = [(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text_1,
            text_2,
        )]
        if viewer_state.show_help:
          menu_text_1, menu_text_2 = vu.get_menu_text()
          overlays.append((
              mujoco.mjtFontScale.mjFONTSCALE_150,
              mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
              menu_text_1,
              menu_text_2,
          ))

        # Add transfer status if available
        if ssh_enabled and viewer_state.transfer_status:
          overlays.append((
              mujoco.mjtFontScale.mjFONTSCALE_250,
              mujoco.mjtGridPos.mjGRID_BOTTOM,
              viewer_state.transfer_status,
              "",
          ))

        viewer.set_texts(overlays)

        # Render figures (metrics).
        if viewer_state.show_metrics:
          if not viewer_state.pause:
            cur_metrics = {
                key: metrics[replay_index]
                for key, metrics in cur_rollout.metrics.items()
            }
            for key in cur_metrics:
              vu.add_data_to_fig(key, cur_metrics[key])
          viewports = vu.get_viewports(
              len(cur_rollout.metrics), viewer.viewport
          )
          viewport_figures = list(zip(viewports, list(vu.figures.values())))
          viewer.set_figures(viewport_figures)
        else:
          viewer.clear_figures()

        # Render pixel observations if available.
        from collections.abc import Mapping

        if isinstance(cur_rollout.obs, Mapping):
          if any(key.startswith("pixels/") for key in cur_rollout.obs.keys()):
            if viewer_state.show_pixel_obs:
              cur_obs = rollout.dict_obs_t_select(cur_rollout.obs, replay_index)
              viewports = vu.get_viewports(len(cur_obs), viewer.viewport)
              processed_obs = {
                  key: image_processing.process_img(
                      cur_obs[key], viewport.height, viewport.width
                  )
                  for key, viewport in zip(cur_obs.keys(), viewports)
              }
              viewer.set_images(
                  list(zip(viewports, list(processed_obs.values())))
              )
            else:
              viewer.clear_images()

      # Advance simulation: update the state.
      def advance_rollout(mj_model, mj_data, idx):
        mj_data.qpos, mj_data.qvel = (
            cur_rollout.qpos[idx],
            cur_rollout.qvel[idx],
        )
        if cur_rollout.mocap_pos.size:
          mj_data.mocap_pos, mj_data.mocap_quat = (
              cur_rollout.mocap_pos[idx],
              cur_rollout.mocap_quat[idx],
          )
        mj_data.time = cur_rollout.time[idx]
        mujoco.mj_forward(mj_model, mj_data)

      advance_rollout(mj_model, mj_data, replay_index)
      if not viewer_state.pause:
        replay_index = (replay_index + 1) % replay_len
        viewer.sync()

      time_until_next_step = float(
          rollout.env_ctrl_dt
      ) / viewer_state.playback_speed - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)

  # Modify cleanup section
  if ssh_enabled:
    stop_event.set()
    watcher_thread.join()
    transfer_thread.join()

  observer.stop()
  observer.join()
