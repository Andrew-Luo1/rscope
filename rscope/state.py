"""Rscope state."""

import glfw

import rscope.rollout as rollout


class ViewerState:

  # fmt: off
  # Logarithmically spaced real-time slow-down coefficients (percent).
  SPEED_PERCENTAGES = [
      100, 80, 66, 50, 40, 33, 25, 20, 16, 13,
      10, 8, 6.6, 5.0, 4, 3.3, 2.5, 2, 1.6, 1.3,
      1, 0.8, 0.66, 0.5, 0.4, 0.33, 0.25, 0.2, 0.16, 0.13,
      0.1
  ]
  # fmt: on

  def __init__(self):
    self.cur_eval = 0
    self.cur_env = 0
    self.change_rollout = True
    self.pause = False
    self.show_metrics = False
    self.show_pixel_obs = False
    self.show_help = True
    self.transfer_status = None  # Current transfer message
    self.transfer_until = None  # Timestamp when message should disappear
    self.playback_speed = 1.0
    self.speed_index = 0

  def key_callback(self, keycode):
    if keycode == glfw.KEY_RIGHT:
      self.change_rollout = True
      self.cur_env += 1
    elif keycode == glfw.KEY_LEFT:
      self.change_rollout = True
      self.cur_env -= 1
    elif keycode == glfw.KEY_UP:
      self.change_rollout = True
      self.cur_eval += 1
    elif keycode == glfw.KEY_DOWN:
      self.change_rollout = True
      self.cur_eval -= 1
    else:
      try:
        char = chr(keycode)
        if char == "M":
          self.show_metrics = not self.show_metrics
        elif char == "O":
          self.show_pixel_obs = not self.show_pixel_obs
        elif char == " ":
          self.pause = not self.pause
        elif char == "H":
          self.show_help = not self.show_help
        elif char == "-":
          self.speed_index = min(
              len(self.SPEED_PERCENTAGES) - 1, self.speed_index + 1
          )
          self.playback_speed = self.SPEED_PERCENTAGES[self.speed_index] / 100.0
        elif char == "+" or char == "=":
          self.speed_index = max(0, self.speed_index - 1)
          self.playback_speed = self.SPEED_PERCENTAGES[self.speed_index] / 100.0
      except ValueError:
        pass

    # Wrap to valid ranges
    self.cur_eval = (self.cur_eval + rollout.num_evals) % rollout.num_evals
    self.cur_env = (self.cur_env + rollout.num_envs) % rollout.num_envs
