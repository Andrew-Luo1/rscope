"""Image processing utilities."""

import numpy as np


def resize_image(image, height, width):
  """Resize the leading two image dimensions."""
  # Resize using numpy slicing
  h_ratio = image.shape[0] / height
  w_ratio = image.shape[1] / width
  h_indices = (np.arange(height) * h_ratio).astype(int)
  w_indices = (np.arange(width) * w_ratio).astype(int)
  resized = image[h_indices][:, w_indices]
  return np.ascontiguousarray(resized)


def process_img(
    obs: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
  """
  Process an image observation:
    1. If of float type, clip to [0, 1] and scale to [0, 255] as uint8.
    2. If single-channel, expand to 3 channels.
    3. If not square, pad the image to make it square.
  """
  # Convert floats to uint8.
  if np.issubdtype(obs.dtype, np.floating):
    obs = np.clip(obs, 0, 1)
    obs = (obs * 255).astype(np.uint8)

  # Expand single-channel to 3 channels.
  if len(obs.shape) == 2 or (len(obs.shape) == 3 and obs.shape[2] == 1):
    if len(obs.shape) == 2:
      obs = obs[:, :, None]
    obs = np.repeat(obs, 3, axis=2)

  # Pad to square if necessary.
  height, width = obs.shape[0], obs.shape[1]
  if height != width:
    max_dim = max(height, width)
    padded = np.zeros((max_dim, max_dim, obs.shape[2]), dtype=obs.dtype)
    h_pad = (max_dim - height) // 2
    w_pad = (max_dim - width) // 2
    padded[h_pad : h_pad + height, w_pad : w_pad + width, :] = obs
    obs = padded

  return resize_image(obs, target_height, target_width)
