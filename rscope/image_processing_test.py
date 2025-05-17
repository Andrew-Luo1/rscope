"""Tests for image processing module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from rscope import image_processing


class ImageProcessingTest(parameterized.TestCase):

  def test_resize_image(self):
    # Test with a simple 2D grayscale image
    image = np.ones((100, 200), dtype=np.uint8) * 128
    resized = image_processing.resize_image(image, 25, 50)
    self.assertEqual(resized.shape, (25, 50))
    self.assertEqual(resized.dtype, np.uint8)
    # Check that values are preserved
    self.assertTrue(np.all(resized == 128))
    # Check memory layout is C-contiguous
    self.assertTrue(resized.flags.c_contiguous)

    # Test with a 3D RGB image
    image = np.ones((100, 200, 3), dtype=np.uint8) * 128
    resized = image_processing.resize_image(image, 25, 50)
    self.assertEqual(resized.shape, (25, 50, 3))
    self.assertEqual(resized.dtype, np.uint8)
    self.assertTrue(np.all(resized == 128))
    self.assertTrue(resized.flags.c_contiguous)

  @parameterized.parameters(
      # Test with float images (grayscale and RGB)
      {'input_shape': (64, 64), 'input_dtype': np.float32, 'channels': None},
      {'input_shape': (64, 64), 'input_dtype': np.float64, 'channels': None},
      {'input_shape': (64, 64, 3), 'input_dtype': np.float32, 'channels': 3},
      # Test with uint8 images (grayscale and RGB)
      {'input_shape': (64, 64), 'input_dtype': np.uint8, 'channels': None},
      {'input_shape': (64, 64, 3), 'input_dtype': np.uint8, 'channels': 3},
      # Test with single-channel images
      {'input_shape': (64, 64, 1), 'input_dtype': np.uint8, 'channels': 1},
      {'input_shape': (64, 64, 1), 'input_dtype': np.float32, 'channels': 1},
      # Test with non-square images
      {'input_shape': (32, 64), 'input_dtype': np.uint8, 'channels': None},
      {'input_shape': (64, 32), 'input_dtype': np.uint8, 'channels': None},
      {'input_shape': (32, 64, 3), 'input_dtype': np.uint8, 'channels': 3},
      {'input_shape': (64, 32, 3), 'input_dtype': np.uint8, 'channels': 3},
  )
  def test_process_img(self, input_shape, input_dtype, channels):
    """Test process_img with various input shapes and types."""
    # Create test image
    if channels is None:
      # Grayscale image without channel dimension
      image = np.ones(input_shape, dtype=input_dtype)
      if np.issubdtype(input_dtype, np.floating):
        image *= 0.5  # Set to 0.5 for float images
      else:
        image *= 128  # Set to 128 for uint8 images
    else:
      # Image with explicit channel dimension
      image = np.ones(input_shape, dtype=input_dtype)
      if np.issubdtype(input_dtype, np.floating):
        image *= 0.5  # Set to 0.5 for float images
      else:
        image *= 128  # Set to 128 for uint8 images

    # Process the image
    target_height, target_width = 32, 32
    processed = image_processing.process_img(image, target_height, target_width)

    # Check the processed image has the correct shape and type
    self.assertEqual(processed.shape, (target_height, target_width, 3))
    self.assertEqual(processed.dtype, np.uint8)

    # For non-square images, there will be padding, so we can't check all pixels
    # Instead, we check that the image was processed correctly and has values in expected ranges

    # For square images, we can verify more exactly
    is_square = (
        (input_shape[0] == input_shape[1])
        if len(input_shape) == 2
        else (input_shape[0] == input_shape[1])
    )

    if is_square:
      if np.issubdtype(input_dtype, np.floating):
        # For floating point, check using a more tolerant statistical approach
        # 0.5 float should convert to values close to 127-128

        # With float conversions and resizing, we should expect values to be close
        # to 127-128, but allow for some variation due to interpolation and rounding

        # Calculate the histogram of values
        values, counts = np.unique(processed, return_counts=True)

        # Find the most common value
        most_common_value = values[np.argmax(counts)]

        # Check that the most common value is close to 127-128
        self.assertTrue(
            abs(most_common_value - 127.5) <= 2,
            f'Most common value {most_common_value} not close to 127-128',
        )

        # Check that at least 90% of the pixels are in the expected range
        expected_range_mask = (processed >= 126) & (processed <= 129)
        percentage_in_range = (
            np.count_nonzero(expected_range_mask) / processed.size
        )
        self.assertGreaterEqual(
            percentage_in_range,
            0.9,
            f'Only {percentage_in_range:.2%} of pixels in expected range'
            ' 126-129',
        )
      else:
        # For uint8, values should be exactly 128
        self.assertTrue(np.all(processed == 128))
    else:
      # For non-square images, we just need to check that:
      # 1. There are zero values (padding)
      # 2. There are non-zero values (original data)
      # 3. For non-zero values, they should be close to our expected value

      has_zeros = np.any(processed == 0)
      has_non_zeros = np.any(processed > 0)

      self.assertTrue(has_zeros, 'Padded image should have some zero values')
      self.assertTrue(
          has_non_zeros, 'Processed image should have non-zero values'
      )

      # Extract non-zero values and check them
      non_zero_mask = processed > 0
      if np.issubdtype(input_dtype, np.floating):
        # For floating point values, use the same statistical approach
        if np.any(non_zero_mask):
          non_zero_values = processed[non_zero_mask]

          # Calculate the histogram of non-zero values
          values, counts = np.unique(non_zero_values, return_counts=True)

          # Find the most common value
          most_common_value = values[np.argmax(counts)]

          # Check that the most common value is close to 127-128
          self.assertTrue(
              abs(most_common_value - 127.5) <= 2,
              f'Most common value {most_common_value} not close to 127-128',
          )

          # Check that at least 90% of the non-zero pixels are in the expected range
          expected_range_mask = (non_zero_values >= 126) & (
              non_zero_values <= 129
          )
          percentage_in_range = (
              np.count_nonzero(expected_range_mask) / non_zero_values.size
          )
          self.assertGreaterEqual(
              percentage_in_range,
              0.9,
              f'Only {percentage_in_range:.2%} of pixels in expected range'
              ' 126-129',
          )
      else:
        # For uint8, non-zero values should be exactly 128
        if np.any(non_zero_mask):
          non_zero_values = processed[non_zero_mask]
          self.assertTrue(np.all(non_zero_values == 128))

  def test_process_img_float_clipping(self):
    """Test that float values are properly clipped to [0, 1] range."""
    # Create test image with values outside [0, 1] range
    image = np.ones((64, 64), dtype=np.float32) * 2.0  # Above 1.0
    negative = np.ones((64, 64), dtype=np.float32) * -0.5  # Below 0.0

    # Process the images
    processed_high = image_processing.process_img(image, 32, 32)
    processed_low = image_processing.process_img(negative, 32, 32)

    # Values above 1.0 should be clipped to 255
    self.assertTrue(np.all(processed_high == 255))
    # Values below 0.0 should be clipped to 0
    self.assertTrue(np.all(processed_low == 0))

  def test_process_img_square_padding(self):
    """Test that non-square images are properly padded."""
    # Test with a wide image
    wide_image = np.ones((32, 64, 3), dtype=np.uint8) * 255
    processed_wide = image_processing.process_img(wide_image, 32, 32)
    self.assertEqual(processed_wide.shape, (32, 32, 3))

    # Test with a tall image
    tall_image = np.ones((64, 32, 3), dtype=np.uint8) * 255
    processed_tall = image_processing.process_img(tall_image, 32, 32)
    self.assertEqual(processed_tall.shape, (32, 32, 3))

    # Create images with a visible pattern to test padding placement
    # Wide image with left side white, right side black
    pattern_wide = np.zeros((32, 64, 3), dtype=np.uint8)
    pattern_wide[:, :32, :] = 255
    processed_pattern_wide = image_processing.process_img(pattern_wide, 32, 32)

    # The processed image should have padding on the sides, with the pattern centered
    # Check if top-left pixel is 0 (should be in the padding region)
    self.assertEqual(processed_pattern_wide[0, 0, 0], 0)

    # Tall image with top half white, bottom half black
    pattern_tall = np.zeros((64, 32, 3), dtype=np.uint8)
    pattern_tall[:32, :, :] = 255
    processed_pattern_tall = image_processing.process_img(pattern_tall, 32, 32)

    # The processed image should have padding on the top and bottom, with the pattern centered
    # Check if top-left pixel is 0 (should be in the padding region)
    self.assertEqual(processed_pattern_tall[0, 0, 0], 0)


if __name__ == '__main__':
  absltest.main()
