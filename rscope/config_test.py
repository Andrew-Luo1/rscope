"""Tests for config values."""

from absl.testing import absltest

from rscope.config import CONFIG


class ConfigTest(absltest.TestCase):
  """Test to ensure default config values are maintained."""

  def test_default_config_values(self):
    """Verify CONFIG has expected default values.

    This test ensures that developers don't accidentally commit their
    personal configuration details.
    """
    # Test ssh_username value
    self.assertEqual(
        CONFIG.ssh_username,
        "foo",
        "Default ssh_username should be 'foo', not a real username",
    )

    # Test ssh_host value
    self.assertEqual(
        CONFIG.ssh_host,
        "bar",
        "Default ssh_host should be 'bar', not a real hostname/IP",
    )

    # Test ssh_port value
    self.assertEqual(CONFIG.ssh_port, 22, "Default ssh_port should be 22")

    # Test ssh_key_path value
    self.assertEqual(
        CONFIG.ssh_key_path,
        "~/.ssh/foobar",
        "Default ssh_key_path should be '~/.ssh/foobar', not a real SSH key"
        " path",
    )


if __name__ == "__main__":
  absltest.main()
