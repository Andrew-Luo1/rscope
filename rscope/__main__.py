#!/usr/bin/env python3

"""Entry point for the rscope package when executed with python -m rscope."""

from absl import app
from absl import flags
from absl import logging

from rscope.main import main

# Configure absl flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'ssh_to', None, 'SSH connection string in the format username@host[:port]'
)
flags.DEFINE_string('ssh_key', None, 'Path to SSH private key file')
flags.DEFINE_integer(
    'polling_interval', 10, 'Interval in seconds for SSH file polling'
)


def _main(argv):
  ssh_enabled = FLAGS.ssh_to is not None
  main(ssh_enabled=ssh_enabled, polling_interval=FLAGS.polling_interval)


if __name__ == '__main__':
  logging.set_verbosity(
      logging.WARNING
  )  # Set to INFO to debug SSH and file watcher, WARNING to mute.
  app.run(_main)
