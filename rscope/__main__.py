#!/usr/bin/env python3

"""Entry point for the rscope package when executed with python -m rscope."""

from absl import app
from absl import flags
from absl import logging

from rscope.main import main

# Configure absl flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('ssh', False, 'Enable SSH file watching and transfer')
flags.DEFINE_integer(
    'polling_interval', 10, 'Interval in seconds for SSH file polling'
)


def _main(argv):
  main(ssh_enabled=FLAGS.ssh, polling_interval=FLAGS.polling_interval)


if __name__ == '__main__':
  logging.set_verbosity(
      logging.INFO
  )  # Set to INFO to debug SSH and file watcher.
  app.run(_main)
