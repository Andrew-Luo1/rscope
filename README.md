# rscope
RL training visualizer for Mujoco Playground environments + Brax training

## Usage

### Local Mode
To visualize locally stored rollouts:
```bash
python -m rscope
```

### SSH Mode
To visualize rollouts stored on a remote server via SSH:
```bash
python -m rscope --ssh-to username@hostname[:port] --ssh-key ~/path/to/private_key
```

### Options
- `--ssh-to`: SSH connection string in the format `username@host[:port]`
- `--ssh-key`: Path to SSH private key file (required when using `--ssh-to`)
- `--polling-interval`: Interval in seconds for SSH file polling (default: 10)

## Notes
- When using SSH mode, the `--ssh-key` parameter is required
- The default SSH port (22) is used if not specified in the `--ssh-to` string
