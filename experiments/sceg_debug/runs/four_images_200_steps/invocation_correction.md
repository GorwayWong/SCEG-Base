# Invocation metadata correction

The initial utility recorded sys.argv, which omitted Python's `-m` flag.
This does not affect model/config/checkpoint/results. The observed original
invocation from the executor was:

```bash
cd /root/projects/aris/sceg_fsfg
.venv/bin/python -m tools.overfit --run-id four_images_200_steps
```

The original command.txt is retained for audit. Future runs record sys.orig_argv
and cwd.txt. Replaying uses a fresh run ID because run directories reject overwrite.
