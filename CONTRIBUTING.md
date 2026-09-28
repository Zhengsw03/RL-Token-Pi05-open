# Contributing

Thanks for your interest in RL-Token-Pi05. Issues and pull requests are welcome, whether they
come from someone running the same cell or from someone reading the code.

## What is useful

- **Bug reports.** A failure that can be reproduced is the most valuable contribution: which
  script was run, how it was configured, what the log said and what was expected instead.
- **Documentation.** The READMEs and `CONFIG.md` should describe the code as it is; corrections
  and clearer wording are welcome.
- **Examples and reproductions.** A different task, another arm, a shorter command sequence or a
  variant of the pipeline.
- **Hardware notes.** Camera models, USB adapters, serial permissions and the pitfalls of a
  particular cell, since these are the parts that cost the most time to rediscover.

## Reporting an issue

Please include:

1. the exact command, with the name of the configuration file it used;
2. the part of the log around the failure, from the file, not from the terminal scrollback
   (`<output_dir>/train.log` keeps the full log of a run);
3. the configuration, with paths, serial ports and account names replaced by placeholders;
4. the environment: GPU, driver, CUDA, Python, and the LeRobot version recorded in
   `lerobot/pyproject.toml`;
5. for hardware problems, how the cameras and the two arms are connected, because device indices
   move when the USB layout changes.

## Pull requests

1. Fork the repository and work on a branch.
2. Keep the change focused: one subject per pull request.
3. Run `pytest` and keep the suite green. Tests that need the robot or the workspace LeRobot
   package skip themselves when those are unavailable, so a laptop can run the rest.
4. Keep the repository conventions: comments, docstrings and console output in English, no
   personal paths, accounts, credentials or absolute machine paths anywhere in the tree, and no
   internal documents.
5. Explain what changed, why, and how it was checked.

## Documentation changes

The two READMEs are kept in step, so a change to one should be mirrored in the other, in the
same style: no first person, no em dashes, and figures centred with a short caption.

## License

Contributions are released under the Apache License 2.0, the license of this repository.
