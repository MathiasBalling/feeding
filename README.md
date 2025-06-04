# mj_sim_feeding

## Setup

This project uses uv for dependency management and virtual environments.
To get started, create a virtual environment and install dependencies:

```bash
uv venv
uv sync
soruce .venv/bin/activate
```

### Test installation

Try testing the installation by running:

```bash
python -m sims.feeder  # Linux
mjpython -m sims.feeder # MacOS
Python -m sims.feeder   # Windows
```

The expected behaviour is that the feeder track vibrates and the parts convey from one end to the other.
Press "space" to stop the simulation before quitting the window (pressing ctrl+c).
