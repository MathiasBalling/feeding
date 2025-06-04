# mj_sim_feeding

This is a fork of Victor Staven's repository (https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim) which I've added components to for this lecture. All credit should go to him.
His is a repository to demonstrate, implement, test and learn control in MuJoCo. Feel free to explore his other simulations if you are interested in simulation with the MuJoCo physics engine

Below is a guide to get started.  I've tested it works on Ubuntu 24.04 and Windows 11, but was not able to do so for MacOS. This does not mean you cannot get it to work, but I had issues with my homebrew finding all the necessary dependencies. 

## Table of Contents

- [mj\_sim](#mj_sim)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Getting Repository](#getting-repository)
    - [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
    - [Activating a Virtual Environmentv](#activating-a-virtual-environmentv)
  - [Installing Dependencies](#installing-dependencies)
- [Real Time](#real-time)
- [Learning](#learning)
- [Docs](#docs)
- [VSCODE Compatibility](#vscode-compatibility)
- [License](#license)
- [Cite this work](#cite-this-work)

## Setup

### Getting Repository
To get the repository clone it into your directory of choice and checkout the teaching branch

```bash
git clone git@gitlab.sdu.dk:simat/mj-sim-feeding.git 	# or git clone https://gitlab.sdu.dk/simat/mj-sim-feeding.git
cd mj-sim-feeding
git checkout teaching
```
then go into the repository by
```bash
cd mj_sim_feeding
```

### Setting up a Virtual Environment

To isolate the project's dependencies, it's recommended to use a virtual environment. If you haven't already **installed Python's venv module**, you can do so by running:

```bash
sudo apt install python3.12-venv   # For Linux (Debian/Ubuntu)
```


```bash
pip install virtualenv            # For Windows, assuming pip is installed (guide: https://phoenixnap.com/kb/install-pip-windows)
```

```bash
brew install virtualenv           # For MacOS
```

Once installed, you can **create a virtual environment** by executing the following commands:

```bash
python3.12 -m venv venv            # Linux (Ubuntu) and MacOS
```
```bash
virtualenv --python C:\Path\To\Python\python.exe venv            # Windows
```

### Activating a Virtual Environmentv
```bash
source venv/bin/activate          # Activate the virtual environment (Linux/Macos)
```
```bash
venv\Scripts\activate             # Activate the virtual environment (Windows)
```

## Installing Dependencies

After activating the virtual environment, you can install the project dependencies using pip. `mj_sim` uses [poetry](https://python-poetry.org/docs/) for managing dependencies and can thus be installed using
```bash
pip install poetry
```
Once poetry is installed in your virtual environment, install the project dependencies using 
```bash
poetry install
```
in the root of the project. Poetry will then install the project's base dependencies.

> [!WARNING]
> In case your poetry installation seems to run forever, kill the process and run the following 
> ```bash
> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
> ```
> and try again.

you can then test it on one of the demo simulation. Either a robot simulation:
```bash
python -m sims.opspace_ur5e  # Linux
mjpython -m sims.opspace_ur5e # MacOS
Python -m sims.opspace_ur5e   # Windows
```

A demo of the robot sim can be seen below for comparison

![here](/public/docs/real_time_sim_demo.gif)


You should also try the feeder simulation we will work on in the lectures:
```bash
python -m sims.feeder  # Linux
mjpython -m sims.feeder # MacOS
Python -m sims.feeder   # Windows
```
The expected behaviour is that the feeder track vibrates and the parts convey from one end to the other. 
For windows. Press "space" to stop the simulation before quitting the window (pressing ctrl+c).



# Docs

A static website is generatd using [`pdoc`](https://pdoc.dev/) and can be found in `public/`. Access the documentation through 
```bash
<your-webbrowser-of-choise> public/index.html # e.g. firefox public/index.html
```
or online through the link found [here](https://sdurobotics.pages.sdu.dk/teaching/mj_sim/).

In case of generating pdoc documentation fails, remember to enable pdoc subprocess execution i.e.

```bash
export PDOC_ALLOW_EXEC=1
```

# VSCODE Compatibility

In case you are using [vscode](https://code.visualstudio.com/download) native type hinting is not a default for the MuJoCo or `ur_rtde` bindings (at the writing of this README.md). To add type hinting look here [here](https://github.com/google-deepmind/mujoco/issues/1292) (`pybind11-stubgen` is installed with the [`pyproject.toml`](pyproject.toml) when you install the dependencies through poetry) i.e. 
For MuJoCo:
```bash
pybind11-stubgen mujoco -o ./typings/
```
For `ur_thde`:
```bash
pybind11-stubgen rtde_control -o ./typings/
```
```bash
pybind11-stubgen rtde_receive -o ./typings/
```
```bash
pybind11-stubgen rtde_io -o ./typings/
```
For more documentation see [here](https://github.com/sizmailov/pybind11-stubgen).

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# To cite Victor's work:
To cite this work using `bibtex` please use the following
```bibtex
@misc{staven2024mjsim,
  author       = {Staven, Victor M},
  title        = {mj\_sim},
  version      = {3.0.0},
  year         = {2024},
  howpublished = {\url{https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim}},
  note         = {A repository to demonstrate, implement, test and learn control in MuJoCo.},
}
```
