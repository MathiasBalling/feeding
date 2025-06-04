# Deep Reinforcement Learning in MuJoCo with Brax (MJX)

Some of the examples presented in this work can be found as MJX tutorial examples
[here](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=jDJLcI0Bv5lD) from DeepMind.

<center>

|   |   |
|---|---|
| ![here](docs/humanoid_running.gif)  | ![here](docs/panda_bring_to_target.gif)  |
| ![here](docs/barkour.gif)  | ![here](docs/barkour_continued.gif)  |

</center>

## Prerequisites

- For tracking progress and analytics we use [Weights and Biases](https://wandb.ai/site/) ([`wandb`](https://pypi.org/project/wandb/)). Be therefore sure that you have an account (it's free)
 - **Recommended**: it is recommended to run this on a device with acceleration hardware i.e. GPU or TPU.

## Quick Start

To start training you can run examples by
```bash
python -m learning.gym.<file-name> # e.g. python -m learning.gym.humanoid_running
```


## Spinning Up

It is recommended to construct the functions such as reward, done, etc. and simulation setup as a [sims](/sims/) simulation, since this enables easy monitoring and debugging. Once confirmed, learning is easier and the potential mistakes are fewer.

### Make your own environment 

Environments are kept in the [`envs/`](/learning/envs/) directory.

For easy of use, please copy any of the presented environments (**recommended** for simplicity: [`humanoid_running.py`](/learning/envs/humanoid_running.py)) and correct it for you own use.


### Train your agent

To train your agent copy any of the training scripts found in [`gym/`](./gym/) and make corrections for you particular use. It can here useful to take a look at some of the optimized algorithms implemented by the Brax team [here](https://github.com/google/brax/tree/main/brax/training/agents). 

### Apply your agent

In [demo/](/learning/demo/) is an example file on how to load in and apply a trained policy for humanoid running. Feel free to copy any and all code for you own particular use.

## Resources

To fully grasp how to perform reinforcement learning using Brax, here are some resources for laying the foundation.

 - [XLA](https://research.google/pubs/xla-compiling-machine-learning-for-peak-performance/)
 - [JAX](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf)
 - [Brax](https://arxiv.org/pdf/2106.13281)
 - [Flax](https://flax.readthedocs.io/en/latest/)
 - MuJoCo [software](https://mujoco.readthedocs.io/en/stable/overview.html) and [paper](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

### Additional resources

Weights and biases have hosted video calls with engineers from the Brax team, some of which might be of interest, such as [Brax Introdution](https://www.youtube.com/watch?v=5HRi5ALE8MQ&t=460s). The same goes for the [MuJoCo presented by Yuval Tassa](https://www.youtube.com/watch?v=2xVN-qY78P4&t=818s) presenting sampling based control methods in MuJoCo, along with [this post](https://github.com/google-deepmind/mujoco/discussions/1101), highlighting some of the newest and most exciting additions to MuJoCo.


# Common Issues

### Why are my Tyro parameters not being loaded in?
For tyro to properly overwrite hyper parameters, type hinting is necessary, thus
```python
num_time_steps = 10_000_000 # This is NOT a proper overwrite
```
```python
num_time_steps: int  = 10_000_000 # This IS a proper overwrite
```
