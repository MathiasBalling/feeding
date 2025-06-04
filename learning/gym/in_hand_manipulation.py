import functools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import tyro
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from orbax import checkpoint as ocp

import wandb
from learning.envs.in_hand_manipulation import InHandManipulation
from utils.args import Args
from utils.learning import create_data_directory


@dataclass(frozen=True)
class InHandManipulationArgs(Args):
    """
    Configuration arguments for training and environment setup.
    """

    env_name: str = Path(__file__).stem

    algo: Callable = ppo.train
    """Algorithm to use for training."""

    num_envs: int = 100
    batch_size: int = 100
    # batch_size: int = 512 // 2
    # num_envs: int = 1000
    # num_minibatches: int = 32


args = tyro.cli(InHandManipulationArgs, description=__doc__)

session = wandb.init(
    # set the wandb project where this run will be logged
    project=args.env_name,
    # track hyperparameters and run metadata
    config=args,
)

# save path init
session_path = create_data_directory(
    environment_name=args.env_name, session_name=session.name
)

# register enfironment
envs.register_environment(args.env_name, InHandManipulation)
env: InHandManipulation = envs.get_environment(args.env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


def checkpoint_callback(current_step, make_policy, params):
    """
    A user-defined callback function that can be used for saving
    policy checkpoints
    """
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = session_path / "checkpoints" / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


def evaluation_callback(num_steps, metrics: dict):
    """
    A user-defined callback function for reporting/plotting metrics
    """
    times.append(datetime.now())
    wandb.log(metrics)


# define training function
train_fn = functools.partial(
    args.algo,
    num_timesteps=args.num_timesteps,
    num_evals=args.num_evals,
    reward_scaling=args.reward_scaling,
    episode_length=args.episode_length,
    normalize_observations=args.normalize_observations,
    action_repeat=args.action_repeat,
    unroll_length=args.unroll_length,
    num_minibatches=args.num_minibatches,
    num_updates_per_batch=args.num_updates_per_batch,
    discounting=args.discounting,
    learning_rate=args.learning_rate,
    entropy_cost=args.entropy_cost,
    num_envs=args.num_envs,
    batch_size=args.batch_size,
    seed=args.seed,
    policy_params_fn=checkpoint_callback,
)

times = [datetime.now()]


# train the algorithm
make_inference_fn, params, metrics = train_fn(
    environment=env, progress_fn=evaluation_callback
)

# add the training time and JIT time to metrics
metrics["time to jit"] = times[1].second - times[0].second
metrics["time to train"] = times[-1].second - times[1].second

# upload all metrics along with times
wandb.log(metrics)

# save model
model.save_params(session_path / "model", params)

# # generate a short video demonstrating the capabilities of the policy
# demo_video_frames = env.get_demo_video(
#     make_inference_fn=make_inference_fn, params=params
# )

# # send demo video to wandb
# wandb.log({f"{session.name} demo": wandb.Video(demo_video_frames, fps=60)})
