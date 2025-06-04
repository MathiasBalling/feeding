import functools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import tyro
from brax import envs
from brax.training.agents.ppo import train as ppo

import wandb
from learning.envs.panda_cable_weaving_primitive import PandaCableWeavingPrimitive
from utils.args import Args
from utils.helpers import timer


@dataclass(frozen=True)
class PandaCableWeavingPrimitiveArgs(Args):
    """
    Configuration arguments for training and environment setup.
    """

    algo: Callable = ppo.train
    """Algorithm to use for training."""

    env_name: str = Path(__file__).stem
    """Name of the environment to train on."""


args = tyro.cli(PandaCableWeavingPrimitiveArgs, description=__doc__)

# session = wandb.init(
#     # set the wandb project where this run will be logged
#     project=args.env_name,
#     # track hyperparameters and run metadata
#     config=args,
# )

# # save path init
# session_path = create_data_directory(
#     environment_name=args.env_name, session_name=session.name
# )

with timer("create env..."):
    # register enfironment
    envs.register_environment(args.env_name, PandaCableWeavingPrimitive)
    env = envs.get_environment(args.env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# define training function
with timer("define training func..."):
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
    )

times = [datetime.now()]


# progress logger
def progress(num_steps, metrics: dict):
    times.append(datetime.now())
    wandb.log(metrics)


# train the algorithm
with timer("training..."):
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

# add the training time and JIT time to metrics
metrics["time to jit"] = times[1].second - times[0].second
metrics["time to train"] = times[-1].second - times[1].second

# upload all metrics along with times
wandb.log(metrics)

# save model
# model.save_params(session_path / "model", params)

# # generate a short video demonstrating the capabilities of the policy
# demo_video_frames = generate_demo_video(
#     args=args,
#     make_inference_fn=make_inference_fn,
#     params=params,
#     session_path=session_path,
# )

# # send demo video to wandb
# wandb.log({f"{session.name} demo": wandb.Video(demo_video_frames, fps=60)})
