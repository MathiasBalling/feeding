import time
from pathlib import Path

import jax
import mujoco as mj
import mujoco.viewer
from brax import envs
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from learning.envs.humanoid_running import HumanoidRunning
from utils.learning import get_dimensions_from_params

_LEARNING = Path(__file__).parent.parent

_MODEL = _LEARNING / "docs" / "humanoid_running"

params = model.load_params(_MODEL)

o_dim, a_dim = get_dimensions_from_params(params)

ppo_network = ppo_networks.make_ppo_networks(
    observation_size=o_dim,
    action_size=a_dim,
    preprocess_observations_fn=running_statistics.normalize,
)

make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

# register enfironment
envs.register_environment("humanoid", HumanoidRunning)
eval_env = envs.get_environment("humanoid")

rng = jax.random.PRNGKey(0)

jit_inference_fn = jax.jit(make_inference_fn(params))
act_rng, rng = jax.random.split(rng)

mj_model: mj.MjModel = eval_env.sys.mj_model
mj_data = mj.MjData(mj_model)


def get_obs(data: mj.MjData) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos
    if True:
        position = position[2:]

    # external_contact_forces are excluded
    return jp.concatenate(
        [
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ]
    )


inference_fn = make_inference_fn(params)

with mujoco.viewer.launch_passive(
    model=mj_model,
    data=mj_data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    while viewer.is_running():
        act_rng, rng = jax.random.split(rng)

        step_start = time.time()

        o = get_obs(mj_data)

        ctrl, _ = inference_fn(o, act_rng)

        mj_data.ctrl = ctrl

        mj.mj_step(mj_model, mj_data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
