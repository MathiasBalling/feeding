import jax
import mujoco
import mujoco.mjx as mjx
from brax import State, System
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from jax import numpy as jp


class UpscaledDataCollection(PipelineEnv):
    def __init__(
        self,
        **kwargs,
    ):
        self.sys = self.init()

        # the number of times to step the physics pipeline for each environment step
        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(self.sys, **kwargs)

    def init(self) -> System:
        _XML = """
        <mujoco>
        <worldbody>
            <light name="top" pos="0 0 1"/>
            <body name="box_and_sphere" euler="0 0 -30">
            <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
            <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
            <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
            </body>
        </worldbody>
        </mujoco>
        """
        mj_model = mujoco.MjModel.from_xml_string(_XML)
        return mjcf.load_model(mj_model)

    def reset(self, rng):
        """Reset the environment for multiple parallel simulations."""

        @jax.vmap
        def _reset(rng):
            qpos = self.sys.qpos0  # Initial positions (e.g., box height)
            qvel = jp.zeros(self.sys.nv)  # Initial velocities
            data = self.pipeline_init(qpos, qvel)
            return State(data, jp.array(0), jp.array(0), jp.array(False), None)

        return _reset(rng)

    @jax.vmap
    def step(self, state: State, action: jp.ndarray) -> State:
        """Step the environment for one time step in parallel."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        return state.replace(
            pipeline_state=data,
            obs=jp.array(0),
            reward=jp.array(0),
            done=jp.array(False),
        )


def domain_randomize(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = (
            jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + sys.actuator_gainprm[:, 0]
        )
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)
    # print("friction", friction)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


env = UpscaledDataCollection()

seed: int = 1
n_timesteps: int = 10
n_sims = 20

rng = jax.random.PRNGKey(seed)

rng = jax.random.split(rng, n_sims)
batched_sys, _ = domain_randomize(env.sys, rng)

# Make model, data, and renderer
mj_model = batched_sys.mj_model
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

for i in range(n_timesteps):
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    batch: mjx.Data = jit_step(mjx_model, batch)

    # see shape of batch
    print("shape of batch data: ", batch.geom_xpos.shape)

# load the batched mjx.Data back into mj
batched_mj_data = mjx.get_data(mj_model, batch)
# display values in batch
print([d.qpos for d in batched_mj_data])
