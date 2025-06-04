import dm_control.mjcf as dm_mjcf
import jax
import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils.epath import Path
from jax import numpy as jp

import utils.mjx as umjx


class InHandManipulation(PipelineEnv):
    # https://github.com/google-deepmind/mujoco/tree/main/mjx/mujoco/mjx/test_data/shadow_hand
    def __init__(
        self,
        **kwargs,
    ):
        # self.sys = self.init_debug()
        self.sys = self.init()

        self.mj_model = self.sys.mj_model

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(self.sys, **kwargs)

        self._reset_noise_scale = 1e-2

        self.t_w_goal_cube = self.mj_model.geom("cube").pos
        self.q_w_goal_cube = self.mj_model.geom("cube").quat

        self.T_w_goal_cube = umjx.make_tf(
            ori=self.q_w_goal_cube, pos=self.t_w_goal_cube
        )
        self._cube_id = self.mj_model.geom("cube").id

        self._time_out = 10
        self._drop_radius = 0.3

    def init(self) -> System:
        # root
        _MJ_SIM = Path(__file__).parent.parent.parent

        # scene path
        _XML_SCENE = Path(_MJ_SIM / "scenes/empty.xml")
        scene = dm_mjcf.from_path(_XML_SCENE)

        # get global solref
        scene.option.o_solref = [0.00000001, 4]

        # prop
        cube = scene.worldbody.add("body", name="cube", pos="0.3 0 1.1")
        cube.add("geom", name="cube", type="box", size="0.02 0.02 0.02")
        cube.add("joint", name="cube", type="free")

        # shadow hand path
        # _XML_SHADOW_HAND = Path(_MJ_SIM / "assets/shadow_hand/mjx_right_hand.xml")
        _XML_SHADOW_HAND = Path(_MJ_SIM / "assets/mjx_shadow_hand/right_hand.xml")
        # _XML_SHADOW_HAND = Path(_MJ_SIM / "assets/shadow_hand/mjx_shadow_hand.xml")

        shadow_hand = dm_mjcf.from_path(_XML_SHADOW_HAND)

        mocap_body = scene.worldbody.add(
            "body",
            name="mocap",
            pos="0 0 1",
            mocap="true",
        )
        mocap_geom = mocap_body.add(
            "geom", type="box", size="0.01 0.01 0.01", contype="0", conaffinity="0"
        )
        mocap_site = mocap_body.add("site")
        mocap_site.attach(shadow_hand)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        return mjcf.load_model(m)

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "T_error": zero,
        }
        return State(data, obs, reward, done, metrics)

    def _get_current_cube_pose(self, data: mjx.Data) -> jaxl.SE3:
        t_w_current_cube = data.geom_xpos[self._cube_id]
        R_w_current_cube = data.geom_xmat[self._cube_id]
        T_w_current_cube = umjx.make_tf(pos=t_w_current_cube, ori=R_w_current_cube)
        return T_w_current_cube

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        T_w_current_cube = self._get_current_cube_pose(data)

        dT = umjx.pose_error(T_w_current_cube, self.T_w_goal_cube)

        return jp.array(dT)

    def step(self, state: State, action: jp.ndarray) -> State:
        """
        Runs one timestep of the environment's dynamics.

        Args:
            state (State): Current state of the environment.
            action (jp.ndarray): Action to be applied.

        Returns:
            State: Updated state after applying the action.
        """
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        obs = self._get_obs(data)

        reward = self._get_reward(state, data0, data, action)

        done = self._get_done(data)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_reward(self, state, data0, data, action) -> float:
        return -self._get_obs(data).sum()

    def _has_timed_out(self, data: mjx.Data) -> bool:
        return (data.time - self._time_out) > 0.0

    def _get_done(self, data: mjx.Data) -> jp.ndarray:
        time_out = jp.where(self._has_timed_out(data), 1.0, 0.0)
        drop = jp.where(self._get_obs(data).sum() > self._drop_radius, 1.0, 0.0)

        return time_out + drop
