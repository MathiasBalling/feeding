from typing import Any, Callable, Optional, Tuple

import jax
import mujoco
import numpy as np
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp
from ml_collections import ConfigDict, config_dict


def default_config() -> ConfigDict:
    """Returns reward config for the environment."""

    return config_dict.create(
        # Environment timestep. Should match the robot decision frequency.
        dt=0.02,
        # Lowers action magnitude for less-jerky motion.  Also sometimes helps
        # sample efficiency.
        action_scale=0.04,
        # The coefficients for all reward terms used for training.
        reward_scales=config_dict.create(
            # Gripper goes to the box.
            gripper_box=4.0,
            # Box goes to the target mocap.
            box_target=8.0,
            # Do not collide the gripper with the floor.
            no_floor_collision=0.25,
            # Arm stays close to target pose.
            robot_target_qpos=0.3,
        ),
    )


def _load_sys() -> System:
    """Load a mujoco model from a path."""

    _HERE = epath.Path(__file__).parent.parent.parent
    path = _HERE / "assets" / "franka_emika_panda" / "mjx_single_cube.xml"

    assets = {}
    for f in path.parent.glob("*.xml"):
        assets[f.name] = f.read_bytes()
    for f in (path.parent / "assets").glob("*"):
        assets[f.name] = f.read_bytes()
    xml = path.read_text()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return mjcf.load_model(model)


def _get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
    if geom1 > geom2:
        geom1, geom2 = geom2, geom1
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def _geoms_colliding(state: Optional[State], geom1: int, geom2: int) -> jax.Array:
    return _get_collision_info(state.contact, geom1, geom2)[0] < 0


class PandaBringToTarget(PipelineEnv):
    """Environment for training franka panda to bring an object to target."""

    def __init__(self, **kwargs):
        sys = _load_sys()

        self.camera_name = "tracker"

        self._config = config = default_config()

        nsteps = int(np.round(config.dt / sys.opt.timestep))
        kwargs["backend"] = "mjx"
        kwargs["n_frames"] = nsteps
        super().__init__(sys, **kwargs)

        # define constants
        model = sys.mj_model
        arm_joints = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        finger_joints = ["finger_joint1", "finger_joint2"]
        all_joints = arm_joints + finger_joints
        self._robot_arm_qposadr = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in arm_joints]
        )
        self._robot_qposadr = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in all_joints]
        )
        self._gripper_site = model.site("gripper").id
        self._left_finger_geom = model.geom("left_finger_pad").id
        self._right_finger_geom = model.geom("right_finger_pad").id
        self._hand_geom = model.geom("hand_capsule").id
        self._box_body = model.body("box").id
        self._box_qposadr = model.jnt_qposadr[model.body("box").jntadr[0]]
        # TODO(btaba): replace with mocap_pos once MJX version 3.2.3 is released.
        self._target_id = model.body("mocap_target").id
        self._floor_geom = model.geom("floor").id
        self._init_q = sys.mj_model.keyframe("home").qpos
        self._init_box_pos = jp.array(
            self._init_q[self._box_qposadr : self._box_qposadr + 3],
            dtype=jp.float32,
        )
        self._init_ctrl = sys.mj_model.keyframe("home").ctrl
        self._lowers = model.actuator_ctrlrange[:, 0]
        self._uppers = model.actuator_ctrlrange[:, 1]

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # intialize box position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.0]),
                maxval=jp.array([0.2, 0.2, 0.0]),
            )
            + self._init_box_pos
        )

        # initialize target position
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, -0.2, 0.2]),
                maxval=jp.array([0.2, 0.2, 0.4]),
            )
            + self._init_box_pos
        )

        # initialize pipeline state
        init_q = (
            jp.array(self._init_q)
            .at[self._box_qposadr : self._box_qposadr + 3]
            .set(box_pos)
        )
        pipeline_state = self.pipeline_init(init_q, jp.zeros(self.sys.nv))
        pipeline_state = pipeline_state.replace(ctrl=self._init_ctrl)
        # set target mocap position
        # TODO(btaba): replace with mocap_pos once MJX version 3.2.3 is released.
        pipeline_state = pipeline_state.replace(
            xpos=pipeline_state.xpos.at[self._target_id, :].set(target_pos)
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0),
            **{k: 0.0 for k in self._config.reward_scales.keys()},
        }
        self.info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
        obs = self._get_obs(pipeline_state, self.info)
        reward, done = jp.zeros(2)
        state = State(pipeline_state, obs, reward, done, metrics, self.info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._config.action_scale
        ctrl = state.pipeline_state.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        # step the physics
        data = self.pipeline_step(state.pipeline_state, ctrl)

        # compute reward terms
        target_pos = state.info["target_pos"]
        box_pos = data.xpos[self._box_body]
        gripper_pos = data.site_xpos[self._gripper_site]
        box_target = 1 - jp.tanh(5 * jp.linalg.norm(target_pos - box_pos))
        gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                state.pipeline_state.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        hand_floor_collision = [
            _geoms_colliding(state.pipeline_state, self._floor_geom, g)
            for g in [
                self._left_finger_geom,
                self._right_finger_geom,
                self._hand_geom,
            ]
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = 1 - floor_collision

        state.info["reached_box"] = 1.0 * jp.maximum(
            state.info["reached_box"],
            (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
        )

        rewards = {
            "box_target": box_target * state.info["reached_box"],
            "gripper_box": gripper_box,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        rewards = {k: v * self._config.reward_scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        state.metrics.update(out_of_bounds=out_of_bounds.astype(float), **rewards)

        obs = self._get_obs(data, state.info)
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def _get_obs(self, data: State, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
                gripper_pos,
                gripper_mat[3:],
                data.xmat[self._box_body].ravel()[3:],
                data.xpos[self._box_body] - data.site_xpos[self._gripper_site],
                info["target_pos"] - data.xpos[self._box_body],
                data.ctrl - data.qpos[self._robot_qposadr[:-1]],
            ]
        )

        return obs

    def get_demo_video(
        self,
        make_inference_fn: Callable,
        params: tuple,
        n_steps: int = 500,
        render_every: int = 2,
    ) -> np.ndarray:
        jit_reset = jax.jit(self.reset)
        jit_step = jax.jit(self.step)
        inference_fn = make_inference_fn(params)
        jit_inference_fn = jax.jit(inference_fn)

        # initialize the state
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)
        rollout = [state.pipeline_state]

        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)

        # convert to numpy array
        frames = np.array(self.render(rollout[::render_every]))

        # proper dimensions of frames
        frames = np.transpose(np.array(frames), axes=(0, 3, 1, 2))

        return frames
