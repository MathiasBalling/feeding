from typing import Callable, List, Sequence

import jax
import mujoco
import mujoco.mjx as mjx
import numpy as np
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp


class HumanoidRunning(PipelineEnv):
    def __init__(
        self,
        **kwargs,
    ):
        """
        HumanoidRunning is a simulation environment for a humanoid robot running task.

        Attributes:
            _forward_reward_weight (float): Weight for the forward reward.
            _ctrl_cost_weight (float): Weight for the control cost penalty.
            _healthy_reward (float): Reward given when the humanoid is in a healthy state.
            _terminate_when_unhealthy (bool): Whether to terminate the episode when unhealthy.
            _healthy_z_range (tuple[float, float]): Range of z-axis values considered healthy.
            _reset_noise_scale (float): Magnitude of random noise added during reset.
            _exclude_current_positions_from_observation (bool): Whether to exclude current positions
                                                                from observations.
        """
        forward_reward_weight = 1.25
        ctrl_cost_weight = 0.1
        healthy_reward = 5.0
        terminate_when_unhealthy = True
        healthy_z_range = (1.0, 2.0)
        reset_noise_scale = 1e-2
        exclude_current_positions_from_observation = True

        path = epath.Path(epath.resource_path("mujoco")) / ("mjx/test_data/humanoid")
        mj_model = mujoco.MjModel.from_xml_path((path / "humanoid.xml").as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        # the number of times to step the physics pipeline for each environment step
        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jp.ndarray): Random key for initializing the environment.

        Returns:
            State: Initial environment state.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))

        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        return State(data, obs, reward, done, metrics)

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
        print(action.shape, action, "----------------------------")
        data = self.pipeline_step(data0, action)

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)

        obs = self._get_obs(data, action)

        reward = self.reward(state, data0, data, action)

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def reward(
        self, state: State, data0: State, data: State, action: jp.ndarray
    ) -> float:
        """
        Computes the reward for the current state transition.

        Args:
            state (State): Current state.
            data0 (State): Previous state data.
            data (State): Current state data.
            action (jp.ndarray): Action applied.

        Returns:
            float: Computed reward.
        """
        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        reward = forward_reward + healthy_reward - ctrl_cost

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return reward

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """
        Observes the humanoid's state, including position, velocity, and other features.

        Args:
            data (mjx.Data): Current environment data.
            action (jp.ndarray): Action applied in the current step.

        Returns:
            jp.ndarray: Observation vector.
        """
        position = data.qpos

        if self._exclude_current_positions_from_observation:
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

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 320,
        height: int = 240,
    ) -> Sequence[jp.ndarray]:
        camera = camera or "side"
        return super().render(trajectory, camera=camera, width=width, height=height)

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
