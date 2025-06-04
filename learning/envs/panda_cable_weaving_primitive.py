import dm_control.mjcf as dm_mjcf
import jax
import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import pandas as pd
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils.epath import Path
from jax import numpy as jp

import utils.mjx as umjx
from ctrl.dmp_cartesian import DMPCartesian
from robots.base_robot import BaseRobot
from utils.helpers import timer
from utils.mj import (
    ObjType,
    RobotInfo,
    get_contact_states,
    get_names,
    get_number_of,
    id2name,
    name2id,
)
from utils.physics import amperes_law


# from mujoco import mjx
class Gripper(BaseRobot):
    def __init__(self, model: mj.MjModel):
        super().__init__()
        self._model = model
        self._info = RobotInfo(self.model, self.name)

    @property
    def name(self) -> str:
        return "panda hand"

    def get_ee_pose(self, data: mjx.Data) -> jaxl.SE3:
        return umjx.get_pose(self._model, data, f"{self.name}/tcp", ObjType.SITE)

    @property
    def data(self) -> mj.MjData:
        return None

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def info(self) -> RobotInfo:
        return self._info

    def step(self):
        return super().step()


class PandaCableWeavingPrimitive(PipelineEnv):
    def __init__(
        self,
        **kwargs,
    ):
        self.sys = self.init()

        self.mj_model = self.sys.mj_model
        # self.all_geoms

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(self.sys, **kwargs)

        self.gripper = Gripper(self.sys.mj_model)

        self.spec = {
            "o0": {"name": "wire_washer_3", "dir": "ccw"},
            "o1": {"name": "wire_washer_2", "dir": "cw"},
            "o2": {"name": "wire_washer_1", "dir": "ccw"},
        }

        self.d_min = 5
        self.d_max = 15
        self.which_spec = 0

        self._keyframe_names = ["unnamed_model/s0_0", "unnamed_model/s0_1"]
        self._keyframe_qpos = [
            jp.array(self.sys.mj_model.key(self._keyframe_names[0]).qpos),
            jp.array(self.sys.mj_model.key(self._keyframe_names[1]).qpos),
        ]

        def load_ref_data(ref_path: str = "data/tmp/ref.json"):
            ref_data = pd.read_json(ref_path)
            ref_data = jp.array([ref_data["x"], ref_data["y"]])
            ref1 = ref_data[:, 100:250]
            ref2 = ref_data[:, 300:400]
            ref3 = ref_data[:, 500:600]
            return [ref1, ref2, ref3]

        _ENVS = Path(__file__).parent

        self.reference_paths = load_ref_data(
            _ENVS / "assets" / Path(__file__).stem / "ref.json"
        )

        self.episode_timeout = 500
        self._reset_noise_scale = 1e-2

        # build two OPP trajectories: one for each of the two cases

        # build two DMP's 1 for s0_0 and one for s0_1
        self.dmp_0 = DMPCartesian()
        self.dmp_1 = DMPCartesian()

        # get the number of keys (init states)
        self.n_key_frames = get_number_of(self.sys.mj_model, ObjType.KEY)

        # only some keyframes are of interest. These are the ones formatted as s0_0 or s0_1
        self.all_keyframe_names = [
            id2name(self.sys.mj_model, key_id, ObjType.KEY)
            for key_id in range(self.n_key_frames)
        ]
        self.key_frame_names = [
            key_name for key_name in self.all_keyframe_names if "_" in key_name
        ]
        self.key_frame_ids = jp.array(
            [
                name2id(self.sys.mj_model, key_name, ObjType.KEY)
                for key_name in self.key_frame_names
            ]
        )
        self.all_geom_names = get_names(self.sys.mj_model, ObjType.GEOM)
        self.cable_names = np.array([gn for gn in self.all_geom_names if "cable" in gn])

    def init(self) -> System:
        # root
        _HERE = Path(__file__).parent.parent.parent

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = dm_mjcf.from_path(_XML_SCENE)

        # create keyframe file
        self.keyframe_path = Path(_HERE, "keyframes", Path(__file__).stem + ".xml")

        # Ensure the parent directory exists
        self.keyframe_path.parent.mkdir(parents=True, exist_ok=True)

        # Create an empty file if it doesn't exist
        if not self.keyframe_path.exists():
            self.keyframe_path.touch()
            # Define the XML content you want to append
            xml_content = """
            <mujoco>
            </mujoco>
            """
            # Open the file in append mode and write the XML content
            with open(self.keyframe_path, "a") as file:
                file.write(xml_content)

            self.keyframe = dm_mjcf.from_path(self.keyframe_path)
        else:
            # find keyframes file and load in
            self.keyframe = dm_mjcf.from_path(self.keyframe_path)
        scene.attach(self.keyframe)

        # panda path
        _XML_PANDA = Path(_HERE / "assets/franka_emika_panda/hand.xml")
        # _XML_panda = Path(_HERE / "assets/robotiq_panda/panda-gs.xml")
        panda_hand = dm_mjcf.from_path(_XML_PANDA)

        mocap_body = scene.worldbody.add(
            "body",
            name="mocap",
            pos="0 0 0.1",
            mocap="true",
        )
        mocap_body.add(
            "geom", type="box", size="0.01 0.01 0.01", contype="0", conaffinity="0"
        )
        mocap_site = mocap_body.add("site", name="mocap_site")

        mocap_site.attach(panda_hand)

        # rgmc practice board
        _XML_BOARD = Path(
            _HERE / "assets/rgmc_practice_task_board_2020/task_board_mjx.xml"
        )
        board = dm_mjcf.from_path(_XML_BOARD)
        scene.attach(board)

        # table
        _XML_TABLE = Path(_HERE / "assets/props/flexcell_top_mjx.xml")
        table = dm_mjcf.from_path(_XML_TABLE)
        scene.attach(table)

        # cable
        _XML_CABLE = Path(_HERE / "assets/props/cable.xml")
        cable = dm_mjcf.from_path(_XML_CABLE)

        cable_body = cable.find("body", "cable")
        cable_body.pos = [0.67521462, 0.27162106, 0.08657647]
        scene.attach(cable)

        # add pseudo fingers
        rf = panda_hand.find("body", "right_finger")
        rf.add(
            "geom",
            name="right_finger",
            type="box",
            size="0.01 0.02 0.01",
            pos="0 0.02 0.1",
            axisangle="1 0 0 -1",
            solref="0.000000001 1",
            friction="0.3 0.3 0.3",
            priority="1",
        )
        lf = panda_hand.find("body", "left_finger")
        lf.add(
            "geom",
            name="left_finger",
            type="box",
            size="0.01 0.02 0.01",
            pos="0 0.02 0.1",
            axisangle="1 0 0 -1",
            solref="0.000000001 1",
            friction="0.3 0.3 0.3",
            priority="1",
        )

        board_body = board.worldbody.find("body", "task_board")
        board_body.pos[0] -= 0.2
        wire_washer_3 = board.worldbody.find("geom", "wire_washer_3")
        wire_washer_2 = board.worldbody.find("geom", "wire_washer_2")
        wire_washer_1 = board.worldbody.find("geom", "wire_washer_1")
        wire_bolt_3 = board.worldbody.find("geom", "wire_bolt_3")
        wire_bolt_2 = board.worldbody.find("geom", "wire_bolt_2")
        wire_bolt_1 = board.worldbody.find("geom", "wire_bolt_1")
        wire_washer_3.pos[0] -= 0.2
        wire_washer_2.pos[0] -= 0.2
        wire_washer_1.pos[0] -= 0.2
        wire_bolt_3.pos[0] -= 0.2
        wire_bolt_2.pos[0] -= 0.2
        wire_bolt_1.pos[0] -= 0.2

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        return mjcf.load_model(m)

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # pick an init state at random
        keyframe_id = jax.random.choice(rng1, self.key_frame_ids)

        # NOTE: Comment this out as it ensures consistent s0
        keyframe_id = 0

        # select the primitive parameters based on the s0
        self.which_spec = keyframe_id

        # get the s0 name
        # keyframe_name = id2name(self.sys.mj_model, keyframe_id, ObjType.KEY)

        # generate a small amount of noise and add is to the current qpos for state randomization
        qpos = self._keyframe_qpos[keyframe_id]
        # qpos = self.sys.qpos0
        # qpos = self.keyframe.qpos
        # qpos = self.sys.qpos0 + jax.random.uniform(
        #     rng1, (self.sys.nq,), minval=low, maxval=hi
        # )
        # generate a small amount of noise and add is to the current qvel for derivative state randomization
        qvel = jax.random.uniform(rng1, (self.sys.nv,), minval=low, maxval=hi)

        print(
            "pipeline init?... //////////////////////////////////////////////////////"
        )
        # get the data corresponding to this qpos and qvel
        with timer("initialize pipeline..."):
            state = self.pipeline_init(qpos, qvel)

        print("loading key... //////////////////////////////////////////////////////")
        # load s0
        # load_keyframe(
        #     self.sys.mj_model,
        #     None,
        #     keyframe_name,
        #     log=False,
        #     file_path=self.keyframe_path,
        # )

        print(
            "loaded key successfully... //////////////////////////////////////////////////////"
        )

        obs = self.get_obs(state, jp.zeros(self.sys.nu))

        reward, done, zero = jp.zeros(3)

        metrics = {
            "in_grasp": zero,
            "homotopy_class": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }

        # reset canonical system
        info = {"x_phase": 1}

        # return the observation form s0 i.e. o0
        return State(state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        # this is the data from the current state
        data0 = state.pipeline_state

        # by applying the action we attain the next state
        data = self.pipeline_step(data0, action)

        # perform the action (normally this is done automatically)
        df = action[:3]
        dtau = action[3:]

        # we here need to build these things:
        #  1. reward
        #  2. observation
        #  3. done
        # we should be using the new data labeled 'data' for checking the state

        # get the done flag:
        done = self.get_done(data)

        # get the current reward as produced by 'action'
        reward = self.get_reward(data)

        # get the observation
        obs = self.get_obs(data, action)

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

        obs = self.get_obs(data, action)

        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
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

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def __get_local_cable_pts(
        self, data: mjx.Data, d_min: int, d_max: int
    ) -> list[list[float]]:
        """
        Get local cable points.

        Parameters:
        - wire_sim: The simulation object.
        - d_min (int): Minimum distance.
        - d_max (int): Maximum distance.

        Returns:
        - list[list[float]]: Local cable points.
        """
        # find the poses of all the cable geometries
        cable_poses = [
            umjx.get_pose(self.mj_model, data, cn, ObjType.GEOM)
            for cn in self.cable_names
        ]
        # print(f"{cable_poses=}")

        # get gripper ee pose
        T_w_tcp = self.gripper.get_ee_pose(data)
        # print(f"{T_w_tcp=}")

        # l = []
        # t_w_tcp = T_w_tcp.as_matrix()[:3, 3]
        # T_mat = T_w_tcp.as_matrix()

        # print(f"{T_mat=}")

        # t_w_tcp = T_mat[:3, -1]  # Get the translation vector
        # for cp in cable_poses:
        #     dt = jp.linalg.norm(cp.translation() - T_w_tcp.translation())
        #     print("Shape of cp.as_matrix():", cp.as_matrix().shape)
        #     print("Shape of T_w_tcp.as_matrix():", T_w_tcp.as_matrix().shape)
        #     print("Shape of cp.translation():", cp.translation().shape)
        #     print("Shape of T_w_tcp.translation():", T_w_tcp.translation().shape)
        #     dt1 = jp.linalg.norm(cp.as_matrix()[:3, 3] - T_w_tcp.as_matrix()[:3, 3])
        #     dt2 = jp.linalg.norm(cp.translation() - T_w_tcp.translation())
        #     l.append(dt)
        # print(f"{dT=}")
        # dT = jp.array(dt)
        # print(f"{dT=}")
        # compute the euclidean distance between each cable segment and the tcp
        dT = jp.array(
            [
                jp.linalg.norm(cp.translation() - T_w_tcp.translation())
                for cp in cable_poses
            ]
        )

        # TODO: Figure out how to perform slicing with array? meaning jax scalar

        jax.device_get(dT)

        print(f"{dT=}")

        # determine the index of the cable segment closest to the tcp
        idT = jp.argmin(dT)

        cnsoi_high = jp.clip(idT + d_max, a_min=0, a_max=len(self.cable_names) + 1)
        cnsoi_low = jp.clip(idT - d_min, a_min=0, a_max=len(self.cable_names) + 1)

        cnsoi_high = jax.device_get(cnsoi_high)
        cnsoi_low = jax.device_get(cnsoi_low)

        print("self.cable_names", self.cable_names)
        print(f"{idT=}")
        print(f"{d_min=}")
        print(f"{d_max=}")
        print(f"{len(self.cable_names)+1=}")
        print(f"{cnsoi_high=}")
        print(f"{cnsoi_low=}")
        cnsoi = self.cable_names[cnsoi_low:cnsoi_high]

        # Ensure the length of cnsoi is 20
        required_length = d_min + d_max
        current_length = len(cnsoi)

        if current_length < required_length:
            # Calculate the number of zero vectors to add
            zeros_to_add = required_length - current_length
            zeros = [[0, 0, 0]] * zeros_to_add

            cnsoi = [
                umjx.get_pose(self.mj_model, data, cnoi, ObjType.GEOM).translation()
                for cnoi in cnsoi
            ]

            # Append zeros on the left if idT < d_min
            if idT < d_min:
                cnsoi = zeros + cnsoi
            # Append zeros on the right if idT > len(cable_names) - d_max
            elif idT > len(self.cable_names) - d_max:
                cnsoi = cnsoi + zeros

        return jp.array(cnsoi)

    def __get_local_cable_name(self, data: mjx.Data):
        # find the poses of all the cable geometries
        cable_poses = [
            umjx.get_pose(self.mj_model, data, cn, ObjType.GEOM)
            for cn in self.cable_names
        ]

        # get gripper ee pose
        T_w_tcp = self.gripper.get_ee_pose(data)

        # compute the euclidean distance between each cable segment and the tcp
        dT = [
            jp.linalg.norm(cp.translation() - T_w_tcp.translation())
            for cp in cable_poses
        ]

        # determine the index of the cable segment closest to the tcp
        idT = jp.argmin(dT)

        return self.cable_names[idT]

    def get_done(self, data: mjx.Data) -> bool:
        timeout = data.time > self.episode_timeout

        # has the cable been dropped
        has_dropped = self._is_dropped(data)

        # is cable below
        csoi = self._get_cs_of_interest(data)
        T_w_csoi = umjx.get_pose(self.mj_model, data, csoi, ObjType.GEOM)
        T_w_obstacle = umjx.get_pose(
            self.mj_model, data, self.spec[self.which_spec]["o"], ObjType.GEOM
        )
        p_obstacle = T_w_obstacle.translation()
        is_cable_below = p_obstacle.at[2] > T_w_csoi.translation().at[2]

        # same homotopy class
        reference_path = self.reference_paths[self.which_spec]
        cable_path = self.__get_local_cable_pts(data, self.d_min, self.d_max)[:, :2].T
        I_wire = 10
        fused_paths = self._fuse_curves(reference_path, cable_path).T
        I_fused_paths = amperes_law(fused_paths, I_wire, p_obstacle[:2])

        curves_enclose_obs = (I_wire - I_fused_paths) < I_wire / 2.0
        same_homotopy_class = not curves_enclose_obs

        success = same_homotopy_class and is_cable_below
        failure = has_dropped or timeout

        done = success or failure

        return done

    def _is_dropped(self, data: mj.MjData, buffer: float = 0.01):
        def in_grasp(data: mj.MjData) -> bool:
            csoi_name = self.__get_local_cable_name(data)
            t_w_csoi = umjx.get_pose(
                self.mj_model, data, csoi_name, ObjType.GEOM
            ).translation()
            t_w_wcp = self.gripper.get_ee_pose().translation()
            dT = jp.linalg.norm(t_w_csoi - t_w_wcp)

            twof85_pad_width = 0.022  # m
            twof85_pad_height = 0.0375  # m
            radius = (
                jp.linalg.norm([twof85_pad_width / 2.0, twof85_pad_height / 2.0])
                + buffer
            )
            return radius > dT

        return not in_grasp(data)

    def get_reward(self, data: mjx.Data) -> float:
        def r_grasp() -> float:
            r_max = 10
            csoi_pose = umjx.get_pose(
                self.sys.mj_model, data, self._get_cs_of_interest(data), ObjType.GEOM
            )
            d = jp.linalg.norm(csoi_pose.t - self.gripper.get_ee_pose().t)
            twof85_pad_width = 0.022  # m
            twof85_pad_height = 0.0375  # m
            R = jp.linalg.norm([twof85_pad_width / 2.0, twof85_pad_height / 2.0]) + 0.01
            s = 200
            w = 0.8 * R
            return r_max * jp.exp(-jp.exp((-s * (d - w / 2))))

        def r_class() -> float:
            T_w_obstacle = umjx.get_pose(
                self.sys.mj_model, data, self.spec[self.which_spec]["o"], ObjType.GEOM
            )
            p_obstacle = T_w_obstacle.translation()
            reference_path = self.reference_paths[self.which_spec]
            cable_path = self.__get_local_cable_pts(data, self.d_min, self.d_max)[
                :, :2
            ].T
            I_wire = 10
            fused_paths = self._fuse_curves(reference_path, cable_path).T
            I_fused_paths = amperes_law(fused_paths, I_wire, p_obstacle[:2])
            return 1 / (max(I_fused_paths, 0.1))

        def r_wrench() -> float:
            l_in_contact, css_l = get_contact_states(
                self.sys.mj_model, data, "left_pad"
            )
            r_in_contact, css_r = get_contact_states(
                self.sys.mj_model, data, "right_pad"
            )

            # no contact
            if not l_in_contact and not r_in_contact:
                return 0

            # right pad only contact
            if r_in_contact and not l_in_contact:
                return -jp.mean([jp.mean(cs.wrench) for cs in css_r])

            # left pad only contact
            if l_in_contact and not r_in_contact:
                return -jp.mean([jp.mean(cs.wrench) for cs in css_l])

            # both pads contact
            mu_css_l = jp.mean([jp.mean(cs.wrench) for cs in css_l])
            mu_css_r = jp.mean([jp.mean(cs.wrench) for cs in css_r])

            return -jp.mean([mu_css_l, mu_css_r])

        r = r_grasp() + r_class() + r_wrench()
        return r

    def _get_cs_of_interest(self, data: mjx.Data) -> str:
        # find the poses of all the cable geometries
        cable_poses = [
            umjx.get_pose(self.sys.mj_model, data, cn, ObjType.GEOM)
            for cn in self.cable_names
        ]

        # get gripper ee pose
        T_w_tcp: jaxl.SE3 = self.gripper.get_ee_pose()

        # compute the euclidean distance between each cable segment and the tcp
        dT = [
            jp.linalg.norm(cp.translation() - T_w_tcp.translation())
            for cp in cable_poses
        ]

        # determine the index of the cable segment closest to the tcp
        idT = jp.argmin(dT)

        # return the name of the cable segment closest to the tcp
        return self.cable_names[idT]

    def get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """
        Placeholder function.

        Returns:
            np.ndarray: An array.
        """

        # observation space [ *[all cable segment points of interest], *[obstacle of interest]     ]
        o = []
        cable_pts = self.__get_local_cable_pts(data, self.d_min, self.d_max)

        flat_cp = [item for sublist in cable_pts for item in sublist]

        flat_pp = umjx.get_pose(
            self.sys.mj_model, data, self.spec[self.which_spec]["o"], ObjType.GEOM
        ).translation()

        o.extend(flat_cp)
        o.extend(flat_pp)

        return jp.array(o, dtype=jp.float32)

    def _fuse_curves(self, curve_1: jp.ndarray, curve_2: jp.ndarray) -> jp.ndarray:
        x = jp.append(curve_1[0], curve_2[0])
        y = jp.append(curve_1[1], curve_2[1])
        x = jp.append(x, x[0])
        y = jp.append(y, y[0])
        return jp.array([x, y])

    # def _opp(model: mj.MjModel, data: mj.MjData, T_init: sm.SE3):
    #     def opp1(h: float, geom_name: str, T_init: sm.SE3 = None) -> sm.SE3:
    #         """
    #         Generate a Cartesian trajectory from the end effector pose to a new pose above an obstacle.

    #         Parameters
    #         ----------
    #         poses : List[sm.SE3]
    #             List of poses (SE3 transformations).
    #         h : float
    #             Height above the obstacle.
    #         obs_name : str
    #             Name of the obstacle geometry.
    #         n_steps : int, optional
    #             Number of steps in the generated trajectory, by default 1000.

    #         Returns
    #         -------
    #         List[sm.SE3]
    #             Updated list of poses with the generated trajectory appended.
    #         """
    #         if T_init is None:
    #             T_w_tcp = self.gripper.get_ee_pose()  # Current TCP pose in world frame
    #         else:
    #             T_w_tcp = T_init

    #         T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)
    #         Tz = sm.SE3.Tz(h + (T_w_geom.t[2] - T_w_tcp.t[2]))
    #         T_w_up = Tz @ T_w_tcp

    #         # T = ctraj(T_start=T_w_tcp, T_end=T_world_up, t=n_steps)
    #         # # T = rtb.ctraj(T0=T_world_tcp, T1=T_world_up, t=n_steps)
    #         # poses.extend(T)
    #         # return poses
    #         return T_w_up

    #     def opp2(
    #         self, theta: float, dir: str, geom_name: str, T_init: sm.SE3 = None
    #     ) -> sm.SE3:
    #         """
    #         Generate a Cartesian trajectory around an obstacle.

    #         Parameters
    #         ----------
    #         poses : List[sm.SE3]
    #             List of poses (SE3 transformations).
    #         delta : float
    #             Distance over the obstacle.
    #         theta : float
    #             Angle of rotation around the obstacle (in radians).
    #         dir : str
    #             Direction of rotation ("ccw" for counterclockwise, "cw" for clockwise).
    #         obs_name : str
    #             Name of the obstacle geometry.
    #         n_steps : int, optional
    #             Number of steps in the generated trajectory, by default 1000.

    #         Returns
    #         -------
    #         List[sm.SE3]
    #             Updated list of poses with the generated trajectory appended.
    #         """

    #         if dir == "ccw":
    #             theta = theta
    #         elif dir == "cw":
    #             theta = -theta
    #         else:
    #             raise ValueError(
    #                 f'dir must be either "ccw" or "cw", but "{dir}" was given'
    #             )

    #         if T_init is None:
    #             T_w_tcp = self.gripper.get_ee_pose()  # Current TCP pose in world frame
    #         else:
    #             T_w_tcp = T_init

    #         T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)

    #         # angle to slign axis with direction towards obstacle
    #         def get_phi() -> float:
    #             tcp_y = T_w_tcp.R[:2, 1]
    #             a = -tcp_y
    #             b = T_w_geom.t[:2] - T_w_tcp.t[:2]
    #             return angle(a, b)

    #         # the angle to correct the direction and the deviation angle theta
    #         theta = get_phi() + theta

    #         return T_w_tcp @ sm.SE3.Rz(theta)

    #     def opp3(
    #         self,
    #         delta: float,
    #         theta: float,
    #         dir: str,
    #         geom_name: str,
    #         T_init: sm.SE3 = None,
    #     ) -> sm.SE3:
    #         """
    #         Compute a target pose for the robot's end-effector based on positional and rotational offsets.

    #         Parameters
    #         ----------
    #         delta : float
    #             Distance offset from the current end-effector position towards the target geometry in the xy-plane.
    #         theta : float
    #             Angle of rotation around the z-axis (in radians).
    #         dir : str
    #             Direction of rotation ("ccw" for counterclockwise, any other value for clockwise).
    #         geom_name : str
    #             Name of the target geometry to compute the offset and rotation with respect to.
    #         T_init : sm.SE3, optional
    #             Initial pose of the end-effector. If None, the current pose is fetched from the robot, by default None.

    #         Returns
    #         -------
    #         sm.SE3
    #             The computed target pose in the world frame.
    #         """
    #         if T_init is None:
    #             T_w_tcp = self.gripper.get_ee_pose()  # Current TCP pose in world frame
    #         else:
    #             T_w_tcp = T_init

    #         T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)

    #         t_tcp_geom = T_w_geom.t[:2] - T_w_tcp.t[:2]
    #         t_tcp_geom_norm = t_tcp_geom / np.linalg.norm(t_tcp_geom)
    #         t_tcp_target = (delta * (t_tcp_geom_norm)) + t_tcp_geom

    #         if dir == "ccw":
    #             theta = -theta

    #         t_tcp_target_rot = rotate_vector_2d(t_tcp_target, theta)

    #         p = T_w_tcp.t + np.append(t_tcp_target_rot, 0)

    #         rz = np.array([0, 0, -1])
    #         rx = np.append(
    #             (T_w_geom.t[:2] - p[:2]) / np.linalg.norm(T_w_geom.t[:2] - p[:2]), 0
    #         )
    #         if dir == "ccw":
    #             rx *= -1
    #         ry = np.cross(rz, rx) / np.linalg.norm(np.cross(rz, rx))

    #         R = np.column_stack((rx, ry, rz))

    #         if np.linalg.det(R) < 0:
    #             ry *= -1
    #             R = np.column_stack((rx, ry, rz))

    #         T_w_target = make_tf(pos=p, ori=R)

    #         return T_w_target

    #     def opp4(self, z_height: float, T_init: sm.SE3 = None) -> sm.SE3:
    #         """
    #         Compute a target pose for the robot's end-effector with a specified z-height.

    #         Parameters
    #         ----------
    #         z_height : float
    #             Desired height (z-coordinate) for the end-effector in the world frame.
    #         T_init : sm.SE3, optional
    #             Initial pose of the end-effector. If None, the current pose is fetched from the robot, by default None.

    #         Returns
    #         -------
    #         sm.SE3
    #             The computed target pose with the specified z-height.
    #         """
    #         if T_init is None:
    #             T_w_tcp = self.gripper.get_ee_pose()  # Current TCP pose in world frame
    #         else:
    #             T_w_tcp = T_init

    #         end_pos = T_w_tcp.t.copy()  # Make a copy of the translation vector
    #         end_pos[2] = z_height
    #         T_target = make_tf(pos=end_pos, ori=T_w_tcp.R)

    #         return T_target

    #     traj = []

    #     return traj
