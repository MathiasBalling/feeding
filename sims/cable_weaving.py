from pathlib import Path
from typing import Dict

import glfw
import mujoco as mj
import numpy as np
import spatialmath as sm
from dm_control import mjcf

from robots.mocap import Mocap
from robots.twof85 import Twof85
from sims import BaseSim
from sims.base_sim import SimSync
from utils.math import angle, rotate_vector_2d
from utils.mj import ObjType, get_pose, load_keyframe
from utils.sm import make_tf


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        self.twof85 = Twof85(self._model, self._data)
        self.mocap = Mocap(self.model, self.data)

        T_w_mocap = get_pose(self.model, self.data, "mocap_site", ObjType.SITE)
        T_w_tcp = self.twof85.get_ee_pose()
        self.T_mocap_tcp = T_w_mocap.inv() @ T_w_tcp

        self.spec = {
            "o0": {"name": "task_board/wire_washer_3", "dir": "ccw"},
            "o1": {"name": "task_board/wire_washer_2", "dir": "cw"},
            "o2": {"name": "task_board/wire_washer_1", "dir": "ccw"},
        }

        self.tasks = [self.spin, self.weaving]

        load_keyframe(self.model, self.data, "unnamed_model/s0", self.keyframe_path)

        self.h = 0.05
        self.theta = np.deg2rad(30)
        self.delta = 0.1

        self.begin = True

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent

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

        else:
            # find keyframes file and load in
            keyframe = mjcf.from_path(self.keyframe_path)

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        scene.attach(keyframe)

        # 2f85 path
        _XML_2F85 = Path(_HERE / "assets/robotiq_2f85/2f85.xml")
        # _XML_2F85 = Path(_HERE / "assets/robotiq_2f85/2f85-gs.xml")
        twof85 = mjcf.from_path(_XML_2F85)

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

        mocap_site.attach(twof85)

        # rgmc practice board
        _XML_BOARD = Path(_HERE / "assets/rgmc_practice_task_board_2020/task_board.xml")
        board = mjcf.from_path(_XML_BOARD)
        scene.attach(board)

        # attach site to usb_clamp
        usb_clamp_body = board.find("body", "usb_clamp")
        usb_clamp_body.add(
            "site",
            name="usb_clamp",
            pos="0.36343917 0.173  0.0279117",
            euler="0 0 1.57",
        )

        # make usb_clamp non colliding
        usb_clamp_geom = board.find("geom", "usb_clamp")
        usb_clamp_geom.contype = 0
        usb_clamp_geom.conaffinity = 0

        # table
        _XML_TABLE = Path(_HERE / "assets/props/flexcell_top.xml")
        table = mjcf.from_path(_XML_TABLE)
        scene.attach(table)

        # cable
        _XML_CABLE = Path(_HERE / "assets/props/cable.xml")
        cable = mjcf.from_path(_XML_CABLE)
        usb_clamp_site = board.find("site", "usb_clamp")

        usb_clamp_site.attach(cable)

        left_pad = twof85.find("body", "left_pad")
        left_pad.axisangle = [1, 0, 0, 0.4]
        right_pad = twof85.find("body", "right_pad")
        right_pad.axisangle = [1, 0, 0, 0.4]

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
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()

    def get_check_point_frames(self) -> Dict[str, sm.SE3]:
        traj = {}
        traj["init"] = self.twof85.get_ee_pose()

        o_name = self.spec["o0"]["name"]
        o_dir = self.spec["o0"]["dir"]
        z_height = self.twof85.get_ee_pose().t[2]
        T = self.init_orientation(o_name)
        traj["init_orientation"] = T
        N_SPEC = len(self.spec.keys())
        for i in range(N_SPEC):
            o_name = self.spec[f"o{i}"]["name"]
            o_dir = self.spec[f"o{i}"]["dir"]
            T = self.move_close(o_name, T_init=T)
            traj[f"move_close_{i}"] = T
            T = self.opp1(self.h, o_name, T_init=T)
            traj[f"opp1_{i}"] = T
            T = self.opp2(self.theta, o_dir, o_name, T_init=T)
            traj[f"opp2_{i}"] = T
            T = self.opp3(self.delta, self.theta, o_dir, o_name, T_init=T)
            traj[f"opp3_{i}"] = T
            T = self.opp4(z_height, T_init=T)
            traj[f"opp4_{i}"] = T
            if i <= N_SPEC - 2:
                o_next_name = self.spec[f"o{i+1}"]["name"]
                phi = self.angle_to_next_alignment(i, o_name, o_next_name, T_init=T)
                T = self.rotate_about(o_name, o_dir, phi, T_init=T)
                for j, Ti in enumerate(T):
                    traj[f"rotate_about_{i}_{j}"] = Ti
                T = T[-1]

        return traj

    def weaving(self, ss: SimSync):
        while not self.begin:
            ss.step()

        DEFAULT_VEL = 0.1
        FAST_VEL = 1
        o_name = self.spec["o0"]["name"]
        o_dir = self.spec["o0"]["dir"]
        z_height = self.twof85.get_ee_pose().t[2]
        T = self.init_orientation(o_name)
        # rotate the gripper towards the first obstacle
        self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=FAST_VEL)

        N_SPEC = len(self.spec.keys())
        for i in range(N_SPEC):
            o_name = self.spec[f"o{i}"]["name"]
            o_dir = self.spec[f"o{i}"]["dir"]
            T = self.move_close(o_name, T_init=T, radius=0.05)
            # move towards the i'th obstacle and keep moving until a distance of (radius) is achieved
            self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=DEFAULT_VEL)
            # lift up the cable
            T = self.opp1(self.h, o_name, T_init=T)
            self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=DEFAULT_VEL)
            # rotate the gripper theta
            T = self.opp2(self.theta, o_dir, o_name, T_init=T)
            self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=FAST_VEL)
            # move over the obstacle distance delta
            T = self.opp3(self.delta, self.theta, o_dir, o_name, T_init=T)
            self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=FAST_VEL)
            # move down again
            T = self.opp4(z_height, T_init=T)
            self.mocap.move_l(T @ self.T_mocap_tcp.inv(), ss, velocity=DEFAULT_VEL)
            if i <= N_SPEC - 2:
                # rotate about the obstacle until the gripper is aligned with the new
                o_next_name = self.spec[f"o{i+1}"]["name"]
                phi = self.angle_to_next_alignment(i, o_name, o_next_name, T_init=T)
                T = self.rotate_about(o_name, o_dir, phi, T_init=T)
                for Ti in T:
                    self.mocap.move_l(
                        Ti @ self.T_mocap_tcp.inv(), ss, velocity=FAST_VEL
                    )
                T = T[-1]

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            print("You pressed space...")
        if key is glfw.KEY_PERIOD:
            print("You pressed period...")

    def init_orientation(self, geom_name: str, vel: float = 0.3, T_init: sm.SE3 = None):
        """
        Align the TCP's y-axis to point directly away from a specified object.

        Parameters:
        - geom_name (str): Name of the geometry to align away from.
        - vel (float): Desired rotation velocity in radians per second (currently unused).
        - T_init (sm.SE3, optional): Initial pose of the TCP. If None, fetches the current TCP pose.

        Returns:
        - sm.SE3: The target pose after rotation.
        """
        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        ry_T_w_tcp = T_w_tcp.R[:2, 1]  # Extract the x-axis of the TCP's rotation matrix

        T_w_o = get_pose(self.model, self.data, geom_name, ObjType.GEOM)

        p_tcp_p_o = (
            T_w_o.t[:2] - T_w_tcp.t[:2]
        )  # Relative 2D position from TCP to object

        # Normalize vectors for angle calculation
        ry_T_w_tcp /= np.linalg.norm(ry_T_w_tcp)
        p_tcp_p_o /= np.linalg.norm(p_tcp_p_o)

        # Compute the angle between the TCP's x-axis and the relative position vector
        phi = angle(
            ry_T_w_tcp, p_tcp_p_o
        )  # Angle between current x-axis and target direction
        cross = np.cross(
            ry_T_w_tcp, p_tcp_p_o
        )  # Determine rotation direction (2D cross product)

        if cross < 0:
            theta = -phi  # Rotate counter-clockwise
        else:
            theta = np.pi - phi  # Rotate clockwise

        # Compute the target pose by rotating the TCP pose around the z-axis
        T_target = T_w_tcp @ sm.SE3.Rz(theta)

        return T_target

    def move_close(
        self,
        geom_name: str,
        radius: float = 0.06,
        T_init: sm.SE3 = None,
    ) -> sm.SE3:
        """
        Move towards the object specified by geom_name until within a given radius.

        Parameters:
        - geom_name: str, the name of the geometry to move towards.
        - radius: float, the distance to maintain from the object.
        - vel: float, the velocity of the movement.

        Returns:
        - sm.SE3: Pose radius away from a geometry.
        """

        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init
        T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)
        dpos = T_w_geom.t - T_w_tcp.t
        dt = np.linalg.norm(dpos)

        # Adjust the translation to stop within the specified radius
        if dt > radius:
            pos_unit = dpos / dt
            scale = dt - radius
            pos_unit *= scale
            T1 = make_tf(pos=T_w_tcp.t + pos_unit, ori=T_w_tcp.R)
            # T1 = sm.SE3.Rt(R=T_gripper.R, t=T_gripper.t + pos_unit)
        else:
            T1 = T_w_geom

        return T1

    def angle_to_next_alignment(
        self,
        geom_indx: int,
        current_geom_name: str,
        next_geom_name: str,
        T_init: sm.SE3 = None,
    ) -> float:
        """
        Compute the angle to align the TCP with a tangent direction between two geometries.

        Parameters:
        - geom_indx (int): Index of the geometry in the specification.
        - current_geom_name (str): Name of the current geometry.
        - next_geom_name (str): Name of the next geometry to align with.
        - T_init (sm.SE3, optional): Initial pose of the TCP. If None, fetches the current TCP pose.

        Returns:
        - float: The angle in radians for the alignment.
        """

        def tangent_points(cx, cy, r, px, py):
            """
            Compute the tangent points on a circle centered at (cx, cy) with radius r
            from an external point (px, py).

            Parameters
            ----------
            cx, cy : float
                Circle center coordinates.
            r : float
                Circle radius.
            px, py : float
                External point coordinates.

            Returns
            -------
            tuple
                Coordinates of the two tangent points as (t1, t2), where t1 and t2
                are (x, y) tuples.
            """
            # Translate the circle center and point to local coordinates
            c = np.array([cx, cy])
            p = np.array([px, py])
            local_p = p - c

            # Compute tangent points in local coordinates
            d0 = np.linalg.norm(local_p)
            if d0 <= r:
                raise ValueError(
                    "Point must be outside the circle for tangent lines to exist."
                )

            i1_local = (r**2 / d0**2) * local_p + (r / d0**2) * np.sqrt(
                d0**2 - r**2
            ) * np.array([-local_p[1], local_p[0]])
            i2_local = (r**2 / d0**2) * local_p - (r / d0**2) * np.sqrt(
                d0**2 - r**2
            ) * np.array([-local_p[1], local_p[0]])

            # Transform tangent points back to global coordinates
            t1 = i1_local + c
            t2 = i2_local + c

            return t1, t2

        def angle_between_vectors(u, v, direction="ccw"):
            """
            Compute the angle between two 2D vectors in the specified direction.

            Parameters
            ----------
            u, v : array-like
                Input 2D vectors.
            direction : str, optional
                Direction of the angle, either "ccw" (counterclockwise) or "cw" (clockwise).
                Default is "ccw".

            Returns
            -------
            float
                Angle in radians (0 to 2π) in the specified direction.

            Raises
            ------
            ValueError
                If the `direction` is not "ccw" or "cw".
            """
            # Normalize the vectors
            u = np.array(u) / np.linalg.norm(u)
            v = np.array(v) / np.linalg.norm(v)

            # Dot product and cross product (z-component only)
            dot = np.dot(u, v)
            cross = u[0] * v[1] - u[1] * v[0]

            # Compute the unsigned angle using the dot product
            angle = np.arccos(np.clip(dot, -1.0, 1.0))  # Clip for numerical stability

            # Adjust for the specified direction
            if direction == "ccw":
                if cross < 0:  # Negative cross-product means CW direction
                    angle = 2 * np.pi - angle
            elif direction == "cw":
                if cross > 0:  # Positive cross-product means CCW direction
                    angle = 2 * np.pi - angle
            else:
                raise ValueError(
                    f"Invalid direction: {direction}. Must be 'ccw' or 'cw'."
                )

            return angle

        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        T_w_geom = get_pose(self.model, self.data, current_geom_name, ObjType.GEOM)
        p = get_pose(self.model, self.data, next_geom_name, ObjType.GEOM).t
        c = T_w_geom.t

        d = np.linalg.norm(T_w_geom.t - T_w_tcp.t)

        t1, t2 = tangent_points(cx=c[0], cy=c[1], r=d, px=p[0], py=p[1])

        u = T_w_tcp.t[:2] - T_w_geom.t[:2]
        v1 = t1 - T_w_geom.t[:2]
        v2 = t2 - T_w_geom.t[:2]

        theta1 = angle_between_vectors(
            u, v1, direction=self.spec[f"o{geom_indx}"]["dir"]
        )
        theta2 = angle_between_vectors(
            u, v2, direction=self.spec[f"o{geom_indx}"]["dir"]
        )
        theta = -min(theta1, theta2)

        return theta

    def rotate_about(
        self,
        geom_name: str,
        dir: str,
        phi: float,
        n_steps: int = 20,
        T_init: sm.SE3 = None,
    ) -> sm.SE3:
        """
        Generate a pose rotating the TCP relative to a specified geometry.

        Parameters:
        - geom_name (str): Name of the geometry to rotate around.
        - dir (str): Direction of rotation, "ccw" for counterclockwise or "cw" for clockwise.
        - phi (float): Angle of rotation in radians.
        - n_steps (int): Number of steps in the trajectory.
        - T_init (sm.SE3, optional): Initial pose of the TCP. If None, fetches the current TCP pose.

        Returns:
        - sm.SE3: Rotated pose.
        """
        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)
        c_traj = [T_w_tcp]
        phi = -phi if dir == "ccw" else phi
        for phi in np.linspace(0, phi, n_steps):
            T_tcp_geom = T_w_geom.inv() * T_w_tcp
            Rz = sm.SE3.Rz(phi)
            T_w_tcp_new = T_w_geom * Rz * T_tcp_geom
            c_traj.append(T_w_tcp_new)
        return c_traj

    def opp1(self, h: float, geom_name: str, T_init: sm.SE3 = None) -> sm.SE3:
        """
        Generate a Cartesian pose from the end effector pose to a new pose above an obstacle.

        Parameters:
        - h (float): Height offset above the obstacle.
        - geom_name (str): Name of the obstacle geometry.
        - T_init (sm.SE3, optional): Initial pose of the TCP. If None, fetches the current TCP pose.

        Returns:
        - sm.SE3: Target pose above the obstacle.
        """
        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)
        Tz = sm.SE3.Tz(h + (T_w_geom.t[2] - T_w_tcp.t[2]))
        T_w_up = Tz @ T_w_tcp

        # T = ctraj(T_start=T_w_tcp, T_end=T_world_up, t=n_steps)
        # # T = rtb.ctraj(T0=T_world_tcp, T1=T_world_up, t=n_steps)
        # poses.extend(T)
        # return poses
        return T_w_up

    def opp2(
        self, theta: float, dir: str, geom_name: str, T_init: sm.SE3 = None
    ) -> sm.SE3:
        """
        Generate a Cartesian pose rotated theta raltive to some obstacle.

        Parameters:
        - theta (float): Angle of rotation in radians.
        - dir (str): Direction of rotation, "ccw" or "cw".
        - geom_name (str): Name of the obstacle geometry.
        - T_init (sm.SE3, optional): Initial pose of the TCP. If None, fetches the current TCP pose.

        Returns:
        - sm.SE3: Target pose after the rotation.
        """

        if dir == "ccw":
            theta = theta
        elif dir == "cw":
            theta = -theta
        else:
            raise ValueError(f'dir must be either "ccw" or "cw", but "{dir}" was given')

        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)

        # angle to slign axis with direction towards obstacle
        def get_phi() -> float:
            tcp_y = T_w_tcp.R[:2, 1]
            a = -tcp_y
            b = T_w_geom.t[:2] - T_w_tcp.t[:2]
            return angle(a, b)

        # the angle to correct the direction and the deviation angle theta
        theta = get_phi() + theta

        return T_w_tcp @ sm.SE3.Rz(theta)

    def opp3(
        self,
        delta: float,
        theta: float,
        dir: str,
        geom_name: str,
        T_init: sm.SE3 = None,
    ) -> sm.SE3:
        """
        Compute a target pose for the robot's end-effector based on positional and rotational offsets.

        Parameters
        ----------
        delta : float
            Distance offset from the current end-effector position towards the target geometry in the xy-plane.
        theta : float
            Angle of rotation around the z-axis (in radians).
        dir : str
            Direction of rotation ("ccw" for counterclockwise, any other value for clockwise).
        geom_name : str
            Name of the target geometry to compute the offset and rotation with respect to.
        T_init : sm.SE3, optional
            Initial pose of the end-effector. If None, the current pose is fetched from the robot, by default None.

        Returns
        -------
        sm.SE3
            The computed target pose in the world frame.
        """
        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        T_w_geom = get_pose(self.model, self.data, geom_name, ObjType.GEOM)

        t_tcp_geom = T_w_geom.t[:2] - T_w_tcp.t[:2]
        t_tcp_geom_norm = t_tcp_geom / np.linalg.norm(t_tcp_geom)
        t_tcp_target = (delta * (t_tcp_geom_norm)) + t_tcp_geom

        if dir == "ccw":
            theta = -theta

        t_tcp_target_rot = rotate_vector_2d(t_tcp_target, theta)

        p = T_w_tcp.t + np.append(t_tcp_target_rot, 0)

        rz = np.array([0, 0, -1])
        rx = np.append(
            (T_w_geom.t[:2] - p[:2]) / np.linalg.norm(T_w_geom.t[:2] - p[:2]), 0
        )
        if dir == "ccw":
            rx *= -1
        ry = np.cross(rz, rx) / np.linalg.norm(np.cross(rz, rx))

        R = np.column_stack((rx, ry, rz))

        if np.linalg.det(R) < 0:
            ry *= -1
            R = np.column_stack((rx, ry, rz))

        T_w_target = make_tf(pos=p, ori=R)

        return T_w_target

    def opp4(self, z_height: float, T_init: sm.SE3 = None) -> sm.SE3:
        """
        Compute a target pose for the robot's end-effector with a specified z-height.

        Parameters
        ----------
        z_height : float
            Desired height (z-coordinate) for the end-effector in the world frame.
        T_init : sm.SE3, optional
            Initial pose of the end-effector. If None, the current pose is fetched from the robot, by default None.

        Returns
        -------
        sm.SE3
            The computed target pose with the specified z-height.
        """
        if T_init is None:
            T_w_tcp = self.twof85.get_ee_pose()  # Current TCP pose in world frame
        else:
            T_w_tcp = T_init

        end_pos = T_w_tcp.t.copy()  # Make a copy of the translation vector
        end_pos[2] = z_height
        T_target = make_tf(pos=end_pos, ori=T_w_tcp.R)

        return T_target


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
