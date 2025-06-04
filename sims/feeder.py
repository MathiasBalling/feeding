from pathlib import Path

import glfw
import mujoco as mj
from dm_control import mjcf

from sims import BaseSim
from sims.base_sim import SimSync
import numpy as np
import spatialmath as sm
from utils.mj import ObjType, get_contact_states, get_number_of, get_pose, set_pose
import random


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()
        self.tasks = [self.spin]

    def init(self):
        # root
        _MJ_SIM = Path(__file__).parent.parent

        # basic scene path
        _XML_SCENE = Path(_MJ_SIM / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # path to import feeder
        _XML_FEEDER = Path(_MJ_SIM / "assets/props/feeder.xml")
        feeder = mjcf.from_path(_XML_FEEDER)
        self._feeder = feeder

        # find the feeder body and attach it to the basic scene
        feeder_body = feeder.worldbody.find("body", "feeder")
        feeder_body.pos = [0.0, 0, 0.1]  # set position
        scene.attach(feeder)

        # add the parts to the xml scene file
        numberOfParts = 1
        for i in range(numberOfParts):
            _XML_PART = Path(_MJ_SIM / "assets/props/part.xml")
            part = mjcf.from_path(_XML_PART)

            ssd = 0.05  # spawn seperation distance, MODIFY IF NEEDED
            part_body = part.worldbody.find("body", "part")
            part_body.pos = [-0.6 - (i * ssd), -0.04, 0.15]

            # random RPY initializing of the part, MODIFY IF NEEDED
            roll = random.uniform(0, 3.14)
            pitch = random.uniform(0, 3.14)
            yaw = random.uniform(0, 3.14)

            # convert to quaternions
            qx, qy, qz, qw = self.getQuaternionFromEuler(roll, pitch, yaw)
            part_body.quat = [qx, qy, qz, qw]

            # attach part to scene
            part_attach = scene.attach(part)
            part_attach.add("joint", type="free")

        # load the scene
        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        self._runSim = True
        return m, d

    def spin(self, ss: SimSync):  # defines the simulation stepping
        t = 0  # start time
        omega = 100  # vibration frequency
        A = 0.000179  # vibration amplitude
        vibAngle = 20 / 180 * np.pi  # vibration angle in radians

        dt = (
            self.model.opt.timestep * 4
        )  # set time step of the simulation for computation of next vibrations position (handled by controller)

        while self._runSim:
            vzAmp = (
                np.sin(vibAngle) * A
            )  # the forward motion composant of the full motion
            vxAmp = np.sqrt(
                np.power(A, 2.0) - np.power(np.sin(vibAngle) * A, 2.0)
            )  # the upwards motion composant of the full motion

            dz = vzAmp * np.sin(omega * t)
            dx = vxAmp * np.sin(omega * t)
            self.data.actuator("feeder/x").ctrl = dx
            self.data.actuator("feeder/z").ctrl = dz

            # print(self.data.joint("feeder/x").qpos)
            # print(self.data.joint("feeder/z").qpos)

            ss.step()
            t += dt

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            self._runSim = False
            print("You pressed space and can now quit")
        if key is glfw.KEY_R:
            pass  # TODO: reset function not yet implemented

    def getQuaternionFromEuler(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)

        return qx, qy, qz, qw


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
