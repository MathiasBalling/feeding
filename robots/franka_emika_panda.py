from collections import deque
from pathlib import Path
from typing import List, Tuple

import mujoco as mj
import mujoco.viewer
from dm_control import mjcf

# from ctrl.dmp_position import DMPPosition
from robots.base_robot import BaseRobot
from sims.base_sim import BaseSim
from utils.mj import (
    RobotInfo,
)


class FrankaPanda(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData) -> None:
        self._data = data
        self._model = model
        self.dt = self._model.opt.timestep

        self.home_qpos = self.model.key(f"{self.name}/home").qpos[:-1]
        self._task_queue = deque()

        self._info = RobotInfo(self._model, self.name)

        self.set_ctrl(self.home_qpos)

    def step(self) -> None:
        pass

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> str:
        return "panda"

    @property
    def actuator_values(self) -> List[float]:
        """
        Get the values of the actuators.

        Returns
        ----------
                List of actuator values.
        """
        return [self.data.actuator(an).ctrl[0] for an in self.info.actuator_names]


if __name__ == "__main__":

    class MjSim(BaseSim):
        def __init__(self):
            super().__init__()

            self._model, self._data = self.init()

            panda = FrankaPanda(self._model, self._data)

            panda.info.print()

        def init(self) -> Tuple[mj.MjModel, mj.MjData]:
            # root
            _HERE = Path(__file__).parent.parent
            # scene path
            _XML_SCENE = Path(_HERE / "scenes/empty.xml")
            scene = mjcf.from_path(_XML_SCENE)

            # shadow hand path
            _XML_FRANKA_PANDA = Path(_HERE / "assets/franka_emika_panda/panda.xml")
            franka_panda = mjcf.from_path(_XML_FRANKA_PANDA)

            scene.attach(franka_panda)

            m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
            d = mj.MjData(m)

            # step once to compute the poses of objects
            mj.mj_step(m, d)

            return m, d

        @property
        def data(self):
            return self._data

        @property
        def model(self):
            return self._model

    sim = MjSim()

    sim.run()
