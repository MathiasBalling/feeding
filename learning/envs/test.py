from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
from brax.base import System
from brax.io import mjcf
from dm_control import mjcf as dm_mjcf


class TestEnv:
    """A pure physics simulation of a falling box using Brax."""

    def __init__(self):
        # Define a simple MJCF model with a falling box
        self.sys = self.init()

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

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        return mjcf.load_model(m)

    def reset(self, rng):
        if rng.ndim == 2:  # If batch dimension exists
            return jax.vmap(self._reset)(rng)  # Vectorized reset
        return self._reset(rng)  # Single environment reset

    def _reset(self, rng):
        """Single environment reset logic."""
        # Your reset logic goes here...

    def step(self, qp):
        """Advances the simulation by one step."""
        qp, _ = self.sys.step(
            qp, jnp.zeros((self.sys.act_size,))
        )  # No action, just physics
        return qp
