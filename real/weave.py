

from pathlib import Path
import pickle

import numpy as np
from scripts.config import T_LIFT_INIT, T_FOLLOW_INIT, T_W_BOARD_CORNER
from utils.sm import make_tf
from .robot import Robot
import spatialmath as sm
from .utils import sm_print

def main() -> None:
    robot = Robot()

    traj = None

    q_home = [-0.12301284471620733, -1.8627401791014613, -1.523547887802124, -1.2507248383811493, 1.5750060081481934, -1.4767621199237269]

    # load in trajectory
    with open("data/checkpoints.pickle","rb") as f:
        traj: dict = pickle.load(f)

    T_base_target_traj = []
    names = []
    # get names and frames
    for frame_name, frame in traj.items():
        names.append(frame_name)
        T_base_target_traj.append(robot.T_w_base.inv() @ frame)

    robot.hande.move_and_wait_for_pos(0)

    key = input("take inital pose?")
    if key.lower() == "n":
            quit()
    # move up
    robot.moveJ(q_home)

    robot.moveL(T_base_target_traj[0], speed=0.1, acc=0.1)
    # print(3)
    key = input("close gripper?")
    if key.lower() == "n":
            quit()
    robot.hande.move_and_wait_for_pos(0.88)
    # robot.hande.move_and_wait_for_pos(0.9)

    key = input("begin?")
    if key.lower() == "n":
        quit()

    for i, Ti in enumerate(T_base_target_traj):
        # key = input(f"continue?, {i} / {len(T_base_target_traj)}, name = {names[i]}")
        # if key.lower() == "n":
        #     quit()
        # print(np.rad2deg(robot.get_q()))
        robot.moveL(Ti, speed=0.1, acc=0.1)
    robot.hande.move_and_wait_for_pos(0)

if __name__ == "__main__":
    main()