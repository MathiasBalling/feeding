
import time

from .robot import Robot
from .config import T_FOLLOW_INIT
import spatialmath as sm

def main() -> None:
    robot = Robot()

    while True:

        robot.hande.move_and_wait_for_pos(0,pct=True)

        key = input("continue?")
        if key.lower() == "n":
            quit()

        robot.moveJ_IK(T_FOLLOW_INIT)
        
        key = input("continue?")
        if key.lower() == "n":
            quit()

        robot.hande.move_and_wait_for_pos(0.9)

        key = input("continue?")
        if key.lower() == "n":
            quit()

        T_w_target = sm.SE3.Tx(0.3) @ robot.T_w_tcp

        T_base_target = robot.T_w_base.inv() @ T_w_target

        robot.moveL(T_base_target)

        key = input("continue?")
        if key.lower() == "n":
            quit()

if __name__ == "__main__":
    main()
