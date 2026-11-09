"""One pick-and-place cycle, written only against the Robot interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from main.config import CellConfig
from main.interfaces import Detection, Robot


@dataclass
class CycleResult:
    pick_ok: bool
    cycle_time_s: float


def grasp_yaw(part_yaw: float, tool_yaw: float) -> float:
    """The part is 180-deg symmetric: pick the grasp yaw closest to the current tool yaw."""
    cands = part_yaw + np.pi * np.arange(-2, 3)
    return float(cands[np.argmin(np.abs(cands - tool_yaw))])


def pick_and_place(robot: Robot, det: Detection, drop_xy: np.ndarray, cfg: CellConfig) -> CycleResult:
    t0 = robot.now()
    xy = np.array([det.x, det.y])
    yaw = grasp_yaw(det.theta, robot.tcp_pose()[1])

    robot.open_gripper()
    robot.move_to(np.array([*xy, cfg.approach_z]), yaw)
    robot.move_to(np.array([*xy, cfg.grasp_z]), yaw)
    robot.close_gripper()
    robot.move_to(np.array([*xy, cfg.safe_z]), yaw)

    pick_ok = robot.gripper_holding()
    if pick_ok:
        robot.move_to(np.array([*drop_xy, cfg.safe_z]), yaw)
        robot.move_to(np.array([*drop_xy, cfg.drop_z]), yaw)
        robot.open_gripper()
        robot.wait(cfg.settle_s)                     # let the part drop and settle
        robot.move_to(np.array([*drop_xy, cfg.safe_z]), yaw)
    else:
        robot.open_gripper()                          # failed pick: open and move on
        # TODO (PackML): raise a fault / go to HELD instead of silently moving on (OQ-04).

    return CycleResult(pick_ok=bool(pick_ok), cycle_time_s=round(robot.now() - t0, 3))
