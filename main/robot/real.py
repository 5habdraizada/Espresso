"""Robot driver for the physical cobot. Implements `main.interfaces.Robot`.

TODO: fill this in once the cobot model is confirmed (project brief, week 1).
On a Universal Robots arm with the External Control URCap this is a thin wrapper around
ur_rtde (`pip install ur_rtde`):

    rtde_control.RTDEControlInterface(host)  -> moveL / moveJ / stopL / speed scaling
    rtde_receive.RTDEReceiveInterface(host)  -> getActualTCPPose, safety/protective-stop status
    gripper: the tool digital outputs (vacuum) or the gripper's own URCap / socket API

Rules from the brief that live here:
  * keep the controller's own safety config on; run at <= 25% speed while developing
  * the shared workspace clamp (main.safety) runs before every move_to, but do not remove
    the controller's safety planes because of it
  * a protective stop or e-stop must raise StopRequested so the sequencer goes to HELD/STOPPED

Pose convention: the interface passes (x, y, z) + a yaw about z with the tool pointing down.
UR wants an axis-angle pose [x, y, z, rx, ry, rz]: build the rotation "flip 180 deg about x,
then yaw about z" (the same as sim.cell.tool_down_quat) and convert it with
scipy.spatial.transform.Rotation(...).as_rotvec().
"""

from __future__ import annotations

import numpy as np

from main.config import CellConfig


class RealRobot:
    def __init__(self, cfg: CellConfig, host: str):
        self.cfg, self.host = cfg, host
        raise NotImplementedError("RealRobot is not written yet: see main/robot/real.py")

    def move_to(self, pos: np.ndarray, yaw: float) -> None:
        raise NotImplementedError

    def move_home(self) -> None:
        raise NotImplementedError

    def open_gripper(self) -> None:
        raise NotImplementedError

    def close_gripper(self) -> None:
        raise NotImplementedError

    def gripper_holding(self) -> bool:
        raise NotImplementedError

    def tcp_pose(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def wait(self, seconds: float) -> None:
        raise NotImplementedError

    def now(self) -> float:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
