"""The contracts between the cell logic and the hardware.

The sequencer only ever talks to these. `robot/sim.py` and `camera/sim.py` implement them on
the MuJoCo cell; `robot/real.py` and `camera/usb.py` implement them on the real hardware.
If the sequencer needs something new from a robot, add it here first, then to every driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class StopRequested(Exception):
    """Raised by a driver when the cell must stop now (viewer closed, e-stop, protective stop)."""


@dataclass
class Detection:
    """One part found on the tray, already in the robot base frame."""
    x: float
    y: float
    theta: float                 # part yaw, radians
    color: str | None = None     # a key of CellConfig.colors, or None if unknown
    defective: bool = False
    score: float = 1.0           # detector / inspection confidence
    part_id: int | None = None   # sim only: which simulated part this is (for scoring)


class Robot(Protocol):
    """Every call blocks until the motion is finished (like ur_rtde's moveL)."""

    def move_to(self, pos: np.ndarray, yaw: float) -> None:
        """Move the TCP to `pos` (x, y, z) with the tool pointing down, rotated `yaw` about z."""

    def move_home(self) -> None: ...
    def open_gripper(self) -> None: ...
    def close_gripper(self) -> None: ...

    def gripper_holding(self) -> bool:
        """True if a part is in the gripper (vacuum sensor, finger width, ...)."""

    def tcp_pose(self) -> tuple[np.ndarray, float]:
        """Current TCP position (x, y, z) and tool yaw about z."""

    def wait(self, seconds: float) -> None: ...

    def now(self) -> float:
        """Seconds on the robot's clock (simulation time in the sim)."""

    def stop(self) -> None:
        """Stop motion and hold position (PackML HELD / STOPPED)."""


class Camera(Protocol):
    def grab(self) -> np.ndarray:
        """One BGR image (OpenCV channel order)."""


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Parts on the tray, in the robot base frame."""
