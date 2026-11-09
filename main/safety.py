"""Software safety layer shared by every robot driver.

Wrap any driver in GuardedRobot: every move target is clamped to the workspace box from
config/cell.toml before it reaches the driver, so the sim tests the exact same check the real
cell runs. The cobot controller's own safety limits still apply on top of this.
"""

from __future__ import annotations

import numpy as np

from main.config import CellConfig
from main.interfaces import Robot


def clamp_to_workspace(pos: np.ndarray, cfg: CellConfig) -> np.ndarray:
    return np.clip(np.asarray(pos, dtype=float), cfg.workspace[:, 0], cfg.workspace[:, 1])


class GuardedRobot:
    """A Robot that clamps move targets, then forwards everything to the wrapped driver."""

    def __init__(self, robot: Robot, cfg: CellConfig):
        self.robot, self.cfg = robot, cfg

    def move_to(self, pos: np.ndarray, yaw: float) -> None:
        safe = clamp_to_workspace(pos, self.cfg)
        if not np.allclose(safe, pos):
            print(f"[safety] target {np.round(pos, 3)} clamped to {np.round(safe, 3)}")
        self.robot.move_to(safe, yaw)

    def __getattr__(self, name):          # every other Robot method passes straight through
        return getattr(self.robot, name)
