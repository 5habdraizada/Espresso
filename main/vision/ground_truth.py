"""Sim-only stand-in for the vision pipeline: reads part poses straight from the simulator.

Use it to test the rest of the cell before your detector works, and as the reference to
score your detector against (detection rate, position and angle error: OQ-01).
"""

from __future__ import annotations

import numpy as np

from main.interfaces import Detection
from sim.cell import Cell


class GroundTruthDetector:
    def __init__(self, cell: Cell):
        self.cell = cell

    def detect(self, image: np.ndarray | None = None) -> list[Detection]:
        cfg = self.cell.cfg
        found = []
        for i in range(self.cell.n_parts):
            pos, yaw = self.cell.part_pose(i)
            on_tray = (np.all(np.abs(pos[:2] - cfg.tray_center) < cfg.tray_half)
                       and pos[2] < cfg.tray_top_z + 0.03)
            if on_tray:
                found.append(Detection(x=float(pos[0]), y=float(pos[1]), theta=yaw,
                                       color=self.cell.colors[i],
                                       defective=bool(self.cell.defective[i]), part_id=i))
        return found
