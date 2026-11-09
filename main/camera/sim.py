"""Camera driver for the MuJoCo cell. Implements `main.interfaces.Camera`."""

from __future__ import annotations

import cv2
import numpy as np

from sim.cell import Cell


class SimCamera:
    def __init__(self, cell: Cell, name: str = "overhead", width: int = 640, height: int = 480):
        self.cell, self.name = cell, name
        self.width, self.height = width, height
        self._renderer = None

    def grab(self) -> np.ndarray:
        if self._renderer is None:            # create lazily, on the thread that renders
            self._renderer = self.cell.make_renderer(self.width, self.height)
        rgb = self.cell.grab_frame(self._renderer, self.name)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
