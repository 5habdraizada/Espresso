"""Live view of a sim run: MuJoCo 3D viewer, optional camera window, real-time pacing.

Pass an instance as SimRobot's `on_frame`; it is called once per 1/60 s of sim time.
"""

from __future__ import annotations

import time

import numpy as np

from main.interfaces import StopRequested
from sim.cell import Cell


class SimDisplay:
    def __init__(self, cell: Cell, viewer=None, cameras: tuple[str, ...] = (), fps: int = 60):
        self.cell, self.viewer, self.cameras = cell, viewer, cameras
        self.frame_dt = 1 / fps
        self._frame = 0
        self._last = time.perf_counter()
        self._renderer = cell.make_renderer(640, 480) if cameras else None

    def __call__(self) -> None:
        if self._renderer is not None and self._frame % 2 == 0:
            import cv2
            views = [self.cell.grab_frame(self._renderer, cam) for cam in self.cameras]
            cv2.imshow(" | ".join(self.cameras) + " camera",
                       cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise StopRequested("camera window closed")
        self._frame += 1

        if self.viewer is not None:
            if not self.viewer.is_running():
                raise StopRequested("viewer closed")
            self.viewer.sync()
            time.sleep(max(0.0, self.frame_dt - (time.perf_counter() - self._last)))
        self._last = time.perf_counter()

    def close(self) -> None:
        if self._renderer is not None:
            import cv2
            self._renderer.close()
            cv2.destroyAllWindows()
