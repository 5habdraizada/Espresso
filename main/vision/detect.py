"""Your vision pipeline: overhead image -> parts in the robot base frame. TODO: build this.

The same code runs on sim images (SimCamera) and real images (UsbCamera); only the
calibration differs. Suggested steps (start classical, as the brief says):

  1. Mask the tray region, convert to HSV.
  2. For each colour in cfg.colors: threshold its hue range -> mask -> findContours.
  3. Filter contours by area (reject noise and touching parts).
  4. minAreaRect -> pixel centre (u, v) and angle; turn the angle into the part's long-axis yaw.
  5. Inspection: look for the defect inside each part's rectangle (start with a dark-red
     threshold; later swap in an anomaly model trained on good parts, per the brief).
  6. calibration.pixel_to_robot(u, v, angle) -> Detection(x, y, theta, color, defective).

Check it on the sim with `python run.py --detector vision` and compare against
GroundTruthDetector to measure detection rate and angle error before going to the bench.
"""

from __future__ import annotations

import numpy as np

from main.calibration import PlanarCalibration
from main.config import CellConfig
from main.interfaces import Detection


class ColorDetector:
    def __init__(self, cfg: CellConfig, calibration: PlanarCalibration):
        self.cfg, self.calibration = cfg, calibration

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError("ColorDetector.detect is not written yet: see main/vision/detect.py")
