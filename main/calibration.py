"""Pixel -> robot base frame for the fixed overhead camera. TODO: build this.

Picks are planar (all parts lie on the tray surface), so a single homography from image
pixels to tray-plane (x, y) in the robot base frame is enough to start with:

  1. calibrateCamera with a ChArUco board -> intrinsics + distortion; undistort pixels first.
  2. Collect >= 4 point pairs (more is better): jog the TCP to marks on the tray, read the
     robot (x, y), and find the same marks in the image (u, v).
  3. cv2.findHomography(pixel_pts, robot_pts) -> H. Save H and the intrinsics to a file and
     record its hash (IQ-01).
  4. Angles: map two points along the part's axis through H and take atan2 of the difference.

In the sim you can check your result against the truth: the overhead camera is at
(0.45, 0, 0.75) looking straight down, fovy 35 deg, image right = world -y, image up = world +x.
"""

from __future__ import annotations

from pathlib import Path


class PlanarCalibration:
    @classmethod
    def load(cls, path: Path | str) -> "PlanarCalibration":
        raise NotImplementedError("PlanarCalibration is not written yet: see main/calibration.py")

    def pixel_to_robot(self, u: float, v: float, angle_px: float) -> tuple[float, float, float]:
        """(u, v) pixel and image-frame angle -> (x, y, yaw) in the robot base frame."""
        raise NotImplementedError
