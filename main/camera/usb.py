"""Camera driver for a USB / UVC camera (the fixed overhead camera on the real cell)."""

from __future__ import annotations

import cv2
import numpy as np


class UsbCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # TODO: lock exposure and white balance once the lighting is fixed, so colours stay stable.

    def grab(self) -> np.ndarray:
        self.cap.grab()                       # drop a buffered frame so the image is current
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def close(self) -> None:
        self.cap.release()
