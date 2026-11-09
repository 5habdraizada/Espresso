"""The cell loop: grab image -> detect -> choose bin -> pick and place -> log -> repeat.

Runs the same way on the sim and on the real cell; only the objects passed in differ.
TODO (PackML): this is the body of the EXECUTE state. Once packml.py exists, the state machine
calls one cycle at a time so it can go to HELD / STOPPED between (or during) cycles.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from main.config import CellConfig
from main.interfaces import Camera, Detection, Detector, Robot
from main.logger import CycleLogger
from main.sequencer.cycle import pick_and_place
from main.sequencer.sorting import choose_bin


def run_cycles(robot: Robot, camera: Camera, detector: Detector, cfg: CellConfig,
               n_cycles: int, logger: CycleLogger, rng: np.random.Generator,
               verify: Callable[[Detection, str], bool] | None = None,
               refill: Callable[[], None] | None = None) -> None:
    """Run `n_cycles` cycles.

    verify: optional check that the part landed in its bin (sim: ground truth; real: TODO,
            e.g. a bin camera). Without it, placed_ok is logged as unknown.
    refill: called when the tray is empty (sim: re-scatter parts). Without it the run ends.
    """
    robot.move_home()
    while logger.count < n_cycles:
        detections = detector.detect(camera.grab())
        if not detections:
            if refill is None:
                print("Tray empty.")
                break
            refill()
            continue

        det = detections[0]                          # TODO: pick order (nearest, most confident, ...)
        bin_name = choose_bin(det)
        drop = cfg.bins[bin_name].xy + rng.uniform(-cfg.drop_jitter, cfg.drop_jitter, 2)
        result = pick_and_place(robot, det, drop, cfg)
        placed_ok = verify(det, bin_name) if verify is not None else None
        robot.move_home()                            # clear the overhead camera's view
        logger.log(det, bin_name, result, placed_ok)
    robot.move_home()
