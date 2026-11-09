"""Run the pick-and-place cell on the simulator or the real cobot.

    py run.py                                   # sim: 3D viewer + wrist camera window
    py run.py --overhead                        # also show the overhead camera
    py run.py --no-camera                       # 3D viewer only
    py run.py --headless --parts 10 --cycles 200 --seed 1   # fast validation-style run
    py run.py --detector vision                 # sim with YOUR vision pipeline instead of ground truth
    py run.py --backend real --host 192.168.0.10    # the physical cobot (once RealRobot exists)

The sequencer, sorting rule, safety clamp and logger are the same objects on both backends;
only the robot, camera and (in the sim) the ground-truth hooks change.
"""

from __future__ import annotations

import argparse
import dataclasses
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from main.calibration import PlanarCalibration
from main.config import DEFAULT_PATH, load_config
from main.interfaces import StopRequested
from main.logger import CycleLogger
from main.safety import GuardedRobot
from main.sequencer.runner import run_cycles
from main.vision.detect import ColorDetector

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--config", default=str(DEFAULT_PATH))
    ap.add_argument("--cycles", type=int, default=None, help="cycles to run (default: one tray)")
    ap.add_argument("--speed", type=float, default=None, help="override motion.speed from the config")
    ap.add_argument("--calibration", default=None, help="calibration file (default config/calibration_<backend>.npz)")
    sim = ap.add_argument_group("sim only")
    sim.add_argument("--parts", type=int, default=12, help="parts on the tray (default 12)")
    sim.add_argument("--defect-rate", type=float, default=0.25)
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--detector", choices=["truth", "vision"], default="truth")
    sim.add_argument("--headless", action="store_true", help="no viewer, run as fast as possible")
    sim.add_argument("--no-camera", action="store_true", help="don't open the wrist camera window")
    sim.add_argument("--overhead", action="store_true", help="show the overhead camera too")
    real = ap.add_argument_group("real only")
    real.add_argument("--host", default=None, help="cobot controller IP address")
    real.add_argument("--camera-index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.speed is not None:
        cfg = dataclasses.replace(cfg, speed=args.speed)
    calib_path = args.calibration or ROOT / "config" / f"calibration_{args.backend}.npz"
    logger = CycleLogger(args.backend)

    if args.backend == "sim":
        run_sim(args, cfg, calib_path, logger)
    else:
        run_real(args, cfg, calib_path, logger)
    logger.summarize()


def run_sim(args, cfg, calib_path, logger) -> None:
    from main.camera.sim import SimCamera
    from main.robot.sim import SimRobot
    from main.vision.ground_truth import GroundTruthDetector
    from sim.cell import Cell
    from sim.display import SimDisplay

    cell = Cell(cfg, n_parts=args.parts, defect_rate=args.defect_rate, seed=args.seed)
    camera = SimCamera(cell, "overhead")
    detector = (GroundTruthDetector(cell) if args.detector == "truth"
                else ColorDetector(cfg, PlanarCalibration.load(calib_path)))

    if args.headless:
        viewer_ctx = nullcontext(None)
    else:
        import mujoco.viewer
        viewer_ctx = mujoco.viewer.launch_passive(cell.model, cell.data, show_left_ui=False)

    with viewer_ctx as viewer:
        display = None
        if viewer is not None:
            viewer.cam.lookat[:] = [0.3, 0.0, 0.15]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 145, -28
            cams = () if args.no_camera else (("wrist", "overhead") if args.overhead else ("wrist",))
            display = SimDisplay(cell, viewer, cams)
        sim_robot = SimRobot(cell, on_frame=display)
        try:
            run_cycles(GuardedRobot(sim_robot, cfg), camera, detector, cfg,
                       n_cycles=args.cycles or args.parts, logger=logger, rng=cell.rng,
                       verify=lambda det, b: sim_robot.part_in_bin(det.part_id, b),
                       refill=sim_robot.refill_tray)
        except StopRequested as e:
            print(f"Stopped: {e}")
        finally:
            camera.close()
            if display is not None:
                display.close()


def run_real(args, cfg, calib_path, logger) -> None:
    from main.camera.usb import UsbCamera
    from main.robot.real import RealRobot

    if not args.host:
        raise SystemExit("--host (the cobot controller's IP) is required for --backend real")
    robot = GuardedRobot(RealRobot(cfg, args.host), cfg)
    camera = UsbCamera(args.camera_index)
    detector = ColorDetector(cfg, PlanarCalibration.load(calib_path))
    try:
        run_cycles(robot, camera, detector, cfg, n_cycles=args.cycles or 10**9,
                   logger=logger, rng=np.random.default_rng())
    except (StopRequested, KeyboardInterrupt) as e:
        robot.stop()
        print(f"Stopped: {e!r}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
