"""Robot driver for the MuJoCo cell. Implements `main.interfaces.Robot`."""

from __future__ import annotations

from typing import Callable, Iterator

import mujoco
import numpy as np

from sim.cell import Cell


class SimRobot:
    FPS = 60

    def __init__(self, cell: Cell, on_frame: Callable[[], None] | None = None):
        """`on_frame` is called every 1/60 s of sim time (viewer sync, camera window, pacing).
        It may raise StopRequested to end the run."""
        self.cell = cell
        self.on_frame = on_frame
        self.steps_per_frame = max(1, round((1 / self.FPS) / cell.model.opt.timestep))
        self._steps = 0

    def _run(self, motion: Iterator[None]) -> None:
        """Step the physics once per yield of a Cell motion generator, until it finishes."""
        m, d = self.cell.model, self.cell.data
        for _ in motion:
            mujoco.mj_step(m, d)
            self._steps += 1
            if self.on_frame is not None and self._steps % self.steps_per_frame == 0:
                self.on_frame()

    # ---------------------------------------------------------- Robot interface

    def move_to(self, pos: np.ndarray, yaw: float) -> None:
        self._run(self.cell.move_to(pos, yaw))

    def move_home(self) -> None:
        self._run(self.cell.move_joints(self.cell.home_q))

    def open_gripper(self) -> None:
        self._run(self.cell.open_gripper())

    def close_gripper(self) -> None:
        self._run(self.cell.close_gripper())

    def gripper_holding(self) -> bool:
        return self.cell.held_part() is not None

    def tcp_pose(self) -> tuple[np.ndarray, float]:
        return self.cell.tcp_pos(), self.cell.tool_yaw()

    def wait(self, seconds: float) -> None:
        self._run(self.cell.wait(seconds))

    def now(self) -> float:
        return self.cell.data.time

    def stop(self) -> None:
        self.cell.hold()

    # ------------------------------------------------- sim-only (ground truth)

    def part_in_bin(self, part_id: int, bin_name: str) -> bool:
        """Did simulated part `part_id` end up inside bin `bin_name`? Used to score sim runs."""
        cfg = self.cell.cfg
        pos = self.cell.part_pose(part_id)[0]
        return bool(np.all(np.abs(pos[:2] - cfg.bins[bin_name].xy) < cfg.bin_inner_half)
                    and pos[2] < 0.08)

    def refill_tray(self) -> None:
        """Re-scatter all parts on the tray (the sim's 'operator loads a new tray')."""
        self.cell.reset()
