"""MuJoCo model of the pick-and-place cell: scene building, IK, motion and the 2-finger gripper.

This is the simulator's internals. The cell logic never imports it directly: it goes through
`main.robot.sim.SimRobot` and `main.camera.sim.SimCamera`, which wrap this class.

Motions here are generators that set actuator targets and `yield` once per physics step;
SimRobot steps the physics for each yield, so its methods block like a real robot driver.
Layout and motion numbers come from config/cell.toml (shared with the real cell).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from main.config import CellConfig

HERE = Path(__file__).resolve().parent

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

# Gripper targets: half-opening in metres (finger pad to tool axis). 5 cm open, fully closed.
GRIP_OPEN = 0.025
GRIP_CLOSED = 0.0

# Seed configuration: elbow up, tool pointing down, above the tray.
SEED_Q = np.array([0.0, 0.5, 1.2, np.pi - 1.7, 0.0, 0.0])


def tool_down_quat(yaw: float) -> np.ndarray:
    """Orientation with the tool axis pointing straight down, rotated `yaw` about world z."""
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    q_flip = np.array([0.0, 1.0, 0.0, 0.0])   # 180 deg about x: tool z -> world -z
    out = np.empty(4)
    mujoco.mju_mulQuat(out, q_yaw, q_flip)
    return out


def _rgba(c) -> str:
    return " ".join(f"{v:g}" for v in c)


def _bin_xml(cfg: CellConfig, name: str, xy: np.ndarray, rgba) -> str:
    """Open-top box: floor plus four 6 cm walls, inner size 2 * bin_inner_half."""
    o, w, h = cfg.bin_inner_half + 0.01, 0.005, 0.03   # outer half-size, wall half-thickness, wall half-height
    c = o - w
    geoms = [f'size="{o} {o} 0.003" pos="0 0 0.003"',
             f'size="{o} {w} {h}" pos="0 {c} {h}"', f'size="{o} {w} {h}" pos="0 {-c} {h}"',
             f'size="{w} {o} {h}" pos="{c} 0 {h}"', f'size="{w} {o} {h}" pos="{-c} 0 {h}"']
    body = "".join(f'<geom class="fixture" rgba="{_rgba(rgba)}" {g}/>' for g in geoms)
    return f'<body name="bin_{name}" pos="{xy[0]:g} {xy[1]:g} 0">{body}</body>'


def _part_xml(cfg: CellConfig, i: int, color: str, defective: bool) -> str:
    hx, hy, hz = cfg.part_half
    marker = ""
    if defective:
        # Seeded defect: a dark-red flash patch on the top face (what the inspection should catch).
        marker = (f'<geom type="box" size="0.007 0.006 0.0006" pos="0.012 0 {hz:.4f}" '
                  f'rgba="0.65 0.08 0.05 1" contype="0" conaffinity="0" mass="0"/>')
    return (f'<body name="part{i}" pos="{0.3 + 0.01 * i:.3f} 0 0.5">'
            f'<freejoint name="part{i}_free"/>'
            f'<geom name="part{i}_geom" type="box" size="{hx} {hy} {hz}" mass="0.015" '
            f'rgba="{_rgba(cfg.colors[color])}" friction="1.0 0.01 0.001"/>'
            f'{marker}</body>')


def build_xml(cfg: CellConfig, colors: list[str], defective: np.ndarray) -> str:
    xml = (HERE / "scene.xml").read_text(encoding="utf-8")
    bins = "\n    ".join(_bin_xml(cfg, b.name, b.xy, b.rgba) for b in cfg.bins.values())
    parts = "\n    ".join(_part_xml(cfg, i, c, bool(d))
                          for i, (c, d) in enumerate(zip(colors, defective)))
    return xml.replace("<!--BINS-->", bins).replace("<!--PARTS-->", parts)


class Cell:
    def __init__(self, cfg: CellConfig, n_parts: int = 6, defect_rate: float = 0.25,
                 seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n_parts = n_parts
        self.colors = [str(c) for c in self.rng.choice(list(cfg.colors), n_parts)]
        self.defective = self.rng.random(n_parts) < defect_rate
        self.model = mujoco.MjModel.from_xml_string(build_xml(cfg, self.colors, self.defective))
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)

        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dofadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qlo = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS], 0]
        self.qhi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS], 1]
        self.tcp = m.site("tcp").id
        self.arm_act = np.array([m.actuator(f"a{k}").id for k in range(1, 7)])
        self.part_body = np.array([m.body(f"part{i}").id for i in range(n_parts)])
        self.part_qadr = np.array([m.joint(f"part{i}_free").qposadr[0] for i in range(n_parts)])
        self.part_geom = {m.geom(f"part{i}_geom").id: i for i in range(n_parts)}

        self.grip_act = m.actuator("grip").id
        self.finger_qadr = np.array([m.joint(j).qposadr[0] for j in ("finger_l", "finger_r")])
        self.finger_dofadr = np.array([m.joint(j).dofadr[0] for j in ("finger_l", "finger_r")])
        self.finger_geom = (m.geom("finger_l").id, m.geom("finger_r").id)

        self.vmax = 1.2 * cfg.speed   # joint speed limit, rad/s
        self.dt = m.opt.timestep
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        """Re-scatter the parts on the tray and put the arm at home."""
        mujoco.mj_resetData(self.model, self.data)
        self._scatter_parts()
        home = self.solve_ik(self.cfg.home, tool_down_quat(0.0), SEED_Q)
        self.data.qpos[self.qadr] = home
        self.data.ctrl[self.arm_act] = home
        self.home_q = home
        self.data.qpos[self.finger_qadr] = GRIP_OPEN
        self.data.ctrl[self.grip_act] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)

    def _random_layout(self) -> list[np.ndarray]:
        """Random part positions, at least part_spacing apart so a gripper fits between them."""
        c = self.cfg
        for _ in range(500):                       # retry whole layouts: greedy fills get stuck
            placed: list[np.ndarray] = []
            for _ in range(self.n_parts):
                for _ in range(200):
                    xy = c.tray_center + self.rng.uniform(-c.tray_usable_half, c.tray_usable_half, 2)
                    if all(np.linalg.norm(xy - p) > c.part_spacing for p in placed):
                        placed.append(xy)
                        break
                else:
                    break
            if len(placed) == self.n_parts:
                return placed
        raise RuntimeError(f"Could not fit {self.n_parts} parts {c.part_spacing * 100:g} cm apart "
                           f"on the tray; use fewer parts or lower part.spacing in cell.toml.")

    def _scatter_parts(self) -> None:
        z = self.cfg.tray_top_z + self.cfg.part_half[2] + 0.0005
        for i, xy in enumerate(self._random_layout()):
            yaw = self.rng.uniform(-np.pi, np.pi)
            a = self.part_qadr[i]
            self.data.qpos[a:a + 3] = [xy[0], xy[1], z]
            self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # -------------------------------------------------------------- queries

    def part_pose(self, i: int) -> tuple[np.ndarray, float]:
        """Ground-truth (position, yaw) of part i."""
        pos = self.data.xpos[self.part_body[i]].copy()
        mat = self.data.xmat[self.part_body[i]].reshape(3, 3)
        return pos, float(np.arctan2(mat[1, 0], mat[0, 0]))

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    def tool_yaw(self) -> float:
        x_axis = self.data.site_xmat[self.tcp].reshape(3, 3)[:, 0]
        return float(np.arctan2(x_axis[1], x_axis[0]))

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    # ------------------------------------------------------------------- IK

    def solve_ik(self, pos: np.ndarray, quat: np.ndarray, q_seed: np.ndarray,
                 iters: int = 300, tol: float = 1e-5) -> np.ndarray:
        """Damped-least-squares IK for the TCP site. Raises if it does not converge."""
        m, d = self.model, self._ik
        d.qpos[:] = self.data.qpos
        q = np.array(q_seed, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cur_q, neg_q, err_q = np.empty(4), np.empty(4), np.empty(4)
        err_r = np.empty(3)
        lam = 1e-2
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = pos - d.site_xpos[self.tcp]
            mujoco.mju_mat2Quat(cur_q, d.site_xmat[self.tcp])
            mujoco.mju_negQuat(neg_q, cur_q)
            mujoco.mju_mulQuat(err_q, quat, neg_q)
            mujoco.mju_quat2Vel(err_r, err_q, 1.0)
            err = np.concatenate([err_p, err_r])
            if err_p @ err_p < tol ** 2 and err_r @ err_r < (10 * tol) ** 2:
                return q
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.qlo, self.qhi)
        raise RuntimeError(f"IK did not converge for target {np.round(pos, 3)}")

    # --------------------------------------------------------------- motion

    def move_to(self, pos, yaw: float = 0.0, settle_tol: float = 0.002):
        """Joint-space min-jerk move to a tool-down TCP pose. Generator: yields once per step."""
        # Last line of defence; the shared clamp in main.safety runs before this.
        target = np.clip(np.asarray(pos, dtype=float), self.cfg.workspace[:, 0], self.cfg.workspace[:, 1])
        q0 = self.data.ctrl[self.arm_act].copy()
        q1 = self.solve_ik(target, tool_down_quat(yaw), q0)
        yield from self.move_joints(q1)
        # Let the servos settle onto the target.
        for _ in range(int(0.5 / self.dt)):
            if np.linalg.norm(self.tcp_pos() - target) < settle_tol:
                break
            yield

    def move_joints(self, q1: np.ndarray):
        q0 = self.data.ctrl[self.arm_act].copy()
        span = np.max(np.abs(q1 - q0))
        # Min-jerk peak speed is 1.875x the average, so size the duration to respect vmax.
        T = max(0.25, 1.875 * span / self.vmax)
        n = max(1, int(T / self.dt))
        for k in range(1, n + 1):
            t = k / n
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            self.data.ctrl[self.arm_act] = q0 + s * (q1 - q0)
            yield

    def wait(self, seconds: float):
        for _ in range(int(seconds / self.dt)):
            yield

    def hold(self) -> None:
        """Freeze the arm where it is."""
        self.data.ctrl[self.arm_act] = self.arm_q()

    # -------------------------------------------------------------- gripper

    # Generators like the arm motions: set the grip actuator's target, then yield once per
    # physics step until the fingers stop moving (blocked by a part, or at the end stop).

    def finger_opening(self) -> float:
        """Current distance between the finger pads, metres."""
        return float(self.data.qpos[self.finger_qadr].sum())

    def _actuate_gripper(self, target: float, timeout: float = 0.6):
        self.data.ctrl[self.grip_act] = target
        min_steps, still, need_still = int(0.06 / self.dt), 0, int(0.03 / self.dt)
        for k in range(int(timeout / self.dt)):
            yield
            moving = np.max(np.abs(self.data.qvel[self.finger_dofadr])) > 2e-3
            still = 0 if moving else still + 1
            if k >= min_steps and still >= need_still:
                break

    def open_gripper(self):
        yield from self._actuate_gripper(GRIP_OPEN)

    def close_gripper(self):
        yield from self._actuate_gripper(GRIP_CLOSED)

    def held_part(self) -> int | None:
        """Index of the part touching both finger pads, if any (a real grasp, not a nudge)."""
        touching: list[set[int]] = [set(), set()]
        for c in self.data.contact[:self.data.ncon]:
            for f, fg in enumerate(self.finger_geom):
                other = c.geom2 if c.geom1 == fg else c.geom1 if c.geom2 == fg else None
                if other in self.part_geom:
                    touching[f].add(self.part_geom[other])
        both = touching[0] & touching[1]
        return min(both) if both else None

    # --------------------------------------------------------------- camera

    def make_renderer(self, width: int = 640, height: int = 480) -> mujoco.Renderer:
        return mujoco.Renderer(self.model, height=height, width=width)

    def grab_frame(self, renderer: mujoco.Renderer, camera: str = "overhead") -> np.ndarray:
        """RGB image from a named camera ("overhead" or "wrist")."""
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()
