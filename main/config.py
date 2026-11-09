"""Load config/cell.toml into a typed, read-only CellConfig."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "cell.toml"


@dataclass(frozen=True)
class Bin:
    name: str
    xy: np.ndarray
    rgba: tuple[float, ...]


@dataclass(frozen=True)
class CellConfig:
    tray_center: np.ndarray
    tray_half: float
    tray_usable_half: float
    tray_top_z: float
    part_half: np.ndarray
    part_spacing: float
    workspace: np.ndarray          # 3 x 2: [min, max] for x, y, z
    home: np.ndarray
    safe_z: float
    approach_z: float
    grasp_z: float
    drop_z: float
    drop_jitter: float
    settle_s: float
    speed: float
    colors: dict[str, tuple[float, ...]]
    bin_inner_half: float
    bins: dict[str, Bin]


def load_config(path: Path | str = DEFAULT_PATH) -> CellConfig:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    tray, part, ws, mo, bins = c["tray"], c["part"], c["workspace"], c["motion"], dict(c["bins"])
    inner_half = bins.pop("inner_half")
    cfg = CellConfig(
        tray_center=np.array(tray["center"], float), tray_half=tray["half"],
        tray_usable_half=tray["usable_half"], tray_top_z=tray["top_z"],
        part_half=np.array(part["half"], float), part_spacing=part["spacing"],
        workspace=np.array([ws["x"], ws["y"], ws["z"]], float),
        home=np.array(mo["home"], float), safe_z=mo["safe_z"], approach_z=mo["approach_z"],
        grasp_z=mo["grasp_z"], drop_z=mo["drop_z"], drop_jitter=mo["drop_jitter"],
        settle_s=mo["settle_s"], speed=mo["speed"],
        colors={k: tuple(v) for k, v in c["colors"].items()},
        bin_inner_half=inner_half,
        bins={k: Bin(k, np.array(v["xy"], float), tuple(v["rgba"])) for k, v in bins.items()},
    )
    missing = set(cfg.colors) - set(cfg.bins)
    if missing:
        raise ValueError(f"{path}: no bin for part colour(s) {sorted(missing)}")
    if "reject" not in cfg.bins:
        raise ValueError(f"{path}: a [bins.reject] entry is required")
    return cfg
