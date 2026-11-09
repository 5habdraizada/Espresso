"""Per-cycle log. Prints each cycle and writes a CSV at the end.

TODO (brief, weeks 6-7): switch to SQLite, one row per cycle, adding a timestamp, the
anomaly score and the path of the saved image (traceability target: 100% of cycles).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from main.interfaces import Detection
from main.sequencer.cycle import CycleResult

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class CycleLogger:
    def __init__(self, backend: str):
        self.backend = backend
        self.rows: list[dict] = []

    @property
    def count(self) -> int:
        return len(self.rows)

    def log(self, det: Detection, bin_name: str, result: CycleResult, placed_ok: bool | None) -> None:
        row = {
            "cycle": self.count + 1, "part": det.part_id,
            "x": round(det.x, 4), "y": round(det.y, 4),
            "theta_deg": round(float(np.degrees(det.theta)), 1),
            "color": det.color, "defective": det.defective, "bin": bin_name,
            "pick_ok": result.pick_ok, "placed_ok": placed_ok,
            "cycle_time_s": result.cycle_time_s,
        }
        self.rows.append(row)
        ok = row["pick_ok"] and placed_ok is not False
        print(f"[{'ok ' if ok else 'FAIL'}] cycle {row['cycle']:>3}  part {row['part']}  "
              f"{str(row['color']):<6} -> {bin_name:<6} "
              f"({row['x']:.3f}, {row['y']:.3f}, {row['theta_deg']:>6.1f} deg)  "
              f"{row['cycle_time_s']:.2f} s")

    def summarize(self) -> Path | None:
        if not self.rows:
            return None
        n = len(self.rows)
        picks = sum(r["pick_ok"] for r in self.rows)
        checked = [r["placed_ok"] for r in self.rows if r["placed_ok"] is not None]
        placed = f"{sum(checked) / len(checked):.1%}" if checked else "n/a"
        times = [r["cycle_time_s"] for r in self.rows if r["pick_ok"]]
        print(f"\n{n} cycles | pick success {picks / n:.1%} | correct bin {placed} | "
              f"mean cycle {np.mean(times) if times else float('nan'):.2f} s")

        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{self.backend}_cycles_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        print(f"Log written to {path}")
        return path
