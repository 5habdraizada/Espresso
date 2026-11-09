"""Sorting rule: which bin each detected part goes to."""

from __future__ import annotations

from main.interfaces import Detection


def choose_bin(det: Detection) -> str:
    """Name of the bin (a key of CellConfig.bins) this part goes to."""
    if det.defective or det.color is None:
        return "reject"
    return det.color
