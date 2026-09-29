"""
Offline checks for Phase 2's per-frame filters that are not the court area
filter (which verify_camera_motion.py section 7 covers).

Usage:
    python backend/scripts/verify_phase2_filters.py

No GPU, no network. Sections:
  1. is_sitting_pose, as it has always behaved (characterization: these
     cases pin the rule the loop used before it was lifted to module level).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modal_supabase_processor import is_sitting_pose  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


def body(hip_y: float, knee_y: float, ankle_y: float, x: float = 100.0) -> np.ndarray:
    """17 keypoints as the loop's shadows carry them: (x, y) only, no confidence column."""
    k = np.full((17, 2), [x, hip_y - 60.0])
    k[11] = (x - 8, hip_y)
    k[12] = (x + 8, hip_y)
    k[13] = (x - 8, knee_y)
    k[14] = (x + 8, knee_y)
    k[15] = (x - 8, ankle_y)
    k[16] = (x + 8, ankle_y)
    return k


def box(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


print("1. is_sitting_pose (unchanged behaviour)")
standing = body(hip_y=400.0, knee_y=450.0, ankle_y=500.0)
check("a standing player in a tall box is not sitting",
      not is_sitting_pose(standing, box(80, 300, 120, 505)))
check("a box under 1.2 times as tall as it is wide reads as sitting",
      is_sitting_pose(standing, box(20, 300, 220, 505)))
check("hips at knee height read as sitting",
      is_sitting_pose(body(hip_y=440.0, knee_y=445.0, ankle_y=500.0), box(80, 300, 120, 505)))
check("hips close above the ankles read as sitting",
      is_sitting_pose(body(hip_y=460.0, knee_y=480.0, ankle_y=500.0), box(80, 300, 120, 505)))
check("no box: not sitting", not is_sitting_pose(standing, None))
missing = standing.copy()
missing[11:17] = 0.0
check("legs not detected (the (0, 0) sentinel) fall back to the box alone",
      not is_sitting_pose(missing, box(80, 300, 120, 505)) and is_sitting_pose(missing, box(20, 300, 220, 505)))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
