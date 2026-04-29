"""
Speed recalculation from skeleton data.

Pure-Python port of `calculateSpeedFromSkeleton` from convex/http.ts.
Uses OpenCV for homography (replaces the TS buildHomographyFromKeypoints +
applyHomography pair with cv2.findHomography + cv2.perspectiveTransform).

Pipeline (matches the TS reference 1:1):
1. Build a homography from manual_court_keypoints if provided. Supports up to
   12 keypoints (4 required corners + 8 optional). cv2.findHomography uses
   least-squares DLT for 4+ points, mirroring the TS Hartley-normalized DLT.
2. For each (frame, player) pair, transform pixel -> court meters. If a point
   maps outside the court (with 2 m margin) it is rejected.
3. Compute per-frame speed; apply 4-step filter:
     - Pixel jump (catches tracking ID swaps)
     - Distance-per-frame in meters (catches position jumps)
     - Hard speed cap (physiological limit ~25 km/h)
     - Median outlier rejection (3x running median, with min threshold)
4. Aggregate per-player avg/max/distance statistics.

Constants are kept in lock-step with convex/http.ts. If any value changes
upstream, update both files.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# --- Court geometry (BWF official, meters) ----------------------------------
# Mirrors convex/http.ts.
COURT_LENGTH = 13.4    # along the long axis
COURT_WIDTH = 6.1      # along the short axis (doubles)
SERVICE_LINE = 1.98    # distance from net to short service line


# Standard court keypoint positions in meters (matches 12-point calibration
# system used in convex/http.ts: TL, TR, BR, BL, NL, NR, SNL, SNR, SFL, SFR,
# CTN, CTF).
COURT_KEYPOINT_POSITIONS: list[tuple[float, float]] = [
    (0.0, 0.0),                                                  # 0:  TL
    (COURT_WIDTH, 0.0),                                          # 1:  TR
    (COURT_WIDTH, COURT_LENGTH),                                 # 2:  BR
    (0.0, COURT_LENGTH),                                         # 3:  BL
    (0.0, COURT_LENGTH / 2),                                     # 4:  NL  - Net left
    (COURT_WIDTH, COURT_LENGTH / 2),                             # 5:  NR  - Net right
    (0.0, COURT_LENGTH / 2 - SERVICE_LINE),                      # 6:  SNL - Service near left
    (COURT_WIDTH, COURT_LENGTH / 2 - SERVICE_LINE),              # 7:  SNR - Service near right
    (0.0, COURT_LENGTH / 2 + SERVICE_LINE),                      # 8:  SFL - Service far left
    (COURT_WIDTH, COURT_LENGTH / 2 + SERVICE_LINE),              # 9:  SFR - Service far right
    (COURT_WIDTH / 2, COURT_LENGTH / 2 - SERVICE_LINE),          # 10: CTN - Center near
    (COURT_WIDTH / 2, COURT_LENGTH / 2 + SERVICE_LINE),          # 11: CTF - Center far
]


# Optional keypoint -> court-position-index mapping (in TS file order).
_OPTIONAL_KEYPOINTS: list[tuple[str, int]] = [
    ("net_left",                4),
    ("net_right",               5),
    ("service_line_near_left",  6),
    ("service_line_near_right", 7),
    ("service_line_far_left",   8),
    ("service_line_far_right",  9),
    ("center_near",            10),
    ("center_far",             11),
]


# --- Filter thresholds (parity with convex/http.ts) -------------------------
# Maximum realistic on-court speed. Badminton players rarely exceed 25 km/h.
MAX_REALISTIC_SPEED_KMH = 25.0
# Maximum allowed pixel jump per frame (catches tracking ID swaps).
MAX_FRAME_JUMP_PIXELS = 80.0
# Maximum allowed distance per frame in meters (at 30 fps, 7 m/s -> 0.23 m).
MAX_DISTANCE_PER_FRAME_M = 0.25
# Sliding window length for the median outlier filter.
SPEED_MEDIAN_WINDOW = 5


def _coerce_xy(pt: Any) -> tuple[float, float] | None:
    """Accept [x, y] (list/tuple) or {'x': ..., 'y': ...} -> (x, y)."""
    if pt is None:
        return None
    if isinstance(pt, dict):
        x = pt.get("x")
        y = pt.get("y")
        if x is None or y is None:
            return None
        return (float(x), float(y))
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return (float(pt[0]), float(pt[1]))
    return None


def build_homography(manual_keypoints: dict | None) -> np.ndarray | None:
    """
    Build a 3x3 homography from 4-12 court keypoints. Returns None if the four
    required corners are missing or cv2.findHomography fails.

    Pixel -> court-meter transform. Court coordinate system:
      origin (0, 0) = top-left corner
      X axis = court width  (0 .. COURT_WIDTH)
      Y axis = court length (0 .. COURT_LENGTH)
    """
    if not manual_keypoints:
        return None

    required = ("top_left", "top_right", "bottom_right", "bottom_left")
    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    for i, name in enumerate(required):
        xy = _coerce_xy(manual_keypoints.get(name))
        if xy is None:
            return None
        src_pts.append(xy)
        dst_pts.append(COURT_KEYPOINT_POSITIONS[i])

    # Optional keypoints — include each one that is present (improves accuracy
    # via least-squares DLT, matching the TS implementation).
    for field, idx in _OPTIONAL_KEYPOINTS:
        xy = _coerce_xy(manual_keypoints.get(field))
        if xy is not None:
            src_pts.append(xy)
            dst_pts.append(COURT_KEYPOINT_POSITIONS[idx])

    import cv2  # imported here so this module is importable in tooling contexts

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    try:
        H, _mask = cv2.findHomography(src, dst, method=0)
    except cv2.error:
        return None
    return H if H is not None else None


def pixel_to_meters(
    homography: np.ndarray | None,
    fallback_mpp: float,
    px: float,
    py: float,
) -> tuple[float, float] | None:
    """
    Map a pixel coordinate to court meters.

    With homography: rejects points more than 2 m outside the court bounds.
    Without homography: linear scaling by `fallback_mpp` (no bounds check).
    """
    if homography is not None:
        import cv2

        pts = np.asarray([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, homography)
        cx, cy = float(out[0, 0, 0]), float(out[0, 0, 1])
        if cx < -2 or cx > COURT_WIDTH + 2 or cy < -2 or cy > COURT_LENGTH + 2:
            return None
        return (cx, cy)
    return (px * fallback_mpp, py * fallback_mpp)


def calculate_speeds_from_skeleton(
    skeleton_data: list[dict],
    fps: float,
    video_width: int,
    video_height: int,
    manual_court_keypoints: dict | None = None,
) -> dict[str, Any]:
    """
    Recompute per-frame speed and per-player aggregates from skeleton data.

    Returns:
        {
          "frame_data": [
              {"frame": int,
               "players": [{"player_id": int, "speed_kmh": float}, ...]},
              ...
          ],
          "statistics": {
              "<player_id>": {
                  "avg":              {"speed_kmh": float},
                  "max":              {"speed_kmh": float},
                  "total_distance_m": float,
              },
              ...
          },
          "manual_keypoints_used": bool,
        }
    """
    has_manual_keypoints = bool(
        manual_court_keypoints
        and manual_court_keypoints.get("top_left")
        and manual_court_keypoints.get("bottom_left")
    )

    homography: np.ndarray | None = None
    if has_manual_keypoints:
        homography = build_homography(manual_court_keypoints)

    # Fallback meters-per-pixel — matches the TS reference and the
    # modal_convex_processor.py heuristic.
    fallback_mpp = 0.0
    if homography is None:
        fallback_mpp = COURT_LENGTH / (max(video_width, video_height) * 0.8)

    # Per-player tracking state
    player_stats: dict[str, dict[str, Any]] = {}
    frame_data: list[dict[str, Any]] = []

    for frame in skeleton_data:
        frame_idx = frame.get("frame")
        if frame_idx is None:
            continue
        frame_speeds: list[dict[str, Any]] = []

        for player in frame.get("players", []):
            # Position extraction — center preferred over position.
            xy = _coerce_xy(player.get("center")) or _coerce_xy(player.get("position"))
            if xy is None:
                continue
            px, py = xy

            court = pixel_to_meters(homography, fallback_mpp, px, py)
            if court is None:
                continue
            cx, cy = court

            pid = str(player["player_id"])
            stats = player_stats.setdefault(pid, {
                "speeds": [],
                "distance": 0.0,
                "prev": None,            # {"cx", "cy", "px", "py", "frame"}
                "speed_window": [],
            })

            speed_kmh = 0.0
            is_valid = True

            prev = stats["prev"]
            if prev is not None:
                d_meters = math.hypot(cx - prev["cx"], cy - prev["cy"])
                d_pixels = math.hypot(px - prev["px"], py - prev["py"])
                frames_elapsed = max(1, frame_idx - prev["frame"])
                dt = frames_elapsed / fps
                dist_per_frame = d_meters / frames_elapsed
                px_per_frame = d_pixels / frames_elapsed

                if dt > 0:
                    speed_kmh = (d_meters / dt) * 3.6

                    # 4-step filter pipeline (parity with convex/http.ts)
                    if px_per_frame > MAX_FRAME_JUMP_PIXELS:
                        speed_kmh = 0.0
                        is_valid = False
                    elif dist_per_frame > MAX_DISTANCE_PER_FRAME_M:
                        speed_kmh = 0.0
                        is_valid = False
                    elif speed_kmh > MAX_REALISTIC_SPEED_KMH:
                        speed_kmh = 0.0
                        is_valid = False
                    else:
                        # Median outlier filter — reject sudden 3x spikes once
                        # we have at least 3 prior samples.
                        win = stats["speed_window"]
                        if len(win) >= 3:
                            sorted_win = sorted(win)
                            median_speed = sorted_win[len(sorted_win) // 2]
                            if median_speed > 2.0 and speed_kmh > median_speed * 3.0:
                                speed_kmh = 0.0
                                is_valid = False

                    if is_valid and speed_kmh > 0:
                        win = stats["speed_window"]
                        win.append(speed_kmh)
                        if len(win) > SPEED_MEDIAN_WINDOW:
                            win.pop(0)
                        stats["speeds"].append(speed_kmh)
                        stats["distance"] += d_meters

            # Only update tracking position when measurement was valid (avoids
            # propagating jump errors).
            if is_valid:
                stats["prev"] = {
                    "cx": cx, "cy": cy,
                    "px": px, "py": py,
                    "frame": frame_idx,
                }

            frame_speeds.append({
                "player_id": player["player_id"],
                "speed_kmh": speed_kmh,
            })

        if frame_speeds:
            frame_data.append({"frame": frame_idx, "players": frame_speeds})

    # Per-player aggregates (per-frame filtering already handled outliers;
    # only the hard cap is applied here, matching the TS reference).
    statistics: dict[str, Any] = {}
    for pid, stats in player_stats.items():
        filtered = [s for s in stats["speeds"] if 0 < s <= MAX_REALISTIC_SPEED_KMH]
        avg_speed = (sum(filtered) / len(filtered)) if filtered else 0.0
        max_speed = max(filtered) if filtered else 0.0
        statistics[pid] = {
            "avg": {"speed_kmh": avg_speed},
            "max": {"speed_kmh": min(max_speed, MAX_REALISTIC_SPEED_KMH)},
            "total_distance_m": stats["distance"],
        }

    return {
        "frame_data": frame_data,
        "statistics": statistics,
        "manual_keypoints_used": has_manual_keypoints,
    }
