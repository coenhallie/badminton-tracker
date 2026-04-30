"""
Shot detection — Python port of src/utils/shotDetection.ts.

Mirrors the client-side detector that powers in-browser rally segmentation
in useAdvancedAnalytics.ts. Used by rally_detection_shot_gap.py to feed the
shot-gap rally detector that runs alongside the gradient-based detector
(rally_detection.py) in the Modal worker.

Algorithmic parity with the TS source:
  * Outlier rejection on visible shuttle positions (single-frame TrackNet
    glitches dropped before velocity computation).
  * Stride='auto': enable backend-style subsampling (>50% coverage) with
    autoStrideSec=0.3.
  * Dot-threshold math: cosAngleMax<0 ⇒ threshold = cosAngleMax * sqrt(s1²s2²);
    otherwise threshold = 0. Trigger when dot < threshold.
  * MAX_GAP_S=2.5 guard against velocity built across inter-rally idle.
  * minSpeedSq=225 (15 px) — both per-step speeds below ⇒ jitter, skip.
  * minShotGapFrames = max(3, floor(fps * 0.6)).
  * Pose-classification fallback (detect_pose_shots) when shuttle data is
    too sparse, then mergeShots() preferring shuttle-detected shots.
  * Wrist-proximity gate is OFF for the rally use case (TS sets
    wristProximityMeters=null), so the Python port omits homography wholly.
    The TS rally caller also passes minAccelMagPx=null, so the accel gate
    is omitted too. (These options exist in the TS source for the
    auto-pause caller, which is not ported here.)

Frame field mapping (TS SkeletonFrame → Python skeleton_frame dict):
  TS f.frame              → py f["frame"]
  TS f.timestamp          → py f["timestamp"]
  TS f.shuttle_position   → py f.get("shuttle_position")  (or None)
  TS f.players[i].player_id → py f["players"][i]["player_id"]
  TS f.players[i].center  → py f["players"][i].get("center")
  TS f.players[i].keypoints[idx]  → py f["players"][i]["keypoints"][idx]
       Note: TS uses dense index lookup (kpts[9] = left wrist). Python
       skeleton_frames also store keypoints as an ordered list of
       {name, x, y, confidence} dicts, indexed by COCO keypoint id, so
       direct integer indexing works. We still bounds-check.
  TS f.pose_classifications → py f.get("pose_classifications") (rarely set
       in the Python pipeline, but we honor it for parity).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Constants — keep aligned with src/utils/shotDetection.ts
# -----------------------------------------------------------------------------
LEFT_WRIST_KP = 9
RIGHT_WRIST_KP = 10

# Min keypoint confidence for a wrist to be considered visible.
WRIST_CONFIDENCE_THRESHOLD = 0.3

# Single-frame jump threshold (squared pixels) for outlier rejection.
OUTLIER_DIST_SQ = 400 * 400

# Max time between sampled shuttle positions before velocity is meaningless.
MAX_GAP_S = 2.5

# High-confidence hitting poses — only poses that unambiguously indicate a
# shot. Mirrors HITTING_POSES in useAdvancedAnalytics.ts.
HITTING_POSES = {"smash", "overhead", "serving", "serve", "offense"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _find_closest_player(
    players: List[Dict[str, Any]], x: float, y: float
) -> Optional[Dict[str, Any]]:
    """Find the player whose center is closest to (x, y). Returns None when
    none have a center; falls back to players[0] when all centers are
    missing — same fallback as the TS findClosestPlayer."""
    closest: Optional[Dict[str, Any]] = None
    closest_dist = math.inf

    for player in players:
        center = player.get("center")
        if not center:
            continue
        cx = center.get("x")
        cy = center.get("y")
        if cx is None or cy is None:
            continue
        dx = cx - x
        dy = cy - y
        d = math.sqrt(dx * dx + dy * dy)
        if d < closest_dist:
            closest_dist = d
            closest = player

    if closest is not None:
        return closest
    return players[0] if players else None


def _filter_outliers(points: List[Tuple[int, float, float, float]]) -> List[
    Tuple[int, float, float, float]
]:
    """Drop a position when its squared distance to BOTH neighbors exceeds
    OUTLIER_DIST_SQ. First and last are always kept.

    Each point is (frame, timestamp, x, y).
    """
    if len(points) < 3:
        return list(points)
    out: List[Tuple[int, float, float, float]] = [points[0]]
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[i + 1]
        d_prev = (curr[2] - prev[2]) ** 2 + (curr[3] - prev[3]) ** 2
        d_next = (curr[2] - nxt[2]) ** 2 + (curr[3] - nxt[3]) ** 2
        if d_prev > OUTLIER_DIST_SQ and d_next > OUTLIER_DIST_SQ:
            continue
        out.append(curr)
    out.append(points[-1])
    return out


# -----------------------------------------------------------------------------
# Shuttle trajectory detector
# -----------------------------------------------------------------------------
def detect_shuttle_shots(
    frames: List[Dict[str, Any]],
    fps: float,
    min_shot_gap_sec: float = 0.6,
    min_speed_sq: float = 225.0,
    cos_angle_max: float = 0.0,
    reject_outliers: bool = True,
    stride_sec: Any = "auto",
    auto_stride_sec: float = 0.3,
) -> List[Dict[str, Any]]:
    """Direct port of detectShuttleShots(frames, opts).

    Returns a list of shot-event dicts. The TS ShotEvent fields are mapped to
    snake_case here for Python convention, but the rally-grouping code only
    reads `frame` and `timestamp`, so naming downstream is not load-bearing.
    """
    if not fps or fps <= 0:
        return []

    min_shot_gap_frames = max(3, int(fps * min_shot_gap_sec))

    # 1. Collect visible shuttle positions
    raw: List[Tuple[int, float, float, float]] = []
    for f in frames:
        s = f.get("shuttle_position")
        if s and s.get("x") is not None and s.get("y") is not None:
            raw.append((f["frame"], f["timestamp"], s["x"], s["y"]))

    if len(raw) < 5:
        return []

    # 2. Outlier rejection
    cleaned = _filter_outliers(raw) if reject_outliers else raw

    # 3 + 4. Stride subsampling
    samples: List[Tuple[int, float, float, float]] = cleaned
    stride_frames = 0

    if stride_sec == "auto":
        total_frames = len(frames) or 1
        coverage = len(raw) / total_frames
        if coverage > 0.5:
            stride_frames = max(3, round(fps * auto_stride_sec))
    elif isinstance(stride_sec, (int, float)) and stride_sec is not None:
        stride_frames = max(3, round(fps * stride_sec))

    if stride_frames > 0 and cleaned:
        subsampled: List[Tuple[int, float, float, float]] = [cleaned[0]]
        for pt in cleaned[1:]:
            if pt[0] - subsampled[-1][0] >= stride_frames:
                subsampled.append(pt)
        samples = subsampled

    if len(samples) < 3:
        return []

    # 5. Candidate triples
    frame_map: Dict[int, Dict[str, Any]] = {f["frame"]: f for f in frames}

    shots: List[Dict[str, Any]] = []
    last_shot_frame = -math.inf

    for i in range(2, len(samples)):
        f0, t0, x0, y0 = samples[i - 2]
        f1, t1, x1, y1 = samples[i - 1]
        f2, t2, x2, y2 = samples[i]

        # Max-time-gap guard.
        if (t1 - t0) > MAX_GAP_S or (t2 - t1) > MAX_GAP_S:
            continue

        vx1 = x1 - x0
        vy1 = y1 - y0
        vx2 = x2 - x1
        vy2 = y2 - y1

        speed1_sq = vx1 * vx1 + vy1 * vy1
        speed2_sq = vx2 * vx2 + vy2 * vy2

        if speed1_sq < min_speed_sq and speed2_sq < min_speed_sq:
            continue

        dot = vx1 * vx2 + vy1 * vy2

        if cos_angle_max < 0:
            threshold = cos_angle_max * math.sqrt(speed1_sq * speed2_sq)
        else:
            threshold = 0.0
        if dot >= threshold:
            continue

        # (Accel gate omitted — rally caller passes minAccelMagPx=null.)

        if (f1 - last_shot_frame) < min_shot_gap_frames:
            continue

        sk = frame_map.get(f1)
        if not sk:
            continue
        players = sk.get("players") or []
        if not players:
            continue

        # (Wrist-proximity gate omitted — rally caller passes
        # wristProximityMeters=null and homography is not available
        # backend-side at this stage anyway.)

        closest = _find_closest_player(players, x1, y1)
        if not closest:
            continue

        center = closest.get("center") or {"x": 0.0, "y": 0.0}

        shots.append({
            "frame": int(f1),
            "timestamp": float(t1),
            "player_id": int(closest.get("player_id", 0)),
            "shot_type": "unknown",
            "shuttle_position": {"x": float(x1), "y": float(y1)},
            "player_position": {
                "x": float(center.get("x", 0.0)),
                "y": float(center.get("y", 0.0)),
            },
            "detection_method": "shuttle_trajectory",
        })
        last_shot_frame = f1

    return shots


# -----------------------------------------------------------------------------
# Pose classification fallback
# -----------------------------------------------------------------------------
def detect_pose_shots(
    frames: List[Dict[str, Any]],
    fps: float,
    hitting_classes: Optional[set] = None,
    min_confidence: float = 0.65,
    min_shot_gap_sec: float = 0.6,
    per_player_gap_sec: float = 0.8,
) -> List[Dict[str, Any]]:
    """Direct port of detectPoseShots. Reads frame.pose_classifications.

    Note: the Python skeleton_frames pipeline currently does NOT populate
    pose_classifications (per the doc-comment in useAdvancedAnalytics.ts:
    "NEVER POPULATED"), so this fallback will typically yield no shots.
    Kept for parity in case the pipeline starts emitting them.
    """
    if not fps or fps <= 0:
        return []
    if hitting_classes is None:
        hitting_classes = HITTING_POSES

    min_shot_gap_frames = max(1, int(fps * min_shot_gap_sec))
    per_player_gap_frames = max(min_shot_gap_frames, int(fps * per_player_gap_sec))

    shots: List[Dict[str, Any]] = []
    last_shot_by_player: Dict[int, int] = {}
    last_shot_frame = -10**9

    for frame in frames:
        classifications = frame.get("pose_classifications")
        if not classifications:
            continue
        if (frame["frame"] - last_shot_frame) < min_shot_gap_frames:
            continue

        players = frame.get("players") or []

        for cls in classifications:
            cname = cls.get("class_name")
            if cname not in hitting_classes:
                continue
            if cls.get("confidence", 0.0) < min_confidence:
                continue

            matched: Optional[Dict[str, Any]] = players[0] if players else None

            bbox = cls.get("bbox")
            if bbox and len(players) > 1:
                cx = bbox["x"] + bbox["width"] / 2
                cy = bbox["y"] + bbox["height"] / 2
                matched = _find_closest_player(players, cx, cy) or matched

            if not matched:
                continue

            pid = int(matched.get("player_id", 0))
            last_for_player = last_shot_by_player.get(pid, -10**9)
            if (frame["frame"] - last_for_player) < per_player_gap_frames:
                continue

            shuttle = frame.get("shuttle_position")
            shuttle_pos = None
            if shuttle and shuttle.get("x") is not None and shuttle.get("y") is not None:
                shuttle_pos = {"x": float(shuttle["x"]), "y": float(shuttle["y"])}

            center = matched.get("center") or {"x": 0.0, "y": 0.0}

            shots.append({
                "frame": int(frame["frame"]),
                "timestamp": float(frame["timestamp"]),
                "player_id": pid,
                "shot_type": cname,
                "shuttle_position": shuttle_pos,
                "player_position": {
                    "x": float(center.get("x", 0.0)),
                    "y": float(center.get("y", 0.0)),
                },
                "detection_method": "pose_classification",
            })
            last_shot_frame = frame["frame"]
            last_shot_by_player[pid] = frame["frame"]
            # Only one shot per frame.
            break

    return shots


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
def merge_shots(
    shots: List[Dict[str, Any]], min_gap_frames: int
) -> List[Dict[str, Any]]:
    """Direct port of mergeShots. Sort by frame; if the current shot is
    within min_gap_frames of the last accepted, prefer whichever has a
    non-null shuttle_position."""
    sorted_shots = sorted(shots, key=lambda s: s["frame"])
    merged: List[Dict[str, Any]] = []
    for shot in sorted_shots:
        if merged:
            last = merged[-1]
            if (shot["frame"] - last["frame"]) < min_gap_frames:
                if last.get("shuttle_position") is None and shot.get("shuttle_position") is not None:
                    merged[-1] = shot
                continue
        merged.append(shot)
    return merged


# -----------------------------------------------------------------------------
# Top-level entry point — mirrors useAdvancedAnalytics.detectAllShots
# -----------------------------------------------------------------------------
def detect_all_shots(frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    """Mirrors useAdvancedAnalytics.detectAllShots(frames, fps).

    Strategy:
      1. Run shuttle-trajectory detector (rally use-case options: wrist gate
         off, accel gate off, outlier rejection on, stride='auto').
      2. If we got >= 4 shuttle shots, return those (and tag shot_type
         'unknown' to match the TS toRallyShot output).
      3. Otherwise add the pose-classification fallback and merge,
         preferring shuttle-detected shots.

    Each returned shot dict has: frame, timestamp, player_id, shot_type,
    shuttle_position, player_position, detection_method.
    """
    shuttle_shots = detect_shuttle_shots(
        frames,
        fps=fps,
        reject_outliers=True,
        stride_sec="auto",
    )

    if len(shuttle_shots) >= 4:
        return shuttle_shots

    pose_shots = detect_pose_shots(
        frames,
        fps=fps,
        hitting_classes=HITTING_POSES,
        min_confidence=0.65,
        min_shot_gap_sec=0.6,
        per_player_gap_sec=0.8,
    )

    if not pose_shots:
        return shuttle_shots

    min_gap_frames = max(3, int(fps * 0.6))
    return merge_shots(shuttle_shots + pose_shots, min_gap_frames)
