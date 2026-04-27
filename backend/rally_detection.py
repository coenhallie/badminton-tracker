"""
Rally Detection — shot-gap approach.

Algorithm mirrors src/utils/shotDetection.ts (TS) exactly. Any change to
thresholds or logic here MUST be applied to that module (and
useAdvancedAnalytics.ts:rallies) to keep client and backend rally counts
comparable. Sync keys:

  min_rally_duration_s      ↔  MIN_RALLY_DURATION_S in useAdvancedAnalytics
  min_gap_duration_s        ↔  RALLY_GAP_SECONDS in useAdvancedAnalytics
  min_speed_sq              ↔  DetectShuttleShotsOptions.minSpeedSq
  dot_threshold             ↔  DetectShuttleShotsOptions.cosAngleMax
  stride_s                  ↔  DetectShuttleShotsOptions.strideSec
  min_shots                 ↔  MIN_SHOTS in useAdvancedAnalytics

Diverges intentionally on:
  - No wrist-proximity gate (backend has no pose keypoints at this stage).
  - No player-movement / speed-peaks fallbacks (backend always has
    TrackNet shuttle data; if shuttle is empty, no rallies are detected).

Shot-gap algorithm:
1. Detect shots as shuttle direction reversals (dot product of velocity
   vectors < threshold).
2. Group shots into rallies using a gap threshold.
3. Enforce min shots + min duration per rally.

TrackNet provides ~40-50% frame coverage vs ~10-40% from YOLO per-frame
detection, giving more complete shot detection and tighter rally boundaries.

Dense frame-by-frame positions are subsampled with a stride so that
velocity vectors span enough displacement to reveal real direction changes
rather than sub-pixel noise.
"""

import math
from typing import Dict, List, Optional


def detect_rallies(
    shuttle_positions: Dict[int, Dict],
    fps: float,
    total_frames: int,
    player_positions: Optional[Dict[int, List[Dict]]] = None,
    pose_data: Optional[Dict[int, List[Dict]]] = None,
    speed_data: Optional[Dict[int, List[float]]] = None,
    min_rally_duration_s: float = 0.8,
    min_gap_duration_s: float = 3.0,
    zero_gradient_window: int = 0,
    zero_gradient_ratio: float = 0.80,
    frame_width: int = 0,
    frame_height: int = 0,
) -> List[Dict]:
    """
    Detect rally boundaries from shuttle tracking data.

    Uses the shot-gap approach: detect individual shots (shuttle direction
    reversals), then group them into rallies by time gaps.

    Args:
        shuttle_positions: Dict of frame_num -> {"x", "y", "visible"}.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.
        min_rally_duration_s: Minimum rally length in seconds.
        min_gap_duration_s: Minimum gap between shots to split rallies.
        Other args: kept for backwards compatibility, unused.

    Returns:
        List of rally dicts with id, start/end frame/timestamp, duration.
    """
    if not shuttle_positions or fps <= 0:
        return []

    # Overhead-camera thresholds. The app only supports an overhead camera
    # centered above the court, which produces stable TrackNet trajectories.
    min_shot_gap_s = 0.6
    min_speed_sq = 15.0 * 15.0  # 225
    min_shots = 2
    dot_threshold = 0.0  # any reversal (>90deg)
    stride_s = 0.3

    min_shot_gap_frames = max(3, int(fps * min_shot_gap_s))

    # Step 1: Detect shots from shuttle direction reversals
    shots = _detect_shots(
        shuttle_positions, total_frames, fps, min_shot_gap_frames,
        min_speed_sq=min_speed_sq, dot_threshold=dot_threshold,
        stride_s=stride_s,
    )

    if len(shots) < min_shots:
        return []

    # Step 2: Group shots into rallies by gap threshold
    rally_gap_frames = max(1, int(min_gap_duration_s * fps))
    min_rally_frames = max(1, int(min_rally_duration_s * fps))
    rallies = _group_shots_into_rallies(
        shots, rally_gap_frames, min_rally_frames, fps,
        min_shots=min_shots,
    )

    # Step 3: Assign IDs and compute timestamps
    result = []
    for i, rally in enumerate(rallies):
        result.append({
            "id": i + 1,
            "start_frame": rally["start_frame"],
            "end_frame": rally["end_frame"],
            "start_timestamp": rally["start_frame"] / fps,
            "end_timestamp": rally["end_frame"] / fps,
            "duration_seconds": (rally["end_frame"] - rally["start_frame"]) / fps,
        })

    return result


def _detect_shots(
    positions: Dict[int, Dict],
    total_frames: int,
    fps: float,
    min_gap_frames: int,
    min_speed_sq: float = 225.0,
    dot_threshold: float = 0.0,
    stride_s: float = 0.3,
) -> List[Dict]:
    """
    Detect shots as shuttle direction reversals.

    A shot occurs when the shuttle changes direction — negative dot product
    between consecutive velocity vectors. This is the same algorithm as the
    client-side detectShotsFromShuttle().

    Positions are subsampled with a stride so velocity vectors span enough
    displacement to show clear direction changes in dense tracking data.
    """
    # Collect visible positions
    all_pts = []
    for f in range(total_frames):
        p = positions.get(f)
        if p and p.get("visible"):
            all_pts.append((f, p["x"], p["y"]))

    if len(all_pts) < 5:
        return []

    # Subsample: ~10 frames apart to smooth noise in dense data.
    # Client-side data is already sparse (~10-40% of frames) so it doesn't
    # need this, but TrackNet gives consecutive-frame positions where
    # velocity vectors change too gradually to detect reversals.
    STRIDE = max(3, int(fps * stride_s))
    pts = [all_pts[0]]
    for pt in all_pts[1:]:
        if pt[0] - pts[-1][0] >= STRIDE:
            pts.append(pt)

    if len(pts) < 3:
        return []

    # Minimum velocity magnitude to count as a real shot (not jitter).
    # A real shot moves the shuttle ~50+ pixels over a stride interval.
    # Stationary jitter is ~1-5 pixels. Threshold at 15px to be safe.
    MIN_SPEED_SQ = min_speed_sq

    # Detect direction reversals (dot product < 0)
    shots = []
    last_shot_frame = -10000

    for i in range(2, len(pts)):
        f0, x0, y0 = pts[i - 2]
        f1, x1, y1 = pts[i - 1]
        f2, x2, y2 = pts[i]

        vx1, vy1 = x1 - x0, y1 - y0
        vx2, vy2 = x2 - x1, y2 - y1

        # Skip if both velocity vectors are too small (jitter, not a shot)
        speed1_sq = vx1 * vx1 + vy1 * vy1
        speed2_sq = vx2 * vx2 + vy2 * vy2
        if speed1_sq < MIN_SPEED_SQ and speed2_sq < MIN_SPEED_SQ:
            continue

        dot = vx1 * vx2 + vy1 * vy2

        # For overhead: dot < 0 (any reversal >90deg)
        # For corner: dot < -0.25*|v1|*|v2| (reversal >~105deg, rejects jitter wobbles)
        if dot_threshold < 0:
            threshold = dot_threshold * math.sqrt(speed1_sq * speed2_sq)
        else:
            threshold = 0
        if dot < threshold and (f1 - last_shot_frame) >= min_gap_frames:
            shots.append({"frame": f1, "x": x1, "y": y1})
            last_shot_frame = f1

    return shots


def _group_shots_into_rallies(
    shots: List[Dict],
    rally_gap_frames: int,
    min_rally_frames: int,
    fps: float,
    min_shots: int = 2,
) -> List[Dict]:
    """
    Group shots into rallies using gap threshold.

    Same logic as client-side: if gap between consecutive shots > threshold,
    start a new rally. Require minimum shots and minimum duration.
    """
    if len(shots) < min_shots:
        return []

    rallies = []
    rally_start_idx = 0

    for i in range(1, len(shots)):
        gap = shots[i]["frame"] - shots[i - 1]["frame"]
        is_last = (i == len(shots) - 1)

        if gap > rally_gap_frames or is_last:
            # End of current rally group
            end_idx = i if (is_last and gap <= rally_gap_frames) else i - 1
            rally_shots = shots[rally_start_idx:end_idx + 1]

            if len(rally_shots) >= min_shots:
                start_frame = rally_shots[0]["frame"]
                end_frame = rally_shots[-1]["frame"]
                # Add small buffer after last shot for shuttle to land
                end_frame = end_frame + max(1, int(0.5 * fps))

                if (end_frame - start_frame) >= min_rally_frames:
                    rallies.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "shot_count": len(rally_shots),
                    })

            rally_start_idx = i

    return rallies
