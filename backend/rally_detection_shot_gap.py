"""
Shot-gap-based rally detection — Python port of the client-side detector
in useAdvancedAnalytics.ts:158-221.

This is the second detector in the pipeline. The first (rally_detection.detect_rallies)
is a gradient-based shuttle-position analysis. This one looks at shot events
(returned by shot_detection.detect_all_shots) and groups them by inter-shot
gap.

The two detectors are unioned in the Modal worker (modal_supabase_processor);
the union becomes the rally set used for clip cutting and the rally_clips
table.

Output shape MATCHES rally_detection.detect_rallies — list[dict] with keys:
  id, start_frame, end_frame, start_timestamp, end_timestamp, duration_seconds.

Frame-field mapping for skeleton_data: see the docstring in shot_detection.py.
"""
from __future__ import annotations

from typing import Any, Dict, List

from shot_detection import detect_all_shots


# Constants — keep aligned with useAdvancedAnalytics.ts lines 165-198.
MIN_SHOTS = 2
RALLY_GAP_SECONDS = 3.1
MIN_RALLY_DURATION_S = 0.8
SHUTTLE_VISIBILITY_THRESHOLD = 0.25  # 25% of frames in window must show shuttle


def detect_rallies_from_shots(
    skeleton_data: List[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    """
    Returns rally dicts in the SAME format as rally_detection.detect_rallies.
    """
    if not skeleton_data or len(skeleton_data) < 10:
        return []

    shots = detect_all_shots(skeleton_data, fps)
    if len(shots) < MIN_SHOTS:
        return []

    def is_shuttle_active_window(start_ts: float, end_ts: float) -> bool:
        total = 0
        visible = 0
        for f in skeleton_data:
            ts = f.get("timestamp")
            if ts is None or ts < start_ts or ts > end_ts:
                continue
            total += 1
            sp = f.get("shuttle_position")
            if sp and sp.get("x") is not None and sp.get("y") is not None:
                visible += 1
        return total == 0 or (visible / total) >= SHUTTLE_VISIBILITY_THRESHOLD

    detected: List[Dict[str, Any]] = []
    rally_start = 0
    for i in range(1, len(shots)):
        gap = shots[i]["timestamp"] - shots[i - 1]["timestamp"]
        is_last = i == len(shots) - 1
        if gap > RALLY_GAP_SECONDS or is_last:
            end_idx = i + 1 if is_last else i
            rally_shots = shots[rally_start:end_idx]
            if len(rally_shots) >= MIN_SHOTS:
                first = rally_shots[0]
                last = rally_shots[-1]
                duration = last["timestamp"] - first["timestamp"]
                if duration >= MIN_RALLY_DURATION_S and is_shuttle_active_window(
                    first["timestamp"], last["timestamp"]
                ):
                    detected.append({
                        "id": len(detected) + 1,
                        "start_frame": int(first["frame"]),
                        "end_frame": int(last["frame"]),
                        "start_timestamp": float(first["timestamp"]),
                        "end_timestamp": float(last["timestamp"]),
                        "duration_seconds": float(duration),
                    })
            rally_start = i

    return detected


def union_rallies(
    a: List[Dict[str, Any]],
    b: List[Dict[str, Any]],
    fps: float,
    overlap_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Combine two rally lists, deduplicating by temporal overlap.

    Two rallies are considered "the same" if their time ranges overlap by
    more than `overlap_threshold` of the SHORTER rally's duration. When two
    overlap, take the union of their bounds (earliest start, latest end).
    Otherwise both are kept.

    Returns rallies sorted by start_timestamp with sequential id starting
    at 1. Frame indices are recomputed as int(timestamp * fps).
    """
    combined: List[Dict[str, float]] = []  # list of {start, end}

    def overlaps(rally: Dict[str, Any], cand: Dict[str, float]) -> bool:
        s = max(rally["start_timestamp"], cand["start"])
        e = min(rally["end_timestamp"], cand["end"])
        if e <= s:
            return False
        overlap_dur = e - s
        shorter = min(
            rally["end_timestamp"] - rally["start_timestamp"],
            cand["end"] - cand["start"],
        )
        return shorter > 0 and (overlap_dur / shorter) >= overlap_threshold

    all_rallies = sorted(list(a) + list(b), key=lambda x: x["start_timestamp"])

    for r in all_rallies:
        absorbed = False
        for c in combined:
            if overlaps(r, c):
                c["start"] = min(c["start"], r["start_timestamp"])
                c["end"] = max(c["end"], r["end_timestamp"])
                absorbed = True
                break
        if not absorbed:
            combined.append({
                "start": float(r["start_timestamp"]),
                "end": float(r["end_timestamp"]),
            })

    safe_fps = fps if fps and fps > 0 else 30.0
    return [
        {
            "id": i + 1,
            "start_frame": int(c["start"] * safe_fps),
            "end_frame": int(c["end"] * safe_fps),
            "start_timestamp": float(c["start"]),
            "end_timestamp": float(c["end"]),
            "duration_seconds": float(c["end"] - c["start"]),
        }
        for i, c in enumerate(combined)
    ]
