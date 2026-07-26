"""
Shot-gap-based rally detection — Python port of the client-side detector
in useAdvancedAnalytics.ts:158-221.

This is the second detector in the pipeline. The first (rally_detection.detect_rallies)
is a gradient-based shuttle-position analysis. This one looks at shot events
(returned by shot_detection.detect_all_shots) and groups them by inter-shot
gap.

The Modal worker (modal_supabase_processor) uses this detector two ways:
its raw-track output is unioned with the gradient detector's (via
`union_rallies`) into the "combined" set stored in results.json — the web
app shows both timelines from that. Separately, filtered-track rallies with
raw-track bound extension (via `refine_rallies`) drive clip cutting and
the rally_clips table, so the apps get an accurate 'client'-style
separation right after Phase 1; Phase 2 later re-cuts the clips with exact
browser parity. The combined set is the clip fallback when this detector
finds nothing.

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
    require_players: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns rally dicts in the SAME format as rally_detection.detect_rallies.

    require_players=True is exact TS/browser parity (shots on frames without
    player data are dropped). Pass False for Phase 1 detection-only frames,
    which never carry player data.
    """
    if not skeleton_data or len(skeleton_data) < 10:
        return []

    shots = detect_all_shots(skeleton_data, fps, require_players=require_players)
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
            # Shot i belongs to the current rally ONLY when it is the final
            # shot AND close enough to its predecessor. When it is both last
            # and beyond the gap threshold it is an isolated trailing shot
            # that starts a new (1-shot, therefore rejected) rally — including
            # it welded the whole inter-rally gap onto the previous rally's
            # end. Matches rally_detection._group_shots_into_rallies.
            end_idx = i + 1 if (is_last and gap <= RALLY_GAP_SECONDS) else i
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


def refine_rallies(
    filtered: List[Dict[str, Any]],
    raw: List[Dict[str, Any]],
    fps: float,
    max_extension_sec: float = RALLY_GAP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Merge Phase 1's two shuttle-track detections into one clip-ready list.

    The filtered track (court ROI + static rejection) provides the rally
    LIST and SPLITS — its noise rejection is trustworthy, but over-filtering
    trims rally tails. The raw track has accurate bounds, yet fabricates
    rallies in replay/idle windows and can weld adjacent rallies into one
    when noise bridges the inter-rally gap. So: keep the filtered rallies,
    widen each boundary toward overlapping raw bounds by at most
    max_extension_sec, and clamp to the neighboring rallies. Raw rallies
    overlapping no filtered rally are dropped entirely.
    """
    safe_fps = fps if fps and fps > 0 else 30.0
    refined: List[Dict[str, Any]] = []
    for i, f in enumerate(filtered):
        start, end = f["start_timestamp"], f["end_timestamp"]
        overlapping = [
            r for r in raw
            if min(end, r["end_timestamp"]) > max(start, r["start_timestamp"])
        ]
        if overlapping:
            raw_start = min(r["start_timestamp"] for r in overlapping)
            raw_end = max(r["end_timestamp"] for r in overlapping)
            start = max(start - max_extension_sec, min(raw_start, start))
            end = min(end + max_extension_sec, max(raw_end, end))
        if i > 0:
            start = max(start, filtered[i - 1]["end_timestamp"] + 0.05)
        if i + 1 < len(filtered):
            end = min(end, filtered[i + 1]["start_timestamp"] - 0.05)
        refined.append({
            "id": i + 1,
            "start_frame": int(start * safe_fps),
            "end_frame": int(end * safe_fps),
            "start_timestamp": float(start),
            "end_timestamp": float(end),
            "duration_seconds": float(end - start),
        })
    return refined


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
