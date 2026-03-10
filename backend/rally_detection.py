"""
Rally Detection from Shuttlecock Tracking Data

Uses the gradient-based approach from:
"A Comparative Analysis of Deep Learning Models and Gradient Computation
 for Rally Detection in Badminton Videos" (Springer, 2025)

Algorithm:
1. Compute distance gradient between consecutive shuttle positions
2. Rally END: >=80% of N consecutive frames have zero gradient
3. Rally START: gradient becomes consistently non-zero
4. Post-filter: minimum rally duration, minimum gap between rallies
"""

import math
from typing import Dict, List, Optional


def detect_rallies(
    shuttle_positions: Dict[int, Dict],
    fps: float,
    total_frames: int,
    min_rally_duration_s: float = 2.0,
    min_gap_duration_s: float = 3.0,
    zero_gradient_window: int = 0,
    zero_gradient_ratio: float = 0.80,
) -> List[Dict]:
    """
    Detect rally boundaries from shuttle tracking data.

    Args:
        shuttle_positions: Dict of frame_num → {"x", "y", "visible"}.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.
        min_rally_duration_s: Minimum rally length in seconds.
        min_gap_duration_s: Minimum gap between rallies in seconds.
        zero_gradient_window: Number of frames for zero-gradient detection.
                              0 = auto (1.5 seconds worth of frames).
        zero_gradient_ratio: Fraction of frames that must have zero gradient
                             to declare rally end (default 0.80).

    Returns:
        List of rally dicts with:
        - id: Rally number (1-indexed)
        - start_frame: First frame of rally
        - end_frame: Last frame of rally
        - start_timestamp: Start time in seconds
        - end_timestamp: End time in seconds
        - duration_seconds: Rally duration
    """
    if not shuttle_positions or fps <= 0:
        return []

    # Auto-compute window size: ~1.5 seconds
    if zero_gradient_window <= 0:
        zero_gradient_window = max(15, int(fps * 1.5))

    min_rally_frames = max(1, int(min_rally_duration_s * fps))
    min_gap_frames = max(1, int(min_gap_duration_s * fps))

    # Step 1: Compute distance gradient for each frame
    gradients = _compute_gradients(shuttle_positions, total_frames)

    # Step 2: Classify each frame as "active" (shuttle moving) or "idle"
    frame_active = _classify_frames(
        gradients, total_frames, zero_gradient_window, zero_gradient_ratio
    )

    # Step 3: Extract rally segments from active/idle classification
    raw_rallies = _extract_segments(frame_active, total_frames)

    # Step 4: Post-filter by duration and merge close rallies
    rallies = _post_filter(
        raw_rallies, fps, min_rally_frames, min_gap_frames
    )

    # Step 5: Assign IDs and compute timestamps
    result = []
    for i, rally in enumerate(rallies):
        result.append({
            "id": i + 1,
            "start_frame": rally["start"],
            "end_frame": rally["end"],
            "start_timestamp": rally["start"] / fps,
            "end_timestamp": rally["end"] / fps,
            "duration_seconds": (rally["end"] - rally["start"]) / fps,
        })

    return result


def _compute_gradients(
    positions: Dict[int, Dict], total_frames: int
) -> List[float]:
    """
    Compute frame-to-frame distance gradient.
    Returns a list of gradient values indexed by frame number.
    Gradient is 0 if either frame has no visible shuttle.
    """
    gradients = [0.0] * total_frames

    for frame in range(1, total_frames):
        curr = positions.get(frame)
        prev = positions.get(frame - 1)

        if (curr and curr.get("visible") and prev and prev.get("visible")):
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            gradients[frame] = math.sqrt(dx * dx + dy * dy)

    return gradients


def _classify_frames(
    gradients: List[float],
    total_frames: int,
    window_size: int,
    zero_ratio: float,
) -> List[bool]:
    """
    Classify each frame as active (True) or idle (False).

    A frame is idle if, within a window centered on it, >=zero_ratio
    of frames have zero (or near-zero) gradient.
    """
    # Threshold for "zero" gradient: shuttle moved less than 2 pixels
    ZERO_THRESHOLD = 2.0

    is_zero = [g < ZERO_THRESHOLD for g in gradients]

    # Compute rolling count of zero-gradient frames
    frame_active = [False] * total_frames
    half_w = window_size // 2

    # Use a sliding window for efficiency
    zero_count = sum(is_zero[0:min(window_size, total_frames)])

    for center in range(total_frames):
        win_start = max(0, center - half_w)
        win_end = min(total_frames, center + half_w + 1)
        actual_window = win_end - win_start

        # Maintain running count (recalculate at boundaries for correctness)
        if center == 0 or win_start == 0:
            zero_count = sum(is_zero[win_start:win_end])
        else:
            # Remove element that left the window
            old_start = max(0, (center - 1) - half_w)
            if old_start < win_start and old_start < total_frames:
                if is_zero[old_start]:
                    zero_count -= 1
            # Add element that entered the window
            new_end = min(total_frames, center + half_w + 1) - 1
            old_end = min(total_frames, (center - 1) + half_w + 1) - 1
            if new_end > old_end and new_end < total_frames:
                if is_zero[new_end]:
                    zero_count += 1

        zero_fraction = zero_count / max(actual_window, 1)
        frame_active[center] = zero_fraction < zero_ratio

    return frame_active


def _extract_segments(
    frame_active: List[bool], total_frames: int
) -> List[Dict]:
    """Extract contiguous active segments as raw rallies."""
    segments = []
    in_rally = False
    start = 0

    for i in range(total_frames):
        if frame_active[i] and not in_rally:
            start = i
            in_rally = True
        elif not frame_active[i] and in_rally:
            segments.append({"start": start, "end": i - 1})
            in_rally = False

    # Close final segment
    if in_rally:
        segments.append({"start": start, "end": total_frames - 1})

    return segments


def _post_filter(
    segments: List[Dict],
    fps: float,
    min_rally_frames: int,
    min_gap_frames: int,
) -> List[Dict]:
    """
    Post-filter rally segments:
    1. Merge segments that are very close together (gap < min_gap)
    2. Remove segments shorter than min_rally_duration
    """
    if not segments:
        return []

    # Merge close segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if gap < min_gap_frames:
            # Merge: extend previous segment
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Filter by minimum duration
    filtered = [s for s in merged if (s["end"] - s["start"]) >= min_rally_frames]

    return filtered


def compute_rally_stats(
    rallies: List[Dict],
    shuttle_positions: Dict[int, Dict],
    fps: float,
) -> Dict:
    """
    Compute summary statistics from detected rallies.

    Returns:
        Dict with rally statistics for logging/display.
    """
    if not rallies:
        return {
            "total_rallies": 0,
            "total_rally_time_s": 0,
            "total_idle_time_s": 0,
            "rally_percentage": 0,
            "avg_rally_duration_s": 0,
            "min_rally_duration_s": 0,
            "max_rally_duration_s": 0,
        }

    durations = [r["duration_seconds"] for r in rallies]
    total_rally_time = sum(durations)

    # Total video time from first rally start to last rally end
    first_start = rallies[0]["start_timestamp"]
    last_end = rallies[-1]["end_timestamp"]
    total_time = last_end - first_start if last_end > first_start else 0

    return {
        "total_rallies": len(rallies),
        "total_rally_time_s": round(total_rally_time, 1),
        "total_idle_time_s": round(max(0, total_time - total_rally_time), 1),
        "rally_percentage": round(
            100 * total_rally_time / total_time if total_time > 0 else 0, 1
        ),
        "avg_rally_duration_s": round(total_rally_time / len(rallies), 1),
        "min_rally_duration_s": round(min(durations), 1),
        "max_rally_duration_s": round(max(durations), 1),
    }
