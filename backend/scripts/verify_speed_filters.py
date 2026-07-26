"""
Offline regression suite for player speed/distance filtering.

Usage:
    python backend/scripts/verify_speed_filters.py

No GPU, no video, no network. Run after any change to speed_calc.py or to the
in-loop filter in modal_supabase_processor._run_full_yolo_loop.

Guards the fixes for docs/2026-07-25-metric-pipeline-audit.md §5 and §6:

  §5  three implementations of player speed/distance disagreed. The Phase 2
      loop capped at 8.5 m/s (30.6 km/h) with a resolution-scaled pixel gate;
      speed_calc capped at 25 km/h with a flat 80px gate (~1.7x stricter at
      1080p). So results.json — the durable record the KMP app will read —
      carried different numbers from what the browser displayed. speed_calc is
      now the single source of truth and the loop imports from it.

  §6  MAX_DISTANCE_PER_FRAME_M = 0.25 was a metres-PER-FRAME gate hardcoded for
      30fps. Since speed_kmh == (d_metres / frames_elapsed) * fps * 3.6, that is
      algebraically the speed cap divided by fps: binding at 25fps (implying a
      22.5 km/h ceiling that rejected legitimate movement) and inert at 60fps
      (implying 54 km/h). It is deleted; the speed cap does the job at any rate.

The headline property is fps-invariance: identical physical motion must yield
the same reported speed regardless of the source frame rate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from speed_calc import (  # noqa: E402
    MAX_REALISTIC_SPEED_KMH,
    MAX_REALISTIC_SPEED_MPS,
    calculate_speeds_from_skeleton,
    max_frame_jump_pixels,
    median_speed_rejects,
)

FAILURES: list[str] = []

# A synthetic court whose homography makes the maths legible: the court
# occupies x 100..700 px and y 100..1440 px, so vertically 1340px maps to
# 13.4m — exactly 0.01 m per pixel.
COURT_KEYPOINTS = {
    "top_left": [100.0, 100.0],
    "top_right": [700.0, 100.0],
    "bottom_right": [700.0, 1440.0],
    "bottom_left": [100.0, 1440.0],
}
M_PER_PX_Y = 0.01
VIDEO_W, VIDEO_H = 1920, 1620


def check(label: str, cond: bool, detail: str = "") -> None:
    suffix = f"  {detail}" if (detail and not cond) else ""
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{suffix}")
    if not cond:
        FAILURES.append(label)


def frames_for_speed(speed_kmh: float, fps: float, n_frames: int = 20) -> list[dict]:
    """Skeleton frames for one player gliding down the court at a constant speed."""
    speed_mps = speed_kmh / 3.6
    px_per_frame = speed_mps / (M_PER_PX_Y * fps)
    frames = []
    for i in range(n_frames):
        y = 200.0 + i * px_per_frame
        frames.append({
            "frame": i,
            "timestamp": i / fps,
            "players": [{"player_id": 0, "center": {"x": 400.0, "y": y}}],
        })
    return frames


def measured_max_kmh(speed_kmh: float, fps: float) -> float:
    out = calculate_speeds_from_skeleton(
        skeleton_data=frames_for_speed(speed_kmh, fps),
        fps=fps,
        video_width=VIDEO_W,
        video_height=VIDEO_H,
        manual_court_keypoints=COURT_KEYPOINTS,
    )
    stats = out["statistics"].get("0")
    return stats["max"]["speed_kmh"] if stats else 0.0


def test_fps_invariance() -> None:
    print("=" * 74)
    print("§6  Identical physical motion must measure the same at any frame rate")
    print("=" * 74)
    target = 20.0  # km/h — comfortably inside the cap at every rate
    results = {fps: measured_max_kmh(target, fps) for fps in (25.0, 30.0, 50.0, 60.0)}
    for fps, got in results.items():
        print(f"    {fps:>4.0f}fps -> {got:5.2f} km/h")
    spread = max(results.values()) - min(results.values())
    check(
        f"a {target} km/h player measures the same across 25/30/50/60fps",
        spread < 0.5,
        f"spread={spread:.2f} km/h across {results}",
    )
    for fps, got in results.items():
        check(
            f"  measured value is correct at {fps:.0f}fps",
            abs(got - target) < 0.5,
            f"expected ~{target}, got {got:.2f}",
        )


def test_old_distance_gate_no_longer_binds() -> None:
    print()
    print("=" * 74)
    print("§6  The deleted metres-per-frame gate must not clip legitimate movement")
    print("=" * 74)
    # 23.5 km/h is under the 25 km/h cap but exceeds the old 0.25 m/frame gate
    # at 25fps (23.5 km/h -> 0.261 m/frame), so it used to be zeroed there while
    # passing at 30fps. That inconsistency is the bug.
    target = 23.5
    got_25 = measured_max_kmh(target, 25.0)
    got_30 = measured_max_kmh(target, 30.0)
    print(f"    {target} km/h at 25fps -> {got_25:5.2f} km/h  (old gate implied a 22.5 km/h ceiling)")
    print(f"    {target} km/h at 30fps -> {got_30:5.2f} km/h")
    check(
        "movement below the speed cap survives at 25fps",
        abs(got_25 - target) < 0.5,
        f"got {got_25:.2f}, expected ~{target}",
    )
    check(
        "25fps and 30fps agree",
        abs(got_25 - got_30) < 0.5,
        f"{got_25:.2f} vs {got_30:.2f}",
    )


def test_speed_cap_still_enforced() -> None:
    print()
    print("=" * 74)
    print("§5  The speed cap must still reject impossible movement, at every rate")
    print("=" * 74)
    print(f"    MAX_REALISTIC_SPEED_KMH = {MAX_REALISTIC_SPEED_KMH}")
    for fps in (25.0, 30.0, 60.0):
        got = measured_max_kmh(45.0, fps)
        check(
            f"45 km/h is rejected at {fps:.0f}fps",
            got == 0.0,
            f"got {got:.2f} km/h",
        )
    check(
        "MAX_REALISTIC_SPEED_MPS is consistent with the km/h constant",
        abs(MAX_REALISTIC_SPEED_MPS * 3.6 - MAX_REALISTIC_SPEED_KMH) < 1e-9,
    )


def test_frame_jump_gate_scales() -> None:
    print()
    print("=" * 74)
    print("§5  The ID-swap pixel gate must scale with resolution and frame rate")
    print("=" * 74)
    at_720 = max_frame_jump_pixels(1280, 720, 30.0)
    at_1080 = max_frame_jump_pixels(1920, 1080, 30.0)
    at_4k = max_frame_jump_pixels(3840, 2160, 30.0)
    print(f"    30fps: 720p={at_720:.0f}px  1080p={at_1080:.0f}px  4K={at_4k:.0f}px")
    check("gate grows with resolution", at_720 < at_1080 < at_4k)

    at_1080_60 = max_frame_jump_pixels(1920, 1080, 60.0)
    print(f"    1080p: 30fps={at_1080:.0f}px  60fps={at_1080_60:.0f}px")
    check(
        "gate shrinks as frame rate rises (each frame spans less time)",
        at_1080_60 < at_1080,
        f"30fps={at_1080:.0f}, 60fps={at_1080_60:.0f}",
    )
    check(
        "a zero/garbage fps does not blow up the gate",
        max_frame_jump_pixels(1920, 1080, 0.0) == at_1080,
    )


def test_median_filter_helper() -> None:
    print()
    print("=" * 74)
    print("§5  The shared median-spike helper behaves as both call sites expect")
    print("=" * 74)
    check("no judgement with fewer than 3 samples", not median_speed_rejects([9.0, 9.0], 99.0))
    check("rejects a 3x spike over an established median", median_speed_rejects([5.0, 5.0, 5.0], 20.0))
    check("accepts a normal sample", not median_speed_rejects([5.0, 5.0, 5.0], 7.0))
    check(
        "does not judge against a tiny median",
        not median_speed_rejects([0.5, 0.5, 0.5], 4.0),
    )


def test_no_duplicate_thresholds_in_worker() -> None:
    print()
    print("=" * 74)
    print("§5  The worker must not reintroduce its own private thresholds")
    print("=" * 74)
    src = (BACKEND_DIR / "modal_supabase_processor.py").read_text()
    banned = {
        r"MAX_VALID_SPEED_MPS\s*=": "a private speed cap (was 8.5 m/s = 30.6 km/h)",
        r"MAX_DISTANCE_PER_FRAME\s*=": "the fps-dependent metres-per-frame gate",
        r"MAX_PX_PER_FRAME\s*=\s*max\(80": "a hand-rolled pixel gate",
    }
    for pattern, what in banned.items():
        hits = re.findall(pattern, src)
        check(
            f"no {what}",
            not hits,
            f"found {len(hits)} occurrence(s) of /{pattern}/",
        )
    check(
        "the worker imports the shared thresholds from speed_calc",
        "from speed_calc import" in src and "MAX_REALISTIC_SPEED_MPS" in src,
    )


def test_fps_normalization() -> None:
    print()
    print("=" * 74)
    print("§9  An unusable probed frame rate must be substituted, not propagated")
    print("=" * 74)
    from modal_supabase_processor import DEFAULT_FPS, normalize_fps

    for bad in (0, 0.0, -5, None, "not a number", float("nan"), float("inf")):
        fps, substituted = normalize_fps(bad)
        check(
            f"normalize_fps({bad!r}) substitutes the default",
            substituted and fps == DEFAULT_FPS,
            f"got {(fps, substituted)}",
        )
    for good in (25.0, 29.97, 60):
        fps, substituted = normalize_fps(good)
        check(
            f"normalize_fps({good!r}) is passed through untouched",
            not substituted and abs(fps - float(good)) < 1e-9,
            f"got {(fps, substituted)}",
        )
    # The Phase 2 loop divides by fps for per-frame dt, so a 0 reaching it is a
    # ZeroDivisionError, not just a bad number. Both callers normalize and the
    # loop guards its own boundary; assert the primitive they all rely on.
    check(
        "the substituted value is safe to divide by",
        normalize_fps(0)[0] > 0,
    )


def main() -> None:
    test_fps_invariance()
    test_old_distance_gate_no_longer_binds()
    test_speed_cap_still_enforced()
    test_frame_jump_gate_scales()
    test_median_filter_helper()
    test_fps_normalization()
    test_no_duplicate_thresholds_in_worker()

    print()
    print("=" * 74)
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("OK: all speed-filter checks passed.")


if __name__ == "__main__":
    main()
