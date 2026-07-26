"""
Offline regression suite for rally grouping and clip padding.

Usage:
    python backend/scripts/verify_rally_bounds.py

No GPU, no video, no network — these are pure functions, so this runs in
milliseconds and is the only executable guard the rally-bounds logic has.
Run it after ANY change to:
    backend/rally_detection_shot_gap.py   (detect_rallies_from_shots)
    backend/modal_supabase_processor.py   (pad_rally_windows / clip constants)
    src/composables/useAdvancedAnalytics.ts  (the TS twin of the grouping loop
                                              — keep the two in sync by hand)

Covers the two defects fixed on 2026-07-25 (see
docs/2026-07-25-rally-detection-clipping-audit.md §1 and §4):
  * an isolated trailing shot welding the whole inter-rally gap onto the
    previous rally's end;
  * clips cut at first-shot/last-shot contact, missing the serve and the
    shuttle's landing.

Exits non-zero on the first failing assertion group.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rally_detection_shot_gap import RALLY_GAP_SECONDS  # noqa: E402
from modal_supabase_processor import (  # noqa: E402
    CLIP_POST_ROLL_S,
    CLIP_PRE_ROLL_S,
    pad_rally_windows,
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def group(shots: list[dict]) -> list[dict]:
    """Run the real rally-grouping loop over hand-built shots.

    detect_rallies_from_shots derives shots from frames via detect_all_shots;
    stubbing that lets us drive the grouping loop directly with exact shot
    timings instead of reverse-engineering a shuttle trajectory.
    """
    import rally_detection_shot_gap as rg

    original = rg.detect_all_shots
    rg.detect_all_shots = lambda frames, fps, require_players=True: shots
    try:
        # Frames only need to satisfy the length guard and the shuttle-active
        # window check (>=25% of frames in the rally window show a shuttle).
        frames = [
            {"frame": i, "timestamp": i / 30.0, "shuttle_position": {"x": 1.0, "y": 1.0}}
            for i in range(2000)
        ]
        return rg.detect_rallies_from_shots(frames, 30.0)
    finally:
        rg.detect_all_shots = original


def S(ts: float) -> dict:
    return {"frame": int(ts * 30), "timestamp": ts}


def R(i: int, s: float, e: float) -> dict:
    return {
        "id": i,
        "start_timestamp": s,
        "end_timestamp": e,
        "duration_seconds": e - s,
        "start_frame": int(s * 30),
        "end_frame": int(e * 30),
    }


def test_weld_bug() -> None:
    print("=" * 72)
    print("1. Isolated trailing shot must NOT extend the last rally")
    print("=" * 72)
    r = group([S(1.0), S(2.0), S(3.0), S(30.0)])
    print(f"  shots 1,2,3 + lone shot at 30 -> {[(x['start_timestamp'], x['end_timestamp']) for x in r]}")
    check("one rally emitted", len(r) == 1, f"got {len(r)}")
    check(
        "rally ends at the last real shot (3.0s), not 30.0s",
        bool(r) and abs(r[0]["end_timestamp"] - 3.0) < 1e-6,
        f"end={r[0]['end_timestamp'] if r else None}",
    )


def test_final_shot_inside_gap_is_kept() -> None:
    print()
    print("=" * 72)
    print("2. A final shot INSIDE the gap threshold must still be included")
    print("=" * 72)
    r = group([S(1.0), S(2.0), S(3.0), S(4.5)])
    print(f"  all gaps <= {RALLY_GAP_SECONDS} -> {[(x['start_timestamp'], x['end_timestamp']) for x in r]}")
    check("one rally", len(r) == 1, f"got {len(r)}")
    check(
        "includes the final shot (ends 4.5s)",
        bool(r) and abs(r[0]["end_timestamp"] - 4.5) < 1e-6,
        f"end={r[0]['end_timestamp'] if r else None}",
    )


def test_two_shot_rally_boundary() -> None:
    print()
    print("=" * 72)
    print("3. Boundary: a 2-shot rally whose 2nd shot is also the LAST shot")
    print("=" * 72)
    # The exact case the weld fix touches: is_last is True and gap <= threshold,
    # so end_idx must be i+1 or the rally is dropped entirely.
    r = group([S(1.0), S(2.5)])
    print(f"  2 shots, gap 1.5s (<= {RALLY_GAP_SECONDS}) -> "
          f"{[(x['start_timestamp'], x['end_timestamp']) for x in r]}")
    check("serve+return kept as a rally", len(r) == 1, f"got {len(r)}")
    check(
        "spans both shots (1.0 -> 2.5)",
        bool(r) and abs(r[0]["start_timestamp"] - 1.0) < 1e-6
        and abs(r[0]["end_timestamp"] - 2.5) < 1e-6,
    )


def test_split_still_works() -> None:
    print()
    print("=" * 72)
    print("4. Two genuine rallies still split, both kept")
    print("=" * 72)
    r = group([S(1.0), S(2.0), S(3.0), S(20.0), S(21.0), S(22.0)])
    print(f"  -> {[(x['start_timestamp'], x['end_timestamp']) for x in r]}")
    check("two rallies", len(r) == 2, f"got {len(r)}")
    check("second rally ends at 22.0", len(r) == 2 and abs(r[1]["end_timestamp"] - 22.0) < 1e-6)


def test_bogus_gap_rally_rejected() -> None:
    print()
    print("=" * 72)
    print("5. Two shots separated by a big gap are not a rally")
    print("=" * 72)
    r = group([S(1.0), S(30.0)])
    print(f"  -> {[(x['start_timestamp'], x['end_timestamp']) for x in r]}")
    check("no rally emitted (was a bogus 29s rally before the fix)", len(r) == 0, f"got {len(r)}")


def test_padding() -> None:
    print()
    print("=" * 72)
    print(f"6. Clip padding (pre={CLIP_PRE_ROLL_S}s, post={CLIP_POST_ROLL_S}s)")
    print("=" * 72)

    p = pad_rally_windows([R(1, 60.0, 70.0)], video_duration=300.0)
    print(f"  isolated rally 60-70 -> clip {p[0]['clip_start']:.2f}-{p[0]['clip_end']:.2f}")
    check("full pre-roll applied", abs(p[0]["clip_start"] - (60.0 - CLIP_PRE_ROLL_S)) < 1e-6)
    check("full post-roll applied", abs(p[0]["clip_end"] - (70.0 + CLIP_POST_ROLL_S)) < 1e-6)
    check(
        "detected window preserved on the dict (results.json stays unpadded)",
        p[0]["start_timestamp"] == 60.0 and p[0]["end_timestamp"] == 70.0,
    )

    p = pad_rally_windows([R(1, 0.5, 10.0)], video_duration=300.0)
    print(f"  rally at video start 0.5-10 -> clip {p[0]['clip_start']:.2f}-{p[0]['clip_end']:.2f}")
    check("clip_start never negative", p[0]["clip_start"] >= 0.0, f"got {p[0]['clip_start']}")

    p = pad_rally_windows([R(1, 90.0, 99.0)], video_duration=100.0)
    print(f"  rally at EOF 90-99 (duration 100) -> clip {p[0]['clip_start']:.2f}-{p[0]['clip_end']:.2f}")
    check("clip_end clamped to duration", abs(p[0]["clip_end"] - 100.0) < 1e-6, f"got {p[0]['clip_end']}")

    p = pad_rally_windows([R(1, 10.0, 20.0), R(2, 21.0, 30.0)], video_duration=300.0)
    print(f"  adjacent 10-20 / 21-30 -> clip1 {p[0]['clip_start']:.2f}-{p[0]['clip_end']:.2f}, "
          f"clip2 {p[1]['clip_start']:.2f}-{p[1]['clip_end']:.2f}")
    # Padding may share the dead air between rallies, but must never reach
    # into a neighbour's detected rally window.
    check("clip1 does not enter rally 2's window", p[0]["clip_end"] <= 21.0 + 1e-9, f"got {p[0]['clip_end']}")
    check("clip2 does not enter rally 1's window", p[1]["clip_start"] >= 20.0 - 1e-9, f"got {p[1]['clip_start']}")

    # refine_rallies can currently emit overlapping rallies (audit §5, open).
    # Padding must never shrink a window below what was detected.
    p = pad_rally_windows([R(1, 10.0, 24.0), R(2, 22.0, 30.0)], video_duration=300.0)
    print(f"  OVERLAPPING input 10-24 / 22-30 -> clip1 {p[0]['clip_start']:.2f}-{p[0]['clip_end']:.2f}, "
          f"clip2 {p[1]['clip_start']:.2f}-{p[1]['clip_end']:.2f}")
    check("clip1 covers its whole detected window", p[0]["clip_start"] <= 10.0 and p[0]["clip_end"] >= 24.0)
    check("clip2 covers its whole detected window", p[1]["clip_start"] <= 22.0 and p[1]["clip_end"] >= 30.0)

    src = [R(1, 60.0, 70.0)]
    pad_rally_windows(src, video_duration=300.0)
    check("caller's dicts not mutated", "clip_start" not in src[0])

    p = pad_rally_windows([R(2, 60.0, 70.0), R(1, 10.0, 20.0)], video_duration=300.0)
    check("output sorted by start_timestamp", p[0]["start_timestamp"] < p[1]["start_timestamp"])

    p = pad_rally_windows([R(1, 60.0, 70.0)], video_duration=None)
    check("missing duration probe still pads", p[0]["clip_end"] > 70.0)


def main() -> None:
    test_weld_bug()
    test_final_shot_inside_gap_is_kept()
    test_two_shot_rally_boundary()
    test_split_still_works()
    test_bogus_gap_rally_rejected()
    test_padding()

    print()
    print("=" * 72)
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("OK: all rally-bounds checks passed.")


if __name__ == "__main__":
    main()
