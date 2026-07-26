"""
Offline regression suite for PlayerIdentityTracker invariants.

Usage:
    python backend/scripts/verify_tracker_invariants.py

No GPU, no video, no network. Run after any change to PlayerIdentityTracker,
to the net-keypoint plumbing in `_process_analytics_worker`, or to
CourtSetup.vue's keypoint emission.

Context: an uploaded video is always exactly ONE game and the players never
switch sides (confirmed 2026-07-25). Court side is therefore ground truth for
identity across a whole clip, and the tracker now treats it as a HARD CONSTRAINT:
a skeleton may only be assigned to the player who owns its side of the net.

What this suite pins down, from docs/2026-07-25-metric-pipeline-audit.md:

  Constraint  a skeleton is never assigned across the net; a player with no
              claimable skeleton is left unassigned rather than guessed at; a
              narrow net band keeps tight net play from dropping frames; and the
              mechanisms the constraint supersedes (swap detection,
              majority-vote smoothing) stay deleted. Swaps are not detected and
              repaired — they cannot be represented.
  §1          degenerate net keypoints ((0,0)/(0,0)) would collapse every
              court-side decision to "bottom". DEFENSIVE: no live writer can
              produce this today (CourtSetup.vue requires all 12 keypoints), so
              this guards future writers — the KMP app, direct DB edits, a
              relaxed CourtSetup.
  §2          calibration must never lock both players to the same side. Sides
              are now derived after the midline is refined, and an
              unrepresentative window extends calibration instead of locking an
              impossible state. This one was live.

All assertions should pass. Run with --document to print observed behaviour
without asserting.

Not covered (deliberately): §3 the end-change identity flip — closed, impossible
under the fixed-sides invariant; §8 the >2-skeleton majority-vote defect — the
mechanism it lived in is now deleted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modal_supabase_processor import (  # noqa: E402
    PlayerIdentityTracker,
    valid_net_line,
)

FAILURES: list[str] = []
DOCUMENT_ONLY = False


def check(label: str, cond: bool, detail: str = "") -> None:
    # `detail` describes the failure, so only show it when the check fails —
    # printing it next to a PASS reads as if the assertion had tripped.
    suffix = f"  {detail}" if (detail and not cond) else ""
    if DOCUMENT_ONLY:
        print(f"  [{'ok' if cond else 'CURRENT BUG'}] {label}{suffix}")
        return
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{suffix}")
    if not cond:
        FAILURES.append(label)


def skel(x: float, y: float, name: str, track_id: int = -1) -> dict:
    return {"center": (x, y), "kpts": name, "area": 1000, "track_id": track_id}


def calibrate(t: PlayerIdentityTracker, skeletons: list[dict], limit: int = 100) -> int:
    """Drive calibration exactly as match_skeletons does: only while incomplete."""
    frame = 0
    while not t.calibration_complete and frame < limit:
        frame += 1
        t.frames_processed = frame
        t._run_calibration(skeletons)
    return frame


def test_degenerate_net_keypoints() -> None:
    print("=" * 74)
    print("§1  Degenerate net keypoints must be rejected before reaching the tracker")
    print("=" * 74)
    # The guard lives in the CALLER (_run_full_yolo_loop), not in the tracker,
    # because the tracker cannot tell a bad net line from a real one. So assert
    # on valid_net_line() itself.
    W, H = 1920.0, 1080.0
    cases = [
        ("both endpoints at the origin (unplaced keypoints)", (0.0, 0.0), (0.0, 0.0), False),
        ("identical endpoints", (900.0, 500.0), (900.0, 500.0), False),
        ("insufficient horizontal separation", (900.0, 500.0), (905.0, 505.0), False),
        ("negative coordinate", (-10.0, 500.0), (1500.0, 500.0), False),
        ("endpoint beyond frame width", (100.0, 500.0), (5000.0, 500.0), False),
        ("realistic net line", (150.0, 520.0), (1770.0, 505.0), True),
        ("realistic tilted net line", (200.0, 610.0), (1700.0, 430.0), True),
    ]
    for label, nl, nr, expected in cases:
        got = valid_net_line(nl, nr, W, H)
        check(f"valid_net_line: {label} -> {expected}", got is expected, f"got {got}")

    # And the fallback the guard produces (net_line=None) must itself be sane:
    # a tracker with no net line still has to calibrate to opposite sides.
    t = PlayerIdentityTracker(video_height=H, fps=30.0, video_width=W)
    opposite = [skel(700.0, 200.0, "A", 1), skel(1100.0, 900.0, "B", 2)]
    calibrate(t, opposite)
    print(f"  midline fallback: court_sides={t.court_sides} midline={t.court_midline_y}")
    check(
        "midline fallback still calibrates players to opposite sides",
        t.court_sides[0] != t.court_sides[1],
        f"both -> {t.court_sides[0]!r}",
    )
    check(
        "midline fallback still assigns a single visible player",
        len(t.match_skeletons([skel(700.0, 200.0, "X", 9)], 200)) == 1,
    )


def test_same_side_calibration() -> None:
    print()
    print("=" * 74)
    print("§2  Calibration must never lock both players to the same court side")
    print("=" * 74)
    # Both players on one half for the whole 15-frame window: a warm-up, a
    # handshake, an intro shot, or a serve with both crowding the net.
    t = PlayerIdentityTracker(video_height=1080.0, fps=30.0, video_width=1920.0)
    both_bottom = [skel(700.0, 800.0, "A", 1), skel(1100.0, 860.0, "B", 2)]
    frames = calibrate(t, both_bottom)
    print(f"  calibration completed at frame {frames}")
    print(f"  court_sides={t.court_sides}  refined court_midline_y={t.court_midline_y}")
    check(
        "the two players hold different court sides",
        t.court_sides[0] != t.court_sides[1],
        f"both -> {t.court_sides[0]!r}",
    )
    check(
        "a single visible player is assignable after calibration",
        len(t.match_skeletons([skel(700.0, 800.0, "X", 9)], 200)) == 1,
    )

    # With a REAL net line, both players genuinely on one half is not a
    # geometry error — it's an unrepresentative window (warm-up, walk-on).
    # Calibration must wait rather than lock an impossible state, then fall
    # back to a relative split if the situation never resolves.
    t2 = PlayerIdentityTracker(
        video_height=1080.0, fps=30.0, video_width=1920.0,
        net_left=(150.0, 520.0), net_right=(1770.0, 520.0),
    )
    both_below_net = [skel(700.0, 800.0, "A", 1), skel(1100.0, 860.0, "B", 2)]
    for frame in range(1, t2.CALIBRATION_FRAMES + 5):
        t2.frames_processed = frame
        if not t2.calibration_complete:
            t2._run_calibration(both_below_net)
    check(
        "calibration WAITS while both players are on one side of a real net",
        not t2.calibration_complete,
        f"completed early at frame {t2.frames_processed}",
    )

    frames = calibrate(t2, both_below_net, limit=t2.CALIBRATION_MAX_FRAMES + 10)
    print(f"  after waiting: completed at frame {frames}, "
          f"court_sides={t2.court_sides}, net_line={t2.net_line}")
    check(
        "calibration eventually completes rather than hanging forever",
        t2.calibration_complete,
    )
    check(
        "the fallback split still yields opposite sides",
        t2.court_sides[0] != t2.court_sides[1],
        f"both -> {t2.court_sides[0]!r}",
    )
    check(
        "the contradicted net line is dropped so live classification agrees",
        t2.net_line is None,
        f"net_line still {t2.net_line}",
    )


def test_hard_side_constraint() -> None:
    print()
    print("=" * 74)
    print("Constraint  A skeleton may only ever be assigned to the player who")
    print("            owns its side of the net")
    print("=" * 74)
    t = PlayerIdentityTracker(video_height=1080.0, fps=30.0, video_width=1920.0)
    top = skel(700.0, 200.0, "TOP", 1)
    bottom = skel(1100.0, 900.0, "BOTTOM", 2)
    calibrate(t, [top, bottom])
    top_pid = 0 if t.court_sides[0] == "top" else 1
    bottom_pid = 1 - top_pid
    print(f"  court_sides={t.court_sides} -> top player is pid {top_pid}")

    # Feed the SAME two skeletons but hand them over in reversed list order, and
    # keep doing it. A cost-based global assignment could flip; a hard side
    # constraint cannot.
    seen = []
    for frame in range(200, 240):
        order = [top, bottom] if frame % 2 == 0 else [bottom, top]
        result = t.match_skeletons(order, frame)
        seen.append({name: pid for pid, name, _ in result})
    always_correct = all(
        m.get("TOP") == top_pid and m.get("BOTTOM") == bottom_pid for m in seen
    )
    check(
        "assignment follows court side across 40 frames of reordered input",
        always_correct,
        f"observed mappings: {seen[:4]} ...",
    )
    check(
        "court_sides never mutates during matching",
        t.court_sides[top_pid] == "top" and t.court_sides[bottom_pid] == "bottom",
        f"got {t.court_sides}",
    )

    # A wrong-side skeleton must NOT be handed to the other player. Both
    # skeletons on the bottom half: the top player has no claimable skeleton.
    t2 = PlayerIdentityTracker(video_height=1080.0, fps=30.0, video_width=1920.0)
    calibrate(t2, [top, bottom])
    top_pid2 = 0 if t2.court_sides[0] == "top" else 1
    before_unsplittable = t2.frames_unsplittable
    result = t2.match_skeletons(
        [skel(700.0, 880.0, "STRAY", 3), skel(1100.0, 950.0, "REAL", 2)], 300
    )
    pids = [pid for pid, _, _ in result]
    print(f"  both skeletons on the bottom half -> assigned pids {pids}")
    check(
        "the top player is left unassigned rather than given a bottom skeleton",
        top_pid2 not in pids,
        f"top player {top_pid2} was assigned anyway",
    )
    check(
        "exactly one player is placed",
        len(result) == 1,
        f"got {len(result)} assignments",
    )
    check(
        "the unsplittable-frame counter records it",
        t2.frames_unsplittable == before_unsplittable + 1,
        f"counter went {before_unsplittable} -> {t2.frames_unsplittable}",
    )

    # A single-skeleton frame is NOT a constraint failure — a disjoint pair is
    # impossible by definition — and single-skeleton frames are common on real
    # footage. Counting them together would bury the signal above in noise.
    t3 = PlayerIdentityTracker(video_height=1080.0, fps=30.0, video_width=1920.0)
    calibrate(t3, [top, bottom])
    t3.match_skeletons([skel(700.0, 200.0, "ONLY", 1)], 310)
    print(f"  one skeleton visible -> unsplittable={t3.frames_unsplittable}, "
          f"single={t3.frames_single_skeleton}")
    check(
        "a single-skeleton frame does not count as a constraint failure",
        t3.frames_unsplittable == 0,
        f"frames_unsplittable={t3.frames_unsplittable}",
    )
    check(
        "it is counted separately as a coverage signal",
        t3.frames_single_skeleton == 1,
        f"frames_single_skeleton={t3.frames_single_skeleton}",
    )


def test_net_band_hysteresis() -> None:
    print()
    print("=" * 74)
    print("Constraint  A narrow band around the net keeps tight net play from")
    print("            dropping frames")
    print("=" * 74)
    t = PlayerIdentityTracker(
        video_height=1080.0, fps=30.0, video_width=1920.0,
        net_left=(150.0, 540.0), net_right=(1770.0, 540.0),
    )
    calibrate(t, [skel(700.0, 200.0, "TOP", 1), skel(1100.0, 900.0, "BOTTOM", 2)])
    top_pid = 0 if t.court_sides[0] == "top" else 1
    print(f"  NET_BAND_PX={t.NET_BAND_PX:.0f}px, net y=540")

    # Front foot lunges just past the net line — a few px onto the wrong side.
    just_over = 540.0 + t.NET_BAND_PX * 0.5
    result = t.match_skeletons(
        [skel(900.0, just_over, "LUNGE", 1), skel(1100.0, 900.0, "BOTTOM", 2)], 400
    )
    mapping = {name: pid for pid, name, _ in result}
    print(f"  top player lunging to y={just_over:.0f} -> {mapping}")
    check(
        "a skeleton inside the band is still claimable by the top player",
        mapping.get("LUNGE") == top_pid,
        f"got {mapping}",
    )

    # Well past the band is genuinely the other side and must be refused.
    well_over = 540.0 + t.NET_BAND_PX * 10
    check(
        "a skeleton well beyond the band is NOT claimable by the top player",
        not t._in_net_band(well_over, 900.0)
        and top_pid not in [
            pid for pid, _, _ in t.match_skeletons(
                [skel(900.0, well_over, "DEEP", 1), skel(1100.0, 900.0, "BOTTOM", 2)], 401
            )
        ],
    )


def test_superseded_mechanisms_are_gone() -> None:
    print()
    print("=" * 74)
    print("Constraint  Swap detection and majority-vote smoothing must not")
    print("            come back — they cannot fire and would only add drift")
    print("=" * 74)
    t = PlayerIdentityTracker(video_height=1080.0, fps=30.0, video_width=1920.0)
    for gone in (
        "_detect_and_correct_swap",
        "_apply_majority_vote",
        "total_swaps_corrected",
        "swap_violation_count",
        "assignment_vote_history",
    ):
        check(f"{gone} is removed", not hasattr(t, gone))
    check(
        "the replacement health signals exist and stay separate",
        hasattr(t, "frames_unsplittable") and hasattr(t, "frames_single_skeleton"),
    )
    check(
        "get_stats reports them instead of a swap count",
        {"frames_unsplittable", "frames_single_skeleton"} <= set(t.get_stats())
        and "total_swaps_corrected" not in t.get_stats(),
        f"keys={sorted(t.get_stats())}",
    )


def main() -> None:
    global DOCUMENT_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--document", action="store_true",
        help="print observed behaviour without failing the run",
    )
    DOCUMENT_ONLY = ap.parse_args().document

    test_degenerate_net_keypoints()
    test_same_side_calibration()
    test_hard_side_constraint()
    test_net_band_hysteresis()
    test_superseded_mechanisms_are_gone()

    print()
    print("=" * 74)
    if DOCUMENT_ONLY:
        print("Documented current behaviour (no assertions enforced).")
        return
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("OK: all tracker invariants hold.")


if __name__ == "__main__":
    main()
