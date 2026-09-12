"""
Offline regression suite for annotation re-anchoring across a clip re-cut.

Usage:
    python backend/scripts/verify_annotation_shift.py

No GPU, no video, no network - this is a pure function, so it runs in
milliseconds. Run it after ANY change to:
    backend/modal_supabase_processor.py  (shift_annotation_offsets,
                                          cut_and_upload_rally_clips)

Covers the defect recorded as §8.5 of
badminton-rally-mobile/docs/plans/2026-08-31-web-analysis-pipeline-reference.md:
the Phase 2 re-cut upserts on (video_id, rally_index), so rally_clips.id
survives while the footage under it moves, and every note's offset into that
footage silently becomes an offset into different footage.

Exits non-zero on the first failing assertion group.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modal_supabase_processor import (  # noqa: E402
    ANNOTATION_SHIFT_EPSILON_S,
    shift_annotation_offsets,
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def note(note_id: str, offset: float) -> dict:
    return {"id": note_id, "timestamp_seconds": offset}


print("a clip that did not move leaves every note alone")
# The re-cut runs on every Phase 2, including ones whose boundaries agree with
# Phase 1's. Writing a row per note on every run would be a needless write and
# would churn updated_at on data nobody touched.
check(
    "no shift under the epsilon",
    shift_annotation_offsets(
        [note("a", 3.0), note("b", 7.5)],
        old_clip_start=10.0,
        new_clip_start=10.0 + ANNOTATION_SHIFT_EPSILON_S / 2,
        new_clip_end=30.0,
    )
    == [],
)

print("a clip that starts EARLIER pushes its notes later in the file")
# New footage is prepended, so the same moment of play now sits further into
# the clip. delta = old_start - new_start = +2.0.
shifted = shift_annotation_offsets(
    [note("a", 3.0)], old_clip_start=10.0, new_clip_start=8.0, new_clip_end=30.0
)
check("one row written", len(shifted) == 1, str(shifted))
check("offset moved by the boundary delta", shifted[0]["timestamp_seconds"] == 5.0, str(shifted))
check("not clamped", shifted[0]["clamped"] is False)
check("id carried", shifted[0]["id"] == "a")

print("a clip that starts LATER pulls its notes earlier in the file")
shifted = shift_annotation_offsets(
    [note("a", 6.0)], old_clip_start=10.0, new_clip_start=12.0, new_clip_end=30.0
)
check("offset moved by the boundary delta", shifted[0]["timestamp_seconds"] == 4.0, str(shifted))

print("a note that falls off the front is clamped and flagged, not moved silently")
# The alternative is a negative offset, which the apps coerce to 0 anyway
# (ClipDetailViewModel.addAnnotation does `coerceAtLeast(0f)`), losing the
# fact that the note no longer describes footage that exists.
shifted = shift_annotation_offsets(
    [note("a", 1.0)], old_clip_start=10.0, new_clip_start=14.0, new_clip_end=30.0
)
check("clamped to the window start", shifted[0]["timestamp_seconds"] == 0.0, str(shifted))
check("and flagged", shifted[0]["clamped"] is True)

print("a note that falls off the end is clamped to the window, not past it")
shifted = shift_annotation_offsets(
    [note("a", 18.0)], old_clip_start=10.0, new_clip_start=6.0, new_clip_end=26.0
)
# delta = +4.0 -> 22.0, but the window is only 26.0 - 6.0 = 20.0 long.
check("clamped to the window length", shifted[0]["timestamp_seconds"] == 20.0, str(shifted))
check("and flagged", shifted[0]["clamped"] is True)

print("a degenerate window does not produce a negative bound")
shifted = shift_annotation_offsets(
    [note("a", 5.0)], old_clip_start=10.0, new_clip_start=20.0, new_clip_end=20.0
)
check("clamped to zero", shifted[0]["timestamp_seconds"] == 0.0, str(shifted))

print("every note on a moved clip is re-anchored, not just the first")
shifted = shift_annotation_offsets(
    [note("a", 1.0), note("b", 2.0), note("c", 3.0)],
    old_clip_start=10.0,
    new_clip_start=9.0,
    new_clip_end=30.0,
)
check("three rows", len(shifted) == 3, str(shifted))
check(
    "each moved by the same delta",
    [row["timestamp_seconds"] for row in shifted] == [2.0, 3.0, 4.0],
    str(shifted),
)

print("an empty note list is not an error")
check(
    "no rows",
    shift_annotation_offsets([], old_clip_start=1.0, new_clip_start=2.0, new_clip_end=9.0) == [],
)

if FAILURES:
    print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("\nall checks passed")
