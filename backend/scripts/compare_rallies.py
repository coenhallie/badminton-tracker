"""
Compare the rallies arrays from two videos' results JSONs.

Useful for diagnosing rally-count differences between the pre-refactor monolithic
pipeline and the post-refactor Phase 1 output.

Usage:
    python -m backend.scripts.compare_rallies --old <old-video-id> --new <new-video-id>

Both video IDs must have `results_storage_path` set. The script downloads each
results JSON, extracts the rallies array, and prints a side-by-side diff with
timing information so you can see which rally is missing or shifted.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional


def _supabase_client():
    import sys
    from pathlib import Path
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from supabase_helpers import supabase_client  # type: ignore
    return supabase_client()


def _fetch_results(video_id: str) -> dict:
    sb = _supabase_client()
    row = sb.table("videos").select("id, filename, results_storage_path, total_frames").eq("id", video_id).single().execute().data
    assert row, f"Video {video_id} not found"
    assert row["results_storage_path"], f"Video {video_id} has no results_storage_path"
    blob = sb.storage.from_("results").download(row["results_storage_path"])
    payload = json.loads(blob.decode("utf-8"))
    return {"row": row, "payload": payload}


def _format_rally(r: dict, fps: float) -> str:
    sf = r.get("start_frame")
    ef = r.get("end_frame")
    ss = r.get("start_timestamp") or (sf / fps if fps and sf is not None else None)
    es = r.get("end_timestamp") or (ef / fps if fps and ef is not None else None)
    dur = (ef - sf) / fps if fps and sf is not None and ef is not None else None
    src = r.get("source") or r.get("origin") or "?"
    return (
        f"rally#{r.get('id', '?')} "
        f"frames=[{sf}..{ef}] "
        f"time=[{_fmt_t(ss)}..{_fmt_t(es)}] "
        f"dur={dur:.1f}s "
        f"src={src}"
    )


def _fmt_t(s: Optional[float]) -> str:
    if s is None:
        return "?"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def _match_by_overlap(a: list[dict], b: list[dict]) -> list[tuple[Optional[dict], Optional[dict], float]]:
    """Match rallies between two lists by max IoU on time intervals. Returns
    (a_rally_or_None, b_rally_or_None, iou) entries covering every rally."""
    used_b: set[int] = set()
    pairs: list[tuple[Optional[dict], Optional[dict], float]] = []
    for ra in a:
        best_idx = None
        best_iou = 0.0
        for i, rb in enumerate(b):
            if i in used_b:
                continue
            inter_start = max(ra["start_frame"], rb["start_frame"])
            inter_end = min(ra["end_frame"], rb["end_frame"])
            if inter_end <= inter_start:
                continue
            inter = inter_end - inter_start
            union = (ra["end_frame"] - ra["start_frame"]) + (rb["end_frame"] - rb["start_frame"]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None and best_iou > 0.1:
            used_b.add(best_idx)
            pairs.append((ra, b[best_idx], best_iou))
        else:
            pairs.append((ra, None, 0.0))
    for i, rb in enumerate(b):
        if i not in used_b:
            pairs.append((None, rb, 0.0))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Video ID with the higher rally count")
    parser.add_argument("--new", required=True, help="Video ID with the lower rally count")
    args = parser.parse_args()

    old = _fetch_results(args.old)
    new = _fetch_results(args.new)

    old_rallies = old["payload"].get("rallies") or []
    new_rallies = new["payload"].get("rallies") or []
    old_fps = float(old["payload"].get("fps") or 30.0)
    new_fps = float(new["payload"].get("fps") or 30.0)

    print(f"OLD video={args.old[:8]} filename={old['row'].get('filename')!r}")
    print(f"  fps={old_fps} total_frames={old['payload'].get('total_frames')} phase={old['payload'].get('phase', 'legacy')}")
    print(f"  rally count: {len(old_rallies)}")
    print()
    print(f"NEW video={args.new[:8]} filename={new['row'].get('filename')!r}")
    print(f"  fps={new_fps} total_frames={new['payload'].get('total_frames')} phase={new['payload'].get('phase', 'legacy')}")
    print(f"  rally count: {len(new_rallies)}")
    print()

    # Sort by start_frame for stable display
    old_sorted = sorted(old_rallies, key=lambda r: r.get("start_frame", 0))
    new_sorted = sorted(new_rallies, key=lambda r: r.get("start_frame", 0))

    print("=== OLD rallies ===")
    for r in old_sorted:
        print("  " + _format_rally(r, old_fps))
    print()
    print("=== NEW rallies ===")
    for r in new_sorted:
        print("  " + _format_rally(r, new_fps))
    print()

    print("=== Diff (IoU matching, threshold > 0.1) ===")
    pairs = _match_by_overlap(old_sorted, new_sorted)
    for a, b, iou in pairs:
        if a and b:
            print(f"  MATCH iou={iou:.2f}")
            print(f"    OLD: {_format_rally(a, old_fps)}")
            print(f"    NEW: {_format_rally(b, new_fps)}")
        elif a:
            print(f"  MISSING IN NEW: {_format_rally(a, old_fps)}")
        elif b:
            print(f"  EXTRA IN NEW:   {_format_rally(b, new_fps)}")

    # Summary
    matched = sum(1 for a, b, _ in pairs if a and b)
    only_old = sum(1 for a, b, _ in pairs if a and not b)
    only_new = sum(1 for a, b, _ in pairs if b and not a)
    print()
    print(f"Summary: {matched} matched · {only_old} in OLD only · {only_new} in NEW only")

    # Shuttle position coverage (helps diagnose if shuttle detection differs)
    def _coverage(payload: dict) -> Optional[float]:
        sp = payload.get("shuttle_positions")
        tf = payload.get("total_frames") or 0
        if not sp or tf == 0:
            # Fallback: count visible shuttle positions in skeleton_data (legacy shape)
            sd = payload.get("skeleton_data") or []
            visible = sum(1 for f in sd if f.get("shuttle_position"))
            return visible / len(sd) if sd else None
        if isinstance(sp, dict):
            visible = sum(1 for v in sp.values() if v and v.get("visible"))
            return visible / tf
        return None

    old_cov = _coverage(old["payload"])
    new_cov = _coverage(new["payload"])
    print(f"Shuttle visibility: OLD={old_cov!r}  NEW={new_cov!r}")


if __name__ == "__main__":
    main()
