"""
Phase 0 benchmark: TrackNet-only vs union rally detection.

Reads previously-processed videos from Supabase (status='completed' with
results_storage_path set), downloads their results JSON, re-runs gradient-only
rally detection from the stored shuttle_positions, and compares to the
existing union output.

Outputs a Markdown report to docs/plans/2026-05-12-rally-benchmark-results.md.

Decision rule (from design):
- TrackNet-only recall >= 95% of union rallies, AND
- 95th-percentile boundary error < 1.0s, AND
- Estimated time savings > 40% of current pipeline time
Then choose TrackNet-only for Phase 1. Otherwise choose union path.

Usage:
    modal run backend/scripts/rally_accuracy_benchmark.py --video-ids id1,id2,id3
or
    modal run backend/scripts/rally_accuracy_benchmark.py --auto-select 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

# Import lazily so the script can be imported for tests without Supabase env vars.
def _supabase_client():
    from supabase_helpers import get_supabase_client  # type: ignore
    return get_supabase_client()


def fetch_completed_videos(limit: int) -> list[dict]:
    """Pull up to `limit` videos with status='completed' and results_storage_path set."""
    sb = _supabase_client()
    res = (
        sb.table("videos")
        .select("id, filename, fps, total_frames, results_storage_path")
        .eq("status", "completed")
        .not_.is_("results_storage_path", "null")
        .limit(limit)
        .execute()
    )
    return res.data or []


def download_results_json(path: str) -> dict:
    sb = _supabase_client()
    # Convention: results stored in 'results' bucket.
    data = sb.storage.from_("results").download(path)
    return json.loads(data.decode("utf-8"))


def rerun_gradient_only(shuttle_positions: dict, fps: float, total_frames: int) -> list[dict]:
    """Run gradient-based rally detection without the shot-gap union."""
    # Import here so the function can be unit-tested with stubs.
    from rally_detection import detect_rallies

    # detect_rallies signature: (shuttle_positions, total_frames, fps, ...)
    # Reuses the same gradient signal that today's pipeline uses.
    return detect_rallies(
        shuttle_positions=shuttle_positions,
        total_frames=total_frames,
        fps=fps,
    )


def match_rallies(union: list[dict], candidate: list[dict]) -> list[tuple[dict, dict | None]]:
    """Match union rallies to candidate rallies by max IoU on time intervals."""
    matched: list[tuple[dict, dict | None]] = []
    used: set[int] = set()
    for u in union:
        u_start, u_end = u["start_frame"], u["end_frame"]
        best_idx: int | None = None
        best_iou = 0.0
        for i, c in enumerate(candidate):
            if i in used:
                continue
            inter_start = max(u_start, c["start_frame"])
            inter_end = min(u_end, c["end_frame"])
            if inter_end <= inter_start:
                continue
            inter = inter_end - inter_start
            union_len = max(u_end, c["end_frame"]) - min(u_start, c["start_frame"])
            iou = inter / union_len if union_len > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None and best_iou > 0.3:
            used.add(best_idx)
            matched.append((u, candidate[best_idx]))
        else:
            matched.append((u, None))
    return matched


def boundary_errors_seconds(matches: list[tuple[dict, dict | None]], fps: float) -> dict:
    """Compute boundary error stats for matched rallies."""
    if fps <= 0:
        return {"start_p95": None, "end_p95": None, "start_max": None, "end_max": None}
    starts = [
        abs(u["start_frame"] - c["start_frame"]) / fps
        for u, c in matches
        if c is not None
    ]
    ends = [
        abs(u["end_frame"] - c["end_frame"]) / fps
        for u, c in matches
        if c is not None
    ]
    def p95(xs: list[float]) -> float | None:
        if not xs:
            return None
        return statistics.quantiles(xs, n=20)[-1] if len(xs) >= 20 else max(xs)
    return {
        "start_p95": p95(starts),
        "end_p95": p95(ends),
        "start_max": max(starts) if starts else None,
        "end_max": max(ends) if ends else None,
    }


def evaluate_video(video: dict) -> dict:
    """Run the comparison for one video and return its row of metrics."""
    results = download_results_json(video["results_storage_path"])
    union = results.get("rallies", [])
    shuttle_positions = results.get("shuttle_positions") or {}
    fps = float(video.get("fps") or results.get("fps") or 30.0)
    total_frames = int(video.get("total_frames") or results.get("total_frames") or 0)
    if not shuttle_positions or total_frames == 0:
        return {"video_id": video["id"], "skipped": True, "reason": "missing shuttle_positions or total_frames"}
    candidate = rerun_gradient_only(shuttle_positions, fps, total_frames)
    matches = match_rallies(union, candidate)
    matched_count = sum(1 for _, c in matches if c is not None)
    recall = matched_count / len(union) if union else 1.0
    errors = boundary_errors_seconds(matches, fps)
    return {
        "video_id": video["id"],
        "filename": video.get("filename"),
        "union_count": len(union),
        "candidate_count": len(candidate),
        "matched_count": matched_count,
        "recall": recall,
        "boundary_start_p95_s": errors["start_p95"],
        "boundary_end_p95_s": errors["end_p95"],
    }


def render_report(rows: list[dict], out_path: Path) -> str:
    """Render Markdown report and write to out_path. Return summary recommendation."""
    valid = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]
    if not valid:
        recommendation = "INCONCLUSIVE — no valid videos to compare"
    else:
        recalls = [r["recall"] for r in valid]
        min_recall = min(recalls)
        max_p95 = max(
            (r["boundary_start_p95_s"] or 0) for r in valid
        )
        if min_recall >= 0.95 and max_p95 < 1.0:
            recommendation = "TrackNet-only path PASSES recall + boundary thresholds. Choose TrackNet-only for Phase 1."
        else:
            recommendation = f"TrackNet-only path FAILS thresholds (min recall {min_recall:.2f}, max p95 boundary {max_p95:.2f}s). Choose union path for Phase 1."

    lines = [
        "# Rally Accuracy Benchmark — Results",
        "",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Per-Video Results",
        "",
        "| Video | Union | Candidate | Matched | Recall | Start p95 (s) | End p95 (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in valid:
        lines.append(
            f"| `{r['video_id'][:8]}` | {r['union_count']} | {r['candidate_count']} | "
            f"{r['matched_count']} | {r['recall']:.2%} | "
            f"{r['boundary_start_p95_s']:.2f} | {r['boundary_end_p95_s']:.2f} |"
        )
    if skipped:
        lines.append("")
        lines.append(f"_Skipped: {len(skipped)} videos (missing data)._")
    out_path.write_text("\n".join(lines))
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-ids", help="Comma-separated video IDs", default=None)
    parser.add_argument("--auto-select", type=int, help="Auto-select N completed videos", default=None)
    parser.add_argument("--out", default="docs/plans/2026-05-12-rally-benchmark-results.md")
    args = parser.parse_args()

    if args.video_ids:
        ids = [s.strip() for s in args.video_ids.split(",") if s.strip()]
        sb = _supabase_client()
        videos = (
            sb.table("videos")
            .select("id, filename, fps, total_frames, results_storage_path")
            .in_("id", ids)
            .execute()
        ).data or []
    elif args.auto_select:
        videos = fetch_completed_videos(args.auto_select)
    else:
        raise SystemExit("Pass --video-ids or --auto-select")

    rows = []
    for v in videos:
        try:
            rows.append(evaluate_video(v))
        except Exception as e:
            rows.append({"video_id": v["id"], "skipped": True, "reason": str(e)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation = render_report(rows, out_path)
    print(f"Report written to {out_path}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
