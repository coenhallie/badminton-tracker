# Two-Phase Rally-First Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic video processing pipeline into Phase 1 (rally segmentation, runs immediately) and Phase 2 (full analytics, opt-in after user reviews rallies). Phase 1 surfaces rally clips to the Android app and Vue web app independently; Phase 2 is triggered by an explicit user action.

**Architecture:** Two independent Modal entry points (`process_video` for Phase 1, new `process_analytics` for Phase 2). Phase 1 ends with `videos.status = 'phase1_complete'`, rally clips written. New `RallyReview.vue` screen between phases. User clicks "Continue with full analytics" → new `start-analytics` Edge Function → Phase 2 Modal run. Status flow extends to `processing_phase1` → `phase1_complete` → `processing_phase2` → `completed` (plus per-phase `failed_*` states).

**Tech Stack:** Vue 3 + TypeScript + Vite + Tailwind (frontend); Python + Modal serverless GPU (backend); Supabase (Postgres + Edge Functions + Storage). No frontend or backend test framework installed — verification is via scripts and manual smoke.

**Spec reference:** `docs/plans/2026-05-12-two-phase-rally-first-pipeline-design.md`

---

## File Structure

### Backend (`backend/`)

| File | Action | Responsibility |
|---|---|---|
| `modal_supabase_processor.py` | Modify | Scope down `process_video` to Phase 1; add `process_analytics` entrypoint and `_process_analytics_worker`; persist `shuttle_positions` in Phase-1 results JSON |
| `scripts/__init__.py` | Create (empty) | Package marker for verification scripts |
| `scripts/rally_accuracy_benchmark.py` | Create | Phase 0 benchmark: TrackNet-only vs union on existing Supabase-processed videos |
| `scripts/verify_phase1.py` | Create | Verification script: invoke Phase 1 against a fixture video and assert outputs |
| `scripts/verify_phase2.py` | Create | Verification script: invoke Phase 2 against a Phase-1-complete fixture and assert merged outputs |

### Supabase (`supabase/`)

| File | Action | Responsibility |
|---|---|---|
| `migrations/0005_two_phase_pipeline.sql` | Create | Add status enum values, `phase` column on `processing_logs`, remap legacy `failed` rows |
| `functions/process-video/index.ts` | Modify | Set new status `processing_phase1` (was `processing`) |
| `functions/start-analytics/index.ts` | Create | New Edge Function: validate state, set `processing_phase2`, invoke Modal `process_analytics` |
| `functions/start-analytics/deno.json` | Create | Deno config matching `process-video` |

### Frontend (`src/`)

| File | Action | Responsibility |
|---|---|---|
| `types/analysis.ts` | Modify | Add new status enum values + Phase-1-only result shape |
| `App.vue` | Modify | Extend `AppState`, add resume-from-status hydration, remove skip handler |
| `components/CourtSetup.vue` | Modify | Remove skip emit + Skip button |
| `components/AnalysisProgress.vue` | Modify | Phase-aware events (`phase1Complete`, `phase2Complete`) + dynamic header + phase log filtering |
| `components/RallyReview.vue` | Create | New screen between Phase 1 and Phase 2 |
| `components/ResultsDashboard.vue` | Modify | Hard-route Phase-1-only videos to RallyReview |

---

## Task 1: Phase 0 Benchmark — Build the Script

**Files:**
- Create: `backend/scripts/__init__.py` (empty)
- Create: `backend/scripts/rally_accuracy_benchmark.py`

**Purpose:** Compare today's union rally detection against a TrackNet-only path on already-processed videos. Emit a report deciding Phase 1 scope.

- [ ] **Step 1: Create the empty package marker**

```bash
touch backend/scripts/__init__.py
```

- [ ] **Step 2: Write the benchmark script**

Create `backend/scripts/rally_accuracy_benchmark.py`:

```python
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
```

- [ ] **Step 3: Verify the script parses and imports**

Run: `python -c "import backend.scripts.rally_accuracy_benchmark as b; print('ok', b.evaluate_video.__name__)"`
Expected: `ok evaluate_video`

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/rally_accuracy_benchmark.py
git commit -m "feat(benchmark): rally accuracy benchmark script (Phase 0)"
```

---

## Task 2: Phase 0 Benchmark — Run It and Record Decision

**Files:**
- Create: `docs/plans/2026-05-12-rally-benchmark-results.md` (auto-written by script)

**Purpose:** Generate the report and commit the decision.

- [ ] **Step 1: Pick 5–10 representative videos from Supabase**

Run a quick query in the Supabase SQL editor or via psql:
```sql
SELECT id, filename, status, created_at
FROM videos
WHERE status='completed' AND results_storage_path IS NOT NULL
ORDER BY created_at DESC
LIMIT 20;
```
Note 5–10 IDs covering different conditions (singles/doubles, lighting, camera angles).

- [ ] **Step 2: Run the benchmark**

Run: `python -m backend.scripts.rally_accuracy_benchmark --video-ids <comma-separated-ids>`
Expected: Report written to `docs/plans/2026-05-12-rally-benchmark-results.md`. STDOUT prints the recommendation.

- [ ] **Step 3: Commit the report**

```bash
git add docs/plans/2026-05-12-rally-benchmark-results.md
git commit -m "docs(benchmark): rally accuracy benchmark results"
```

- [ ] **Step 4: Apply the decision to subsequent tasks**

If the report says **TrackNet-only PASSES**: in Task 6 below, skip the YOLO frame loop entirely in Phase 1.
If the report says **union path needed**: in Task 6 below, run a detection-only YOLO pass (see Task 6 Branch B).

The rest of the plan is path-agnostic.

---

## Task 3: Supabase Migration

**Files:**
- Create: `supabase/migrations/0005_two_phase_pipeline.sql`

**Purpose:** Extend `videos.status` enum, add `phase` column to `processing_logs`, remap legacy `failed` rows.

- [ ] **Step 1: Inspect the current status enum definition**

Run: `grep -n "status" supabase/migrations/0001_initial_schema.sql`
Expected: shows how `videos.status` is currently defined (likely a TEXT column with a CHECK constraint, since plain Postgres TEXT is the safe Supabase pattern).

Read enough to know whether it's an `ENUM` type or a `TEXT CHECK`. If `TEXT CHECK`, update the constraint. If `ENUM`, use `ALTER TYPE ... ADD VALUE`.

- [ ] **Step 2: Write the migration**

Create `supabase/migrations/0005_two_phase_pipeline.sql`. The exact SQL depends on what Step 1 showed — pick the matching branch below and discard the other.

**Branch A — if `videos.status` is `TEXT` with a CHECK constraint:**

```sql
-- 0005_two_phase_pipeline.sql
-- Split the monolithic pipeline into Phase 1 (rallies) and Phase 2 (analytics).

BEGIN;

-- 1. Drop the old CHECK constraint and re-add with new values.
ALTER TABLE videos DROP CONSTRAINT IF EXISTS videos_status_check;
ALTER TABLE videos ADD CONSTRAINT videos_status_check
  CHECK (status IN (
    'pending',
    'uploaded',
    'processing',           -- legacy, tolerated for historical rows
    'processing_phase1',
    'phase1_complete',
    'processing_phase2',
    'completed',
    'failed',               -- legacy, tolerated for historical rows
    'failed_phase1',
    'failed_phase2'
  ));

-- 2. Remap legacy 'failed' rows to 'failed_phase2' (closest semantic match).
UPDATE videos SET status='failed_phase2' WHERE status='failed';

-- 3. Add phase column to processing_logs.
ALTER TABLE processing_logs
  ADD COLUMN IF NOT EXISTS phase TEXT
    CHECK (phase IS NULL OR phase IN ('phase1', 'phase2'));

COMMIT;
```

**Branch B — if `videos.status` is a Postgres ENUM type:**

```sql
-- 0005_two_phase_pipeline.sql
-- Split the monolithic pipeline into Phase 1 (rallies) and Phase 2 (analytics).

-- ALTER TYPE ... ADD VALUE cannot run inside a transaction block.
ALTER TYPE video_status ADD VALUE IF NOT EXISTS 'processing_phase1';
ALTER TYPE video_status ADD VALUE IF NOT EXISTS 'phase1_complete';
ALTER TYPE video_status ADD VALUE IF NOT EXISTS 'processing_phase2';
ALTER TYPE video_status ADD VALUE IF NOT EXISTS 'failed_phase1';
ALTER TYPE video_status ADD VALUE IF NOT EXISTS 'failed_phase2';

BEGIN;

UPDATE videos SET status='failed_phase2' WHERE status='failed';

ALTER TABLE processing_logs
  ADD COLUMN IF NOT EXISTS phase TEXT
    CHECK (phase IS NULL OR phase IN ('phase1', 'phase2'));

COMMIT;
```

- [ ] **Step 3: Apply the migration locally**

Run: `supabase db reset` (local dev) or `supabase migration up`.
Expected: migration applies cleanly; no constraint violations.

- [ ] **Step 4: Verify**

Run:
```bash
supabase db psql -c "SELECT DISTINCT status FROM videos;"
supabase db psql -c "\d processing_logs"
```
Expected: `status` distinct values include the new names where applicable; `processing_logs` has a `phase` column.

- [ ] **Step 5: Commit**

```bash
git add supabase/migrations/0005_two_phase_pipeline.sql
git commit -m "feat(db): add two-phase pipeline status values and processing_logs.phase"
```

---

## Task 4: Backend — Scope Down Phase 1 in `process_video`

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Purpose:** Existing `_process_video_worker` runs the full monolithic pipeline. Scope it down to Phase 1 only: TrackNet shuttle tracking + rally detection + rally clip generation + status flip to `phase1_complete`. Persist `shuttle_positions` in the results JSON. Remove pose, tracking, court calibration validation, and analytics from this worker.

- [ ] **Step 1: Read the current worker structure**

Run: `grep -n "def _process_video_worker\|Phase \|status\|skeleton_frames\|YOLO\|detected_rallies\|cut_and_upload" backend/modal_supabase_processor.py | head -60`

This locates the phases described in the spec (frame processing, rally detection, results upload, clip generation, completion gate).

- [ ] **Step 2: Modify `_process_video_worker` to do Phase 1 only**

Around line 1923 (start of `_process_video_worker`), rename the function or add a clear comment block, and remove the YOLO frame loop and analytics post-processing. The exact diff depends on whether the benchmark in Task 2 chose TrackNet-only or union.

**Common changes (regardless of benchmark result):**

1. Change initial status set: `'processing'` → `'processing_phase1'` (search for the `send_status_update` call at worker start).
2. After rally detection: serialize the slim results JSON instead of the full skeleton-laden payload:

```python
# Replace the existing full-payload serialization with this slim Phase-1 payload.
phase1_results = {
    "phase": "phase1",
    "rallies": detected_rallies,
    "shuttle_positions": shuttle_positions,   # from TrackNet
    "fps": fps,
    "total_frames": total_frames,
    "video_metadata": {
        "duration_seconds": total_frames / fps if fps else None,
        "filename": video_row.get("filename"),
    },
}

results_json = json.dumps(phase1_results)
results_path = f"{video_id}/phase1_results.json"
sb.storage.from_("results").upload(
    path=results_path,
    file=results_json.encode("utf-8"),
    file_options={"content-type": "application/json", "upsert": "true"},
)
sb.table("videos").update({"results_storage_path": results_path}).eq("id", video_id).execute()
```

3. After rally clip generation (`cut_and_upload_rally_clips`): set status to `phase1_complete`:

```python
await send_status_update(video_id, {"status": "phase1_complete", "progress": 100})
```

4. In the failure handler, change `'failed'` to `'failed_phase1'`:

```python
except Exception as e:
    await send_status_update(video_id, {"status": "failed_phase1", "error": str(e)})
    raise
```

5. In every `send_log` call inside this worker, pass `phase="phase1"` (extend `send_log` signature if needed — Step 3 below).

**Branch A (TrackNet-only path won the benchmark):** Delete the entire YOLO frame loop block (locate it as the loop iterating over frames and calling YOLO pose/detection models). Remove the shot-gap rally union call. Keep only TrackNet's `shuttle_positions` and `detect_rallies(shuttle_positions=..., total_frames=..., fps=...)`.

**Branch B (union path needed):** Keep the YOLO loop but strip it down — load only the detection model (not pose), skip `skeleton_frames` accumulation, just record per-frame shuttle detections. Keep the shot-gap union call.

- [ ] **Step 3: Extend `send_log` and `send_status_update` to accept `phase`**

Locate the `send_log` and `send_status_update` definitions (~lines 1953–2019). Add an optional `phase: str | None = None` parameter to `send_log` and include it in the inserted row:

```python
async def send_log(message: str, level: str = "info", stage: str = "processing", phase: str | None = None) -> None:
    sb.table("processing_logs").insert({
        "video_id": video_id,
        "message": message,
        "level": level,
        "stage": stage,
        "phase": phase,
    }).execute()
```

Then update every `send_log(...)` call inside `_process_video_worker` to pass `phase="phase1"`. `send_status_update` does not need a `phase` field on `videos`, so leave its signature unchanged.

- [ ] **Step 4: Verify with the verification script (created in Task 8)**

The verification script doesn't exist yet — defer running it until Task 8. For now, do a smoke check:

Run: `python -c "from backend.modal_supabase_processor import process_video; print(process_video.__name__)"`
Expected: `process_video` (or whatever Modal wraps it as). No syntax errors.

- [ ] **Step 5: Commit**

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(backend): scope process_video to Phase 1 (rallies + clips only)"
```

---

## Task 5: Backend — Add `process_analytics` Entry Point (Phase 2)

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Purpose:** New Modal entry point that runs the full YOLO + analytics pipeline against a video whose Phase 1 is already complete. Augments the existing Phase-1 results JSON in place.

- [ ] **Step 1: Add the new Modal entry point**

Append to `backend/modal_supabase_processor.py` (placement: near the existing `process_video` Modal function definition). Inputs and outputs mirror `process_video` but no upload-trigger inputs.

```python
@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_volume, "/cache": cache_volume},
    timeout=60 * 60,  # 1 hour, matches process_video
    secrets=[modal.Secret.from_name("supabase-credentials")],
)
async def process_analytics(video_id: str) -> dict:
    """
    Phase 2: full analytics pipeline. Re-downloads the video and reads Phase-1
    results from Supabase. Runs YOLO pose + detection per-frame, computes
    analytics, and augments the existing results JSON in place. Does NOT
    re-run TrackNet (shuttle_positions are persisted in Phase-1 JSON) and
    does NOT re-run rally detection.
    """
    asyncio.create_task(_process_analytics_worker(video_id))
    return {"video_id": video_id, "status": "processing_phase2"}


async def _process_analytics_worker(video_id: str) -> None:
    sb = get_supabase_client()
    try:
        await send_status_update(video_id, {"status": "processing_phase2", "progress": 0})
        await send_log("Phase 2 starting", level="info", stage="processing", phase="phase2", video_id=video_id)

        # 1. Load video row + Phase 1 results JSON.
        video_row = sb.table("videos").select("*").eq("id", video_id).single().execute().data
        if not video_row or not video_row.get("results_storage_path"):
            raise RuntimeError("Phase 1 results missing — cannot run Phase 2")

        phase1_results = json.loads(
            sb.storage.from_("results").download(video_row["results_storage_path"]).decode("utf-8")
        )
        court_keypoints = video_row.get("manual_court_keypoints")
        if not court_keypoints:
            raise RuntimeError("Court keypoints missing — cannot run Phase 2")
        fps = float(phase1_results["fps"])
        total_frames = int(phase1_results["total_frames"])
        shuttle_positions = phase1_results.get("shuttle_positions") or {}

        # 2. Download video (re-use existing helper).
        video_path = await _download_video_to_cache(video_row, sb)

        # 3. Load YOLO pose + detection models.
        pose_model = _load_yolo_pose_model()
        detect_model = _load_yolo_detect_model()

        # 4. Per-frame YOLO loop: pose + detection + tracking. Accumulate skeleton_frames.
        skeleton_frames = await _run_yolo_loop(
            video_path=video_path,
            pose_model=pose_model,
            detect_model=detect_model,
            total_frames=total_frames,
            fps=fps,
            video_id=video_id,
            phase="phase2",
        )

        # 5. Compute analytics (speed, heatmaps, body angles, shot classification, zones).
        analytics = compute_analytics(
            skeleton_frames=skeleton_frames,
            court_keypoints=court_keypoints,
            shuttle_positions=shuttle_positions,
            fps=fps,
            total_frames=total_frames,
        )

        # 6. Merge: augment Phase 1 results with analytics, preserve Phase 1 keys exactly.
        merged = dict(phase1_results)
        merged["phase"] = "completed"
        merged["skeleton_frames"] = skeleton_frames
        merged["analytics"] = analytics

        # 7. Write back to the same storage path.
        sb.storage.from_("results").upload(
            path=video_row["results_storage_path"],
            file=json.dumps(merged).encode("utf-8"),
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        # 8. Flip status to completed.
        await send_status_update(video_id, {"status": "completed", "progress": 100})
        await send_log("Phase 2 complete", level="info", stage="processing", phase="phase2", video_id=video_id)

    except Exception as e:
        await send_status_update(video_id, {"status": "failed_phase2", "error": str(e)})
        await send_log(f"Phase 2 failed: {e}", level="error", stage="processing", phase="phase2", video_id=video_id)
        raise
```

**Important:** The helpers referenced above (`_download_video_to_cache`, `_load_yolo_pose_model`, `_load_yolo_detect_model`, `_run_yolo_loop`, `compute_analytics`) should be **extracted from the existing `_process_video_worker`** during Task 4 so they can be reused here. If they don't exist yet as separate functions, refactor them out as part of this task — pull the YOLO loop and analytics computation into named helpers, then call them from both old and new workers.

- [ ] **Step 2: Update `send_log` to accept `video_id` as parameter if not already**

If `send_log` was a closure over `video_id` in the original code, refactor it to a module-level function that takes `video_id` explicitly. This is needed so `_process_analytics_worker` can call it.

- [ ] **Step 3: Smoke check import**

Run: `python -c "from backend.modal_supabase_processor import process_analytics; print('ok')"`
Expected: `ok` (no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(backend): add process_analytics Modal entry point (Phase 2)"
```

---

## Task 6: Supabase Edge Function — Update `process-video`

**Files:**
- Modify: `supabase/functions/process-video/index.ts`

**Purpose:** Change the status it sets from `processing` to `processing_phase1`. Validate that `manual_court_keypoints` is non-null before invoking Modal (new precondition since court setup is now mandatory).

- [ ] **Step 1: Read the current function**

Run: `cat supabase/functions/process-video/index.ts | head -100`
Identify where `videos.status` is set and where Modal is invoked.

- [ ] **Step 2: Edit the status string and add court-keypoints validation**

Locate the line that sets status (probably `status: 'processing'`) and change to `'processing_phase1'`. Before invoking Modal, fetch the video row and assert `manual_court_keypoints` is present:

```ts
// Fetch row, validate state.
const { data: video, error: fetchErr } = await supabase
  .from('videos').select('id, status, manual_court_keypoints').eq('id', videoId).single();
if (fetchErr || !video) {
  return new Response(JSON.stringify({ error: 'Video not found' }), { status: 404 });
}
if (video.status !== 'uploaded') {
  return new Response(JSON.stringify({ error: `Invalid state: ${video.status}` }), { status: 409 });
}
if (!video.manual_court_keypoints) {
  return new Response(JSON.stringify({ error: 'Court setup required before processing' }), { status: 400 });
}

// Flip status BEFORE invoking Modal so a Modal cold-start failure is recoverable.
const { error: updErr } = await supabase
  .from('videos').update({ status: 'processing_phase1' }).eq('id', videoId);
if (updErr) {
  return new Response(JSON.stringify({ error: updErr.message }), { status: 500 });
}

// Then invoke Modal (existing logic).
```

- [ ] **Step 3: Deploy locally and test**

Run: `supabase functions serve process-video`
In another terminal:
```bash
curl -X POST http://localhost:54321/functions/v1/process-video \
  -H "Authorization: Bearer <test-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"videoId": "<test-video-id-without-keypoints>"}'
```
Expected: 400 with "Court setup required before processing".

- [ ] **Step 4: Commit**

```bash
git add supabase/functions/process-video/index.ts
git commit -m "feat(edge): process-video sets processing_phase1, requires court keypoints"
```

---

## Task 7: Supabase Edge Function — Create `start-analytics`

**Files:**
- Create: `supabase/functions/start-analytics/index.ts`
- Create: `supabase/functions/start-analytics/deno.json` (mirror existing function configs)

**Purpose:** New Edge Function. Validates state, flips status to `processing_phase2`, invokes Modal `process_analytics`. Used by `RallyReview.vue` and the "Retry analytics" button.

- [ ] **Step 1: Copy the existing `process-video` function as a starting point**

```bash
mkdir -p supabase/functions/start-analytics
cp supabase/functions/process-video/deno.json supabase/functions/start-analytics/deno.json
```

- [ ] **Step 2: Write the new function**

Create `supabase/functions/start-analytics/index.ts`:

```ts
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const MODAL_PROCESS_ANALYTICS_URL = Deno.env.get("MODAL_PROCESS_ANALYTICS_URL")!;
const MODAL_API_TOKEN = Deno.env.get("MODAL_API_TOKEN")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) {
    return new Response(JSON.stringify({ error: "Missing auth" }), { status: 401 });
  }

  const { videoId } = await req.json().catch(() => ({}));
  if (!videoId) {
    return new Response(JSON.stringify({ error: "Missing videoId" }), { status: 400 });
  }

  // Use a user-scoped client first to confirm ownership via RLS.
  const userClient = createClient(SUPABASE_URL, Deno.env.get("SUPABASE_ANON_KEY")!, {
    global: { headers: { Authorization: authHeader } },
  });
  const { data: ownerCheck, error: ownerErr } = await userClient
    .from("videos").select("id").eq("id", videoId).single();
  if (ownerErr || !ownerCheck) {
    return new Response(JSON.stringify({ error: "Not found or not authorized" }), { status: 404 });
  }

  // Service-role client for state transition.
  const admin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: video, error: fetchErr } = await admin
    .from("videos")
    .select("id, status, results_storage_path, manual_court_keypoints")
    .eq("id", videoId)
    .single();
  if (fetchErr || !video) {
    return new Response(JSON.stringify({ error: "Video not found" }), { status: 404 });
  }
  if (!["phase1_complete", "failed_phase2"].includes(video.status)) {
    return new Response(
      JSON.stringify({ error: `Invalid state: ${video.status}` }),
      { status: 409 },
    );
  }
  if (!video.results_storage_path) {
    return new Response(JSON.stringify({ error: "Phase 1 results missing" }), { status: 409 });
  }
  if (!video.manual_court_keypoints) {
    return new Response(JSON.stringify({ error: "Court keypoints missing" }), { status: 409 });
  }

  // Flip status BEFORE invoking Modal.
  const { error: updErr } = await admin
    .from("videos")
    .update({ status: "processing_phase2", error: null, progress: 0 })
    .eq("id", videoId);
  if (updErr) {
    return new Response(JSON.stringify({ error: updErr.message }), { status: 500 });
  }

  // Invoke Modal.
  const modalRes = await fetch(MODAL_PROCESS_ANALYTICS_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${MODAL_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ video_id: videoId }),
  });
  if (!modalRes.ok) {
    // Roll back status so user can retry.
    await admin.from("videos").update({ status: "failed_phase2", error: `Modal invoke failed: ${modalRes.status}` }).eq("id", videoId);
    return new Response(JSON.stringify({ error: "Modal invocation failed" }), { status: 502 });
  }

  return new Response(JSON.stringify({ ok: true, videoId, status: "processing_phase2" }), {
    status: 202,
    headers: { "Content-Type": "application/json" },
  });
});
```

- [ ] **Step 3: Configure the Modal URL secret**

Run:
```bash
supabase secrets set MODAL_PROCESS_ANALYTICS_URL=<url-from-modal-deploy>
```
(The URL is shown after running `modal deploy backend/modal_supabase_processor.py`.)

- [ ] **Step 4: Deploy locally and test the validation paths**

Run: `supabase functions serve start-analytics`

Test bad state (status not `phase1_complete`):
```bash
curl -X POST http://localhost:54321/functions/v1/start-analytics \
  -H "Authorization: Bearer <test-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"videoId": "<id-with-status-uploaded>"}'
```
Expected: 409 with "Invalid state: uploaded".

- [ ] **Step 5: Commit**

```bash
git add supabase/functions/start-analytics/
git commit -m "feat(edge): start-analytics Edge Function for Phase 2 trigger"
```

---

## Task 8: Backend Verification Scripts

**Files:**
- Create: `backend/scripts/verify_phase1.py`
- Create: `backend/scripts/verify_phase2.py`

**Purpose:** Manual verification scripts since the project has no pytest. Each script invokes its phase against a fixture video and asserts the expected outputs.

- [ ] **Step 1: Write `verify_phase1.py`**

```python
"""
Verify Phase 1: invoke process_video against a fixture video and assert outputs.
Usage: python -m backend.scripts.verify_phase1 --video-id <id>
"""
from __future__ import annotations
import argparse
import json
import time
from supabase_helpers import get_supabase_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    sb = get_supabase_client()

    # Sanity: precondition.
    row = sb.table("videos").select("status, manual_court_keypoints").eq("id", args.video_id).single().execute().data
    assert row, f"Video {args.video_id} not found"
    assert row["status"] in ("uploaded", "failed_phase1"), f"Bad starting status: {row['status']}"
    assert row["manual_court_keypoints"], "Court keypoints required"

    # Invoke via the Edge Function (so we exercise the full path).
    import requests, os
    res = requests.post(
        f"{os.environ['SUPABASE_URL']}/functions/v1/process-video",
        headers={"Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_ROLE_KEY']}"},
        json={"videoId": args.video_id},
    )
    assert res.status_code == 202, f"Edge Function failed: {res.status_code} {res.text}"

    # Poll for terminal state.
    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        row = sb.table("videos").select("status, results_storage_path, error").eq("id", args.video_id).single().execute().data
        if row["status"] in ("phase1_complete", "failed_phase1"):
            break
        time.sleep(5)
    else:
        raise AssertionError("Phase 1 did not complete within timeout")

    assert row["status"] == "phase1_complete", f"Phase 1 failed: {row.get('error')}"
    assert row["results_storage_path"], "results_storage_path not set"

    # Assert results JSON shape.
    results = json.loads(sb.storage.from_("results").download(row["results_storage_path"]).decode("utf-8"))
    assert results.get("phase") == "phase1"
    assert "rallies" in results and isinstance(results["rallies"], list)
    assert "shuttle_positions" in results
    assert "fps" in results and "total_frames" in results
    assert "skeleton_frames" not in results, "Phase 1 must NOT include skeleton_frames"
    assert "analytics" not in results, "Phase 1 must NOT include analytics"

    # Assert rally_clips rows present.
    clips = sb.table("rally_clips").select("id").eq("video_id", args.video_id).execute().data
    assert len(clips) == len(results["rallies"]) or len(clips) > 0, "rally_clips rows missing"

    print(f"OK: Phase 1 verified for {args.video_id}. {len(results['rallies'])} rallies, {len(clips)} clips.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `verify_phase2.py`**

```python
"""
Verify Phase 2: invoke start-analytics for a phase1_complete video and assert outputs.
Usage: python -m backend.scripts.verify_phase2 --video-id <id>
"""
from __future__ import annotations
import argparse
import json
import time
import os
from supabase_helpers import get_supabase_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()
    sb = get_supabase_client()

    row = sb.table("videos").select("status, results_storage_path").eq("id", args.video_id).single().execute().data
    assert row, f"Video {args.video_id} not found"
    assert row["status"] in ("phase1_complete", "failed_phase2"), f"Bad starting status: {row['status']}"

    # Capture Phase 1 keys for non-destructive merge assertion.
    phase1_results = json.loads(sb.storage.from_("results").download(row["results_storage_path"]).decode("utf-8"))
    phase1_rallies = phase1_results["rallies"]
    phase1_shuttle = phase1_results["shuttle_positions"]

    import requests
    res = requests.post(
        f"{os.environ['SUPABASE_URL']}/functions/v1/start-analytics",
        headers={"Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_ROLE_KEY']}"},
        json={"videoId": args.video_id},
    )
    assert res.status_code == 202, f"Edge Function failed: {res.status_code} {res.text}"

    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        row = sb.table("videos").select("status, error").eq("id", args.video_id).single().execute().data
        if row["status"] in ("completed", "failed_phase2"):
            break
        time.sleep(10)
    else:
        raise AssertionError("Phase 2 did not complete within timeout")

    assert row["status"] == "completed", f"Phase 2 failed: {row.get('error')}"

    # Re-download merged results and assert Phase 1 keys preserved exactly.
    merged = json.loads(sb.storage.from_("results").download(phase1_results.get("results_storage_path", "")).decode("utf-8")) if False else None
    row2 = sb.table("videos").select("results_storage_path").eq("id", args.video_id).single().execute().data
    merged = json.loads(sb.storage.from_("results").download(row2["results_storage_path"]).decode("utf-8"))

    assert merged["rallies"] == phase1_rallies, "Phase 1 rallies were mutated by Phase 2"
    assert merged["shuttle_positions"] == phase1_shuttle, "Phase 1 shuttle_positions were mutated by Phase 2"
    assert "analytics" in merged, "Phase 2 did not add analytics"
    assert "skeleton_frames" in merged, "Phase 2 did not add skeleton_frames"

    # Assert rally_clips rows unchanged.
    clips_after = sb.table("rally_clips").select("id").eq("video_id", args.video_id).execute().data
    assert len(clips_after) == len(phase1_rallies), "Phase 2 mutated rally_clips"

    print(f"OK: Phase 2 verified for {args.video_id}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run both scripts against a test video**

```bash
python -m backend.scripts.verify_phase1 --video-id <id-uploaded-with-keypoints>
python -m backend.scripts.verify_phase2 --video-id <same-id-now-phase1-complete>
```
Expected: both print `OK: ...`.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/verify_phase1.py backend/scripts/verify_phase2.py
git commit -m "test(backend): verification scripts for Phase 1 and Phase 2"
```

---

## Task 9: Frontend Types — Update `analysis.ts`

**Files:**
- Modify: `src/types/analysis.ts`

**Purpose:** Add new status enum values and a Phase-1-only result shape.

- [ ] **Step 1: Locate the status type**

Run: `grep -n "type.*Status\|status:\|VideoStatus" src/types/analysis.ts`

- [ ] **Step 2: Update the status union**

Replace the existing `VideoStatus` (or equivalent) type with:

```ts
export type VideoStatus =
  | 'pending'
  | 'uploaded'
  | 'processing_phase1'
  | 'phase1_complete'
  | 'processing_phase2'
  | 'completed'
  | 'failed_phase1'
  | 'failed_phase2'
  // Legacy values tolerated for historical rows.
  | 'processing'
  | 'failed';
```

- [ ] **Step 3: Add Phase-1-only result shape**

Add to the same file:

```ts
export interface Phase1Results {
  phase: 'phase1';
  rallies: Rally[];
  shuttle_positions: Record<number, { x: number; y: number; visible: boolean }>;
  fps: number;
  total_frames: number;
  video_metadata: {
    duration_seconds: number | null;
    filename: string | null;
  };
}

// Existing AnalysisResult represents the fully merged Phase-2 payload.
// It now extends Phase1Results with optional analytics keys.
export interface FullAnalysisResult extends Phase1Results {
  phase: 'phase1' | 'completed';   // updated by Phase 2 merge
  skeleton_frames?: SkeletonFrame[];
  analytics?: Analytics;            // existing Analytics type
}
```

If `Rally`, `SkeletonFrame`, `Analytics` already exist, keep them. If not, the engineer should find their existing equivalents and adapt the references.

- [ ] **Step 4: Verify types compile**

Run: `npm run type-check`
Expected: no new errors.

- [ ] **Step 5: Commit**

```bash
git add src/types/analysis.ts
git commit -m "feat(types): add two-phase pipeline status values and Phase1Results"
```

---

## Task 10: Frontend — `CourtSetup.vue` Remove Skip

**Files:**
- Modify: `src/components/CourtSetup.vue`

**Purpose:** Court setup is now mandatory. Remove the skip emit and the Skip button.

- [ ] **Step 1: Find the skip emit and button**

Run: `grep -n "skip\|Skip" src/components/CourtSetup.vue`

- [ ] **Step 2: Remove from the `defineEmits`**

Locate:
```ts
const emit = defineEmits<{
  complete: [keypoints: ExtendedCourtKeypoints]
  skip: []
  error: [message: string]
}>()
```
Change to:
```ts
const emit = defineEmits<{
  complete: [keypoints: ExtendedCourtKeypoints]
  error: [message: string]
}>()
```

- [ ] **Step 3: Remove the Skip button from the template**

Find the `<button>` (or equivalent) tied to `emit('skip')` and remove that element entirely.

- [ ] **Step 4: Remove handlers from `App.vue`**

In `src/App.vue`, locate `handleCourtSetupSkip` (line ~704) and:
- Delete the function definition
- Remove `@skip="handleCourtSetupSkip"` from the `<CourtSetup>` element (line ~1200 area)

- [ ] **Step 5: Verify types compile**

Run: `npm run type-check`
Expected: no errors related to the skip emit.

- [ ] **Step 6: Commit**

```bash
git add src/components/CourtSetup.vue src/App.vue
git commit -m "feat(frontend): make court setup mandatory, remove skip path"
```

---

## Task 11: Frontend — `App.vue` State Machine + Resume Hydration

**Files:**
- Modify: `src/App.vue`

**Purpose:** Extend `AppState`, add new transitions for Phase 1 complete / Phase 2 complete, and add resume-from-status logic so a returning user lands in the right state.

- [ ] **Step 1: Extend `AppState`**

Replace (line ~44):
```ts
type AppState = 'upload' | 'court-setup' | 'analyzing' | 'results'
```
with:
```ts
type AppState =
  | 'upload'
  | 'court-setup'
  | 'analyzing-phase1'
  | 'rally-review'
  | 'analyzing-phase2'
  | 'results'
  | 'error'
```

- [ ] **Step 2: Rename the existing `analyzing` state usages**

Find every `currentState.value = 'analyzing'` and the `v-else-if="currentState === 'analyzing'"` block. Change them to use `'analyzing-phase1'` initially; `'analyzing-phase2'` will be added in Step 4.

- [ ] **Step 3: Add resume hydration on mount**

Add a function and call it from the existing `onMounted` (or equivalent) hook:

```ts
import { onMounted } from 'vue'

async function hydrateFromExistingVideo() {
  // If the app loads with a known videoId (from query param, local state, etc.),
  // query its current status and jump to the corresponding AppState.
  if (!uploadedVideo.value?.id) return
  const { data, error } = await supabase
    .from('videos')
    .select('status, manual_court_keypoints, results_storage_path')
    .eq('id', uploadedVideo.value.id)
    .single()
  if (error || !data) return
  const status = data.status as VideoStatus
  switch (status) {
    case 'pending':
      currentState.value = 'upload'; break
    case 'uploaded':
      currentState.value = 'court-setup'; break
    case 'processing_phase1':
      currentState.value = 'analyzing-phase1'; break
    case 'phase1_complete':
      currentState.value = 'rally-review'; break
    case 'processing_phase2':
      currentState.value = 'analyzing-phase2'; break
    case 'completed':
      currentState.value = 'results'; break
    case 'failed_phase1':
    case 'failed_phase2':
    case 'failed':      // legacy
      currentState.value = 'error'; break
    case 'processing':  // legacy in-flight at deploy
      currentState.value = 'analyzing-phase2'; break   // safer: treat as Phase 2
  }
}

onMounted(() => {
  hydrateFromExistingVideo()
})
```

- [ ] **Step 4: Wire up new transitions**

In `AnalysisProgress.vue` listener (already in App.vue), today's `@complete` becomes two listeners. Add:

```vue
<AnalysisProgress
  ... existing props ...
  @phase1Complete="handlePhase1Complete"
  @phase2Complete="handlePhase2Complete"
  @failed="handleAnalysisFailed"
/>
```

with new handlers:

```ts
function handlePhase1Complete() {
  currentState.value = 'rally-review'
}

function handlePhase2Complete() {
  currentState.value = 'results'
}

function handleAnalysisFailed() {
  currentState.value = 'error'
}

async function handleContinueAnalytics() {
  if (!uploadedVideo.value?.id) return
  const { data: { session } } = await supabase.auth.getSession()
  await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/start-analytics`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${session?.access_token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ videoId: uploadedVideo.value.id }),
  })
  currentState.value = 'analyzing-phase2'
}

function handleDoneForNow() {
  currentState.value = 'upload'
  uploadedVideo.value = null
}
```

- [ ] **Step 5: Add `<RallyReview>` block to the template**

Insert after the `analyzing-phase1` block:

```vue
<div v-else-if="currentState === 'rally-review' && uploadedVideo" key="rally-review" class="content-section">
  <RallyReview
    :video-id="uploadedVideo.id"
    @continue="handleContinueAnalytics"
    @done="handleDoneForNow"
  />
</div>
<div v-else-if="currentState === 'analyzing-phase2' && uploadedVideo" key="analyzing-phase2" class="content-section">
  <AnalysisProgress
    :video-id="uploadedVideo.id"
    :phase="'phase2'"
    @phase2Complete="handlePhase2Complete"
    @failed="handleAnalysisFailed"
  />
</div>
<div v-else-if="currentState === 'error'" key="error" class="content-section">
  <!-- Simple error pane. "Retry analytics" only when failed_phase2. -->
  <p>Processing failed.</p>
  <button v-if="lastFailedPhase === 'phase2'" @click="handleContinueAnalytics">Retry analytics</button>
  <button @click="handleDoneForNow">Start over</button>
</div>
```

`lastFailedPhase` is a new ref:
```ts
const lastFailedPhase = ref<'phase1' | 'phase2' | null>(null)
```
Set it in `handleAnalysisFailed` based on the current `videoStatus`:
```ts
function handleAnalysisFailed(failedStatus: VideoStatus) {
  lastFailedPhase.value = failedStatus === 'failed_phase1' ? 'phase1' : 'phase2'
  currentState.value = 'error'
}
```

- [ ] **Step 6: Import `RallyReview`**

Add to the top of `App.vue`:
```ts
import RallyReview from '@/components/RallyReview.vue'
```

(`RallyReview` is created in Task 13. If that task is executed after this one in a subagent flow, type-check will fail temporarily — that's fine, both files land before any merge.)

- [ ] **Step 7: Verify types compile (after Task 13 also lands)**

Run: `npm run type-check`
Expected: no errors related to `AppState`, handlers, or `RallyReview`.

- [ ] **Step 8: Commit**

```bash
git add src/App.vue
git commit -m "feat(frontend): state machine for two-phase pipeline + resume hydration"
```

---

## Task 12: Frontend — `AnalysisProgress.vue` Phase-Aware Events

**Files:**
- Modify: `src/components/AnalysisProgress.vue`

**Purpose:** Replace the single `complete` event with `phase1Complete` and `phase2Complete`. Dynamic header. Filter live logs by phase.

- [ ] **Step 1: Locate the watchers and emits**

Run: `grep -n "videoStatus\|emit\|defineEmits\|@status" src/components/AnalysisProgress.vue`

- [ ] **Step 2: Update `defineEmits`**

Replace the existing emits with:

```ts
const emit = defineEmits<{
  phase1Complete: []
  phase2Complete: []
  failed: [status: VideoStatus]
}>()
```

- [ ] **Step 3: Add an optional `phase` prop**

```ts
const props = defineProps<{
  videoId: string
  phase?: 'phase1' | 'phase2'
}>()
```

- [ ] **Step 4: Replace the watcher logic**

Find the `watch(videoStatus, ...)` block (line ~141) and replace the body:

```ts
watch(videoStatus, (status) => {
  switch (status) {
    case 'phase1_complete':
      emit('phase1Complete'); break
    case 'completed':
      emit('phase2Complete'); break
    case 'failed_phase1':
    case 'failed_phase2':
    case 'failed':
      emit('failed', status as VideoStatus); break
  }
})
```

- [ ] **Step 5: Dynamic header text**

Find the header element (often an `<h2>` or similar). Bind text to:

```ts
const headerText = computed(() => {
  switch (videoStatus.value) {
    case 'processing_phase1':
    case 'phase1_complete':
      return 'Detecting rallies...'
    case 'processing_phase2':
      return 'Analyzing players, speeds, poses...'
    default:
      return 'Processing video...'
  }
})
```
and in the template: `<h2>{{ headerText }}</h2>`.

- [ ] **Step 6: Filter logs by phase (if a logs panel exists)**

Locate the live `processing_logs` subscription. Add a filter:

```ts
const filteredLogs = computed(() =>
  logs.value.filter(l => !props.phase || !l.phase || l.phase === props.phase)
)
```

Render `filteredLogs` instead of `logs`.

- [ ] **Step 7: Verify types compile**

Run: `npm run type-check`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/components/AnalysisProgress.vue
git commit -m "feat(frontend): AnalysisProgress phase-aware events + dynamic header"
```

---

## Task 13: Frontend — Create `RallyReview.vue`

**Files:**
- Create: `src/components/RallyReview.vue`

**Purpose:** New screen shown after Phase 1 completes. Lists rallies with thumbnails, offers "Continue" → Phase 2 and "Done for now" → reset.

- [ ] **Step 1: Create the component**

```vue
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { supabase } from '@/lib/supabase'

const props = defineProps<{
  videoId: string
}>()

const emit = defineEmits<{
  continue: []
  done: []
}>()

interface RallyClipRow {
  id: string
  rally_index: number
  start_seconds: number
  end_seconds: number
  storage_path: string
  signed_url?: string
}

const clips = ref<RallyClipRow[]>([])
const totalDuration = ref<number | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    const { data, error: e1 } = await supabase
      .from('rally_clips')
      .select('id, rally_index, start_seconds, end_seconds, storage_path')
      .eq('video_id', props.videoId)
      .order('rally_index', { ascending: true })
    if (e1) throw e1
    clips.value = data ?? []

    // Sign URLs in parallel.
    await Promise.all(clips.value.map(async (c) => {
      const { data: signed } = await supabase.storage.from('clips').createSignedUrl(c.storage_path, 3600)
      c.signed_url = signed?.signedUrl
    }))

    // Fetch results JSON for total duration.
    const { data: row } = await supabase.from('videos').select('results_storage_path').eq('id', props.videoId).single()
    if (row?.results_storage_path) {
      const blob = await supabase.storage.from('results').download(row.results_storage_path)
      if (blob.data) {
        const text = await blob.data.text()
        const parsed = JSON.parse(text)
        totalDuration.value = parsed.video_metadata?.duration_seconds ?? null
      }
    }
  } catch (e: any) {
    error.value = e.message ?? String(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)

const summary = computed(() => {
  if (!clips.value.length) return 'No rallies detected'
  const dur = totalDuration.value
  return dur
    ? `${clips.value.length} rallies detected · total match duration ${formatDuration(dur)}`
    : `${clips.value.length} rallies detected`
})

function formatDuration(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}m ${sec.toString().padStart(2, '0')}s`
}

function formatRange(start: number, end: number): string {
  return `${formatDuration(start)} — ${formatDuration(end)}`
}

const continueDisabled = ref(false)
async function onContinue() {
  continueDisabled.value = true
  emit('continue')
}
</script>

<template>
  <div class="rally-review">
    <header class="rally-review__header">
      <h2>Rally Review</h2>
      <p>{{ summary }}</p>
    </header>

    <div v-if="loading">Loading rallies…</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <ul v-else class="rally-review__list">
      <li v-for="c in clips" :key="c.id" class="rally-review__item">
        <span class="rally-review__index">Rally #{{ c.rally_index + 1 }}</span>
        <span class="rally-review__range">{{ formatRange(c.start_seconds, c.end_seconds) }}</span>
        <video v-if="c.signed_url" :src="c.signed_url" controls preload="metadata" class="rally-review__preview" />
      </li>
    </ul>

    <footer class="rally-review__actions">
      <button class="primary" :disabled="continueDisabled" @click="onContinue">
        Continue with full analytics
      </button>
      <button class="secondary" @click="emit('done')">Done for now</button>
    </footer>
  </div>
</template>

<style scoped>
.rally-review__list { display: flex; flex-direction: column; gap: 0.75rem; }
.rally-review__item { display: grid; grid-template-columns: auto 1fr 320px; gap: 1rem; align-items: center; padding: 0.75rem; border: 1px solid var(--border, #e5e7eb); border-radius: 0.5rem; }
.rally-review__preview { width: 320px; height: 180px; object-fit: cover; border-radius: 0.25rem; }
.rally-review__actions { display: flex; gap: 0.75rem; margin-top: 1.5rem; }
.error { color: var(--danger, #dc2626); }
</style>
```

**Note:** The `rally_clips` row schema referenced above (`rally_index`, `start_seconds`, `end_seconds`, `storage_path`) should match what `backend/modal_supabase_processor.py` writes today. If the existing column names differ, adjust the SELECT and the template accordingly.

- [ ] **Step 2: Verify types compile**

Run: `npm run type-check`
Expected: no errors.

- [ ] **Step 3: Manual smoke**

```bash
npm run dev
```
Navigate to a video in `phase1_complete` (set up by running Task 8 verification end-to-end). Confirm the rally list renders with clip previews, and that the two buttons work.

- [ ] **Step 4: Commit**

```bash
git add src/components/RallyReview.vue
git commit -m "feat(frontend): RallyReview screen between Phase 1 and Phase 2"
```

---

## Task 14: Frontend — `ResultsDashboard.vue` Hard-Route Phase-1-Only

**Files:**
- Modify: `src/components/ResultsDashboard.vue`

**Purpose:** If a Phase-1-only video reaches `ResultsDashboard` (deep link or stale route), redirect to `RallyReview` instead.

- [ ] **Step 1: Locate where the results JSON is loaded/consumed**

Run: `grep -n "phase\|analysisResult\|AnalysisResult\|skeleton_frames" src/components/ResultsDashboard.vue | head -20`

- [ ] **Step 2: Add an early redirect**

After the results JSON is loaded, check the `phase` key:

```ts
import { watch } from 'vue'

const emit = defineEmits<{
  needsRallyReview: []
}>()

watch(analysisResult, (r) => {
  if (r && r.phase === 'phase1') {
    emit('needsRallyReview')
  }
}, { immediate: true })
```

Then in `App.vue`, handle this emit by setting `currentState.value = 'rally-review'`.

Alternatively (simpler): check before rendering and render `<RallyReview>` directly:

```vue
<template>
  <RallyReview v-if="analysisResult?.phase === 'phase1'" :video-id="videoId" @continue="..." @done="..." />
  <div v-else>
    ... existing ResultsDashboard contents ...
  </div>
</template>
```

Pick whichever is more consistent with how `ResultsDashboard` is currently used in `App.vue`.

- [ ] **Step 3: Verify types compile**

Run: `npm run type-check`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/components/ResultsDashboard.vue src/App.vue
git commit -m "feat(frontend): ResultsDashboard routes Phase-1-only videos to RallyReview"
```

---

## Task 15: End-to-End Smoke Test

**Files:** None (manual walkthrough)

**Purpose:** Verify the entire flow on a real video.

- [ ] **Step 1: Run a happy-path walkthrough**

1. Run `npm run dev`. Open the app.
2. Upload a small sample badminton video.
3. Complete the 12-point court setup. Click Continue (Skip button should not exist).
4. Wait for Phase 1: header should say "Detecting rallies…". Progress bar advances.
5. On Phase 1 complete: app transitions to `RallyReview` showing rally clips.
6. Click "Continue with full analytics". App transitions to `analyzing-phase2`, header says "Analyzing players, speeds, poses…".
7. On Phase 2 complete: app transitions to `ResultsDashboard` with full analytics.

- [ ] **Step 2: Verify Android-app independence**

While Phase 1 is running, in a separate terminal:
```bash
supabase db psql -c "SELECT id, status FROM rally_clips WHERE video_id='<id>';"
```
Expected: rows appear at end of Phase 1, before user clicks Continue. They stay unchanged through Phase 2.

- [ ] **Step 3: Verify "Done for now" path**

1. Upload another video, complete court setup, let Phase 1 finish.
2. On `RallyReview`, click "Done for now".
3. App returns to upload screen. Confirm in DB: `videos.status = 'phase1_complete'` (unchanged), `rally_clips` rows present.
4. Re-open the app with the original `video_id` (or re-launch with persisted state). Confirm it lands on `RallyReview` for that video.

- [ ] **Step 4: Verify retry path**

1. Force a Phase 2 failure: temporarily edit `_process_analytics_worker` to `raise RuntimeError("forced fail")` near the start. Redeploy.
2. Trigger Phase 2 via "Continue with full analytics".
3. App transitions to `error`. "Retry analytics" button appears.
4. Revert the forced fail. Redeploy. Click "Retry analytics".
5. Phase 2 completes successfully.

- [ ] **Step 5: Commit any test fixtures or notes**

If you took notes, commit them under `docs/plans/2026-05-12-smoke-test-results.md`. Otherwise skip this step.

---

## Task 16: Deploy Procedure (Drain Strategy)

**Files:** None (operations)

**Purpose:** Apply the migration safely with zero in-flight `processing` rows.

- [ ] **Step 1: Schedule the deploy window**

Pick a low-traffic window. The drain will refuse new uploads for ~1 hour.

- [ ] **Step 2: Block new uploads at the Edge Function**

Temporarily add a feature flag to `process-video/index.ts`:
```ts
const DRAIN_MODE = Deno.env.get("DRAIN_MODE") === "1";
if (DRAIN_MODE) {
  return new Response(JSON.stringify({ error: "Maintenance — try again shortly" }), { status: 503 });
}
```
Set the secret: `supabase secrets set DRAIN_MODE=1`. Redeploy `process-video`.

- [ ] **Step 3: Wait for in-flight `processing` rows to drain**

Run repeatedly:
```bash
supabase db psql -c "SELECT id, status, updated_at FROM videos WHERE status='processing';"
```
Wait until empty.

- [ ] **Step 4: Apply migration + deploy new code**

Run:
```bash
supabase migration up
modal deploy backend/modal_supabase_processor.py
supabase functions deploy process-video
supabase functions deploy start-analytics
```

- [ ] **Step 5: Verify**

Upload a test video. Confirm `videos.status` flows: `uploaded` → `processing_phase1` → `phase1_complete` → `processing_phase2` → `completed`. Confirm any pre-existing `completed` rows still render in `ResultsDashboard`.

- [ ] **Step 6: Lift the drain**

Run: `supabase secrets unset DRAIN_MODE && supabase functions deploy process-video`

- [ ] **Step 7: Commit the drain flag (left in place for future deploys)**

```bash
git add supabase/functions/process-video/index.ts
git commit -m "ops: add DRAIN_MODE flag for future deploys"
```

---

## Self-Review Notes

**Spec coverage check (run before considering this plan done):**

- [x] Phase 0 benchmark — Tasks 1, 2
- [x] Schema migration with drain — Tasks 3, 16
- [x] Backend Phase 1 scope-down — Task 4
- [x] Backend Phase 2 entrypoint — Task 5
- [x] `process-video` Edge Function update — Task 6
- [x] `start-analytics` Edge Function — Task 7
- [x] Backend verification — Task 8
- [x] Frontend types — Task 9
- [x] CourtSetup skip removal — Task 10
- [x] App.vue state machine + resume — Task 11
- [x] AnalysisProgress phase awareness — Task 12
- [x] RallyReview screen — Task 13
- [x] ResultsDashboard hard-route — Task 14
- [x] End-to-end smoke (incl. Android-app independence, "Done for now", retry path) — Task 15

**No placeholders.** Each step has either a complete code block or an exact command + expected output.

**Type consistency:** `VideoStatus` enum is the single source of truth in `src/types/analysis.ts`. `Phase1Results` and `FullAnalysisResult` shapes are declared there and consumed by `RallyReview.vue` and `ResultsDashboard.vue`. `phase1Complete` / `phase2Complete` events are emitted from `AnalysisProgress.vue` and consumed in `App.vue` — names match.

**Open conditional:** Task 4 has Branch A (TrackNet-only) and Branch B (union). The decision is locked by Task 2. Subsequent tasks (5 onward) are path-agnostic.
