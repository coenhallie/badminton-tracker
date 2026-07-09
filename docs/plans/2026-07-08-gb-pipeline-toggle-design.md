# Good-Badminton Pipeline Toggle — Design

**Date:** 2026-07-08
**Status:** Approved
**Goal:** Let the user choose, per video, between the existing Phase 1 pipeline
("legacy") and a Good-Badminton-enhanced pipeline ("gb_fusion"), so two runs of
the same video can be compared side by side.

## Background

A validation spike (2026-07-08, run on Modal A10G against previously processed
videos) showed that Good-Badminton's Apache-2.0 `yolo11s-ball.pt` shuttlecock
detector complements our TrackNetV3 tracking:

| In-rally shuttle coverage | Ours | GB detector | Union |
|---|---|---|---|
| Yamaguchi vs Chen (12,032 frames) | 30.8% | 45.8% | **58.5%** |
| short_test1 (3,602 frames) | 35.0% | 19.9% | **45.8%** |

Where both detect, median position difference is 4–5 px (same object). Shuttle
coverage feeds rally-boundary detection — our weakest metric (recall 80–87%,
see `2026-05-12-rally-benchmark-results.md`). Spike artifacts live in the
session scratchpad (`spike_good_badminton.py`, `spike_results.json`).

## Scope decisions (user-approved)

- The toggle switches **Phase 1 shuttle detection only**. Court setup stays
  manual and identical in both variants so rally results are directly
  comparable. Good-Badminton's court auto-detection is explicitly out of scope
  for this feature.
- Two variants only: `legacy`, `gb_fusion`. No GB-only mode (add later via a
  two-line CHECK migration if wanted).
- Comparison workflow: **duplicate & re-run**. A completed video gets a
  "re-run with the other pipeline" action that creates a sibling video row with
  identical inputs (same storage bytes, same manual court keypoints).
- Comparison UI: **badges + sibling links** only. No merged comparison view;
  the user opens both results dashboards.
- Toggle placement: **CourtSetup**, next to the start-analysis action.
  **Default: legacy** — existing behavior is unchanged unless opted in.
  - *Amended 2026-07-09:* moved to **VideoUpload** (file-preview card, set at
    row-insert time) — in CourtSetup the selector sat inside the
    keypoint-controls overlay on top of the video frame. CourtSetup no longer
    reads or writes `pipeline_variant` (which also moots the
    duplicate-reset-to-legacy concern); `duplicate-video` still sets the
    variant server-side.

## 1. Data model — migration `0006_pipeline_variant`

```sql
ALTER TABLE videos
  ADD COLUMN pipeline_variant text NOT NULL DEFAULT 'legacy'
    CHECK (pipeline_variant IN ('legacy', 'gb_fusion')),
  ADD COLUMN source_video_id uuid REFERENCES videos(id) ON DELETE SET NULL;
```

- `pipeline_variant` is the single source of truth for which pipeline a video
  uses; Modal reads it from the row.
- `source_video_id` links a duplicate to its source. Sibling discovery is
  bidirectional: a duplicate points at its source via `source_video_id`; a
  source finds duplicates via reverse lookup (`source_video_id = my_id`).
- RLS: both columns ride on the existing single-owner row policies; no policy
  changes needed.
- The Phase 1 results JSON gains a top-level `pipeline_variant` field so
  stored results are self-describing.

## 2. Modal Phase 1 — shuttle fusion

**Weight distribution.** `yolo11s-ball.pt` (19 MB, single class `badminton`,
Apache-2.0, from Good-Badminton release v0.1.0) is uploaded to the existing
`badminton-tracker-models` volume at `/models/gb_ball/yolo11s-ball.pt` by a new
`backend/upload_gb_ball.py`, following the `upload_tracknet.py` pattern.

**Worker changes** (`backend/modal_supabase_processor.py`, Phase 1 only):

- `_process_video_worker` already fetches the video row; it reads
  `pipeline_variant` from it.
- For `gb_fusion`, a new `_run_gb_ball_pass()` mirrors `_run_tracknet_pass`:
  batched Ultralytics inference at conf 0.18 with Good-Badminton's candidate
  filters — box area ≤ 0.4% of frame area, aspect ratio ≤ 4.0 (from their
  `ShuttlecockTracker` defaults). Output: `{frame: {x, y, visible}}`.
- Per-frame fusion priority: **TrackNet → GB → existing YOLO fallback**, each
  position tagged with `source` (`"tracknet"` / `"gb_yolo"` / `"yolo"`). The
  fused positions dict feeds the existing union rally detectors
  (`rally_detection.py` + `rally_detection_shot_gap.py`) unchanged.
- **Fail loudly:** if `pipeline_variant = 'gb_fusion'` and the weight cannot be
  loaded, the run ends `failed_phase1` with an error naming
  `upload_gb_ball.py`. No silent degrade — a silently-legacy run would poison
  the A/B comparison.
- Processing logs include per-source shuttle coverage counts so GB's
  contribution is visible per run.
- Note: Good-Badminton's `opencv-python==4.10.0.84` pin only affects their
  court detector (HoughLinesP output shape). The ball weight runs on our
  existing Ultralytics/OpenCV stack; no pin changes.

## 3. Edge functions

**New: `duplicate-video`** (`supabase/functions/duplicate-video/index.ts`)

- Auth: JWT bearer, same pattern as `process-video`.
- Input: `{ video_id }`.
- Validations: caller owns the video; `manual_court_keypoints` is present;
  source status is not `processing_phase1`/`processing_phase2`.
- Actions (service role):
  1. Copy the storage object: `videos` bucket, source `storage_path` →
     `{owner_id}/{newId}.mp4` (server-side copy, no re-upload).
  2. Insert the sibling row: new id, same `filename`/`size`, copied
     `manual_court_keypoints` and `player_labels`,
     `pipeline_variant` = the **opposite** of the source's,
     `source_video_id` = source id, `status` = `uploaded`.
  3. On insert failure, best-effort delete the copied object; return 500.
- Output: `{ new_video_id }`.

`process-video` is **unchanged** — it already validates keypoints and status,
and Modal reads the variant from the row.

## 4. Frontend

- **CourtSetup.vue**: a two-option "Pipeline" selector (radio/segmented):
  "Current pipeline" (default) and "Good-Badminton fusion" with a one-line
  description each. Persisted in the same `.update()` call that saves
  `manual_court_keypoints`.
- **ResultsDashboard.vue**:
  - Variant badge chip (Legacy / GB fusion) in the header.
  - "Re-run with [other] pipeline" action: invokes `duplicate-video`, then
    `process-video` for the new id, then App.vue navigates directly into
    `analyzing-phase1` for the sibling (court setup is skipped — keypoints
    were copied).
  - "Compare: open [other] run" link when a sibling exists (lookup on load:
    follow `source_video_id`, plus reverse query).
- **RallyReview.vue**: same variant badge.
- **Types/API**: `src/types/analysis.ts` and `src/services/api.ts` gain
  `pipeline_variant: 'legacy' | 'gb_fusion'` and
  `source_video_id: string | null`.

## 5. Error handling

- `duplicate-video`: storage-copy failure → 500, no row created; insert
  failure → best-effort object cleanup, 500. Source mid-processing → 409.
- Missing GB weight on a `gb_fusion` run → `failed_phase1`, actionable error.
- Old rows: `DEFAULT 'legacy'` means every existing video reads as legacy;
  no backfill needed.

## 6. Verification (no test framework in this repo)

1. `vue-tsc` type-check passes.
2. Manual smoke: process a short video as legacy → re-run via the new action →
   sibling processes as `gb_fusion` → badges and sibling links render on both
   dashboards → rally clips exist on both.
3. Quantitative: `backend/scripts/compare_rallies.py` against the two sibling
   rows for the rally-count/boundary diff; processing logs show per-source
   shuttle coverage.
