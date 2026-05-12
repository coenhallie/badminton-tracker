# Two-Phase Rally-First Pipeline — Design

**Date:** 2026-05-12
**Status:** Draft — pending implementation plan
**Branch context:** `feat/supabase-migration`

## Summary

Split the current monolithic video-processing pipeline into two independently invoked phases:

1. **Phase 1 — Rally Segmentation.** Runs immediately after upload + court setup. Produces rally boundaries and per-rally MP4 clips. Surfaces to the user in a new "Rally Review" screen and to the planned Android app via existing `rally_clips` rows.
2. **Phase 2 — Full Analytics.** Triggered explicitly by the user from Rally Review. Adds pose, tracking, speed, heatmaps, body angles, and other analytics on top of phase-1 results.

The goal is to make rally segmentation available quickly (priority for the Android consumer) without forcing users to wait for full analytics they may not need.

## Motivation

Today's pipeline (`backend/modal_supabase_processor.py:_process_video_worker`) is a single monolithic GPU worker that gates the user-visible `videos.status` flip to `completed` behind the slowest stage (rally clip generation, which itself runs after the full YOLO frame loop and analytics). The web UI hides everything until that flip.

The user's primary current consumer of this pipeline is a planned/in-progress KMP Android app (`docs/plans/2026-05-04-kmp-rally-clips-*.md`) whose only need is accurate rally segmentation and per-rally clips. Forcing it to wait for full analytics is wasted time and wasted GPU cost. The Vue web app also benefits — many sessions never need the full analytics view, and the user can opt in only when they want it.

A prior `analysis-mode-selector` design (`docs/plans/2026-03-22-analysis-mode-selector-design.md`, marked obsolete) attempted an upfront "rally-only vs full" toggle. This design is different: rally segmentation always runs first; the user makes the analytics decision **after seeing the rallies**, not before processing starts.

## Goals

- Rally segmentation runs first, gated only by upload + court setup
- Per-rally clips are durable in Supabase and queryable by the Android app immediately after phase 1
- The Vue web app shows rally results between phases via a dedicated screen
- Full analytics is opt-in, triggered by an explicit user action
- Phase 2 is independently invocable (no idle GPU between phases)
- Existing `completed` videos remain accessible with no UX regression

## Non-goals

- Multi-video session UI (resume of a *prior* video is supported, but no video history listing)
- Sharing intermediate state in-memory between phase 1 and phase 2 (each phase is a cold Modal invocation)
- Re-running rally detection in phase 2 (rallies from phase 1 are authoritative)
- Phase-1 retry without re-upload (re-upload is simpler than a retry path for the smaller phase-1 work)
- Backward compatibility for in-flight `processing` rows at deploy time (drain instead)

## Architecture

### Pipeline flow

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Upload  │ -> │  Court setup │ -> │  Phase 1     │ -> │  Rally Review   │ -> │  Phase 2     │
│  (web)   │    │  (mandatory) │    │  (Modal #1)  │    │  (web, user     │    │  (Modal #2)  │
└──────────┘    └──────────────┘    │  rallies +   │    │   decides)      │    │  full        │
                                    │  clips       │    │                 │    │  analytics   │
                                    └──────────────┘    └─────────────────┘    └──────────────┘
                                           │                    │                      │
                                           ▼                    ▼                      ▼
                                    rally_clips rows    Android consumes       results_storage_path
                                    phase-1 results     rally_clips            augmented with full
                                    JSON                independently          analytics
```

### Modal entry points

| Entry point | Phase | Inputs | Output |
|---|---|---|---|
| `process_video(video_id)` (existing, scoped down) | 1 | `video_id` only | rally JSON, rally_clips rows, status `phase1_complete` |
| `process_analytics(video_id)` (new) | 2 | `video_id` only | augmented results JSON, status `completed` |

Both entry points read all needed state from Supabase (video file, court keypoints, phase-1 results). Cold starts; no inter-phase in-memory state.

### Supabase Edge Functions

| Edge Function | Behavior |
|---|---|
| `process-video` (existing) | Validates upload + court keypoints exist, sets `status='processing_phase1'`, invokes Modal `process_video`. Returns 202. |
| `start-analytics` (new) | Validates `status in ('phase1_complete', 'failed_phase2')`, sets `status='processing_phase2'`, invokes Modal `process_analytics`. Returns 202. |

Edge Functions set the new status **before** invoking Modal, so a Modal cold-start failure leaves the row in a phase-N processing state that a retry can reset.

### Status flow

`videos.status` enum (extended):

| Status | Set by | Meaning |
|---|---|---|
| `pending` | upload row insert | row created, no upload yet |
| `uploaded` | upload completion | file in storage, awaiting court setup |
| `processing_phase1` | `process-video` Edge Function | phase 1 running |
| `phase1_complete` | Modal phase-1 worker | rallies + clips ready; awaiting user decision (terminal if user never opts into analytics) |
| `processing_phase2` | `start-analytics` Edge Function | phase 2 running |
| `completed` | Modal phase-2 worker | full analytics ready (today's terminal state) |
| `failed_phase1` | Modal phase-1 worker on error | phase 1 failed; re-upload required |
| `failed_phase2` | Modal phase-2 worker on error | phase 2 failed; retry via "Retry analytics" button |

## Phase 0 — Rally accuracy benchmark (prerequisite)

Before locking phase 1's scope (TrackNet-only vs full YOLO frame loop), run a one-off benchmark to settle the accuracy/speed tradeoff.

### Comparison

| Path | Inputs | Rally signals |
|---|---|---|
| **Baseline (today's union)** | TrackNet shuttle + YOLO per-frame detections | gradient ∪ shot-gap |
| **TrackNet-only candidate** | TrackNet shuttle only | gradient only |

### Metrics

- Per-video rally **count delta** (union vs TrackNet-only)
- For matched rallies (by IoU on time intervals): **boundary error** in seconds (start delta, end delta)
- Per-video **wall-clock time delta** (estimated savings if YOLO frame loop is skipped in phase 1)

### Decision rule

Go TrackNet-only in phase 1 if and only if:

- TrackNet-only recall ≥ **95%** of union rallies, AND
- Boundary error (95th percentile) **< 1s**, AND
- Estimated phase-1 time savings vs union path is **> 40%** of current total pipeline time

Otherwise, phase 1 runs the YOLO frame loop with detection-only (no pose) so the shot-gap signal is preserved. Phase 1's main win in that case is UX — rallies visible before analytics — rather than wall-clock speed.

### Test set

Pull 5–10 videos already processed by today's pipeline from Supabase storage. They already have union output, so the diff is a re-run of TrackNet+gradient-only against existing data. Supplement with user-provided samples if coverage gaps exist (camera angle, indoor/outdoor, singles/doubles).

### Deliverable

`backend/benchmarks/rally_accuracy_benchmark.py` — runs both paths, persists outputs, emits a comparison report committed to `docs/plans/`. The implementation plan branches phase-1 scope on this report's recommendation.

## Phase 1 — Rally Segmentation Pipeline

### Inputs

- `video_id`
- `videos.manual_court_keypoints` (12-point payload, persisted by `CourtSetup.vue`)
- Video file from `videos` Supabase Storage bucket

### Steps

1. **Download video** to Modal worker scratch.
2. **Shuttle tracking** — TrackNet full-video pass.
3. **YOLO frame loop** — *branches on benchmark result:*
   - If TrackNet-only path wins: skip entirely.
   - If union path wins: YOLO detection per frame only (no pose) to feed shot-gap signal.
4. **Rally detection** — `rally_detection.detect_rallies` (gradient signal). If step 3 ran, also `rally_detection_shot_gap.detect_rallies_from_shots` and union via `union_rallies`.
5. **Rally JSON serialization** — slim payload:
   ```json
   {
     "rallies": [...],
     "shuttle_positions": {...},
     "fps": ...,
     "total_frames": ...,
     "video_metadata": {...},
     "phase": "phase1"
   }
   ```
   Written to `results_storage_path` (same bucket as today's results).
6. **Rally clip generation** — ffmpeg stream-copy per rally, upload to `clips` bucket, insert one `rally_clips` row per rally (today's behavior, moved earlier).
7. **Flip `videos.status` to `phase1_complete`**.

### Outputs (visible to clients)

- `rally_clips` rows — Android app consumes immediately
- `results_storage_path` JSON — Vue web app fetches for `RallyReview.vue`
- `videos.status = phase1_complete` — gates the "Continue" UI

### Key design choices

- **TrackNet shuttle_positions are persisted in the phase-1 JSON.** Phase 2 reuses them rather than re-running TrackNet — saves a redundant GPU pass and real Modal cost. JSON size impact is negligible.
- **Rally clips are cut in phase 1 unconditionally.** Even if the user never opts into analytics, the Android app has everything it needs.
- **No pose, no tracking, no court calibration validation, no speed, no heatmaps in phase 1.** All deferred to phase 2.
- **Rally clip generation soft-failure preserved.** If some clips fail to cut/upload (today's behavior), `status='phase1_complete'` still flips; missing clips are logged.

## Phase 2 — Full Analytics Pipeline

### Trigger

`start-analytics` Edge Function, invoked from `RallyReview.vue` when user clicks "Continue with full analytics".

### Inputs

- `video_id`
- Video file from `videos` bucket
- `videos.manual_court_keypoints`
- Phase-1 results JSON from `results_storage_path` (rallies + shuttle_positions)

### Steps

1. **Download video** to Modal worker scratch.
2. **Load YOLO pose + detection models.** No TrackNet — shuttle data is already on disk.
3. **YOLO per-frame loop** — pose estimation, player detection, tracking, court calibration validation.
4. **Analytics** — speed, heatmaps, body angles, shot classification, zone analytics (everything `useAdvancedAnalytics.ts` consumes today).
5. **Results JSON merge** — load phase-1 JSON, augment with analytics fields, write back to `results_storage_path`. Phase-1 keys (`rallies`, `shuttle_positions`, `fps`, `total_frames`, `video_metadata`) are preserved exactly.
6. **Flip `videos.status` to `completed`**.

### Key design choices

- **Phase 2 does NOT re-run rally detection.** Phase-1 rallies are authoritative. If phase 2 produced different rallies, it would invalidate the clips the Android app already has.
- **Phase 2 does NOT touch `rally_clips` rows.** Those are phase 1's territory.
- **Results JSON is augmented in place, not replaced.** Phase 2 adds keys; it never overwrites phase-1 keys.
- **Failure is non-destructive.** On `failed_phase2`, phase-1 results remain intact and the user can retry. Unconditional retries (no cap).

## Data Model

Minimal, additive changes. No table renames, no data migration risk.

### `videos` table

| Field | Today | After |
|---|---|---|
| `status` | enum: `pending`, `uploaded`, `processing`, `completed`, `failed` | enum: `pending`, `uploaded`, `processing_phase1`, `phase1_complete`, `processing_phase2`, `completed`, `failed_phase1`, `failed_phase2` |
| `manual_court_keypoints` | nullable JSON | nullable JSON (unchanged; now required before phase-1 trigger — enforced at app layer) |
| `results_storage_path` | path to single results JSON | path to results JSON (phase 1 writes; phase 2 augments in place) |
| `progress`, `current_frame`, `total_frames` | unchanged | unchanged — both phases write to these independently (each phase's progress is 0–100 within its own scope) |
| `error` | unchanged | unchanged — carries either phase's error |

### `processing_logs` table

| Field | Change |
|---|---|
| `phase` (new) | TEXT NULL, values `'phase1' \| 'phase2'`. NULL acceptable for legacy rows. Used to filter live logs in `AnalysisProgress.vue`. |

### `rally_clips` table

No schema change. Rows written earlier (end of phase 1) instead of end of monolithic pipeline.

### Storage buckets

No change. `videos`, `clips`, `results` keep current responsibilities.

### RLS / policies

No change. Same `user_id` ownership rules apply across both phases.

### Migration

Single Supabase migration that:

1. Adds new enum values to `videos.status` (Postgres `ALTER TYPE ... ADD VALUE`, non-breaking — legacy `processing` and `failed` values continue to exist).
2. One-time SQL remap of legacy rows for clarity:
   - `UPDATE videos SET status='failed_phase2' WHERE status='failed'` — closest semantic match for prior monolithic failures.
   - No remap needed for `status='processing'` — drain (below) guarantees zero such rows at deploy time.
3. Adds the new `phase` column on `processing_logs` (TEXT NULL).

Legacy `processing` and `failed` values remain in the enum after migration for historical safety, but no live code path emits them after deploy.

### Deploy procedure (drain strategy)

Per "stable and recommended" preference, given low traffic:

1. Block new uploads at Edge Function layer (return 503) ~1h before deploy.
2. Wait for all rows with `status='processing'` to reach a terminal state (`completed` or `failed`).
3. Deploy schema migration + new Edge Function + new Modal entrypoint atomically.
4. Re-enable uploads.

No in-flight remap needed because drain guarantees zero `processing` rows at deploy time.

## Frontend (Vue Web App)

### App state machine (`src/App.vue`)

Today's `AppState` (line 44):

```ts
type AppState = 'upload' | 'court-setup' | 'analyzing' | 'results'
```

New:

```ts
type AppState =
  | 'upload'
  | 'court-setup'
  | 'analyzing-phase1'
  | 'rally-review'      // new — between phases
  | 'analyzing-phase2'
  | 'results'
  | 'error'             // distinguishes failed_phase1 vs failed_phase2 via videoStatus
```

### Transitions

| From | To | Trigger |
|---|---|---|
| `upload` | `court-setup` | upload row inserted with `status='uploaded'` |
| `court-setup` | `analyzing-phase1` | `handleCourtSetupComplete` → invokes `process-video` Edge Function |
| `analyzing-phase1` | `rally-review` | live subscription sees `status='phase1_complete'` |
| `analyzing-phase1` | `error` | live subscription sees `status='failed_phase1'` |
| `rally-review` | `analyzing-phase2` | user clicks "Continue with full analytics" → invokes `start-analytics` Edge Function |
| `rally-review` | end-of-session | user clicks "Done for now" or closes tab |
| `analyzing-phase2` | `results` | live subscription sees `status='completed'` |
| `analyzing-phase2` | `error` | live subscription sees `status='failed_phase2'` |
| `error` | `analyzing-phase2` | "Retry analytics" button (only when `failed_phase2`) |

### Resume from any status

On app load with an existing `video_id`, hydrate `AppState` from `videos.status`:

| Loaded status | Initial state |
|---|---|
| `pending` | `upload` |
| `uploaded` | `court-setup` |
| `processing_phase1` | `analyzing-phase1` |
| `phase1_complete` | `rally-review` |
| `processing_phase2` | `analyzing-phase2` |
| `completed` | `results` |
| `failed_phase1`, `failed_phase2` | `error` |

Legacy `failed` rows (pre-migration) are remapped to `failed_phase2` by the migration (see Data Model), so they hydrate to `error` via the row above.

Each session shows one active video. Re-uploading creates a fresh `video_id`; the prior video's row stays in DB and its rally clips remain available for the Android app. No multi-video history UI in this scope.

### `CourtSetup.vue` changes

- **Remove `skip` emit.**
- **Remove Skip button from UI.**
- `Continue` button stays disabled until all 12 points placed (today's behavior).
- `App.vue:704` `handleCourtSetupSkip` handler is deleted.

### New screen: `RallyReview.vue`

Shown in state `rally-review`. Renders:

- **Rally list with thumbnails** — one row per `rally_clips` entry: rally number, start/end timestamp, duration, clickable thumbnail/clip preview (signed URLs from `clips` bucket).
- **Summary header**: "{N} rallies detected, total match duration {X}". Pulled from phase-1 results JSON.
- **Primary action**: "Continue with full analytics" → invokes `start-analytics`.
- **Secondary action**: "Done for now" → returns to upload screen. State persists server-side; user can come back via resume.
- Re-uses existing rally-related components (e.g., `RallyTimeline.vue`) where their inputs can be satisfied from rally-only data.

### `AnalysisProgress.vue` changes

- **Replace binary `complete` event** with `phase1Complete` and `phase2Complete`. `App.vue` listens to both.
- **Header text** reflects active phase: "Detecting rallies..." (phase 1), "Analyzing players, speeds, poses..." (phase 2). Derived from `videos.status`.
- **Progress bar** keeps using `videos.progress` — each phase writes to it independently (0–100 within its own scope).
- **Live logs panel** filters by `processing_logs.phase` for the active phase.

### `ResultsDashboard.vue` changes

- **Hard-route phase-1-only videos away from `ResultsDashboard`.** If anything deep-links to a `phase1_complete` video, redirect to `RallyReview.vue`. No fallback banner.
- No other changes when full analytics are present — same behavior as today.

### New Supabase Edge Function: `start-analytics`

Mirrors today's `process-video`. Inputs: `{ video_id }`. Validates:

- Caller owns the video (RLS + explicit check)
- `status in ('phase1_complete', 'failed_phase2')`
- `results_storage_path` is non-null

Sets `status='processing_phase2'` before invoking Modal `process_analytics`. Returns 202 on success, 4xx on validation, 5xx on Modal invocation failure.

## Error Handling & Edge Cases

| Failure | Detection | User-visible behavior |
|---|---|---|
| Phase 1 fails | Worker writes `status='failed_phase1'` + `error` | `error` state; "Rally detection failed: {error}"; primary action: re-upload |
| Phase 2 fails | Worker writes `status='failed_phase2'` + `error` | `error` state; "Analytics failed: {error}"; primary action: "Retry analytics" (unconditional retries) |
| Modal timeout | `on_failure` hook sets `failed_phaseN` | Same as above |
| Rally clip cut partial failure | Soft failure (today's behavior preserved) | `status='phase1_complete'` proceeds; missing clips logged; Android sees existing clips |
| User clicks "Continue" twice rapidly | Edge Function validates status; UI disables button on click | Idempotent; second click 409s server-side, no-op client-side |
| User refreshes during `processing_phase2` | Resume table hydrates to `analyzing-phase2`; live subscription resumes | Seamless |
| User uploads new video while old is `phase1_complete` | New row inserted; becomes session active; old row untouched | Old `rally_clips` persist for Android |
| `start-analytics` called with missing `results_storage_path` | Edge Function pre-flight | 409 with clear error; user re-uploads |
| Court keypoints missing at phase 1 invoke | Edge Function pre-flight | 400 "Complete court setup first"; UI shouldn't allow this state |

### Edge Function invariants

Both `process-video` and `start-analytics`:

- Validate caller owns `video_id`
- Validate current `status` matches expected pre-state (race protection)
- Set new status **before** invoking Modal
- Return 202 on success, 4xx on validation, 5xx on Modal invocation failure

### Observability

- All phase transitions logged to `processing_logs` with `phase` column populated
- Worker failures write structured error to `videos.error`
- Drain-at-deploy procedure documented in this design doc

## Testing Strategy

### Backend (Python / Modal)

- **Unit tests** for rally detection wrappers — phase-1 output identical to today's pipeline on fixture (regression guard).
- **Phase 0 benchmark script** — `backend/benchmarks/rally_accuracy_benchmark.py`. Itself a deliverable.
- **Integration test for `process_analytics`** — given a `video_id` with phase-1 results fixture in Supabase, runs phase 2 on a small clip and verifies merged results JSON has all expected analytics keys plus untouched phase-1 keys.
- **Schema-merge unit test** — phase-1 keys preserved exactly after phase-2 augmentation.

### Edge Functions

- **`process-video`** — input validation (owns video, status is `uploaded`, keypoints present). Reject all other states.
- **`start-analytics`** — input validation (owns video, status is `phase1_complete` or `failed_phase2`, `results_storage_path` non-null). 409 on bad state.

### Frontend (Vue)

- **State machine unit tests** — for each `videos.status` value, resume hydration produces the correct `AppState`.
- **`RallyReview.vue` component tests** — rally rows render from `rally_clips` fixture; "Continue" triggers Edge Function call; "Done for now" cleanly resets session.
- **`CourtSetup.vue`** — verify Skip is gone, Continue disabled until 12 points placed.
- **`AnalysisProgress.vue`** — emits `phase1Complete` on `phase1_complete` status, `phase2Complete` on `completed`.

### End-to-end smoke

Scripted smoke on a sample video:

1. Upload → court setup → phase 1 runs → `rally-review` screen with N rally clips.
2. Click "Continue" → phase 2 runs → results screen with full analytics.
3. Verify `rally_clips` rows queryable by a separate (Android-simulating) client during step 1 and unchanged through step 2.
4. "Done for now" path: close tab after step 1, re-open → lands on `rally-review` for the same video.

### Migration & deploy verification

- Pre-deploy: zero rows with `status='processing'`.
- Post-deploy: fresh upload lands in `processing_phase1`.
- Existing `completed` rows still render in `ResultsDashboard.vue`.

## Open Questions

None at design time. The one residual unknown — phase-1 scope (TrackNet-only vs union) — is explicitly settled by the Phase 0 benchmark before implementation begins.

## Files Touched (preview — final list in implementation plan)

**Backend:**
- `backend/modal_supabase_processor.py` — scope down `process_video` to phase 1; add new `process_analytics` entrypoint
- `backend/benchmarks/rally_accuracy_benchmark.py` — new (Phase 0)
- `backend/rally_detection.py`, `backend/rally_detection_shot_gap.py` — no changes expected; invoked from new entry points

**Supabase:**
- `supabase/migrations/<timestamp>_two_phase_pipeline.sql` — new (enum values + `phase` column)
- `supabase/functions/process-video/` — scope-down logic for phase 1
- `supabase/functions/start-analytics/` — new

**Frontend:**
- `src/App.vue` — state machine, transitions, resume hydration
- `src/components/CourtSetup.vue` — remove skip
- `src/components/AnalysisProgress.vue` — phase-aware events and headers
- `src/components/ResultsDashboard.vue` — hard-route phase-1-only videos away
- `src/components/RallyReview.vue` — new
- `src/types/analysis.ts` — new status enum values, possibly split `AnalysisResult` into phase-1 vs phase-2 shapes

## Deliverables (order)

1. Phase 0 benchmark + decision on phase-1 scope
2. Schema migration + drain procedure execution
3. Backend split (Modal entry points + Edge Function)
4. Frontend state machine + new `RallyReview.vue`
5. `CourtSetup.vue` skip removal
6. End-to-end smoke + retry / resume tests
