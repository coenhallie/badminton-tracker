# OC-SORT A/B Testing Design

**Date:** 2026-04-16
**Status:** Approved
**Goal:** Add per-video tracker toggle (BoT-SORT vs OC-SORT) for A/B testing player tracking quality. Remove the losing tracker once evaluation is complete.

## Context

Current player tracking uses Ultralytics' built-in BoT-SORT via `model.track(persist=True)` with a custom badminton config (90-frame buffer, low thresholds). Research shows OC-SORT outperforms BoT-SORT on SportsMOT (+5 HOTA, 60% fewer ID switches) because BoT-SORT's appearance model gets confused by identical uniforms — a problem in badminton where players wear similar kit.

OC-SORT uses an observation-centric approach: virtual trajectory interpolation during occlusions (ORU), velocity direction consistency (OCM), and last-observation recovery (OCR) — all pure motion, no ReID.

## Integration Library

**Roboflow `trackers` v2.3.0** (Apache 2.0 license):
- Clean-room OC-SORT implementation
- Lightweight deps (numpy, supervision, scipy, opencv)
- API: `OCSORTTracker.update(sv.Detections) → sv.Detections` with `.tracker_id`
- No AGPL concerns (unlike BoxMOT)

Ultralytics does not support OC-SORT natively — only BoT-SORT and ByteTrack. OC-SORT requires decoupled detection + external tracking.

## Data Flow

Mirrors the existing `cameraAngle` pattern:

```
VideoUpload.vue (toggle: botsort | ocsort)
  → Convex createVideo mutation (stored in videos table)
    → Convex processVideo action (read from DB, pass to Modal)
      → modal_convex_processor.py (branch tracker logic in frame loop)
```

Default: `"botsort"` — zero impact on existing videos.

## Schema

```typescript
// convex/schema.ts — videos table
trackerType: v.optional(v.union(v.literal("botsort"), v.literal("ocsort"))),
```

## Backend Changes

### modal_convex_processor.py

**Before the frame loop** — conditionally set up tracker:
- BoT-SORT: write YAML config (unchanged from today)
- OC-SORT: instantiate `OCSORTTracker` with badminton-tuned params

**In the frame loop** — branch on tracker_type:
- BoT-SORT: `pose_model.track(frame, persist=True, tracker=config_path, ...)`
- OC-SORT: `pose_model(frame, ...)` → `sv.Detections.from_ultralytics()` → `tracker.update()` → map `tracker_id` back to keypoints by index

**OC-SORT parameters (tuned for badminton):**
- `lost_track_buffer=90` (3s at 30fps, matches BoT-SORT's track_buffer)
- `frame_rate=fps` (dynamic from video)
- `minimum_consecutive_frames=2` (fast track confirmation)
- `minimum_iou_threshold=0.3` (lenient for far player)
- `high_conf_det_threshold=0.3` (low for far player detection)
- `direction_consistency_weight=0.2` (OC-SORT's OCM feature)

**Keypoint mapping:** The tracker only uses bounding boxes for association. Keypoints are extracted from the original Ultralytics result. Index correspondence between `pose_results[0]` detections and `sv.Detections` is maintained since `from_ultralytics()` preserves detection order.

### modal_inference.py

Add `trackers>=2.3.0` and `supervision>=0.26.0` to Modal image `pip_install`.

## Frontend Changes

### VideoUpload.vue

- State: `const trackerType = ref<'botsort' | 'ocsort'>('botsort')`
- UI: Two-button toggle group (same style as camera angle selector)
- Pass to `createVideo` mutation alongside `cameraAngle`

### No changes needed in:
- `AnalysisProgress.vue` — already passes just `videoId`; action reads from DB
- Playback/analysis components — skeleton data format is identical regardless of tracker
- `PlayerIdentityTracker` — receives `track_id` per skeleton, source-agnostic

## Files Changed

| File | Change |
|------|--------|
| `convex/schema.ts` | Add `trackerType` field |
| `convex/videos.ts` | Accept in `createVideo`, pass in `processVideo` |
| `src/components/VideoUpload.vue` | Tracker toggle UI + state |
| `backend/modal_convex_processor.py` | Receive `tracker_type`, branch tracker logic |
| `backend/modal_inference.py` | Add deps to Modal image |

## Temporary Nature

This is an A/B test. Once evaluation is complete, the losing tracker and its code path will be removed. The toggle, schema field, and branching logic are designed to be cleanly removable.
