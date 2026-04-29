# Camera Angle Rally Detection Presets

> **Status: obsolete** — the feature described here was removed in v1.9. Kept as a historical design record; do not implement or reference.

**Date**: 2026-04-09
**Status**: Approved

## Problem

Rally detection produces too many false rallies on amateur footage filmed from a corner/side angle. The current algorithm is tuned for professional overhead cameras and works well there. Corner angles introduce more TrackNet coordinate jitter (shuttle varies in apparent size, more occlusion) which creates phantom direction reversals that pass the current thresholds.

## Solution

Add a "Camera Position" selector (`"overhead"` | `"corner"`) that switches rally detection parameters on both backend and client-side. Default to `"overhead"` so existing professional footage is unchanged.

## Data Flow

Same path as `analysisMode`:

```
VideoUpload.vue (selector)
  -> convex videos.createVideo (stored in DB as cameraAngle)
    -> convex videos.ts startProcessing (reads from DB, sends to Modal)
      -> Modal modal_convex_processor.py (passes to detect_rallies)
        -> rally_detection.py (applies parameter preset)

App.vue (reads from AnalysisResult)
  -> useAdvancedAnalytics (applies client-side parameter preset)
```

## Parameter Presets

### Backend (`rally_detection.py`)

| Parameter | Overhead | Corner | Rationale |
|---|---|---|---|
| `MIN_SPEED_SQ` | 225 (15px) | 900 (30px) | Corner jitter can be 15-20px; 30px filters noise while real shots still produce large displacement |
| `STRIDE` | 0.3s | 0.5s | Longer stride smooths per-frame jitter; trade-off is missing very fast exchanges (<1s), rare in amateur play |
| `min_shot_gap_s` | 0.6s | 0.8s | Larger cooldown prevents jitter registering as rapid-fire shots |
| `min_shots_per_rally` | 2 | 3 | 2 shots from noisy corner data is likely false; require more evidence |
| `min_rally_duration_s` | 2.0s | 3.0s | Eliminates short false bursts |
| `min_gap_duration_s` | 3.0s | 4.0s | Prevents closely-spaced noise clusters from merging into one false rally |
| Dot product threshold | `dot < 0` | `dot < -0.25 * |v1| * |v2|` | Require >105 degree direction change, rejecting slight wobbles from noise |

### Client-side (`useAdvancedAnalytics.ts`)

| Parameter | Overhead | Corner | Rationale |
|---|---|---|---|
| `MIN_SPEED_SQ` | 900 (30px) | 2500 (50px) | Client data is already sparse (YOLO ~10-40%); corner needs even higher threshold |
| `MIN_GAP_S` | 0.6s | 0.8s | Match backend |
| `RALLY_GAP_SECONDS` | 3.1s | 4.0s | Match backend gap threshold |
| `MIN_RALLY_DURATION_S` | 2.0s | 3.0s | Match backend |
| Min shots per rally | 2 | 3 | Match backend |
| Dot product threshold | `dot < 0` | `dot < -0.25 * |v1| * |v2|` | Match backend |

### Static cluster filtering (backend `modal_convex_processor.py`)

For corner angles without court keypoints, increase static cluster aggressiveness:
- `SHUTTLE_STATIC_COUNT_THRESHOLD`: 3 -> 2 (confirm static clusters faster)
- `_RALLY_STATIC_DIST` multiplier: 1.0x -> 1.5x (wider radius for clustering)

## Frontend UI

Second selector row below "Analysis Mode" in `VideoUpload.vue`, same button styling:

```
Camera Position
[Overhead]  [Corner / Side]
```

- Default: "Overhead"
- Only visible when not uploading (same as analysis mode)

## Files to Change

1. **`src/components/VideoUpload.vue`** -- Add camera position selector UI + pass `cameraAngle` on upload
2. **`src/types/analysis.ts`** -- Add `cameraAngle` to `UploadResponse` / `AnalysisResult` types
3. **`convex/schema.ts`** -- Add `cameraAngle` field to videos table
4. **`convex/videos.ts`** -- Accept + store + forward `cameraAngle`
5. **`convex/http.ts`** -- Pass `cameraAngle` through to Modal (if needed)
6. **`backend/modal_convex_processor.py`** -- Accept `cameraAngle`, apply static cluster presets, pass to `detect_rallies`
7. **`backend/rally_detection.py`** -- Accept `camera_angle` parameter, apply parameter presets
8. **`src/composables/useAdvancedAnalytics.ts`** -- Accept `cameraAngle`, apply client-side parameter presets
9. **`src/App.vue`** -- Thread `cameraAngle` through to `useAdvancedAnalytics`
10. **`src/components/AdvancedAnalytics.vue`** -- Pass `cameraAngle` prop through
