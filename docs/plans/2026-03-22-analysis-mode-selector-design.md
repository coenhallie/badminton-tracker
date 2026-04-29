# Analysis Mode Selector

> **Status: obsolete** — the feature described here was removed in v1.9. Kept as a historical design record; do not implement or reference.

## Summary

Add a mode selector on the upload page: **Rally Separation** (fast, TrackNet + rally detection only) vs **Full Analysis** (current pipeline). The `analysisMode` parameter flows through the full stack. Rally-only mode skips YOLO model loading and the per-frame analysis loop.

## Frontend — Mode Selector

After file selection in `VideoUpload.vue`, show a toggle/selector with two options:

- **Rally Separation** (`"rally_only"`) — "Detect rally boundaries only (faster)"
- **Full Analysis** (`"full"`) — "Player tracking, poses, speed, heatmaps + rallies"

The selected mode is emitted with the upload response. `App.vue` passes it through to `AnalysisProgress.vue` and stores it on the Convex video record.

## Data Flow — Convex & Modal

- `convex/schema.ts`: Add optional `analysisMode` string field to `videos` table.
- `convex/videos.ts`: `createVideo` accepts `analysisMode`, `processVideo` passes it to Modal.
- `modal_convex_processor.py`: `process_video` and `_process_video_worker` accept `analysisMode`.

### Rally-only pipeline (skip list)

When `analysisMode == "rally_only"`, the worker:
1. Downloads video (same as full)
2. Loads TrackNet only (skips YOLO pose + detection models)
3. Runs TrackNet shuttle tracking pass (same as full)
4. **Skips** the entire frame-by-frame YOLO loop
5. Runs rally detection on TrackNet positions
6. Uploads results with empty `skeleton_data: []`, `players: []`

## Results View — Conditional UI

When `analysisMode === "rally_only"`, the results page shows only:
- Video player (no overlays — no skeleton data exists)
- Rally timeline

Hidden: MiniCourt, SpeedGraph, ShotSpeedList, AdvancedAnalytics, overlay toggles, heatmap toggle.

Guard with `v-if="analysisMode !== 'rally_only'"` on analytics components.
