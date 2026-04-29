# OC-SORT A/B Testing Implementation Plan

> **Status: obsolete** — the feature described here was removed in v1.9. Kept as a historical design record; do not implement or reference.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-video tracker toggle (BoT-SORT vs OC-SORT) so users can A/B test player tracking quality.

**Architecture:** Mirrors the existing `cameraAngle` flow: schema field → mutation → action → Modal endpoint → worker. OC-SORT uses Roboflow's `trackers` library (Apache 2.0) with decoupled detection + external tracking. BoT-SORT path stays unchanged.

**Tech Stack:** Roboflow `trackers>=2.3.0`, `supervision>=0.26.0`, Convex, Vue 3, Modal.com

---

### Task 1: Add `trackerType` to Convex Schema

**Files:**
- Modify: `convex/schema.ts:14` (add field after `cameraAngle`)

**Step 1: Add the schema field**

In `convex/schema.ts`, add after line 14 (`cameraAngle` field):

```typescript
// Tracker algorithm for player tracking (A/B testing)
trackerType: v.optional(v.union(v.literal("botsort"), v.literal("ocsort"))),
```

**Step 2: Verify Convex schema pushes cleanly**

Run: `npx convex dev --once`
Expected: Schema update succeeds with no errors.

**Step 3: Commit**

```bash
git add convex/schema.ts
git commit -m "feat: add trackerType field to videos schema"
```

---

### Task 2: Thread `trackerType` Through Convex Mutations and Actions

**Files:**
- Modify: `convex/videos.ts:89-96` (`createVideo` mutation args + handler)
- Modify: `convex/videos.ts:358-434` (`processVideo` action — read from DB, pass to Modal)

**Step 1: Update `createVideo` mutation**

In `convex/videos.ts`, add `trackerType` to the mutation args (after `cameraAngle` on line 95):

```typescript
trackerType: v.optional(v.union(v.literal("botsort"), v.literal("ocsort"))),
```

In the handler's `ctx.db.insert` call (after line 104), add:

```typescript
trackerType: args.trackerType ?? "botsort",
```

**Step 2: Update `processVideo` action**

In `convex/videos.ts`, after line 359 (`const cameraAngle = ...`), add:

```typescript
const trackerType = video.trackerType ?? "botsort"
```

In the `JSON.stringify` body (after line 433 `cameraAngle,`), add:

```typescript
trackerType,
```

**Step 3: Verify Convex pushes cleanly**

Run: `npx convex dev --once`
Expected: No errors.

**Step 4: Commit**

```bash
git add convex/videos.ts
git commit -m "feat: thread trackerType through Convex mutations and actions"
```

---

### Task 3: Add Tracker Toggle UI in VideoUpload.vue

**Files:**
- Modify: `src/components/VideoUpload.vue:20` (add state)
- Modify: `src/components/VideoUpload.vue:189-195` (pass to mutation)
- Modify: `src/components/VideoUpload.vue:341-342` (add UI after camera selector)

**Step 1: Add state variable**

In `src/components/VideoUpload.vue`, after line 20 (`const cameraAngle = ...`), add:

```typescript
const trackerType = ref<'botsort' | 'ocsort'>('botsort')
```

**Step 2: Pass to createVideo mutation**

In the `createVideo` call (around line 189-195), add `trackerType`:

```typescript
const videoId = await client.mutation(api.videos.createVideo, {
  storageId,
  filename: selectedFile.value.name,
  size: selectedFile.value.size,
  analysisMode: analysisMode.value,
  cameraAngle: cameraAngle.value,
  trackerType: trackerType.value,
})
```

**Step 3: Add toggle UI**

After the Camera Position selector closing `</div>` (line 341), add a new selector block. Only show when analysis mode is "full" (tracker is irrelevant for rally-only mode):

```html
<!-- Player Tracker Selector (only for full analysis) -->
<div v-if="!isUploading && analysisMode === 'full'" class="mode-selector">
  <span class="mode-label">Player Tracker</span>
  <div class="mode-options">
    <button
      class="mode-option"
      :class="{ active: trackerType === 'botsort' }"
      @click="trackerType = 'botsort'"
      type="button"
    >
      <span class="mode-title">BoT-SORT</span>
      <span class="mode-desc">Default tracker with motion compensation</span>
    </button>
    <button
      class="mode-option"
      :class="{ active: trackerType === 'ocsort' }"
      @click="trackerType = 'ocsort'"
      type="button"
    >
      <span class="mode-title">OC-SORT</span>
      <span class="mode-desc">Motion-only tracker, better for uniform players</span>
    </button>
  </div>
</div>
```

No new CSS needed — reuses existing `.mode-selector`, `.mode-options`, `.mode-option` classes.

**Step 4: Verify frontend compiles**

Run: `npm run build`
Expected: No errors.

**Step 5: Commit**

```bash
git add src/components/VideoUpload.vue
git commit -m "feat: add tracker type selector in video upload UI"
```

---

### Task 4: Add Dependencies to Modal Processor Image

**Files:**
- Modify: `backend/modal_convex_processor.py:1001-1010` (add to `pip_install`)

**Step 1: Add trackers and supervision to the Modal image**

In `backend/modal_convex_processor.py`, in the `.pip_install()` call (lines 1001-1010), add two new packages after `"torchvision>=0.15.0"`:

```python
.pip_install(
    "fastapi[standard]",
    "opencv-python-headless",
    "numpy",
    "ultralytics>=8.2.0",
    "httpx",
    "python-dotenv",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "supervision>=0.26.0",
    "trackers>=2.3.0",
)
```

**Step 2: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "feat: add supervision and trackers deps to Modal image"
```

---

### Task 5: Receive `tracker_type` in Modal Endpoint and Worker

**Files:**
- Modify: `backend/modal_convex_processor.py:1031` (endpoint — extract from request)
- Modify: `backend/modal_convex_processor.py:1037-1044` (endpoint — pass to spawn)
- Modify: `backend/modal_convex_processor.py:1057-1064` (worker — add parameter)

**Step 1: Extract from request in `process_video` endpoint**

In `backend/modal_convex_processor.py`, after line 1031 (`camera_angle = ...`), add:

```python
tracker_type = request.get("trackerType", "botsort")
```

**Step 2: Pass to worker spawn**

In the `_process_video_worker.spawn()` call (lines 1037-1044), add `tracker_type`:

```python
_process_video_worker.spawn(
    video_id=video_id,
    video_url=video_url,
    callback_url=callback_url,
    manual_court_keypoints=manual_court_keypoints,
    analysis_mode=analysis_mode,
    camera_angle=camera_angle,
    tracker_type=tracker_type,
)
```

**Step 3: Add parameter to worker function**

In `_process_video_worker` signature (lines 1057-1064), add after `camera_angle`:

```python
async def _process_video_worker(
    video_id: str,
    video_url: str,
    callback_url: str,
    manual_court_keypoints: Optional[Dict] = None,
    analysis_mode: str = "full",
    camera_angle: str = "overhead",
    tracker_type: str = "botsort",
) -> Dict[str, Any]:
```

**Step 4: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "feat: thread tracker_type through Modal endpoint to worker"
```

---

### Task 6: Implement OC-SORT Tracker Initialization

**Files:**
- Modify: `backend/modal_convex_processor.py:1220-1249` (tracker setup section)

**Step 1: Branch tracker initialization based on tracker_type**

Replace the BoT-SORT config block (lines 1221-1249) with branched initialization. The BoT-SORT path stays identical. The OC-SORT path instantiates the Roboflow tracker.

```python
            # =========================================================
            # TRACKER SETUP — BoT-SORT (built-in) or OC-SORT (external)
            # =========================================================
            ocsort_tracker = None  # Only set when tracker_type == "ocsort"

            if tracker_type == "ocsort":
                import supervision as sv
                from trackers import OCSORTTracker

                ocsort_tracker = OCSORTTracker(
                    lost_track_buffer=90,               # 3s at 30fps — match BoT-SORT's track_buffer
                    frame_rate=float(fps),               # scale buffer to actual FPS
                    minimum_consecutive_frames=2,        # fast track confirmation for 2 players
                    minimum_iou_threshold=0.3,           # lenient for far-player small bboxes
                    high_conf_det_threshold=0.3,         # low threshold — far player is often 0.2-0.4
                    direction_consistency_weight=0.2,    # OCM: velocity direction matching
                    delta_t=3,                           # past frames for velocity estimation
                )
                await send_log(f"OC-SORT tracker initialized (lost_track_buffer=90, frame_rate={fps})", "info", "model")
            else:
                # Create custom BoT-SORT tracker config optimized for badminton (2 players)
                # Key changes from defaults:
                # - track_buffer: 90 (3 seconds at 30fps) instead of 30 (1 second)
                #   -> Far player's track survives longer occlusions/low-confidence gaps
                # - track_high_thresh: 0.3 instead of 0.5
                #   -> Far player (small, low-confidence) gets tracked more consistently
                # - new_track_thresh: 0.4 instead of 0.6
                #   -> Faster track creation when far player reappears
                # - match_thresh: 0.9 instead of 0.8
                #   -> More lenient matching to prevent track fragmentation
                tracker_config_path = Path("/cache/botsort_badminton.yaml")
                tracker_config_path.parent.mkdir(parents=True, exist_ok=True)
                tracker_config_content = """# BoT-SORT tracker config optimized for badminton (2 players)
tracker_type: botsort
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: 90
match_thresh: 0.9
fuse_score: True
# GMC (Global Motion Compensation) for camera movement
gmc_method: sparseOptFlow
# Proximity and appearance thresholds
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
                tracker_config_path.write_text(tracker_config_content)
                await send_log("Custom BoT-SORT tracker config created (track_buffer=90, track_high=0.3)", "info", "model")
```

**Step 2: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "feat: initialize OC-SORT or BoT-SORT based on tracker_type"
```

---

### Task 7: Branch the Frame-Loop Tracking Logic

**Files:**
- Modify: `backend/modal_convex_processor.py:1476-1486` (pose tracking call)
- Modify: `backend/modal_convex_processor.py:1656-1658` (track_ids extraction)

This is the core change. The existing code calls `pose_model.track()` which does detection + BoT-SORT in one call. For OC-SORT, we call `pose_model()` (detection only) then run the external tracker.

**Step 1: Replace the single tracking call with a branched block**

Replace the current tracking call at line 1478:

```python
pose_results = pose_model.track(
    frame,
    persist=True,
    verbose=False,
    tracker=str(tracker_config_path),
    conf=0.15,
    iou=0.5,
    imgsz=960,
)
```

With a branched block:

```python
if ocsort_tracker is not None:
    # OC-SORT: detection only, then external tracking
    pose_results = pose_model(
        frame,
        verbose=False,
        conf=0.15,
        iou=0.5,
        imgsz=960,
    )
else:
    # BoT-SORT: built-in Ultralytics tracking
    pose_results = pose_model.track(
        frame,
        persist=True,
        verbose=False,
        tracker=str(tracker_config_path),
        conf=0.15,
        iou=0.5,
        imgsz=960,
    )
```

**Step 2: Branch the track_ids extraction**

The current code (around lines 1656-1658) extracts track IDs:

```python
has_tracking = result.boxes is not None and result.boxes.is_track
track_ids = result.boxes.id.int().cpu().tolist() if has_tracking and result.boxes.id is not None else None
```

Replace with branched extraction:

```python
if ocsort_tracker is not None and result.boxes is not None and len(result.boxes) > 0:
    # OC-SORT: feed detections to external tracker
    import supervision as sv
    detections = sv.Detections.from_ultralytics(result)
    tracked = ocsort_tracker.update(detections)
    # Build track_ids list aligned to original detection order.
    # from_ultralytics preserves order; update() may reorder or drop,
    # so map back by matching bbox coordinates.
    track_ids = [None] * len(result.boxes)
    if tracked.tracker_id is not None and len(tracked) > 0:
        orig_boxes = result.boxes.xyxy.cpu().numpy()
        for t_idx in range(len(tracked)):
            t_box = tracked.xyxy[t_idx]
            # Find closest original detection by IoU
            best_iou = 0.0
            best_orig = -1
            for o_idx in range(len(orig_boxes)):
                iou = _bbox_iou(orig_boxes[o_idx], t_box)
                if iou > best_iou:
                    best_iou = iou
                    best_orig = o_idx
            if best_orig >= 0 and best_iou > 0.5:
                track_ids[best_orig] = int(tracked.tracker_id[t_idx])
    has_tracking = any(tid is not None for tid in track_ids)
    # Replace None with -1 for downstream compatibility
    track_ids = [tid if tid is not None else -1 for tid in track_ids]
else:
    has_tracking = result.boxes is not None and result.boxes.is_track
    track_ids = result.boxes.id.int().cpu().tolist() if has_tracking and result.boxes.id is not None else None
```

**Step 3: Add the IoU helper function**

Add this near the top of the file (after the existing `calculate_iou` function around line 47, or near the tracker setup section):

```python
def _bbox_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] arrays."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

**Step 4: Include tracker_type in results**

Find where `camera_angle` is included in the results dict (around line 2400) and add `tracker_type` next to it:

```python
"tracker_type": tracker_type,
```

**Step 5: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "feat: branch frame-loop tracking for OC-SORT vs BoT-SORT"
```

---

### Task 8: Deploy and Verify

**Step 1: Deploy Modal processor**

Run: `modal deploy backend/modal_convex_processor.py`
Expected: Deploys successfully. Watch for dependency installation in the build log (supervision, trackers).

**Step 2: Deploy Convex schema**

Run: `npx convex deploy`
Expected: Schema and functions push successfully.

**Step 3: Test BoT-SORT path (regression)**

Upload a video with the default "BoT-SORT" tracker selected. Verify:
- Processing completes without errors
- Skeleton data has consistent player IDs
- No behavioral change from before

**Step 4: Test OC-SORT path**

Upload the same video with "OC-SORT" tracker selected. Verify:
- Processing completes without errors
- Skeleton data has track IDs assigned
- Player tracking quality can be visually compared with the BoT-SORT result

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: OC-SORT A/B testing complete — per-video tracker toggle"
```
