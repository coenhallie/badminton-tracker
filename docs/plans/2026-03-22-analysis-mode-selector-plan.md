# Analysis Mode Selector — Implementation Plan

> **Status: obsolete** — the feature described here was removed in v1.9. Kept as a historical design record; do not implement or reference.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a mode selector so users can choose between fast rally-only analysis and full analytics.

**Architecture:** An `analysisMode` field (`"rally_only"` | `"full"`) flows from frontend selector through Convex to Modal. The Modal worker conditionally skips YOLO model loading and the per-frame loop for rally-only mode. The results view conditionally hides analytics components.

**Tech Stack:** Vue 3, TypeScript, Convex (schema + actions), Python/Modal (backend pipeline)

---

### Task 1: Add `analysisMode` to Convex schema and mutations

**Files:**
- Modify: `convex/schema.ts:5-17` (videos table definition)
- Modify: `convex/videos.ts:89-114` (createVideo mutation)

**Step 1: Add field to schema**

In `convex/schema.ts`, add `analysisMode` field to the `videos` table, after line 17 (after the `status` field):

```typescript
// Analysis mode
analysisMode: v.optional(v.union(v.literal("rally_only"), v.literal("full"))),
```

**Step 2: Accept `analysisMode` in `createVideo` mutation**

In `convex/videos.ts:89-114`, add `analysisMode` to args and insert:

```typescript
export const createVideo = mutation({
  args: {
    storageId: v.id("_storage"),
    filename: v.string(),
    size: v.number(),
    analysisMode: v.optional(v.union(v.literal("rally_only"), v.literal("full"))),
  },
  handler: async (ctx, args) => {
    const videoId = await ctx.db.insert("videos", {
      storageId: args.storageId,
      filename: args.filename,
      size: args.size,
      status: "uploaded",
      analysisMode: args.analysisMode ?? "full",
      createdAt: Date.now(),
    })
    // ... rest unchanged
```

**Step 3: Pass `analysisMode` from `processVideo` to Modal**

In `convex/videos.ts:339-458`, read `analysisMode` from the video record and include it in the Modal request body.

At line ~343, after fetching `video`, extract the mode:
```typescript
const analysisMode = video.analysisMode ?? "full"
```

At line ~418, add `analysisMode` to the JSON body:
```typescript
body: JSON.stringify({
  videoId,
  videoUrl: video.videoUrl,
  callbackUrl: convexSiteUrl,
  manualCourtKeypoints: hasCourtKeypoints ? keypointsData.keypoints : null,
  analysisMode,
}),
```

**Step 4: Verify Convex dev server accepts the schema change**

Run: `npx convex dev` — check for schema errors.

**Step 5: Commit**

```
feat: add analysisMode field to Convex schema and mutations
```

---

### Task 2: Add mode selector UI to VideoUpload

**Files:**
- Modify: `src/components/VideoUpload.vue` (script + template + styles)
- Modify: `src/types/analysis.ts:295-300` (UploadResponse type)

**Step 1: Add `analysisMode` to UploadResponse type**

In `src/types/analysis.ts`, update `UploadResponse`:

```typescript
export interface UploadResponse {
  video_id: string
  filename: string
  size: number
  status: string
  analysisMode: 'rally_only' | 'full'
}
```

**Step 2: Add mode state and pass it through upload flow**

In `src/components/VideoUpload.vue` `<script setup>`, add a ref after `selectedFile`:

```typescript
const analysisMode = ref<'rally_only' | 'full'>('full')
```

In the `createVideo` mutation call (~line 187-190), add `analysisMode`:

```typescript
const videoId = await client.mutation(api.videos.createVideo, {
  storageId,
  filename: selectedFile.value.name,
  size: selectedFile.value.size,
  analysisMode: analysisMode.value,
})
```

In the `emit('uploaded', ...)` call (~line 196-201), add `analysisMode`:

```typescript
emit('uploaded', {
  video_id: videoId,
  filename: selectedFile.value.name,
  size: selectedFile.value.size,
  status: 'uploaded',
  analysisMode: analysisMode.value,
})
```

**Step 3: Add selector UI to template**

In the template, inside the `.file-preview` div, add the mode selector before the upload button (before line 287):

```html
<!-- Analysis Mode Selector -->
<div class="mode-selector">
  <span class="mode-label">Analysis Mode</span>
  <div class="mode-options">
    <button
      class="mode-option"
      :class="{ active: analysisMode === 'rally_only' }"
      @click="analysisMode = 'rally_only'"
    >
      <span class="mode-title">Rally Separation</span>
      <span class="mode-desc">Detect rally boundaries only (faster)</span>
    </button>
    <button
      class="mode-option"
      :class="{ active: analysisMode === 'full' }"
      @click="analysisMode = 'full'"
    >
      <span class="mode-title">Full Analysis</span>
      <span class="mode-desc">Player tracking, poses, speed + rallies</span>
    </button>
  </div>
</div>
```

**Step 4: Add styles**

Add to the `<style scoped>` section:

```css
.mode-selector {
  margin-bottom: 20px;
}

.mode-label {
  display: block;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  font-weight: 500;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.mode-options {
  display: flex;
  gap: 8px;
}

.mode-option {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 16px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-border-secondary);
  border-radius: 0;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s ease;
}

.mode-option:hover {
  border-color: var(--color-text-tertiary);
}

.mode-option.active {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
}

.mode-title {
  color: var(--color-text-heading);
  font-weight: 600;
  font-size: 0.9rem;
}

.mode-desc {
  color: var(--color-text-tertiary);
  font-size: 0.75rem;
}

.mode-option.active .mode-title {
  color: var(--color-accent);
}
```

**Step 5: Commit**

```
feat: add analysis mode selector to VideoUpload component
```

---

### Task 3: Thread `analysisMode` through App.vue and AnalysisProgress

**Files:**
- Modify: `src/App.vue:54-56,564-568,1270-1276` (state + handler + template)
- Modify: `src/components/AnalysisProgress.vue:1-17,235-248` (props + processVideo call)

**Step 1: Store mode in App.vue state**

In `src/App.vue`, add a ref after `uploadedVideo` (~line 55):

```typescript
const analysisMode = ref<'rally_only' | 'full'>('full')
```

Update `handleUploadComplete` (~line 564):

```typescript
function handleUploadComplete(response: UploadResponse) {
  uploadedVideo.value = response
  analysisMode.value = response.analysisMode
  currentState.value = 'court-setup'
  errorMessage.value = ''
}
```

Update session save/restore to include `analysisMode` — in `loadSessionState` return the mode, in `saveSessionState` include it.

**Step 2: Pass mode to AnalysisProgress**

In the template (~line 1270), add the prop:

```html
<AnalysisProgress
  :video-id="uploadedVideo.video_id"
  :filename="uploadedVideo.filename"
  :analysis-mode="analysisMode"
  @complete="handleAnalysisComplete"
  @error="handleAnalysisError"
  @cancel="handleAnalysisCancel"
/>
```

**Step 3: Accept prop in AnalysisProgress and pass to processVideo**

In `src/components/AnalysisProgress.vue`, update the props definition (~line 8):

```typescript
const props = defineProps<{
  videoId: string
  filename: string
  analysisMode: 'rally_only' | 'full'
}>()
```

The `processVideo` Convex action reads `analysisMode` from the video record (set during `createVideo`), so no changes needed to the action call itself. The mode is already stored on the video document.

**Step 4: Commit**

```
feat: thread analysisMode through App.vue to AnalysisProgress
```

---

### Task 4: Backend — rally-only mode in Modal worker

**Files:**
- Modify: `backend/modal_convex_processor.py:1021-1042` (process_video endpoint)
- Modify: `backend/modal_convex_processor.py:1053-1058` (_process_video_worker signature)
- Modify: `backend/modal_convex_processor.py:1199-1500` (skip YOLO + frame loop)
- Modify: `backend/modal_convex_processor.py:2366-2460` (results assembly)

**Step 1: Pass `analysisMode` through the endpoint**

In `process_video` (~line 1021), extract and forward the mode:

```python
analysis_mode = request.get("analysisMode", "full")
```

Add it to `_process_video_worker.spawn(...)`:

```python
_process_video_worker.spawn(
    video_id=video_id,
    video_url=video_url,
    callback_url=callback_url,
    manual_court_keypoints=manual_court_keypoints,
    analysis_mode=analysis_mode,
)
```

**Step 2: Accept in worker signature**

Update `_process_video_worker` (~line 1053):

```python
async def _process_video_worker(
    video_id: str,
    video_url: str,
    callback_url: str,
    manual_court_keypoints: Optional[Dict] = None,
    analysis_mode: str = "full",
) -> Dict[str, Any]:
```

**Step 3: Conditionally skip YOLO and frame loop**

After TrackNet pass completes (~line 1311), add a branch:

```python
if analysis_mode == "rally_only":
    await send_log("Rally-only mode — skipping player analysis", "info", "processing")

    # Jump directly to rally detection (no skeleton data, no players)
    skeleton_frames = []
    players_summary = []
    player_positions = {0: [], 1: []}
    player_distances = {0: 0.0, 1: 0.0}
    processed_count = 0
else:
    # ... existing YOLO loading + frame-by-frame loop code ...
```

This means wrapping the existing code from model loading (~line 1199) through end of frame loop/player summary (~line 2210) in an `else` block.

**Step 4: Handle rally detection for rally-only mode**

The rally detection section (~line 2212-2363) should run in both modes. For rally-only mode, it will use TrackNet data (which was collected). The `player_wrist_data` will be empty (no skeleton frames), which is fine — `detect_rallies` accepts `player_positions=None`.

For rally-only mode, build rally data from TrackNet without the wrist filtering:

```python
if analysis_mode == "rally_only":
    # No skeleton frames → no wrist data for carried-shuttle detection
    player_wrist_data = {}
```

**Step 5: Adjust results assembly for rally-only mode**

The results dict (~line 2366-2383) already works — `skeleton_frames` will be `[]` and `players_summary` will be `[]`.

**Step 6: Commit**

```
feat: add rally-only mode to Modal pipeline (skip YOLO + frame loop)
```

---

### Task 5: Conditional results view in App.vue

**Files:**
- Modify: `src/App.vue:1280-1617` (results template section)

**Step 1: Hide settings panel in rally-only mode**

Wrap the settings toggle button and settings panel (~lines 1306-1498) in:

```html
<template v-if="analysisMode !== 'rally_only'">
  <!-- settings toggle button + settings panel -->
</template>
```

Also hide the export button in rally-only mode (no overlays to export).

**Step 2: Hide analytics components**

Wrap the MiniCourt section (~lines 1528-1548) in `v-if="analysisMode !== 'rally_only'"`.

Wrap the SpeedGraph section (~lines 1570-1584) in `v-if="analysisMode !== 'rally_only'"`.

Wrap the ShotSpeedList section (~lines 1587-1599) in `v-if="analysisMode !== 'rally_only'"`.

Wrap the ResultsDashboard section (~lines 1601-1607) in `v-if="analysisMode !== 'rally_only'"`.

Wrap the AdvancedAnalytics section (~lines 1609-1616) in `v-if="analysisMode !== 'rally_only'"`.

**Step 3: Simplify VideoPlayer props for rally-only mode**

In the VideoPlayer component, conditionally disable overlays:

```html
<VideoPlayer
  ref="videoPlayerRef"
  :video-url="videoUrl"
  :skeleton-data="analysisMode !== 'rally_only' ? analysisResult.skeleton_data : []"
  :show-skeleton="analysisMode !== 'rally_only' && showSkeleton"
  :show-bounding-boxes="analysisMode !== 'rally_only' && showBoundingBoxes"
  ...
/>
```

**Step 4: Rally timeline stays visible for both modes**

The RallyTimeline (~lines 1551-1566) is already outside the guarded sections. No changes needed — it renders when `detectedRallies.length > 0`.

**Step 5: Commit**

```
feat: conditionally hide analytics components in rally-only mode
```

---

### Task 6: Skip court setup for rally-only mode

**Files:**
- Modify: `src/App.vue:564-568` (handleUploadComplete)

**Step 1: Skip court-setup state when rally-only**

Court keypoints are only useful for player filtering and speed calibration — not needed for rally-only. Update `handleUploadComplete`:

```typescript
function handleUploadComplete(response: UploadResponse) {
  uploadedVideo.value = response
  analysisMode.value = response.analysisMode
  // Skip court setup for rally-only mode (no player analysis)
  currentState.value = response.analysisMode === 'rally_only' ? 'analyzing' : 'court-setup'
  errorMessage.value = ''
}
```

**Step 2: Commit**

```
feat: skip court setup for rally-only mode
```

---

### Task 7: End-to-end test

**Step 1: Run the frontend dev server**

Run: `npm run dev`

**Step 2: Manual verification checklist**

1. Upload page shows mode selector after file selection
2. Rally Separation mode selected → skips court setup → goes to analyzing
3. Full Analysis mode selected → goes to court setup as before
4. Rally-only processing completes (check Modal logs for skipped YOLO)
5. Rally-only results show only video player + rally timeline
6. Full analysis results show all components as before

**Step 3: Commit final adjustments if needed**
