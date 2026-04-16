# Camera Angle Rally Detection Presets — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a camera position selector ("Overhead" / "Corner / Side") that tunes rally detection parameters on both backend and client-side to reduce false rallies from corner-angle amateur footage.

**Architecture:** New `cameraAngle` field flows through the same path as `analysisMode` — from `VideoUpload.vue` through Convex DB to Modal backend and also to the client-side analytics composable. Each layer reads the value and applies parameter presets.

**Tech Stack:** Vue 3 + TypeScript (frontend), Convex (DB/API), Python (backend rally detection)

---

### Task 1: Add `cameraAngle` to types and schema

**Files:**
- Modify: `src/types/analysis.ts:294-300` (UploadResponse)
- Modify: `src/types/analysis.ts:278-292` (AnalysisResult)
- Modify: `convex/schema.ts:11-12` (videos table)

**Step 1: Add `cameraAngle` to TypeScript types**

In `src/types/analysis.ts`, add to `UploadResponse`:
```typescript
export interface UploadResponse {
  video_id: string
  filename: string
  size: number
  status: string
  analysisMode: 'rally_only' | 'full'
  cameraAngle?: 'overhead' | 'corner'
}
```

Add to `AnalysisResult` (so client-side analytics can read it from results):
```typescript
export interface AnalysisResult {
  // ... existing fields ...
  rallies?: BackendRally[] | null
  rally_stats?: RallyStats | null
  camera_angle?: 'overhead' | 'corner'
}
```

**Step 2: Add `cameraAngle` to Convex schema**

In `convex/schema.ts`, add after the `analysisMode` field (line 12):
```typescript
    // Camera position (affects rally detection parameters)
    cameraAngle: v.optional(v.union(v.literal("overhead"), v.literal("corner"))),
```

**Step 3: Commit**

```bash
git add src/types/analysis.ts convex/schema.ts
git commit -m "feat: add cameraAngle to types and schema"
```

---

### Task 2: Add camera position selector to VideoUpload.vue

**Files:**
- Modify: `src/components/VideoUpload.vue:19` (add ref)
- Modify: `src/components/VideoUpload.vue:188-203` (pass in mutation + emit)
- Modify: `src/components/VideoUpload.vue:291-313` (add UI after analysis mode selector)

**Step 1: Add `cameraAngle` ref**

After line 19 (`const analysisMode = ref<...>`), add:
```typescript
const cameraAngle = ref<'overhead' | 'corner'>('overhead')
```

**Step 2: Pass `cameraAngle` to Convex mutation and emit**

In `createVideo` call (~line 188-193), add `cameraAngle`:
```typescript
    const videoId = await client.mutation(api.videos.createVideo, {
      storageId,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
      analysisMode: analysisMode.value,
      cameraAngle: cameraAngle.value,
    })
```

In the `emit('uploaded', ...)` call (~line 198-204), add `cameraAngle`:
```typescript
    emit('uploaded', {
      video_id: videoId,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
      status: 'uploaded',
      analysisMode: analysisMode.value,
      cameraAngle: cameraAngle.value,
    })
```

**Step 3: Add camera position selector UI**

After the closing `</div>` of the analysis mode selector (after line 313), add a second selector block using identical styling:
```html
      <!-- Camera Position Selector -->
      <div v-if="!isUploading" class="mode-selector">
        <span class="mode-label">Camera Position</span>
        <div class="mode-options">
          <button
            class="mode-option"
            :class="{ active: cameraAngle === 'overhead' }"
            @click="cameraAngle = 'overhead'"
            type="button"
          >
            <span class="mode-title">Overhead</span>
            <span class="mode-desc">Camera centered above the court</span>
          </button>
          <button
            class="mode-option"
            :class="{ active: cameraAngle === 'corner' }"
            @click="cameraAngle = 'corner'"
            type="button"
          >
            <span class="mode-title">Corner / Side</span>
            <span class="mode-desc">Camera at court level from corner or side</span>
          </button>
        </div>
      </div>
```

**Step 4: Commit**

```bash
git add src/components/VideoUpload.vue
git commit -m "feat: add camera position selector UI"
```

---

### Task 3: Thread `cameraAngle` through Convex (DB + API)

**Files:**
- Modify: `convex/videos.ts:89-94` (createVideo args)
- Modify: `convex/videos.ts:97-103` (createVideo handler)
- Modify: `convex/videos.ts:355-429` (processVideo — read + forward to Modal)

**Step 1: Accept `cameraAngle` in `createVideo` mutation**

Add to args (after `analysisMode` at line 94):
```typescript
    cameraAngle: v.optional(v.union(v.literal("overhead"), v.literal("corner"))),
```

Add to the `ctx.db.insert` call (after `analysisMode` at line 102):
```typescript
      cameraAngle: args.cameraAngle ?? "overhead",
```

**Step 2: Forward `cameraAngle` in `processVideo` action**

After line 356 (`const analysisMode = video.analysisMode ?? "full"`), add:
```typescript
    const cameraAngle = video.cameraAngle ?? "overhead"
```

In the `body: JSON.stringify(...)` call (~line 423-430), add `cameraAngle`:
```typescript
        body: JSON.stringify({
          videoId,
          videoUrl: video.videoUrl,
          callbackUrl: convexSiteUrl,
          manualCourtKeypoints: hasCourtKeypoints ? keypointsData.keypoints : null,
          analysisMode,
          cameraAngle,
        }),
```

**Step 3: Commit**

```bash
git add convex/videos.ts
git commit -m "feat: thread cameraAngle through Convex mutations and actions"
```

---

### Task 4: Update backend rally_detection.py with parameter presets

**Files:**
- Modify: `backend/rally_detection.py:21-34` (detect_rallies signature)
- Modify: `backend/rally_detection.py:55-68` (apply presets)
- Modify: `backend/rally_detection.py:86-101` (shot detection with presets)
- Modify: `backend/rally_detection.py:128-150` (dot product threshold)
- Modify: `backend/rally_detection.py:157-196` (min shots per rally)

**Step 1: Add `camera_angle` parameter to `detect_rallies`**

Update the function signature to accept and apply presets:
```python
def detect_rallies(
    shuttle_positions: Dict[int, Dict],
    fps: float,
    total_frames: int,
    player_positions: Optional[Dict[int, List[Dict]]] = None,
    pose_data: Optional[Dict[int, List[Dict]]] = None,
    speed_data: Optional[Dict[int, List[float]]] = None,
    min_rally_duration_s: float = 2.0,
    min_gap_duration_s: float = 3.0,
    zero_gradient_window: int = 0,
    zero_gradient_ratio: float = 0.80,
    frame_width: int = 0,
    frame_height: int = 0,
    camera_angle: str = "overhead",
) -> List[Dict]:
```

**Step 2: Apply camera-angle presets at the top of `detect_rallies`**

After the docstring and the empty-check (line 52), add preset logic:
```python
    # Camera-angle presets: corner angles produce more TrackNet jitter,
    # requiring stricter thresholds to avoid false rallies.
    if camera_angle == "corner":
        min_rally_duration_s = max(min_rally_duration_s, 3.0)
        min_gap_duration_s = max(min_gap_duration_s, 4.0)
        min_shot_gap_s = 0.8
        min_speed_sq = 30.0 * 30.0
        min_shots = 3
        dot_threshold = -0.25  # cos(~105°), reject slight wobbles
    else:
        min_shot_gap_s = 0.6
        min_speed_sq = 15.0 * 15.0
        min_shots = 2
        dot_threshold = 0.0  # any reversal (>90°)

    min_shot_gap_frames = max(3, int(fps * min_shot_gap_s))
```

Remove the old hardcoded `min_shot_gap_s` and `min_shot_gap_frames` lines (lines 55-56).

**Step 3: Pass presets to `_detect_shots`**

Update the call and signature of `_detect_shots` to accept `min_speed_sq` and `dot_threshold`:
```python
    shots = _detect_shots(
        shuttle_positions, total_frames, fps, min_shot_gap_frames,
        min_speed_sq=min_speed_sq, dot_threshold=dot_threshold,
    )
```

Update `_detect_shots` signature:
```python
def _detect_shots(
    positions: Dict[int, Dict],
    total_frames: int,
    fps: float,
    min_gap_frames: int,
    min_speed_sq: float = 225.0,
    dot_threshold: float = 0.0,
) -> List[Dict]:
```

Replace the hardcoded `MIN_SPEED_SQ = 15.0 * 15.0` (line 128) with the parameter:
```python
    MIN_SPEED_SQ = min_speed_sq
```

Update the dot product check (line 150) to use `dot_threshold`:
```python
        # For overhead: dot < 0 (any reversal >90°)
        # For corner: dot < -0.25*|v1|*|v2| (reversal >~105°, rejects jitter wobbles)
        if dot_threshold < 0:
            threshold = dot_threshold * math.sqrt(speed1_sq * speed2_sq)
        else:
            threshold = 0
        if dot < threshold and (f1 - last_shot_frame) >= min_gap_frames:
```

**Step 4: Pass `min_shots` to `_group_shots_into_rallies`**

Update the call:
```python
    rallies = _group_shots_into_rallies(
        shots, rally_gap_frames, min_rally_frames, fps,
        min_shots=min_shots,
    )
```

Update `_group_shots_into_rallies` signature and the check:
```python
def _group_shots_into_rallies(
    shots: List[Dict],
    rally_gap_frames: int,
    min_rally_frames: int,
    fps: float,
    min_shots: int = 2,
) -> List[Dict]:
```

Replace the hardcoded `if len(rally_shots) >= 2:` (line 184) with:
```python
            if len(rally_shots) >= min_shots:
```

Also update the early return check (line 169):
```python
    if len(shots) < min_shots:
        return []
```

**Step 5: Update STRIDE for corner angle**

In `_detect_shots`, update the STRIDE calculation to accept the camera angle or compute it from the parameters. Simplest approach — add a `stride_s` parameter:

Update `_detect_shots` signature to also accept `stride_s`:
```python
def _detect_shots(
    positions: Dict[int, Dict],
    total_frames: int,
    fps: float,
    min_gap_frames: int,
    min_speed_sq: float = 225.0,
    dot_threshold: float = 0.0,
    stride_s: float = 0.3,
) -> List[Dict]:
```

Replace the hardcoded STRIDE (line 116):
```python
    STRIDE = max(3, int(fps * stride_s))
```

Pass from `detect_rallies`:
```python
    if camera_angle == "corner":
        # ... existing preset lines ...
        stride_s = 0.5
    else:
        # ... existing preset lines ...
        stride_s = 0.3

    shots = _detect_shots(
        shuttle_positions, total_frames, fps, min_shot_gap_frames,
        min_speed_sq=min_speed_sq, dot_threshold=dot_threshold,
        stride_s=stride_s,
    )
```

**Step 6: Commit**

```bash
git add backend/rally_detection.py
git commit -m "feat: camera-angle parameter presets in rally detection"
```

---

### Task 5: Pass `cameraAngle` through Modal processor

**Files:**
- Modify: `backend/modal_convex_processor.py:1030` (parse from request)
- Modify: `backend/modal_convex_processor.py:1036-1041` (pass to worker)
- Modify: `backend/modal_convex_processor.py:1055-1060` (worker signature)
- Modify: `backend/modal_convex_processor.py:2275-2285` (static cluster presets)
- Modify: `backend/modal_convex_processor.py:2353-2357` (pass to detect_rallies)

**Step 1: Parse `cameraAngle` from request**

After line 1030 (`analysis_mode = request.get("analysisMode", "full")`), add:
```python
    camera_angle = request.get("cameraAngle", "overhead")
```

Pass it to the spawn call (~line 1036-1041):
```python
    _process_video_worker.spawn(
        video_id=video_id,
        video_url=video_url,
        callback_url=callback_url,
        manual_court_keypoints=manual_court_keypoints,
        analysis_mode=analysis_mode,
        camera_angle=camera_angle,
    )
```

**Step 2: Add to worker signature**

Update `_process_video_worker` signature (~line 1055-1060):
```python
async def _process_video_worker(
    video_id: str,
    video_url: str,
    callback_url: str,
    manual_court_keypoints: Optional[Dict] = None,
    analysis_mode: str = "full",
    camera_angle: str = "overhead",
) -> Dict[str, Any]:
```

**Step 3: Apply stricter static cluster filtering for corner angles**

In the rally detection filtering section (~line 2276-2280), after setting `_RALLY_STATIC_DIST` and `_RALLY_MIN_MOVE`, add corner adjustments:
```python
            # Corner angles: more aggressive static filtering (no court polygon available)
            if camera_angle == "corner":
                _RALLY_STATIC_DIST = int(_RALLY_STATIC_DIST * 1.5)
                _RALLY_STATIC_COUNT = 2
            else:
                _RALLY_STATIC_COUNT = 3
```

Then update the static cluster pruning line (~line 2327) to use `_RALLY_STATIC_COUNT` instead of hardcoded `3`:
```python
                _rally_static_clusters = [c for c in _rally_static_clusters if c["count"] >= _RALLY_STATIC_COUNT]
```

**Step 4: Pass `camera_angle` to `detect_rallies`**

Update the call (~line 2353-2357):
```python
                detected_rallies = detect_rallies(
                    rally_shuttle_positions,
                    fps=fps,
                    total_frames=total_frames,
                    camera_angle=camera_angle,
                )
```

**Step 5: Include `camera_angle` in results data**

In the `results_data` dict (~line 2381), add:
```python
            "camera_angle": camera_angle,
```

**Step 6: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "feat: pass cameraAngle through Modal processor to rally detection"
```

---

### Task 6: Update client-side analytics with camera-angle presets

**Files:**
- Modify: `src/composables/useAdvancedAnalytics.ts:74-77` (add cameraAngle parameter)
- Modify: `src/composables/useAdvancedAnalytics.ts:113-117` (shot detection presets)
- Modify: `src/composables/useAdvancedAnalytics.ts:188-198` (MIN_SPEED_SQ + dot product)
- Modify: `src/composables/useAdvancedAnalytics.ts:293-320` (rally grouping presets)

**Step 1: Add `cameraAngle` parameter to composable**

Update the function signature:
```typescript
export function useAdvancedAnalytics(
  analysisResult: Ref<AnalysisResult | null>,
  currentFrame: Ref<number>,
  courtKeypoints?: Ref<number[][] | null>,
  cameraAngle?: Ref<'overhead' | 'corner'>
) {
```

**Step 2: Apply presets in `detectAllShots`**

Update `detectAllShots` (~line 115-117):
```typescript
  function detectAllShots(frames: SkeletonFrame[], fps: number): RallyShot[] {
    const isCorner = cameraAngle?.value === 'corner'
    const MIN_GAP_S = isCorner ? 0.8 : 0.6
    const MIN_GAP_FRAMES = Math.max(3, Math.floor(fps * MIN_GAP_S))
```

**Step 3: Apply presets in `detectShotsFromShuttle`**

Update MIN_SPEED_SQ and dot product check (~line 193-198):
```typescript
      const MIN_SPEED_SQ = isCorner ? 50 * 50 : 30 * 30
      if (speed1sq < MIN_SPEED_SQ && speed2sq < MIN_SPEED_SQ) continue

      const dot = vx1 * vx2 + vy1 * vy2

      // Corner: require >~105° reversal to reject jitter wobbles
      const dotThreshold = isCorner
        ? -0.25 * Math.sqrt(speed1sq * speed2sq)
        : 0
      if (dot >= dotThreshold || (p1.frame - lastShotFrame) < minGapFrames) continue
```

Note: `isCorner` needs to be accessible inside `detectShotsFromShuttle`. Since it's a nested function inside the composable, capture it at the `detectAllShots` level and pass it down, OR read `cameraAngle?.value` directly since it's in closure scope. The simplest approach: read `cameraAngle?.value` at the start of `detectShotsFromShuttle`:
```typescript
  function detectShotsFromShuttle(
    frames: SkeletonFrame[],
    frameMap: Map<number, SkeletonFrame>,
    minGapFrames: number
  ): RallyShot[] {
    const isCorner = cameraAngle?.value === 'corner'
    // ... rest of function
```

**Step 4: Apply presets in rally grouping**

Update the rally grouping section (~line 293-320):
```typescript
  const rallies = computed(() => {
    // ... existing result/frames checks ...
    const shots = detectAllShots(frames, fps)
    const isCorner = cameraAngle?.value === 'corner'
    const MIN_SHOTS = isCorner ? 3 : 2
    if (shots.length < MIN_SHOTS) return []

    const RALLY_GAP_SECONDS = isCorner ? 4.0 : 3.1
    const MIN_RALLY_DURATION_S = isCorner ? 3.0 : 2.0

    // ... isRealGameplay function stays the same ...

    // Update the rally shot count check:
    // Change: if (rallyShots.length >= 2)
    // To:     if (rallyShots.length >= MIN_SHOTS)
```

**Step 5: Commit**

```bash
git add src/composables/useAdvancedAnalytics.ts
git commit -m "feat: camera-angle presets in client-side rally detection"
```

---

### Task 7: Thread `cameraAngle` through App.vue and components

**Files:**
- Modify: `src/App.vue:56` (add cameraAngle ref)
- Modify: `src/App.vue:130-133` (pass to useAdvancedAnalytics)
- Modify: `src/App.vue:564-570` (handleUploadComplete)
- Modify: `src/App.vue:1633-1635` (pass to AdvancedAnalytics component)
- Modify: `src/components/AdvancedAnalytics.vue:25-28` (accept + forward prop)

**Step 1: Add `cameraAngle` ref in App.vue**

After line 56 (`const analysisMode = ref<...>`), add:
```typescript
const cameraAngle = ref<'overhead' | 'corner'>(restored.video?.cameraAngle ?? 'overhead')
```

**Step 2: Pass to `useAdvancedAnalytics`**

Update the call (~line 130-133):
```typescript
const { rallies: detectedRallies, backendRallies, rallySource, rallySpeedStats } = useAdvancedAnalytics(
  computed(() => analysisResult.value),
  currentFrame,
  undefined,
  cameraAngle,
)
```

**Step 3: Set `cameraAngle` on upload complete**

In `handleUploadComplete` (~line 565-570), add:
```typescript
function handleUploadComplete(response: UploadResponse) {
  uploadedVideo.value = response
  analysisMode.value = response.analysisMode
  cameraAngle.value = response.cameraAngle ?? 'overhead'
  currentState.value = response.analysisMode === 'rally_only' ? 'analyzing' : 'court-setup'
  errorMessage.value = ''
}
```

**Step 4: Also read `cameraAngle` from analysis result when available**

In `handleAnalysisComplete` (~line 613), after setting the result, also update from server result:
```typescript
async function handleAnalysisComplete(result: AnalysisResult) {
  analysisResult.value = result
  if (result.camera_angle) {
    cameraAngle.value = result.camera_angle
  }
  // ... rest of function
}
```

**Step 5: Pass to AdvancedAnalytics component**

In the AdvancedAnalytics template usage (~line 1633), add the prop:
```html
              <AdvancedAnalytics
                :result="analysisResult"
                :current-frame="currentFrame"
                :court-keypoints="courtCornersForMiniCourt"
                :camera-angle="cameraAngle"
              />
```

Update `AdvancedAnalytics.vue` to accept and forward the prop. In the props definition, add `cameraAngle`:
```typescript
const props = defineProps<{
  result: AnalysisResult
  currentFrame: number
  courtKeypoints?: number[][] | null
  cameraAngle?: 'overhead' | 'corner'
}>()
```

Update its `useAdvancedAnalytics` call to pass `cameraAngle`:
```typescript
} = useAdvancedAnalytics(
  computed(() => props.result),
  computed(() => props.currentFrame),
  computed(() => props.courtKeypoints ?? null),
  computed(() => props.cameraAngle ?? 'overhead'),
)
```

**Step 6: Commit**

```bash
git add src/App.vue src/components/AdvancedAnalytics.vue
git commit -m "feat: thread cameraAngle through App and AdvancedAnalytics"
```

---

### Task 8: Verify build and do a final review

**Step 1: Run TypeScript check**

```bash
npx vue-tsc --noEmit
```

Expected: No type errors.

**Step 2: Run Vite build**

```bash
npm run build
```

Expected: Build succeeds.

**Step 3: Run Convex codegen**

```bash
npx convex dev --once
```

Expected: Schema and API types regenerate without errors.

**Step 4: Verify Python syntax**

```bash
python -c "import ast; ast.parse(open('backend/rally_detection.py').read()); print('OK')"
```

Expected: `OK`

**Step 5: Final commit if any fixes needed, then squash or leave as-is**
