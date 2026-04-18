# Synthetic Court View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Video/Court view-mode toggle in the results dashboard. In Court mode, the video element is visually hidden and replaced by a synthetic badminton court drawn from the manual keypoints (camera-perspective, via inverse homography), with the existing skeleton overlay and a shuttle-trail layer drawn on top.

**Architecture:** Pure client-side presentation change. `App.vue` owns the new `viewMode` state and a mode selector UI; passes `viewMode`, the manual keypoints, skeleton data, current frame, and fps into `VideoPlayer.vue`. `VideoPlayer.vue` keeps the `<video>` element mounted (for timing + audio) but hides it with `opacity: 0` when in Court mode, and renders a new `SyntheticCourtView.vue` sibling canvas which draws the court lines + shuttle trail. `PoseOverlay.vue` is unchanged — skeletons stay in pixel space in both modes. No backend, schema, or Convex changes.

**Tech Stack:** Vue 3 `<script setup>` + TypeScript, Vite, Canvas 2D API, existing `@/utils/homography` utilities.

**Design doc:** `docs/plans/2026-04-18-synthetic-court-view-design.md`

---

### Verification primitives used throughout the plan

No unit-test framework is configured in this project (`package.json` has only `dev`, `build`, `type-check`). Each task uses:

- **Type check**: `npm run type-check` — must pass with zero errors.
- **Build check**: `npm run build` — must produce no new warnings.
- **Manual browser check**: `npm run dev` + open `http://localhost:5173`, navigate to a completed video with manual keypoints set. Steps are called out explicitly per task.

---

### Task 1: Add `viewMode` state + mode selector UI in `App.vue`

**Goal:** Wire the selector (no rendering effect yet). Plumbs the prop down so Task 2 can act on it.

**Files:**
- Modify: `src/App.vue`

**Step 1 — Add state (after line 173, near `manualCourtKeypoints`)**

```typescript
// Playback view mode: 'video' = real video + overlays (default),
// 'court' = synthetic court redrawn from keypoints + overlays.
// Only available when manual keypoints are set.
const viewMode = ref<'video' | 'court'>('video')

// Guard: if keypoints get cleared while viewing court mode, snap back to video.
watch(manualCourtKeypoints, (kp) => {
  if (kp === null && viewMode.value === 'court') {
    viewMode.value = 'video'
  }
})
```

Confirm `watch` is already imported from `vue`. If not, add it to the existing `vue` import at the top of the `<script setup>` block.

**Step 2 — Add the selector UI directly above the `<VideoPlayer>` element**

Find the `<VideoPlayer ... />` at roughly line 1531. Immediately before the existing `<div ref="videoSectionRef" class="video-section">` (line 1530), inject the selector:

```html
<div class="mode-selector view-mode-selector">
  <span class="mode-label">Playback View</span>
  <div class="mode-options">
    <button
      type="button"
      class="mode-option"
      :class="{ active: viewMode === 'video' }"
      @click="viewMode = 'video'"
    >
      <span class="mode-title">Video</span>
      <span class="mode-desc">Real footage with skeleton overlay</span>
    </button>
    <button
      type="button"
      class="mode-option"
      :class="{ active: viewMode === 'court' }"
      :disabled="manualCourtKeypoints === null"
      :title="manualCourtKeypoints === null ? 'Set manual court keypoints to enable' : ''"
      @click="viewMode = 'court'"
    >
      <span class="mode-title">Court</span>
      <span class="mode-desc">Synthetic court with skeleton + shuttle trail</span>
    </button>
  </div>
</div>
```

Reuses the `.mode-selector`, `.mode-options`, `.mode-option` classes already defined in `VideoUpload.vue` (lines 615–680). If those styles are not globally available in `App.vue`, check `VideoUpload.vue` — the classes may be scoped. If scoped, copy the three rules (`.mode-selector`, `.mode-options`, `.mode-option` + `.mode-option.active` + `:disabled`) into the `<style scoped>` block of `App.vue`. Pattern to copy verbatim from `VideoUpload.vue:615-680`; add a `:disabled` rule:

```css
.mode-option:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
```

**Step 3 — Pass `viewMode` into `VideoPlayer`**

In the `<VideoPlayer ...>` tag (around line 1531), add a new prop binding between `:court-keypoints` and `@court-keypoints-set`:

```html
:view-mode="viewMode"
```

This will emit a type error until Task 2 defines the prop on the component — that's expected.

**Step 4 — Verify**

```bash
npm run type-check
```
Expected: one type error about `view-mode` not being a valid prop on `VideoPlayer`. Leave it — Task 2 resolves it.

Run `npm run dev`, open a completed video with manual keypoints → verify the selector renders and the "Court" button is clickable. Then open a video WITHOUT manual keypoints → verify "Court" is greyed out with the tooltip.

**Step 5 — Commit**

```bash
git add src/App.vue
git commit -m "feat: add Video/Court view-mode selector in results view"
```

---

### Task 2: Accept `viewMode` prop in `VideoPlayer.vue` + hide video in court mode

**Goal:** Make `viewMode = 'court'` visually hide the video without breaking playback, seeking, or audio. No synthetic court drawing yet — just a dark background so the intent is visible.

**Files:**
- Modify: `src/components/VideoPlayer.vue`

**Step 1 — Add the prop**

In the `defineProps` block (around line 265–282), add `viewMode` alongside the other view-related props:

```typescript
viewMode?: 'video' | 'court'
```

**Step 2 — Apply the CSS hiding in the template**

Find the `<video>` element (around line 2142). Add a class binding that flips on `opacity: 0` when in court mode. Change:

```html
<video
  ref="videoRef"
  :src="videoUrl"
  crossorigin="anonymous"
  :class="{ 'video-dimmed': showHeatmap }"
```

to:

```html
<video
  ref="videoRef"
  :src="videoUrl"
  crossorigin="anonymous"
  :class="{ 'video-dimmed': showHeatmap, 'video-hidden': viewMode === 'court' }"
```

**Step 3 — Add the CSS rule**

In the `<style scoped>` block of `VideoPlayer.vue`, add near the existing `.video-dimmed` rule (search for it):

```css
.video-hidden {
  /* Keep video in layout + keep playback state (audio, timing) alive,
     but render it invisible so the synthetic court canvas shows through.
     opacity:0 is chosen over display:none so the video keeps decoding
     frames (the timing events drive skeleton/shuttle sync). */
  opacity: 0;
  pointer-events: none;
}

.video-wrapper:has(.video-hidden) {
  /* Dark backdrop behind the (invisible) video so the synthetic court
     has a canvas-native dark background even before SyntheticCourtView
     paints. */
  background: #0f1419;
}
```

If `:has()` triggers a lint/build issue (older Tailwind config), use a Vue class-binding on `.video-wrapper` instead: bind `:class="{ 'court-mode': viewMode === 'court' }"` and write `.video-wrapper.court-mode { background: #0f1419; }`.

**Step 4 — Verify**

```bash
npm run type-check
```
Expected: PASS (the Task 1 error is now resolved).

```bash
npm run dev
```
Open a video with keypoints, press play, toggle to Court. Expected:
- Video disappears (dark background).
- Audio keeps playing.
- Skeleton overlay keeps rendering in the same pixel positions as in Video mode.
- Timeline scrub still works.

Toggle back to Video → video reappears, playback continues seamlessly.

**Step 5 — Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "feat: hide video element when VideoPlayer is in court view mode"
```

---

### Task 3: Create `SyntheticCourtView.vue` (court lines only, no shuttle)

**Goal:** New component that computes the inverse homography and draws all standard badminton court lines onto a full-size canvas. Nothing else yet.

**Files:**
- Create: `src/components/SyntheticCourtView.vue`

**Design notes (read before coding):**

- The existing `@/utils/homography` `calculateHomography(src, dst)` maps `src` → `dst`. To draw court-meter lines into video-pixel space, we want the inverse direction: meters → pixels. Cleanest approach is to call `calculateHomography(courtMeterPoints, videoPixelPoints)` with **swapped** arguments — this gives us the matrix we need directly. No need to invert a matrix manually.
- The 12 keypoint labels in `ExtendedCourtKeypoints` already map 1:1 to the 12 entries in `COURT_KEYPOINT_POSITIONS` (exported from `@/utils/homography`). Order must match: TL, TR, BR, BL, NL, NR, SNL, SNR, SFL, SFR, CTN, CTF.
- Court dimensions come from `COURT_DIMENSIONS` already defined in `@/types/analysis` and already imported in `homography.ts`.

**Step 1 — Create the file**

`src/components/SyntheticCourtView.vue`:

```vue
<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { calculateHomography, applyHomography } from '@/utils/homography'
import { COURT_DIMENSIONS } from '@/types/analysis'

interface ExtendedCourtKeypoints {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  net_left: number[]
  net_right: number[]
  service_near_left: number[]
  service_near_right: number[]
  service_far_left: number[]
  service_far_right: number[]
  center_near: number[]
  center_far: number[]
}

const props = defineProps<{
  courtKeypoints: ExtendedCourtKeypoints
  videoWidth: number
  videoHeight: number
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)

// Court geometry in meters (origin top-left, y increasing toward back court).
const COURT_LEN = COURT_DIMENSIONS.length          // 13.4
const COURT_WID_D = COURT_DIMENSIONS.width_doubles // 6.1
const COURT_WID_S = COURT_DIMENSIONS.width_singles // 5.18
const NET_Y = COURT_LEN / 2                        // 6.7
const SERVICE = COURT_DIMENSIONS.service_line      // 1.98 from net
const DOUBLES_BACK_LINE_OFFSET = COURT_DIMENSIONS.doubles_back_line ?? 0.76

// Meters→pixels homography, recomputed when keypoints change.
const metersToPixels = computed((): number[][] | null => {
  const kp = props.courtKeypoints
  // Order must match COURT_KEYPOINT_POSITIONS in @/utils/homography.
  const videoPts = [
    kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left,
    kp.net_left, kp.net_right,
    kp.service_near_left, kp.service_near_right,
    kp.service_far_left, kp.service_far_right,
    kp.center_near, kp.center_far,
  ]
  // Court-meter positions matching the 12 keypoints above.
  const singlesOffset = (COURT_WID_D - COURT_WID_S) / 2
  const courtPts: number[][] = [
    [0, 0], [COURT_WID_D, 0], [COURT_WID_D, COURT_LEN], [0, COURT_LEN],
    [0, NET_Y], [COURT_WID_D, NET_Y],
    [0, NET_Y - SERVICE], [COURT_WID_D, NET_Y - SERVICE],
    [0, NET_Y + SERVICE], [COURT_WID_D, NET_Y + SERVICE],
    [COURT_WID_D / 2, NET_Y - SERVICE], [COURT_WID_D / 2, NET_Y + SERVICE],
  ]
  // Validate all 12 points are present (length >= 2, numeric).
  for (const p of videoPts) {
    if (!p || p.length < 2 || typeof p[0] !== 'number' || typeof p[1] !== 'number') {
      console.warn('[SyntheticCourtView] Incomplete keypoints; skipping homography.')
      return null
    }
  }
  const H = calculateHomography(courtPts, videoPts)
  if (!H) {
    console.warn('[SyntheticCourtView] Degenerate homography; court lines will not render.')
  }
  return H
})

// Helper: project a court-meter point to video-pixel coords.
function m2p(H: number[][], xm: number, ym: number): [number, number] | null {
  const p = applyHomography(H, xm, ym)
  return p ? [p.x, p.y] : null
}

// All standard badminton court lines, as pairs of meter endpoints.
function courtLineSegments(): Array<[[number, number], [number, number], 'normal' | 'net']> {
  const singlesOffset = (COURT_WID_D - COURT_WID_S) / 2
  const leftSingles = singlesOffset
  const rightSingles = COURT_WID_D - singlesOffset
  const shortSvcNear = NET_Y - SERVICE
  const shortSvcFar = NET_Y + SERVICE
  const longSvcNear = DOUBLES_BACK_LINE_OFFSET
  const longSvcFar = COURT_LEN - DOUBLES_BACK_LINE_OFFSET

  return [
    // Outer boundary (doubles)
    [[0, 0], [COURT_WID_D, 0], 'normal'],
    [[COURT_WID_D, 0], [COURT_WID_D, COURT_LEN], 'normal'],
    [[COURT_WID_D, COURT_LEN], [0, COURT_LEN], 'normal'],
    [[0, COURT_LEN], [0, 0], 'normal'],
    // Singles sidelines (full length)
    [[leftSingles, 0], [leftSingles, COURT_LEN], 'normal'],
    [[rightSingles, 0], [rightSingles, COURT_LEN], 'normal'],
    // Short service lines
    [[0, shortSvcNear], [COURT_WID_D, shortSvcNear], 'normal'],
    [[0, shortSvcFar], [COURT_WID_D, shortSvcFar], 'normal'],
    // Long service line (doubles)
    [[0, longSvcNear], [COURT_WID_D, longSvcNear], 'normal'],
    [[0, longSvcFar], [COURT_WID_D, longSvcFar], 'normal'],
    // Center line (does NOT cross the net zone)
    [[COURT_WID_D / 2, 0], [COURT_WID_D / 2, shortSvcNear], 'normal'],
    [[COURT_WID_D / 2, shortSvcFar], [COURT_WID_D / 2, COURT_LEN], 'normal'],
    // Net
    [[0, NET_Y], [COURT_WID_D, NET_Y], 'net'],
  ]
}

// Draw court lines to an offscreen canvas; re-created whenever homography changes.
const offscreenCourt = ref<HTMLCanvasElement | null>(null)

function buildOffscreenCourt() {
  const H = metersToPixels.value
  if (!H) { offscreenCourt.value = null; return }

  const off = document.createElement('canvas')
  off.width = props.videoWidth
  off.height = props.videoHeight
  const ctx = off.getContext('2d')
  if (!ctx) return

  ctx.fillStyle = '#0f1419'
  ctx.fillRect(0, 0, off.width, off.height)

  ctx.strokeStyle = '#f5f5f5'
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  for (const [[x1, y1], [x2, y2], kind] of courtLineSegments()) {
    const a = m2p(H, x1, y1)
    const b = m2p(H, x2, y2)
    if (!a || !b) continue
    ctx.lineWidth = kind === 'net' ? 3 : 2
    ctx.beginPath()
    ctx.moveTo(a[0], a[1])
    ctx.lineTo(b[0], b[1])
    ctx.stroke()
  }

  offscreenCourt.value = off
}

function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.width = props.videoWidth
  canvas.height = props.videoHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (offscreenCourt.value) {
    ctx.drawImage(offscreenCourt.value, 0, 0)
  } else {
    // Fallback: solid dark background when homography is unavailable.
    ctx.fillStyle = '#0f1419'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
}

onMounted(() => {
  buildOffscreenCourt()
  render()
})

watch([() => props.courtKeypoints, () => props.videoWidth, () => props.videoHeight], () => {
  buildOffscreenCourt()
  render()
}, { deep: true })

onUnmounted(() => {
  offscreenCourt.value = null
})
</script>

<template>
  <canvas ref="canvasRef" class="synthetic-court-canvas" />
</template>

<style scoped>
.synthetic-court-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 2; /* Above <video>, below PoseOverlay (which sits at higher z). */
}
</style>
```

**Step 2 — Verify `COURT_DIMENSIONS` exports what we need**

```bash
grep -n "doubles_back_line\|width_singles\|service_line" src/types/analysis.ts
```

If `doubles_back_line` is not in `COURT_DIMENSIONS`, the code above uses `?? 0.76` as a safe default. If `width_singles` is also missing, either add it to `COURT_DIMENSIONS` (with value `5.18`) or hard-code `5.18` in the component. Prefer adding to `COURT_DIMENSIONS` if other files could benefit — but YAGNI: if nothing else references it, inline it here.

**Step 3 — Verify**

```bash
npm run type-check
```
Expected: PASS.

Manual verification deferred to Task 4 (component not yet mounted anywhere).

**Step 4 — Commit**

```bash
git add src/components/SyntheticCourtView.vue
git commit -m "feat: add SyntheticCourtView component with court-line rendering"
```

---

### Task 4: Mount `SyntheticCourtView` inside `VideoPlayer` when in court mode

**Goal:** Wire the new component into the player so Court mode actually shows a court.

**Files:**
- Modify: `src/components/VideoPlayer.vue`

**Step 1 — Import and type the new prop**

At the top of `<script setup>` in `VideoPlayer.vue`, add the import:

```typescript
import SyntheticCourtView from './SyntheticCourtView.vue'
```

In the existing `defineProps` block, add (alongside the existing `courtKeypoints?: number[][] | null` at roughly line 281):

```typescript
manualCourtKeypoints?: ExtendedCourtKeypoints | null
```

Note: there is ALREADY a `courtKeypoints?: number[][] | null` prop (array-of-pairs, used elsewhere). Do NOT overwrite it. Add a separate `manualCourtKeypoints` typed as the structured object, since `SyntheticCourtView` needs the named-field form.

**Step 2 — Mount conditionally in the template**

In the `.video-wrapper` (around line 2141), immediately after the `<video>` element (line 2152) and before the first `<canvas>` (line 2153), insert:

```html
<SyntheticCourtView
  v-if="viewMode === 'court' && manualCourtKeypoints && videoRef?.videoWidth"
  :court-keypoints="manualCourtKeypoints"
  :video-width="videoRef.videoWidth"
  :video-height="videoRef.videoHeight"
/>
```

**Step 3 — Pass the prop from `App.vue`**

In `src/App.vue`, on the `<VideoPlayer ...>` tag (around line 1531), add:

```html
:manual-court-keypoints="manualCourtKeypoints"
```

**Step 4 — Verify**

```bash
npm run type-check
```
Expected: PASS.

```bash
npm run dev
```
Open a video with keypoints → toggle to Court → expected: video disappears, court lines appear drawn onto a dark background in the same camera perspective, skeletons continue to render at identical pixel positions as in Video mode (the skeleton-to-court alignment is the key visual correctness check). Play the video — court lines should NOT move (they're static per keypoints); skeletons should move.

Known-correct sanity check: the manually-clicked keypoints (TL, TR, BR, BL, net line ends, service line ends) should appear exactly on the corresponding drawn lines. A visible offset means either keypoint ordering in `m2p`-input array is wrong or the homography is degenerate.

**Step 5 — Commit**

```bash
git add src/components/VideoPlayer.vue src/App.vue
git commit -m "feat: render SyntheticCourtView inside VideoPlayer when in court mode"
```

---

### Task 5: Add shuttle trail rendering to `SyntheticCourtView`

**Goal:** Draw the shuttle's current position + fading trail of last ~0.5s of positions.

**Files:**
- Modify: `src/components/SyntheticCourtView.vue`
- Modify: `src/components/VideoPlayer.vue`
- Modify: `src/App.vue`

**Step 1 — Extend `SyntheticCourtView` props**

Add to `defineProps`:

```typescript
skeletonData?: Array<{ frame: number; shuttle_position?: { x: number; y: number; visible?: boolean } | null }>
currentFrame?: number
fps?: number
```

Use a loose inline type rather than importing `SkeletonFrame` so the component stays decoupled from the broader analytics types. It only needs `frame` + `shuttle_position` fields.

**Step 2 — Add trail rendering + tick loop**

Add to `<script setup>` after `render()`:

```typescript
// Trail = last ~0.5s of shuttle positions. At 30fps = 15 frames.
const TRAIL_SECONDS = 0.5

function shuttleFrames(): Array<{ x: number; y: number; age: number }> {
  const cf = props.currentFrame ?? 0
  const fps = props.fps && props.fps > 0 ? props.fps : 30
  const window = Math.max(1, Math.round(TRAIL_SECONDS * fps))
  const frames = props.skeletonData
  if (!frames || frames.length === 0) return []

  // skeletonData is frame-indexed but may skip frames. Walk backward from
  // currentFrame, take up to `window` visible shuttle samples.
  const result: Array<{ x: number; y: number; age: number }> = []
  // Binary search would be nicer but frames are small + we only look at a window.
  for (let i = frames.length - 1; i >= 0 && result.length < window; i--) {
    const f = frames[i]
    if (!f || f.frame > cf) continue
    const sp = f.shuttle_position
    if (!sp || sp.visible === false) continue
    if (typeof sp.x !== 'number' || typeof sp.y !== 'number') continue
    const age = cf - f.frame
    if (age >= window) break
    result.push({ x: sp.x, y: sp.y, age })
  }
  return result // result[0] is newest
}

function drawShuttle(ctx: CanvasRenderingContext2D) {
  const pts = shuttleFrames()
  if (pts.length === 0) return
  const window = Math.max(1, Math.round(TRAIL_SECONDS * (props.fps && props.fps > 0 ? props.fps : 30)))

  // Trail (oldest → newest) so newer points paint over older.
  for (let i = pts.length - 1; i >= 0; i--) {
    const { x, y, age } = pts[i]!
    const alpha = Math.max(0.05, 1 - age / window)
    ctx.globalAlpha = alpha * 0.9
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(x, y, 3, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1

  // Current position: bright dot + glow.
  const head = pts[0]!
  ctx.shadowColor = '#ffffff'
  ctx.shadowBlur = 12
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  ctx.arc(head.x, head.y, 6, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
}
```

Change `render()` to also draw the shuttle:

```typescript
function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.width = props.videoWidth
  canvas.height = props.videoHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (offscreenCourt.value) {
    ctx.drawImage(offscreenCourt.value, 0, 0)
  } else {
    ctx.fillStyle = '#0f1419'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }

  drawShuttle(ctx)
}
```

Add a rAF-driven tick so the trail updates every frame without a watcher chain:

```typescript
let rafId: number | null = null
function tick() {
  render()
  rafId = requestAnimationFrame(tick)
}

onMounted(() => {
  buildOffscreenCourt()
  rafId = requestAnimationFrame(tick)
})

onUnmounted(() => {
  if (rafId !== null) cancelAnimationFrame(rafId)
  offscreenCourt.value = null
})
```

Remove the old plain `render()` call from `onMounted` — it's replaced by the tick loop. Keep the `watch` on `courtKeypoints` / dimensions calling `buildOffscreenCourt()` (but drop the `render()` call from inside the watch — the tick loop handles re-renders):

```typescript
watch([() => props.courtKeypoints, () => props.videoWidth, () => props.videoHeight], () => {
  buildOffscreenCourt()
}, { deep: true })
```

**Step 3 — Forward the new props from `VideoPlayer`**

In `VideoPlayer.vue`, the component already receives `skeletonData`. Expose the current frame number — there should already be reactive state tracking this (search for `emit('frameUpdate', ...)` to find it). Identify the reactive variable (likely `currentFrame` or derived from video time). Also compute fps or accept it as a prop — simplest path: derive from `videoRef?.videoWidth`-style availability. If fps is not currently a prop in `VideoPlayer`, accept it from the parent:

```typescript
// In defineProps block:
videoFps?: number
```

Pass these to `SyntheticCourtView` in the template:

```html
<SyntheticCourtView
  v-if="viewMode === 'court' && manualCourtKeypoints && videoRef?.videoWidth"
  :court-keypoints="manualCourtKeypoints"
  :video-width="videoRef.videoWidth"
  :video-height="videoRef.videoHeight"
  :skeleton-data="skeletonData"
  :current-frame="currentFrameInternal"
  :fps="videoFps"
/>
```

Replace `currentFrameInternal` with the actual reactive ref name used inside `VideoPlayer.vue` (find it via grep: `grep -n "frameUpdate\|currentFrame" src/components/VideoPlayer.vue`).

**Step 4 — Pass `videoFps` from `App.vue`**

`App.vue` already has access to analysis results; fps lives on `analysisResult.value.fps`. On the `<VideoPlayer>` tag:

```html
:video-fps="analysisResult?.fps ?? 30"
```

**Step 5 — Verify**

```bash
npm run type-check
```
Expected: PASS.

```bash
npm run dev
```
Open a video with keypoints AND shuttle data (TrackNet was enabled when processing). Toggle to Court mode, press play. Expected:
- Shuttle dot visible at the shuttle's current pixel location.
- Short fading tail behind it showing recent trajectory.
- Tail disappears during occlusion frames (where `visible === false`).
- Tail disappears when video is paused at a frame the shuttle is absent.
- Toggle back to Video — shuttle trail stops being drawn (component unmounted), everything still works.

Performance sanity: open DevTools Performance → record 10 seconds of playback in Court mode → no frame drops, rAF loop sits comfortably within 16ms.

**Step 6 — Commit**

```bash
git add src/components/SyntheticCourtView.vue src/components/VideoPlayer.vue src/App.vue
git commit -m "feat: draw shuttle trail in synthetic court view"
```

---

### Task 6: Edge cases + polish

**Goal:** Make sure the feature behaves sanely on the boundary cases we scoped in the design.

**Files:**
- Potentially: `src/App.vue`, `src/components/SyntheticCourtView.vue`

**Step 1 — Video without keypoints**

Open a video where the user never ran Court Setup.
Expected: "Court" button disabled with tooltip; clicking does nothing.
If not behaving: check that `manualCourtKeypoints.value === null` is the condition used for `:disabled` in Task 1.

**Step 2 — Keypoints get cleared while in Court mode**

In-app path: currently there may not be a UI to clear keypoints mid-session. If there is (check for a "Clear keypoints" button), test it. Expected: the `watch(manualCourtKeypoints, ...)` from Task 1 flips back to `video`. If this guard isn't firing, confirm the watcher is installed.

**Step 3 — Video with keypoints but no shuttle data**

Open an older video processed before TrackNet was enabled (or one where `shuttle_position` is absent).
Expected: Court view renders court lines + skeletons, no shuttle dot, no errors in console.
If seeing errors: the `shuttleFrames()` function should return `[]` cleanly — double-check the early `return []` when `frames.length === 0`.

**Step 4 — Degenerate keypoints**

Manually edit a video's keypoints in Convex to have four collinear points (you can do this via `npx convex dashboard` → videos table → edit the `manualCourtKeypoints` JSON — or simpler: temporarily set two keypoints to identical positions in the app via re-running keypoint selection with bad clicks).
Expected: `calculateHomography` returns `null`; component logs the warning and draws just the dark background. Skeletons still render (they're handled by `PoseOverlay`, not this component).

**Step 5 — Commit any fixes**

```bash
git add -A
git commit -m "fix: handle edge cases in synthetic court view"
```

(If no fixes were needed, skip this commit.)

---

### Task 7: Final verification + branch cleanup

**Step 1 — Full build**

```bash
npm run type-check && npm run build
```
Expected: both pass with no new errors/warnings.

**Step 2 — Review diff**

```bash
git log --oneline feat/camera-angle-rally-presets..HEAD -- src/ docs/plans/2026-04-18-synthetic-court-view-*
git diff feat/camera-angle-rally-presets..HEAD -- src/ | wc -l
```

Confirm all changes are in:
- `src/App.vue`
- `src/components/VideoPlayer.vue`
- `src/components/SyntheticCourtView.vue` (new)
- `docs/plans/2026-04-18-synthetic-court-view-*.md`

No changes should appear in `backend/`, `convex/`, or unrelated components (`MiniCourt.vue`, `PoseOverlay.vue`, etc.).

**Step 3 — Final manual pass**

Run through the scenarios one last time on `npm run dev`:
1. Video with keypoints + shuttle data → toggle Video ↔ Court multiple times during playback.
2. Seek around while in Court mode → court stays put, skeletons + shuttle follow the timeline.
3. Enter fullscreen in Court mode → layout doesn't break.
4. Toggle `showPoseOverlay` while in Court mode → skeletons appear/disappear correctly on top of the synthetic court.

---

## Out of scope (do NOT attempt in this plan)

- Keyboard shortcut for the toggle.
- Tinted service boxes, net mesh, court color background (broadcast polish).
- Player ID labels on the court view.
- Shot/rally annotations on the court.
- Supporting auto-detected court keypoints (feature currently requires manual keypoints).
- Any test framework introduction.

Each of these can be added in a follow-up plan once the v1 feature is in users' hands.
