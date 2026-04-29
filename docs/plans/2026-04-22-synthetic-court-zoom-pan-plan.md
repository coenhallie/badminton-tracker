# Synthetic Court Zoom & Pan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the user zoom (up to 8×) and pan within the synthetic court view to inspect skeleton posture and numeric angle overlays up close, with state that persists across scrub, play/pause, and view-mode toggle.

**Architecture:** A framework-agnostic `useViewportCamera` composable owns `{ scale, tx, ty }` and exposes pan/zoom/reset helpers plus `applyToContext(ctx)` and `pixelSize(v)`. The camera is instantiated in `VideoPlayer.vue` (which stays mounted across the `viewMode` toggle) and passed as a prop to `SyntheticCourtView.vue`. A transparent event-capture overlay (pointer-events: auto, mounted only in court mode) translates wheel/drag/dblclick into camera updates. Both the synthetic-court canvas and the skeleton-overlay canvas apply the camera's transform at the top of their render paths, and rely on `pixelSize()` so font sizes, line widths, and keypoint radii stay a constant on-screen pixel size at any zoom.

**Tech Stack:** Vue 3 `<script setup>` + TypeScript, Canvas 2D (`ctx.setTransform`), `ResizeObserver` for canvas resize re-clamping.

**Design doc:** `docs/plans/2026-04-22-synthetic-court-zoom-pan-design.md`

---

### Design discoveries that shift implementation

Three things the design doc got almost right but not quite, uncovered while reading source:

1. **Two canvases, not one.** Court mode stacks `SyntheticCourtView`'s canvas (court lines + shuttle) AND `VideoPlayer`'s `.skeleton-canvas` (bones + angle arcs + leg-stretch). The camera transform has to be applied in both render paths. Same composable, applied twice.

2. **SyntheticCourtView pre-rasterises the court to an offscreen canvas** (`buildOffscreenCourt`, lines 114–145) and blits via `ctx.drawImage()`. Blitting under a zoom transform would re-sample with bilinear filtering and blur the lines at 2× or more. The offscreen cache has to go — court lines will be drawn directly each frame, under the transform, with widths fed through `camera.pixelSize()`. The per-frame cost is ~13 line segments + a net; trivial.

3. **Both canvases have `pointer-events: none`** (SyntheticCourtView.vue:321; the skeleton canvas almost certainly the same). A transparent `<div class="zoom-capture">` with `pointer-events: auto` has to sit on top of both, visible only in court mode, to receive input. The `ViewportControls` HUD lives inside that same capture layer.

4. **Camera state lives in `VideoPlayer.vue`, not `App.vue`.** Rationale: `VideoPlayer.vue` stays mounted across the `viewMode` toggle (the `<video>` element is hidden via a class at `VideoPlayer.vue:1804`, not unmounted), so keeping state here already satisfies the persistence requirement with one fewer prop layer. When the port to the video view ships in the follow-up, the same camera instance will naturally drive both.

---

### Verification primitives used throughout

No unit-test framework in this repo (confirmed against `package.json`). Each task uses:
- `npm run type-check` — must pass with zero NEW errors. Pre-existing errors to ignore match the prior plan's list: `convex/http.ts:1267` (`process`), `convex/videos.ts:461,472` (`process`), `src/App.vue` rally-possibly-undefined lines, `src/components/VideoUpload.vue:191`, `src/composables/useVideoExport.ts:131,142`.
- `npm run build` — clean build.
- **Manual browser check** — required per task where behaviour changes visually or on input; test video should have an analysed rally with visible skeletons and at least one angle overlay enabled.

The composable math is small (one homography-free 2D affine transform) and is tested via manual zoom-in/zoom-out checks — the inverse-transform correctness is visible: if `zoomAt` is wrong, zooming toward a joint will visibly drift off-target.

---

### Task 1: `useViewportCamera` composable

**Goal:** Pure-math composable with reactive `scale`, `tx`, `ty` and all helpers. No DOM, no Vue lifecycle — callable from any component.

**Files:**
- Create: `src/composables/useViewportCamera.ts`

**Step 1 — Write the composable**

```typescript
import { ref, readonly, type Ref } from 'vue'

export interface ViewportCameraOptions {
  /** Inclusive [min, max] zoom range. Defaults to [1, 8]. */
  scaleRange?: [number, number]
  /** Fraction of the viewport that the anchored bbox must keep on-screen
   *  when clamping pan. 0.2 means at least 20% visible. Defaults to 0.2. */
  minVisibleFraction?: number
}

export interface ViewportCamera {
  scale: Readonly<Ref<number>>
  tx: Readonly<Ref<number>>
  ty: Readonly<Ref<number>>
  /** Reset to scale=1, tx=0, ty=0. */
  reset(): void
  /** Zoom toward (clientX, clientY) on the canvas element `el`. Delta is
   *  a multiplicative scale factor (e.g. 1.15 = zoom in one notch). */
  zoomAt(el: HTMLElement, clientX: number, clientY: number, delta: number): void
  /** Pan by dx, dy screen pixels. */
  panBy(el: HTMLElement, dx: number, dy: number): void
  /** ctx.setTransform(scale, 0, 0, scale, tx, ty). */
  applyToContext(ctx: CanvasRenderingContext2D): void
  /** Convert a "how many on-screen pixels do I want?" value into the
   *  pre-transform world-space value that yields that many on-screen px. */
  pixelSize(screenPx: number): number
  /** Screen-space (x, y) → world-space (pre-transform). */
  screenToWorld(x: number, y: number): { x: number; y: number }
  /** World-space (x, y) → screen-space (post-transform). */
  worldToScreen(x: number, y: number): { x: number; y: number }
  /** Re-clamp pan against the current viewport size. Call after canvas resize. */
  reclamp(el: HTMLElement): void
}

export function useViewportCamera(options: ViewportCameraOptions = {}): ViewportCamera {
  const [MIN_SCALE, MAX_SCALE] = options.scaleRange ?? [1, 8]
  const MIN_VISIBLE = options.minVisibleFraction ?? 0.2

  const scale = ref(1)
  const tx = ref(0)
  const ty = ref(0)

  function clampScale(s: number): number {
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s))
  }

  /** Clamp tx/ty so at least MIN_VISIBLE of the viewport's bbox stays on-screen. */
  function clampPan(el: HTMLElement): void {
    const w = el.clientWidth
    const h = el.clientHeight
    if (w <= 0 || h <= 0) return
    // The transformed viewport covers world-rect [−tx/s, (w−tx)/s] × [−ty/s, (h−ty)/s].
    // Equivalent: the world's [0,w] × [0,h] rect maps to screen [tx, tx + w*s] × [ty, ty + h*s].
    // Require at least MIN_VISIBLE * w (resp. h) of that rect to stay inside [0, w] (resp. [0, h]).
    const s = scale.value
    const minOverlap = MIN_VISIBLE
    const maxTx = w * (1 - minOverlap)
    const minTx = w * minOverlap - w * s
    tx.value = Math.min(maxTx, Math.max(minTx, tx.value))
    const maxTy = h * (1 - minOverlap)
    const minTy = h * minOverlap - h * s
    ty.value = Math.min(maxTy, Math.max(minTy, ty.value))
  }

  function reset(): void {
    scale.value = 1
    tx.value = 0
    ty.value = 0
  }

  function zoomAt(el: HTMLElement, clientX: number, clientY: number, delta: number): void {
    const rect = el.getBoundingClientRect()
    // Screen-space point on the canvas, in canvas pixels.
    const sx = clientX - rect.left
    const sy = clientY - rect.top
    const oldScale = scale.value
    const newScale = clampScale(oldScale * delta)
    if (newScale === oldScale) return
    // Keep the world point under the cursor fixed:
    //   screen = world * s + t   =>   world = (screen − t) / s
    // After zoom, want screen' = screen, so: t' = screen − world * s'
    const worldX = (sx - tx.value) / oldScale
    const worldY = (sy - ty.value) / oldScale
    scale.value = newScale
    tx.value = sx - worldX * newScale
    ty.value = sy - worldY * newScale
    clampPan(el)
  }

  function panBy(el: HTMLElement, dx: number, dy: number): void {
    tx.value += dx
    ty.value += dy
    clampPan(el)
  }

  function applyToContext(ctx: CanvasRenderingContext2D): void {
    ctx.setTransform(scale.value, 0, 0, scale.value, tx.value, ty.value)
  }

  function pixelSize(screenPx: number): number {
    return screenPx / scale.value
  }

  function screenToWorld(x: number, y: number): { x: number; y: number } {
    return { x: (x - tx.value) / scale.value, y: (y - ty.value) / scale.value }
  }

  function worldToScreen(x: number, y: number): { x: number; y: number } {
    return { x: x * scale.value + tx.value, y: y * scale.value + ty.value }
  }

  function reclamp(el: HTMLElement): void {
    clampPan(el)
  }

  return {
    scale: readonly(scale),
    tx: readonly(tx),
    ty: readonly(ty),
    reset,
    zoomAt,
    panBy,
    applyToContext,
    pixelSize,
    screenToWorld,
    worldToScreen,
    reclamp,
  }
}
```

**Step 2 — Verify type-check**

Run: `npm run type-check`
Expected: no new errors (only the pre-existing 8).

**Step 3 — Commit**

```bash
git add src/composables/useViewportCamera.ts
git commit -m "feat: add useViewportCamera composable for 2D pan/zoom"
```

---

### Task 2: Event-capture overlay + camera wiring in `VideoPlayer.vue`

**Goal:** Instantiate the camera in `VideoPlayer.vue`, pass it to `SyntheticCourtView`, and add a transparent input-capture layer (visible only in court mode) that feeds wheel/drag/dblclick into the camera.

**Files:**
- Modify: `src/components/VideoPlayer.vue` (script + template + styles)

**Step 1 — Import & instantiate**

Near the other composable imports at the top of `<script setup>` (search for `import { ... } from 'vue'`), add:

```typescript
import { useViewportCamera } from '@/composables/useViewportCamera'
```

Then, inside the setup body (place it near other refs like `canvasRef`, `containerRef`), add:

```typescript
const camera = useViewportCamera()
const zoomCaptureRef = ref<HTMLDivElement | null>(null)

// Drag state for pan.
let isPanning = false
let lastPanX = 0
let lastPanY = 0

function onZoomWheel(e: WheelEvent) {
  e.preventDefault()
  const el = zoomCaptureRef.value
  if (!el) return
  // Wheel up (deltaY < 0) = zoom in. One notch ≈ 1.15×.
  const delta = e.deltaY < 0 ? 1.15 : 1 / 1.15
  camera.zoomAt(el, e.clientX, e.clientY, delta)
}

function onZoomMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  isPanning = true
  lastPanX = e.clientX
  lastPanY = e.clientY
  window.addEventListener('mousemove', onZoomMouseMove)
  window.addEventListener('mouseup', onZoomMouseUp)
}

function onZoomMouseMove(e: MouseEvent) {
  if (!isPanning) return
  const el = zoomCaptureRef.value
  if (!el) return
  const dx = e.clientX - lastPanX
  const dy = e.clientY - lastPanY
  lastPanX = e.clientX
  lastPanY = e.clientY
  camera.panBy(el, dx, dy)
}

function onZoomMouseUp() {
  isPanning = false
  window.removeEventListener('mousemove', onZoomMouseMove)
  window.removeEventListener('mouseup', onZoomMouseUp)
}

function onZoomDoubleClick() {
  camera.reset()
}
```

**Step 2 — Attach ResizeObserver so pan re-clamps on viewport changes**

Near `onMounted` / `onUnmounted` blocks, add:

```typescript
let zoomResizeObserver: ResizeObserver | null = null

function attachZoomResizeObserver() {
  const el = zoomCaptureRef.value
  if (!el || zoomResizeObserver) return
  zoomResizeObserver = new ResizeObserver(() => camera.reclamp(el))
  zoomResizeObserver.observe(el)
}

function detachZoomResizeObserver() {
  if (zoomResizeObserver) {
    zoomResizeObserver.disconnect()
    zoomResizeObserver = null
  }
}
```

Call `attachZoomResizeObserver()` in `onMounted`, and `detachZoomResizeObserver()` in `onUnmounted`. The observer attaches only if the element exists, so it's safe if the capture layer mounts later via `v-if`.

**Step 3 — Template: capture overlay + pass camera to `SyntheticCourtView`**

In the `<template>`, find `<SyntheticCourtView>` (around line 1811) and pass the camera as a prop. Then add the capture overlay immediately after the `<canvas class="skeleton-canvas">` block (around line 1824). It must be positioned absolutely over the wrapper, z-index above both canvases but below the existing `.play-overlay`.

```vue
<SyntheticCourtView
  v-if="viewMode === 'court' && manualCourtKeypoints && videoRef?.videoWidth"
  :court-keypoints="manualCourtKeypoints"
  :video-width="videoRef.videoWidth"
  :video-height="videoRef.videoHeight"
  :skeleton-data="skeletonData"
  :current-frame="currentFrame"
  :fps="videoFps"
  :camera="camera"
/>
<canvas
  v-if="(showSkeleton || showBoundingBoxes || showHeatmap) && skeletonData"
  ref="canvasRef"
  class="skeleton-canvas"
/>
<div
  v-if="viewMode === 'court'"
  ref="zoomCaptureRef"
  class="zoom-capture"
  @wheel="onZoomWheel"
  @mousedown="onZoomMouseDown"
  @dblclick="onZoomDoubleClick"
>
  <!-- ViewportControls HUD mounts here in Task 5 -->
</div>
```

**Step 4 — Styles**

In the `<style scoped>` block, add:

```css
.zoom-capture {
  position: absolute;
  inset: 0;
  z-index: 5; /* above .skeleton-canvas (z=3) and synthetic court (z=2),
                 below .play-overlay which doesn't use z-index but is ordered later */
  cursor: grab;
  touch-action: none; /* prevent default scroll on trackpad pinch */
}
.zoom-capture:active { cursor: grabbing; }
```

Adjust z-index value after confirming the skeleton canvas's existing z-index via Grep for `skeleton-canvas` in VideoPlayer.vue styles.

**Step 5 — Verify type-check + build**

Run: `npm run type-check && npm run build`
Expected: passes (only pre-existing errors).

**Step 6 — Manual smoke test**

- Load a video, run analysis, switch to court mode.
- Confirm the court canvas renders (it won't zoom yet — that's Task 3/4).
- Confirm wheel / drag / dblclick DON'T throw console errors.
- `camera.scale` should visibly change in Vue Devtools when scrolling.

**Step 7 — Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "feat: wire useViewportCamera + input capture overlay in VideoPlayer"
```

---

### Task 3: Apply camera transform to the skeleton canvas

**Goal:** The skeleton canvas (bones + angle arcs + leg stretch + keypoint dots) now zooms and pans with the camera; font sizes, line widths, and keypoint radii stay a constant on-screen pixel size.

**Files:**
- Modify: `src/components/VideoPlayer.vue` — `drawOverlay`, `drawSkeleton`, `drawAngleArc`, `drawLegStretch`

**Step 1 — Apply transform at the top of the skeleton draw**

Find `drawOverlay` (search for `function drawOverlay`). Inside, after obtaining `ctx` and `clearRect`, add before any subsequent draw calls:

```typescript
camera.applyToContext(ctx)
```

Make sure `clearRect` uses the **identity** transform so the entire canvas clears regardless of pan/zoom state; do the clear with `ctx.setTransform(1, 0, 0, 1, 0, 0)` first, THEN call `camera.applyToContext(ctx)`. Concretely:

```typescript
ctx.setTransform(1, 0, 0, 1, 0, 0) // identity for clear
ctx.clearRect(0, 0, canvas.width, canvas.height)
camera.applyToContext(ctx)
// ... rest of the existing draw code unchanged
```

Without this reset, `clearRect(0, 0, w, h)` under a zoom transform clears only a sub-region, leaving ghost frames.

**Step 2 — Replace hardcoded font sizes in `drawAngleArc` (`VideoPlayer.vue:1345`)**

Change:
```typescript
ctx.font = 'bold 11px Inter, system-ui, sans-serif'
```
to:
```typescript
ctx.font = `bold ${camera.pixelSize(11)}px Inter, system-ui, sans-serif`
```

Do the same for `ctx.lineWidth` values on lines 1335 (`= 2`) and 1348 (`= 2.5`):
```typescript
ctx.lineWidth = camera.pixelSize(2)
// ...
ctx.lineWidth = camera.pixelSize(2.5)
```

And the arc radius on line 1331:
```typescript
const radius = camera.pixelSize(20)
```

Label offset on lines 1342–1343 uses `radius + 14`; since `radius` is already in world-space, the `+ 14` needs to become `+ camera.pixelSize(14)` so the label offset from the arc stays a constant screen distance:
```typescript
const labelX = vx + Math.cos(midAngle) * (radius + camera.pixelSize(14))
const labelY = vy + Math.sin(midAngle) * (radius + camera.pixelSize(14))
```

The text-offset constants on lines 1349–1350 (`labelX - 10, labelY + 4`) similarly: use `camera.pixelSize(10)` and `camera.pixelSize(4)`.

**Step 3 — Same treatment for `drawLegStretch` (`VideoPlayer.vue:1353–1391`)**

Lines 1375, 1388 (`lineWidth = 2`, `= 2.5`) → `camera.pixelSize(2)`, `camera.pixelSize(2.5)`.
Line 1385 font → `camera.pixelSize(12)`px.
Line 1371 `setLineDash([6, 4])` → `setLineDash([camera.pixelSize(6), camera.pixelSize(4)])`.
Lines 1389–1390 offsets `(mx - 15, my - 8)` → use `camera.pixelSize(15)` / `camera.pixelSize(8)`.

**Step 4 — Skeleton bone lineWidth + keypoint radii (in `drawSkeleton`)**

Open `drawSkeleton` at `VideoPlayer.vue:1393–1529`. Find:
- Any `ctx.lineWidth = N` for bone strokes → `camera.pixelSize(N)`.
- Any `ctx.arc(x, y, radius, ...)` for keypoint dots → `ctx.arc(x, y, camera.pixelSize(radius), ...)`.
- Player-id label / name fonts (if any) → same treatment.

Use Grep on this file if needed: search for `lineWidth =` and `arc(` within lines 1393–1529 to make sure none are missed.

**Step 5 — Watcher: redraw skeleton when camera changes while video is paused**

The skeleton animation loop only runs while playing. If the video is paused and the user zooms, the last-drawn frame sits there stale under the old transform. Add a watcher:

```typescript
watch([() => camera.scale.value, () => camera.tx.value, () => camera.ty.value], () => {
  if (!isPlaying.value) {
    drawOverlay() // re-draws last frame under new transform
  }
})
```

Place this near other watchers in the setup body.

**Step 6 — Verify type-check + build**

Run: `npm run type-check && npm run build`
Expected: passes.

**Step 7 — Manual verification**

- In court mode, pause on a frame with visible skeletons.
- Zoom in with the wheel: skeletons should grow; bones should stay uniform-width, not ballooning.
- Angle arc labels (if enabled) should stay the same readable size and stay anchored to their joint.
- Drag-pan: skeletons and angle labels move together.
- Double-click: snaps back to fit.

**Step 8 — Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "feat: apply camera transform + pixelSize to skeleton overlay"
```

---

### Task 4: Direct-render `SyntheticCourtView` under camera transform

**Goal:** Remove the offscreen-court cache; draw court lines + net + shuttle directly each frame with the camera transform active. Line widths stay uniform-width at any zoom.

**Files:**
- Modify: `src/components/SyntheticCourtView.vue`

**Step 1 — Accept the camera prop**

Add to `defineProps`:
```typescript
import type { ViewportCamera } from '@/composables/useViewportCamera'

const props = defineProps<{
  courtKeypoints: ExtendedCourtKeypoints
  videoWidth: number
  videoHeight: number
  skeletonData?: Array<{ frame: number; shuttle_position?: { x: number; y: number; visible?: boolean } | null }>
  currentFrame?: number
  fps?: number
  camera: ViewportCamera
}>()
```

**Step 2 — Remove the offscreen court cache**

Delete:
- The `offscreenCourt` ref (line 112).
- The `buildOffscreenCourt()` function (lines 114–145).
- The `watch(...)` call that rebuilds it (lines 300–302).
- The `buildOffscreenCourt()` call in `onMounted` (line 296).
- The `offscreenCourt.value = null` in `onUnmounted` (line 306).

**Step 3 — Inline the court draw in `render()`**

Replace the current `render()` body with:

```typescript
function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.width = props.videoWidth
  canvas.height = props.videoHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Identity for clear, then apply camera.
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#0f1419'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  props.camera.applyToContext(ctx)

  const H = metersToPixels.value
  if (H) {
    ctx.strokeStyle = '#f5f5f5'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    for (const [[x1, y1], [x2, y2], kind] of courtLineSegments()) {
      const a = m2p(H, x1, y1)
      const b = m2p(H, x2, y2)
      if (!a || !b) continue
      ctx.lineWidth = props.camera.pixelSize(kind === 'net' ? 3 : 2)
      ctx.beginPath()
      ctx.moveTo(a[0], a[1])
      ctx.lineTo(b[0], b[1])
      ctx.stroke()
    }
    drawNet(ctx, H)
  }

  drawShuttle(ctx)
}
```

Note: the `fillRect` for the dark background happens BEFORE `applyToContext` so it fills the entire canvas regardless of pan. Court lines and shuttle are drawn AFTER, under transform. This means panning reveals dark background on the edges — correct.

**Step 4 — Apply `pixelSize` inside `drawNet`**

Change `ctx.lineWidth = 3` (line 196) → `ctx.lineWidth = props.camera.pixelSize(3)`.
Change `ctx.lineWidth = 2` (line 202) → `ctx.lineWidth = props.camera.pixelSize(2)`.

**Step 5 — Apply `pixelSize` inside `drawShuttle`**

Change `ctx.arc(x, y, 3, 0, Math.PI * 2)` (line 273) → `ctx.arc(x, y, props.camera.pixelSize(3), 0, Math.PI * 2)`.
Change `ctx.arc(head.x, head.y, 6, ...)` (line 284) → `ctx.arc(head.x, head.y, props.camera.pixelSize(6), ...)`.
Change `ctx.shadowBlur = 12` (line 281) → `ctx.shadowBlur = props.camera.pixelSize(12)`.

**Step 6 — Verify type-check + build**

Run: `npm run type-check && npm run build`
Expected: passes.

**Step 7 — Manual verification**

- Court mode at 1× zoom should look identical to before (same lines, same net).
- Wheel-zoom to 4×: court lines stay thin (uniform screen width), no blurring or re-sampling artefacts.
- Pan: court moves smoothly; dark background fills revealed edges.
- Shuttle trail (if present): dots stay uniform size.
- Return to 1× via double-click: identical to baseline.

**Step 8 — Commit**

```bash
git add src/components/SyntheticCourtView.vue
git commit -m "refactor: direct-render synthetic court under camera transform

Offscreen rasterised cache removed so line widths stay crisp under zoom.
Per-frame cost is ~13 segments + net; negligible."
```

---

### Task 5: `ViewportControls` HUD

**Goal:** Small HUD with zoom % readout, `−` / `+` / ⟲ buttons. Mounts inside the `.zoom-capture` overlay in court mode only.

**Files:**
- Create: `src/components/ViewportControls.vue`
- Modify: `src/components/VideoPlayer.vue` (template — mount the HUD inside the capture div)

**Step 1 — Create the component**

```vue
<script setup lang="ts">
import { computed } from 'vue'
import type { ViewportCamera } from '@/composables/useViewportCamera'

const props = defineProps<{ camera: ViewportCamera; captureEl?: HTMLElement | null }>()

const zoomPercent = computed(() => Math.round(props.camera.scale.value * 100))

function stepZoom(factor: number, event: MouseEvent) {
  event.stopPropagation()
  const el = props.captureEl ?? (event.currentTarget as HTMLElement).parentElement
  if (!el) return
  // Zoom toward the center of the capture element.
  const rect = el.getBoundingClientRect()
  const cx = rect.left + rect.width / 2
  const cy = rect.top + rect.height / 2
  props.camera.zoomAt(el, cx, cy, factor)
}

function reset(event: MouseEvent) {
  event.stopPropagation()
  props.camera.reset()
}
</script>

<template>
  <div class="viewport-controls" @mousedown.stop @dblclick.stop>
    <button type="button" title="Zoom out" @click="stepZoom(1 / 1.5, $event)">−</button>
    <span class="zoom-readout">{{ zoomPercent }}%</span>
    <button type="button" title="Zoom in" @click="stepZoom(1.5, $event)">+</button>
    <button type="button" title="Reset view" class="reset" @click="reset">⟲</button>
  </div>
</template>

<style scoped>
.viewport-controls {
  position: absolute;
  right: 12px;
  bottom: 12px;
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 4px 6px;
  background: rgba(15, 20, 25, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  color: #f5f5f5;
  font: 500 12px Inter, system-ui, sans-serif;
  user-select: none;
  pointer-events: auto;
}
.viewport-controls button {
  width: 24px;
  height: 24px;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: inherit;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
}
.viewport-controls button:hover { background: rgba(255, 255, 255, 0.08); }
.viewport-controls .zoom-readout {
  min-width: 42px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.viewport-controls .reset { font-size: 12px; }
</style>
```

The `@mousedown.stop` on the root + `event.stopPropagation()` inside the handlers prevents HUD clicks from being interpreted as pan-starts on the capture overlay.

**Step 2 — Mount the HUD inside the capture overlay in `VideoPlayer.vue`**

Import it:
```typescript
import ViewportControls from '@/components/ViewportControls.vue'
```

Replace the empty comment inside the `<div class="zoom-capture">` with:
```vue
<ViewportControls :camera="camera" :capture-el="zoomCaptureRef" />
```

**Step 3 — Verify type-check + build**

Run: `npm run type-check && npm run build`
Expected: passes.

**Step 4 — Manual verification**

- Court mode: HUD visible bottom-right.
- `+` zooms toward the center; `−` zooms out; readout updates.
- `⟲` resets to 100%.
- Clicking HUD buttons does NOT start a pan drag.
- Readout stays readable (white text on dark pill) over any part of the court background.

**Step 5 — Commit**

```bash
git add src/components/ViewportControls.vue src/components/VideoPlayer.vue
git commit -m "feat: ViewportControls HUD for zoom readout + reset"
```

---

### Task 6: Full-feature manual verification + polish

**Goal:** Drive the feature end-to-end, confirm persistence + edge cases match the design, and capture any gaps before shipping.

**Step 1 — Run the full checklist against a real processed rally:**

- [ ] Zoom toward a joint: the joint stays under the cursor as zoom changes. (Tests `zoomAt` correctness.)
- [ ] Enable 2+ angle overlays (e.g. `L Elbow`, `R Knee`); at 4× zoom the degree labels are the same screen size as at 1×, still anchored to the joint.
- [ ] Leg-stretch dashed line: dashes stay uniform-length at any zoom; meter label readable.
- [ ] Pause the video, zoom, scrub ±5 frames with arrow keys: skeleton updates under the transform; no ghost frames.
- [ ] Play the video from zoomed-in state: skeleton keeps drawing in the same framing while frames advance.
- [ ] Toggle video → court → video → court: zoom/pan state preserved.
- [ ] Double-click the court area (but not the HUD): resets to 100%.
- [ ] Press `+` on HUD 8 times: caps at 800%, does not go higher.
- [ ] Press `−` on HUD from 100%: does not go below 100%.
- [ ] Pan aggressively to one corner: court doesn't fully disappear; at least 20% of its bbox stays on-screen.
- [ ] Resize the browser window while zoomed: pan re-clamps so court stays visible; scale preserved.
- [ ] Release mouse button outside the browser window during a drag: drag ends cleanly; next click doesn't think we're still dragging.
- [ ] Video-mode: zoom does nothing to the video overlay (feature is court-mode-only per v1 scope); no console errors.

**Step 2 — Bump version in `package.json` (optional, match project pattern)**

Recent commits bump a version tag (`v1.8-alpha`); if that convention is in use check `package.json` "version" and recent commit messages. If yes, bump and commit.

**Step 3 — If everything passes, final commit and/or annotate the branch**

No code change expected — this task is the gate. If a new issue surfaces, file a follow-up task (don't widen scope here).

```bash
# Optional — if the manual pass uncovered a tweak:
git add <files>
git commit -m "polish: <what changed>"
```

---

### Out of scope (explicit YAGNI list)

- Touch / pinch gestures.
- Follow-player tracking (static camera only, per Q2 answer).
- Zoom in real-video mode (deferred per Q1 (c); camera is now in place, so the follow-up port is ~1 task).
- Mini-map inset.
- Keyboard shortcuts for zoom/pan.
- `localStorage` persistence across page reload.
- Unit-test framework for composable math (repo has none; would add scope disproportionate to the feature).
