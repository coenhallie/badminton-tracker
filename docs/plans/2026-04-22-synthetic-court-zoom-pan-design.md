# Synthetic Court View — Zoom & Pan Design

**Date:** 2026-04-22
**Status:** Approved
**Goal:** Let the user zoom and pan within the synthetic court view (`SyntheticCourtView.vue`) to inspect player posture, skeleton bones, and numeric angle overlays up close. Ship as a reusable camera composable so the same controls can later be applied to the real-video skeleton overlay.

## Context

The synthetic view renders the court and the 17-COCO-keypoint skeletons as flat Canvas 2D, mapped from court-meters into video-pixel space via a 2D homography. Backend-computed joint angles (`BodyAngles` in `src/types/analysis.ts:172–182`) are already overlaid as degree labels via `drawAngleArc` (`VideoPlayer.vue:1307–1351`). The data model is strictly 2D — no per-joint `z` is exported from the Modal pipeline or YOLO26 pose, so a real 3D orbit camera is not meaningful without adding a pose-lift stage. Zoom + pan on the existing 2D render is the minimal change that directly solves "see the numbers and skeleton close up."

## Scope

**In scope (v1):**
- Zoom + pan on `SyntheticCourtView.vue`.
- Zoom/pan state persists across video scrubbing, play/pause, and toggling between video ↔ court view (reset only via explicit button or double-click).
- Static camera — user-driven only, no player-follow.
- Extractable `useViewportCamera` composable, ready for a later port to `VideoPlayer.vue`.

**Out of scope (deferred):**
- Touch / pinch gestures.
- Follow-player tracking (auto-center on a selected player).
- Applying zoom/pan to the real-video view (Q1(c) — second phase).
- Mini-map inset, keyboard shortcuts, per-player "frame to joint" shortcuts.
- Any 3D camera behaviour (blocked on absence of per-joint depth; see Rejected Alternatives).

## Architecture

**Single-transform approach.** Because the existing synthetic render is one Canvas 2D context driven by a `render()` function on a `requestAnimationFrame` loop, we add exactly one `ctx.setTransform(scale, 0, 0, scale, tx, ty)` call at the top of `render()`. Every downstream draw (court lines, skeleton bones, angle arcs, degree labels, shuttle marker) is transformed consistently with zero per-drawable changes.

State lives in a new framework-agnostic composable, not in the Vue component, so it can be ported to `VideoPlayer.vue` later without rewriting.

State is hoisted up to `App.vue`'s setup scope so zoom survives the `viewMode` toggle between `'video'` and `'court'` (currently at `App.vue:227`). When the ported version ships on the video view, the same camera instance will drive both.

## Components

### New files

1. **`src/composables/useViewportCamera.ts`** — owns `scale`, `tx`, `ty` refs plus:
   - `reset()` — snap back to `{ scale: 1, tx: 0, ty: 0 }`.
   - `zoomAt(clientX, clientY, deltaScale)` — zoom toward a screen point (keeps the cursor anchored to its world-space point).
   - `panBy(dx, dy)` — incremental pan in screen pixels.
   - `applyToContext(ctx)` — calls `ctx.setTransform(...)` so callers don't need to know the matrix layout.
   - `screenToWorld(x, y)` / `worldToScreen(x, y)` — inverse transforms, needed later for DOM overlays or hit-testing.
   - `pixelSize(value)` — returns `value / scale`. Draw code uses this for font size, line width, and keypoint radii so they stay a constant on-screen pixel size at any zoom level.
   - Clamping: `scale ∈ [1, 8]`; pan clamped so at least 20% of the court bounding box remains on-screen.

2. **`src/components/ViewportControls.vue`** — small HUD overlay, absolute-positioned on the canvas. Shows current zoom % (e.g., "245%") and `−` / `+` / ⟲ buttons. Accepts the camera composable as a prop.

### Modified files

3. **`src/components/SyntheticCourtView.vue`**:
   - Accept the camera instance as a prop (injected from `App.vue`).
   - Attach `wheel`, `mousedown` / `mousemove` / `mouseup`, `dblclick` listeners on the canvas element. `mousemove` and `mouseup` bind to `window` while a drag is active so releasing outside the canvas doesn't leave the drag stuck.
   - Call `camera.applyToContext(ctx)` as the first line of `render()`.
   - Replace hardcoded font sizes, `lineWidth`, and circle radii with `camera.pixelSize(...)` so labels and strokes stay crisp at any zoom.
   - Mount `<ViewportControls>` overlaying the canvas.
   - Attach a `ResizeObserver` to the canvas; on resize, re-clamp pan (scale is preserved).

4. **`src/App.vue`** — instantiate the camera composable and pass it to `SyntheticCourtView`. Ready to pass to `VideoPlayer` later.

5. **`src/components/VideoPlayer.vue`** — **no change in v1**. Touch-points are documented below for the follow-up port: the same `pixelSize` factoring will need to happen at:
   - `VideoPlayer.vue:1447` — skeleton bone `lineWidth`.
   - `VideoPlayer.vue:1459–1465` — keypoint `arc` radius.
   - `VideoPlayer.vue:1349` — angle label `ctx.font` size.

## Data Flow

```
wheel / mousedown+drag / dblclick event on canvas
  → SyntheticCourtView.vue event handler
  → camera.zoomAt / panBy / reset
  → scale / tx / ty refs updated
  → existing rAF render() tick picks up new values
  → ctx.setTransform(scale, 0, 0, scale, tx, ty)      ← single added line
  → existing court + skeleton + angle-arc draw calls run unchanged
  → font / lineWidth / radius values divide by scale via camera.pixelSize()
```

The render loop is already framerate-capped by `requestAnimationFrame`; zoom math is O(1) per frame and adds no measurable cost.

## Interaction Model

- **Mouse wheel / trackpad two-finger scroll** → zoom toward cursor. Delta mapped so a full notch is ~1.15× scale.
- **Left-drag** → pan. Drag starts on `mousedown` over the canvas; `mousemove` / `mouseup` listeners on `window` so release anywhere ends the drag cleanly.
- **Double-click** → reset to fit (`scale: 1, tx: 0, ty: 0`).
- **HUD buttons** — `−` zoom out 1 step, `+` zoom in 1 step (around canvas center), ⟲ reset. Zoom % readout is read-only text.

No cursor change during drag in v1 (defer to polish).

## Bounds & Clamping

- **Scale:** `[1, 8]`. No zoom-out past the fit baseline; 8× is enough to read a "45°" label clearly on a 1080p canvas.
- **Pan:** clamped so at least 20% of the court's axis-aligned bounding box in screen space remains inside the canvas viewport. Prevents the user from panning the court fully off-screen and getting lost.
- **On canvas resize:** scale is preserved; pan is re-clamped against the new viewport dims. The fit baseline (scale = 1) recomputes from the new canvas dims automatically since it's defined in screen units.

## Persistence

- Composable state lives in `App.vue`, not inside `SyntheticCourtView.vue`. So:
  - Scrubbing / play / pause — no change, same mounted component.
  - Toggle court → video — `SyntheticCourtView` unmounts, camera state survives in parent.
  - Toggle back video → court — `SyntheticCourtView` remounts, reads the still-live camera, draws at the prior zoom/pan.
- Not persisted across page reload (YAGNI; easy to add later by serialising to `localStorage`).

## Edge Cases

- **Drag released off-canvas** — `mouseup` on `window`, not on the canvas. Drag ends cleanly.
- **Window resize while zoomed** — `ResizeObserver` on the canvas wrapper; on each tick, re-clamp pan against new dims.
- **Zoom during playback** — no special handling; render loop runs every frame anyway, camera just reads current refs.
- **Pre-existing `ctx.save()/restore()` usage** inside `SyntheticCourtView.vue` (e.g., the net stroke at lines 175–209) — unaffected. Our `setTransform` runs once at top; inner `save/restore` blocks still stack on top of it correctly.
- **Text legibility at extreme zoom** — addressed by `camera.pixelSize()`. A "45°" label rendered with `ctx.font = \`${12 / scale}px sans-serif\`` stays at 12 on-screen pixels regardless of zoom, so the label itself stays readable while the underlying bone it's attached to grows.

## Testing

**Unit (Vitest) — `useViewportCamera`:**
- `screenToWorld(worldToScreen(p)) ≈ p` within 1e-9 at various `(scale, tx, ty)` triples.
- `zoomAt(cx, cy, delta)` keeps the world-space point under `(cx, cy)` invariant after the zoom.
- `scale` clamps at both ends of `[1, 8]`.
- Pan clamping respects the 20%-visible-bbox rule.
- `pixelSize(12)` returns `12 / scale` across a range of scales.

**Manual verification — loaded processed rally:**
- Zoom into a player's arm mid-smash; scrub ±10 frames; confirm labels stay crisp and the same pixel size; bones don't balloon into fat bars; keypoint dots stay tight.
- Reset button returns to fit exactly.
- Toggle video → court → video → court; zoom state preserved.
- Drag release off the canvas doesn't leave the drag stuck.
- Resize the window while zoomed; court stays partially visible, no crash.

## Rejected Alternatives

- **True 3D orbit.** Requires per-joint depth. Options (MotionBERT / VideoPose3D pose-lift on the Modal pipeline, or monocular depth via MiDaS / Depth-Anything) are a backend-side feature in their own right. Not rejected permanently — deferred behind this 2D change, which covers the user's stated need ("see the numbers and degrees close up").
- **Fake 3D (tilting the court plane).** Rotating the homography so the court tilts in a virtual 3D space is visually fancy but unhelpful for posture analysis: the skeletons remain flat card-cutouts glued to the floor plane, so you gain no new information about arm angles or torso lean from another viewpoint.
- **CSS transforms on the canvas element instead of canvas-internal transform.** Would re-sample the bitmap and blur at high zoom; canvas-internal `setTransform` re-rasterises at native resolution and stays sharp.
- **Per-drawable scale compensation instead of a shared `pixelSize` helper.** More repetition, more places to forget. Helper keeps it one line per draw call.
- **Persisting across reload via `localStorage`.** YAGNI for v1; trivial to add once we know users actually want it.
