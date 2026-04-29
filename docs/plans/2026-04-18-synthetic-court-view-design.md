# Synthetic Court View Design

**Date:** 2026-04-18
**Status:** Approved
**Goal:** Add a view-mode toggle that switches playback between the existing video-with-skeletons view and a new synthetic view where the video is replaced by a line-art badminton court (derived from the manually-added court keypoints), with player skeletons and the shuttlecock drawn on top.

## Context

Users already manually annotate 12 court keypoints per video (`CourtSetup.vue`), and those keypoints are already used to compute a homography matrix in `MiniCourt.vue` for the top-down tactical view. The existing `PoseOverlay.vue` draws per-frame skeletons onto the video. Shuttle positions from TrackNetV3 are available per-frame in the results JSON but are not currently drawn in the main player.

The new view uses those same keypoints (via inverse homography: court-meter → video-pixel) to redraw the court lines in the same camera perspective, so skeletons stay in their original pixel positions with full anatomical detail. This is a presentation-layer change only — no backend or schema changes.

## View Geometry — Camera-perspective synthetic court

Rejected alternatives:
- **2D top-down bird's-eye view** — already covered by `MiniCourt.vue`; flattens skeleton detail onto the ground plane.
- **3D/isometric render** — disproportionate implementation effort for a v1.

Chosen approach keeps the camera perspective and swaps out the video pixels for a synthetic court drawing. Skeletons render in unchanged pixel positions (via the existing `PoseOverlay`), preserving full pose detail.

## Toggle Placement — Mode selector above the player

Two buttons ("Video" / "Court") styled using the existing `.mode-selector` / `.mode-options` classes (same as camera-angle and tracker-type selectors). Lives in the parent (`ResultsDashboard.vue` or wherever `VideoPlayer` is mounted), outside the video control bar. The "Court" button is disabled with an explanatory tooltip when the video has no manual court keypoints.

## Court Detail — Full badminton line set

All lines drawn from the 12-keypoint-derived homography:
- Outer boundary (doubles): `0×0` → `6.1×13.4` (meters)
- Singles sideline
- Short service line (1.98m from net, each side)
- Long service line doubles (0.76m from back boundary, each side)
- Long service line singles (= back boundary)
- Center line (between service lines, excluding net zone)
- Net line (center at 6.7m)

All white, 2px anti-aliased; net line 3px for emphasis.

## Shuttle Rendering — Current position + fading trail

Every render tick, find the shuttle position for the current video frame and draw:
- Fading tail: last 15 frames (~0.5s at 30fps), opacity ramp 0.1 → 1.0
- Current position: 6px white dot with subtle glow
- Skip frames where TrackNet reported occluded (`visible === false`)

Rejected: full-rally arc (too busy during long rallies; better suited to a dedicated shot-review view).

## Components

### New — `src/components/SyntheticCourtView.vue`

Absolutely-positioned full-size `<canvas>` placed above the (hidden) `<video>` and below the `PoseOverlay` canvas. Responsibilities:

1. On mount / keypoints-change, compute inverse homography (court-meter → video-pixel) using `@/utils/homography`.
2. Pre-render court lines to an offscreen canvas once.
3. Every `requestAnimationFrame`, clear the main canvas, blit the offscreen court lines, then draw the shuttle trail + current position based on the current video frame.
4. Fill background with `#0f1419` (matches dark theme).

Props:
- `videoWidth: number`, `videoHeight: number` — for canvas sizing
- `courtKeypoints: ExtendedCourtKeypoints` — the 12 manual keypoints
- `shuttlePositions: ShuttlePosition[]` — per-frame shuttle data
- `currentFrame: number` — reactive current frame index (same source as `PoseOverlay`)
- `fps: number` — for time-based trail length

### Modified — `src/components/VideoPlayer.vue`

New props:
- `viewMode: 'video' | 'court'` (default `'video'`)
- `courtKeypoints?: ExtendedCourtKeypoints`
- `shuttlePositions?: ShuttlePosition[]`

Behavior when `viewMode === 'court'`:
- Hide the `<video>` element visually (`visibility: hidden` — keep it in DOM so playback/seek/audio work).
- Render `SyntheticCourtView` in its place.
- `PoseOverlay` continues rendering unchanged on top.

### Modified — parent (likely `src/App.vue` or `ResultsDashboard.vue`)

- Add `viewMode` reactive state (default `'video'`).
- Add mode selector UI with two buttons ("Video" / "Court"); "Court" disabled + tooltip when keypoints absent.
- Pass `viewMode`, `courtKeypoints`, `shuttlePositions` down to `VideoPlayer`.

## Data Flow

```
videos table (Convex)
  └─ manualCourtKeypoints (existing, already fetched)
  └─ resultsStorageId → JSON
        └─ shuttle_data / skeleton_data / ... (existing)

Parent component
  ├─ reads keypoints + results
  ├─ owns viewMode state + mode selector UI
  └─ passes viewMode + courtKeypoints + shuttlePositions → VideoPlayer

VideoPlayer
  ├─ <video> (always mounted; hidden when court mode)
  ├─ SyntheticCourtView (only when court mode)
  └─ PoseOverlay (unchanged)
```

No changes to backend, Convex schema, or Modal processor.

## Performance

- Court lines drawn once to an offscreen canvas, re-blit each frame — no per-frame geometry math.
- Shuttle trail = 15 draws per frame. Trivial.
- Video element stays mounted and playing (for timing + audio), even when hidden. Browser may still decode frames; acceptable overhead for v1.

## Error Handling

- **Missing keypoints** — toggle disabled at UI level. No runtime branch needed in the component.
- **Degenerate homography** (colinear/duplicate points) — `@/utils/homography` already returns `null`; `SyntheticCourtView` falls back to drawing only outer-boundary endpoints and emits a warning log. Skeletons + shuttle still render in raw pixel positions.
- **Missing shuttle data** (e.g., TrackNet disabled on older videos) — skip shuttle layer silently; court + skeletons still render.

## Testing

- **Unit**: `SyntheticCourtView` with fixed keypoints renders expected court-line endpoints (mock canvas `getContext`, assert path calls).
- **Integration**: toggle between modes during playback — skeletons must remain at identical pixel positions across both modes (visual regression guard).
- **Edge**: video without keypoints shows disabled "Court" button with tooltip. Video with partial shuttle data renders trail with gaps.
- **Manual**: eye-check court line alignment against a well-calibrated video (keypoints visibly match real court lines).

## Non-goals (explicit v1 scope)

- No 2D top-down court view (already covered by `MiniCourt.vue`).
- No tinted service boxes / broadcast polish (v2 if desired).
- No shot-type annotations or rally boundaries on the court view.
- No keyboard shortcut for the toggle (could be trivially added later).
