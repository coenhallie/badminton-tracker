# Auto-Pause Between Shots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a per-shot auto-pause feature — when enabled, the video pauses for 1/1.5/2/3 seconds after each detected shot and overlays a card showing the preceding movement segment's speed, distance, duration, and peak body-mechanics (leg stretch in meters, knee flex in degrees). Updates on the next shot.

**Architecture:** Hoist existing shot detection + segment logic out of `ShotSpeedList.vue` into a shared composable so both the list view and the auto-pause watcher consume the same pipeline. Extract a standalone `bodyAngles.ts` helper for leg-stretch (ankle-to-ankle in meters via homography) — the only piece that isn't already on `player.pose.body_angles`. Mirror the exact rally-pause state-machine pattern already in `App.vue` so the two features stay symmetric. Add one new overlay component and one new control row in `RallyTimeline.vue`. No backend, Convex, or Modal changes.

**Tech Stack:** Vue 3 `<script setup>` + TypeScript, existing `@/utils/homography`, existing `SkeletonFrame`/`BodyAngles`/`FramePlayer` types.

**Design doc:** `docs/plans/2026-04-18-auto-pause-between-shots-design.md`

---

### Critical pre-task discoveries (read before starting)

I inspected the actual code — several details in the design doc need small corrections that are baked into this plan:

1. **`leg_stretch` is NOT an angle on `BodyAngles`.** The `BodyAngles` interface (`src/types/analysis.ts:131`) only has: `left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`, `left_knee`, `right_knee`, `left_hip`, `right_hip`, `torso_lean`. There is NO `leg_stretch` field.
   
   In `VideoPlayer.vue:1692` (`drawLegStretch`), leg stretch is computed on-demand from raw keypoints: it's the **distance in METERS between the two ankles** via homography. So the overlay will display leg stretch as a distance in meters, not degrees. Example: `1.42 m`.

2. **Homography is required for leg stretch.** The ankle-to-meters conversion uses `applyHomography(H, x, y)` where `H` is computed from the manual court keypoints. If no manual keypoints are set, leg stretch displays "—" and the rest of the card still renders.

3. **Knee flex.** Use `player.pose?.body_angles?.left_knee` and `right_knee` — already per-frame on the data. For "peak knee flex" take the *minimum* (smaller knee angle = deeper bend = bigger flex) across the segment's frames for the moving player.

4. **`detectedRallies` already lives in `App.vue`** as a reactive from `useAdvancedAnalytics` — use it directly for the "last shot of rally" suppression guard.

5. **Verification primitives** — no unit-test framework in this repo. Each task ends with `npm run type-check` + manual browser check. Pre-existing errors to ignore: `convex/http.ts:1393`, `convex/videos.ts:383`, `convex/videos.ts:394`, `src/App.vue:348,351`, `src/components/VideoUpload.vue:191`, `src/composables/useVideoExport.ts:131,142`.

---

### Task 1: Extract body-angle helpers to `src/utils/bodyAngles.ts`

**Goal:** Pull the leg-stretch computation out of `VideoPlayer.vue` into a pure helper so `useShotSegments` (Task 3) can reuse it without importing a component. This is a refactor with zero behavior change — `VideoPlayer.vue` should continue to render leg stretch identically.

**Files:**
- Create: `src/utils/bodyAngles.ts`
- Modify: `src/components/VideoPlayer.vue:1692-1738` (import + replace `drawLegStretch` internals)

**Step 1 — Create the helper file**

```typescript
// src/utils/bodyAngles.ts
import type { Keypoint } from '@/types/analysis'
import { applyHomography } from '@/utils/homography'

export const KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

// COCO keypoint indices for ankles (matches KP constants in VideoPlayer).
const LEFT_ANKLE = 15
const RIGHT_ANKLE = 16

/**
 * Leg stretch in METERS — distance between the two ankles in court-plane
 * coordinates via homography. Returns null when either ankle is missing,
 * below the confidence threshold, or the homography projection fails.
 */
export function legStretchMeters(
  keypoints: Keypoint[] | undefined,
  H: number[][] | null,
): number | null {
  if (!keypoints || !H) return null
  const la = keypoints[LEFT_ANKLE]
  const ra = keypoints[RIGHT_ANKLE]
  if (!la?.x || !la?.y || !ra?.x || !ra?.y) return null
  if (la.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      ra.confidence < KEYPOINT_CONFIDENCE_THRESHOLD) return null

  const leftM = applyHomography(H, la.x, la.y)
  const rightM = applyHomography(H, ra.x, ra.y)
  if (!leftM || !rightM) return null

  return Math.hypot(leftM.x - rightM.x, leftM.y - rightM.y)
}

/**
 * "Knee flex" = smaller knee angle (tighter bend). Returns the minimum of
 * left_knee / right_knee from already-computed body_angles, ignoring nulls.
 */
export function kneeFlexDegrees(
  leftKnee: number | null | undefined,
  rightKnee: number | null | undefined,
): number | null {
  const vals = [leftKnee, rightKnee].filter(
    (v): v is number => typeof v === 'number',
  )
  if (vals.length === 0) return null
  return Math.min(...vals)
}
```

**Step 2 — Refactor `VideoPlayer.vue:drawLegStretch` to use the helper**

In `src/components/VideoPlayer.vue`, at the top of `<script setup>`, add to existing imports:

```typescript
import { legStretchMeters } from '@/utils/bodyAngles'
```

Replace the *internals* of `drawLegStretch` (lines 1692–1738) — keep the function signature identical, just delegate the math:

```typescript
function drawLegStretch(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  scaleX: number,
  scaleY: number,
  color: string,
  H: number[][] | null,
) {
  const distMeters = legStretchMeters(keypoints, H)
  if (distMeters == null) return

  const la = keypoints[KP.left_ankle], ra = keypoints[KP.right_ankle]
  if (!la?.x || !la?.y || !ra?.x || !ra?.y) return

  const lax = la.x * scaleX, lay = la.y * scaleY
  const rax = ra.x * scaleX, ray = ra.y * scaleY

  ctx.beginPath()
  ctx.setLineDash([6, 4])
  ctx.moveTo(lax, lay)
  ctx.lineTo(rax, ray)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.globalAlpha = 0.7
  ctx.stroke()
  ctx.setLineDash([])
  ctx.globalAlpha = 1.0

  const mx = (lax + rax) / 2
  const my = (lay + ray) / 2
  const label = `${distMeters.toFixed(2)}m`

  ctx.font = 'bold 12px Inter, system-ui, sans-serif'
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2.5
  ctx.strokeText(label, mx - 15, my - 8)
  ctx.fillText(label, mx - 15, my - 8)
}
```

**Step 3 — Verify**

```bash
npm run type-check
```
Expected: PASS (only the 8 pre-existing errors).

```bash
npm run dev
```
Open a video with manual keypoints, enable "Leg Stretch" angle overlay. Expected: identical rendering to before this refactor (same dashed line, same meters label).

**Step 4 — Commit**

```bash
git add src/utils/bodyAngles.ts src/components/VideoPlayer.vue
git commit -m "refactor: extract legStretchMeters + kneeFlexDegrees helpers"
```

---

### Task 2: Hoist shot detection + segments to `src/composables/useShotSegments.ts`

**Goal:** Move `detectShots`, sub-detectors, and `movementSegments` out of `ShotSpeedList.vue` into a composable so both the list view and the new auto-pause watcher consume the same shot pipeline. Zero behavior change for the existing list.

**Files:**
- Create: `src/composables/useShotSegments.ts`
- Modify: `src/components/ShotSpeedList.vue` — import from composable, delete hoisted code

**Step 1 — Create the composable**

Create `src/composables/useShotSegments.ts`. Structure:

```typescript
import { computed, type Ref } from 'vue'
import type { SkeletonFrame } from '@/types/analysis'
import type { SpeedZone } from '@/types/analysis'
import { getSpeedZone } from '@/types/analysis' // verify the actual import path; if it lives elsewhere move accordingly

// =============================================================================
// TYPES (exported so ShotSpeedList and the new overlay can both use them)
// =============================================================================

export interface ShotEvent {
  frame: number
  timestamp: number
  playerId: number
  shuttlePosition: { x: number; y: number }
  playerPosition: { x: number; y: number }
  detectionMethod: 'shuttle_trajectory' | 'pose_classification' | 'player_movement' | 'speed_peaks'
}

export interface ShotMovementSegment {
  id: number
  startShot: ShotEvent
  endShot: ShotEvent
  movingPlayerId: number
  maxSpeedKmh: number
  avgSpeedKmh: number
  maxSpeedZone: SpeedZone
  distanceCoveredM: number | null
  startFrame: number
  endFrame: number
  startTimestamp: number
  endTimestamp: number
  durationSeconds: number
  speedProfile: number[]
}

// =============================================================================
// SHOT DETECTION (moved verbatim from ShotSpeedList.vue)
// =============================================================================

export function detectShots(frames: SkeletonFrame[]): ShotEvent[] {
  // PASTE the full body of detectShots from ShotSpeedList.vue:110-148 here
}

function detectShotsFromShuttleTrajectory(frames: SkeletonFrame[]): ShotEvent[] {
  // PASTE from ShotSpeedList.vue:149-220
}

function detectShotsFromPoseClassification(frames: SkeletonFrame[]): ShotEvent[] {
  // PASTE from ShotSpeedList.vue:221-280
}

function detectShotsFromPlayerMovement(frames: SkeletonFrame[]): ShotEvent[] {
  // PASTE from ShotSpeedList.vue:281-387
}

function detectShotsFromSpeedPeaks(
  // PASTE from ShotSpeedList.vue:388 through its closing brace
) {
  // ...
}

// =============================================================================
// SEGMENT BUILDING (moved from movementSegments computed in ShotSpeedList.vue:525-597)
// =============================================================================

export function buildMovementSegments(
  shots: ShotEvent[],
  frames: SkeletonFrame[],
): ShotMovementSegment[] {
  if (shots.length < 2) return []

  const segments: ShotMovementSegment[] = []
  const MAX_VALID_SPEED_KMH = 25

  for (let i = 0; i < shots.length - 1; i++) {
    const startShot = shots[i]!
    const endShot = shots[i + 1]!
    const movingPlayerId = endShot.playerId

    const duration = endShot.timestamp - startShot.timestamp
    if (duration < 0.2 || duration > 15) continue

    const speedProfile: number[] = []
    let maxSpeedKmh = 0
    let sumSpeedKmh = 0
    let speedCount = 0

    for (const frame of frames) {
      if (frame.frame < startShot.frame || frame.frame > endShot.frame) continue
      const player = frame.players.find(p => p.player_id === movingPlayerId)
      if (player) {
        const speed = player.current_speed ?? 0
        const validSpeed = speed > MAX_VALID_SPEED_KMH ? 0 : speed
        speedProfile.push(validSpeed)
        if (validSpeed > maxSpeedKmh) maxSpeedKmh = validSpeed
        sumSpeedKmh += validSpeed
        speedCount++
      }
    }

    const avgSpeedKmh = speedCount > 0 ? sumSpeedKmh / speedCount : 0

    segments.push({
      id: i,
      startShot,
      endShot,
      movingPlayerId,
      maxSpeedKmh,
      avgSpeedKmh,
      maxSpeedZone: getSpeedZone(maxSpeedKmh),
      distanceCoveredM: null,
      startFrame: startShot.frame,
      endFrame: endShot.frame,
      startTimestamp: startShot.timestamp,
      endTimestamp: endShot.timestamp,
      durationSeconds: duration,
      speedProfile,
    })
  }

  return segments
}

// =============================================================================
// COMPOSABLE WRAPPER
// =============================================================================

/**
 * Given reactive skeleton data, returns reactive shot events and segments
 * (recomputed only when skeletonData reference changes).
 */
export function useShotSegments(skeletonData: Ref<SkeletonFrame[] | undefined>) {
  const shotEvents = computed<ShotEvent[]>(() => {
    const frames = skeletonData.value
    if (!frames || frames.length === 0) return []
    return detectShots(frames)
  })

  const segments = computed<ShotMovementSegment[]>(() => {
    const frames = skeletonData.value
    if (!frames) return []
    return buildMovementSegments(shotEvents.value, frames)
  })

  return { shotEvents, segments }
}
```

**IMPORTANT — paste verbatim:** when moving `detectShots` and its four sub-detectors from `ShotSpeedList.vue`, copy the code BYTE-FOR-BYTE. Zero re-writes. The hoisted code must produce identical output; any behavior change here is a regression.

Also verify the exact import of `getSpeedZone` / `SpeedZone` — search the codebase:
```bash
grep -n "export.*getSpeedZone\|export.*SpeedZone" src/types/analysis.ts src/composables/*.ts src/utils/*.ts
```
Use the actual export location found. If `getSpeedZone` is defined inside `ShotSpeedList.vue` rather than exported from a shared file, move it into `src/utils/speedZones.ts` first as a prerequisite mini-step; otherwise import directly from wherever it already lives.

**Step 2 — Refactor `ShotSpeedList.vue` to consume the composable**

In `src/components/ShotSpeedList.vue`:

- Replace the local `ShotEvent` and `ShotMovementSegment` interfaces (lines 36–67) with an import from the composable:
  ```typescript
  import { useShotSegments, type ShotEvent, type ShotMovementSegment } from '@/composables/useShotSegments'
  ```
- Delete the `detectShots` function and its four sub-detectors (lines 110–524).
- Replace `shotEvents` and `movementSegments` computeds with the composable hookup:
  ```typescript
  const skeletonDataRef = computed(() => props.skeletonData)
  const { shotEvents, segments: movementSegments } = useShotSegments(skeletonDataRef)
  ```
  (Naming kept as `movementSegments` so the rest of the file — `filteredSegments`, `summaryStats`, etc. — needs no changes.)

**Step 3 — Verify**

```bash
npm run type-check
```
Expected: PASS.

```bash
npm run dev
```
Open the Shot Speed List view for a processed video. Expected: the list looks identical to before (same shots, same segments, same ordering, same max/avg speeds). This is the regression check for the refactor.

**Step 4 — Commit**

```bash
git add src/composables/useShotSegments.ts src/components/ShotSpeedList.vue
git commit -m "refactor: hoist shot detection + segments into useShotSegments composable"
```

If `getSpeedZone` had to be extracted to its own file as a prerequisite, include that file in the commit.

---

### Task 3: Extend `useShotSegments` with body-angle aggregation

**Goal:** Add a per-segment `bodyAnglePeaks` computation so the overlay card has the peak leg-stretch and knee-flex values ready to display without doing work at pause-time.

**Files:**
- Modify: `src/composables/useShotSegments.ts`

**Step 1 — Add the aggregation**

Add to `src/composables/useShotSegments.ts`:

```typescript
import type { SkeletonFrame, Keypoint } from '@/types/analysis'
import { legStretchMeters, kneeFlexDegrees } from '@/utils/bodyAngles'

export interface BodyAnglePeaks {
  peakLegStretchM: number | null   // Max ankle-to-ankle distance in meters
  peakKneeFlexDeg: number | null   // Minimum knee angle (deepest bend)
  peakTorsoLeanDeg: number | null  // Max absolute torso-lean angle
}

/**
 * Walk the frames of a single segment for the moving player and compute
 * peak body-mechanics values. H is the meters→pixels inverse homography
 * (meters-output direction); when null, leg stretch cannot be computed.
 *
 * Actually — for legStretchMeters we need pixels→meters. The caller supplies
 * the homography computed via computeHomographyFromKeypoints(), which maps
 * video-pixels → court-meters. Pass that matrix as H.
 */
export function aggregateBodyAngles(
  segment: ShotMovementSegment,
  frames: SkeletonFrame[],
  H: number[][] | null,
): BodyAnglePeaks {
  let peakLeg: number | null = null
  let peakKnee: number | null = null
  let peakTorso: number | null = null

  for (const frame of frames) {
    if (frame.frame < segment.startFrame || frame.frame > segment.endFrame) continue
    const player = frame.players.find(p => p.player_id === segment.movingPlayerId)
    if (!player) continue

    // Leg stretch (meters) — requires homography.
    const leg = legStretchMeters(player.keypoints as Keypoint[], H)
    if (leg != null && (peakLeg == null || leg > peakLeg)) peakLeg = leg

    // Knee flex (smaller angle = deeper bend).
    const knee = kneeFlexDegrees(
      player.pose?.body_angles?.left_knee,
      player.pose?.body_angles?.right_knee,
    )
    if (knee != null && (peakKnee == null || knee < peakKnee)) peakKnee = knee

    // Torso lean (max absolute value).
    const torso = player.pose?.body_angles?.torso_lean
    if (typeof torso === 'number') {
      const absT = Math.abs(torso)
      if (peakTorso == null || absT > peakTorso) peakTorso = absT
    }
  }

  return {
    peakLegStretchM: peakLeg,
    peakKneeFlexDeg: peakKnee,
    peakTorsoLeanDeg: peakTorso,
  }
}

/**
 * Full per-segment bundle that the overlay consumes.
 */
export interface ShotMovementSegmentWithPeaks extends ShotMovementSegment {
  peaks: BodyAnglePeaks
}
```

Extend `useShotSegments` to accept an optional reactive homography and attach peaks:

```typescript
export function useShotSegments(
  skeletonData: Ref<SkeletonFrame[] | undefined>,
  homography?: Ref<number[][] | null>,
) {
  const shotEvents = computed<ShotEvent[]>(() => { /* unchanged */ })

  const segments = computed<ShotMovementSegmentWithPeaks[]>(() => {
    const frames = skeletonData.value
    if (!frames) return []
    const H = homography?.value ?? null
    const base = buildMovementSegments(shotEvents.value, frames)
    return base.map(seg => ({ ...seg, peaks: aggregateBodyAngles(seg, frames, H) }))
  })

  return { shotEvents, segments }
}
```

**Step 2 — Verify the refactor still compiles**

```bash
npm run type-check
```
Expected: PASS. `ShotSpeedList.vue` now receives `ShotMovementSegmentWithPeaks[]` instead of `ShotMovementSegment[]`, but since `ShotMovementSegmentWithPeaks extends ShotMovementSegment`, all existing usages still typecheck. The ShotSpeedList view just ignores the extra `peaks` field — that's fine.

Open the Shot Speed List view to confirm still unchanged visually.

**Step 3 — Commit**

```bash
git add src/composables/useShotSegments.ts
git commit -m "feat: aggregate per-segment body-angle peaks in useShotSegments"
```

---

### Task 4: Create `ShotSummaryOverlay.vue`

**Goal:** New presentational component that renders the bottom-center summary card. No state, no timers — pure props-in, pixels-out. The card reads `ShotMovementSegmentWithPeaks` + a countdown and displays the fields.

**Files:**
- Create: `src/components/ShotSummaryOverlay.vue`

**Step 1 — Create the component**

```vue
<!-- src/components/ShotSummaryOverlay.vue -->
<script setup lang="ts">
import type { ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'

const props = defineProps<{
  segment: ShotMovementSegmentWithPeaks
  countdownSec: number
}>()

function fmtNumber(v: number | null, digits: number, unit: string): string {
  if (v == null || !isFinite(v)) return '—'
  return `${v.toFixed(digits)} ${unit}`
}
</script>

<template>
  <div class="shot-summary-overlay">
    <div class="shot-summary-header">
      <span class="shot-summary-player">Player {{ segment.movingPlayerId + 1 }} responded</span>
      <span class="shot-summary-countdown">⏱ {{ countdownSec.toFixed(1) }}s</span>
    </div>

    <div class="shot-summary-grid">
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Max speed</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.maxSpeedKmh, 1, 'km/h') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Avg speed</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.avgSpeedKmh, 1, 'km/h') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Distance</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.distanceCoveredM, 2, 'm') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Duration</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.durationSeconds, 2, 's') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Leg stretch</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.peaks.peakLegStretchM, 2, 'm') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Knee flex</span>
        <span class="shot-summary-value">{{ segment.peaks.peakKneeFlexDeg != null ? Math.round(segment.peaks.peakKneeFlexDeg) + '°' : '—' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.shot-summary-overlay {
  position: absolute;
  left: 50%;
  bottom: 64px;
  transform: translateX(-50%);
  width: 320px;
  padding: 12px 14px;
  background: rgba(0, 0, 0, 0.88);
  border: 1px solid var(--color-accent);
  color: #f5f5f5;
  font: 500 12px/1.2 Inter, system-ui, sans-serif;
  /* Above SyntheticCourtView (2), skeleton-canvas (3), PoseOverlay (20),
     controls (25). See VideoPlayer.vue z-index table. */
  z-index: 26;
  pointer-events: none;
}

.shot-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}

.shot-summary-player {
  font-weight: 600;
  font-size: 13px;
}

.shot-summary-countdown {
  font-variant-numeric: tabular-nums;
  color: var(--color-accent);
}

.shot-summary-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 12px;
}

.shot-summary-cell {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.shot-summary-label {
  color: rgba(255, 255, 255, 0.6);
  font-size: 11px;
}

.shot-summary-value {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
</style>
```

**Step 2 — Verify**

```bash
npm run type-check
```
Expected: PASS. Component unused so far — visual verification comes in Task 6.

**Step 3 — Commit**

```bash
git add src/components/ShotSummaryOverlay.vue
git commit -m "feat: add ShotSummaryOverlay presentational component"
```

---

### Task 5: Wire `useShotSegments` + pause state machine into `App.vue`

**Goal:** Call the composable at the App level; mirror the rally-pause state machine for shots; emit the active segment and countdown to `VideoPlayer`. No UI yet — the toggle is added in Task 7.

**Files:**
- Modify: `src/App.vue`

**Step 1 — Instantiate the composable + add state**

In `src/App.vue`, near where `detectedRallies` is set up (around line 131), add:

```typescript
import { useShotSegments, type ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'
import { computeHomographyFromKeypoints } from '@/utils/homography'

// Reactive skeletonData for the composable.
const skeletonDataRef = computed(() => analysisResult.value?.skeleton_data)

// Homography for leg-stretch meters (null when no manual keypoints).
const shotHomography = computed((): number[][] | null => {
  const kp = manualCourtKeypoints.value
  if (!kp) return null
  const videoPts = [
    kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left,
    kp.net_left, kp.net_right,
    kp.service_near_left, kp.service_near_right,
    kp.service_far_left, kp.service_far_right,
    kp.center_near, kp.center_far,
  ]
  return computeHomographyFromKeypoints(videoPts)
})

const { segments: shotSegments } = useShotSegments(skeletonDataRef, shotHomography)
```

Then add the pause state (next to the existing rally-pause state at line 122–123):

```typescript
// Shot auto-pause state
const pauseBetweenShots = ref(false)
const shotPauseDurationSec = ref<1 | 1.5 | 2 | 3>(1.5)
const shotPauseCountdown = ref(0)
const currentShotSegment = ref<ShotMovementSegmentWithPeaks | null>(null)
const lastTriggeredShotTime = ref(-1)
let shotPauseTimer: ReturnType<typeof setInterval> | null = null
```

**Step 2 — Add the watcher (mirrors rally-pause at lines 334–357)**

Immediately after the rally-pause watcher (around line 357), append:

```typescript
// ── Shot auto-pause logic ────────────────────────────────────────────────
// Mirrors the rally-pause watcher's shape exactly.

watch(currentVideoTime, (time) => {
  // Reset tracking on seek-backward (same 1s slack as rally-pause).
  if (time < lastTriggeredShotTime.value - 1) {
    lastTriggeredShotTime.value = -1
  }

  if (!pauseBetweenShots.value) return
  if (shotPauseCountdown.value > 0) return          // already in a pause
  if (rallyPauseCountdown.value > 0) return         // rally-pause has priority
  if (!shotSegments.value.length) return

  for (let i = 0; i < shotSegments.value.length; i++) {
    const seg = shotSegments.value[i]!
    const t = seg.endTimestamp // the "just-happened" shot is endShot of segment i
    if (t <= lastTriggeredShotTime.value) continue
    if (time < t || time >= t + 0.5) continue       // only within 0.5s window

    // Suppress if this is the last shot of its rally AND rally-pause is on.
    if (pauseBetweenRallies.value && isLastShotOfRally(seg.endShot.timestamp)) {
      lastTriggeredShotTime.value = t
      continue
    }

    lastTriggeredShotTime.value = t
    currentShotSegment.value = seg
    startShotPause()
    break
  }
})

function isLastShotOfRally(shotTimestamp: number): boolean {
  // A shot is the "last of its rally" if the next shot (if any) starts after
  // the end of the rally that contains this shot, or if there is no next shot.
  const segs = shotSegments.value
  const idx = segs.findIndex(s => s.endShot.timestamp === shotTimestamp)
  if (idx === -1) return false
  const nextShot = segs[idx + 1]?.endShot
  const rally = detectedRallies.value.find(
    r => shotTimestamp >= r.startTimestamp && shotTimestamp <= r.endTimestamp,
  )
  if (!rally) return false
  if (!nextShot) return true
  return nextShot.timestamp > rally.endTimestamp
}

function startShotPause() {
  videoPlayerRef.value?.pause()
  shotPauseCountdown.value = shotPauseDurationSec.value

  // Tick every 100ms so the countdown displays fractional seconds smoothly.
  shotPauseTimer = setInterval(() => {
    shotPauseCountdown.value = Math.max(0, shotPauseCountdown.value - 0.1)
    if (shotPauseCountdown.value <= 0) {
      endShotPause()
    }
  }, 100)
}

function endShotPause() {
  clearShotPause()
  videoPlayerRef.value?.play()
}

function clearShotPause() {
  if (shotPauseTimer) {
    clearInterval(shotPauseTimer)
    shotPauseTimer = null
  }
  shotPauseCountdown.value = 0
  currentShotSegment.value = null
}
```

Extend the existing `onUnmounted` at line 394–396 to also clear the shot timer:

```typescript
onUnmounted(() => {
  clearRallyPause()
  clearShotPause()
})
```

**Step 3 — Pass props to VideoPlayer**

On the `<VideoPlayer>` tag (around line 1568), add:

```html
:shot-summary-segment="currentShotSegment"
:shot-summary-countdown="shotPauseCountdown"
```

These props will be type-errors until Task 6 defines them. Expected.

**Step 4 — Verify**

```bash
npm run type-check
```
Expected: two new errors about unknown props on `VideoPlayer` (resolved in Task 6). Pre-existing errors unchanged.

**Step 5 — Commit**

```bash
git add src/App.vue
git commit -m "feat: shot-pause state machine + useShotSegments wiring in App.vue"
```

---

### Task 6: Accept overlay props in `VideoPlayer.vue` and mount `ShotSummaryOverlay`

**Goal:** Receive the active segment + countdown; render `ShotSummaryOverlay` when both are present.

**Files:**
- Modify: `src/components/VideoPlayer.vue`

**Step 1 — Import + props**

At the top of `<script setup>`:

```typescript
import ShotSummaryOverlay from './ShotSummaryOverlay.vue'
import type { ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'
```

In `defineProps` (around line 265–285), add:

```typescript
shotSummarySegment?: ShotMovementSegmentWithPeaks | null
shotSummaryCountdown?: number
```

**Step 2 — Mount in template**

Inside `.video-wrapper`, immediately after the `<PoseOverlay>` element (around line 2175), add:

```html
<ShotSummaryOverlay
  v-if="shotSummarySegment && (shotSummaryCountdown ?? 0) > 0"
  :segment="shotSummarySegment"
  :countdown-sec="shotSummaryCountdown ?? 0"
/>
```

**Step 3 — Verify**

```bash
npm run type-check
```
Expected: PASS. All props now declared; App.vue errors from Task 5 resolve.

```bash
npm run dev
```
Manual check is still blocked on the UI toggle (Task 7). You can temporarily force-enable by setting `pauseBetweenShots.value = true` in App.vue's setup, play a video, and confirm the overlay appears at each detected shot's timestamp — then revert that override before committing.

**Step 4 — Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "feat: render ShotSummaryOverlay in VideoPlayer during shot pause"
```

---

### Task 7: Add toggle + duration picker in `RallyTimeline.vue`

**Goal:** Expose the new feature in the UI next to the existing rally-pause toggle.

**Files:**
- Modify: `src/components/RallyTimeline.vue` (props + template + style)
- Modify: `src/App.vue` (pass v-models)

**Step 1 — Add props + emits in `RallyTimeline.vue`**

Find the existing `pauseBetweenRallies` prop (line 12) and its emit (line 20). Add siblings:

```typescript
// In defineProps (around line 8-15):
pauseBetweenShots: boolean
shotPauseDurationSec: 1 | 1.5 | 2 | 3

// In defineEmits (around line 17-22):
'update:pauseBetweenShots': [value: boolean]
'update:shotPauseDurationSec': [value: 1 | 1.5 | 2 | 3]
```

**Step 2 — Add the control row in the template**

Immediately after the existing `<label class="rally-tl-autopause">` block (lines 186–194), add:

```html
<label class="rally-tl-autopause" :title="pauseBetweenShots ? 'Auto-pause between shots is ON' : 'Auto-pause between shots is OFF'">
  <input
    type="checkbox"
    :checked="pauseBetweenShots"
    @change="emit('update:pauseBetweenShots', ($event.target as HTMLInputElement).checked)"
  />
  <span class="rally-tl-autopause-slider" />
  <span class="rally-tl-autopause-label">Pause Between Shots</span>
</label>

<select
  v-if="pauseBetweenShots"
  class="rally-tl-shot-duration"
  :value="shotPauseDurationSec"
  @change="emit('update:shotPauseDurationSec', Number(($event.target as HTMLSelectElement).value) as 1 | 1.5 | 2 | 3)"
  :title="'Pause duration after each shot'"
>
  <option :value="1">1s</option>
  <option :value="1.5">1.5s</option>
  <option :value="2">2s</option>
  <option :value="3">3s</option>
</select>
```

Add a minimal style rule in the `<style scoped>` block (match the visual weight of the existing toggle):

```css
.rally-tl-shot-duration {
  padding: 2px 6px;
  margin-left: 6px;
  background: rgba(255, 255, 255, 0.06);
  color: inherit;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}
```

**Step 3 — Bind v-models from `App.vue`**

On the `<RallyTimeline ... />` (around line 1621–1632), next to the existing rally-pause bindings, add:

```html
:pause-between-shots="pauseBetweenShots"
@update:pause-between-shots="pauseBetweenShots = $event"
:shot-pause-duration-sec="shotPauseDurationSec"
@update:shot-pause-duration-sec="shotPauseDurationSec = $event"
```

**Step 4 — Verify**

```bash
npm run type-check
```
Expected: PASS.

```bash
npm run dev
```
Full manual smoke test:
1. Open a processed video with manual keypoints set.
2. Enable "Pause Between Shots". Verify the duration dropdown appears.
3. Play the video. At each mid-rally shot, the video should pause ~1.5s with the overlay visible. Overlay should show player, speeds, distance, duration, leg stretch in meters, knee flex in degrees. The countdown badge should tick down.
4. Let the pause complete — video auto-resumes.
5. Change duration to 3s → next pause should be longer.
6. Enable "Pause Between Rallies" AS WELL. At the last shot of a rally, the shot-pause should SKIP and the rally-pause should fire at the rally's end. No double-pause.
7. Click play during a shot-pause countdown — should resume immediately (video was paused; clicking play in the control bar ends it).
8. Seek backward across a shot you already saw — the pause should fire again when you re-cross.
9. Toggle to Court view during a pause — overlay should render identically over the synthetic court.
10. Open a video WITHOUT manual keypoints → toggle still works, but "Leg stretch" shows "—" (no homography); other fields populate normally.

**Step 5 — Commit**

```bash
git add src/components/RallyTimeline.vue src/App.vue
git commit -m "feat: shot auto-pause toggle + duration picker in RallyTimeline"
```

---

### Task 8: Final verification + branch audit

**Step 1 — Full build + type check**

```bash
npm run type-check && npm run build
```
Expected: both pass, only the 8 pre-existing type errors; no new build warnings beyond the already-known large-chunk warning.

**Step 2 — Scope audit**

```bash
git log --oneline <plan-commit-SHA>..HEAD
git diff --name-only <plan-commit-SHA>..HEAD
```

Changed files should be exactly:
- `docs/plans/2026-04-18-auto-pause-between-shots-design.md` (existing)
- `docs/plans/2026-04-18-auto-pause-between-shots-plan.md` (this file)
- `src/App.vue`
- `src/components/VideoPlayer.vue`
- `src/components/ShotSpeedList.vue`
- `src/components/RallyTimeline.vue`
- `src/components/ShotSummaryOverlay.vue` (new)
- `src/composables/useShotSegments.ts` (new)
- `src/utils/bodyAngles.ts` (new)
- possibly `src/utils/speedZones.ts` (only if `getSpeedZone` had to be relocated as a Task-2 prereq)

Nothing should appear in `backend/`, `convex/`, or unrelated components (`MiniCourt.vue`, `PoseOverlay.vue`, `SyntheticCourtView.vue`, etc.).

**Step 3 — Final manual pass**

Play a full match-length video end-to-end with both toggles on. Confirm:
- No unhandled exceptions in the console.
- Pauses fire rhythmically at shots; no stuck pauses.
- Seeking around during playback doesn't queue ghost pauses.
- Countdown display is readable (smooth tick at 100ms resolution).

---

## Out of scope (do NOT attempt in this plan)

- Per-player split overlay.
- Configurable metric selection (always the same six fields).
- Shot-type / strike-mechanics display for the hitter.
- Exporting per-shot aggregate CSVs.
- Keyboard shortcut for the toggle.
- Unifying rally-pause and shot-pause into a single dropdown.

Each can be a follow-up once v1 is in users' hands.
