# Shot/Rally Detection Unification + Accuracy Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the triplicate shot-detection code across `useShotSegments.ts`, `useAdvancedAnalytics.ts`, and `backend/rally_detection.py`; fix the accuracy issues that make rally detection miss short 2-shot rallies and reject real play as "replays"; align client and backend thresholds so they produce comparable results.

**Architecture:** Extract one canonical TypeScript shot-detection module (`src/utils/shotDetection.ts`) used by both client composables. Python mirrors it algorithmically. Both composables keep their composable APIs unchanged; internally they delegate to the shared module with different option objects (pause-use-case vs rally-use-case).

**Tech Stack:** Vue 3 + TypeScript (client), Python (backend). No framework additions.

---

### Pre-task context

Conversation so far established:
- `useShotSegments.ts:detectShots` — detector used by auto-pause and ShotSpeedList. Has Layers B (wrist proximity) + C (tight thresholds: `cosAngle < 0`, `accelMag > 200 px/s²`, `speed > 80 px/s`, `gap > 0.5s`).
- `useAdvancedAnalytics.ts:detectAllShots` + `detectShotsFromShuttle` + `detectShotsFromPose` + `mergeShots` — detector used by client-side rally detection. Has outlier rejection, camera-angle-aware thresholds, pose fallback, but NO wrist gate and NO stride subsampling.
- `backend/rally_detection.py:_detect_shots` — Python mirror. Has stride subsampling (TrackNet data is dense); no wrist gate; otherwise aligned with client rally detector.

User-reported issues:
1. Client-side rally detector returns far fewer rallies than backend on the same video.
2. Both miss 2-shot rallies entirely (serve + return → ~1.3s total → fails `MIN_RALLY_DURATION_S = 2.0`).
3. Real play is sometimes rejected by `isRealGameplay` (≥50% two-player frames) when pose detection drops the far player.

Quantified divergences (overhead camera):
- Client `MIN_SPEED_SQ = 30² = 900`; backend `15² = 225` — client rejects 4× more candidates as jitter.
- Client has no stride subsampling on dense TrackNet data; backend uses 0.3s stride.
- Client has `isRealGameplay ≥ 50%` guard; backend has none.
- Both have `MIN_RALLY_DURATION_S = 2.0s`.

Pre-existing `npm run type-check` errors to IGNORE throughout: `convex/http.ts:1393`, `convex/videos.ts:383`, `convex/videos.ts:394`, `src/App.vue:~377,~380`, `src/components/VideoUpload.vue:191`, `src/composables/useVideoExport.ts:131,142`. No test framework in the repo — use `npm run type-check` + `npm run build` + manual browser check.

---

### Task 1: Extract `src/utils/shotDetection.ts` — shared primitives

**Goal:** Create a single source of truth for shuttle-trajectory shot detection and pose-fallback shot detection, consumable by both composables. No callers touched yet.

**Files:**
- Create: `src/utils/shotDetection.ts`

**Step 1 — Read the existing implementations FIRST**

Before writing anything, read both files completely:
- `src/composables/useShotSegments.ts` (focus: `detectShots`, `detectShotsFromShuttleTrajectory`, `detectShotsFromPoseClassification`, `detectShotsFromPlayerMovement`, `detectShotsFromSpeedPeaks`, `smoothPositions`, `findClosestPlayer`, `nearestWristMeters`)
- `src/composables/useAdvancedAnalytics.ts` (focus: `detectAllShots`, `detectShotsFromShuttle`, `detectShotsFromPose`, `mergeShots`)
- `backend/rally_detection.py` (focus: `_detect_shots`, for algorithmic parity — do not try to port Python into TS, just understand the algorithm)

You need to hold both detectors in your head before you can unify them well. Do not skip this.

**Step 2 — Design the shared API**

`src/utils/shotDetection.ts` exports:

```typescript
import type { SkeletonFrame, FramePlayer, Keypoint } from '@/types/analysis'
import { applyHomography } from '@/utils/homography'

export type DetectionMethod =
  | 'shuttle_trajectory'
  | 'pose_classification'
  | 'player_movement'
  | 'speed_peaks'

export interface ShotEvent {
  frame: number
  timestamp: number
  playerId: number
  shuttlePosition: { x: number; y: number } | null
  playerPosition: { x: number; y: number }
  detectionMethod: DetectionMethod
  // Optional, populated when the fallback pose detector assigns a class.
  shotType?: string | null
}

// -----------------------------------------------------------------------------
// SHUTTLE TRAJECTORY DETECTOR
// -----------------------------------------------------------------------------

export interface DetectShuttleShotsOptions {
  /** Source frame rate. Required for minimum-gap + stride calculations. */
  fps: number
  /** Camera preset. Switches default thresholds. */
  cameraAngle?: 'overhead' | 'corner'
  /** Optional video-pixels → court-meters matrix. Needed for wrist gate. */
  homography?: number[][] | null

  // --- Core thresholds (defaults are camera-angle aware) ----------------------
  /** Minimum seconds between accepted shots. */
  minShotGapSec?: number
  /** Squared-pixel velocity threshold for "real movement, not jitter". */
  minSpeedSq?: number
  /**
   * Direction-reversal strictness.
   *   0  → any reversal (>90°) — loose
   *  -0.25 → >~105°  — corner default
   *  -0.5 → >~120°  — very strict
   * Interpreted as cos-angle threshold.
   */
  cosAngleMax?: number

  // --- Optional quality gates (off = null) -----------------------------------
  /** Pixel-s⁻² acceleration magnitude minimum. Off by default. */
  minAccelMagPx?: number | null
  /** Require shuttle within N meters of some player's wrist. Needs homography. */
  wristProximityMeters?: number | null

  // --- Pre-processing --------------------------------------------------------
  /** Drop single-frame position outliers before velocity calc. Default true. */
  rejectOutliers?: boolean
  /**
   * Subsample positions every N seconds so velocity vectors span enough
   * displacement. Critical for dense TrackNet data where consecutive-frame
   * reversals are too gradual to detect. Pass 'auto' to enable only when
   * shuttle-frame coverage > 50 %.
   */
  strideSec?: number | 'auto' | null
}

export function detectShuttleShots(
  frames: SkeletonFrame[],
  opts: DetectShuttleShotsOptions,
): ShotEvent[] {
  // 1. Collect visible shuttle positions
  // 2. Optional: reject single-frame outliers
  // 3. Optional: stride-subsample
  // 4. Resolve camera-angle defaults (see table below)
  // 5. Iterate consecutive (p0, p1, p2) triples:
  //    - skip if either velocity is sub-threshold
  //    - compute dot product; reject unless cos-angle < cosAngleMax
  //    - if minAccelMagPx set, compute |dv| and gate
  //    - if wristProximityMeters + homography set, gate by nearestWristMeters()
  //    - assign to closest player (by center distance)
  //    - push ShotEvent
  // 6. Return in timestamp order.
}
```

Camera-angle defaults — resolve these when caller omits:

| Option | `overhead` default | `corner` default |
|---|---|---|
| `minShotGapSec` | `0.6` | `0.8` |
| `minSpeedSq` | `225` (was 30² client / 15² backend — unified on 15² = 225) | `900` (30²) |
| `cosAngleMax` | `0` (any reversal) | `-0.25` |
| `strideSec` (when `'auto'`) | `0.3` if shuttle coverage > 50%, else `null` | `0.5` if shuttle coverage > 50%, else `null` |

**Step 3 — Helpers**

Extract these helpers into the same file (all currently duplicated):

```typescript
// COCO keypoint indices
export const LEFT_WRIST_KP = 9
export const RIGHT_WRIST_KP = 10
const WRIST_CONFIDENCE_THRESHOLD = 0.3

function nearestWristMeters(
  players: FramePlayer[],
  shuttleX: number,
  shuttleY: number,
  H: number[][],
): number | null {
  // Same logic as current useShotSegments.ts:nearestWristMeters
}

function findClosestPlayer(
  players: FramePlayer[],
  x: number,
  y: number,
): FramePlayer | null {
  // Same logic as current implementations
}

// Single-frame outlier: position jumps far from BOTH predecessor and successor.
function rejectPositionOutliers(
  positions: ReadonlyArray<{ frame: number; ts: number; x: number; y: number }>,
): typeof positions {
  // Same logic as useAdvancedAnalytics.ts (OUTLIER_DIST_SQ = 400 * 400)
}

function strideSubsample<T extends { frame: number }>(
  positions: ReadonlyArray<T>,
  strideFrames: number,
): T[] {
  // Same logic as backend/rally_detection.py:_detect_shots stride loop
}
```

**Step 4 — Pose fallback + merge**

```typescript
export interface DetectPoseShotsOptions {
  fps: number
  hittingClasses: Set<string>      // e.g. {'smash','offense','backhand-general','serve','lift'}
  minConfidence?: number            // default 0.5
  minShotGapSec?: number             // default 0.5 (any two shots regardless of player)
  perPlayerGapSec?: number           // default 0.8 (same player re-hit gate)
}

export function detectPoseShots(
  frames: SkeletonFrame[],
  opts: DetectPoseShotsOptions,
): ShotEvent[] {
  // Union of the existing detectShotsFromPoseClassification (useShotSegments)
  // and detectShotsFromPose (useAdvancedAnalytics):
  //   - per-player gap (from useAdvancedAnalytics)
  //   - overall gap (from useShotSegments)
  //   - bbox-aware player match when classification has a bbox (from useShotSegments)
  //   - fill shotType = classification.class_name
  //   - detectionMethod = 'pose_classification'
}

export function mergeShots(shots: ShotEvent[], minGapFrames: number): ShotEvent[] {
  // Sort by frame, drop any shot within minGapFrames of the previous,
  // preferring shuttle-detected shots over pose-detected (shuttlePosition != null).
  // Preserves existing useAdvancedAnalytics.ts:mergeShots semantics.
}
```

**Step 5 — No player-movement / speed-peaks detectors in the shared module**

`detectShotsFromPlayerMovement` and `detectShotsFromSpeedPeaks` (currently in `useShotSegments.ts`) are last-resort fallbacks only relevant when shuttle data is missing. Keep them local to `useShotSegments.ts` for now — they don't participate in the duplication problem and rally detection (`useAdvancedAnalytics.ts`) doesn't use them. Flag this in a header comment so future maintainers know.

**Step 6 — Verify**

```bash
npm run type-check
```
Expected: no new errors (file is defined but not imported anywhere yet).

**Step 7 — Commit**

```bash
git add src/utils/shotDetection.ts
git commit -m "feat: extract shared shot-detection primitives to utils/shotDetection"
```

---

### Task 2: Refactor `useShotSegments.ts` to consume the shared module

**Goal:** Delete the local `detectShotsFromShuttleTrajectory`, `detectShotsFromPoseClassification`, `nearestWristMeters`, `findClosestPlayer`, `smoothPositions`, `LEFT_WRIST/RIGHT_WRIST` constants. Keep `detectShotsFromPlayerMovement` + `detectShotsFromSpeedPeaks` as local fallbacks. The composable's public API stays identical.

**Files:**
- Modify: `src/composables/useShotSegments.ts`

**Step 1 — Replace body of `detectShots`**

```typescript
import { detectShuttleShots, detectPoseShots, mergeShots } from '@/utils/shotDetection'

export function detectShots(
  frames: SkeletonFrame[],
  homography: number[][] | null = null,
  fps = 30,
): ShotEvent[] {
  // 1. Count shuttle/pose frames (same as before, for routing)
  // 2. If shuttle ≥ 5 frames: call detectShuttleShots with pause-use-case options:
  //      { fps, cameraAngle: 'overhead', homography,
  //        minAccelMagPx: 200,
  //        wristProximityMeters: homography ? 3.5 : null,
  //        rejectOutliers: true,
  //        strideSec: 'auto' }
  //    → accept if ≥3 shots
  // 3. Else pose path: call detectPoseShots with existing hitting classes
  // 4. Else local fallbacks: detectShotsFromPlayerMovement / detectShotsFromSpeedPeaks
}
```

Keep the console logging that already exists; rename the `[ShotDetection]` summary line to come out of the *shared module* itself (add a debug option to `DetectShuttleShotsOptions` like `logStats?: boolean`) so both callers get the same instrumentation without reimplementing it.

**Step 2 — Delete dead code**

Remove from `useShotSegments.ts`:
- `detectShotsFromShuttleTrajectory` and its body
- `detectShotsFromPoseClassification` and its body
- `nearestWristMeters`
- `LEFT_WRIST`, `RIGHT_WRIST`, `WRIST_CONFIDENCE_THRESHOLD` constants
- `smoothPositions` (if defined locally)
- `findClosestPlayer` (if not used by the kept detectors)
- The `SHOT_PROXIMITY_METERS` constant (move to shared module as the default)

Keep: `detectShotsFromPlayerMovement`, `detectShotsFromSpeedPeaks`, the `buildMovementSegments`, `aggregateBodyAngles`, the `useShotSegments` composable.

**Step 3 — Verify**

```bash
npm run type-check && npm run build
```
Expected: zero new errors. Diagnostic logging still appears in the console when opening a video.

**Step 4 — Commit**

```bash
git add src/composables/useShotSegments.ts
git commit -m "refactor: useShotSegments delegates shot detection to shared module"
```

---

### Task 3: Refactor `useAdvancedAnalytics.ts` to consume the shared module + apply accuracy fixes

**Goal:** Delete the local shuttle/pose detectors. Apply threshold + gameplay-guard fixes that restore accuracy for short rallies and far-player occlusion.

**Files:**
- Modify: `src/composables/useAdvancedAnalytics.ts`

**Step 1 — Replace `detectAllShots` internals**

```typescript
import { detectShuttleShots, detectPoseShots, mergeShots, type ShotEvent } from '@/utils/shotDetection'

// Inside the composable:
function detectAllShots(frames: SkeletonFrame[], fps: number): RallyShot[] {
  const camera = (cameraAngle?.value === 'corner' ? 'corner' : 'overhead')

  const shuttleShots = detectShuttleShots(frames, {
    fps,
    cameraAngle: camera,
    // rally-use-case: wrist proximity OFF (false negatives hurt more than
    // false positives here — we want to see every real shot for rally bounds).
    wristProximityMeters: null,
    // accel gate OFF — wrist gate + basic speed/angle are enough for rally bounds.
    minAccelMagPx: null,
    rejectOutliers: true,
    strideSec: 'auto',
  })
  if (shuttleShots.length >= 4) {
    return shuttleShots.map(toRallyShot)  // thin mapper, see below
  }

  const poseShots = detectPoseShots(frames, {
    fps,
    hittingClasses: HITTING_POSES,          // existing constant
    minConfidence: 0.65,
    minShotGapSec: Math.max(0.6, camera === 'corner' ? 0.8 : 0.6),
    perPlayerGapSec: 0.8,
  })

  const merged = mergeShots([...shuttleShots, ...poseShots], Math.max(3, Math.floor(fps * 0.6)))
  return merged.map(toRallyShot)
}

function toRallyShot(e: ShotEvent): RallyShot {
  return {
    frame: e.frame,
    timestamp: e.timestamp,
    playerId: e.playerId,
    shotType: e.shotType || 'unknown',
    shuttlePosition: e.shuttlePosition,
    playerPosition: e.playerPosition,
  }
}
```

Delete: `detectShotsFromShuttle`, `detectShotsFromPose`, local `mergeShots`, the OUTLIER_DIST_SQ constant, MAX_GAP_S etc. (all handled inside the shared module now).

**Step 2 — Fix rally-level thresholds + guards in the `rallies` computed**

In the same file, find the `rallies = computed(...)` block and apply:

```typescript
// Was: const MIN_RALLY_DURATION_S = isCorner ? 3.0 : 2.0
const MIN_RALLY_DURATION_S = isCorner ? 1.2 : 0.8
```

Justification in a comment: a serve + return is a legitimate 2-shot rally lasting ~1.3s; 2.0s was silently dropping them. 1.2s for corner (noisier) still catches them.

Remove the `isRealGameplay` call and helper entirely. Rationale — pose detection frequently drops the far player (user's logs show frequent `Players: 1` frames during active play), so this check rejects real rallies. A better replay-rejection signal is shuttle visibility coverage within the rally window: if `<25%` of frames in the rally window have shuttle data, treat as replay and skip. Add that check in place of `isRealGameplay`:

```typescript
function isShuttleActiveWindow(startFrame: number, endFrame: number): boolean {
  let total = 0
  let visible = 0
  for (const f of frames) {
    if (f.frame < startFrame || f.frame > endFrame) continue
    total++
    if (f.shuttle_position?.x != null && f.shuttle_position?.y != null) visible++
  }
  return total === 0 || (visible / total) >= 0.25
}
```

Keep camera-angle-aware `RALLY_GAP_SECONDS` unchanged (3.1s overhead / 4.0s corner).

**Step 3 — Verify**

```bash
npm run type-check && npm run build
```
Zero new errors. Open a video with clearly short rallies — more rallies should now be detected than before.

**Step 4 — Commit**

```bash
git add src/composables/useAdvancedAnalytics.ts
git commit -m "fix: rally detector catches short rallies; delegates shot detection to shared module"
```

---

### Task 4: Align `backend/rally_detection.py` thresholds

**Goal:** Mirror the TS changes so backend and client give comparable rally counts. Algorithm already parallel; just update constants.

**Files:**
- Modify: `backend/rally_detection.py`

**Step 1 — Update default args + overhead branch in `detect_rallies`**

Current signature has `min_rally_duration_s: float = 2.0`. Change default to `0.8`:

```python
def detect_rallies(
    ...
    min_rally_duration_s: float = 0.8,
    ...
):
```

Inside the `if camera_angle == "corner":` branch:

```python
if camera_angle == "corner":
    min_rally_duration_s = max(min_rally_duration_s, 1.2)  # was 3.0
    min_gap_duration_s = max(min_gap_duration_s, 4.0)
    min_shot_gap_s = 0.8
    min_speed_sq = 30.0 * 30.0
    min_shots = 2           # was 3 — match client MIN_SHOTS
    dot_threshold = -0.25
    stride_s = 0.5
else:
    min_shot_gap_s = 0.6
    min_speed_sq = 15.0 * 15.0
    min_shots = 2
    dot_threshold = 0.0
    stride_s = 0.3
```

**Step 2 — Add a header comment documenting the parity contract**

At the top of the file, add:

```python
"""
Rally Detection — shot-gap approach.

Algorithm mirrors src/utils/shotDetection.ts (TS) exactly. Any change to
thresholds or logic here MUST be applied to that module (and
useAdvancedAnalytics.ts:rallies) to keep client and backend rally counts
comparable. Sync keys:

  min_rally_duration_s      ↔  MIN_RALLY_DURATION_S
  min_gap_duration_s        ↔  RALLY_GAP_SECONDS
  min_speed_sq              ↔  DetectShuttleShotsOptions.minSpeedSq
  dot_threshold             ↔  DetectShuttleShotsOptions.cosAngleMax
  stride_s                  ↔  DetectShuttleShotsOptions.strideSec
  min_shots                 ↔  MIN_SHOTS

Diverges intentionally on:
  - No wrist-proximity gate (backend has no pose keypoints at this stage).
  - No player-movement / speed-peaks fallbacks (backend always has
    TrackNet shuttle data; if shuttle is empty, no rallies are detected).
"""
```

**Step 3 — Verify + commit**

Python has no type-check in this project layout. Just confirm the file parses:

```bash
python3 -c "import ast; ast.parse(open('backend/rally_detection.py').read()); print('ok')"
```

```bash
git add backend/rally_detection.py
git commit -m "fix(backend): align rally-detection thresholds with client; accept 2-shot rallies"
```

Deploy is user's call — flag in the handoff that backend changes require `modal deploy backend/modal_convex_processor.py` to take effect.

---

### Task 5: Cleanup — drop diagnostic logs from earlier tuning

**Goal:** The per-gate rejection counters and LayerA logs were for tuning. Now that thresholds are settled, remove them to reduce console noise.

**Files:**
- Modify: `src/utils/shotDetection.ts` (the `logStats` flag — leave the capability, but default off)
- Modify: `src/composables/useShotSegments.ts` (stop passing `logStats: true`)
- Modify: `src/App.vue` (remove the `[ShotDetection/LayerA]` console.log; the kept/dropped logic stays, only the log goes)

Keep the `[ShotSpeedList] Using shuttle trajectory detection: N shots` single line — it's a useful high-level signal.

**Commit:**

```bash
git add -A
git commit -m "chore: drop shot-detection diagnostic logs (tuning done)"
```

---

### Task 6: Final verification

**Step 1 — Full sweep**

```bash
npm run type-check && npm run build
```
Expected: only the 8 pre-existing errors. Build clean.

**Step 2 — Scope audit**

```bash
git log --oneline <plan-commit>..HEAD -- src/ backend/
git diff --name-only <plan-commit>..HEAD
```

Files that SHOULD appear:
- `src/utils/shotDetection.ts` (new)
- `src/composables/useShotSegments.ts` (smaller — lots of deletes)
- `src/composables/useAdvancedAnalytics.ts` (smaller — lots of deletes)
- `backend/rally_detection.py` (constants + header)
- `src/App.vue` (only the diagnostic-log removal)
- `docs/plans/2026-04-20-shot-rally-detection-unification-plan.md` (this file)

Files that SHOULD NOT appear: `convex/`, `src/components/`, anything else.

**Step 3 — Manual browser check**

Run `npm run dev`, open a video that previously had too few rallies:
- Expect more rallies detected, including 2-shot ones.
- Shot-pause modal fires at real shot moments; frequency matches perceived shot tempo.
- `[ShotSpeedList] Using shuttle trajectory detection: N shots` is visible in the console.
- `[ShotDetection]` per-gate counters should NOT be visible anymore (diagnostic removal).

---

## Out of scope (explicitly)

- Changes to `convex/` or the Modal worker pipeline.
- Re-tuning shot-level gates (B/C) — those are already calibrated.
- Re-implementing the player-movement / speed-peaks fallback in the shared module. Deferred; only used when shuttle data is absent, which is rare.
- Python type hints beyond what's already there.
- Any changes to rally *display* code (`RallyTimeline.vue`, etc.).
