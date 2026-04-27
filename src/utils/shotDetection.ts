// =============================================================================
// SHARED SHOT-DETECTION PRIMITIVES
// =============================================================================
// Single source of truth for shuttle-trajectory + pose-classification shot
// detection. Consumed by useShotSegments (auto-pause + shot speed list) and
// useAdvancedAnalytics (rally detection). Python mirror in
// backend/rally_detection.py — keep algorithms and threshold defaults aligned.
//
// Algorithmic parity notes:
//   * dot-threshold math matches backend Python `_detect_shots`:
//       if cosAngleMax < 0: threshold = cosAngleMax * sqrt(s1sq * s2sq)
//       else:              threshold = 0
//       reject when dot >= threshold  (i.e. trigger when dot < threshold)
//     Stable against zero-speed edge cases (no division by zero).
//   * minSpeedSq is compared against per-step displacement squared
//     (Δx² + Δy² between sampled points), the same metric backend and
//     useAdvancedAnalytics use. It is NOT a per-second velocity magnitude.
//   * Outlier rejection filters single-frame TrackNet glitches before
//     velocity computation (useAdvancedAnalytics behavior).
//   * MAX_GAP_S (2.5s) guard prevents velocity from being built across
//     inter-rally idle periods.
//   * Stride subsampling ('auto'): when visible-shuttle coverage > 50% of
//     total frames, enable backend-style subsampling by keeping points at
//     least `stride` frames apart; below 50% coverage, use every point.
//
// Extra gates (opt-in, for auto-pause callers that need the stricter Layer
// B + C semantics from useShotSegments):
//   * minAccelMagPx:       reject candidates whose |v2 - v1| is below this
//                          (raw per-step velocity difference magnitude).
//   * wristProximityMeters: reject candidates where the nearest visible
//                           wrist is farther than this many court-meters
//                           from the shuttle. Requires `homography`.
// =============================================================================

import type {
  SkeletonFrame,
  FramePlayer,
  PoseClassificationResult,
} from '@/types/analysis'
import { applyHomography } from '@/utils/homography'

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------
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
  shotType?: string | null
}

// -----------------------------------------------------------------------------
// Constants (exported for reuse)
// -----------------------------------------------------------------------------
const LEFT_WRIST_KP = 9
const RIGHT_WRIST_KP = 10

/** Minimum keypoint confidence for a wrist to be considered visible. */
const WRIST_CONFIDENCE_THRESHOLD = 0.3

/** Single-frame jump threshold (squared pixels) for TrackNet outlier rejection. */
const OUTLIER_DIST_SQ = 400 * 400

/**
 * Max time (seconds) between consecutive sampled shuttle positions before
 * the pair's velocity is treated as meaningless. Prevents inter-rally gaps
 * from producing phantom direction reversals.
 */
const MAX_GAP_S = 2.5

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Find the player whose center is closest to (x, y). Returns null when none. */
function findClosestPlayer(
  players: FramePlayer[],
  x: number,
  y: number,
): FramePlayer | null {
  let closest: FramePlayer | null = null
  let closestDist = Infinity

  for (const player of players) {
    if (!player.center) continue
    const dx = player.center.x - x
    const dy = player.center.y - y
    const d = Math.sqrt(dx * dx + dy * dy)
    if (d < closestDist) {
      closestDist = d
      closest = player
    }
  }

  return closest ?? players[0] ?? null
}

/**
 * Smallest distance (in court meters) between the shuttle and any confident
 * wrist keypoint across the given players. Returns null when the homography
 * is missing, no wrist is visible, or a keypoint cannot be projected.
 */
function nearestWristMeters(
  players: FramePlayer[],
  shuttleX: number,
  shuttleY: number,
  H: number[][] | null,
): number | null {
  if (!H) return null

  const shuttleM = applyHomography(H, shuttleX, shuttleY)
  if (!shuttleM) return null

  let best: number | null = null
  for (const player of players) {
    const kpts = player.keypoints
    if (!kpts) continue
    for (const idx of [LEFT_WRIST_KP, RIGHT_WRIST_KP]) {
      const w = kpts[idx]
      if (!w?.x || !w?.y) continue
      if (w.confidence < WRIST_CONFIDENCE_THRESHOLD) continue
      const wM = applyHomography(H, w.x, w.y)
      if (!wM) continue
      const d = Math.hypot(shuttleM.x - wM.x, shuttleM.y - wM.y)
      if (best == null || d < best) best = d
    }
  }
  return best
}

// -----------------------------------------------------------------------------
// Shuttle trajectory detector
// -----------------------------------------------------------------------------

export interface DetectShuttleShotsOptions {
  /** Video frame rate (frames per second). Required for stride + gap math. */
  fps: number
  /** Forward homography (video pixels → court meters). Optional. */
  homography?: number[][] | null
  /** Minimum gap between consecutive shots in seconds. */
  minShotGapSec?: number
  /**
   * Minimum per-step displacement squared (px²). A candidate pair is rejected
   * if BOTH `|p1-p0|²` and `|p2-p1|²` are below this threshold (pure jitter).
   */
  minSpeedSq?: number
  /**
   * Angle threshold between consecutive velocity vectors.
   *   * >= 0 → require `dot < 0` (any reversal past 90°); value ignored.
   *   * < 0  → require `dot < cosAngleMax * sqrt(s1sq * s2sq)`.
   * Matches backend Python dot_threshold semantics.
   */
  cosAngleMax?: number
  /**
   * Optional. When set, reject candidates whose |v2 - v1| magnitude is below
   * this threshold. Useful for auto-pause to reject smooth gravity-arc flips.
   */
  minAccelMagPx?: number | null
  /**
   * Optional. When set AND `homography` is provided, reject candidates whose
   * nearest visible wrist is farther than this many court-meters from the
   * shuttle at the candidate frame.
   */
  wristProximityMeters?: number | null
  /** Filter single-frame shuttle outliers before velocity. Default: true. */
  rejectOutliers?: boolean
  /**
   * Stride subsampling control (seconds between kept samples).
   *   * number: always subsample with stride = max(3, round(fps * strideSec)).
   *   * 'auto': enable subsampling only when coverage > 50%, using the
   *            overhead-camera default stride of 0.3s.
   *   * null/undefined: no subsampling.
   */
  strideSec?: number | 'auto' | null
}

interface ShuttlePoint {
  frame: number
  timestamp: number
  x: number
  y: number
}

/**
 * Resolve overhead-camera defaults. Only fills values that are `undefined`.
 * The app only supports an overhead camera centered above the court — these
 * thresholds are tuned for that single viewpoint.
 */
function resolveShuttleDefaults(opts: DetectShuttleShotsOptions): {
  minShotGapSec: number
  minSpeedSq: number
  cosAngleMax: number
  autoStrideSec: number
} {
  return {
    minShotGapSec: opts.minShotGapSec ?? 0.6,
    minSpeedSq: opts.minSpeedSq ?? 225,
    cosAngleMax: opts.cosAngleMax ?? 0,
    autoStrideSec: 0.3,
  }
}

/**
 * Filter single-frame shuttle outliers. A position is dropped when its
 * squared distance to BOTH its predecessor and successor exceeds
 * OUTLIER_DIST_SQ (TrackNet glitches mis-locating the shuttle for one
 * frame). First and last points are always kept.
 */
function filterOutliers(points: ShuttlePoint[]): ShuttlePoint[] {
  if (points.length < 3) return points.slice()
  const out: ShuttlePoint[] = [points[0]!]
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1]!
    const curr = points[i]!
    const next = points[i + 1]!
    const dPrev = (curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2
    const dNext = (curr.x - next.x) ** 2 + (curr.y - next.y) ** 2
    if (dPrev > OUTLIER_DIST_SQ && dNext > OUTLIER_DIST_SQ) continue
    out.push(curr)
  }
  out.push(points[points.length - 1]!)
  return out
}

/**
 * Detect shots from shuttle trajectory direction reversals.
 *
 * Implements, in order:
 *   1. Collect visible shuttle positions.
 *   2. Optional outlier rejection (default on).
 *   3. Resolve overhead-camera defaults for undefined options.
 *   4. Optional stride subsampling ('auto' gates on >50% coverage).
 *   5. For each triple (p0, p1, p2):
 *        - skip when both per-step speeds are below `minSpeedSq`.
 *        - skip when MAX_GAP_S is exceeded between p0→p1 or p1→p2.
 *        - compute dot product; compare against threshold per `cosAngleMax`
 *          (backend-compatible formulation).
 *        - optional accel gate (|v2 - v1| < minAccelMagPx).
 *        - enforce min-shot-gap in frames.
 *        - lookup skeleton frame for p1; optional wrist-proximity gate.
 *        - assign to closest player and emit ShotEvent.
 */
export function detectShuttleShots(
  frames: SkeletonFrame[],
  opts: DetectShuttleShotsOptions,
): ShotEvent[] {
  const { fps, homography = null } = opts
  if (!fps || fps <= 0) return []

  const {
    minShotGapSec,
    minSpeedSq,
    cosAngleMax,
    autoStrideSec,
  } = resolveShuttleDefaults(opts)

  const minShotGapFrames = Math.max(3, Math.floor(fps * minShotGapSec))

  // ---------------------------------------------------------------------------
  // 1. Collect visible shuttle positions
  // ---------------------------------------------------------------------------
  const raw: ShuttlePoint[] = []
  for (const f of frames) {
    const s = f.shuttle_position
    if (s && s.x != null && s.y != null) {
      raw.push({ frame: f.frame, timestamp: f.timestamp, x: s.x, y: s.y })
    }
  }

  if (raw.length < 5) return []

  // ---------------------------------------------------------------------------
  // 2. Outlier rejection
  // ---------------------------------------------------------------------------
  const rejectOutliers = opts.rejectOutliers ?? true
  const cleaned = rejectOutliers ? filterOutliers(raw) : raw

  // ---------------------------------------------------------------------------
  // 3 + 4. Stride subsampling
  // ---------------------------------------------------------------------------
  let samples: ShuttlePoint[] = cleaned
  let strideFrames = 0

  const strideSec = opts.strideSec
  if (strideSec === 'auto') {
    const totalFrames = frames.length || 1
    const coverage = raw.length / totalFrames
    if (coverage > 0.5) {
      strideFrames = Math.max(3, Math.round(fps * autoStrideSec))
    }
  } else if (typeof strideSec === 'number') {
    strideFrames = Math.max(3, Math.round(fps * strideSec))
  }

  if (strideFrames > 0 && cleaned.length > 0) {
    const subsampled: ShuttlePoint[] = [cleaned[0]!]
    for (let i = 1; i < cleaned.length; i++) {
      if (cleaned[i]!.frame - subsampled[subsampled.length - 1]!.frame >= strideFrames) {
        subsampled.push(cleaned[i]!)
      }
    }
    samples = subsampled
  }

  if (samples.length < 3) return []

  // ---------------------------------------------------------------------------
  // 5. Candidate triples
  // ---------------------------------------------------------------------------
  const frameMap = new Map<number, SkeletonFrame>()
  for (const f of frames) frameMap.set(f.frame, f)

  const shots: ShotEvent[] = []
  let lastShotFrame = -Infinity

  for (let i = 2; i < samples.length; i++) {
    const p0 = samples[i - 2]!
    const p1 = samples[i - 1]!
    const p2 = samples[i]!

    // Max-time-gap guard: velocity is meaningless across long idle gaps.
    if ((p1.timestamp - p0.timestamp) > MAX_GAP_S || (p2.timestamp - p1.timestamp) > MAX_GAP_S) {
      continue
    }

    const vx1 = p1.x - p0.x
    const vy1 = p1.y - p0.y
    const vx2 = p2.x - p1.x
    const vy2 = p2.y - p1.y

    const speed1sq = vx1 * vx1 + vy1 * vy1
    const speed2sq = vx2 * vx2 + vy2 * vy2

    // Both near-stationary → jitter, not a hit.
    if (speed1sq < minSpeedSq && speed2sq < minSpeedSq) {
      continue
    }

    const dot = vx1 * vx2 + vy1 * vy2

    // Dot-threshold math — matches backend Python formulation.
    const threshold = cosAngleMax < 0
      ? cosAngleMax * Math.sqrt(speed1sq * speed2sq)
      : 0
    if (dot >= threshold) {
      continue
    }

    // Optional accel gate (auto-pause semantics).
    if (opts.minAccelMagPx != null) {
      const dvx = vx2 - vx1
      const dvy = vy2 - vy1
      const accelMag = Math.sqrt(dvx * dvx + dvy * dvy)
      if (accelMag < opts.minAccelMagPx) {
        continue
      }
    }

    // Min-shot-gap (frame-based, matches backend).
    if ((p1.frame - lastShotFrame) < minShotGapFrames) {
      continue
    }

    // Skeleton lookup for player assignment.
    const skFrame = frameMap.get(p1.frame)
    if (!skFrame || skFrame.players.length === 0) {
      continue
    }

    // Optional wrist-proximity gate.
    if (opts.wristProximityMeters != null && homography) {
      const wristDist = nearestWristMeters(skFrame.players, p1.x, p1.y, homography)
      if (wristDist != null && wristDist > opts.wristProximityMeters) {
        continue
      }
    }

    const closest = findClosestPlayer(skFrame.players, p1.x, p1.y)
    if (!closest) {
      continue
    }

    shots.push({
      frame: p1.frame,
      timestamp: p1.timestamp,
      playerId: closest.player_id,
      shuttlePosition: { x: p1.x, y: p1.y },
      playerPosition: closest.center ? { x: closest.center.x, y: closest.center.y } : { x: 0, y: 0 },
      detectionMethod: 'shuttle_trajectory',
    })
    lastShotFrame = p1.frame
  }

  return shots
}

// -----------------------------------------------------------------------------
// Pose classification fallback
// -----------------------------------------------------------------------------

export interface DetectPoseShotsOptions {
  /** Video frame rate (frames per second). */
  fps: number
  /** Set of class_name values considered hitting poses (e.g. 'smash'). */
  hittingClasses: Set<string>
  /** Minimum classification confidence. Default 0.5. */
  minConfidence?: number
  /** Inter-shot minimum gap across all players, in seconds. Default 0.5. */
  minShotGapSec?: number
  /** Same-player re-hit minimum gap in seconds. Default 0.8. */
  perPlayerGapSec?: number
}

/**
 * Detect shots from frame-level pose_classifications.
 *
 * For each frame with classifications:
 *   - Respect the overall `minShotGapSec` guard (frame-based).
 *   - For each classification, require class in `hittingClasses` and
 *     confidence >= `minConfidence`.
 *   - Match player: if the classification has a bbox AND the frame has
 *     multiple players, pick the player closest to the bbox center;
 *     otherwise default to `frame.players[0]`.
 *   - Respect a per-player gap (frame-based) to prevent double-counting a
 *     held pose.
 */
export function detectPoseShots(
  frames: SkeletonFrame[],
  opts: DetectPoseShotsOptions,
): ShotEvent[] {
  const { fps, hittingClasses } = opts
  if (!fps || fps <= 0) return []

  const minConfidence = opts.minConfidence ?? 0.5
  const minShotGapSec = opts.minShotGapSec ?? 0.5
  const perPlayerGapSec = opts.perPlayerGapSec ?? 0.8

  const minShotGapFrames = Math.max(1, Math.floor(fps * minShotGapSec))
  const perPlayerGapFrames = Math.max(minShotGapFrames, Math.floor(fps * perPlayerGapSec))

  const shots: ShotEvent[] = []
  const lastShotByPlayer = new Map<number, number>()
  let lastShotFrame = -Infinity

  for (const frame of frames) {
    const classifications = frame.pose_classifications
    if (!classifications || classifications.length === 0) continue
    if ((frame.frame - lastShotFrame) < minShotGapFrames) continue

    for (const cls of classifications as PoseClassificationResult[]) {
      if (!hittingClasses.has(cls.class_name)) continue
      if (cls.confidence < minConfidence) continue

      let matchedPlayer: FramePlayer | null = frame.players[0] ?? null

      if (cls.bbox && frame.players.length > 1) {
        const cx = cls.bbox.x + cls.bbox.width / 2
        const cy = cls.bbox.y + cls.bbox.height / 2
        matchedPlayer = findClosestPlayer(frame.players, cx, cy) ?? matchedPlayer
      }

      if (!matchedPlayer) continue

      const lastForPlayer = lastShotByPlayer.get(matchedPlayer.player_id) ?? -Infinity
      if ((frame.frame - lastForPlayer) < perPlayerGapFrames) continue

      const shuttle = frame.shuttle_position
      shots.push({
        frame: frame.frame,
        timestamp: frame.timestamp,
        playerId: matchedPlayer.player_id,
        shuttlePosition: shuttle && shuttle.x != null && shuttle.y != null
          ? { x: shuttle.x, y: shuttle.y }
          : null,
        playerPosition: matchedPlayer.center
          ? { x: matchedPlayer.center.x, y: matchedPlayer.center.y }
          : { x: 0, y: 0 },
        detectionMethod: 'pose_classification',
        shotType: cls.class_name,
      })
      lastShotFrame = frame.frame
      lastShotByPlayer.set(matchedPlayer.player_id, frame.frame)
      // Only one shot per frame.
      break
    }
  }

  return shots
}

// -----------------------------------------------------------------------------
// Merge
// -----------------------------------------------------------------------------

/**
 * Merge shot events from multiple detectors, preferring shuttle-detected
 * shots over pose-only ones when they collide within `minGapFrames`.
 *
 * Mirrors useAdvancedAnalytics.ts:mergeShots:
 *   - Sort by frame.
 *   - Walk in order; if the current shot is within minGapFrames of the last
 *     accepted one, keep whichever has a non-null `shuttlePosition` (shuttle
 *     wins); otherwise drop the current shot.
 */
export function mergeShots(shots: ShotEvent[], minGapFrames: number): ShotEvent[] {
  const sorted = shots.slice().sort((a, b) => a.frame - b.frame)
  const merged: ShotEvent[] = []
  for (const shot of sorted) {
    const last = merged[merged.length - 1]
    if (last && (shot.frame - last.frame) < minGapFrames) {
      if (last.shuttlePosition == null && shot.shuttlePosition != null) {
        merged[merged.length - 1] = shot
      }
      continue
    }
    merged.push(shot)
  }
  return merged
}
