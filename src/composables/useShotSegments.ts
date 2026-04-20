import { computed, type Ref } from 'vue'
import type { SkeletonFrame, Keypoint, SpeedZone } from '@/types/analysis'
import { getSpeedZone } from '@/utils/speedZones'
import { legStretchMeters, kneeFlexDegrees } from '@/utils/bodyAngles'
import {
  detectShuttleShots,
  detectPoseShots,
  type ShotEvent as SharedShotEvent,
  type DetectionMethod,
} from '@/utils/shotDetection'

// Re-export ShotEvent so downstream consumers (ShotSpeedList.vue, App.vue)
// can continue importing it from this module. The canonical definition now
// lives in @/utils/shotDetection.
export type { ShotEvent } from '@/utils/shotDetection'

export interface ShotMovementSegment {
  id: number
  startShot: SharedShotEvent
  endShot: SharedShotEvent
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

/**
 * Detect shot events using a cascade of methods:
 * 1. Shuttle trajectory analysis (via shared @/utils/shotDetection).
 * 2. Pose classification (via shared @/utils/shotDetection).
 * 3. Player movement pattern analysis (local fallback - uses acceleration/deceleration).
 *
 * Method 3 is the reliable fallback: in badminton, a shot correlates with
 * a player decelerating (reaching the shuttle) after rapid movement. We detect
 * these deceleration events and alternate between players.
 */
export function detectShots(
  frames: SkeletonFrame[],
  homography: number[][] | null = null,
  fps = 30,
): SharedShotEvent[] {
  // Routing data availability
  let shuttleFrameCount = 0
  let poseClassCount = 0
  for (const frame of frames) {
    if (frame.shuttle_position && frame.shuttle_position.x != null) shuttleFrameCount++
    if (frame.pose_classifications && frame.pose_classifications.length > 0) poseClassCount++
  }
  console.log(
    `[ShotSpeedList] Data available: ${shuttleFrameCount} shuttle frames, ` +
    `${poseClassCount} pose classification frames, ${frames.length} total frames`
  )

  // Primary: shared shuttle-trajectory detector with pause-use-case options.
  if (shuttleFrameCount >= 5) {
    const shots = detectShuttleShots(frames, {
      fps,
      cameraAngle: 'overhead',
      homography,
      minAccelMagPx: 200,
      wristProximityMeters: homography ? 3.5 : null,
      rejectOutliers: true,
      strideSec: 'auto',
      logStats: true,
    })
    if (shots.length >= 3) {
      console.log(`[ShotSpeedList] Using shuttle trajectory detection: ${shots.length} shots`)
      return shots
    }
  }

  // Pose fallback
  if (poseClassCount >= 3) {
    const shots = detectPoseShots(frames, {
      fps,
      hittingClasses: new Set(['smash', 'offense', 'backhand-general', 'serve', 'lift']),
      minConfidence: 0.5,
      minShotGapSec: 0.5,
      perPlayerGapSec: 0.8,
    })
    if (shots.length >= 3) {
      console.log(`[ShotSpeedList] Using pose classification detection: ${shots.length} shots`)
      return shots
    }
  }

  // Last resort: local player-movement fallback (keeps its own detectShotsFromSpeedPeaks path).
  const shots = detectShotsFromPlayerMovement(frames)
  console.log(`[ShotSpeedList] Using player movement detection: ${shots.length} shots`)
  return shots
}

/**
 * Primary fallback: Detect shots from player movement patterns.
 *
 * In badminton, a shot happens when a player decelerates after fast movement
 * (they've reached the shuttle and are hitting it). We detect these deceleration
 * peaks and alternately assign them to players.
 *
 * Algorithm:
 * 1. For each player, compute per-frame speed from position changes
 * 2. Smooth the speed signal
 * 3. Find "deceleration peaks" — moments where speed drops significantly after being high
 * 4. These peaks indicate the player just reached the shuttle (shot moment)
 * 5. Merge events from both players, sorted by time, alternating where possible
 */
function detectShotsFromPlayerMovement(frames: SkeletonFrame[]): SharedShotEvent[] {
  if (frames.length < 10) return []

  // Build per-player position timeline with speeds
  const playerTimelines: Map<number, {
    frame: number
    timestamp: number
    x: number
    y: number
    speed: number  // km/h from skeleton data
  }[]> = new Map()

  for (const frame of frames) {
    for (const player of frame.players) {
      if (!player.center) continue

      let timeline = playerTimelines.get(player.player_id)
      if (!timeline) {
        timeline = []
        playerTimelines.set(player.player_id, timeline)
      }

      timeline.push({
        frame: frame.frame,
        timestamp: frame.timestamp,
        x: player.center.x,
        y: player.center.y,
        speed: player.current_speed ?? 0
      })
    }
  }

  // For each player, detect "shot events" as deceleration peaks
  const playerShots: SharedShotEvent[] = []
  const MIN_SHOT_GAP = 0.6 // seconds between shots by same player
  const DECEL_THRESHOLD_KMH = 2.0 // Minimum speed drop to count as deceleration
  const MIN_SPEED_BEFORE_DECEL = 3.0 // Player must be moving at least this fast (km/h) before decel

  for (const [playerId, timeline] of playerTimelines) {
    if (timeline.length < 5) continue

    // Smooth speeds (moving average window=5)
    const smoothedSpeeds: number[] = []
    for (let i = 0; i < timeline.length; i++) {
      let sum = 0, count = 0
      for (let j = Math.max(0, i - 2); j <= Math.min(timeline.length - 1, i + 2); j++) {
        sum += timeline[j]!.speed
        count++
      }
      smoothedSpeeds.push(sum / count)
    }

    // Detect deceleration peaks: speed was high, then drops significantly
    let lastShotTimestamp = -Infinity

    for (let i = 3; i < smoothedSpeeds.length - 1; i++) {
      const prev3Avg = (smoothedSpeeds[i-3]! + smoothedSpeeds[i-2]! + smoothedSpeeds[i-1]!) / 3
      const curr = smoothedSpeeds[i]!
      const decel = prev3Avg - curr

      const entry = timeline[i]!

      // Deceleration peak: was moving fast, now slowing down
      if (decel > DECEL_THRESHOLD_KMH &&
          prev3Avg > MIN_SPEED_BEFORE_DECEL &&
          entry.timestamp - lastShotTimestamp > MIN_SHOT_GAP) {

        playerShots.push({
          frame: entry.frame,
          timestamp: entry.timestamp,
          playerId: playerId,
          shuttlePosition: { x: entry.x, y: entry.y }, // Use player position as proxy
          playerPosition: { x: entry.x, y: entry.y },
          detectionMethod: 'player_movement' as DetectionMethod
        })

        lastShotTimestamp = entry.timestamp
      }
    }
  }

  // Sort all shots by timestamp
  playerShots.sort((a, b) => a.timestamp - b.timestamp)

  // If we have very few shots, lower thresholds and try again with just speed peaks
  if (playerShots.length < 3) {
    return detectShotsFromSpeedPeaks(frames, playerTimelines)
  }

  // Post-process: ensure minimum gap between any two shots
  const filteredShots: SharedShotEvent[] = []
  let lastTs = -Infinity
  for (const shot of playerShots) {
    if (shot.timestamp - lastTs > 0.4) {
      filteredShots.push(shot)
      lastTs = shot.timestamp
    }
  }

  return filteredShots
}

/**
 * Fallback within fallback: detect shots from raw speed peaks.
 * Simply finds local maxima in player speed data, which indicate
 * peak movement moments (player reaching for shuttle).
 */
function detectShotsFromSpeedPeaks(
  frames: SkeletonFrame[],
  playerTimelines: Map<number, { frame: number; timestamp: number; x: number; y: number; speed: number }[]>
): SharedShotEvent[] {
  const shots: SharedShotEvent[] = []
  const MIN_PEAK_SPEED = 2.0 // km/h - very low threshold to catch anything
  const MIN_GAP = 0.8 // seconds

  for (const [playerId, timeline] of playerTimelines) {
    if (timeline.length < 5) continue

    let lastShotTs = -Infinity

    // Find local speed maxima (peaks in movement)
    for (let i = 2; i < timeline.length - 2; i++) {
      const speed = timeline[i]!.speed
      const prevSpeed = timeline[i-1]!.speed
      const nextSpeed = timeline[i+1]!.speed
      const prev2Speed = timeline[i-2]!.speed
      const next2Speed = timeline[i+2]!.speed

      // Local maximum across a symmetric 5-sample window (i-2..i+2):
      // current speed strictly greater than its two neighbors on each side.
      const isLocalMax = speed > prevSpeed && speed > nextSpeed &&
                         speed > prev2Speed && speed > next2Speed &&
                         speed > MIN_PEAK_SPEED

      const entry = timeline[i]!

      if (isLocalMax && entry.timestamp - lastShotTs > MIN_GAP) {
        shots.push({
          frame: entry.frame,
          timestamp: entry.timestamp,
          playerId,
          shuttlePosition: { x: entry.x, y: entry.y },
          playerPosition: { x: entry.x, y: entry.y },
          detectionMethod: 'speed_peaks' as DetectionMethod
        })
        lastShotTs = entry.timestamp
      }
    }
  }

  shots.sort((a, b) => a.timestamp - b.timestamp)

  // Filter minimum gap between any two shots
  const filtered: SharedShotEvent[] = []
  let lastTs = -Infinity
  for (const shot of shots) {
    if (shot.timestamp - lastTs > 0.5) {
      filtered.push(shot)
      lastTs = shot.timestamp
    }
  }

  return filtered
}

/**
 * Build movement segments from consecutive shot events.
 * Each segment represents the period between opponent's hit and player's responding hit.
 */
export function buildMovementSegments(
  shots: SharedShotEvent[],
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

export interface BodyAnglePeaks {
  peakLegStretchM: number | null   // Max ankle-to-ankle distance in meters
  peakKneeFlexDeg: number | null   // Minimum knee angle (deepest bend)
  peakTorsoLeanDeg: number | null  // Max absolute torso-lean angle
}

/**
 * Walk the frames of a single segment for the moving player and compute
 * peak body-mechanics values. H is the video-pixels → court-meters homography
 * (the forward matrix produced by computeHomographyFromKeypoints). When null,
 * leg stretch cannot be computed and is returned as null.
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

    const leg = legStretchMeters(player.keypoints as Keypoint[], H)
    if (leg != null && (peakLeg == null || leg > peakLeg)) peakLeg = leg

    const knee = kneeFlexDegrees(
      player.pose?.body_angles?.left_knee,
      player.pose?.body_angles?.right_knee,
    )
    if (knee != null && (peakKnee == null || knee < peakKnee)) peakKnee = knee

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

export interface ShotMovementSegmentWithPeaks extends ShotMovementSegment {
  peaks: BodyAnglePeaks
}

export function useShotSegments(
  skeletonData: Ref<SkeletonFrame[] | undefined>,
  homography?: Ref<number[][] | null>,
) {
  // Infer fps from frame timestamps when available. Falls back to 30 fps when
  // the data is insufficient or timestamps collapse (e.g. single-frame clip).
  const fpsEstimate = computed(() => {
    const frames = skeletonData.value
    if (!frames || frames.length < 2) return 30
    const span = frames[frames.length - 1]!.timestamp - frames[0]!.timestamp
    return span > 0 ? (frames.length - 1) / span : 30
  })

  const shotEvents = computed<SharedShotEvent[]>(() => {
    const frames = skeletonData.value
    if (!frames || frames.length === 0) return []
    return detectShots(frames, homography?.value ?? null, fpsEstimate.value)
  })

  const segments = computed<ShotMovementSegmentWithPeaks[]>(() => {
    const frames = skeletonData.value
    if (!frames) return []
    const H = homography?.value ?? null
    const base = buildMovementSegments(shotEvents.value, frames)
    return base.map(seg => ({ ...seg, peaks: aggregateBodyAngles(seg, frames, H) }))
  })

  return { shotEvents, segments }
}
