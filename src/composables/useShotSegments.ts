import { computed, type Ref } from 'vue'
import type { SkeletonFrame, Keypoint, SpeedZone } from '@/types/analysis'
import { getSpeedZone } from '@/utils/speedZones'
import { legStretchMeters, kneeFlexDegrees } from '@/utils/bodyAngles'

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

/**
 * Detect shot events using a cascade of methods:
 * 1. Shuttle trajectory analysis (if shuttle_position data available)
 * 2. Pose classification (if pose_classifications data available)
 * 3. Player movement pattern analysis (always available - uses acceleration/deceleration)
 *
 * Method 3 is the reliable fallback: in badminton, a shot correlates with
 * a player decelerating (reaching the shuttle) after rapid movement. We detect
 * these deceleration events and alternate between players.
 */
export function detectShots(frames: SkeletonFrame[]): ShotEvent[] {
  // Count data availability
  let shuttleFrameCount = 0
  let poseClassCount = 0
  
  for (const frame of frames) {
    if (frame.shuttle_position && frame.shuttle_position.x != null) shuttleFrameCount++
    if (frame.pose_classifications && frame.pose_classifications.length > 0) poseClassCount++
  }
  
  console.log(`[ShotSpeedList] Data available: ${shuttleFrameCount} shuttle frames, ${poseClassCount} pose classification frames, ${frames.length} total frames`)
  
  // Try shuttle trajectory first if enough data
  if (shuttleFrameCount >= 5) {
    const shots = detectShotsFromShuttleTrajectory(frames)
    if (shots.length >= 3) {
      console.log(`[ShotSpeedList] Using shuttle trajectory detection: ${shots.length} shots`)
      return shots
    }
  }
  
  // Try pose classification if available
  if (poseClassCount >= 3) {
    const shots = detectShotsFromPoseClassification(frames)
    if (shots.length >= 3) {
      console.log(`[ShotSpeedList] Using pose classification detection: ${shots.length} shots`)
      return shots
    }
  }
  
  // Primary fallback: Player movement analysis (always works with skeleton data)
  const shots = detectShotsFromPlayerMovement(frames)
  console.log(`[ShotSpeedList] Using player movement detection: ${shots.length} shots`)
  return shots
}

/**
 * Detect shots from shuttle position trajectory direction changes.
 */
function detectShotsFromShuttleTrajectory(frames: SkeletonFrame[]): ShotEvent[] {
  const shots: ShotEvent[] = []
  
  const shuttleFrames: { frame: number; timestamp: number; x: number; y: number }[] = []
  for (const frame of frames) {
    if (frame.shuttle_position && frame.shuttle_position.x != null && frame.shuttle_position.y != null) {
      shuttleFrames.push({
        frame: frame.frame,
        timestamp: frame.timestamp,
        x: frame.shuttle_position.x,
        y: frame.shuttle_position.y
      })
    }
  }
  
  if (shuttleFrames.length < 5) return shots
  
  const smoothed = smoothPositions(shuttleFrames)
  
  const velocities: { frame: number; timestamp: number; vx: number; vy: number; x: number; y: number }[] = []
  for (let i = 1; i < smoothed.length; i++) {
    const prev = smoothed[i - 1]!
    const curr = smoothed[i]!
    const dt = curr.timestamp - prev.timestamp
    if (dt > 0) {
      velocities.push({
        frame: curr.frame,
        timestamp: curr.timestamp,
        vx: (curr.x - prev.x) / dt,
        vy: (curr.y - prev.y) / dt,
        x: curr.x,
        y: curr.y
      })
    }
  }
  
  // Layer C thresholds. Earlier config (dot<0 / accel>50 / gap>0.3s) fired
  // on any direction change with any acceleration bump, including the
  // gravity-induced apex-of-arc flip (vy crosses zero) and TrackNet jitter
  // while the shuttle wobbles near-stationary between rallies. New gates:
  //   - cosAngle < -0.5  → require a sharp >120° reversal, not a smooth arc
  //   - accelMag > 200   → raise the bar 4× in raw pixel units
  //   - min speed mag    → reject when BOTH sides are near-stationary
  //   - gap > 0.5s       → match real shot tempo, kill double-triggers
  const MIN_SHOT_GAP_SECONDS = 0.5
  const MIN_ANGLE_CHANGE_COS = -0.5
  const MIN_ACCEL_MAG_PX = 200
  const MIN_SPEED_MAG_PX = 80
  let lastShotTimestamp = -Infinity

  for (let i = 1; i < velocities.length; i++) {
    const prev = velocities[i - 1]!
    const curr = velocities[i]!

    const prevMag = Math.hypot(prev.vx, prev.vy)
    const currMag = Math.hypot(curr.vx, curr.vy)
    // Both sides near-stationary → this is shuttle wobble, not a hit.
    if (prevMag < MIN_SPEED_MAG_PX && currMag < MIN_SPEED_MAG_PX) continue

    const dot = prev.vx * curr.vx + prev.vy * curr.vy
    const denom = prevMag * currMag
    const cosAngle = denom > 0 ? dot / denom : 0
    // Require a sharp direction reversal (>120°). An apex-of-arc gravity
    // flip produces a gradual change per-frame; a racket hit nearly
    // reverses the velocity vector.
    if (cosAngle > MIN_ANGLE_CHANGE_COS) continue

    const dvx = curr.vx - prev.vx
    const dvy = curr.vy - prev.vy
    const accelMag = Math.sqrt(dvx * dvx + dvy * dvy)
    if (accelMag < MIN_ACCEL_MAG_PX) continue

    if (curr.timestamp - lastShotTimestamp < MIN_SHOT_GAP_SECONDS) continue

    const skeletonFrame = frames.find(f => f.frame === curr.frame)
    if (skeletonFrame && skeletonFrame.players.length > 0) {
      const closest = findClosestPlayer(skeletonFrame.players, curr.x, curr.y)
      if (closest) {
        shots.push({
          frame: curr.frame,
          timestamp: curr.timestamp,
          playerId: closest.player_id,
          shuttlePosition: { x: curr.x, y: curr.y },
          playerPosition: closest.center ? { x: closest.center.x, y: closest.center.y } : { x: 0, y: 0 },
          detectionMethod: 'shuttle_trajectory'
        })
        lastShotTimestamp = curr.timestamp
      }
    }
  }

  return shots
}

/**
 * Detect shots from pose classification data (smash, offense, backhand, serve, lift).
 */
function detectShotsFromPoseClassification(frames: SkeletonFrame[]): ShotEvent[] {
  const shots: ShotEvent[] = []
  const HITTING_CLASSES = new Set(['smash', 'offense', 'backhand-general', 'serve', 'lift'])
  const MIN_CONFIDENCE = 0.5
  const MIN_SHOT_GAP_SECONDS = 0.5
  let lastShotTimestamp = -Infinity
  
  for (const frame of frames) {
    if (!frame.pose_classifications || frame.pose_classifications.length === 0) continue
    if (frame.timestamp - lastShotTimestamp < MIN_SHOT_GAP_SECONDS) continue
    
    for (const classification of frame.pose_classifications) {
      if (HITTING_CLASSES.has(classification.class_name) && classification.confidence >= MIN_CONFIDENCE) {
        let matchedPlayer = frame.players[0]
        
        if (classification.bbox && frame.players.length > 1) {
          const bboxCenterX = classification.bbox.x + classification.bbox.width / 2
          const bboxCenterY = classification.bbox.y + classification.bbox.height / 2
          matchedPlayer = findClosestPlayer(frame.players, bboxCenterX, bboxCenterY) || matchedPlayer
        }
        
        if (matchedPlayer) {
          shots.push({
            frame: frame.frame,
            timestamp: frame.timestamp,
            playerId: matchedPlayer.player_id,
            shuttlePosition: frame.shuttle_position
              ? { x: frame.shuttle_position.x, y: frame.shuttle_position.y }
              : matchedPlayer.center
                ? { x: matchedPlayer.center.x, y: matchedPlayer.center.y }
                : { x: 0, y: 0 },
            playerPosition: matchedPlayer.center
              ? { x: matchedPlayer.center.x, y: matchedPlayer.center.y }
              : { x: 0, y: 0 },
            detectionMethod: 'pose_classification'
          })
          lastShotTimestamp = frame.timestamp
          break
        }
      }
    }
  }
  
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
function detectShotsFromPlayerMovement(frames: SkeletonFrame[]): ShotEvent[] {
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
  const playerShots: ShotEvent[] = []
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
          detectionMethod: 'player_movement'
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
  const filteredShots: ShotEvent[] = []
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
): ShotEvent[] {
  const shots: ShotEvent[] = []
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
          detectionMethod: 'speed_peaks'
        })
        lastShotTs = entry.timestamp
      }
    }
  }
  
  shots.sort((a, b) => a.timestamp - b.timestamp)
  
  // Filter minimum gap between any two shots
  const filtered: ShotEvent[] = []
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
 * Find the closest player to a given position.
 */
function findClosestPlayer(
  players: SkeletonFrame['players'],
  x: number,
  y: number
): SkeletonFrame['players'][0] | null {
  let closest: SkeletonFrame['players'][0] | null = null
  let closestDist = Infinity
  
  for (const player of players) {
    if (!player.center) continue
    const dx = player.center.x - x
    const dy = player.center.y - y
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < closestDist) {
      closestDist = dist
      closest = player
    }
  }
  
  return closest || players[0] || null
}

/**
 * Smooth positions using a moving average to reduce noise.
 */
function smoothPositions(
  points: { frame: number; timestamp: number; x: number; y: number }[],
  windowSize: number = 3
): { frame: number; timestamp: number; x: number; y: number }[] {
  const result: { frame: number; timestamp: number; x: number; y: number }[] = []
  const half = Math.floor(windowSize / 2)
  
  for (let i = 0; i < points.length; i++) {
    let sumX = 0, sumY = 0, count = 0
    for (let j = Math.max(0, i - half); j <= Math.min(points.length - 1, i + half); j++) {
      sumX += points[j]!.x
      sumY += points[j]!.y
      count++
    }
    result.push({
      frame: points[i]!.frame,
      timestamp: points[i]!.timestamp,
      x: sumX / count,
      y: sumY / count
    })
  }
  
  return result
}

/**
 * Build movement segments from consecutive shot events.
 * Each segment represents the period between opponent's hit and player's responding hit.
 */
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
  const shotEvents = computed<ShotEvent[]>(() => {
    const frames = skeletonData.value
    if (!frames || frames.length === 0) return []
    return detectShots(frames)
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
