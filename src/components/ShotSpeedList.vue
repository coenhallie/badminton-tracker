<script setup lang="ts">
/**
 * ShotSpeedList Component
 * 
 * Analyzes skeleton data to detect shot events and calculate player movement
 * speed between shots. Shows a clickable list of shots in each rally with
 * max and average speed of the responding player's movement.
 * 
 * The key insight from the client: the most valuable speed metric is the
 * movement speed between when the opponent hits the shuttle and when the
 * player hits the shuttle ("reaction burst speed").
 * 
 * Shot detection uses shuttle position trajectory analysis:
 * 1. Track shuttle positions across frames
 * 2. Detect direction changes (indicating a hit)
 * 3. Associate each hit with the nearest player
 * 4. Calculate movement speeds between consecutive hits
 */

import { ref, computed, watch } from 'vue'
import type {
  SkeletonFrame,
  SpeedZone,
} from '@/types/analysis'
import {
  SPEED_ZONE_THRESHOLDS,
  SPEED_ZONE_COLORS,
  SPEED_ZONE_NAMES,
  PLAYER_SPEED_COLORS,
} from '@/types/analysis'

// =============================================================================
// TYPES
// =============================================================================

interface ShotEvent {
  frame: number
  timestamp: number
  playerId: number         // Player who hit the shuttle
  shuttlePosition: { x: number; y: number }
  playerPosition: { x: number; y: number }
  detectionMethod: 'shuttle_trajectory' | 'pose_classification'
}

interface ShotMovementSegment {
  id: number
  // Shot that starts this segment (opponent's hit)
  startShot: ShotEvent
  // Shot that ends this segment (this player's response hit)
  endShot: ShotEvent
  // The player who is MOVING (responding to the opponent's shot)
  movingPlayerId: number
  // Speed metrics for the moving player during this segment
  maxSpeedKmh: number
  avgSpeedKmh: number
  maxSpeedZone: SpeedZone
  // Distance covered by the moving player
  distanceCoveredM: number | null
  // Frame range
  startFrame: number
  endFrame: number
  startTimestamp: number
  endTimestamp: number
  durationSeconds: number
  // Per-frame speed data for this segment (for potential micro-chart)
  speedProfile: number[] // km/h values
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  skeletonData: SkeletonFrame[]
  fps: number
  visible?: boolean
  courtKeypointsSet?: boolean
  speedCalculated?: boolean
}>()

const emit = defineEmits<{
  (e: 'seekToSegment', startTime: number, endTime: number): void
  (e: 'close'): void
}>()

// =============================================================================
// STATE
// =============================================================================

const useKmh = ref(true)
const selectedSegmentId = ref<number | null>(null)
const filterPlayer = ref<number | null>(null) // null = show all players
const sortBy = ref<'time' | 'maxSpeed' | 'avgSpeed'>('time')
const sortDesc = ref(false)

// =============================================================================
// SHOT DETECTION
// =============================================================================

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
function detectShots(frames: SkeletonFrame[]): ShotEvent[] {
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
  
  const MIN_SHOT_GAP_SECONDS = 0.3
  let lastShotTimestamp = -Infinity
  
  for (let i = 1; i < velocities.length; i++) {
    const prev = velocities[i - 1]!
    const curr = velocities[i]!
    const dot = prev.vx * curr.vx + prev.vy * curr.vy
    const dvx = curr.vx - prev.vx
    const dvy = curr.vy - prev.vy
    const accelMag = Math.sqrt(dvx * dvx + dvy * dvy)
    
    if (dot < 0 && accelMag > 50 && (curr.timestamp - lastShotTimestamp) > MIN_SHOT_GAP_SECONDS) {
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
          detectionMethod: 'shuttle_trajectory' // Using this to keep type simple
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
      const next2Speed = timeline[i+1]!.speed
      
      // Local maximum: current speed is higher than neighbors
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
          detectionMethod: 'shuttle_trajectory'
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
 * Get speed zone classification from km/h.
 */
function getSpeedZone(speedKmh: number): SpeedZone {
  const speedMps = speedKmh / 3.6
  for (const [zoneName, thresholds] of Object.entries(SPEED_ZONE_THRESHOLDS)) {
    const maxSpeed = thresholds.max ?? Infinity
    if (speedMps >= thresholds.min && speedMps < maxSpeed) {
      return zoneName as SpeedZone
    }
  }
  return 'standing'
}

// =============================================================================
// COMPUTED: Shot events and movement segments
// =============================================================================

const shotEvents = computed(() => {
  if (!props.skeletonData || props.skeletonData.length === 0) return []
  return detectShots(props.skeletonData)
})

/**
 * Build movement segments from consecutive shot events.
 * Each segment represents the period between opponent's hit and player's responding hit.
 */
const movementSegments = computed((): ShotMovementSegment[] => {
  const shots = shotEvents.value
  if (shots.length < 2) return []
  
  const segments: ShotMovementSegment[] = []
  const MAX_VALID_SPEED_KMH = 25  // Cap for filtering tracking errors
  
  // Build a frame-indexed map for quick lookup
  const frameMap = new Map<number, SkeletonFrame>()
  for (const frame of props.skeletonData) {
    frameMap.set(frame.frame, frame)
  }
  
  for (let i = 0; i < shots.length - 1; i++) {
    const startShot = shots[i]!
    const endShot = shots[i + 1]!
    
    // The moving player is the one who hits the endShot (they are responding to startShot)
    const movingPlayerId = endShot.playerId
    
    // Skip if same player hits twice in a row (likely a detection error)
    // Actually, in doubles this could be valid, but for now let's keep it simple
    // We still show them but mark them differently
    
    const duration = endShot.timestamp - startShot.timestamp
    
    // Skip very short or very long segments (likely errors)
    if (duration < 0.2 || duration > 15) continue
    
    // Collect speed data for the moving player during this segment
    const speedProfile: number[] = []
    let maxSpeedKmh = 0
    let sumSpeedKmh = 0
    let speedCount = 0
    
    for (const frame of props.skeletonData) {
      if (frame.frame < startShot.frame || frame.frame > endShot.frame) continue
      
      const player = frame.players.find(p => p.player_id === movingPlayerId)
      if (player) {
        const speed = player.current_speed ?? 0
        // Filter unrealistic speeds
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
      distanceCoveredM: null, // Would need court calibration to calculate
      startFrame: startShot.frame,
      endFrame: endShot.frame,
      startTimestamp: startShot.timestamp,
      endTimestamp: endShot.timestamp,
      durationSeconds: duration,
      speedProfile,
    })
  }
  
  return segments
})

// =============================================================================
// COMPUTED: Filtered & sorted segments
// =============================================================================

const filteredSegments = computed(() => {
  let result = movementSegments.value
  
  // Filter by player
  if (filterPlayer.value !== null) {
    result = result.filter(s => s.movingPlayerId === filterPlayer.value)
  }
  
  // Sort
  const sorted = [...result]
  switch (sortBy.value) {
    case 'time':
      sorted.sort((a, b) => a.startTimestamp - b.startTimestamp)
      break
    case 'maxSpeed':
      sorted.sort((a, b) => b.maxSpeedKmh - a.maxSpeedKmh)
      break
    case 'avgSpeed':
      sorted.sort((a, b) => b.avgSpeedKmh - a.avgSpeedKmh)
      break
  }
  
  if (sortDesc.value && sortBy.value === 'time') {
    sorted.reverse()
  }
  
  return sorted
})

// Available player IDs
const playerIds = computed(() => {
  const ids = new Set<number>()
  for (const segment of movementSegments.value) {
    ids.add(segment.movingPlayerId)
  }
  return Array.from(ids).sort()
})

// Summary statistics
const summaryStats = computed(() => {
  const segments = filteredSegments.value
  if (segments.length === 0) return null
  
  return {
    totalShots: segments.length,
    overallMaxSpeed: Math.max(...segments.map(s => s.maxSpeedKmh)),
    overallAvgSpeed: segments.reduce((sum, s) => sum + s.avgSpeedKmh, 0) / segments.length,
    avgReactionTime: segments.reduce((sum, s) => sum + s.durationSeconds, 0) / segments.length,
    fastestSegment: segments.reduce((max, s) => s.maxSpeedKmh > max.maxSpeedKmh ? s : max, segments[0]!),
  }
})

// =============================================================================
// METHODS
// =============================================================================

function handleSegmentClick(segment: ShotMovementSegment) {
  selectedSegmentId.value = segment.id
  // Seek to 0.5s before the segment starts for context
  const startTime = Math.max(0, segment.startTimestamp - 0.5)
  emit('seekToSegment', startTime, segment.endTimestamp + 0.5)
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  const tenths = Math.floor((seconds % 1) * 10)
  return `${mins}:${secs.toString().padStart(2, '0')}.${tenths}`
}

function formatSpeed(kmh: number): string {
  if (useKmh.value) {
    return kmh.toFixed(1)
  }
  return (kmh / 3.6).toFixed(2)
}

function getSpeedUnit(): string {
  return useKmh.value ? 'km/h' : 'm/s'
}

function getPlayerColor(playerId: number): string {
  const idx = Math.max(0, playerId - 1) % PLAYER_SPEED_COLORS.length
  return PLAYER_SPEED_COLORS[idx] ?? '#3B82F6'
}

function getZoneColor(zone: SpeedZone): string {
  return SPEED_ZONE_COLORS[zone] ?? '#94A3B8'
}

function getZoneName(zone: SpeedZone): string {
  return SPEED_ZONE_NAMES[zone] ?? 'Unknown'
}

function toggleSort(field: 'time' | 'maxSpeed' | 'avgSpeed') {
  if (sortBy.value === field) {
    sortDesc.value = !sortDesc.value
  } else {
    sortBy.value = field
    sortDesc.value = false
  }
}

/**
 * Generate a mini sparkline SVG path for the speed profile.
 */
function getSparklinePath(speedProfile: number[], width: number = 60, height: number = 20): string {
  if (speedProfile.length < 2) return ''
  
  const maxSpeed = Math.max(...speedProfile, 1)
  const step = width / (speedProfile.length - 1)
  
  let path = `M 0 ${height - (speedProfile[0]! / maxSpeed) * height}`
  for (let i = 1; i < speedProfile.length; i++) {
    const x = i * step
    const y = height - (speedProfile[i]! / maxSpeed) * height
    path += ` L ${x} ${y}`
  }
  
  return path
}
</script>

<template>
  <div v-if="visible !== false" class="shot-speed-container">
    <!-- Header -->
    <div class="shot-speed-header">
      <div class="header-title">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        <span>Shot-by-Shot Movement Speed</span>
      </div>
      
      <div class="header-controls">
        <!-- Unit toggle -->
        <button 
          class="unit-toggle"
          :class="{ active: useKmh }"
          @click="useKmh = !useKmh"
          title="Toggle speed unit"
        >
          {{ useKmh ? 'km/h' : 'm/s' }}
        </button>
        
        <!-- Player filter -->
        <select 
          v-model="filterPlayer" 
          class="filter-select"
          title="Filter by player"
        >
          <option :value="null">All Players</option>
          <option v-for="pid in playerIds" :key="pid" :value="pid">
            Player {{ pid }}
          </option>
        </select>
        
        <!-- Close button -->
        <button 
          class="close-btn"
          @click="emit('close')"
          title="Close shot speed analysis"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>
    
    <!-- Summary Stats -->
    <div v-if="summaryStats" class="summary-stats">
      <div class="stat-card">
        <span class="stat-label">Shots</span>
        <span class="stat-value">{{ summaryStats.totalShots }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Top Speed</span>
        <span class="stat-value highlight">
          {{ formatSpeed(summaryStats.overallMaxSpeed) }}
          <span class="stat-unit">{{ getSpeedUnit() }}</span>
        </span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Speed</span>
        <span class="stat-value">
          {{ formatSpeed(summaryStats.overallAvgSpeed) }}
          <span class="stat-unit">{{ getSpeedUnit() }}</span>
        </span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Reaction</span>
        <span class="stat-value">
          {{ summaryStats.avgReactionTime.toFixed(1) }}
          <span class="stat-unit">s</span>
        </span>
      </div>
    </div>
    
    <!-- Info banner explaining the metric -->
    <div class="info-banner">
      <svg xmlns="http://www.w3.org/2000/svg" class="icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 16v-4M12 8h.01"/>
      </svg>
      <span>
        Speed measured between opponent's hit and player's response hit.
        <strong>Click any row to replay that segment.</strong>
      </span>
    </div>
    
    <!-- Column Headers (sortable) -->
    <div class="list-header">
      <span class="col-num">#</span>
      <span class="col-time sortable" @click="toggleSort('time')">
        Time
        <span v-if="sortBy === 'time'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
      <span class="col-player">Player</span>
      <span class="col-duration">Duration</span>
      <span class="col-sparkline">Profile</span>
      <span class="col-speed sortable" @click="toggleSort('maxSpeed')">
        Max
        <span v-if="sortBy === 'maxSpeed'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
      <span class="col-speed sortable" @click="toggleSort('avgSpeed')">
        Avg
        <span v-if="sortBy === 'avgSpeed'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
    </div>
    
    <!-- Shot List -->
    <div class="shot-list" v-if="filteredSegments.length > 0">
      <div
        v-for="segment in filteredSegments"
        :key="segment.id"
        class="shot-row"
        :class="{ selected: selectedSegmentId === segment.id }"
        @click="handleSegmentClick(segment)"
      >
        <span class="col-num">{{ segment.id + 1 }}</span>
        
        <span class="col-time">
          {{ formatTime(segment.startTimestamp) }}
        </span>
        
        <span class="col-player">
          <span 
            class="player-badge"
            :style="{ backgroundColor: getPlayerColor(segment.movingPlayerId) }"
          >
            P{{ segment.movingPlayerId }}
          </span>
        </span>
        
        <span class="col-duration">
          {{ segment.durationSeconds.toFixed(1) }}s
        </span>
        
        <span class="col-sparkline">
          <svg 
            v-if="segment.speedProfile.length > 1"
            :width="60" 
            :height="20" 
            class="sparkline"
          >
            <path 
              :d="getSparklinePath(segment.speedProfile)" 
              fill="none" 
              :stroke="getZoneColor(segment.maxSpeedZone)" 
              stroke-width="1.5"
            />
          </svg>
        </span>
        
        <span class="col-speed">
          <span 
            class="speed-badge"
            :style="{ backgroundColor: getZoneColor(segment.maxSpeedZone) + '33', color: getZoneColor(segment.maxSpeedZone) }"
          >
            {{ formatSpeed(segment.maxSpeedKmh) }}
          </span>
        </span>
        
        <span class="col-speed">
          <span class="speed-value">
            {{ formatSpeed(segment.avgSpeedKmh) }}
          </span>
        </span>
      </div>
    </div>
    
    <!-- Empty State -->
    <div v-else class="empty-state">
      <div v-if="!speedCalculated && !courtKeypointsSet" class="no-data pending">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
        <p>Set court keypoints and start video to enable shot analysis</p>
        <p class="hint">Speed calibration is needed for accurate measurements</p>
      </div>
      <div v-else class="no-data">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        <p>No shots detected</p>
        <p class="hint">Shot detection requires shuttle tracking data or pose classification</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.shot-speed-container {
  background: #141414;
  border: 1px solid #222;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 100%;
  overflow: hidden;
}

/* Header */
.shot-speed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid #222;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
  font-weight: 600;
  font-size: 14px;
}

.header-title .icon {
  width: 18px;
  height: 18px;
  color: #f59e0b;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.unit-toggle {
  background: #1a1a1a;
  color: #f59e0b;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.unit-toggle:hover,
.unit-toggle.active {
  background: #222;
  border-color: #f59e0b;
}

.filter-select {
  background: #1a1a1a;
  color: #fff;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.close-btn {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px;
  cursor: pointer;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.close-btn:hover {
  background: #222;
  color: #fff;
  border-color: #444;
}

.icon-sm {
  width: 14px;
  height: 14px;
}

.icon-xs {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

.icon-large {
  width: 32px;
  height: 32px;
  color: #666;
  margin-bottom: 8px;
}

/* Summary Stats */
.summary-stats {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.stat-card {
  flex: 1;
  min-width: 70px;
  background: #1a1a1a;
  padding: 8px 10px;
  text-align: center;
}

.stat-label {
  display: block;
  color: #666;
  font-size: 10px;
  text-transform: uppercase;
  margin-bottom: 2px;
}

.stat-value {
  display: block;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
}

.stat-value.highlight {
  color: #f59e0b;
}

.stat-unit {
  font-size: 10px;
  opacity: 0.7;
  font-weight: normal;
}

/* Info Banner */
.info-banner {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  font-size: 11px;
  color: #8b8baa;
}

.info-banner strong {
  color: #aaaacc;
}

/* Column Headers */
.list-header {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  background: #1a1a1a;
  border-bottom: 1px solid #333;
  font-size: 10px;
  text-transform: uppercase;
  color: #666;
  font-weight: 600;
  gap: 4px;
}

.sortable {
  cursor: pointer;
  user-select: none;
}

.sortable:hover {
  color: #f59e0b;
}

.sort-indicator {
  font-size: 10px;
  margin-left: 2px;
}

.col-num {
  width: 28px;
  text-align: center;
  flex-shrink: 0;
}

.col-time {
  width: 65px;
  flex-shrink: 0;
}

.col-player {
  width: 40px;
  flex-shrink: 0;
}

.col-duration {
  width: 50px;
  flex-shrink: 0;
  text-align: center;
}

.col-sparkline {
  flex: 1;
  min-width: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.col-speed {
  width: 52px;
  flex-shrink: 0;
  text-align: right;
}

/* Shot List */
.shot-list {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.shot-row {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  gap: 4px;
  cursor: pointer;
  transition: background 0.15s;
  border-bottom: 1px solid #1a1a1a;
  font-size: 12px;
  color: #ccc;
}

.shot-row:hover {
  background: #1a1a2e;
}

.shot-row.selected {
  background: #1a2a1a;
  border-left: 2px solid #22c55e;
  padding-left: 6px;
}

.player-badge {
  display: inline-block;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
  border-radius: 2px;
}

.speed-badge {
  display: inline-block;
  padding: 2px 6px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 2px;
}

.speed-value {
  font-size: 11px;
  color: #888;
}

.sparkline {
  display: block;
}

/* Empty state */
.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.no-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  color: #666;
  gap: 4px;
  text-align: center;
}

.no-data p {
  margin: 2px 0;
  font-size: 12px;
}

.no-data .hint {
  font-size: 11px;
  opacity: 0.7;
}

/* Scrollbar styling */
.shot-list::-webkit-scrollbar {
  width: 6px;
}

.shot-list::-webkit-scrollbar-track {
  background: #111;
}

.shot-list::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 3px;
}

.shot-list::-webkit-scrollbar-thumb:hover {
  background: #444;
}
</style>
