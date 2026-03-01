/**
 * Advanced Analytics Composable
 *
 * Pure computation engine that derives advanced metrics from existing
 * skeleton_data, shuttle positions, and player tracking data.
 * No backend calls needed — everything is computed client-side.
 *
 * DATA REALITY (from backend audit):
 * - player.center: ALWAYS populated, in VIDEO PIXELS (not meters)
 * - player.current_speed: ALWAYS populated, in KM/H, ~50% are 0 (filtered by backend)
 * - player.pose?.pose_type: ALWAYS populated per-player (smash, overhead, serving, lunge, etc.)
 * - player.keypoints: ALWAYS populated, 17 COCO keypoints in VIDEO PIXELS
 * - frame.pose_classifications: NEVER POPULATED (always null/empty) — do NOT use
 * - frame.shuttle_position: SPARSE (~10-40% of frames), in VIDEO PIXELS
 * - frame.shuttle_speed_kmh: RARELY populated
 * - Frame sample rate: 1:1 with video FPS (no skipping)
 */

import { computed, ref, type Ref } from 'vue'
import type {
  AnalysisResult,
  SkeletonFrame,
  FramePlayer,
  Rally,
  RallyShot,
  ShotPlacement,
  ShotPlacementHeatmap,
  RecoveryEvent,
  FatigueSegment,
  FatigueProfile,
  ReactionEvent,
  MovementEfficiency,
  PressureEvent,
} from '@/types/analysis'

// =============================================================================
// UTILITY HELPERS
// =============================================================================

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function getCenter(player: FramePlayer): { x: number; y: number } {
  return player.center || { x: 0, y: 0 }
}

/** Hitting pose types from backend's classify_pose() */
const HITTING_POSES = new Set([
  'smash', 'overhead', 'serving', 'lunge', 'forehand', 'backhand',
  // Also accept trained model class names
  'serve', 'offense', 'lift',
])

/** Get the pose type string from a player, reading from player.pose (the ONLY source) */
function getPlayerPoseType(player: FramePlayer): string | null {
  const pose = (player as FramePlayer & { pose?: { pose_type?: string; confidence?: number } | null }).pose
  if (!pose) return null
  if ((pose.confidence ?? 0) < 0.3) return null
  return pose.pose_type || null
}

// =============================================================================
// COMPOSABLE
// =============================================================================

export function useAdvancedAnalytics(
  analysisResult: Ref<AnalysisResult | null>,
  currentFrame: Ref<number>
) {
  const isComputing = ref(false)

  // =========================================================================
  // 1. SHOT DETECTION (foundation for everything else)
  // =========================================================================

  /**
   * Detect shots using 3 methods in cascade:
   * 1. Shuttle trajectory direction changes (when shuttle data exists)
   * 2. Per-player pose detection (player.pose — always available)
   * 3. Player deceleration peaks (proven method from ShotSpeedList)
   */
  function detectAllShots(frames: SkeletonFrame[], fps: number): RallyShot[] {
    const MIN_GAP_FRAMES = Math.max(3, Math.floor(fps * 0.25))
    const MIN_GAP_SECONDS = 0.3

    // Build frame lookup map for O(1) access by frame number
    const frameMap = new Map<number, SkeletonFrame>()
    for (const f of frames) frameMap.set(f.frame, f)

    // --- Method 1: Shuttle trajectory direction changes ---
    const shots: RallyShot[] = []
    const shuttlePositions: { frame: number; ts: number; x: number; y: number }[] = []
    for (const f of frames) {
      if (f.shuttle_position?.x != null && f.shuttle_position?.y != null) {
        shuttlePositions.push({
          frame: f.frame,
          ts: f.timestamp,
          x: f.shuttle_position.x,
          y: f.shuttle_position.y,
        })
      }
    }

    if (shuttlePositions.length >= 5) {
      let lastShotFrame = -Infinity
      for (let i = 2; i < shuttlePositions.length; i++) {
        const p0 = shuttlePositions[i - 2]!
        const p1 = shuttlePositions[i - 1]!
        const p2 = shuttlePositions[i]!
        const vx1 = p1.x - p0.x
        const vy1 = p1.y - p0.y
        const vx2 = p2.x - p1.x
        const vy2 = p2.y - p1.y
        const dot = vx1 * vx2 + vy1 * vy2

        if (dot < 0 && (p1.frame - lastShotFrame) >= MIN_GAP_FRAMES) {
          const skFrame = frameMap.get(p1.frame)
          if (skFrame && skFrame.players.length > 0) {
            // Find closest player to shuttle
            let closest = skFrame.players[0]!
            let minD = dist(getCenter(closest), p1)
            for (const pl of skFrame.players) {
              const d = dist(getCenter(pl), p1)
              if (d < minD) { minD = d; closest = pl }
            }

            // Get shot type from player.pose (NOT frame.pose_classifications)
            const shotType = getPlayerPoseType(closest) || 'unknown'

            shots.push({
              frame: p1.frame,
              timestamp: p1.ts,
              playerId: closest.player_id,
              shotType,
              shuttlePosition: { x: p1.x, y: p1.y },
              playerPosition: getCenter(closest),
            })
            lastShotFrame = p1.frame
          }
        }
      }
    }

    if (shots.length >= 6) {
      shots.sort((a, b) => a.frame - b.frame)
      return shots
    }

    // --- Method 2: Per-player pose detection ---
    // player.pose is always populated; look for hitting poses
    const poseShots: RallyShot[] = []
    const lastPoseShotFrame = new Map<number, number>() // per player

    for (const f of frames) {
      for (const player of f.players) {
        const poseType = getPlayerPoseType(player)
        if (!poseType || !HITTING_POSES.has(poseType)) continue

        const pose = (player as FramePlayer & { pose?: { confidence?: number } }).pose
        if ((pose?.confidence ?? 0) < 0.5) continue

        const lastFrame = lastPoseShotFrame.get(player.player_id) ?? -Infinity
        if ((f.frame - lastFrame) < Math.floor(fps * 0.5)) continue

        poseShots.push({
          frame: f.frame,
          timestamp: f.timestamp,
          playerId: player.player_id,
          shotType: poseType,
          shuttlePosition: f.shuttle_position || null,
          playerPosition: getCenter(player),
        })
        lastPoseShotFrame.set(player.player_id, f.frame)
      }
    }

    if (poseShots.length >= 6) {
      // Merge with shuttle shots, deduplicate by frame proximity
      const merged = mergeShots([...shots, ...poseShots], MIN_GAP_FRAMES)
      if (merged.length >= 6) return merged
    }

    // --- Method 3: Deceleration-based detection (proven from ShotSpeedList) ---
    const decelShots = detectShotsFromDeceleration(frames, fps, frameMap)

    // Merge all methods, deduplicate
    const all = mergeShots([...shots, ...poseShots, ...decelShots], MIN_GAP_FRAMES)
    return all
  }

  /** Proven deceleration method adapted from ShotSpeedList.vue */
  function detectShotsFromDeceleration(frames: SkeletonFrame[], fps: number, frameMap: Map<number, SkeletonFrame>): RallyShot[] {
    if (frames.length < 10) return []

    const playerTimelines = new Map<number, {
      frame: number; timestamp: number; x: number; y: number; speed: number
    }[]>()

    for (const f of frames) {
      for (const player of f.players) {
        if (!player.center) continue
        let tl = playerTimelines.get(player.player_id)
        if (!tl) { tl = []; playerTimelines.set(player.player_id, tl) }
        tl.push({
          frame: f.frame,
          timestamp: f.timestamp,
          x: player.center.x,
          y: player.center.y,
          speed: player.current_speed ?? 0,
        })
      }
    }

    const result: RallyShot[] = []
    const MIN_SHOT_GAP = 0.6
    const DECEL_THRESHOLD = 2.0  // km/h drop
    const MIN_SPEED_BEFORE = 3.0 // km/h

    for (const [playerId, timeline] of playerTimelines) {
      if (timeline.length < 5) continue

      // 5-frame moving average
      const smooth: number[] = []
      for (let i = 0; i < timeline.length; i++) {
        let sum = 0, cnt = 0
        for (let j = Math.max(0, i - 2); j <= Math.min(timeline.length - 1, i + 2); j++) {
          sum += timeline[j]!.speed; cnt++
        }
        smooth.push(sum / cnt)
      }

      let lastShotTs = -Infinity
      for (let i = 3; i < smooth.length - 1; i++) {
        const prev3 = (smooth[i - 3]! + smooth[i - 2]! + smooth[i - 1]!) / 3
        const curr = smooth[i]!
        const decel = prev3 - curr
        const entry = timeline[i]!

        if (decel > DECEL_THRESHOLD && prev3 > MIN_SPEED_BEFORE && entry.timestamp - lastShotTs > MIN_SHOT_GAP) {
          // Get pose type from this frame
          const skFrame = frameMap.get(entry.frame)
          const player = skFrame?.players.find(p => p.player_id === playerId)
          const poseType = player ? (getPlayerPoseType(player) || 'unknown') : 'unknown'

          result.push({
            frame: entry.frame,
            timestamp: entry.timestamp,
            playerId,
            shotType: poseType,
            shuttlePosition: skFrame?.shuttle_position || null,
            playerPosition: { x: entry.x, y: entry.y },
          })
          lastShotTs = entry.timestamp
        }
      }
    }

    result.sort((a, b) => a.frame - b.frame)
    return result
  }

  /** Merge and deduplicate shots from multiple methods */
  function mergeShots(shots: RallyShot[], minGap: number): RallyShot[] {
    shots.sort((a, b) => a.frame - b.frame)
    const merged: RallyShot[] = []
    for (const shot of shots) {
      // Skip if too close to an already-accepted shot
      const last = merged[merged.length - 1]
      if (last && (shot.frame - last.frame) < minGap) {
        // Prefer the one with a known shot type
        if (last.shotType === 'unknown' && shot.shotType !== 'unknown') {
          merged[merged.length - 1] = shot
        }
        continue
      }
      merged.push(shot)
    }
    return merged
  }

  // =========================================================================
  // 2. RALLY SEGMENTATION
  // =========================================================================

  const rallies = computed<Rally[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length < 10) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const shots = detectAllShots(frames, fps)
    if (shots.length < 2) return []

    const RALLY_GAP_SECONDS = 3.0
    const detected: Rally[] = []
    let rallyStart = 0

    for (let i = 1; i < shots.length; i++) {
      const gap = shots[i]!.timestamp - shots[i - 1]!.timestamp
      if (gap > RALLY_GAP_SECONDS || i === shots.length - 1) {
        const rallyShots = shots.slice(rallyStart, i === shots.length - 1 ? i + 1 : i)
        if (rallyShots.length >= 2) {
          const first = rallyShots[0]!
          const last = rallyShots[rallyShots.length - 1]!
          detected.push({
            id: detected.length + 1,
            startFrame: first.frame,
            endFrame: last.frame,
            startTimestamp: first.timestamp,
            endTimestamp: last.timestamp,
            durationSeconds: last.timestamp - first.timestamp,
            shotCount: rallyShots.length,
            shots: rallyShots,
            winner: null,
          })
        }
        rallyStart = i
      }
    }

    return detected
  })

  // =========================================================================
  // 3. RALLY LENGTH DISTRIBUTION
  // =========================================================================

  const rallyLengthDistribution = computed(() => {
    const r = rallies.value
    if (r.length === 0) return { bins: [] as { label: string; count: number; avgDuration: number }[], avgShots: 0, avgDuration: 0 }

    const bins = [
      { label: '1-3', min: 1, max: 3, count: 0, totalDur: 0 },
      { label: '4-6', min: 4, max: 6, count: 0, totalDur: 0 },
      { label: '7-10', min: 7, max: 10, count: 0, totalDur: 0 },
      { label: '11-15', min: 11, max: 15, count: 0, totalDur: 0 },
      { label: '16-20', min: 16, max: 20, count: 0, totalDur: 0 },
      { label: '21+', min: 21, max: Infinity, count: 0, totalDur: 0 },
    ]

    let totalShots = 0
    let totalDuration = 0
    for (const rally of r) {
      totalShots += rally.shotCount
      totalDuration += rally.durationSeconds
      const bin = bins.find(b => rally.shotCount >= b.min && rally.shotCount <= b.max)
      if (bin) { bin.count++; bin.totalDur += rally.durationSeconds }
    }

    return {
      bins: bins.map(b => ({
        label: b.label,
        count: b.count,
        avgDuration: b.count > 0 ? b.totalDur / b.count : 0,
      })),
      avgShots: r.length > 0 ? totalShots / r.length : 0,
      avgDuration: r.length > 0 ? totalDuration / r.length : 0,
    }
  })

  // =========================================================================
  // 4. SHOT PLACEMENT HEATMAP
  // =========================================================================

  const shotPlacements = computed<ShotPlacement[]>(() => {
    if (!analysisResult.value?.skeleton_data) return []
    const placements: ShotPlacement[] = []
    for (const rally of rallies.value) {
      for (const shot of rally.shots) {
        if (shot.shuttlePosition) {
          placements.push({
            shotType: shot.shotType || 'unknown',
            position: shot.shuttlePosition,
            playerId: shot.playerId,
            frame: shot.frame,
          })
        }
      }
    }
    return placements
  })

  const shotPlacementsByType = computed<ShotPlacementHeatmap[]>(() => {
    const placements = shotPlacements.value
    if (placements.length < 3) return [] // Need minimum data for meaningful heatmap

    const GRID_SIZE = 8
    const typeMap = new Map<string, { grid: number[][]; total: number }>()

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const p of placements) {
      minX = Math.min(minX, p.position.x); maxX = Math.max(maxX, p.position.x)
      minY = Math.min(minY, p.position.y); maxY = Math.max(maxY, p.position.y)
    }
    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1

    const allGrid = Array.from({ length: GRID_SIZE }, () => new Array(GRID_SIZE).fill(0) as number[])
    let allTotal = 0

    for (const p of placements) {
      if (!typeMap.has(p.shotType)) {
        typeMap.set(p.shotType, {
          grid: Array.from({ length: GRID_SIZE }, () => new Array(GRID_SIZE).fill(0)),
          total: 0,
        })
      }
      const gx = clamp(Math.floor(((p.position.x - minX) / rangeX) * GRID_SIZE), 0, GRID_SIZE - 1)
      const gy = clamp(Math.floor(((p.position.y - minY) / rangeY) * GRID_SIZE), 0, GRID_SIZE - 1)
      const entry = typeMap.get(p.shotType)!
      entry.grid[gy]![gx]!++
      entry.total++
      allGrid[gy]![gx]!++
      allTotal++
    }

    const result: ShotPlacementHeatmap[] = [{ shotType: 'all', grid: allGrid, total: allTotal }]
    for (const [shotType, data] of typeMap) {
      result.push({ shotType, grid: data.grid, total: data.total })
    }
    return result
  })

  // =========================================================================
  // 5. RECOVERY POSITION ANALYSIS
  // =========================================================================

  const recoveryEvents = computed<RecoveryEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const events: RecoveryEvent[] = []

    const allShots: RallyShot[] = []
    for (const rally of rallies.value) allShots.push(...rally.shots)
    if (allShots.length < 2) return []

    // Build frame-number-to-index map for O(1) lookups
    const frameIndexMap = new Map<number, number>()
    for (let i = 0; i < frames.length; i++) frameIndexMap.set(frames[i]!.frame, i)

    // Precompute base positions (average position per player — always in pixels)
    const basePositions = new Map<number, { x: number; y: number }>()
    const posAccum = new Map<number, { sx: number; sy: number; n: number }>()
    for (const f of frames) {
      for (const p of f.players) {
        const c = getCenter(p)
        let acc = posAccum.get(p.player_id)
        if (!acc) { acc = { sx: 0, sy: 0, n: 0 }; posAccum.set(p.player_id, acc) }
        acc.sx += c.x; acc.sy += c.y; acc.n++
      }
    }
    for (const [pid, acc] of posAccum) {
      if (acc.n > 0) basePositions.set(pid, { x: acc.sx / acc.n, y: acc.sy / acc.n })
    }

    // Compute noise floor per player: median frame-to-frame displacement
    // This reflects natural keypoint jitter at this camera distance
    const playerNoiseFloor = new Map<number, number>()
    for (const [pid] of posAccum) {
      const displacements: number[] = []
      let prev: { x: number; y: number } | null = null
      for (const f of frames) {
        const player = f.players.find(p => p.player_id === pid)
        if (!player) continue
        const pos = getCenter(player)
        if (prev) {
          displacements.push(dist(prev, pos))
        }
        prev = pos
      }
      if (displacements.length > 10) {
        displacements.sort((a, b) => a - b)
        const median = displacements[Math.floor(displacements.length / 2)]!
        // Use 2x the noise floor as the stationary threshold
        playerNoiseFloor.set(pid, Math.max(2, median * 2))
      } else {
        playerNoiseFloor.set(pid, 8) // fallback
      }
    }

    for (const shot of allShots) {
      const pid = shot.playerId
      const basePos = basePositions.get(pid)
      if (!basePos) continue

      const shotFrameIdx = frameIndexMap.get(shot.frame) ?? frames.findIndex(f => f.frame >= shot.frame)
      if (shotFrameIdx < 0) continue

      const RECOVERY_WINDOW = Math.floor(fps * 2)
      let recoveryFrame: SkeletonFrame | null = null
      let recoveryPos: { x: number; y: number } | null = null

      // Use position displacement to detect when player stops moving
      // Threshold is calibrated to this video's noise floor
      const stationaryThreshold = playerNoiseFloor.get(pid) ?? 8
      let prevPos: { x: number; y: number } | null = null
      let stationaryCount = 0

      for (let j = shotFrameIdx + 1; j < Math.min(shotFrameIdx + RECOVERY_WINDOW, frames.length); j++) {
        const f = frames[j]!
        const player = f.players.find(pl => pl.player_id === pid)
        if (!player) continue

        const pos = getCenter(player)
        if (prevPos) {
          const displacement = dist(prevPos, pos)
          if (displacement < stationaryThreshold) {
            stationaryCount++
            if (stationaryCount >= 3) { // Stationary for 3+ consecutive frames
              recoveryFrame = f
              recoveryPos = pos
              break
            }
          } else {
            stationaryCount = 0
          }
        }
        prevPos = pos
      }

      if (recoveryFrame && recoveryPos) {
        const recoveryTime = recoveryFrame.timestamp - shot.timestamp
        if (recoveryTime < 0.1) continue // Too fast — likely noise
        const distFromBase = dist(recoveryPos, basePos)

        // Quality based PURELY on recovery time (reliable, unit-independent)
        let quality: RecoveryEvent['quality'] = 'poor'
        if (recoveryTime < 0.8) quality = 'excellent'
        else if (recoveryTime < 1.2) quality = 'good'
        else if (recoveryTime < 1.8) quality = 'fair'

        events.push({
          playerId: pid,
          shotFrame: shot.frame,
          shotTimestamp: shot.timestamp,
          recoveryFrame: recoveryFrame.frame,
          recoveryTimestamp: recoveryFrame.timestamp,
          recoveryTimeSeconds: recoveryTime,
          shotPosition: shot.playerPosition,
          recoveryPosition: recoveryPos,
          basePosition: basePos,
          distanceFromBase: distFromBase, // pixels — used only for relative comparison
          quality,
        })
      }
    }

    return events
  })

  const recoveryStats = computed(() => {
    const events = recoveryEvents.value
    if (events.length < 3) return null // Minimum 3 events for meaningful stats

    const playerMap = new Map<number, RecoveryEvent[]>()
    for (const e of events) {
      if (!playerMap.has(e.playerId)) playerMap.set(e.playerId, [])
      playerMap.get(e.playerId)!.push(e)
    }

    const perPlayer = Array.from(playerMap.entries()).map(([pid, evts]) => {
      const times = evts.map(e => e.recoveryTimeSeconds)
      const qualityCounts = { excellent: 0, good: 0, fair: 0, poor: 0 }
      for (const e of evts) qualityCounts[e.quality]++

      return {
        playerId: pid,
        avgRecoveryTime: times.reduce((a, b) => a + b, 0) / times.length,
        minRecoveryTime: Math.min(...times),
        maxRecoveryTime: Math.max(...times),
        totalRecoveries: evts.length,
        qualityDistribution: qualityCounts,
        avgDistanceFromBase: evts.reduce((a, e) => a + e.distanceFromBase, 0) / evts.length,
      }
    })

    return { perPlayer }
  })

  // =========================================================================
  // 6. FATIGUE DETECTION
  // =========================================================================

  const videoDuration = computed(() => {
    const frames = analysisResult.value?.skeleton_data
    if (!frames || frames.length < 2) return 0
    return frames[frames.length - 1]!.timestamp - frames[0]!.timestamp
  })

  const fatigueProfiles = computed<FatigueProfile[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length < 30) return []

    const frames = result.skeleton_data
    const totalDuration = videoDuration.value
    // Fatigue develops over 15-60+ minutes; analyzing short clips is meaningless
    if (totalDuration < 600) return [] // Minimum 10 minutes

    const NUM_SEGMENTS = 4
    const segmentDuration = totalDuration / NUM_SEGMENTS

    const playerIds = new Set<number>()
    for (const f of frames) for (const p of f.players) playerIds.add(p.player_id)

    // Compute position range per player for relative grid sizing
    const playerPosRange = new Map<number, { minX: number; maxX: number; minY: number; maxY: number }>()
    for (const f of frames) {
      for (const p of f.players) {
        const c = getCenter(p)
        let range = playerPosRange.get(p.player_id)
        if (!range) {
          range = { minX: c.x, maxX: c.x, minY: c.y, maxY: c.y }
          playerPosRange.set(p.player_id, range)
        }
        range.minX = Math.min(range.minX, c.x)
        range.maxX = Math.max(range.maxX, c.x)
        range.minY = Math.min(range.minY, c.y)
        range.maxY = Math.max(range.maxY, c.y)
      }
    }

    const profiles: FatigueProfile[] = []
    const GRID_CELLS = 10
    const totalGridCells = GRID_CELLS * GRID_CELLS
    const startTs = frames[0]!.timestamp

    // Precompute per-player grid config
    const playerGridConfig = new Map<number, { cellX: number; cellY: number; offX: number; offY: number }>()
    for (const pid of playerIds) {
      const posRange = playerPosRange.get(pid)
      playerGridConfig.set(pid, {
        cellX: posRange ? Math.max(1, (posRange.maxX - posRange.minX) / GRID_CELLS) : 50,
        cellY: posRange ? Math.max(1, (posRange.maxY - posRange.minY) / GRID_CELLS) : 50,
        offX: posRange?.minX ?? 0,
        offY: posRange?.minY ?? 0,
      })
    }

    // Single-pass: accumulate segment metrics for all players simultaneously
    type SegAccum = { totalSpeed: number; maxSpeed: number; nonZeroCount: number; positions: Set<string> }
    const segData = new Map<number, SegAccum[]>() // pid -> segments
    for (const pid of playerIds) {
      segData.set(pid, Array.from({ length: NUM_SEGMENTS }, () => ({
        totalSpeed: 0, maxSpeed: 0, nonZeroCount: 0, positions: new Set<string>(),
      })))
    }

    for (const f of frames) {
      const seg = Math.min(Math.floor((f.timestamp - startTs) / segmentDuration), NUM_SEGMENTS - 1)
      for (const player of f.players) {
        const accum = segData.get(player.player_id)?.[seg]
        if (!accum) continue

        const speed = player.current_speed || 0
        if (speed > 0) {
          accum.totalSpeed += speed
          accum.maxSpeed = Math.max(accum.maxSpeed, speed)
          accum.nonZeroCount++
        }

        const grid = playerGridConfig.get(player.player_id)!
        const pos = getCenter(player)
        const gx = Math.floor((pos.x - grid.offX) / grid.cellX)
        const gy = Math.floor((pos.y - grid.offY) / grid.cellY)
        accum.positions.add(`${gx},${gy}`)
      }
    }

    // Pre-bucket recovery events by player+segment
    const recoveryBuckets = new Map<string, RecoveryEvent[]>()
    for (const e of recoveryEvents.value) {
      const seg = Math.min(Math.floor((e.shotTimestamp - startTs) / segmentDuration), NUM_SEGMENTS - 1)
      const key = `${e.playerId}:${seg}`
      let bucket = recoveryBuckets.get(key)
      if (!bucket) { bucket = []; recoveryBuckets.set(key, bucket) }
      bucket.push(e)
    }

    for (const pid of playerIds) {
      const playerSegs = segData.get(pid)!
      const segments: FatigueSegment[] = []
      let hasEnoughData = true

      for (let seg = 0; seg < NUM_SEGMENTS; seg++) {
        const accum = playerSegs[seg]!
        if (accum.nonZeroCount < 5) { hasEnoughData = false; break }

        const segStart = startTs + seg * segmentDuration
        const segEnd = segStart + segmentDuration

        const segRecoveries = recoveryBuckets.get(`${pid}:${seg}`) || []
        const avgRecovery = segRecoveries.length > 0
          ? segRecoveries.reduce((a, e) => a + e.recoveryTimeSeconds, 0) / segRecoveries.length
          : 0

        segments.push({
          segmentIndex: seg,
          startTimestamp: segStart,
          endTimestamp: segEnd,
          playerId: pid,
          avgSpeed: accum.totalSpeed / accum.nonZeroCount,
          maxSpeed: accum.maxSpeed,
          distanceCovered: 0,
          avgRecoveryTime: avgRecovery,
          courtCoverage: (accum.positions.size / totalGridCells) * 100,
        })
      }

      if (!hasEnoughData || segments.length < NUM_SEGMENTS) continue

      const firstSeg = segments[0]!
      const lastSeg = segments[segments.length - 1]!

      const speedDecline = firstSeg.avgSpeed > 0
        ? ((firstSeg.avgSpeed - lastSeg.avgSpeed) / firstSeg.avgSpeed) * 100
        : 0
      const recoveryDecline = firstSeg.avgRecoveryTime > 0
        ? ((lastSeg.avgRecoveryTime - firstSeg.avgRecoveryTime) / firstSeg.avgRecoveryTime) * 100
        : 0
      const coverageDecline = firstSeg.courtCoverage > 0
        ? ((firstSeg.courtCoverage - lastSeg.courtCoverage) / firstSeg.courtCoverage) * 100
        : 0

      let fatigueOnset: number | null = null
      const peakSpeed = Math.max(...segments.map(s => s.avgSpeed))
      for (let i = 1; i < segments.length; i++) {
        if (segments[i]!.avgSpeed < peakSpeed * 0.9) { fatigueOnset = i; break }
      }

      profiles.push({
        playerId: pid,
        segments,
        speedDeclinePercent: speedDecline,
        recoveryDeclinePercent: recoveryDecline,
        coverageDeclinePercent: coverageDecline,
        fatigueOnsetSegment: fatigueOnset,
      })
    }

    return profiles
  })

  // =========================================================================
  // SHARED: Inter-player distance metrics (used by reaction + pressure)
  // =========================================================================

  const interPlayerDistMetrics = computed(() => {
    const frames = analysisResult.value?.skeleton_data
    if (!frames) return { avg: 500, max: 500 }
    let total = 0, count = 0, max = 0
    for (const f of frames) {
      if (f.players.length >= 2) {
        const d = dist(getCenter(f.players[0]!), getCenter(f.players[1]!))
        total += d; count++; max = Math.max(max, d)
      }
    }
    return { avg: count > 0 ? total / count : 500, max: max || 500 }
  })

  // =========================================================================
  // 7. REACTION TIME
  // =========================================================================

  const reactionEvents = computed<ReactionEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const events: ReactionEvent[] = []

    // Build frame-number-to-index map for O(1) lookups
    const frameIndexMap = new Map<number, number>()
    for (let i = 0; i < frames.length; i++) frameIndexMap.set(frames[i]!.frame, i)

    // Use 2.5% of average inter-player distance as displacement threshold
    // This self-calibrates to the video's scale/zoom level
    const DISPLACEMENT_THRESHOLD = Math.max(5, interPlayerDistMetrics.value.avg * 0.025)

    for (const rally of rallies.value) {
      for (let i = 1; i < rally.shots.length; i++) {
        const prevShot = rally.shots[i - 1]!
        const currShot = rally.shots[i]!

        if (prevShot.playerId === currShot.playerId) continue

        const respondingPlayer = currShot.playerId
        const opponentShotFrame = prevShot.frame

        const startIdx = frameIndexMap.get(opponentShotFrame) ?? frames.findIndex(f => f.frame >= opponentShotFrame)
        if (startIdx < 0) continue

        const basePlayer = frames[startIdx]?.players.find(p => p.player_id === respondingPlayer)
        if (!basePlayer) continue
        const basePos = getCenter(basePlayer)

        let firstMovementFrame: number | null = null

        for (let j = startIdx + 1; j < Math.min(startIdx + Math.floor(fps * 1.5), frames.length); j++) {
          const f = frames[j]!
          const player = f.players.find(p => p.player_id === respondingPlayer)
          if (!player) continue

          const displacement = dist(basePos, getCenter(player))
          if (displacement > DISPLACEMENT_THRESHOLD) {
            firstMovementFrame = f.frame
            break
          }
        }

        if (firstMovementFrame !== null) {
          const reactionTimeMs = ((firstMovementFrame - opponentShotFrame) / fps) * 1000
          if (reactionTimeMs > 50 && reactionTimeMs < 1500) {
            events.push({
              playerId: respondingPlayer,
              opponentShotFrame,
              firstMovementFrame,
              reactionTimeMs,
              opponentShotTimestamp: prevShot.timestamp,
              movementStartTimestamp: prevShot.timestamp + reactionTimeMs / 1000,
            })
          }
        }
      }
    }

    return events
  })

  const reactionStats = computed(() => {
    const events = reactionEvents.value
    if (events.length < 3) return null // Minimum 3 measurements for reliability

    const playerMap = new Map<number, number[]>()
    for (const e of events) {
      if (!playerMap.has(e.playerId)) playerMap.set(e.playerId, [])
      playerMap.get(e.playerId)!.push(e.reactionTimeMs)
    }

    return Array.from(playerMap.entries()).map(([pid, times]) => {
      const sorted = [...times].sort((a, b) => a - b)
      return {
        playerId: pid,
        avgReactionMs: times.reduce((a, b) => a + b, 0) / times.length,
        minReactionMs: Math.min(...times),
        maxReactionMs: Math.max(...times),
        medianReactionMs: sorted[Math.floor(sorted.length / 2)] || 0,
        totalMeasured: times.length,
      }
    })
  })

  // =========================================================================
  // 10. MOVEMENT EFFICIENCY
  // =========================================================================

  const movementEfficiency = computed<MovementEfficiency[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length < 10) return []

    const frames = result.skeleton_data
    const playerIds = new Set<number>()
    for (const f of frames) for (const p of f.players) playerIds.add(p.player_id)

    const efficiencies: MovementEfficiency[] = []

    for (const pid of playerIds) {
      const playerPositions: { x: number; y: number }[] = []
      for (const f of frames) {
        const player = f.players.find(p => p.player_id === pid)
        if (player) playerPositions.push(getCenter(player))
      }

      if (playerPositions.length < 30) continue

      // Directness ratio is unit-independent (works in pixels)
      // displacement / path-length for each segment
      let segmentCount = 0
      let directnessSum = 0
      const SEGMENT_SIZE = 30

      for (let i = SEGMENT_SIZE; i < playerPositions.length; i += SEGMENT_SIZE) {
        const segStart = playerPositions[i - SEGMENT_SIZE]!
        const segEnd = playerPositions[i]!
        const displacement = dist(segStart, segEnd)
        let segDist = 0
        for (let j = i - SEGMENT_SIZE; j < i; j++) {
          segDist += dist(playerPositions[j]!, playerPositions[j + 1]!)
        }
        if (segDist > 5) { // Minimum movement to count (avoid div-by-near-zero)
          directnessSum += displacement / segDist
          segmentCount++
        }
      }

      if (segmentCount < 3) continue // Need enough segments for reliable score

      const avgDirectness = directnessSum / segmentCount

      efficiencies.push({
        playerId: pid,
        totalDistance: 0,    // Not displayed — pixel distance is misleading
        usefulDistance: 0,   // Not displayed
        wastedDistance: 0,   // Not displayed
        efficiencyScore: clamp(avgDirectness * 100, 0, 100),
        avgDirectness,
      })
    }

    return efficiencies
  })

  // =========================================================================
  // 11. PRESSURE INDEX
  // =========================================================================

  const pressureEvents = computed<PressureEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const events: PressureEvent[] = []

    // Build frame lookup map for O(1) access
    const frameMap = new Map<number, SkeletonFrame>()
    for (const f of frames) frameMap.set(f.frame, f)

    const maxPlayerDist = interPlayerDistMetrics.value.max

    for (const rally of rallies.value) {
      for (let i = 0; i < rally.shots.length; i++) {
        const shot = rally.shots[i]!
        const hitterId = shot.playerId

        const skFrame = frameMap.get(shot.frame)
        if (!skFrame) continue

        const hitter = skFrame.players.find(p => p.player_id === hitterId)
        const opponent = skFrame.players.find(p => p.player_id !== hitterId)
        if (!hitter || !opponent) continue

        const opponentPos = getCenter(opponent)

        // Simplified to 2 reliable factors (50/50 weighting)

        // Factor 1: Opponent travel distance normalized by max observed distance (50%)
        const opponentTravelDist = shot.shuttlePosition
          ? dist(shot.shuttlePosition, opponentPos)
          : dist(shot.playerPosition, opponentPos)
        const travelFactor = clamp((opponentTravelDist / maxPlayerDist) * 100, 0, 100)

        // Factor 2: Time pressure — shorter time between shots = more pressure (50%)
        let timeFactor = 50 // default mid-range
        if (i + 1 < rally.shots.length) {
          const nextShot = rally.shots[i + 1]!
          const timeBetween = nextShot.timestamp - shot.timestamp
          // 0.3s = max pressure (100), 2.0s = low pressure (0)
          timeFactor = clamp((1 - (timeBetween - 0.3) / 1.7) * 100, 0, 100)
        }

        const pressureScore = clamp(
          travelFactor * 0.5 + timeFactor * 0.5,
          0, 100
        )

        events.push({
          frame: shot.frame,
          timestamp: shot.timestamp,
          playerId: hitterId,
          pressureScore,
          factors: {
            shotSpeed: 0,
            placementDifficulty: travelFactor,
            recoveryTimeForced: timeFactor,
            courtPositionAdvantage: 0,
          },
        })
      }
    }

    return events
  })

  // =========================================================================
  // SUMMARY / CURRENT STATE
  // =========================================================================

  const currentRally = computed<Rally | null>(() => {
    const frame = currentFrame.value
    return rallies.value.find(r => frame >= r.startFrame && frame <= r.endFrame) || null
  })

  // =========================================================================
  // RETURN
  // =========================================================================

  return {
    rallies,
    rallyLengthDistribution,
    currentRally,
    shotPlacements,
    shotPlacementsByType,
    recoveryEvents,
    recoveryStats,
    fatigueProfiles,
    videoDuration,
    reactionEvents,
    reactionStats,
    movementEfficiency,
    pressureEvents,
    isComputing,
  }
}
