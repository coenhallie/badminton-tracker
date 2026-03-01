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
  MomentumPoint,
  ShotPattern,
  MovementEfficiency,
  PressureEvent,
  KineticChainEvent,
  BenchmarkComparison,
} from '@/types/analysis'
import { PRO_BENCHMARKS } from '@/types/analysis'

// =============================================================================
// UTILITY HELPERS
// =============================================================================

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function percentile(value: number, min: number, max: number): number {
  if (max <= min) return 50
  return clamp(((value - min) / (max - min)) * 100, 0, 100)
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

/** Aggressive pose types for momentum calculation */
const AGGRESSIVE_POSES = new Set(['smash', 'overhead', 'offense', 'serving', 'serve', 'forehand'])
const DEFENSIVE_POSES = new Set(['defense', 'lift', 'lunge', 'recovery'])

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
          const skFrame = frames.find(f => f.frame === p1.frame)
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
    const decelShots = detectShotsFromDeceleration(frames, fps)

    // Merge all methods, deduplicate
    const all = mergeShots([...shots, ...poseShots, ...decelShots], MIN_GAP_FRAMES)
    return all
  }

  /** Proven deceleration method adapted from ShotSpeedList.vue */
  function detectShotsFromDeceleration(frames: SkeletonFrame[], fps: number): RallyShot[] {
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
          const skFrame = frames.find(f => f.frame === entry.frame)
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

    for (const p of placements) {
      if (!typeMap.has(p.shotType)) {
        typeMap.set(p.shotType, {
          grid: Array.from({ length: GRID_SIZE }, () => new Array(GRID_SIZE).fill(0)),
          total: 0,
        })
      }
      const entry = typeMap.get(p.shotType)!
      const gx = clamp(Math.floor(((p.position.x - minX) / rangeX) * GRID_SIZE), 0, GRID_SIZE - 1)
      const gy = clamp(Math.floor(((p.position.y - minY) / rangeY) * GRID_SIZE), 0, GRID_SIZE - 1)
      entry.grid[gy]![gx]!++
      entry.total++
    }

    const allGrid = Array.from({ length: GRID_SIZE }, () => new Array(GRID_SIZE).fill(0) as number[])
    let allTotal = 0
    for (const p of placements) {
      const gx = clamp(Math.floor(((p.position.x - minX) / rangeX) * GRID_SIZE), 0, GRID_SIZE - 1)
      const gy = clamp(Math.floor(((p.position.y - minY) / rangeY) * GRID_SIZE), 0, GRID_SIZE - 1)
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

    for (const shot of allShots) {
      const pid = shot.playerId
      const basePos = basePositions.get(pid)
      if (!basePos) continue

      const shotFrameIdx = frames.findIndex(f => f.frame >= shot.frame)
      if (shotFrameIdx < 0) continue

      const RECOVERY_WINDOW = Math.floor(fps * 2)
      let recoveryFrame: SkeletonFrame | null = null
      let recoveryPos: { x: number; y: number } | null = null

      // Use position displacement to detect when player stops moving
      // (more reliable than current_speed which has 50% zero values)
      let prevPos: { x: number; y: number } | null = null
      let stationaryCount = 0

      for (let j = shotFrameIdx + 1; j < Math.min(shotFrameIdx + RECOVERY_WINDOW, frames.length); j++) {
        const f = frames[j]!
        const player = f.players.find(pl => pl.player_id === pid)
        if (!player) continue

        const pos = getCenter(player)
        if (prevPos) {
          const displacement = dist(prevPos, pos)
          if (displacement < 8) { // pixels — nearly stationary between frames
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

  const fatigueProfiles = computed<FatigueProfile[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length < 30) return []

    const frames = result.skeleton_data
    const totalDuration = frames[frames.length - 1]!.timestamp - frames[0]!.timestamp
    if (totalDuration < 10) return []

    const NUM_SEGMENTS = 4
    const segmentDuration = totalDuration / NUM_SEGMENTS

    const playerIds = new Set<number>()
    for (const f of frames) for (const p of f.players) playerIds.add(p.player_id)

    const profiles: FatigueProfile[] = []

    for (const pid of playerIds) {
      const segments: FatigueSegment[] = []
      let hasEnoughData = true

      for (let seg = 0; seg < NUM_SEGMENTS; seg++) {
        const segStart = frames[0]!.timestamp + seg * segmentDuration
        const segEnd = segStart + segmentDuration
        const segFrames = frames.filter(f => f.timestamp >= segStart && f.timestamp < segEnd)

        // FIXED: Only include non-zero speeds (backend sets speed=0 for unreliable measurements)
        let totalSpeed = 0
        let maxSpeed = 0
        let nonZeroSpeedCount = 0
        const positions = new Set<string>()

        for (const f of segFrames) {
          const player = f.players.find(p => p.player_id === pid)
          if (!player) continue

          const speed = player.current_speed || 0
          if (speed > 0) {
            totalSpeed += speed
            maxSpeed = Math.max(maxSpeed, speed)
            nonZeroSpeedCount++
          }

          // Grid-based court coverage (relative metric — works in pixels)
          const pos = getCenter(player)
          const gx = Math.floor(pos.x / 50)
          const gy = Math.floor(pos.y / 50)
          positions.add(`${gx},${gy}`)
        }

        // Require minimum non-zero speed samples for reliable segment data
        if (nonZeroSpeedCount < 5) {
          hasEnoughData = false
          break
        }

        // Get recovery events for this segment
        const segRecoveries = recoveryEvents.value.filter(
          e => e.playerId === pid && e.shotTimestamp >= segStart && e.shotTimestamp < segEnd
        )
        const avgRecovery = segRecoveries.length > 0
          ? segRecoveries.reduce((a, e) => a + e.recoveryTimeSeconds, 0) / segRecoveries.length
          : 0

        segments.push({
          segmentIndex: seg,
          startTimestamp: segStart,
          endTimestamp: segEnd,
          playerId: pid,
          avgSpeed: totalSpeed / nonZeroSpeedCount, // Only non-zero speeds
          maxSpeed,
          distanceCovered: 0, // Not displayed — pixel distance is meaningless without calibration
          avgRecoveryTime: avgRecovery,
          courtCoverage: (positions.size / 100) * 100,
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
  // 7. REACTION TIME
  // =========================================================================

  const reactionEvents = computed<ReactionEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const events: ReactionEvent[] = []

    for (const rally of rallies.value) {
      for (let i = 1; i < rally.shots.length; i++) {
        const prevShot = rally.shots[i - 1]!
        const currShot = rally.shots[i]!

        if (prevShot.playerId === currShot.playerId) continue

        const respondingPlayer = currShot.playerId
        const opponentShotFrame = prevShot.frame

        const startIdx = frames.findIndex(f => f.frame >= opponentShotFrame)
        if (startIdx < 0) continue

        // FIXED: Use position displacement instead of filtered current_speed
        // current_speed has ~50% zero values from backend filtering — unreliable for detecting movement start
        const basePlayer = frames[startIdx]?.players.find(p => p.player_id === respondingPlayer)
        if (!basePlayer) continue
        const basePos = getCenter(basePlayer)

        const DISPLACEMENT_THRESHOLD = 15 // pixels — meaningful movement start
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
  // 8. RALLY MOMENTUM
  // =========================================================================

  const momentumTimeline = computed<MomentumPoint[]>(() => {
    if (!analysisResult.value?.skeleton_data) return []

    const frames = analysisResult.value.skeleton_data
    const points: MomentumPoint[] = []
    let momentum = 0

    for (const rally of rallies.value) {
      for (const shot of rally.shots) {
        const direction = shot.playerId === 0 ? 1 : -1

        // FIXED: Use player.pose types (now correctly read), with speed-based fallback
        let aggressiveness = 0.05 // base — neutral shot
        if (AGGRESSIVE_POSES.has(shot.shotType)) {
          aggressiveness = 0.15
        } else if (DEFENSIVE_POSES.has(shot.shotType)) {
          aggressiveness = -0.05
        } else if (shot.shotType === 'unknown') {
          // Fallback: use player speed as aggression proxy
          const skFrame = frames.find(f => f.frame === shot.frame)
          const player = skFrame?.players.find(p => p.player_id === shot.playerId)
          const speed = player?.current_speed || 0
          if (speed > 8) aggressiveness = 0.12       // fast movement = aggressive
          else if (speed > 4) aggressiveness = 0.07
          // else keep 0.05 default
        }

        momentum += direction * aggressiveness
        momentum = clamp(momentum, -1, 1)
        momentum *= 0.92

        points.push({
          frame: shot.frame,
          timestamp: shot.timestamp,
          momentum,
          reason: `P${shot.playerId + 1} ${shot.shotType}`,
        })
      }
      momentum *= 0.5
    }

    return points
  })

  // =========================================================================
  // 9. SHOT SEQUENCE PATTERNS
  // =========================================================================

  const shotPatterns = computed<ShotPattern[]>(() => {
    // Need minimum shots for meaningful patterns
    const totalShots = rallies.value.reduce((s, r) => s + r.shotCount, 0)
    if (totalShots < 5) return []

    const patternMap = new Map<string, { count: number; playerId: number }>()

    for (const rally of rallies.value) {
      // 2-shot sequences from same player (alternating shots)
      for (let i = 0; i < rally.shots.length - 2; i++) {
        const s1 = rally.shots[i]!
        const s3 = rally.shots[i + 2]!

        if (s1.playerId === s3.playerId && s1.shotType !== 'unknown' && s3.shotType !== 'unknown') {
          const seq = [s1.shotType, s3.shotType]
          const key = `${s1.playerId}:${seq.join('\u2192')}`
          if (!patternMap.has(key)) patternMap.set(key, { count: 0, playerId: s1.playerId })
          patternMap.get(key)!.count++
        }
      }

      // 3-consecutive-shot patterns from same player
      const playerShots = new Map<number, RallyShot[]>()
      for (const shot of rally.shots) {
        if (!playerShots.has(shot.playerId)) playerShots.set(shot.playerId, [])
        playerShots.get(shot.playerId)!.push(shot)
      }

      for (const [pid, shots] of playerShots) {
        for (let i = 0; i < shots.length - 2; i++) {
          // Skip patterns with 'unknown' types — not informative
          if (shots[i]!.shotType === 'unknown' || shots[i + 1]!.shotType === 'unknown' || shots[i + 2]!.shotType === 'unknown') continue
          const seq = [shots[i]!.shotType, shots[i + 1]!.shotType, shots[i + 2]!.shotType]
          const key = `${pid}:${seq.join('\u2192')}`
          if (!patternMap.has(key)) patternMap.set(key, { count: 0, playerId: pid })
          patternMap.get(key)!.count++
        }
      }
    }

    return Array.from(patternMap.entries())
      .map(([key, data]) => ({
        sequence: key.split(':')[1]?.split('\u2192') || [],
        count: data.count,
        successRate: null,
        playerId: data.playerId,
      }))
      .filter(p => p.count >= 2)
      .sort((a, b) => b.count - a.count)
      .slice(0, 15)
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

    // Precompute max distance for normalization
    let maxPlayerDist = 0
    for (const f of frames) {
      if (f.players.length >= 2) {
        const d = dist(getCenter(f.players[0]!), getCenter(f.players[1]!))
        maxPlayerDist = Math.max(maxPlayerDist, d)
      }
    }
    if (maxPlayerDist === 0) maxPlayerDist = 500 // fallback

    for (const rally of rallies.value) {
      for (let i = 0; i < rally.shots.length; i++) {
        const shot = rally.shots[i]!
        const hitterId = shot.playerId

        const skFrame = frames.find(f => f.frame === shot.frame)
        if (!skFrame) continue

        const hitter = skFrame.players.find(p => p.player_id === hitterId)
        const opponent = skFrame.players.find(p => p.player_id !== hitterId)
        if (!hitter || !opponent) continue

        const hitterPos = getCenter(hitter)
        const opponentPos = getCenter(opponent)

        // FIXED: Only use reliably-available factors (removed shuttle_speed_kmh)

        // Factor 1: Distance opponent must travel (40%)
        // Higher distance from opponent to shot position = more pressure
        const opponentTravelDist = shot.shuttlePosition
          ? dist(shot.shuttlePosition, opponentPos)
          : dist(shot.playerPosition, opponentPos) // Use hitter position as proxy
        const travelFactor = clamp((opponentTravelDist / maxPlayerDist) * 100, 0, 100)

        // Factor 2: Time pressure — shorter time between shots = more pressure (30%)
        let timeFactor = 50 // default mid-range
        if (i + 1 < rally.shots.length) {
          const nextShot = rally.shots[i + 1]!
          const timeBetween = nextShot.timestamp - shot.timestamp
          // 0.3s = max pressure (100), 2.0s = low pressure (0)
          timeFactor = clamp((1 - (timeBetween - 0.3) / 1.7) * 100, 0, 100)
        }

        // Factor 3: Court position — how displaced the opponent is from their base (30%)
        const opponentDisplacement = dist(opponentPos, hitterPos)
        const positionFactor = clamp((opponentDisplacement / maxPlayerDist) * 100, 0, 100)

        const pressureScore = clamp(
          travelFactor * 0.4 + timeFactor * 0.3 + positionFactor * 0.3,
          0, 100
        )

        events.push({
          frame: shot.frame,
          timestamp: shot.timestamp,
          playerId: hitterId,
          pressureScore,
          factors: {
            shotSpeed: 0, // Not available reliably
            placementDifficulty: travelFactor,
            recoveryTimeForced: timeFactor,
            courtPositionAdvantage: positionFactor,
          },
        })
      }
    }

    return events
  })

  // =========================================================================
  // 12. KINETIC CHAIN ANALYSIS
  // =========================================================================

  const kineticChainEvents = computed<KineticChainEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const events: KineticChainEvent[] = []

    const JOINTS = [
      { name: 'hip', indices: [11, 12] },
      { name: 'shoulder', indices: [5, 6] },
      { name: 'elbow', indices: [7, 8] },
      { name: 'wrist', indices: [9, 10] },
    ]

    // FIXED: Accept power shots from player.pose (now correctly read via detectAllShots)
    const POWER_POSES = new Set(['smash', 'overhead', 'serving', 'serve', 'offense', 'forehand', 'clear'])

    for (const rally of rallies.value) {
      for (const shot of rally.shots) {
        // Accept shots with known power pose types
        if (!POWER_POSES.has(shot.shotType)) continue

        const shotIdx = frames.findIndex(f => f.frame === shot.frame)
        if (shotIdx < 0) continue

        const WINDOW = 10
        const startIdx = Math.max(0, shotIdx - WINDOW)
        const endIdx = Math.min(frames.length - 1, shotIdx + WINDOW)
        const windowFrames = frames.slice(startIdx, endIdx + 1)

        const player = frames[shotIdx]?.players.find(p => p.player_id === shot.playerId)
        if (!player?.keypoints || player.keypoints.length < 17) continue

        const chainSequence: KineticChainEvent['chainSequence'] = []

        for (const joint of JOINTS) {
          const velocities: { frame: number; vel: number }[] = []

          for (let i = 1; i < windowFrames.length; i++) {
            const prevFrame = windowFrames[i - 1]!
            const currFrame = windowFrames[i]!
            const prevPlayer = prevFrame.players.find(p => p.player_id === shot.playerId)
            const currPlayer = currFrame.players.find(p => p.player_id === shot.playerId)

            if (!prevPlayer?.keypoints?.length || !currPlayer?.keypoints?.length) continue

            let totalVel = 0
            let validCount = 0
            for (const idx of joint.indices) {
              const prevKp = prevPlayer.keypoints[idx]
              const currKp = currPlayer.keypoints[idx]
              if (prevKp?.x != null && prevKp?.y != null && currKp?.x != null && currKp?.y != null) {
                const dt = (currFrame.frame - prevFrame.frame) / fps
                if (dt > 0) {
                  totalVel += dist({ x: prevKp.x, y: prevKp.y }, { x: currKp.x, y: currKp.y }) / dt
                  validCount++
                }
              }
            }

            if (validCount > 0) {
              velocities.push({ frame: currFrame.frame, vel: totalVel / validCount })
            }
          }

          if (velocities.length > 0) {
            const peak = velocities.reduce((max, v) => v.vel > max.vel ? v : max, velocities[0]!)
            chainSequence.push({
              joint: joint.name,
              peakFrame: peak.frame,
              peakVelocity: peak.vel,
              timing: ((peak.frame - shot.frame) / fps) * 1000,
            })
          }
        }

        if (chainSequence.length >= 3) {
          let score = 50
          for (let i = 1; i < chainSequence.length; i++) {
            if (chainSequence[i]!.timing > chainSequence[i - 1]!.timing) score += 12.5
            else score -= 10
          }

          events.push({
            frame: shot.frame,
            timestamp: shot.timestamp,
            playerId: shot.playerId,
            shotType: shot.shotType,
            chainSequence,
            chainScore: clamp(score, 0, 100),
          })
        }
      }
    }

    return events
  })

  // =========================================================================
  // 13. BENCHMARK AGAINST PROS
  // =========================================================================

  const benchmarkComparisons = computed<BenchmarkComparison[]>(() => {
    const result = analysisResult.value
    if (!result?.players?.length) return []

    const comparisons: BenchmarkComparison[] = []
    const benchmarks = PRO_BENCHMARKS

    // Average speed (already in km/h from backend)
    const avgSpeed = result.players.reduce((s, p) => s + p.avg_speed, 0) / result.players.length
    comparisons.push({
      metric: 'Average Speed',
      playerValue: avgSpeed,
      proAverage: benchmarks.avgSpeed.avg,
      proRange: { min: benchmarks.avgSpeed.min, max: benchmarks.avgSpeed.max },
      percentile: percentile(avgSpeed, benchmarks.avgSpeed.min, benchmarks.avgSpeed.max),
      unit: benchmarks.avgSpeed.unit,
    })

    // Max speed
    const maxSpd = Math.max(...result.players.map(p => p.max_speed))
    comparisons.push({
      metric: 'Max Speed',
      playerValue: maxSpd,
      proAverage: benchmarks.maxSpeed.avg,
      proRange: { min: benchmarks.maxSpeed.min, max: benchmarks.maxSpeed.max },
      percentile: percentile(maxSpd, benchmarks.maxSpeed.min, benchmarks.maxSpeed.max),
      unit: benchmarks.maxSpeed.unit,
    })

    // REMOVED: totalDistance comparison — clip is NOT a full game, comparing to "per game" stats is misleading
    // REMOVED: smashSpeed — shuttle speed rarely available

    // Reaction time (if enough data)
    const rStats = reactionStats.value
    if (rStats && rStats.length > 0) {
      const avgReaction = rStats.reduce((s, p) => s + p.avgReactionMs, 0) / rStats.length
      comparisons.push({
        metric: 'Reaction Time',
        playerValue: avgReaction,
        proAverage: benchmarks.reactionTime.avg,
        proRange: { min: benchmarks.reactionTime.min, max: benchmarks.reactionTime.max },
        percentile: 100 - percentile(avgReaction, benchmarks.reactionTime.min, benchmarks.reactionTime.max),
        unit: benchmarks.reactionTime.unit,
      })
    }

    // Recovery time
    const recStats = recoveryStats.value
    if (recStats && recStats.perPlayer.length > 0) {
      const avgRecovery = recStats.perPlayer.reduce((s, p) => s + p.avgRecoveryTime, 0) / recStats.perPlayer.length
      comparisons.push({
        metric: 'Recovery Time',
        playerValue: avgRecovery,
        proAverage: benchmarks.recoveryTime.avg,
        proRange: { min: benchmarks.recoveryTime.min, max: benchmarks.recoveryTime.max },
        percentile: 100 - percentile(avgRecovery, benchmarks.recoveryTime.min, benchmarks.recoveryTime.max),
        unit: benchmarks.recoveryTime.unit,
      })
    }

    // Rally stats
    if (rallies.value.length >= 3) { // Need minimum rallies for meaningful comparison
      const avgShotsPerRally = rallies.value.reduce((s, r) => s + r.shotCount, 0) / rallies.value.length
      comparisons.push({
        metric: 'Shots per Rally',
        playerValue: avgShotsPerRally,
        proAverage: benchmarks.shotsPerRally.avg,
        proRange: { min: benchmarks.shotsPerRally.min, max: benchmarks.shotsPerRally.max },
        percentile: percentile(avgShotsPerRally, benchmarks.shotsPerRally.min, benchmarks.shotsPerRally.max),
        unit: benchmarks.shotsPerRally.unit,
      })

      const avgRallyDur = rallies.value.reduce((s, r) => s + r.durationSeconds, 0) / rallies.value.length
      comparisons.push({
        metric: 'Rally Duration',
        playerValue: avgRallyDur,
        proAverage: benchmarks.rallyDuration.avg,
        proRange: { min: benchmarks.rallyDuration.min, max: benchmarks.rallyDuration.max },
        percentile: percentile(avgRallyDur, benchmarks.rallyDuration.min, benchmarks.rallyDuration.max),
        unit: benchmarks.rallyDuration.unit,
      })
    }

    // Movement efficiency
    if (movementEfficiency.value.length > 0) {
      const avgEff = movementEfficiency.value.reduce((s, e) => s + e.efficiencyScore, 0) / movementEfficiency.value.length
      comparisons.push({
        metric: 'Movement Efficiency',
        playerValue: avgEff,
        proAverage: benchmarks.movementEfficiency.avg,
        proRange: { min: benchmarks.movementEfficiency.min, max: benchmarks.movementEfficiency.max },
        percentile: percentile(avgEff, benchmarks.movementEfficiency.min, benchmarks.movementEfficiency.max),
        unit: benchmarks.movementEfficiency.unit,
      })
    }

    return comparisons
  })

  // =========================================================================
  // SUMMARY / CURRENT STATE
  // =========================================================================

  const currentRally = computed<Rally | null>(() => {
    const frame = currentFrame.value
    return rallies.value.find(r => frame >= r.startFrame && frame <= r.endFrame) || null
  })

  const currentMomentum = computed(() => {
    const frame = currentFrame.value
    const pts = momentumTimeline.value
    if (pts.length === 0) return 0
    let latest = pts[0]!
    for (const p of pts) {
      if (p.frame <= frame) latest = p
      else break
    }
    return latest.momentum
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
    reactionEvents,
    reactionStats,
    momentumTimeline,
    currentMomentum,
    shotPatterns,
    movementEfficiency,
    pressureEvents,
    kineticChainEvents,
    benchmarkComparisons,
    isComputing,
  }
}
