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
  ReactionEvent,
  MovementEfficiency,
} from '@/types/analysis'
import { COURT_DIMENSIONS } from '@/types/analysis'
import { computeHomographyFromKeypoints, applyHomography } from '@/utils/homography'

const COURT_LENGTH = COURT_DIMENSIONS.length      // 13.4m
const COURT_WIDTH = COURT_DIMENSIONS.width_doubles // 6.1m

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
  currentFrame: Ref<number>,
  courtKeypoints?: Ref<number[][] | null>
) {
  const isComputing = ref(false)

  // =========================================================================
  // HOMOGRAPHY (video pixels → court meters)
  // =========================================================================

  const homographyMatrix = computed(() => {
    const kp = courtKeypoints?.value
    if (!kp || kp.length < 4) return null
    return computeHomographyFromKeypoints(kp)
  })

  /** Transform a video pixel position to court meters. Returns null if no homography. */
  function toCourtMeters(px: { x: number; y: number }): { x: number; y: number } | null {
    const H = homographyMatrix.value
    if (!H) return null
    const result = applyHomography(H, px.x, px.y)
    if (!result) return null
    // Clamp to court bounds with small margin for noise
    if (result.x < -1 || result.x > COURT_WIDTH + 1 || result.y < -1 || result.y > COURT_LENGTH + 1) return null
    return result
  }

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
    const poseShots: RallyShot[] = []
    const lastPoseShotFrame = new Map<number, number>()

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
      const merged = mergeShots([...shots, ...poseShots], MIN_GAP_FRAMES)
      if (merged.length >= 6) return merged
    }

    // --- Method 3: Deceleration-based detection ---
    const decelShots = detectShotsFromDeceleration(frames, fps, frameMap)

    const all = mergeShots([...shots, ...poseShots, ...decelShots], MIN_GAP_FRAMES)
    return all
  }

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
    const DECEL_THRESHOLD = 2.0
    const MIN_SPEED_BEFORE = 3.0

    for (const [playerId, timeline] of playerTimelines) {
      if (timeline.length < 5) continue

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

  function mergeShots(shots: RallyShot[], minGap: number): RallyShot[] {
    shots.sort((a, b) => a.frame - b.frame)
    const merged: RallyShot[] = []
    for (const shot of shots) {
      const last = merged[merged.length - 1]
      if (last && (shot.frame - last.frame) < minGap) {
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
  // 4. SHOT PLACEMENT HEATMAP (court-aware with homography)
  // =========================================================================

  const shotPlacements = computed<ShotPlacement[]>(() => {
    if (!analysisResult.value?.skeleton_data) return []
    const placements: ShotPlacement[] = []
    for (const rally of rallies.value) {
      for (const shot of rally.shots) {
        // Use shuttle position if available, otherwise fall back to player position
        const pixelPos = shot.shuttlePosition || shot.playerPosition
        const courtPos = toCourtMeters(pixelPos)
        if (courtPos) {
          placements.push({
            shotType: shot.shotType || 'unknown',
            position: courtPos,
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
    if (placements.length < 3) return []

    // Use fixed court grid: 6 columns (width) x 8 rows (length)
    // Each cell ~1.0m x 1.675m — aligns roughly with court zones
    const GRID_COLS = 6
    const GRID_ROWS = 8
    const cellW = COURT_WIDTH / GRID_COLS
    const cellH = COURT_LENGTH / GRID_ROWS

    const typeMap = new Map<string, { grid: number[][]; total: number }>()
    const allGrid = Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(0) as number[])
    let allTotal = 0

    for (const p of placements) {
      if (!typeMap.has(p.shotType)) {
        typeMap.set(p.shotType, {
          grid: Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(0)),
          total: 0,
        })
      }
      const gx = clamp(Math.floor(p.position.x / cellW), 0, GRID_COLS - 1)
      const gy = clamp(Math.floor(p.position.y / cellH), 0, GRID_ROWS - 1)
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
  // 5. RECOVERY POSITION ANALYSIS (court-aware)
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

    const frameIndexMap = new Map<number, number>()
    for (let i = 0; i < frames.length; i++) frameIndexMap.set(frames[i]!.frame, i)

    // Determine each player's court half center (ideal recovery position)
    // Player closer to top (lower Y in video) → top half center, other → bottom half
    const playerAvgY = new Map<number, { sumY: number; n: number }>()
    for (const f of frames) {
      for (const p of f.players) {
        const c = getCenter(p)
        let acc = playerAvgY.get(p.player_id)
        if (!acc) { acc = { sumY: 0, n: 0 }; playerAvgY.set(p.player_id, acc) }
        acc.sumY += c.y; acc.n++
      }
    }

    // Court center positions in meters for each half
    const courtCenterTop = { x: COURT_WIDTH / 2, y: COURT_LENGTH / 4 }     // 3.05, 3.35
    const courtCenterBottom = { x: COURT_WIDTH / 2, y: 3 * COURT_LENGTH / 4 } // 3.05, 10.05

    // Assign court half to each player based on average video Y position
    // (Lower pixel Y = top of video = one side, higher = other side)
    const playerBaseCourtPos = new Map<number, { x: number; y: number }>()
    const playerIds = Array.from(playerAvgY.keys())
    if (playerIds.length >= 2) {
      const avgYs = playerIds.map(pid => {
        const acc = playerAvgY.get(pid)!
        return { pid, avgY: acc.sumY / acc.n }
      })
      avgYs.sort((a, b) => a.avgY - b.avgY) // lower Y first
      // Use homography to check which court half each player occupies
      const firstCourtPos = toCourtMeters({ x: 0, y: avgYs[0]!.avgY })
      if (firstCourtPos && firstCourtPos.y < COURT_LENGTH / 2) {
        playerBaseCourtPos.set(avgYs[0]!.pid, courtCenterTop)
        playerBaseCourtPos.set(avgYs[1]!.pid, courtCenterBottom)
      } else {
        playerBaseCourtPos.set(avgYs[0]!.pid, courtCenterBottom)
        playerBaseCourtPos.set(avgYs[1]!.pid, courtCenterTop)
      }
    } else if (playerIds.length === 1) {
      playerBaseCourtPos.set(playerIds[0]!, courtCenterTop)
    }

    // Compute noise floor per player
    const playerNoiseFloor = new Map<number, number>()
    for (const pid of playerIds) {
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
        playerNoiseFloor.set(pid, Math.max(2, median * 2))
      } else {
        playerNoiseFloor.set(pid, 8)
      }
    }

    for (const shot of allShots) {
      const pid = shot.playerId
      const basePos = playerBaseCourtPos.get(pid)
      if (!basePos) continue

      const shotFrameIdx = frameIndexMap.get(shot.frame) ?? frames.findIndex(f => f.frame >= shot.frame)
      if (shotFrameIdx < 0) continue

      const RECOVERY_WINDOW = Math.floor(fps * 2)
      let recoveryFrame: SkeletonFrame | null = null
      let recoveryPos: { x: number; y: number } | null = null

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
            if (stationaryCount >= 3) {
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
        if (recoveryTime < 0.1) continue

        // Transform recovery position to court meters for distance calculation
        const recoveryCourtPos = toCourtMeters(recoveryPos)
        const distFromBase = recoveryCourtPos
          ? dist(recoveryCourtPos, basePos) // meters from court center
          : dist(recoveryPos, { x: 0, y: 0 }) // fallback pixel (won't be meaningful)

        // Quality factors in both time AND position (when homography available)
        let quality: RecoveryEvent['quality'] = 'poor'
        if (recoveryCourtPos) {
          // Combined score: time (60%) + position (40%)
          // Time score: <0.8s = 1.0, 0.8-1.2s = 0.75, 1.2-1.8s = 0.5, >1.8s = 0.25
          let timeScore = 0.25
          if (recoveryTime < 0.8) timeScore = 1.0
          else if (recoveryTime < 1.2) timeScore = 0.75
          else if (recoveryTime < 1.8) timeScore = 0.5

          // Position score: <1.0m = 1.0, 1.0-2.0m = 0.75, 2.0-3.0m = 0.5, >3.0m = 0.25
          let posScore = 0.25
          if (distFromBase < 1.0) posScore = 1.0
          else if (distFromBase < 2.0) posScore = 0.75
          else if (distFromBase < 3.0) posScore = 0.5

          const combined = timeScore * 0.6 + posScore * 0.4
          if (combined >= 0.85) quality = 'excellent'
          else if (combined >= 0.65) quality = 'good'
          else if (combined >= 0.45) quality = 'fair'
        } else {
          // No homography — fall back to time-only
          if (recoveryTime < 0.8) quality = 'excellent'
          else if (recoveryTime < 1.2) quality = 'good'
          else if (recoveryTime < 1.8) quality = 'fair'
        }

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
          distanceFromBase: distFromBase,
          quality,
        })
      }
    }

    return events
  })

  const recoveryStats = computed(() => {
    const events = recoveryEvents.value
    if (events.length < 3) return null

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
  // SHARED: Inter-player distance metrics (used by reaction time)
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
  // 6. REACTION TIME (with anticipation filtering)
  // =========================================================================

  const reactionEvents = computed<ReactionEvent[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const events: ReactionEvent[] = []

    const frameIndexMap = new Map<number, number>()
    for (let i = 0; i < frames.length; i++) frameIndexMap.set(frames[i]!.frame, i)

    // Self-calibrating displacement threshold
    const DISPLACEMENT_THRESHOLD = Math.max(5, interPlayerDistMetrics.value.avg * 0.025)
    // Pre-shot movement check window: 5 frames before opponent's shot
    const PRE_SHOT_WINDOW = 5

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

        // Filter anticipatory movement: check if player was already moving
        // significantly before the opponent's shot
        let wasAlreadyMoving = false
        if (startIdx >= PRE_SHOT_WINDOW) {
          const prePos = frames[startIdx - PRE_SHOT_WINDOW]?.players.find(p => p.player_id === respondingPlayer)
          if (prePos) {
            const preDisplacement = dist(getCenter(prePos), basePos)
            // If already displaced more than threshold before the shot, skip
            if (preDisplacement > DISPLACEMENT_THRESHOLD * 1.5) {
              wasAlreadyMoving = true
            }
          }
        }
        if (wasAlreadyMoving) continue

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
    if (events.length < 3) return null

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
  // 7. MOVEMENT EFFICIENCY
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
        if (segDist > 5) {
          directnessSum += displacement / segDist
          segmentCount++
        }
      }

      if (segmentCount < 3) continue

      const avgDirectness = directnessSum / segmentCount

      efficiencies.push({
        playerId: pid,
        totalDistance: 0,
        usefulDistance: 0,
        wastedDistance: 0,
        efficiencyScore: clamp(avgDirectness * 100, 0, 100),
        avgDirectness,
      })
    }

    return efficiencies
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
    reactionEvents,
    reactionStats,
    movementEfficiency,
    isComputing,
  }
}
