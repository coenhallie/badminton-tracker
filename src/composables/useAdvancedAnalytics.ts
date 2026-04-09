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
  RallySpeedStats,
  RallyPlayerSpeed,
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

/** High-confidence hitting poses — only poses that unambiguously indicate a shot */
const HITTING_POSES = new Set([
  'smash', 'overhead', 'serving',
  'serve', 'offense',
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
  courtKeypoints?: Ref<number[][] | null>,
  cameraAngle?: Ref<'overhead' | 'corner'>
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
   * Detect shots from shuttle trajectory direction changes.
   *
   * A shot occurs when the shuttle reverses direction — this is the only
   * physically grounded signal. Falls back to pose detection only when
   * shuttle data is too sparse.
   *
   * Minimum gap between shots: 0.6s (fastest realistic rally exchange).
   */
  function detectAllShots(frames: SkeletonFrame[], fps: number): RallyShot[] {
    const isCorner = cameraAngle?.value === 'corner'
    const MIN_GAP_S = isCorner ? 0.8 : 0.6
    const MIN_GAP_FRAMES = Math.max(3, Math.floor(fps * MIN_GAP_S))

    const frameMap = new Map<number, SkeletonFrame>()
    for (const f of frames) frameMap.set(f.frame, f)

    // --- Primary: shuttle direction changes ---
    const shuttleShots = detectShotsFromShuttle(frames, frameMap, MIN_GAP_FRAMES)

    if (shuttleShots.length >= 4) {
      return shuttleShots
    }

    // --- Fallback: pose detection (only when shuttle data insufficient) ---
    const poseShots = detectShotsFromPose(frames, fps, MIN_GAP_FRAMES)

    // Merge shuttle + pose, preferring shuttle-detected shots
    return mergeShots([...shuttleShots, ...poseShots], MIN_GAP_FRAMES)
  }

  /**
   * Detect shots from shuttle trajectory direction reversals.
   * A negative dot product between consecutive velocity vectors = direction change.
   */
  function detectShotsFromShuttle(
    frames: SkeletonFrame[],
    frameMap: Map<number, SkeletonFrame>,
    minGapFrames: number
  ): RallyShot[] {
    const isCorner = cameraAngle?.value === 'corner'
    const rawPositions: { frame: number; ts: number; x: number; y: number }[] = []
    for (const f of frames) {
      if (f.shuttle_position?.x != null && f.shuttle_position?.y != null) {
        rawPositions.push({ frame: f.frame, ts: f.timestamp, x: f.shuttle_position.x, y: f.shuttle_position.y })
      }
    }

    if (rawPositions.length < 5) return []

    // Filter out single-frame outliers: if a position jumps far from both
    // its predecessor and successor, it's a TrackNet glitch, not the shuttle.
    const OUTLIER_DIST_SQ = 400 * 400
    const positions: typeof rawPositions = [rawPositions[0]!]
    for (let i = 1; i < rawPositions.length - 1; i++) {
      const prev = rawPositions[i - 1]!
      const curr = rawPositions[i]!
      const next = rawPositions[i + 1]!
      const dPrev = (curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2
      const dNext = (curr.x - next.x) ** 2 + (curr.y - next.y) ** 2
      if (dPrev > OUTLIER_DIST_SQ && dNext > OUTLIER_DIST_SQ) continue
      positions.push(curr)
    }
    positions.push(rawPositions[rawPositions.length - 1]!)

    const shots: RallyShot[] = []
    let lastShotFrame = -Infinity

    // Max time gap between consecutive positions for velocity to be meaningful.
    // Positions spanning idle periods (e.g., 3s gap between rallies) produce
    // meaningless velocity vectors that create false direction reversals.
    const MAX_GAP_S = 2.5

    for (let i = 2; i < positions.length; i++) {
      const p0 = positions[i - 2]!
      const p1 = positions[i - 1]!
      const p2 = positions[i]!

      // Skip if positions are too far apart in time — velocity vectors are meaningless
      if ((p1.ts - p0.ts) > MAX_GAP_S || (p2.ts - p1.ts) > MAX_GAP_S) continue

      const vx1 = p1.x - p0.x, vy1 = p1.y - p0.y
      const vx2 = p2.x - p1.x, vy2 = p2.y - p1.y

      // Skip if both velocity vectors are too small — jitter on a
      // nearly-stationary shuttle, not a real shot. Real shots produce
      // at least one large velocity vector (50+ px displacement).
      const speed1sq = vx1 * vx1 + vy1 * vy1
      const speed2sq = vx2 * vx2 + vy2 * vy2
      const MIN_SPEED_SQ = isCorner ? 50 * 50 : 30 * 30
      if (speed1sq < MIN_SPEED_SQ && speed2sq < MIN_SPEED_SQ) continue

      const dot = vx1 * vx2 + vy1 * vy2

      // Corner: require >~105deg reversal to reject jitter wobbles
      const dotThreshold = isCorner
        ? -0.25 * Math.sqrt(speed1sq * speed2sq)
        : 0
      if (dot >= dotThreshold || (p1.frame - lastShotFrame) < minGapFrames) continue

      const skFrame = frameMap.get(p1.frame)
      if (!skFrame || skFrame.players.length === 0) continue

      // Assign to closest player
      let closest = skFrame.players[0]!
      let minD = dist(getCenter(closest), p1)
      for (const pl of skFrame.players) {
        const d = dist(getCenter(pl), p1)
        if (d < minD) { minD = d; closest = pl }
      }

      shots.push({
        frame: p1.frame,
        timestamp: p1.ts,
        playerId: closest.player_id,
        shotType: getPlayerPoseType(closest) || 'unknown',
        shuttlePosition: { x: p1.x, y: p1.y },
        playerPosition: getCenter(closest),
      })
      lastShotFrame = p1.frame
    }

    return shots
  }

  /**
   * Fallback: detect shots from hitting poses.
   * Only uses high-confidence, unambiguous hitting poses (smash, overhead, serving).
   * Requires confidence >= 0.65 and minimum 0.8s gap per player.
   */
  function detectShotsFromPose(
    frames: SkeletonFrame[],
    fps: number,
    minGapFrames: number
  ): RallyShot[] {
    const perPlayerGap = Math.max(minGapFrames, Math.floor(fps * 0.8))
    const lastShotFrame = new Map<number, number>()
    const shots: RallyShot[] = []

    for (const f of frames) {
      for (const player of f.players) {
        const poseType = getPlayerPoseType(player)
        if (!poseType || !HITTING_POSES.has(poseType)) continue

        const pose = (player as FramePlayer & { pose?: { confidence?: number } }).pose
        if ((pose?.confidence ?? 0) < 0.65) continue

        const last = lastShotFrame.get(player.player_id) ?? -Infinity
        if ((f.frame - last) < perPlayerGap) continue

        shots.push({
          frame: f.frame,
          timestamp: f.timestamp,
          playerId: player.player_id,
          shotType: poseType,
          shuttlePosition: f.shuttle_position || null,
          playerPosition: getCenter(player),
        })
        lastShotFrame.set(player.player_id, f.frame)
      }
    }

    return shots
  }

  function mergeShots(shots: RallyShot[], minGap: number): RallyShot[] {
    shots.sort((a, b) => a.frame - b.frame)
    const merged: RallyShot[] = []
    for (const shot of shots) {
      const last = merged[merged.length - 1]
      if (last && (shot.frame - last.frame) < minGap) {
        // Prefer shuttle-detected shots over pose-detected
        if (last.shuttlePosition === null && shot.shuttlePosition !== null) {
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

  /** Detect rallies from shuttle direction changes (shot-gap heuristic). */
  const rallies = computed<Rally[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length < 10) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const shots = detectAllShots(frames, fps)
    const isCorner = cameraAngle?.value === 'corner'
    const MIN_SHOTS = isCorner ? 3 : 2
    if (shots.length < MIN_SHOTS) return []

    const RALLY_GAP_SECONDS = isCorner ? 4.0 : 3.1

    // Check if a rally candidate is real gameplay (not a replay/close-up).
    // Real gameplay from the standard court camera shows 2 players in 50%+
    // of frames. Replays use close-up angles showing 0-1 players.
    function isRealGameplay(startTs: number, endTs: number): boolean {
      let total = 0
      let twoPlayers = 0
      for (const f of frames) {
        if (f.timestamp < startTs || f.timestamp > endTs) continue
        total++
        if (f.players.length >= 2) twoPlayers++
      }
      return total === 0 || (twoPlayers / total) >= 0.50
    }

    const detected: Rally[] = []
    let rallyStart = 0

    for (let i = 1; i < shots.length; i++) {
      const gap = shots[i]!.timestamp - shots[i - 1]!.timestamp
      if (gap > RALLY_GAP_SECONDS || i === shots.length - 1) {
        const rallyShots = shots.slice(rallyStart, i === shots.length - 1 ? i + 1 : i)
        const MIN_RALLY_DURATION_S = isCorner ? 3.0 : 2.0
        if (rallyShots.length >= MIN_SHOTS) {
          const first = rallyShots[0]!
          const last = rallyShots[rallyShots.length - 1]!
          if ((last.timestamp - first.timestamp) >= MIN_RALLY_DURATION_S && isRealGameplay(first.timestamp, last.timestamp)) {
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
        }
        rallyStart = i
      }
    }

    return detected
  })

  const backendRallies = computed(() => {
    const result = analysisResult.value
    if (!result?.rallies || result.rallies.length === 0) return []
    return result.rallies.map(r => ({
      startTimestamp: r.start_timestamp,
      endTimestamp: r.end_timestamp,
      durationSeconds: r.duration_seconds,
    }))
  })

  const rallySource = computed<'client' | 'backend' | 'both' | null>(() => {
    const hasClient = rallies.value.length > 0
    const hasBackend = backendRallies.value.length > 0
    if (hasClient && hasBackend) return 'both'
    if (hasClient) return 'client'
    if (hasBackend) return 'backend'
    return null
  })

  // =========================================================================
  // 2b. PER-RALLY SPEED STATS
  // =========================================================================

  /**
   * Compute per-rally, per-player speed metrics from current_speed data.
   *
   * current_speed is already in km/h, already filtered by the backend's 5-stage
   * pipeline (pixel-jump, distance-jump, hard limit, median filter, tracking
   * validation). We aggregate only non-zero values (zeros = filtered/invalid).
   *
   * Distance is integrated per frame: distance_m += (speed_kmh / 3.6) * dt
   * where dt = 1/fps for consecutive frames.
   *
   * Reliability gate: stats are marked reliable only when >= 30% of the rally's
   * frames have non-zero speed data for a player. Below that, tracking was too
   * sparse to trust the aggregates.
   */
  const rallySpeedStats = computed<RallySpeedStats[]>(() => {
    const result = analysisResult.value
    if (!result?.skeleton_data || result.skeleton_data.length === 0) return []

    const r = rallies.value
    if (r.length === 0) return []

    const frames = result.skeleton_data
    const fps = result.fps || 30
    const dt = 1 / fps

    // Build frame index for O(1) lookup
    const frameMap = new Map<number, SkeletonFrame>()
    for (const f of frames) frameMap.set(f.frame, f)

    const MIN_RELIABLE_RATIO = 0.3

    return r.map(rally => {
      // Collect per-player speed samples within this rally's frame range
      const playerData = new Map<number, { speeds: number[]; totalFrames: number }>()

      for (let frameNum = rally.startFrame; frameNum <= rally.endFrame; frameNum++) {
        const f = frameMap.get(frameNum)
        if (!f) continue

        for (const player of f.players) {
          let data = playerData.get(player.player_id)
          if (!data) {
            data = { speeds: [], totalFrames: 0 }
            playerData.set(player.player_id, data)
          }
          data.totalFrames++
          if (player.current_speed > 0) {
            data.speeds.push(player.current_speed)
          }
        }
      }

      const players: RallyPlayerSpeed[] = []
      let anyReliable = false

      for (const [playerId, data] of playerData) {
        const { speeds, totalFrames } = data
        if (speeds.length === 0 || totalFrames === 0) continue

        const sampleRatio = speeds.length / totalFrames
        const avgSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length
        const maxSpeed = Math.max(...speeds)
        // Integrate distance: each non-zero speed sample covers dt seconds
        const distanceCovered = speeds.reduce((acc, s) => acc + (s / 3.6) * dt, 0)

        if (sampleRatio >= MIN_RELIABLE_RATIO) {
          anyReliable = true
        }

        players.push({
          playerId,
          avgSpeed,
          maxSpeed,
          distanceCovered,
          sampleCount: speeds.length,
          totalFrames,
        })
      }

      // Sort by player ID for consistent ordering
      players.sort((a, b) => a.playerId - b.playerId)

      return {
        rallyId: rally.id,
        players,
        reliable: anyReliable,
      }
    })
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
    backendRallies,
    rallySource,
    rallySpeedStats,
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
