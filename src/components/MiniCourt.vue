<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import type { FramePlayer } from '@/types/analysis'
import { PLAYER_COLORS, COURT_DIMENSIONS } from '@/types/analysis'

// Debug mode - set to true to see coordinate transformation logs
const DEBUG_MODE = import.meta.env.DEV && false

// =============================================================================
// MINI COURT COMPONENT
// =============================================================================
// Displays a top-down view of a badminton court with player positions
// transformed from video coordinates using the detected court homography.
//
// The component receives court corners (from auto-detection or manual input)
// and player positions, then uses a perspective transform to map players
// onto the standardized court representation.
//
// PERFORMANCE OPTIMIZATIONS:
// - Cached canvas 2D context to avoid repeated getContext() calls
// - Throttled render function to prevent excessive redraws
// - Skip render when player positions haven't changed
// - ShallowRef for homography matrix to avoid deep reactivity
// =============================================================================

interface CourtCorners {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
}

interface ShuttlePosition {
  x: number
  y: number
}

// Hit marker representing a detected shuttlecock strike location
interface HitMarker {
  playerId: number
  hitNumber: number  // Sequential per player, starting from 1
  courtX: number     // Court coordinates in meters
  courtY: number     // Court coordinates in meters
  frame: number      // Frame number when hit was detected
}

// Per-player wrist tracking state for hit detection
interface PlayerWristState {
  prevLeft: { x: number; y: number; frame: number } | null
  prevRight: { x: number; y: number; frame: number } | null
  speeds: number[]   // Recent max wrist speeds (for peak detection)
  lastHitFrame: number
}

// Import SkeletonFrame type for trail computation
import type { SkeletonFrame } from '@/types/analysis'

const props = withDefaults(defineProps<{
  // Court corners in video pixel coordinates [x, y]
  // Order: top-left, top-right, bottom-right, bottom-left
  courtCorners?: number[][] | null
  // Alternative: manual court keypoints object
  manualKeypoints?: CourtCorners | null
  // Current frame's player data
  players?: FramePlayer[]
  // Shuttle position in video coordinates
  shuttlePosition?: ShuttlePosition | null
  // Canvas dimensions
  width?: number
  height?: number
  // Display options
  showGrid?: boolean
  showLabels?: boolean
  showShuttle?: boolean
  showTrails?: boolean
  showHitMarkers?: boolean
  // Keypoint selection mode
  isKeypointSelectionMode?: boolean
  keypointSelectionCount?: number
  // Full skeleton data for computing trails dynamically
  // When provided, trails are computed from frame 0 to currentFrame
  skeletonData?: SkeletonFrame[]
  // Current frame number (for dynamic trail computation)
  currentFrame?: number
  // Maximum trail length (number of points to show)
  maxTrailLength?: number
}>(), {
  width: 240,
  height: 440,
  showGrid: true,
  showLabels: true,
  showShuttle: true,
  showTrails: true, // Enable trails by default
  showHitMarkers: true, // Enable hit markers by default
  isKeypointSelectionMode: false,
  keypointSelectionCount: 0,
  maxTrailLength: 100 // Show last 100 positions (~3.3 seconds at 30fps)
})

const canvasRef = ref<HTMLCanvasElement | null>(null)

// PERFORMANCE OPTIMIZATION: Cache the canvas 2D context
let cachedCtx: CanvasRenderingContext2D | null = null

// =============================================================================
// SMOOTH ANIMATION: Interpolated player/shuttle positions for fluid rendering
// =============================================================================
// The video overlay renders interpolated skeleton data at 60fps via rAF, but
// MiniCourt receives discrete position updates at ~20-30fps from the parent.
// To match the video's fluidity, we use exponential smoothing (EMA) to
// interpolate between position updates, producing buttery-smooth court motion.
// =============================================================================

interface SmoothedPlayerPos {
  courtX: number
  courtY: number
  targetCourtX: number
  targetCourtY: number
  speed: number
  targetSpeed: number
  /** Whether this player is currently recognized (skeleton/bbox detected) */
  isRecognized: boolean
  /** Number of consecutive frames the player has NOT been recognized */
  unrecognizedFrames: number
}

interface SmoothedShuttlePos {
  courtX: number
  courtY: number
  targetCourtX: number
  targetCourtY: number
}

const smoothedPlayerPositions = new Map<number, SmoothedPlayerPos>()
let smoothedShuttle: SmoothedShuttlePos | null = null
let animFrameId: number | null = null
let lastAnimTime = 0
// Exponential smoothing speed (higher = faster convergence to target)
// At 60fps with speed=18, positions converge ~95% within ~165ms
const SMOOTHING_SPEED = 18
let isAnimating = false

// =============================================================================
// PLAYER TRAIL ACCUMULATION from smoothed positions
// =============================================================================
// Trails record the actual smoothed court positions used by the player circles,
// following YOLO26's track_history pattern: append the tracked position each frame.
// This ensures trails exactly follow the displayed player dots.
// =============================================================================
const playerTrails = ref<Map<number, { x: number; y: number }[]>>(new Map())
let lastTrailFrame = -1

// =============================================================================
// HIT DETECTION: Detect shuttlecock strikes from wrist acceleration patterns
// =============================================================================
// Analyzes wrist keypoint velocity across frames to identify the precise moment
// each player strikes the shuttlecock. A hit is detected when wrist speed reaches
// a local maximum (peak) that exceeds a minimum threshold, with a cooldown to
// prevent double-counting. At each detected contact point, the player's court
// position is recorded for cumulative marker overlay.
//
// COCO keypoint indices: 9 = left_wrist, 10 = right_wrist
// =============================================================================

const LEFT_WRIST_IDX = 9
const RIGHT_WRIST_IDX = 10

// Hit detection tuning constants — calibrated to reduce false positives.
// A genuine badminton stroke produces a wrist speed of 30-80+ px/frame at 30fps/1080p,
// while normal ready-position arm sway is typically 3-12 px/frame.
const HIT_DETECTION_CONFIG = {
  MIN_WRIST_SPEED: 30,       // Minimum pixels/frame for wrist speed peak (raised from 15)
  PEAK_PROMINENCE: 2.0,      // Peak must be ≥ 2× the baseline (median of window) to qualify
  DECEL_RATIO: 0.6,          // Speed after peak must drop to ≤ 60% of peak (i.e., ≥40% deceleration)
  COOLDOWN_FRAMES: 25,       // Min frames between consecutive hits (~0.8s at 30fps)
  CONFIDENCE_THRESHOLD: 0.3, // Min keypoint confidence to use wrist position
  MAX_SPEED_HISTORY: 10,     // Number of recent speeds to keep for baseline & peak detection
} as const

// Pose types that indicate an active stroke (used for cross-referencing)
const OFFENSIVE_POSE_TYPES = new Set([
  'smash', 'overhead', 'forehand', 'backhand', 'serving', // skeleton-based
  'offense', 'smash', 'serve', 'lift', 'backhand-general'  // trained model
])

// Hit marker colors - vivid, distinct per player (slightly different from trail colors)
const HIT_MARKER_COLORS = [
  '#FF2952', // Player 1 - Bright magenta-red
  '#00E5CC', // Player 2 - Bright cyan-green
  '#2196F3', // Player 3 - Bright blue
  '#FFD700', // Player 4 - Gold
]

// Cumulative hit markers per player
const hitMarkers = ref<Map<number, HitMarker[]>>(new Map())
// Per-player sequential hit counter
const playerHitCounters = new Map<number, number>()
// Per-player wrist tracking state
const playerWristStates = new Map<number, PlayerWristState>()
// Last skeleton frame processed for hit detection
let lastHitDetectionFrame = -1

/**
 * Record current smoothed player positions into trails.
 * Called from updateTargetPositions() so trails accumulate at the prop-update rate
 * (typically ~30fps), matching the pace of the player circle movement.
 * Like YOLO26's track_history[track_id].append((x, y)), this appends the
 * actual displayed position for each recognized player.
 */
function recordTrailPositions() {
  if (!props.showTrails) return
  
  // Only record once per frame to avoid duplicates from re-renders
  const frame = props.currentFrame ?? 0
  if (frame === lastTrailFrame) return
  lastTrailFrame = frame
  
  const maxLength = props.maxTrailLength ?? 100
  
  for (const [playerId, smoothed] of smoothedPlayerPositions) {
    // Only record positions for recognized (actively tracked) players
    if (!smoothed.isRecognized) continue
    
    let trail = playerTrails.value.get(playerId)
    if (!trail) {
      trail = []
      playerTrails.value.set(playerId, trail)
    }
    
    // Append current smoothed court position (same as what drawPlayers renders)
    trail.push({ x: smoothed.courtX, y: smoothed.courtY })
    
    // Keep trail length bounded (like YOLO26's `if len(track) > 30: track.pop(0)`)
    if (trail.length > maxLength) {
      trail.splice(0, trail.length - maxLength)
    }
  }
}

/**
 * Get wrist positions from a player's keypoints.
 * Returns both left and right wrist positions if available and confident.
 */
function getWristPositions(player: import('@/types/analysis').FramePlayer): {
  left: { x: number; y: number } | null
  right: { x: number; y: number } | null
} {
  const keypoints = player.keypoints
  if (!keypoints || keypoints.length < 11) return { left: null, right: null }
  
  const threshold = HIT_DETECTION_CONFIG.CONFIDENCE_THRESHOLD
  const leftWrist = keypoints[LEFT_WRIST_IDX]
  const rightWrist = keypoints[RIGHT_WRIST_IDX]
  
  return {
    left: (leftWrist && leftWrist.x !== null && leftWrist.y !== null && leftWrist.confidence > threshold)
      ? { x: leftWrist.x, y: leftWrist.y } : null,
    right: (rightWrist && rightWrist.x !== null && rightWrist.y !== null && rightWrist.confidence > threshold)
      ? { x: rightWrist.x, y: rightWrist.y } : null
  }
}

/**
 * Compute distance between two 2D points, normalized by frame gap.
 * Returns speed in pixels per frame.
 */
function computeWristSpeed(
  prev: { x: number; y: number; frame: number },
  curr: { x: number; y: number; frame: number }
): number {
  const dx = curr.x - prev.x
  const dy = curr.y - prev.y
  const dist = Math.sqrt(dx * dx + dy * dy)
  const frameGap = Math.max(1, curr.frame - prev.frame)
  return dist / frameGap
}

/**
 * Compute the median of a numeric array (for robust baseline estimation).
 */
function median(arr: number[]): number {
  if (arr.length === 0) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

/**
 * Check if the player's current pose classification suggests an active stroke.
 * Uses both skeleton-based pose and trained model classifications.
 */
function isOffensivePose(player: import('@/types/analysis').FramePlayer, skFrame: import('@/types/analysis').SkeletonFrame): boolean {
  // Check skeleton-based pose
  if (player.pose?.pose_type && OFFENSIVE_POSE_TYPES.has(player.pose.pose_type)) {
    return true
  }
  // Check trained model classifications (matched by position proximity)
  if (skFrame.pose_classifications) {
    for (const pc of skFrame.pose_classifications) {
      if (OFFENSIVE_POSE_TYPES.has(pc.class_name) && pc.confidence > 0.4) {
        return true
      }
    }
  }
  return false
}

/**
 * Process skeleton frames to detect shuttlecock strikes up to the current frame.
 *
 * Multi-layered detection to minimize false positives:
 * 1. Velocity peak: wrist speed local max (speed[n-1] > both neighbors)
 * 2. Absolute threshold: peak speed > MIN_WRIST_SPEED (30 px/frame)
 * 3. Prominence: peak > PEAK_PROMINENCE × baseline median (filters out oscillations)
 * 4. Deceleration: speed must drop to ≤ DECEL_RATIO of peak (confirms stroke follow-through)
 * 5. Cooldown: minimum frames between hits per player
 * 6. Pose cross-reference: offensive pose lowers threshold by 20% (optional signal boost)
 *
 * Processes incrementally — only new frames since last call.
 * Returns true if new hits were detected.
 */
function detectHits(): boolean {
  if (!props.skeletonData || props.skeletonData.length === 0) return false
  if (props.currentFrame === undefined) return false
  
  const H = homographyMatrix.value
  if (!H) return false
  
  const targetFrame = props.currentFrame
  let newHitsDetected = false
  
  for (const skFrame of props.skeletonData) {
    if (skFrame.frame <= lastHitDetectionFrame) continue
    if (skFrame.frame > targetFrame) break
    
    for (const player of skFrame.players) {
      const wrists = getWristPositions(player)
      if (!wrists.left && !wrists.right) continue
      
      // Get or create wrist state for this player
      let state = playerWristStates.get(player.player_id)
      if (!state) {
        state = {
          prevLeft: null,
          prevRight: null,
          speeds: [],
          lastHitFrame: -100
        }
        playerWristStates.set(player.player_id, state)
      }
      
      // Compute speed of each wrist independently, take max (racket hand)
      let leftSpeed = 0
      let rightSpeed = 0
      
      if (wrists.left && state.prevLeft) {
        leftSpeed = computeWristSpeed(state.prevLeft, { ...wrists.left, frame: skFrame.frame })
      }
      if (wrists.right && state.prevRight) {
        rightSpeed = computeWristSpeed(state.prevRight, { ...wrists.right, frame: skFrame.frame })
      }
      
      const maxSpeed = Math.max(leftSpeed, rightSpeed)
      
      // Update previous wrist positions
      if (wrists.left) state.prevLeft = { ...wrists.left, frame: skFrame.frame }
      if (wrists.right) state.prevRight = { ...wrists.right, frame: skFrame.frame }
      
      // Record speed for peak detection
      state.speeds.push(maxSpeed)
      if (state.speeds.length > HIT_DETECTION_CONFIG.MAX_SPEED_HISTORY) {
        state.speeds.shift()
      }
      
      // Need at least 4 speed values for robust peak detection (previous, peak, current, + baseline)
      if (state.speeds.length < 4) continue
      
      const n = state.speeds.length
      const currSpeed = state.speeds[n - 1]!    // Current frame speed (after suspected peak)
      const peakSpeed = state.speeds[n - 2]!    // Suspected peak
      const prePeakSpeed = state.speeds[n - 3]! // Speed before the peak
      
      // --- Layer 1: Velocity peak (local maximum) ---
      const isPeak = peakSpeed > currSpeed && peakSpeed > prePeakSpeed
      if (!isPeak) continue
      
      // --- Layer 2: Absolute speed threshold ---
      // If the player is in an offensive pose, lower threshold by 20%
      const hasOffensivePose = isOffensivePose(player, skFrame)
      const effectiveThreshold = hasOffensivePose
        ? HIT_DETECTION_CONFIG.MIN_WRIST_SPEED * 0.8
        : HIT_DETECTION_CONFIG.MIN_WRIST_SPEED
      
      if (peakSpeed < effectiveThreshold) continue
      
      // --- Layer 3: Peak prominence above baseline ---
      // Compute baseline as the median of all speeds in the window (excluding the peak itself)
      const baselineSpeeds = [...state.speeds.slice(0, n - 2), ...state.speeds.slice(n - 1)]
      const baselineMedian = median(baselineSpeeds)
      const prominenceOk = baselineMedian < 1 || (peakSpeed >= baselineMedian * HIT_DETECTION_CONFIG.PEAK_PROMINENCE)
      if (!prominenceOk) continue
      
      // --- Layer 4: Significant deceleration after peak ---
      // The current speed (post-peak) must have dropped substantially
      const decelOk = currSpeed <= peakSpeed * HIT_DETECTION_CONFIG.DECEL_RATIO
      if (!decelOk) continue
      
      // --- Layer 5: Cooldown between hits ---
      const cooldownPassed = (skFrame.frame - state.lastHitFrame) >= HIT_DETECTION_CONFIG.COOLDOWN_FRAMES
      if (!cooldownPassed) continue
      
      // All checks passed — register hit
      const feetPos = getFeetPosition(player)
      if (!feetPos) continue
      
      const courtPos = applyHomography(H, feetPos.x, feetPos.y)
      if (!courtPos) continue
      
      // Bounds check (allow small margin for near-boundary hits)
      if (courtPos.x < -1 || courtPos.x > COURT_WIDTH + 1 ||
          courtPos.y < -1 || courtPos.y > COURT_LENGTH + 1) continue
      
      // Increment sequential hit counter for this player
      let hitCount = playerHitCounters.get(player.player_id) ?? 0
      hitCount++
      playerHitCounters.set(player.player_id, hitCount)
      
      // Create and store hit marker
      const marker: HitMarker = {
        playerId: player.player_id,
        hitNumber: hitCount,
        courtX: courtPos.x,
        courtY: courtPos.y,
        frame: skFrame.frame
      }
      
      let playerMarkers = hitMarkers.value.get(player.player_id)
      if (!playerMarkers) {
        playerMarkers = []
        hitMarkers.value.set(player.player_id, playerMarkers)
      }
      playerMarkers.push(marker)
      
      state.lastHitFrame = skFrame.frame
      newHitsDetected = true
    }
    
    lastHitDetectionFrame = skFrame.frame
  }
  
  return newHitsDetected
}

/**
 * Reset all hit detection state (e.g., on seek backward or new analysis).
 */
function resetHitDetection() {
  hitMarkers.value.clear()
  playerHitCounters.clear()
  playerWristStates.clear()
  lastHitDetectionFrame = -1
}

// Court dimensions in meters (standard badminton court)
const COURT_LENGTH = COURT_DIMENSIONS.length // 13.4m
const COURT_WIDTH = COURT_DIMENSIONS.width_doubles // 6.1m
const SINGLES_WIDTH = COURT_DIMENSIONS.width_singles // 5.18m
const SERVICE_LINE = COURT_DIMENSIONS.service_line // 1.98m from net
const BACK_SERVICE_LINE = COURT_DIMENSIONS.back_boundary_service // 0.76m from back

// Check if we have valid court data (either 4 or 12 points)
const hasValidCourtData = computed((): boolean => {
  if (props.courtCorners && (props.courtCorners.length === 4 || props.courtCorners.length === 12)) {
    return true
  }
  if (props.manualKeypoints) {
    return true
  }
  return false
})

// Computed court corners from props (either from array or manual keypoints)
// Now supports both 4-point and 12-point arrays
const effectiveCourtCorners = computed((): number[][] | null => {
  // Support 4-point or 12-point arrays
  if (props.courtCorners && (props.courtCorners.length === 4 || props.courtCorners.length === 12)) {
    return props.courtCorners
  }
  if (props.manualKeypoints) {
    return [
      props.manualKeypoints.top_left,
      props.manualKeypoints.top_right,
      props.manualKeypoints.bottom_right,
      props.manualKeypoints.bottom_left
    ]
  }
  return null
})

// Number of keypoints available (4 or 12)
const keypointCount = computed((): number => {
  return effectiveCourtCorners.value?.length ?? 0
})

// Calculate canvas height (container height minus header and legend)
// Header is ~32px, legend is ~40px, padding is 24px (12px top + 12px bottom)
const HEADER_HEIGHT = 32
const LEGEND_HEIGHT = 40
const CONTAINER_PADDING = 24

const canvasHeight = computed(() => {
  // Subtract header, legend, and padding from total height
  const availableHeight = props.height - HEADER_HEIGHT - LEGEND_HEIGHT - CONTAINER_PADDING
  return Math.max(200, availableHeight) // Minimum 200px
})

// Calculate drawing scale and padding
const drawConfig = computed(() => {
  const padding = 20
  const availableWidth = props.width - padding * 2
  const availableHeight = canvasHeight.value - padding * 2
  
  // Scale to fit, maintaining aspect ratio
  const scaleX = availableWidth / COURT_WIDTH
  const scaleY = availableHeight / COURT_LENGTH
  const scale = Math.min(scaleX, scaleY)
  
  // Calculate actual court size in pixels
  const courtWidthPx = COURT_WIDTH * scale
  const courtLengthPx = COURT_LENGTH * scale
  
  // Center the court
  const offsetX = (props.width - courtWidthPx) / 2
  const offsetY = (canvasHeight.value - courtLengthPx) / 2
  
  return {
    scale,
    courtWidthPx,
    courtLengthPx,
    offsetX,
    offsetY,
    padding
  }
})

/**
 * Convert court coordinates (meters) to canvas coordinates (pixels)
 */
function courtToCanvas(courtX: number, courtY: number): { x: number; y: number } {
  const { scale, offsetX, offsetY } = drawConfig.value
  return {
    x: offsetX + courtX * scale,
    y: offsetY + courtY * scale
  }
}

/**
 * Normalize a set of 2D points for numerical stability in homography estimation
 * Returns the normalized points and a 3x3 denormalization matrix
 */
function normalizePoints(points: number[][]): { normalized: number[][]; T: number[][] } {
  // Compute centroid
  let sumX = 0, sumY = 0, count = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      sumX += p[0] ?? 0
      sumY += p[1] ?? 0
      count++
    }
  }
  if (count === 0) return { normalized: points, T: [[1,0,0],[0,1,0],[0,0,1]] }
  
  const meanX = sumX / count
  const meanY = sumY / count
  
  // Compute average distance from centroid
  let sumDist = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      const dx = (p[0] ?? 0) - meanX
      const dy = (p[1] ?? 0) - meanY
      sumDist += Math.sqrt(dx * dx + dy * dy)
    }
  }
  const avgDist = sumDist / count
  const scale = avgDist > 0 ? Math.sqrt(2) / avgDist : 1
  
  // Normalize points: translate to origin, scale to sqrt(2) average distance
  const normalized: number[][] = []
  for (const p of points) {
    if (p && p.length >= 2) {
      normalized.push([
        ((p[0] ?? 0) - meanX) * scale,
        ((p[1] ?? 0) - meanY) * scale
      ])
    } else {
      normalized.push([0, 0])
    }
  }
  
  // Normalization transform matrix T: p_norm = T * p
  // T = [[scale, 0, -scale*meanX], [0, scale, -scale*meanY], [0, 0, 1]]
  const T: number[][] = [
    [scale, 0, -scale * meanX],
    [0, scale, -scale * meanY],
    [0, 0, 1]
  ]
  
  return { normalized, T }
}

/**
 * Invert a 3x3 matrix
 */
function invertMatrix3x3(M: number[][]): number[][] | null {
  const a = M[0]![0]!, b = M[0]![1]!, c = M[0]![2]!
  const d = M[1]![0]!, e = M[1]![1]!, f = M[1]![2]!
  const g = M[2]![0]!, h = M[2]![1]!, i = M[2]![2]!
  
  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
  if (Math.abs(det) < 1e-12) return null
  
  const invDet = 1.0 / det
  return [
    [(e*i - f*h) * invDet, (c*h - b*i) * invDet, (b*f - c*e) * invDet],
    [(f*g - d*i) * invDet, (a*i - c*g) * invDet, (c*d - a*f) * invDet],
    [(d*h - e*g) * invDet, (b*g - a*h) * invDet, (a*e - b*d) * invDet]
  ]
}

/**
 * Multiply two 3x3 matrices
 */
function multiplyMatrix3x3(A: number[][], B: number[][]): number[][] {
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0
      for (let k = 0; k < 3; k++) {
        sum += (A[i]![k] ?? 0) * (B[k]![j] ?? 0)
      }
      result[i]![j] = sum
    }
  }
  return result
}

/**
 * Calculate homography matrix for perspective transformation
 * Converts video pixel coordinates to court coordinates (meters)
 * Supports both 4-point exact solution and overdetermined least-squares for 12+ points
 *
 * Uses coordinate normalization for numerical stability (Hartley normalization)
 */
function calculateHomography(srcPoints: number[][], dstPoints: number[][]): number[][] | null {
  const n = Math.min(srcPoints.length, dstPoints.length)
  if (n < 4) return null
  
  // Filter valid points
  const validSrc: number[][] = []
  const validDst: number[][] = []
  for (let i = 0; i < n; i++) {
    const srcPoint = srcPoints[i]
    const dstPoint = dstPoints[i]
    if (srcPoint && srcPoint.length >= 2 && dstPoint && dstPoint.length >= 2) {
      validSrc.push(srcPoint)
      validDst.push(dstPoint)
    }
  }
  
  if (validSrc.length < 4) return null
  
  // Normalize coordinates for numerical stability
  const { normalized: normSrc, T: T_src } = normalizePoints(validSrc)
  const { normalized: normDst, T: T_dst } = normalizePoints(validDst)
  
  // Build the matrix equation using DLT algorithm with normalized coordinates
  const A: number[][] = []
  const b: number[] = []
  
  for (let i = 0; i < normSrc.length; i++) {
    const sx = normSrc[i]![0] ?? 0
    const sy = normSrc[i]![1] ?? 0
    const dx = normDst[i]![0] ?? 0
    const dy = normDst[i]![1] ?? 0
    
    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    b.push(dx)
    b.push(dy)
  }
  
  if (A.length < 8) return null
  
  let h: number[] | null
  
  if (A.length === 8) {
    // Exactly determined: use direct solution
    h = solveLinearSystem(A, b)
  } else {
    // Overdetermined: use least-squares (normal equations: A^T * A * x = A^T * b)
    h = solveLeastSquares(A, b)
  }
  
  if (!h) return null
  
  // Construct normalized homography matrix H_norm
  const H_norm: number[][] = [
    [h[0]!, h[1]!, h[2]!],
    [h[3]!, h[4]!, h[5]!],
    [h[6]!, h[7]!, 1]
  ]
  
  // Denormalize: H = T_dst^(-1) * H_norm * T_src
  const T_dst_inv = invertMatrix3x3(T_dst)
  if (!T_dst_inv) return null
  
  const temp = multiplyMatrix3x3(H_norm, T_src)
  const H = multiplyMatrix3x3(T_dst_inv, temp)
  
  return H
}

/**
 * Solve overdetermined linear system using least-squares (normal equations)
 * Computes: x = (A^T * A)^-1 * A^T * b
 */
function solveLeastSquares(A: number[][], b: number[]): number[] | null {
  const m = A.length // rows
  const n = A[0]?.length ?? 0 // cols
  
  if (n === 0) return null
  
  // Compute A^T * A (n x n matrix)
  const AtA: number[][] = Array(n).fill(null).map(() => Array(n).fill(0)) as number[][]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0
      for (let k = 0; k < m; k++) {
        sum += (A[k]![i] ?? 0) * (A[k]![j] ?? 0)
      }
      AtA[i]![j] = sum
    }
  }
  
  // Compute A^T * b (n x 1 vector)
  const Atb: number[] = Array(n).fill(0) as number[]
  for (let i = 0; i < n; i++) {
    let sum = 0
    for (let k = 0; k < m; k++) {
      sum += (A[k]![i] ?? 0) * (b[k] ?? 0)
    }
    Atb[i] = sum
  }
  
  // Solve AtA * x = Atb using Gaussian elimination
  return solveLinearSystem(AtA, Atb)
}

/**
 * Solve linear system Ax = b using Gaussian elimination with partial pivoting
 */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length
  const augmented: number[][] = A.map((row, i) => [...row, b[i]!])
  
  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row]![col]!) > Math.abs(augmented[maxRow]![col]!)) {
        maxRow = row
      }
    }
    
    // Swap rows
    [augmented[col], augmented[maxRow]] = [augmented[maxRow]!, augmented[col]!]
    
    // Check for singular matrix
    if (Math.abs(augmented[col]![col]!) < 1e-10) return null
    
    // Eliminate column
    for (let row = col + 1; row < n; row++) {
      const factor = augmented[row]![col]! / augmented[col]![col]!
      for (let j = col; j <= n; j++) {
        augmented[row]![j]! -= factor * augmented[col]![j]!
      }
    }
  }
  
  // Back substitution
  const x: number[] = new Array(n)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i]![n]!
    for (let j = i + 1; j < n; j++) {
      x[i]! -= augmented[i]![j]! * x[j]!
    }
    x[i]! /= augmented[i]![i]!
  }
  
  return x
}

/**
 * Apply homography transformation to a point
 */
function applyHomography(H: number[][], x: number, y: number): { x: number; y: number } | null {
  const w = H[2]![0]! * x + H[2]![1]! * y + H[2]![2]!
  if (Math.abs(w) < 1e-10) return null
  
  const outX = (H[0]![0]! * x + H[0]![1]! * y + H[0]![2]!) / w
  const outY = (H[1]![0]! * x + H[1]![1]! * y + H[1]![2]!) / w
  
  return { x: outX, y: outY }
}

// Standard court positions in meters for each keypoint
// Order matches the 12-point system from VideoPlayer
const COURT_KEYPOINT_POSITIONS: number[][] = [
  // 4 outer corners
  [0, 0],                              // 0: TL - Top-left corner
  [COURT_WIDTH, 0],                    // 1: TR - Top-right corner
  [COURT_WIDTH, COURT_LENGTH],         // 2: BR - Bottom-right corner
  [0, COURT_LENGTH],                   // 3: BL - Bottom-left corner
  // Net intersections (at center length)
  [0, COURT_LENGTH / 2],               // 4: NL - Net left
  [COURT_WIDTH, COURT_LENGTH / 2],     // 5: NR - Net right
  // Service line near (top half) - 1.98m from net towards top
  [0, COURT_LENGTH / 2 - SERVICE_LINE],                  // 6: SNL - Service near left
  [COURT_WIDTH, COURT_LENGTH / 2 - SERVICE_LINE],        // 7: SNR - Service near right
  // Service line far (bottom half) - 1.98m from net towards bottom
  [0, COURT_LENGTH / 2 + SERVICE_LINE],                  // 8: SFL - Service far left
  [COURT_WIDTH, COURT_LENGTH / 2 + SERVICE_LINE],        // 9: SFR - Service far right
  // Center line endpoints at service lines
  [COURT_WIDTH / 2, COURT_LENGTH / 2 - SERVICE_LINE],    // 10: CTN - Center near (at service line)
  [COURT_WIDTH / 2, COURT_LENGTH / 2 + SERVICE_LINE]     // 11: CTF - Center far (at service line)
]

// Cached homography matrix
const homographyMatrix = computed((): number[][] | null => {
  const corners = effectiveCourtCorners.value
  if (!corners) return null
  
  // Source: video pixel keypoints (4 or 12 points)
  const srcPoints = corners
  const n = srcPoints.length
  
  // Destination: court coordinates in meters
  // Use matching number of destination points
  let dstPoints: number[][]
  
  if (n === 4) {
    // 4-point mode: just the corners
    dstPoints = [
      COURT_KEYPOINT_POSITIONS[0]!,
      COURT_KEYPOINT_POSITIONS[1]!,
      COURT_KEYPOINT_POSITIONS[2]!,
      COURT_KEYPOINT_POSITIONS[3]!
    ]
  } else if (n === 12) {
    // 12-point mode: all reference points for better accuracy
    dstPoints = COURT_KEYPOINT_POSITIONS.slice(0, 12)
  } else {
    // Fallback: use available points up to the number we have
    dstPoints = COURT_KEYPOINT_POSITIONS.slice(0, Math.min(n, 12))
  }
  
  const H = calculateHomography(srcPoints, dstPoints)
  
  if (DEBUG_MODE && H) {
    console.log('[MiniCourt] Computed homography matrix with', n, 'points')
    console.log('[MiniCourt] Source points (video px):', srcPoints.map(p => `(${p[0]?.toFixed(0)}, ${p[1]?.toFixed(0)})`).join(', '))
    console.log('[MiniCourt] Dest points (court m):', dstPoints.map(p => `(${p[0]?.toFixed(2)}, ${p[1]?.toFixed(2)})`).join(', '))
  }
  
  return H
})

/**
 * Transform player position from video pixels to court meters
 */
function transformPlayerPosition(pixelX: number, pixelY: number): { x: number; y: number } | null {
  const H = homographyMatrix.value
  if (!H) return null
  
  return applyHomography(H, pixelX, pixelY)
}

/**
 * Draw the badminton court lines
 */
function drawCourt(ctx: CanvasRenderingContext2D) {
  const { courtWidthPx, courtLengthPx, offsetX, offsetY, scale } = drawConfig.value
  
  // Background
  ctx.fillStyle = '#1a472a' // Dark green court
  ctx.fillRect(offsetX, offsetY, courtWidthPx, courtLengthPx)
  
  // Court line style
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = 2
  
  // Outer boundary (doubles court)
  ctx.strokeRect(offsetX, offsetY, courtWidthPx, courtLengthPx)
  
  // Singles sidelines
  const singlesOffset = (COURT_WIDTH - SINGLES_WIDTH) / 2 * scale
  ctx.beginPath()
  ctx.moveTo(offsetX + singlesOffset, offsetY)
  ctx.lineTo(offsetX + singlesOffset, offsetY + courtLengthPx)
  ctx.moveTo(offsetX + courtWidthPx - singlesOffset, offsetY)
  ctx.lineTo(offsetX + courtWidthPx - singlesOffset, offsetY + courtLengthPx)
  ctx.stroke()
  
  // Net (center line)
  const netY = offsetY + courtLengthPx / 2
  ctx.strokeStyle = '#FF4444'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(offsetX, netY)
  ctx.lineTo(offsetX + courtWidthPx, netY)
  ctx.stroke()
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = 2
  
  // Short service lines (1.98m from net on each side)
  const serviceLineDist = SERVICE_LINE * scale
  
  // Top half service line
  ctx.beginPath()
  ctx.moveTo(offsetX, netY - serviceLineDist)
  ctx.lineTo(offsetX + courtWidthPx, netY - serviceLineDist)
  ctx.stroke()
  
  // Bottom half service line
  ctx.beginPath()
  ctx.moveTo(offsetX, netY + serviceLineDist)
  ctx.lineTo(offsetX + courtWidthPx, netY + serviceLineDist)
  ctx.stroke()
  
  // Long service lines for doubles (0.76m from back)
  const longServiceDist = BACK_SERVICE_LINE * scale
  ctx.setLineDash([5, 5])
  
  // Top long service line
  ctx.beginPath()
  ctx.moveTo(offsetX, offsetY + longServiceDist)
  ctx.lineTo(offsetX + courtWidthPx, offsetY + longServiceDist)
  ctx.stroke()
  
  // Bottom long service line
  ctx.beginPath()
  ctx.moveTo(offsetX, offsetY + courtLengthPx - longServiceDist)
  ctx.lineTo(offsetX + courtWidthPx, offsetY + courtLengthPx - longServiceDist)
  ctx.stroke()
  
  ctx.setLineDash([])
  
  // Center lines (in service boxes)
  const centerX = offsetX + courtWidthPx / 2
  
  // Top service box center line
  ctx.beginPath()
  ctx.moveTo(centerX, offsetY)
  ctx.lineTo(centerX, netY - serviceLineDist)
  ctx.stroke()
  
  // Bottom service box center line
  ctx.beginPath()
  ctx.moveTo(centerX, netY + serviceLineDist)
  ctx.lineTo(centerX, offsetY + courtLengthPx)
  ctx.stroke()
  
  // Draw labels if enabled
  if (props.showLabels) {
    ctx.font = '10px Inter, sans-serif'
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
    ctx.textAlign = 'center'
    
    // Net label
    ctx.fillText('NET', centerX, netY - 5)
    
    // Zone labels
    ctx.font = '9px Inter, sans-serif'
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'
    
    const topY = offsetY + courtLengthPx * 0.25
    const bottomY = offsetY + courtLengthPx * 0.75
    
    ctx.fillText('REAR COURT', centerX, topY - 20)
    ctx.fillText('FRONT COURT', centerX, topY + 30)
    ctx.fillText('FRONT COURT', centerX, bottomY - 30)
    ctx.fillText('REAR COURT', centerX, bottomY + 20)
  }
  
  // Draw grid if enabled
  if (props.showGrid) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    
    // Vertical grid (every 1m)
    for (let x = 1; x < COURT_WIDTH; x++) {
      const gridX = offsetX + x * scale
      ctx.beginPath()
      ctx.moveTo(gridX, offsetY)
      ctx.lineTo(gridX, offsetY + courtLengthPx)
      ctx.stroke()
    }
    
    // Horizontal grid (every 1m)
    for (let y = 1; y < COURT_LENGTH; y++) {
      const gridY = offsetY + y * scale
      ctx.beginPath()
      ctx.moveTo(offsetX, gridY)
      ctx.lineTo(offsetX + courtWidthPx, gridY)
      ctx.stroke()
    }
  }
}

/**
 * Draw player trails (position history)
 * Uses accumulated smoothed positions that match the actual player circle positions.
 * This follows YOLO26's track visualization pattern: draw polylines from track_history.
 */
function drawPlayerTrails(ctx: CanvasRenderingContext2D) {
  if (!props.showTrails) return
  
  const trails = playerTrails.value
  
  trails.forEach((trail, playerId) => {
    if (trail.length < 2) return
    
    // Use same color mapping as drawPlayers (player_id is 0-based)
    const color = PLAYER_COLORS[playerId % PLAYER_COLORS.length] ?? '#FF6B6B'
    
    // Draw trail as a gradient line that fades from old to new
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    
    // Draw trail segments with gradient opacity
    for (let i = 1; i < trail.length; i++) {
      const prev = trail[i - 1]!
      const curr = trail[i]!
      const prevCanvas = courtToCanvas(prev.x, prev.y)
      const currCanvas = courtToCanvas(curr.x, curr.y)
      
      // Fade opacity based on position in trail (older = more transparent)
      const progress = i / trail.length
      const alpha = 0.15 + progress * 0.65 // Range from 0.15 to 0.80
      const lineWidth = 1.5 + progress * 2 // Range from 1.5 to 3.5
      
      ctx.beginPath()
      ctx.moveTo(prevCanvas.x, prevCanvas.y)
      ctx.lineTo(currCanvas.x, currCanvas.y)
      ctx.strokeStyle = hexToRgba(color, alpha)
      ctx.lineWidth = lineWidth
      ctx.stroke()
    }
    
    // Draw small dots at key positions along the trail
    const dotInterval = Math.max(1, Math.floor(trail.length / 10)) // Show ~10 dots max
    for (let i = 0; i < trail.length; i += dotInterval) {
      const point = trail[i]!
      const canvasPos = courtToCanvas(point.x, point.y)
      const progress = i / trail.length
      const alpha = 0.2 + progress * 0.5
      const radius = 2 + progress * 2
      
      ctx.beginPath()
      ctx.arc(canvasPos.x, canvasPos.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = hexToRgba(color, alpha)
      ctx.fill()
    }
  })
}

/**
 * Convert hex color to rgba with specified alpha
 */
function hexToRgba(hex: string, alpha: number): string {
  // Remove # if present
  const cleanHex = hex.replace('#', '')
  const r = parseInt(cleanHex.substring(0, 2), 16)
  const g = parseInt(cleanHex.substring(2, 4), 16)
  const b = parseInt(cleanHex.substring(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/**
 * Draw cumulative hit markers on the court.
 * Each marker is a small circle at the court position where a hit was detected,
 * labeled with a sequential number per player. All previous markers persist on
 * screen to build a spatial history of strike locations.
 */
function drawHitMarkers(ctx: CanvasRenderingContext2D) {
  if (!props.showHitMarkers) return
  
  const markers = hitMarkers.value
  if (markers.size === 0) return
  
  ctx.save()
  
  for (const [playerId, playerMarkers] of markers) {
    const color = HIT_MARKER_COLORS[playerId % HIT_MARKER_COLORS.length] ?? '#FF2952'
    
    for (const marker of playerMarkers) {
      const canvasPos = courtToCanvas(marker.courtX, marker.courtY)
      
      // Outer glow ring
      ctx.beginPath()
      ctx.arc(canvasPos.x, canvasPos.y, 9, 0, Math.PI * 2)
      ctx.fillStyle = hexToRgba(color, 0.15)
      ctx.fill()
      
      // Outer ring (stroke)
      ctx.beginPath()
      ctx.arc(canvasPos.x, canvasPos.y, 8, 0, Math.PI * 2)
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.stroke()
      
      // Filled inner circle
      ctx.beginPath()
      ctx.arc(canvasPos.x, canvasPos.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = hexToRgba(color, 0.75)
      ctx.fill()
      
      // White border for legibility
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.lineWidth = 0.5
      ctx.stroke()
      
      // Hit number label (sequential per player)
      ctx.font = 'bold 7px Inter, sans-serif'
      ctx.fillStyle = '#FFFFFF'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(`${marker.hitNumber}`, canvasPos.x, canvasPos.y)
    }
  }
  
  ctx.restore()
}

/**
 * Get the total number of detected hits across all players (for legend display).
 */
function getTotalHitCount(): number {
  let total = 0
  for (const [, markers] of hitMarkers.value) {
    total += markers.length
  }
  return total
}

/**
 * Get feet position from player keypoints (ankle midpoint)
 * Falls back to center if ankles are not available
 */
function getFeetPosition(player: FramePlayer): { x: number; y: number } | null {
  // COCO keypoint indices for ankles
  const LEFT_ANKLE_IDX = 15
  const RIGHT_ANKLE_IDX = 16
  const LEFT_KNEE_IDX = 13
  const RIGHT_KNEE_IDX = 14
  const LEFT_HIP_IDX = 11
  const RIGHT_HIP_IDX = 12
  
  const keypoints = player.keypoints
  const CONFIDENCE_THRESHOLD = 0.3
  
  if (keypoints && keypoints.length >= 17) {
    const leftAnkle = keypoints[LEFT_ANKLE_IDX]
    const rightAnkle = keypoints[RIGHT_ANKLE_IDX]
    
    // Primary: use ankle midpoint (most accurate for court position)
    if (leftAnkle && rightAnkle &&
        leftAnkle.x !== null && leftAnkle.y !== null &&
        rightAnkle.x !== null && rightAnkle.y !== null &&
        leftAnkle.confidence > CONFIDENCE_THRESHOLD &&
        rightAnkle.confidence > CONFIDENCE_THRESHOLD) {
      return {
        x: (leftAnkle.x + rightAnkle.x) / 2,
        y: (leftAnkle.y + rightAnkle.y) / 2
      }
    }
    
    // Fallback 1: use just one ankle if the other isn't visible
    if (leftAnkle && leftAnkle.x !== null && leftAnkle.y !== null &&
        leftAnkle.confidence > CONFIDENCE_THRESHOLD) {
      return { x: leftAnkle.x, y: leftAnkle.y }
    }
    if (rightAnkle && rightAnkle.x !== null && rightAnkle.y !== null &&
        rightAnkle.confidence > CONFIDENCE_THRESHOLD) {
      return { x: rightAnkle.x, y: rightAnkle.y }
    }
    
    // Fallback 2: use knee midpoint if ankles not visible
    const leftKnee = keypoints[LEFT_KNEE_IDX]
    const rightKnee = keypoints[RIGHT_KNEE_IDX]
    if (leftKnee && rightKnee &&
        leftKnee.x !== null && leftKnee.y !== null &&
        rightKnee.x !== null && rightKnee.y !== null &&
        leftKnee.confidence > CONFIDENCE_THRESHOLD &&
        rightKnee.confidence > CONFIDENCE_THRESHOLD) {
      return {
        x: (leftKnee.x + rightKnee.x) / 2,
        y: (leftKnee.y + rightKnee.y) / 2
      }
    }
    
    // Fallback 3: use hip midpoint if lower body not visible
    const leftHip = keypoints[LEFT_HIP_IDX]
    const rightHip = keypoints[RIGHT_HIP_IDX]
    if (leftHip && rightHip &&
        leftHip.x !== null && leftHip.y !== null &&
        rightHip.x !== null && rightHip.y !== null &&
        leftHip.confidence > CONFIDENCE_THRESHOLD &&
        rightHip.confidence > CONFIDENCE_THRESHOLD) {
      return {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
      }
    }
  }
  
  // Final fallback: use player.center (torso/mean of keypoints)
  return player.center || null
}

// Color used when a player is not recognized (gray/muted)
const UNRECOGNIZED_COLOR = '#888888'

/**
 * Draw players on the court using smoothed positions for fluid motion.
 * Reads from smoothedPlayerPositions map which is updated by the animation loop.
 * Players not currently recognized are shown with a gray circle at their last known position.
 */
function drawPlayers(ctx: CanvasRenderingContext2D) {
  if (smoothedPlayerPositions.size === 0) return
  
  for (const [playerId, smoothed] of smoothedPlayerPositions) {
    // Convert smoothed court coordinates to canvas coordinates
    const canvasPos = courtToCanvas(smoothed.courtX, smoothed.courtY)
    
    // Use gray when not recognized, normal player color when recognized
    const normalColor = PLAYER_COLORS[playerId % PLAYER_COLORS.length] ?? '#FF6B6B'
    const color = smoothed.isRecognized ? normalColor : UNRECOGNIZED_COLOR
    
    // Draw player circle with glow effect
    ctx.shadowColor = color
    ctx.shadowBlur = smoothed.isRecognized ? 10 : 4
    
    // Outer glow
    ctx.beginPath()
    ctx.arc(canvasPos.x, canvasPos.y, 14, 0, Math.PI * 2)
    ctx.fillStyle = smoothed.isRecognized ? (color + '40') : 'rgba(136, 136, 136, 0.15)'
    ctx.fill()
    
    // Main circle
    ctx.beginPath()
    ctx.arc(canvasPos.x, canvasPos.y, 10, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    
    // Dashed border for unrecognized players, solid for recognized
    if (!smoothed.isRecognized) {
      ctx.setLineDash([3, 3])
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
    } else {
      ctx.setLineDash([])
      ctx.strokeStyle = '#FFFFFF'
    }
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.setLineDash([])
    
    ctx.shadowBlur = 0
    
    // Player label (player_id is 0-indexed, display as 1-indexed: Player 1, Player 2)
    ctx.font = 'bold 11px Inter, sans-serif'
    ctx.fillStyle = smoothed.isRecognized ? '#FFFFFF' : 'rgba(255, 255, 255, 0.7)'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`P${playerId + 1}`, canvasPos.x, canvasPos.y)
    
    // Speed indicator below player (only when recognized and moving)
    if (smoothed.isRecognized && smoothed.speed > 0.1) {
      ctx.font = '9px Inter, sans-serif'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.fillText(`${smoothed.speed.toFixed(1)} km/h`, canvasPos.x, canvasPos.y + 20)
    }
  }
}

/**
 * Draw shuttle position
 */
function drawShuttle(ctx: CanvasRenderingContext2D) {
  if (!props.showShuttle || !smoothedShuttle) return
  
  // Use smoothed shuttle position for fluid motion
  const canvasPos = courtToCanvas(smoothedShuttle.courtX, smoothedShuttle.courtY)
  
  // Draw shuttle as orange diamond
  ctx.save()
  ctx.translate(canvasPos.x, canvasPos.y)
  ctx.rotate(Math.PI / 4)
  
  ctx.shadowColor = '#FFA500'
  ctx.shadowBlur = 8
  
  ctx.fillStyle = '#FFA500'
  ctx.fillRect(-5, -5, 10, 10)
  
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = 1.5
  ctx.strokeRect(-5, -5, 10, 10)
  
  ctx.restore()
}

/**
 * Draw keypoint selection guide overlay
 * Shows numbered positions on the court for each keypoint
 */
function drawKeypointSelectionGuide(ctx: CanvasRenderingContext2D) {
  if (!props.isKeypointSelectionMode) return
  
  const count = props.keypointSelectionCount
  
  // Keypoint labels (abbreviated)
  const KEYPOINT_LABELS = ['TL', 'TR', 'BR', 'BL', 'NL', 'NR', 'SNL', 'SNR', 'SFL', 'SFR', 'CTN', 'CTF']
  
  // Draw each keypoint position
  COURT_KEYPOINT_POSITIONS.forEach((pos, index) => {
    const canvasPos = courtToCanvas(pos[0]!, pos[1]!)
    const isCompmleted = index < count
    const isActive = index === count
    const isPending = index > count
    
    // Draw the point circle
    ctx.beginPath()
    ctx.arc(canvasPos.x, canvasPos.y, isActive ? 14 : 10, 0, Math.PI * 2)
    
    if (isActive) {
      // Active point - bright green with pulse effect (simulated with glow)
      ctx.fillStyle = '#22c55e'
      ctx.shadowColor = '#22c55e'
      ctx.shadowBlur = 12
    } else if (isCompmleted) {
      // Completed point - dimmed green
      ctx.fillStyle = 'rgba(34, 197, 94, 0.4)'
      ctx.shadowBlur = 0
    } else {
      // Pending point - gray
      ctx.fillStyle = 'rgba(100, 100, 100, 0.6)'
      ctx.shadowBlur = 0
    }
    
    ctx.fill()
    ctx.shadowBlur = 0
    
    // Draw border
    ctx.strokeStyle = isActive ? '#22c55e' : isCompmleted ? 'rgba(34, 197, 94, 0.6)' : 'rgba(150, 150, 150, 0.5)'
    ctx.lineWidth = 2
    ctx.stroke()
    
    // Draw number
    ctx.font = isActive ? 'bold 11px Inter, sans-serif' : '10px Inter, sans-serif'
    ctx.fillStyle = isActive ? '#000' : isCompmleted ? '#22c55e' : 'rgba(255, 255, 255, 0.6)'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${index + 1}`, canvasPos.x, canvasPos.y)
    
    // Draw label below for active or completed points
    if (isActive || isCompmleted) {
      ctx.font = '8px Inter, sans-serif'
      ctx.fillStyle = isActive ? '#22c55e' : 'rgba(34, 197, 94, 0.7)'
      ctx.fillText(KEYPOINT_LABELS[index]!, canvasPos.x, canvasPos.y + 18)
    }
  })
  
  // Draw instruction at the bottom of the canvas
  const instructionY = canvasHeight.value - 8
  if (count < 12) {
    ctx.font = 'bold 11px Inter, sans-serif'
    ctx.fillStyle = '#22c55e'
    ctx.textAlign = 'center'
    ctx.fillText(`Click point ${count + 1}`, props.width / 2, instructionY)
  } else {
    ctx.font = 'bold 11px Inter, sans-serif'
    ctx.fillStyle = '#22c55e'
    ctx.textAlign = 'center'
    ctx.fillText('✓ All set!', props.width / 2, instructionY)
  }
}

/**
 * Draw "No Court Detected" message
 */
function drawNoCourtMessage(ctx: CanvasRenderingContext2D) {
  const cHeight = canvasHeight.value
  ctx.fillStyle = '#1e1e2e'
  ctx.fillRect(0, 0, props.width, cHeight)
  
  // Draw placeholder court outline
  const padding = 30
  const courtWidth = props.width - padding * 2
  const courtHeight = cHeight - padding * 2
  
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
  ctx.lineWidth = 2
  ctx.setLineDash([10, 10])
  ctx.strokeRect(padding, padding, courtWidth, courtHeight)
  ctx.setLineDash([])
  
  // Draw message
  ctx.font = 'bold 14px Inter, sans-serif'
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('Court Detection Required', props.width / 2, cHeight / 2 - 20)
  
  ctx.font = '12px Inter, sans-serif'
  ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'
  ctx.fillText('Set court corners manually or', props.width / 2, cHeight / 2 + 5)
  ctx.fillText('wait for automatic detection', props.width / 2, cHeight / 2 + 22)
}

// =============================================================================
// SMOOTH ANIMATION LOOP
// =============================================================================

/**
 * Update target positions from current props.
 * Transforms video-space player/shuttle positions to court-space targets.
 * Called when player/shuttle props change.
 */
// Maximum consecutive unrecognized frames before removing a player from the court
const MAX_UNRECOGNIZED_FRAMES = 300 // ~10 seconds at 30fps

// Minimum distance (in court meters) between two players to accept a position update.
// If a player's new position is closer than this to another player, it's likely a
// misassignment from the detection system and the update is rejected.
const MIN_PLAYER_DISTANCE = 0.8 // meters (~0.8m, players can't realistically overlap)

// Maximum distance a player can move per frame (in court meters).
// At 30fps, a player sprinting at ~25 km/h covers ~0.23m per frame.
// We use a generous threshold to allow lunges/dives but reject teleportation.
const MAX_JUMP_PER_FRAME = 1.0 // meters per frame

/**
 * Squared distance helper
 */
function distSq(x1: number, y1: number, x2: number, y2: number): number {
  return (x1 - x2) ** 2 + (y1 - y2) ** 2
}

function updateTargetPositions() {
  const H = homographyMatrix.value
  if (!H) return
  
  // Update player targets
  if (props.players && props.players.length > 0) {
    const activeIds = new Set<number>()
    
    // Phase 1: Compute candidate court positions for all detected players
    const candidates: { playerId: number; courtX: number; courtY: number; speed: number }[] = []
    
    for (const player of props.players) {
      const feetPos = getFeetPosition(player)
      if (!feetPos) continue
      
      const courtPos = applyHomography(H, feetPos.x, feetPos.y)
      if (!courtPos) continue
      
      // Bounds check - if out of bounds, treat as unrecognized (likely bad detection)
      const margin = 2
      if (courtPos.x < -margin || courtPos.x > COURT_WIDTH + margin ||
          courtPos.y < -margin || courtPos.y > COURT_LENGTH + margin) {
        continue
      }
      
      candidates.push({
        playerId: player.player_id,
        courtX: courtPos.x,
        courtY: courtPos.y,
        speed: player.current_speed ?? 0
      })
    }
    
    // Phase 2: Detect and fix ID swaps + proximity conflicts
    // An ID swap is when the detector temporarily assigns P1's position to P2 and vice versa.
    // We detect this by comparing "normal" vs "swapped" assignment costs for each pair.
    const rejectedIds = new Set<number>()
    
    for (let i = 0; i < candidates.length; i++) {
      for (let j = i + 1; j < candidates.length; j++) {
        const a = candidates[i]!
        const b = candidates[j]!
        
        const existingA = smoothedPlayerPositions.get(a.playerId)
        const existingB = smoothedPlayerPositions.get(b.playerId)
        
        // --- Proximity check: two candidates landing on the same spot ---
        const pairDist = Math.sqrt(distSq(a.courtX, a.courtY, b.courtX, b.courtY))
        if (pairDist < MIN_PLAYER_DISTANCE) {
          if (existingA && existingB) {
            const moveA = Math.sqrt(distSq(a.courtX, a.courtY, existingA.courtX, existingA.courtY))
            const moveB = Math.sqrt(distSq(b.courtX, b.courtY, existingB.courtX, existingB.courtY))
            rejectedIds.add(moveA > moveB ? a.playerId : b.playerId)
          } else if (existingA) {
            rejectedIds.add(b.playerId)
          } else if (existingB) {
            rejectedIds.add(a.playerId)
          }
          continue
        }
        
        // --- ID swap detection: check if swapping IDs gives a better fit ---
        if (existingA && existingB) {
          // Normal assignment cost: A→A_old + B→B_old
          const normalCost = Math.sqrt(distSq(a.courtX, a.courtY, existingA.courtX, existingA.courtY))
                           + Math.sqrt(distSq(b.courtX, b.courtY, existingB.courtX, existingB.courtY))
          
          // Swapped assignment cost: A→B_old + B→A_old
          const swappedCost = Math.sqrt(distSq(a.courtX, a.courtY, existingB.courtX, existingB.courtY))
                            + Math.sqrt(distSq(b.courtX, b.courtY, existingA.courtX, existingA.courtY))
          
          // If swapped assignment is significantly better, it's an ID swap
          // The swapped cost must be less than half the normal cost to trigger
          // (this avoids false positives when players are near each other legitimately)
          if (swappedCost < normalCost * 0.5) {
            // ID swap detected - reject both updates, keep last known positions
            rejectedIds.add(a.playerId)
            rejectedIds.add(b.playerId)
          }
        }
      }
    }
    
    // Phase 2b: Single-player jump check
    // Reject any player whose position jumped unrealistically far in one frame
    for (const candidate of candidates) {
      if (rejectedIds.has(candidate.playerId)) continue
      
      const existing = smoothedPlayerPositions.get(candidate.playerId)
      if (existing && existing.isRecognized) {
        const jumpDist = Math.sqrt(distSq(
          candidate.courtX, candidate.courtY,
          existing.courtX, existing.courtY
        ))
        if (jumpDist > MAX_JUMP_PER_FRAME) {
          rejectedIds.add(candidate.playerId)
        }
      }
    }
    
    // Phase 3: Apply accepted candidate positions
    for (const candidate of candidates) {
      if (rejectedIds.has(candidate.playerId)) continue
      
      // Mark this player as successfully recognized in this frame
      activeIds.add(candidate.playerId)
      
      const existing = smoothedPlayerPositions.get(candidate.playerId)
      if (existing) {
        // Update target - animation loop will smoothly interpolate
        existing.targetCourtX = candidate.courtX
        existing.targetCourtY = candidate.courtY
        existing.targetSpeed = candidate.speed
        existing.isRecognized = true
        existing.unrecognizedFrames = 0
      } else {
        // New player - snap to position immediately (no lag on first appearance)
        smoothedPlayerPositions.set(candidate.playerId, {
          courtX: candidate.courtX,
          courtY: candidate.courtY,
          targetCourtX: candidate.courtX,
          targetCourtY: candidate.courtY,
          speed: candidate.speed,
          targetSpeed: candidate.speed,
          isRecognized: true,
          unrecognizedFrames: 0
        })
      }
    }
    
    // For players no longer recognized: keep last position, mark as unrecognized
    for (const [id, pos] of smoothedPlayerPositions) {
      if (!activeIds.has(id)) {
        pos.isRecognized = false
        pos.unrecognizedFrames++
        // Freeze position: set target to current so smoothing stops
        pos.targetCourtX = pos.courtX
        pos.targetCourtY = pos.courtY
        pos.targetSpeed = 0
        pos.speed = 0
        
        // Remove player after being unrecognized for too long
        if (pos.unrecognizedFrames > MAX_UNRECOGNIZED_FRAMES) {
          smoothedPlayerPositions.delete(id)
        }
      }
    }
  } else {
    // No players at all - mark all as unrecognized but keep their positions
    for (const [id, pos] of smoothedPlayerPositions) {
      pos.isRecognized = false
      pos.unrecognizedFrames++
      pos.targetCourtX = pos.courtX
      pos.targetCourtY = pos.courtY
      pos.targetSpeed = 0
      pos.speed = 0
      
      if (pos.unrecognizedFrames > MAX_UNRECOGNIZED_FRAMES) {
        smoothedPlayerPositions.delete(id)
      }
    }
  }
  
  // Update shuttle target
  if (props.shuttlePosition && props.showShuttle) {
    const courtPos = applyHomography(H, props.shuttlePosition.x, props.shuttlePosition.y)
    if (courtPos &&
        courtPos.x >= -1 && courtPos.x <= COURT_WIDTH + 1 &&
        courtPos.y >= -1 && courtPos.y <= COURT_LENGTH + 1) {
      if (smoothedShuttle) {
        smoothedShuttle.targetCourtX = courtPos.x
        smoothedShuttle.targetCourtY = courtPos.y
      } else {
        smoothedShuttle = {
          courtX: courtPos.x,
          courtY: courtPos.y,
          targetCourtX: courtPos.x,
          targetCourtY: courtPos.y
        }
      }
    }
  } else {
    smoothedShuttle = null
  }
  
  // Record smoothed positions into trails (follows YOLO26 track_history pattern)
  recordTrailPositions()
  
  // Run hit detection on skeleton data up to current frame
  if (props.showHitMarkers) {
    detectHits()
  }
}

/**
 * Interpolate smoothed positions toward targets using exponential smoothing.
 * Returns true if any position was updated (requires re-render).
 */
function interpolatePositions(deltaTime: number): boolean {
  let anyMoved = false
  const factor = 1 - Math.exp(-SMOOTHING_SPEED * deltaTime)
  
  for (const [, pos] of smoothedPlayerPositions) {
    const dx = pos.targetCourtX - pos.courtX
    const dy = pos.targetCourtY - pos.courtY
    const ds = pos.targetSpeed - pos.speed
    
    if (Math.abs(dx) > 0.002 || Math.abs(dy) > 0.002) {
      pos.courtX += dx * factor
      pos.courtY += dy * factor
      pos.speed += ds * factor
      anyMoved = true
    } else {
      // Snap to target when close enough to avoid endless micro-updates
      pos.courtX = pos.targetCourtX
      pos.courtY = pos.targetCourtY
      pos.speed = pos.targetSpeed
    }
  }
  
  if (smoothedShuttle) {
    const dx = smoothedShuttle.targetCourtX - smoothedShuttle.courtX
    const dy = smoothedShuttle.targetCourtY - smoothedShuttle.courtY
    
    if (Math.abs(dx) > 0.002 || Math.abs(dy) > 0.002) {
      smoothedShuttle.courtX += dx * factor
      smoothedShuttle.courtY += dy * factor
      anyMoved = true
    } else {
      smoothedShuttle.courtX = smoothedShuttle.targetCourtX
      smoothedShuttle.courtY = smoothedShuttle.targetCourtY
    }
  }
  
  return anyMoved
}

/**
 * Animation loop for smooth position rendering.
 * Runs via requestAnimationFrame, interpolating positions each display frame.
 * Automatically stops when all positions have converged to save CPU.
 */
function animationLoop(timestamp: number) {
  if (!isAnimating) return
  
  const deltaTime = lastAnimTime > 0
    ? Math.min((timestamp - lastAnimTime) / 1000, 0.05) // Cap at 50ms to avoid huge jumps
    : 1 / 60
  lastAnimTime = timestamp
  
  const moved = interpolatePositions(deltaTime)
  if (moved) {
    renderFrame()
    animFrameId = requestAnimationFrame(animationLoop)
  } else {
    // All positions converged - stop loop to save CPU
    // Will restart when new target positions arrive
    isAnimating = false
    animFrameId = null
  }
}

/**
 * Start the smooth animation loop (idempotent)
 */
function startAnimLoop() {
  if (isAnimating) return
  isAnimating = true
  lastAnimTime = 0
  animFrameId = requestAnimationFrame(animationLoop)
}

/**
 * Stop the smooth animation loop
 */
function stopAnimLoop() {
  isAnimating = false
  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId)
    animFrameId = null
  }
}

/**
 * Render a single frame - draws court, trails, players, and shuttle.
 * Called from the animation loop for smooth rendering, or directly when config changes.
 */
function renderFrame() {
  const canvas = canvasRef.value
  if (!canvas) return
  
  // Cache canvas context
  if (!cachedCtx) {
    cachedCtx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true // Reduces latency on supported browsers
    })
  }
  const ctx = cachedCtx
  if (!ctx) return
  
  // Clear canvas with dark background (ensures no transparent gaps)
  ctx.fillStyle = '#111111'
  ctx.fillRect(0, 0, props.width, canvasHeight.value)
  
  // Always draw the court first (even without court corner data)
  drawCourt(ctx)
  
  // In keypoint selection mode, show the keypoint guide overlay
  if (props.isKeypointSelectionMode) {
    drawKeypointSelectionGuide(ctx)
    return
  }
  
  // Only draw players/trails/shuttle when we have valid court corners for tracking
  if (!effectiveCourtCorners.value) {
    return
  }
  
  // Draw player trails
  drawPlayerTrails(ctx)
  
  // Draw cumulative hit markers (below players, above trails)
  drawHitMarkers(ctx)
  
  // Draw shuttle
  drawShuttle(ctx)
  
  // Draw players
  drawPlayers(ctx)
}

/**
 * Full render: update targets from current props, then draw.
 * Used for initial render and when non-positional config changes.
 */
function render() {
  updateTargetPositions()
  renderFrame()
}

// =============================================================================
// WATCHERS
// =============================================================================

// Watch for player/shuttle position changes → update targets & start animating
watch([() => props.players, () => props.shuttlePosition], () => {
  updateTargetPositions()
  // Start animation loop if we have data to animate
  if (smoothedPlayerPositions.size > 0 || smoothedShuttle) {
    startAnimLoop()
  }
  // Also render immediately for responsiveness
  renderFrame()
}, { deep: true })

// Watch for court/display config changes → full re-render
watch([
  () => props.courtCorners,
  () => props.manualKeypoints,
  () => props.showGrid,
  () => props.showLabels,
  () => props.showShuttle,
  () => props.showTrails,
  () => props.showHitMarkers,
  () => props.isKeypointSelectionMode,
  () => props.keypointSelectionCount,
  () => props.height,
  () => props.width,
  () => props.skeletonData
], () => {
  // Reset cached context when dimensions or court data change
  cachedCtx = null
  // Re-compute targets in case homography changed
  updateTargetPositions()
  renderFrame()
}, { deep: true })

// Watch for frame changes — clear trails on seek (backward jump)
watch(() => props.currentFrame, (newFrame, oldFrame) => {
  // Detect seeking backward: clear accumulated trails and hit markers
  if (oldFrame !== undefined && newFrame !== undefined && newFrame < oldFrame - 1) {
    playerTrails.value.clear()
    lastTrailFrame = -1
    // Reset hit detection so markers are recomputed from frame 0 to new position
    resetHitDetection()
  }
  if (props.showTrails || props.showHitMarkers) {
    renderFrame()
  }
})

// Initial render
onMounted(() => {
  updateTargetPositions()
  renderFrame()
})

// Cleanup
onUnmounted(() => {
  playerTrails.value.clear()
  resetHitDetection()
  stopAnimLoop()
  smoothedPlayerPositions.clear()
  smoothedShuttle = null
  cachedCtx = null
})

// Expose for parent components
defineExpose({
  render: () => { updateTargetPositions(); renderFrame() },
  clearTrails: () => playerTrails.value.clear(),
  clearHitMarkers: () => resetHitDetection(),
  getHitMarkers: () => hitMarkers.value,
  getTotalHitCount
})
</script>

<template>
  <div class="mini-court-container" :style="{ height: `${height}px` }">
    <div class="mini-court-header">
      <span class="title">🏸 Court View</span>
      <span v-if="effectiveCourtCorners && keypointCount === 12" class="status active precision">12pt ✓</span>
      <span v-else-if="effectiveCourtCorners" class="status active">4pt</span>
      <span v-else class="status inactive">No Data</span>
    </div>
    <div class="canvas-wrapper">
      <canvas
        ref="canvasRef"
        :width="width"
        :height="canvasHeight"
        class="mini-court-canvas"
      />
    </div>
    <div v-if="players && players.length > 0" class="player-legend">
      <div
        v-for="(player, index) in players"
        :key="player.player_id"
        class="legend-item"
      >
        <span
          class="legend-dot"
          :style="{ backgroundColor: PLAYER_COLORS[index % PLAYER_COLORS.length] }"
        />
        <span class="legend-label">P{{ player.player_id + 1 }}</span>
        <span
          v-if="showHitMarkers && hitMarkers.get(player.player_id)?.length"
          class="hit-count"
          :style="{ color: HIT_MARKER_COLORS[index % HIT_MARKER_COLORS.length] }"
        >
          {{ hitMarkers.get(player.player_id)?.length }} hits
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mini-court-container {
  display: flex;
  flex-direction: column;
  background: #141414;
  border-radius: 0;
  padding: 12px;
  border: 1px solid #222;
}

.mini-court-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  padding: 0 4px;
}

.title {
  font-size: 14px;
  font-weight: 600;
  color: #fff;
}

.status {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 0;
}

.status.active {
  background-color: #1a1a1a;
  color: #22c55e;
  border: 1px solid #333;
}

.status.active.precision {
  background-color: #1a1a1a;
  color: #3b82f6;
  border: 1px solid #333;
}

.status.inactive {
  background-color: #1a1a1a;
  color: #666;
  border: 1px solid #333;
}

.canvas-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 0;
}

.mini-court-canvas {
  border-radius: 0;
  /* Background is drawn programmatically on the canvas to avoid transparency issues */
}

.player-legend {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #222;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1px solid #444;
}

.legend-label {
  font-size: 11px;
  color: #888;
  font-weight: 500;
}

.hit-count {
  font-size: 9px;
  font-weight: 600;
  margin-left: 2px;
  opacity: 0.9;
}
</style>
