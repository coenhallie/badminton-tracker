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
  isKeypointSelectionMode: false,
  keypointSelectionCount: 0,
  maxTrailLength: 100 // Show last 100 positions (~3.3 seconds at 30fps)
})

const canvasRef = ref<HTMLCanvasElement | null>(null)

// PERFORMANCE OPTIMIZATION: Cache the canvas 2D context
let cachedCtx: CanvasRenderingContext2D | null = null

// PERFORMANCE OPTIMIZATION: Throttle render calls
let lastRenderTime = 0
let pendingRenderFrame: number | null = null
const RENDER_THROTTLE_MS = 33 // ~30fps max

// PERFORMANCE OPTIMIZATION: Track last player positions to skip redundant renders
let lastPlayerHash = ''

// Store position history for trails (legacy - only used when skeletonData is not provided)
const playerTrails = ref<Map<number, { x: number; y: number }[]>>(new Map())
const MAX_TRAIL_LENGTH = 30

// =============================================================================
// DYNAMIC TRAIL COMPUTATION from skeleton data
// =============================================================================
// When skeletonData and currentFrame are provided, compute trails dynamically
// by looking back through the history of frames up to the current frame.
// This gives a smooth trail that shows where each player has been.
// =============================================================================

/**
 * Compute dynamic player trails from skeleton data up to the current frame
 * Returns a Map of player_id -> array of court positions (in meters)
 */
const dynamicPlayerTrails = computed((): Map<number, { x: number; y: number }[]> => {
  const trails = new Map<number, { x: number; y: number }[]>()
  
  // If no skeleton data or no homography matrix, return empty trails
  if (!props.skeletonData || props.skeletonData.length === 0 || !homographyMatrix.value) {
    return trails
  }
  
  const currentFrameNum = props.currentFrame ?? 0
  const maxLength = props.maxTrailLength ?? 100
  
  // Look back through frames to build trails
  // Start from the frame that gives us maxTrailLength points back
  const startIdx = Math.max(0, currentFrameNum - maxLength)
  const endIdx = Math.min(props.skeletonData.length - 1, currentFrameNum)
  
  for (let i = startIdx; i <= endIdx; i++) {
    const frame = props.skeletonData[i]
    if (!frame?.players) continue
    
    for (const player of frame.players) {
      // Get feet position for accurate court positioning
      const feetPos = getPlayerFeetPosition(player)
      if (!feetPos) continue
      
      // Transform to court coordinates
      const courtPos = applyHomography(homographyMatrix.value!, feetPos.x, feetPos.y)
      if (!courtPos) continue
      
      // Check bounds
      const margin = 1
      if (courtPos.x < -margin || courtPos.x > COURT_WIDTH + margin ||
          courtPos.y < -margin || courtPos.y > COURT_LENGTH + margin) {
        continue
      }
      
      // Add to player's trail
      let trail = trails.get(player.player_id)
      if (!trail) {
        trail = []
        trails.set(player.player_id, trail)
      }
      trail.push({ x: courtPos.x, y: courtPos.y })
    }
  }
  
  return trails
})

/**
 * Get feet position from player (similar to getFeetPosition but works with FramePlayer type)
 */
function getPlayerFeetPosition(player: FramePlayer): { x: number; y: number } | null {
  const LEFT_ANKLE_IDX = 15
  const RIGHT_ANKLE_IDX = 16
  const LEFT_KNEE_IDX = 13
  const RIGHT_KNEE_IDX = 14
  const LEFT_HIP_IDX = 11
  const RIGHT_HIP_IDX = 12
  const CONFIDENCE_THRESHOLD = 0.3
  
  const keypoints = player.keypoints
  
  if (keypoints && keypoints.length >= 17) {
    const leftAnkle = keypoints[LEFT_ANKLE_IDX]
    const rightAnkle = keypoints[RIGHT_ANKLE_IDX]
    
    // Primary: use ankle midpoint
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
    
    // Fallback: single ankle
    if (leftAnkle?.x !== null && leftAnkle?.y !== null && (leftAnkle?.confidence ?? 0) > CONFIDENCE_THRESHOLD) {
      return { x: leftAnkle!.x!, y: leftAnkle!.y! }
    }
    if (rightAnkle?.x !== null && rightAnkle?.y !== null && (rightAnkle?.confidence ?? 0) > CONFIDENCE_THRESHOLD) {
      return { x: rightAnkle!.x!, y: rightAnkle!.y! }
    }
    
    // Fallback: knees
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
    
    // Fallback: hips
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
  
  // Final fallback: use player center
  return player.center || null
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
 * Uses dynamicPlayerTrails when skeletonData is available, otherwise falls back to legacy playerTrails
 */
function drawPlayerTrails(ctx: CanvasRenderingContext2D) {
  if (!props.showTrails) return
  
  // Use dynamic trails from skeleton data if available, otherwise use legacy trails
  const trails = props.skeletonData && props.skeletonData.length > 0
    ? dynamicPlayerTrails.value
    : playerTrails.value
  
  trails.forEach((trail, playerId) => {
    if (trail.length < 2) return
    
    // Use player index for consistent coloring
    const colorIndex = playerId - 1 // player_id is 1-based
    const color = PLAYER_COLORS[colorIndex >= 0 ? colorIndex % PLAYER_COLORS.length : 0] ?? '#FF6B6B'
    
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

/**
 * Draw players on the court
 */
function drawPlayers(ctx: CanvasRenderingContext2D) {
  if (!props.players || props.players.length === 0) return
  
  props.players.forEach((player, index) => {
    // Get feet position (or fallback to center)
    const feetPos = getFeetPosition(player)
    if (!feetPos) return
    
    // Transform pixel position to court coordinates
    const courtPos = transformPlayerPosition(feetPos.x, feetPos.y)
    if (!courtPos) {
      if (DEBUG_MODE) {
        console.log('[MiniCourt] Player', player.player_id, 'transform failed for position:', feetPos)
      }
      return
    }
    
    if (DEBUG_MODE) {
      console.log('[MiniCourt] Player', player.player_id,
        'feet pos:', `(${feetPos.x.toFixed(0)}, ${feetPos.y.toFixed(0)})`,
        '-> court pos:', `(${courtPos.x.toFixed(2)}m, ${courtPos.y.toFixed(2)}m)`)
    }
    
    // Check if position is within court bounds (with some margin)
    const margin = 2 // meters
    if (courtPos.x < -margin || courtPos.x > COURT_WIDTH + margin ||
        courtPos.y < -margin || courtPos.y > COURT_LENGTH + margin) {
      if (DEBUG_MODE) {
        console.log('[MiniCourt] Player', player.player_id, 'outside bounds, skipping')
      }
      return // Skip players outside the court
    }
    
    // Update trail
    if (props.showTrails) {
      const trail = playerTrails.value.get(player.player_id) ?? []
      trail.push({ x: courtPos.x, y: courtPos.y })
      if (trail.length > MAX_TRAIL_LENGTH) {
        trail.shift()
      }
      playerTrails.value.set(player.player_id, trail)
    }
    
    // Convert to canvas coordinates
    const canvasPos = courtToCanvas(courtPos.x, courtPos.y)
    
    // Player color
    const color = PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B'
    
    // Draw player circle with glow effect
    ctx.shadowColor = color
    ctx.shadowBlur = 10
    
    // Outer glow
    ctx.beginPath()
    ctx.arc(canvasPos.x, canvasPos.y, 14, 0, Math.PI * 2)
    ctx.fillStyle = color + '40'
    ctx.fill()
    
    // Main circle
    ctx.beginPath()
    ctx.arc(canvasPos.x, canvasPos.y, 10, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = '#FFFFFF'
    ctx.lineWidth = 2
    ctx.stroke()
    
    ctx.shadowBlur = 0
    
    // Player label
    ctx.font = 'bold 11px Inter, sans-serif'
    ctx.fillStyle = '#FFFFFF'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`P${player.player_id}`, canvasPos.x, canvasPos.y)
    
    // Speed indicator below player
    if (player.current_speed !== undefined && player.current_speed > 0) {
      ctx.font = '9px Inter, sans-serif'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.fillText(`${player.current_speed.toFixed(1)} km/h`, canvasPos.x, canvasPos.y + 20)
    }
  })
}

/**
 * Draw shuttle position
 */
function drawShuttle(ctx: CanvasRenderingContext2D) {
  if (!props.showShuttle || !props.shuttlePosition) return
  
  // Transform shuttle position
  const courtPos = transformPlayerPosition(props.shuttlePosition.x, props.shuttlePosition.y)
  if (!courtPos) return
  
  // Check bounds
  if (courtPos.x < -1 || courtPos.x > COURT_WIDTH + 1 ||
      courtPos.y < -1 || courtPos.y > COURT_LENGTH + 1) {
    return
  }
  
  const canvasPos = courtToCanvas(courtPos.x, courtPos.y)
  
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

/**
 * Compute a simple hash of player positions to detect changes
 */
function computePlayerHash(): string {
  if (!props.players || props.players.length === 0) return ''
  return props.players.map(p =>
    `${p.player_id}:${p.center?.x?.toFixed(1) ?? ''}:${p.center?.y?.toFixed(1) ?? ''}`
  ).join('|')
}

/**
 * Main render function with throttling
 * PERFORMANCE OPTIMIZATION: Throttles renders to ~30fps and skips
 * redundant renders when player positions haven't changed
 */
function render(force = false) {
  const canvas = canvasRef.value
  if (!canvas) return
  
  // PERFORMANCE OPTIMIZATION: Skip render if positions haven't changed
  // But always render in keypoint selection mode
  const currentHash = computePlayerHash()
  if (!force && !props.isKeypointSelectionMode && currentHash === lastPlayerHash) return
  lastPlayerHash = currentHash
  
  // PERFORMANCE OPTIMIZATION: Throttle render calls
  const now = performance.now()
  if (!force && now - lastRenderTime < RENDER_THROTTLE_MS) {
    // Schedule a render for later if not already pending
    if (pendingRenderFrame === null) {
      pendingRenderFrame = requestAnimationFrame(() => {
        pendingRenderFrame = null
        render(true)
      })
    }
    return
  }
  lastRenderTime = now
  
  // PERFORMANCE OPTIMIZATION: Cache canvas context
  if (!cachedCtx) {
    cachedCtx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true // Reduces latency on supported browsers
    })
  }
  const ctx = cachedCtx
  if (!ctx) return
  
  // Clear canvas
  ctx.clearRect(0, 0, props.width, canvasHeight.value)
  
  // Always draw the court first (even without court corner data)
  drawCourt(ctx)
  
  // In keypoint selection mode, show the keypoint guide overlay
  if (props.isKeypointSelectionMode) {
    // Draw keypoint selection guide on top
    drawKeypointSelectionGuide(ctx)
    return
  }
  
  // Only draw players/trails/shuttle when we have valid court corners for tracking
  if (!effectiveCourtCorners.value) {
    // No court corners - show the court but no player tracking
    return
  }
  
  // Draw player trails
  drawPlayerTrails(ctx)
  
  // Draw shuttle
  drawShuttle(ctx)
  
  // Draw players
  drawPlayers(ctx)
}

// Watch for changes and re-render
watch([
  () => props.players,
  () => props.courtCorners,
  () => props.manualKeypoints,
  () => props.shuttlePosition,
  () => props.showGrid,
  () => props.showLabels,
  () => props.showShuttle,
  () => props.showTrails,
  () => props.isKeypointSelectionMode,
  () => props.keypointSelectionCount,
  () => props.height,
  () => props.width,
  () => props.currentFrame,  // Re-render when frame changes for dynamic trails
  () => props.skeletonData   // Re-render when skeleton data changes
], () => {
  // Reset cached context when dimensions change
  cachedCtx = null
  render(true) // Force render when props change
}, { deep: true })

// Optimized watch for currentFrame changes (high frequency during playback)
// Use a separate watcher with throttling for smooth trail updates
watch(() => props.currentFrame, () => {
  // Don't force render, let the throttle in render() handle it
  render(false)
})

// Initial render
onMounted(() => {
  render(true)
})

// Cleanup
onUnmounted(() => {
  playerTrails.value.clear()
  // Cancel any pending render
  if (pendingRenderFrame !== null) {
    cancelAnimationFrame(pendingRenderFrame)
    pendingRenderFrame = null
  }
  // Clear cached context
  cachedCtx = null
})

// Expose for parent components
defineExpose({
  render,
  clearTrails: () => playerTrails.value.clear()
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
        <span class="legend-label">P{{ player.player_id }}</span>
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
  background: #1a472a;
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
</style>
