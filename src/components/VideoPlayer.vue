<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, shallowRef, nextTick, toRef, inject, type Ref } from 'vue'
import type { SkeletonFrame, FramePlayer, Keypoint, BadmintonDetections, BoundingBoxDetection, ExtendedCourtKeypoints } from '@/types/analysis'
import { SKELETON_CONNECTIONS, PLAYER_COLORS } from '@/types/analysis'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'
import PoseOverlay from './PoseOverlay.vue'
import ShotSummaryOverlay from './ShotSummaryOverlay.vue'
import type { ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'
import SyntheticCourtView from './SyntheticCourtView.vue'
import ViewportControls from '@/components/ViewportControls.vue'
import { useVideoExport } from '@/composables/useVideoExport'
import { useViewportCamera } from '@/composables/useViewportCamera'
import { computeHomographyFromKeypoints, applyHomography } from '@/utils/homography'
import { legStretchMeters } from '@/utils/bodyAngles'

// =============================================================================
// PERFORMANCE OPTIMIZATION: Debug mode flag
// =============================================================================
// Set to true only during development to see debug logs
// MUST be false in production to avoid performance overhead
// =============================================================================
const DEBUG_MODE = import.meta.env.DEV && false // Disabled even in dev by default

// Injected lazily from App.vue; null until a videoId is available and the
// composable has been instantiated. Consumers must tolerate null fallback.
const playerLabelsRef = inject(PLAYER_LABELS_KEY)
const pidDisplayFor = (canonical: number): number =>
  playerLabelsRef?.value?.displayId(canonical) ?? canonical
const pidLabelFor = (canonical: number): string =>
  playerLabelsRef?.value?.labelFor(canonical) ?? `Player ${canonical + 1}`

// NOTE: CourtDetectionResult interface removed - automatic court detection disabled
// Manual court keypoints are now the only method for court calibration

// Heatmap data type
interface HeatmapData {
  video_id: string
  width: number
  height: number
  colormap: string
  combined_heatmap?: number[][]
  player_heatmaps?: Record<string, number[][]>
  total_frames: number
  player_position_counts: Record<string, number>
  video_width: number
  video_height: number
}

// =============================================================================
// PROGRESSIVE HEATMAP: Real-time heatmap generation during video playback
// =============================================================================
// Instead of showing a static pre-computed heatmap, we generate the heatmap
// progressively as the video plays, accumulating player positions up to the
// current frame. This gives a more intuitive visualization of player movement.
// =============================================================================

interface ProgressiveHeatmapState {
  // 2D array of heat values (0-1 range, normalized later)
  heatmap: Float32Array[]
  // Last frame that was processed
  lastProcessedFrame: number
  // Dimensions
  width: number
  height: number
  // Video dimensions for scaling
  videoWidth: number
  videoHeight: number
  // Max value for normalization
  maxValue: number
}

// Progressive heatmap constants
// High resolution for maximum accuracy (near PDF export quality)
// Modern browsers handle these sizes easily (~3.5MB memory for Float32 array)
const PROGRESSIVE_HEATMAP_WIDTH = 960   // Half of 1080p width - excellent quality
const PROGRESSIVE_HEATMAP_HEIGHT = 540  // Half of 1080p height - 16:9 aspect
const PROGRESSIVE_HEATMAP_RADIUS = 25   // Matches PDF's sigma=25 for smooth Gaussian blending

// Progressive heatmap state - persists across frames
let progressiveHeatmapState: ProgressiveHeatmapState | null = null
let lastProgressiveHeatmapFrame = -1

/**
 * Initialize or reset the progressive heatmap state
 */
function initProgressiveHeatmap(videoWidth: number, videoHeight: number): ProgressiveHeatmapState {
  const width = PROGRESSIVE_HEATMAP_WIDTH
  const height = PROGRESSIVE_HEATMAP_HEIGHT
  
  // Create 2D array of zeros
  const heatmap: Float32Array[] = []
  for (let y = 0; y < height; y++) {
    heatmap.push(new Float32Array(width))
  }
  
  return {
    heatmap,
    lastProcessedFrame: -1,
    width,
    height,
    videoWidth,
    videoHeight,
    maxValue: 0
  }
}

/**
 * Add heat at a position (applies Gaussian blur around the point)
 */
function addHeatAtPosition(
  state: ProgressiveHeatmapState,
  videoX: number,
  videoY: number,
  intensity: number = 1.0
) {
  // Convert video coordinates to heatmap coordinates
  const heatmapX = Math.floor((videoX / state.videoWidth) * state.width)
  const heatmapY = Math.floor((videoY / state.videoHeight) * state.height)
  
  // Apply Gaussian-like heat around the position
  const radius = PROGRESSIVE_HEATMAP_RADIUS
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const x = heatmapX + dx
      const y = heatmapY + dy
      
      // Bounds check
      if (x < 0 || x >= state.width || y < 0 || y >= state.height) continue
      
      // Gaussian weight based on distance
      const distance = Math.sqrt(dx * dx + dy * dy)
      const weight = Math.exp(-(distance * distance) / (2 * (radius / 2) ** 2)) * intensity
      
      const row = state.heatmap[y]
      if (row) {
        row[x] = (row[x] || 0) + weight
        state.maxValue = Math.max(state.maxValue, row[x] || 0)
      }
    }
  }
}

/**
 * Compute progressive heatmap up to the current frame
 * This accumulates player positions from the beginning of the video
 * up to the current playback position.
 */
function computeProgressiveHeatmap(
  skeletonData: SkeletonFrame[],
  currentFrameNum: number,
  videoWidth: number,
  videoHeight: number,
  frameRange?: { start: number; end: number } | null
): ProgressiveHeatmapState | null {
  if (!skeletonData || skeletonData.length === 0) return null
  if (videoWidth <= 0 || videoHeight <= 0) return null

  // Initialize or check if we need to reset (if seeking backwards)
  if (!progressiveHeatmapState ||
      progressiveHeatmapState.videoWidth !== videoWidth ||
      progressiveHeatmapState.videoHeight !== videoHeight ||
      currentFrameNum < progressiveHeatmapState.lastProcessedFrame) {
    // Reset - either first time, dimensions changed, or we seeked backwards
    progressiveHeatmapState = initProgressiveHeatmap(videoWidth, videoHeight)
    if (DEBUG_MODE) {
      console.log('[Progressive Heatmap] Initialized/reset state for frame', currentFrameNum)
    }
  }

  // Process frames from last processed to current
  const startIdx = progressiveHeatmapState.lastProcessedFrame < 0 ? 0 :
    skeletonData.findIndex(f => f.frame > progressiveHeatmapState!.lastProcessedFrame)

  if (startIdx < 0) return progressiveHeatmapState
  
  let framesProcessed = 0
  for (let i = startIdx; i < skeletonData.length; i++) {
    const frame = skeletonData[i]
    if (!frame || frame.frame > currentFrameNum) break

    // Skip frames outside the rally frame range (if filtering by rally)
    if (frameRange && (frame.frame < frameRange.start || frame.frame > frameRange.end)) {
      progressiveHeatmapState.lastProcessedFrame = frame.frame
      continue
    }

    // Add heat for each player's position
    for (const player of frame.players) {
      // Use center point if available, otherwise try to compute from keypoints
      let posX: number | null = null
      let posY: number | null = null
      
      if (player.center) {
        posX = player.center.x
        posY = player.center.y
      } else if (player.keypoints && player.keypoints.length > 0) {
        // Use hip keypoints (11, 12) or any valid keypoint as fallback
        const hipLeft = player.keypoints[11]
        const hipRight = player.keypoints[12]
        
        if (hipLeft && hipRight &&
            hipLeft.x !== null && hipLeft.y !== null &&
            hipRight.x !== null && hipRight.y !== null &&
            hipLeft.confidence > 0.3 && hipRight.confidence > 0.3) {
          posX = (hipLeft.x + hipRight.x) / 2
          posY = (hipLeft.y + hipRight.y) / 2
        } else {
          // Fallback: use average of all valid keypoints
          let sumX = 0, sumY = 0, count = 0
          for (const kp of player.keypoints) {
            if (kp && kp.x !== null && kp.y !== null && kp.confidence > 0.3) {
              sumX += kp.x
              sumY += kp.y
              count++
            }
          }
          if (count > 0) {
            posX = sumX / count
            posY = sumY / count
          }
        }
      }
      
      // Add heat at player position
      if (posX !== null && posY !== null) {
        addHeatAtPosition(progressiveHeatmapState, posX, posY)
      }
    }
    
    progressiveHeatmapState.lastProcessedFrame = frame.frame
    framesProcessed++
  }
  
  if (DEBUG_MODE && framesProcessed > 0) {
    console.log('[Progressive Heatmap] Processed', framesProcessed, 'frames up to', currentFrameNum,
      'maxValue:', progressiveHeatmapState.maxValue.toFixed(2))
  }
  
  return progressiveHeatmapState
}

/**
 * Convert progressive heatmap state to HeatmapData format for rendering
 */
function progressiveHeatmapToRenderFormat(state: ProgressiveHeatmapState): HeatmapData {
  // Normalize and convert to 0-255 range
  const combined_heatmap: number[][] = []
  const normalizer = state.maxValue > 0 ? 255 / state.maxValue : 1
  
  for (let y = 0; y < state.height; y++) {
    const row: number[] = []
    const sourceRow = state.heatmap[y]
    for (let x = 0; x < state.width; x++) {
      const value = sourceRow ? (sourceRow[x] || 0) : 0
      row.push(Math.min(255, Math.round(value * normalizer)))
    }
    combined_heatmap.push(row)
  }
  
  return {
    video_id: 'progressive',
    width: state.width,
    height: state.height,
    colormap: 'turbo',
    combined_heatmap,
    total_frames: state.lastProcessedFrame,
    player_position_counts: {},
    video_width: state.videoWidth,
    video_height: state.videoHeight
  }
}

const props = defineProps<{
  videoUrl: string
  skeletonData?: SkeletonFrame[]
  // NOTE: courtDetection prop removed - automatic court detection disabled
  // Manual court keypoints are now the only method for court calibration
  showSkeleton?: boolean
  showBoundingBoxes?: boolean
  showPlayers?: boolean
  showShuttles?: boolean
  showRackets?: boolean
  showPoseOverlay?: boolean
  poseSource?: 'skeleton' | 'trained' | 'both'
  // NOTE: showCourtOverlay prop removed - no automatic court detection to display
  showHeatmap?: boolean
  showShuttleTracking?: boolean
  heatmapFrameRange?: { start: number; end: number } | null
  courtKeypoints?: number[][] | null
  manualCourtKeypoints?: ExtendedCourtKeypoints | null
  viewMode?: 'video' | 'court'
  videoFps?: number
  shotSummarySegment?: ShotMovementSegmentWithPeaks | null
  shotSummaryCountdown?: number
}>()

// Colors for bounding boxes (matching backend colors)
const PLAYER_BOX_COLOR = '#00FF00'      // Green for players
const PLAYER_UNASSIGNED_BOX_COLOR = '#FFD60A'  // Amber when pid is unknown
const SHUTTLE_BOX_COLOR = '#FFA500'     // Orange for shuttlecock (YOLO)
const SHUTTLE_TRACKNET_COLOR = '#00E5FF' // Cyan for TrackNet shuttle detection
const SHUTTLE_YOLO_COLOR = '#FFA500'     // Orange for YOLO shuttle detection
const RACKET_BOX_COLOR = '#FF00FF'      // Magenta for rackets
const OTHER_BOX_COLOR = '#00FFFF'       // Cyan for other detections

// NOTE: COURT_REGION_COLORS removed - automatic court detection disabled

// Canonical ExtendedCourtKeypoints type lives in @/types/analysis.

const emit = defineEmits<{
  timeUpdate: [time: number]
  frameUpdate: [frame: number]
  play: []
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)

const camera = useViewportCamera()
const zoomCaptureRef = ref<HTMLDivElement | null>(null)

// Follow-player state. null = free camera; 0 or 1 = lock onto that player_id
// and re-center every frame. Click a player's skeleton to lock; grab-pan the
// canvas to release.
const followedPlayerId = ref<number | null>(null)

// Drag vs click distinction: a mousedown+mouseup within CLICK_THRESHOLD_PX
// is treated as a click (for follow hit-test); anything beyond that becomes
// a pan drag and exits follow mode.
let isPanning = false
let didPanMove = false
let panStartX = 0
let panStartY = 0
let lastPanX = 0
let lastPanY = 0
const CLICK_THRESHOLD_PX = 5

function onZoomWheel(e: WheelEvent) {
  e.preventDefault()
  const el = zoomCaptureRef.value
  if (!el) return
  // Wheel up (deltaY < 0) = zoom in. One notch ≈ 1.15×.
  const delta = e.deltaY < 0 ? 1.15 : 1 / 1.15
  camera.zoomAt(el, e.clientX, e.clientY, delta)
}

function onZoomMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  isPanning = true
  didPanMove = false
  panStartX = e.clientX
  panStartY = e.clientY
  lastPanX = e.clientX
  lastPanY = e.clientY
  window.addEventListener('mousemove', onZoomMouseMove)
  window.addEventListener('mouseup', onZoomMouseUp)
}

function onZoomMouseMove(e: MouseEvent) {
  if (!isPanning) return
  const el = zoomCaptureRef.value
  if (!el) return

  if (!didPanMove) {
    const totalDx = e.clientX - panStartX
    const totalDy = e.clientY - panStartY
    if (Math.hypot(totalDx, totalDy) > CLICK_THRESHOLD_PX) {
      didPanMove = true
      // Deliberate pan movement — release follow-lock so the user's pan sticks.
      followedPlayerId.value = null
    }
  }

  if (didPanMove) {
    const dx = e.clientX - lastPanX
    const dy = e.clientY - lastPanY
    camera.panBy(el, dx, dy)
  }
  lastPanX = e.clientX
  lastPanY = e.clientY
}

function onZoomMouseUp(e: MouseEvent) {
  const wasClick = isPanning && !didPanMove
  isPanning = false
  window.removeEventListener('mousemove', onZoomMouseMove)
  window.removeEventListener('mouseup', onZoomMouseUp)
  if (wasClick) handleZoomCaptureClick(e)
}

function onZoomDoubleClick() {
  camera.reset()
}

// Click hit-test: find which player the click lands on and lock the
// follow-camera onto them. Prefers the tracker's detection bbox (what the
// user sees when showBoundingBoxes is on), then falls back to the
// keypoint bbox (for when boxes are hidden and the user clicks a bone).
// Clicks on empty area do nothing (to avoid stealing clicks away from
// ongoing follow sessions).
function handleZoomCaptureClick(e: MouseEvent) {
  const el = zoomCaptureRef.value
  const video = videoRef.value
  if (!el || !video) return

  const rect = el.getBoundingClientRect()
  const screenX = e.clientX - rect.left
  const screenY = e.clientY - rect.top
  const world = camera.screenToWorld(screenX, screenY)

  // Convert world (canvas-pixel) coords back to video-pixel coords so we can
  // hit-test against raw detection / keypoint coordinates. The canvas-pixel
  // grid matches the video-pixel grid after resizeCanvas(), so sx/sy are
  // ≈ 1; we still compute them defensively in case resizeCanvas hasn't run.
  const vw = video.videoWidth || video.clientWidth || 1
  const vh = video.videoHeight || video.clientHeight || 1
  const canvasW = canvasRef.value?.width || vw
  const canvasH = canvasRef.value?.height || vh
  const sx = canvasW / vw
  const sy = canvasH / vh
  const videoX = world.x / sx
  const videoY = world.y / sy

  const frame = findInterpolatedFrame(video.currentTime ?? 0)
  if (!frame) return

  // 1. Preferred: detection bbox from the backend tracker. x/y are the
  //    bbox CENTER (see drawBoundingBoxes). A small pad handles slightly-
  //    off clicks near the edge.
  const BBOX_PAD = 8
  for (const det of frame.badminton_detections?.players ?? []) {
    if (det.player_id == null) continue
    const minX = det.x - det.width / 2
    const maxX = det.x + det.width / 2
    const minY = det.y - det.height / 2
    const maxY = det.y + det.height / 2
    if (videoX >= minX - BBOX_PAD && videoX <= maxX + BBOX_PAD &&
        videoY >= minY - BBOX_PAD && videoY <= maxY + BBOX_PAD) {
      followedPlayerId.value = det.player_id
      return
    }
  }

  // 2. Fallback: keypoint bbox. Generous pad because users rarely click a
  //    bone exactly, and the keypoint bbox can be much tighter than the
  //    player's visible silhouette.
  const KP_PAD = 40
  for (const player of frame.players ?? []) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    let any = false
    for (const k of player.keypoints ?? []) {
      if (k == null || k.x == null || k.y == null) continue
      if (k.confidence <= KEYPOINT_CONFIDENCE_THRESHOLD) continue
      any = true
      if (k.x < minX) minX = k.x
      if (k.x > maxX) maxX = k.x
      if (k.y < minY) minY = k.y
      if (k.y > maxY) maxY = k.y
    }
    if (!any) continue
    if (videoX >= minX - KP_PAD && videoX <= maxX + KP_PAD &&
        videoY >= minY - KP_PAD && videoY <= maxY + KP_PAD) {
      followedPlayerId.value = player.player_id
      return
    }
  }
}

let zoomResizeObserver: ResizeObserver | null = null

function attachZoomResizeObserver() {
  const el = zoomCaptureRef.value
  if (!el || zoomResizeObserver) return
  zoomResizeObserver = new ResizeObserver(() => camera.reclamp(el))
  zoomResizeObserver.observe(el)
}

function detachZoomResizeObserver() {
  if (zoomResizeObserver) {
    zoomResizeObserver.disconnect()
    zoomResizeObserver = null
  }
}

watch(zoomCaptureRef, (el) => {
  if (el) {
    attachZoomResizeObserver()
  } else {
    detachZoomResizeObserver()
  }
})

const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const volume = ref(1)
const isMuted = ref(false)
const playbackRate = ref(1)
const isFullscreen = ref(false)
const showControls = ref(true)

// Performance optimization: Pre-built index for O(1) frame lookup
const frameIndex = shallowRef<Map<number, number>>(new Map())
const timestampIndex = shallowRef<number[]>([])

// Smoothing state for Kalman-like filtering
interface SmoothState {
  x: number
  y: number
  vx: number
  vy: number
}
const playerSmoothState = shallowRef<Map<number, SmoothState>>(new Map())
const shuttleSmoothState = ref<SmoothState | null>(null)

let animationFrameId: number | null = null
let videoFrameCallbackId: number | null = null
let controlsTimeout: number | null = null

// PERFORMANCE OPTIMIZATION: Cached canvas context and pre-computed values
let cachedCtx: CanvasRenderingContext2D | null = null
let lastFrameNumber = -1
let lastMediaTime = -1

const formattedCurrentTime = computed(() => formatTime(currentTime.value))
const formattedDuration = computed(() => formatTime(duration.value))
const progressPercent = computed(() =>
  duration.value > 0 ? (currentTime.value / duration.value) * 100 : 0
)

const currentFrame = computed(() => {
  if (!videoRef.value) return 0
  // Derive FPS from skeleton data timestamps if available, else fall back to 30
  const skData = props.skeletonData
  let fps = 30
  if (skData && skData.length >= 2) {
    const totalTs = skData[skData.length - 1]!.timestamp - skData[0]!.timestamp
    if (totalTs > 0) fps = (skData.length - 1) / totalTs
  }
  return Math.floor(currentTime.value * fps)
})

// Build index when skeleton data changes for O(1) lookup
watch(() => props.skeletonData, (newData) => {
  // Reset progressive heatmap when skeleton data changes
  progressiveHeatmapState = null

  if (!newData || newData.length === 0) {
    frameIndex.value = new Map()
    timestampIndex.value = []
    return
  }

  // Build frame number -> index map
  const fIndex = new Map<number, number>()
  const tIndex: number[] = []

  // Detect and correct off-by-one timestamp offset from old backend data.
  // Old backend used `frame_count / fps` where frame_count starts at 1,
  // producing timestamps starting at 1/fps instead of 0.
  // New backend uses actual PTS from the video container (starts at ~0).
  // If the first frame's timestamp is suspiciously close to 1/fps, shift all
  // timestamps back so the first frame aligns with video currentTime = 0.
  const firstTs = newData[0]!.timestamp
  let timeOffset = 0
  if (newData.length >= 2 && firstTs > 0.001) {
    const frameDuration = newData[1]!.timestamp - newData[0]!.timestamp
    if (frameDuration > 0 && Math.abs(firstTs - frameDuration) / frameDuration < 0.2) {
      timeOffset = firstTs
    }
  }

  newData.forEach((frame, idx) => {
    fIndex.set(frame.frame, idx)
    tIndex.push(frame.timestamp - timeOffset)
  })

  frameIndex.value = fIndex
  timestampIndex.value = tIndex
  
  // BUGFIX: When skeleton data arrives after video metadata has loaded,
  // the canvas may not have been sized yet. Wait for Vue to mount the canvas,
  // then resize it and draw the initial overlay.
  nextTick(() => {
    resizeCanvas()
    // Reset cached context since canvas may have been recreated
    cachedCtx = null
    lastFrameNumber = -1
    drawOverlay()
  })
}, { immediate: true })

/**
 * Binary search to find the closest frame to a target timestamp
 * O(log n) instead of O(n) for large skeleton data arrays
 */
function binarySearchTimestamp(targetTime: number): number {
  const timestamps = timestampIndex.value
  if (timestamps.length === 0) return -1
  
  let left = 0
  let right = timestamps.length - 1
  
  while (left < right) {
    const mid = Math.floor((left + right) / 2)
    const midVal = timestamps[mid]
    if (midVal !== undefined && midVal < targetTime) {
      left = mid + 1
    } else {
      right = mid
    }
  }
  
  // Check if previous frame is closer
  if (left > 0) {
    const prevTimestamp = timestamps[left - 1]
    const currTimestamp = timestamps[left]
    if (prevTimestamp !== undefined && currTimestamp !== undefined) {
      const prevDiff = Math.abs(prevTimestamp - targetTime)
      const currDiff = Math.abs(currTimestamp - targetTime)
      if (prevDiff < currDiff) {
        return left - 1
      }
    }
  }
  
  return left
}

/**
 * Find the skeleton frame at or just before the given timestamp.
 * Uses floor semantics to ensure we never show skeleton data from
 * a future video frame, preventing the overlay from leading the video.
 * O(log n) binary search.
 */
function findFrameAtOrBefore(targetTime: number): SkeletonFrame | null {
  if (!props.skeletonData || props.skeletonData.length === 0) return null

  const timestamps = timestampIndex.value
  if (timestamps.length === 0) return null

  // Binary search for the last frame with timestamp <= targetTime
  let left = 0
  let right = timestamps.length - 1
  let result = -1

  while (left <= right) {
    const mid = Math.floor((left + right) / 2)
    const midVal = timestamps[mid]
    if (midVal !== undefined && midVal <= targetTime) {
      result = mid
      left = mid + 1
    } else {
      right = mid - 1
    }
  }

  if (result < 0) {
    // All frames are after targetTime; return the very first frame
    // only if targetTime is extremely close (within half a frame period)
    const firstTimestamp = timestamps[0]
    if (firstTimestamp !== undefined && firstTimestamp - targetTime < 0.02) {
      return props.skeletonData[0] ?? null
    }
    return null
  }

  return props.skeletonData[result] ?? null
}

/**
 * Get skeleton frame for current video time
 * Uses floor-semantic binary search so we never return a future frame.
 */
const currentSkeletonFrame = computed(() => {
  return findFrameAtOrBefore(currentTime.value)
})

/**
 * Find interpolated skeleton frame for smooth overlay rendering.
 * Returns a synthetic SkeletonFrame with positions lerped between the
 * floor frame (at-or-before targetTime) and the ceiling frame (after targetTime).
 *
 * This does NOT cause the skeleton to lead the video because:
 * - The blend factor t is clamped to [0, 1] between two real frames
 * - At t=0, we show the floor frame exactly (same as findFrameAtOrBefore)
 * - At t=1, we show the ceiling frame — but only when targetTime has reached it
 * - We never extrapolate beyond the ceiling frame
 *
 * The interpolation ONLY affects rendering coordinates (keypoints, center,
 * bounding boxes). Analytics reads raw skeleton_data directly and is unaffected.
 */
function findInterpolatedFrame(targetTime: number): SkeletonFrame | null {
  if (!props.skeletonData || props.skeletonData.length === 0) return null

  const timestamps = timestampIndex.value
  if (timestamps.length === 0) return null

  // Binary search for floor index (last frame with timestamp <= targetTime)
  let left = 0
  let right = timestamps.length - 1
  let floorIdx = -1

  while (left <= right) {
    const mid = Math.floor((left + right) / 2)
    const midVal = timestamps[mid]
    if (midVal !== undefined && midVal <= targetTime) {
      floorIdx = mid
      left = mid + 1
    } else {
      right = mid - 1
    }
  }

  if (floorIdx < 0) {
    const firstTimestamp = timestamps[0]
    if (firstTimestamp !== undefined && firstTimestamp - targetTime < 0.02) {
      return props.skeletonData[0] ?? null
    }
    return null
  }

  const floorFrame = props.skeletonData[floorIdx]!
  const ceilIdx = floorIdx + 1

  // If no ceiling frame or we're exactly on a frame, return floor
  if (ceilIdx >= timestamps.length) return floorFrame

  const ceilFrame = props.skeletonData[ceilIdx]!
  const floorTs = timestamps[floorIdx]!
  const ceilTs = timestamps[ceilIdx]!
  const span = ceilTs - floorTs
  if (span <= 0) return floorFrame

  const t = (targetTime - floorTs) / span

  // If very close to floor frame, just return it (avoid unnecessary work)
  if (t < 0.01) return floorFrame
  // If very close to ceiling frame, return ceiling
  if (t > 0.99 && targetTime >= (timestamps[ceilIdx] ?? Infinity)) return ceilFrame

  // Interpolate player positions
  const interpolatedPlayers: FramePlayer[] = floorFrame.players.map(floorPlayer => {
    // Find matching player in ceiling frame by player_id
    const ceilPlayer = ceilFrame.players.find(p => p.player_id === floorPlayer.player_id)
    if (!ceilPlayer) return floorPlayer // No match — use floor as-is

    // Lerp center
    const center = {
      x: floorPlayer.center.x + (ceilPlayer.center.x - floorPlayer.center.x) * t,
      y: floorPlayer.center.y + (ceilPlayer.center.y - floorPlayer.center.y) * t,
    }

    // Lerp keypoints
    const keypoints: Keypoint[] = floorPlayer.keypoints.map((floorKp, i) => {
      const ceilKp = ceilPlayer.keypoints[i]
      if (!ceilKp || floorKp.x === null || floorKp.y === null ||
          ceilKp.x === null || ceilKp.y === null) {
        return floorKp
      }
      return {
        name: floorKp.name,
        x: floorKp.x + (ceilKp.x - floorKp.x) * t,
        y: floorKp.y + (ceilKp.y - floorKp.y) * t,
        confidence: Math.min(floorKp.confidence, ceilKp.confidence),
      }
    })

    return {
      ...floorPlayer,
      center,
      keypoints,
      // Keep speed/pose from floor frame (non-interpolatable)
    }
  })

  // Interpolate shuttle position if both frames have one
  let shuttle_position = floorFrame.shuttle_position
  if (floorFrame.shuttle_position && ceilFrame.shuttle_position) {
    shuttle_position = {
      x: floorFrame.shuttle_position.x + (ceilFrame.shuttle_position.x - floorFrame.shuttle_position.x) * t,
      y: floorFrame.shuttle_position.y + (ceilFrame.shuttle_position.y - floorFrame.shuttle_position.y) * t,
      source: floorFrame.shuttle_position.source,
    }
  }

  // Interpolate bounding box positions
  let badminton_detections = floorFrame.badminton_detections
  if (floorFrame.badminton_detections && ceilFrame.badminton_detections) {
    const lerpDetections = (
      floorDets: BoundingBoxDetection[],
      ceilDets: BoundingBoxDetection[]
    ): BoundingBoxDetection[] => {
      return floorDets.map(fd => {
        // Match by player_id when both frames carry it (player bboxes).
        // Falling back to class would cross-wire Player 1's label onto
        // Player 2's interpolated position when one player is missing
        // in the ceil frame. For non-player detections (shuttle/racket)
        // player_id is absent; match by class.
        const cd = fd.player_id != null
          ? ceilDets.find(d => d.player_id === fd.player_id)
          : ceilDets.find(d => d.class === fd.class)
        if (!cd) return fd
        return {
          ...fd,
          x: fd.x + (cd.x - fd.x) * t,
          y: fd.y + (cd.y - fd.y) * t,
          width: fd.width + (cd.width - fd.width) * t,
          height: fd.height + (cd.height - fd.height) * t,
        }
      })
    }

    badminton_detections = {
      ...floorFrame.badminton_detections,
      players: lerpDetections(floorFrame.badminton_detections.players, ceilFrame.badminton_detections.players),
      shuttlecocks: lerpDetections(floorFrame.badminton_detections.shuttlecocks, ceilFrame.badminton_detections.shuttlecocks),
      rackets: lerpDetections(floorFrame.badminton_detections.rackets, ceilFrame.badminton_detections.rackets),
    }
  }

  return {
    ...floorFrame,
    players: interpolatedPlayers,
    shuttle_position,
    badminton_detections,
  }
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function togglePlay() {
  if (!videoRef.value || isExporting.value) return

  if (isPlaying.value) {
    videoRef.value.pause()
  } else {
    videoRef.value.play()
  }
}

function handlePlay() {
  isPlaying.value = true
  startSkeletonAnimation()
  emit('play')
}

function handlePause() {
  isPlaying.value = false
  stopSkeletonAnimation()
}

// PERFORMANCE OPTIMIZATION: Throttle time update emits to reduce overhead
let lastTimeUpdateEmit = 0
const TIME_UPDATE_THROTTLE_MS = 50 // Emit at most every 50ms (20fps)

function handleTimeUpdate() {
  if (!videoRef.value) return
  currentTime.value = videoRef.value.currentTime
  
  // Throttle emit to parent - prevents excessive React re-renders
  const now = performance.now()
  if (now - lastTimeUpdateEmit >= TIME_UPDATE_THROTTLE_MS) {
    lastTimeUpdateEmit = now
    emit('timeUpdate', currentTime.value)
    emit('frameUpdate', currentFrame.value)
  }
}

function handleLoadedMetadata() {
  if (!videoRef.value) return
  duration.value = videoRef.value.duration
  resizeCanvas()
}

function handleSeek(event: MouseEvent) {
  if (isExporting.value) return
  const target = event.currentTarget as HTMLElement
  const rect = target.getBoundingClientRect()
  const percent = (event.clientX - rect.left) / rect.width
  if (videoRef.value) {
    videoRef.value.currentTime = percent * duration.value
  }
}

function handleVolumeChange(event: Event) {
  const input = event.target as HTMLInputElement
  volume.value = parseFloat(input.value)
  if (videoRef.value) {
    videoRef.value.volume = volume.value
    isMuted.value = volume.value === 0
  }
}

function toggleMute() {
  if (!videoRef.value) return
  isMuted.value = !isMuted.value
  videoRef.value.muted = isMuted.value
}

function setPlaybackRate(rate: number) {
  playbackRate.value = rate
  if (videoRef.value) {
    videoRef.value.playbackRate = rate
  }
}

function toggleFullscreen() {
  if (!containerRef.value) return

  if (!isFullscreen.value) {
    if (containerRef.value.requestFullscreen) {
      containerRef.value.requestFullscreen()
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen()
    }
  }
}

function handleFullscreenChange() {
  isFullscreen.value = !!document.fullscreenElement
}

function skipTime(seconds: number) {
  if (!videoRef.value || isExporting.value) return
  videoRef.value.currentTime = Math.max(0, Math.min(duration.value, videoRef.value.currentTime + seconds))
}

// Step the video by an integer number of frames. Pauses playback first so
// the seek isn't immediately overwritten by the next playing frame.
function stepFrames(frames: number) {
  const v = videoRef.value
  if (!v || isExporting.value) return
  if (!v.paused) v.pause()
  const fps = props.videoFps && props.videoFps > 0 ? props.videoFps : 30
  v.currentTime = Math.max(0, Math.min(duration.value, v.currentTime + frames / fps))
}

function showControlsTemporarily() {
  showControls.value = true
  if (controlsTimeout) {
    clearTimeout(controlsTimeout)
  }
  if (isPlaying.value) {
    controlsTimeout = window.setTimeout(() => {
      showControls.value = false
    }, 3000)
  }
}

// =============================================================================
// SHUTTLE TRACKING TRAIL: Draw TrackNet shuttle trajectory trail on video overlay
// =============================================================================
const SHUTTLE_TRAIL_LENGTH = 20

function drawShuttleTrail(
  ctx: CanvasRenderingContext2D,
  currentFrameData: SkeletonFrame,
  scaleX: number,
  scaleY: number
) {
  if (!props.skeletonData || props.skeletonData.length === 0) return
  if (!currentFrameData.shuttle_position) return

  const currentFrameNum = currentFrameData.frame
  const trailPositions: { x: number; y: number; source?: 'tracknet' | 'yolo' }[] = []

  const currentIdx = frameIndex.value.get(currentFrameNum)
  if (currentIdx === undefined) return

  for (let i = currentIdx; i >= Math.max(0, currentIdx - SHUTTLE_TRAIL_LENGTH); i--) {
    const frame = props.skeletonData[i]
    if (frame?.shuttle_position) {
      trailPositions.unshift(frame.shuttle_position)
    }
  }

  if (trailPositions.length === 0) return

  const totalPoints = trailPositions.length
  const current = trailPositions[totalPoints - 1]!
  const trailColor = current.source === 'tracknet' ? SHUTTLE_TRACKNET_COLOR : SHUTTLE_YOLO_COLOR

  if (totalPoints >= 2) {
    // White outline pass for contrast against any background
    for (let i = 1; i < totalPoints; i++) {
      const prev = trailPositions[i - 1]!
      const curr = trailPositions[i]!
      const progress = i / (totalPoints - 1)

      const x1 = prev.x * scaleX
      const y1 = prev.y * scaleY
      const x2 = curr.x * scaleX
      const y2 = curr.y * scaleY

      const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
      if (dist > 300) continue

      const alpha = 0.1 + progress * 0.5
      const lineWidth = 4 + progress * 5

      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.strokeStyle = 'rgba(0,0,0,' + alpha.toFixed(2) + ')'
      ctx.lineWidth = camera.pixelSize(lineWidth + 3)
      ctx.lineCap = 'round'
      ctx.stroke()
    }

    // Colored trail pass on top
    for (let i = 1; i < totalPoints; i++) {
      const prev = trailPositions[i - 1]!
      const curr = trailPositions[i]!
      const progress = i / (totalPoints - 1)

      const x1 = prev.x * scaleX
      const y1 = prev.y * scaleY
      const x2 = curr.x * scaleX
      const y2 = curr.y * scaleY

      const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
      if (dist > 300) continue

      const alpha = 0.25 + progress * 0.75
      const lineWidth = 4 + progress * 5

      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.strokeStyle = trailColor + Math.round(alpha * 255).toString(16).padStart(2, '0')
      ctx.lineWidth = camera.pixelSize(lineWidth)
      ctx.lineCap = 'round'
      ctx.stroke()
    }
  }

  // Current position — large glowing dot
  const cx = current.x * scaleX
  const cy = current.y * scaleY
  const glowRadius = camera.pixelSize(20)

  // Soft outer glow
  const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowRadius)
  gradient.addColorStop(0, trailColor + '66')
  gradient.addColorStop(0.5, trailColor + '22')
  gradient.addColorStop(1, trailColor + '00')
  ctx.beginPath()
  ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2)
  ctx.fillStyle = gradient
  ctx.fill()

  // Dark outline ring for contrast
  ctx.beginPath()
  ctx.arc(cx, cy, camera.pixelSize(10), 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(0,0,0,0.6)'
  ctx.lineWidth = camera.pixelSize(4)
  ctx.stroke()

  // Main dot
  ctx.beginPath()
  ctx.arc(cx, cy, camera.pixelSize(9), 0, Math.PI * 2)
  ctx.fillStyle = trailColor
  ctx.fill()
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = camera.pixelSize(2.5)
  ctx.stroke()

  // Inner bright center
  ctx.beginPath()
  ctx.arc(cx, cy, camera.pixelSize(4), 0, Math.PI * 2)
  ctx.fillStyle = '#FFFFFF'
  ctx.fill()
}

// Canvas skeleton drawing
function resizeCanvas() {
  if (!canvasRef.value || !videoRef.value) return

  canvasRef.value.width = videoRef.value.videoWidth || videoRef.value.clientWidth
  canvasRef.value.height = videoRef.value.videoHeight || videoRef.value.clientHeight
}

function drawOverlay(exactFrame?: SkeletonFrame | null) {
  if (!canvasRef.value) {
    if (DEBUG_MODE) console.log('[Overlay Debug] No canvas ref')
    return
  }
  if (!props.showSkeleton && !props.showBoundingBoxes && !props.showHeatmap && !props.showShuttleTracking) {
    if (DEBUG_MODE) console.log('[Overlay Debug] All overlays are disabled')
    return
  }

  // Use the exact frame when provided (during playback animation loop via
  // requestVideoFrameCallback or rAF with floor-semantic lookup).
  // Fall back to computed frame (when called from watchers while paused).
  const frame = exactFrame !== undefined ? exactFrame : currentSkeletonFrame.value
  
  // Allow heatmap rendering even without skeleton frame
  const hasHeatmapToRender = props.showHeatmap && props.skeletonData && props.skeletonData.length > 0
  const hasFrameToRender = frame && (props.showSkeleton || props.showBoundingBoxes || props.showShuttleTracking)
  
  if (!frame && !hasHeatmapToRender) {
    if (DEBUG_MODE) console.log('[Overlay Debug] No current skeleton frame and no heatmap')
    return
  }
  
  // BUGFIX: Ensure canvas has valid dimensions before drawing
  // If canvas dimensions are 0, try to resize first
  if (canvasRef.value.width === 0 || canvasRef.value.height === 0) {
    if (DEBUG_MODE) console.log('[Overlay Debug] Canvas has 0 dimensions, resizing...')
    resizeCanvas()
    // If still no valid dimensions, we can't draw
    if (!canvasRef.value || canvasRef.value.width === 0 || canvasRef.value.height === 0) {
      if (DEBUG_MODE) console.log('[Overlay Debug] Canvas still has 0 dimensions after resize')
      return
    }
    // Reset cached context after resize
    cachedCtx = null
  }

  // PERFORMANCE OPTIMIZATION: Skip rendering if nothing has changed.
  // During playback, interpolated frames have continuously changing positions
  // so we always redraw. When paused (no exactFrame), skip if same frame number.
  if (frame && !hasHeatmapToRender) {
    if (exactFrame === undefined) {
        if (frame.frame === lastFrameNumber) return
    } else {
        const mediaTime = videoRef.value?.currentTime ?? -1
        if (mediaTime === lastMediaTime && frame.frame === lastFrameNumber) return
        lastMediaTime = mediaTime
    }
  }
  if (frame) {
    lastFrameNumber = frame.frame
  }
  
  // DEBUG: Log frame info (only in debug mode)
  if (DEBUG_MODE) {
    console.log('[Overlay Debug] Drawing frame:', frame?.frame ?? 'N/A',
      'showSkeleton:', props.showSkeleton,
      'showBoundingBoxes:', props.showBoundingBoxes,
      'showHeatmap:', props.showHeatmap,
      'hasHeatmapData:', hasHeatmapToRender,
      'players:', frame?.players?.length ?? 0,
      'badminton_detections:', !!frame?.badminton_detections,
      'canvas:', canvasRef.value.width, 'x', canvasRef.value.height)
  }

  // PERFORMANCE OPTIMIZATION: Cache and reuse canvas context
  // NOTE: desynchronized is intentionally NOT used here. While it reduces latency,
  // it also causes the canvas to composite independently of the page, which can
  // desynchronize the overlay from the video element. For frame-accurate overlay
  // alignment, we need synchronized compositing.
  if (!cachedCtx) {
    cachedCtx = canvasRef.value.getContext('2d', {
      alpha: true,
    })
  }
  const ctx = cachedCtx
  if (!ctx) return

  // Scale factors for canvas (pre-compute once)
  // BUGFIX: Use video dimensions for accurate scaling, fallback to 1 to prevent NaN
  const videoWidth = videoRef.value?.videoWidth || videoRef.value?.clientWidth || 1
  const videoHeight = videoRef.value?.videoHeight || videoRef.value?.clientHeight || 1
  const scaleX = canvasRef.value.width / videoWidth
  const scaleY = canvasRef.value.height / videoHeight

  // Follow-player: update camera tx/ty so the locked player is centered
  // BEFORE applying the transform below. Needs scaleX/Y to map the player's
  // video-pixel center into canvas-pixel (world) space.
  if (followedPlayerId.value !== null && frame && zoomCaptureRef.value) {
    const target = frame.players.find(p => p.player_id === followedPlayerId.value)
    if (target?.center) {
      camera.centerAt(
        zoomCaptureRef.value,
        target.center.x * scaleX,
        target.center.y * scaleY,
      )
    }
  }

  // Clear canvas under identity transform so we wipe the entire canvas
  // regardless of the current pan/zoom state; otherwise clearRect only
  // clears the transformed sub-region and leaves ghost frames.
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)
  camera.applyToContext(ctx)

  // NOTE: Court overlay drawing removed - automatic court detection disabled
  // Manual court keypoints are now the only method for court calibration

  // Draw heatmap overlay if enabled (behind skeleton and bounding boxes)
  // PROGRESSIVE HEATMAP: Compute heatmap dynamically based on skeleton data up to current frame
  if (props.showHeatmap && props.skeletonData && props.skeletonData.length > 0 && frame) {
    const progressiveState = computeProgressiveHeatmap(
      props.skeletonData,
      frame.frame,
      videoWidth,
      videoHeight,
      props.heatmapFrameRange
    )

    if (progressiveState && progressiveState.maxValue > 0) {
      const heatmapToDraw = progressiveHeatmapToRenderFormat(progressiveState)
      drawHeatmap(ctx, heatmapToDraw, canvasRef.value.width, canvasRef.value.height, 0.6)
    }
  }

  // Draw bounding boxes if enabled (requires frame data)
  if (props.showBoundingBoxes && frame?.badminton_detections) {
    drawBoundingBoxes(ctx, frame.badminton_detections, scaleX, scaleY)
  }

  // Draw shuttle tracking trail + current position dot.
  // Skipped in court view — SyntheticCourtView draws its own (cleaner) trail
  // designed for the synthetic-court background, and we don't want both
  // trails overlapping.
  if (props.showShuttleTracking && frame && props.viewMode !== 'court') {
    drawShuttleTrail(ctx, frame, scaleX, scaleY)
  }

  // Draw skeleton if enabled (requires frame data)
  if (props.showSkeleton && frame) {
    drawSkeleton(ctx, frame, scaleX, scaleY)
  }
}

// NOTE: drawCourtOverlay function removed - automatic court detection disabled
// Manual court keypoints are now the only method for court calibration

/**
 * Draw heatmap overlay showing player position density
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Pre-rendered heatmap data from backend (no client-side computation)
 * - Uses ImageData for efficient pixel manipulation
 * - Turbo colormap applied directly from backend-provided values
 */
function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  heatmap: HeatmapData,
  canvasWidth: number,
  canvasHeight: number,
  alpha: number = 0.5
) {
  if (!heatmap.combined_heatmap || heatmap.combined_heatmap.length === 0) {
    if (DEBUG_MODE) console.log('[Heatmap Debug] No heatmap data to render')
    return
  }
  
  const heatmapHeight = heatmap.combined_heatmap.length
  const heatmapWidth = heatmap.combined_heatmap[0]?.length || 0
  
  if (heatmapWidth === 0) {
    if (DEBUG_MODE) console.log('[Heatmap Debug] Invalid heatmap dimensions')
    return
  }
  
  // Create an offscreen canvas for the heatmap
  const offscreen = document.createElement('canvas')
  offscreen.width = heatmapWidth
  offscreen.height = heatmapHeight
  const offCtx = offscreen.getContext('2d')
  if (!offCtx) return
  
  // Create ImageData for efficient pixel manipulation
  const imageData = offCtx.createImageData(heatmapWidth, heatmapHeight)
  const data = imageData.data
  
  // Turbo colormap LUT (256 entries) - matches backend cv2.COLORMAP_TURBO
  // This is a pre-computed lookup table for fast color mapping
  const turboColormap = getTurboColormap()
  
  // Fill the ImageData with heatmap colors
  for (let y = 0; y < heatmapHeight; y++) {
    const row = heatmap.combined_heatmap[y]
    if (!row) continue
    
    for (let x = 0; x < heatmapWidth; x++) {
      const intensity = row[x] || 0
      const idx = (y * heatmapWidth + x) * 4
      
      if (intensity > 0) {
        // Map intensity (0-255) to turbo colormap
        const colorIdx = Math.min(255, Math.max(0, Math.round(intensity)))
        const color = turboColormap[colorIdx]
        if (color) {
          data[idx] = color[0]     // R
          data[idx + 1] = color[1] // G
          data[idx + 2] = color[2] // B
          data[idx + 3] = Math.round(alpha * 255 * (intensity / 255)) // A - fade with intensity
        }
      } else {
        // Transparent for zero intensity
        data[idx + 3] = 0
      }
    }
  }
  
  // Put the image data to offscreen canvas
  offCtx.putImageData(imageData, 0, 0)
  
  // Draw the heatmap scaled to canvas size
  ctx.drawImage(offscreen, 0, 0, canvasWidth, canvasHeight)
  
  if (DEBUG_MODE) {
    console.log('[Heatmap Debug] Rendered heatmap', heatmapWidth, 'x', heatmapHeight, 'to', canvasWidth, 'x', canvasHeight)
  }
}

/**
 * Turbo colormap lookup table (256 RGB values)
 * Matches OpenCV's cv2.COLORMAP_TURBO for consistency with backend
 *
 * PERFORMANCE: Pre-computed LUT means O(1) color lookup per pixel
 */
function getTurboColormap(): [number, number, number][] {
  // Pre-computed turbo colormap values (Google AI turbo colormap)
  // Format: [R, G, B] for indices 0-255
  return [
    [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81], [53, 30, 88], [54, 33, 95],
    [55, 36, 102], [56, 39, 109], [57, 42, 115], [58, 45, 121], [59, 47, 128], [60, 50, 134],
    [61, 53, 139], [62, 56, 145], [63, 59, 151], [63, 62, 156], [64, 64, 162], [65, 67, 167],
    [65, 70, 172], [66, 73, 177], [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195],
    [68, 86, 199], [69, 89, 203], [69, 92, 207], [69, 94, 211], [70, 97, 214], [70, 100, 218],
    [70, 102, 221], [70, 105, 224], [70, 107, 227], [71, 110, 230], [71, 113, 233], [71, 115, 235],
    [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244], [70, 128, 246], [70, 130, 248],
    [70, 133, 250], [70, 135, 251], [69, 138, 252], [69, 140, 253], [68, 143, 254], [67, 145, 254],
    [66, 148, 255], [65, 150, 255], [64, 153, 255], [62, 155, 254], [61, 158, 254], [59, 160, 253],
    [58, 163, 252], [56, 165, 251], [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245],
    [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238], [40, 188, 235], [39, 190, 233],
    [37, 192, 231], [35, 195, 228], [34, 197, 226], [32, 199, 223], [31, 201, 221], [30, 203, 218],
    [28, 205, 216], [27, 208, 213], [26, 210, 210], [26, 212, 208], [25, 213, 205], [24, 215, 202],
    [24, 217, 200], [24, 219, 197], [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187],
    [25, 227, 185], [26, 228, 182], [28, 230, 180], [29, 231, 178], [31, 233, 175], [32, 234, 172],
    [34, 235, 170], [37, 236, 167], [39, 238, 164], [42, 239, 161], [44, 240, 158], [47, 241, 155],
    [50, 242, 152], [53, 243, 148], [56, 244, 145], [60, 245, 142], [63, 246, 138], [67, 247, 135],
    [70, 248, 132], [74, 248, 128], [78, 249, 125], [82, 250, 122], [85, 250, 118], [89, 251, 115],
    [93, 252, 111], [97, 252, 108], [101, 253, 105], [105, 253, 102], [109, 254, 98], [113, 254, 95],
    [117, 254, 92], [121, 254, 89], [125, 255, 86], [128, 255, 83], [132, 255, 81], [136, 255, 78],
    [139, 255, 75], [143, 255, 73], [146, 255, 71], [150, 254, 68], [153, 254, 66], [156, 254, 64],
    [159, 253, 63], [161, 253, 61], [164, 252, 60], [167, 252, 58], [169, 251, 57], [172, 251, 56],
    [175, 250, 55], [177, 249, 54], [180, 248, 54], [183, 247, 53], [185, 246, 53], [188, 245, 52],
    [190, 244, 52], [193, 243, 52], [195, 241, 52], [198, 240, 52], [200, 239, 52], [203, 237, 52],
    [205, 236, 52], [208, 234, 52], [210, 233, 53], [212, 231, 53], [215, 229, 53], [217, 228, 54],
    [219, 226, 54], [221, 224, 55], [223, 223, 55], [225, 221, 55], [227, 219, 56], [229, 217, 56],
    [231, 215, 57], [233, 213, 57], [235, 211, 57], [236, 209, 58], [238, 207, 58], [239, 205, 58],
    [241, 203, 58], [242, 201, 58], [244, 199, 58], [245, 197, 58], [246, 195, 58], [247, 193, 58],
    [248, 190, 57], [249, 188, 57], [250, 186, 57], [251, 184, 56], [251, 182, 55], [252, 179, 54],
    [252, 177, 54], [253, 174, 53], [253, 172, 52], [254, 169, 51], [254, 167, 50], [254, 164, 49],
    [254, 161, 48], [254, 158, 47], [254, 155, 45], [254, 153, 44], [254, 150, 43], [254, 147, 42],
    [254, 144, 41], [253, 141, 39], [253, 138, 38], [252, 135, 37], [252, 132, 35], [251, 129, 34],
    [251, 126, 33], [250, 123, 31], [249, 120, 30], [249, 117, 29], [248, 114, 28], [247, 111, 26],
    [246, 108, 25], [245, 105, 24], [244, 102, 23], [243, 99, 21], [242, 96, 20], [241, 93, 19],
    [240, 91, 18], [239, 88, 17], [237, 85, 16], [236, 83, 15], [235, 80, 14], [234, 78, 13],
    [232, 75, 12], [231, 73, 12], [229, 71, 11], [228, 69, 10], [226, 67, 10], [225, 65, 9],
    [223, 63, 8], [221, 61, 8], [220, 59, 7], [218, 57, 7], [216, 55, 6], [214, 53, 6],
    [212, 51, 5], [210, 49, 5], [208, 47, 5], [206, 45, 4], [204, 43, 4], [202, 42, 4],
    [200, 40, 3], [197, 38, 3], [195, 37, 3], [193, 35, 2], [190, 33, 2], [188, 32, 2],
    [185, 30, 2], [183, 29, 2], [180, 27, 1], [178, 26, 1], [175, 24, 1], [172, 23, 1],
    [169, 22, 1], [167, 20, 1], [164, 19, 1], [161, 18, 1], [158, 16, 1], [155, 15, 1],
    [152, 14, 1], [149, 13, 1], [146, 11, 1], [142, 10, 1], [139, 9, 2], [136, 8, 2],
    [133, 7, 2], [129, 6, 2], [126, 5, 2], [122, 4, 3]
  ]
}

function drawBoundingBoxes(
  ctx: CanvasRenderingContext2D,
  detections: BadmintonDetections,
  scaleX: number,
  scaleY: number,
) {
  // Helper function to draw a single bounding box
  function drawBox(det: BoundingBoxDetection, color: string, label: string) {
    // Convert center-based coordinates to corner coordinates
    const x = (det.x - det.width / 2) * scaleX
    const y = (det.y - det.height / 2) * scaleY
    const width = det.width * scaleX
    const height = det.height * scaleY

    // Draw box
    ctx.strokeStyle = color
    ctx.lineWidth = camera.pixelSize(2)
    ctx.strokeRect(x, y, width, height)

    // Draw label background
    ctx.font = `bold ${camera.pixelSize(12)}px Inter, system-ui, sans-serif`
    const labelText = `${label}: ${(det.confidence * 100).toFixed(0)}%`
    const textMetrics = ctx.measureText(labelText)
    const textHeight = camera.pixelSize(16)
    const padding = camera.pixelSize(4)

    ctx.fillStyle = color
    ctx.fillRect(x, y - textHeight - padding, textMetrics.width + padding * 2, textHeight + padding)

    // Draw label text
    ctx.fillStyle = '#000000'
    ctx.fillText(labelText, x + padding, y - padding - camera.pixelSize(2))
  }

  // Draw players - only if showPlayers is true. When player_id is
  // missing, the tracker was not confident enough to assign an identity
  // this frame; render the bbox in a distinct color with a generic
  // label instead of guessing.
  if (props.showPlayers !== false) {
    detections.players?.forEach((player) => {
      if (player.player_id == null) {
        drawBox(player, PLAYER_UNASSIGNED_BOX_COLOR, 'Player')
      } else {
        drawBox(player, PLAYER_BOX_COLOR, pidLabelFor(player.player_id))
      }
    })
  }

  // Draw shuttlecocks (orange) - only if showShuttles is true
  // Note: center dot is now drawn separately via shuttle_position in drawOverlay
  if (props.showShuttles !== false) {
    detections.shuttlecocks?.forEach(shuttle => {
      drawBox(shuttle, SHUTTLE_BOX_COLOR, 'Shuttle')
    })
  }

  // Draw rackets (magenta) - only if showRackets is true
  if (props.showRackets !== false) {
    detections.rackets?.forEach(racket => {
      drawBox(racket, RACKET_BOX_COLOR, 'Racket')
    })
  }

  // Draw other detections (cyan) - always show other
  detections.other?.forEach(other => {
    drawBox(other, OTHER_BOX_COLOR, other.class || 'Other')
  })
}

// Minimum confidence for drawing keypoints (lower = draw more, higher = draw fewer)
const KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

// Body angle overlay configuration
type AngleOverlay =
  | 'left_elbow' | 'right_elbow'
  | 'left_shoulder' | 'right_shoulder'
  | 'left_knee' | 'right_knee'
  | 'left_hip' | 'right_hip'
  | 'torso_lean'
  | 'leg_stretch'

const ANGLE_OVERLAY_LABELS: Record<AngleOverlay, string> = {
  left_elbow: 'L Elbow',
  right_elbow: 'R Elbow',
  left_shoulder: 'L Shoulder',
  right_shoulder: 'R Shoulder',
  left_knee: 'L Knee',
  right_knee: 'R Knee',
  left_hip: 'L Hip',
  right_hip: 'R Hip',
  torso_lean: 'Torso Lean',
  leg_stretch: 'Leg Stretch',
}

const ALL_ANGLE_OVERLAYS = Object.keys(ANGLE_OVERLAY_LABELS) as AngleOverlay[]
const enabledOverlays = ref(new Set<AngleOverlay>())
const showAngleMenu = ref(false)

function toggleOverlay(overlay: AngleOverlay) {
  const s = new Set(enabledOverlays.value)
  if (s.has(overlay)) s.delete(overlay)
  else s.add(overlay)
  enabledOverlays.value = s
}

const homographyMatrix = computed(() => {
  const kp = props.courtKeypoints
  if (!kp || kp.length < 4) return null
  return computeHomographyFromKeypoints(kp)
})

/** COCO keypoint index map for angle joint lookups */
const KP = {
  left_shoulder: 5, right_shoulder: 6,
  left_elbow: 7, right_elbow: 8,
  left_wrist: 9, right_wrist: 10,
  left_hip: 11, right_hip: 12,
  left_knee: 13, right_knee: 14,
  left_ankle: 15, right_ankle: 16,
} as const

/**
 * Map each angle overlay to the 3 keypoint indices forming the angle.
 * Angle is measured at the MIDDLE keypoint (index 1).
 */
const ANGLE_JOINTS: Record<string, [number, number, number]> = {
  left_elbow:     [KP.left_shoulder,  KP.left_elbow,     KP.left_wrist],
  right_elbow:    [KP.right_shoulder, KP.right_elbow,    KP.right_wrist],
  left_shoulder:  [KP.left_elbow,     KP.left_shoulder,  KP.left_hip],
  right_shoulder: [KP.right_elbow,    KP.right_shoulder, KP.right_hip],
  left_knee:      [KP.left_hip,       KP.left_knee,      KP.left_ankle],
  right_knee:     [KP.right_hip,      KP.right_knee,     KP.right_ankle],
  left_hip:       [KP.left_shoulder,  KP.left_hip,       KP.left_knee],
  right_hip:      [KP.right_shoulder, KP.right_hip,      KP.right_knee],
}

function drawAngleArc(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  jointIndices: [number, number, number],
  angleDegrees: number,
  scaleX: number,
  scaleY: number,
  color: string,
) {
  const [aIdx, vIdx, bIdx] = jointIndices
  const a = keypoints[aIdx], v = keypoints[vIdx], b = keypoints[bIdx]
  if (!a?.x || !a?.y || !v?.x || !v?.y || !b?.x || !b?.y) return
  if (a.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      v.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      b.confidence < KEYPOINT_CONFIDENCE_THRESHOLD) return

  const vx = v.x * scaleX, vy = v.y * scaleY
  const ax = a.x * scaleX, ay = a.y * scaleY
  const bx = b.x * scaleX, by = b.y * scaleY

  const angle1 = Math.atan2(ay - vy, ax - vx)
  const angle2 = Math.atan2(by - vy, bx - vx)

  // Draw arc — radius and label scale with zoom so the angle decoration
  // grows with the joint. Only the arc stroke stays thin (pixelSize).
  const radius = 20
  ctx.beginPath()
  ctx.arc(vx, vy, radius, Math.min(angle1, angle2), Math.max(angle1, angle2))
  ctx.strokeStyle = color
  ctx.lineWidth = camera.pixelSize(2)
  ctx.globalAlpha = 0.8
  ctx.stroke()
  ctx.globalAlpha = 1.0

  // Draw label — font and offsets scale with zoom so the number is
  // readable at high magnification (key reason to zoom in).
  const midAngle = (angle1 + angle2) / 2
  const labelX = vx + Math.cos(midAngle) * (radius + 14)
  const labelY = vy + Math.sin(midAngle) * (radius + 14)

  ctx.font = 'bold 11px Inter, system-ui, sans-serif'
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2.5
  ctx.strokeText(`${Math.round(angleDegrees)}°`, labelX - 10, labelY + 4)
  ctx.fillText(`${Math.round(angleDegrees)}°`, labelX - 10, labelY + 4)
}

function drawLegStretch(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  scaleX: number,
  scaleY: number,
  color: string,
  H: number[][] | null,
) {
  const distMeters = legStretchMeters(keypoints, H)
  if (distMeters == null) return

  const la = keypoints[KP.left_ankle], ra = keypoints[KP.right_ankle]
  if (!la?.x || !la?.y || !ra?.x || !ra?.y) return

  const lax = la.x * scaleX, lay = la.y * scaleY
  const rax = ra.x * scaleX, ray = ra.y * scaleY

  // Dashed line stays a uniform thin indicator (pixelSize on the stroke
  // and dash pattern). The meter label scales with zoom.
  ctx.beginPath()
  ctx.setLineDash([camera.pixelSize(6), camera.pixelSize(4)])
  ctx.moveTo(lax, lay)
  ctx.lineTo(rax, ray)
  ctx.strokeStyle = color
  ctx.lineWidth = camera.pixelSize(2)
  ctx.globalAlpha = 0.7
  ctx.stroke()
  ctx.setLineDash([])
  ctx.globalAlpha = 1.0

  const mx = (lax + rax) / 2
  const my = (lay + ray) / 2
  const label = `${distMeters.toFixed(2)}m`

  ctx.font = 'bold 12px Inter, system-ui, sans-serif'
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 2.5
  ctx.strokeText(label, mx - 15, my - 8)
  ctx.fillText(label, mx - 15, my - 8)
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  frame: SkeletonFrame,
  scaleX: number,
  scaleY: number
) {
  // DEBUG: Log skeleton data on first few draws
  if (frame.frame <= 5 || frame.frame % 100 === 0) {
    console.log('[Skeleton Debug] Frame:', frame.frame,
      'Players:', frame.players.length,
      'First player keypoints:', frame.players[0]?.keypoints?.length ?? 0,
      'Sample keypoint:', frame.players[0]?.keypoints?.[0])
  }
  
  // If no players, nothing to draw
  if (!frame.players || frame.players.length === 0) {
    return
  }
  
  // Sort players by player_id for consistent rendering order
  // This ensures Player 0 always gets the first color, Player 1 the second,
  // regardless of what order the backend returns them in the array
  const sortedPlayers = [...frame.players].sort((a, b) => a.player_id - b.player_id)
  
  // Draw each player's skeleton
  sortedPlayers.forEach((player) => {
    // Use player_id (not array index) for color assignment
    // Player 0 (far/top) always gets color[0], Player 1 (near/bottom) always gets color[1]
    const displayPid = pidDisplayFor(player.player_id)
    const color = PLAYER_COLORS[displayPid % PLAYER_COLORS.length] ?? '#FF6B6B'
    const keypoints = player.keypoints
    
    // Skip if no keypoints
    if (!keypoints || keypoints.length === 0) {
      console.warn('[Skeleton Debug] Player', player.player_id, 'has no keypoints')
      return
    }

    // Draw skeleton connections
    ctx.strokeStyle = color
    ctx.lineWidth = camera.pixelSize(5)
    ctx.lineCap = 'round'

    let connectionsDrawn = 0
    for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
      const startKp = keypoints[startIdx]
      const endKp = keypoints[endIdx]

      // Check if both keypoints exist and have valid coordinates
      if (startKp && endKp &&
          startKp.x !== null && startKp.y !== null &&
          endKp.x !== null && endKp.y !== null &&
          startKp.confidence > KEYPOINT_CONFIDENCE_THRESHOLD &&
          endKp.confidence > KEYPOINT_CONFIDENCE_THRESHOLD) {
        ctx.beginPath()
        ctx.moveTo(startKp.x * scaleX, startKp.y * scaleY)
        ctx.lineTo(endKp.x * scaleX, endKp.y * scaleY)
        ctx.stroke()
        connectionsDrawn++
      }
    }

    // Draw keypoints
    let keypointsDrawn = 0
    for (const kp of keypoints) {
      if (kp && kp.x !== null && kp.y !== null && kp.confidence > KEYPOINT_CONFIDENCE_THRESHOLD) {
        ctx.beginPath()
        ctx.arc(kp.x * scaleX, kp.y * scaleY, camera.pixelSize(6), 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = camera.pixelSize(2.5)
        ctx.stroke()
        keypointsDrawn++
      }
    }
    
    // DEBUG: Log drawing stats
    if (frame.frame <= 5 || frame.frame % 100 === 0) {
      console.log('[Skeleton Debug] Player', player.player_id,
        'drew', keypointsDrawn, 'keypoints,', connectionsDrawn, 'connections')
    }

    // Draw player label and speed (always draw if center exists).
    // Text grows with zoom so it's readable when inspecting posture close-up.
    if (player.center) {
      ctx.font = 'bold 14px Inter, system-ui, sans-serif'
      ctx.fillStyle = color
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 3

      // player_id is 0-indexed, display as 1-indexed (Player 1, Player 2)
      const labelName = pidLabelFor(player.player_id)
      // Shorten "Player 1" → "P1" for the compact on-skeleton label; keep custom
      // names as-is if the user has set one.
      const labelPrefix = labelName.startsWith('Player ') ? `P${labelName.slice(7)}` : labelName
      const label = `${labelPrefix}: ${player.current_speed?.toFixed(1) ?? 0} km/h`
      const x = player.center.x * scaleX
      const y = player.center.y * scaleY - 30

      ctx.strokeText(label, x - 30, y)
      ctx.fillText(label, x - 30, y)
    }

    // Draw enabled angle overlays
    const angles = player.pose?.body_angles
    if (angles && enabledOverlays.value.size > 0) {
      for (const [key, joints] of Object.entries(ANGLE_JOINTS)) {
        if (!enabledOverlays.value.has(key as AngleOverlay)) continue
        const value = angles[key as keyof typeof angles]
        if (value != null) {
          drawAngleArc(ctx, keypoints, joints, value, scaleX, scaleY, color)
        }
      }

      // Torso lean — label near shoulder midpoint. Scales with zoom.
      if (enabledOverlays.value.has('torso_lean') && angles.torso_lean != null) {
        const ls = keypoints[KP.left_shoulder], rs = keypoints[KP.right_shoulder]
        if (ls?.x && ls?.y && rs?.x && rs?.y) {
          const mx = ((ls.x + rs.x) / 2) * scaleX
          const my = ((ls.y + rs.y) / 2) * scaleY
          ctx.font = 'bold 11px Inter, system-ui, sans-serif'
          ctx.fillStyle = '#ffffff'
          ctx.strokeStyle = '#000000'
          ctx.lineWidth = 2.5
          const lbl = `${Math.round(angles.torso_lean)}°`
          ctx.strokeText(lbl, mx + 15, my)
          ctx.fillText(lbl, mx + 15, my)
        }
      }

      // Leg stretch
      if (enabledOverlays.value.has('leg_stretch')) {
        drawLegStretch(ctx, keypoints, scaleX, scaleY, color, homographyMatrix.value)
      }
    }
  })
}

function startSkeletonAnimation() {
  // =============================================================================
  // FRAME-ACCURATE SYNC STRATEGY
  // =============================================================================
  // We use HTMLVideoElement.requestVideoFrameCallback() when available.
  // This callback fires exactly when a new video frame is presented to the
  // compositor, and provides the precise mediaTime of that frame. This
  // guarantees the skeleton overlay is drawn for the exact frame being displayed.
  //
  // When not available, we fall back to requestAnimationFrame but still use
  // floor-semantic frame lookup (findFrameAtOrBefore) instead of interpolation,
  // ensuring we never show skeleton data from a future video frame.
  // =============================================================================

  const video = videoRef.value
  if (!video) return

  if ('requestVideoFrameCallback' in video) {
    // Use requestVideoFrameCallback for frame-perfect sync
    const onVideoFrame = (_now: DOMHighResTimeStamp, metadata: { mediaTime: number }) => {
      if (!videoRef.value || !isPlaying.value) return

      // Use the exact media time from the video frame just presented
      currentTime.value = metadata.mediaTime

      // Interpolate skeleton between floor/ceiling frames for smooth overlay
      // This never leads the video — t is clamped within [floorTime, ceilTime]
      const frame = findInterpolatedFrame(metadata.mediaTime)
      drawOverlay(frame)

      // Request callback for the next video frame
      videoFrameCallbackId = videoRef.value.requestVideoFrameCallback(onVideoFrame)
    }
    videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame)
  } else {
    // Fallback: requestAnimationFrame loop with interpolation
    const animate = () => {
      if (videoRef.value) {
        currentTime.value = videoRef.value.currentTime
      }

      // Interpolate skeleton between floor/ceiling frames for smooth overlay
      const frame = findInterpolatedFrame(currentTime.value)
      drawOverlay(frame)

      animationFrameId = requestAnimationFrame(animate)
    }
    animationFrameId = requestAnimationFrame(animate)
  }
}

function stopSkeletonAnimation() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
  if (videoFrameCallbackId !== null && videoRef.value &&
      'cancelVideoFrameCallback' in videoRef.value) {
    ;(videoRef.value as any).cancelVideoFrameCallback(videoFrameCallbackId)
    videoFrameCallbackId = null
  }
  // Draw one final frame at the exact current position
  drawOverlay()
}

// Keyboard shortcuts
function handleKeydown(event: KeyboardEvent) {
  if (event.target instanceof HTMLInputElement || isExporting.value) return

  switch (event.key) {
    case ' ':
      event.preventDefault()
      togglePlay()
      break
    case 'ArrowLeft':
      event.preventDefault()
      stepFrames(-1)
      break
    case 'ArrowRight':
      event.preventDefault()
      stepFrames(1)
      break
    case 'ArrowUp':
      event.preventDefault()
      volume.value = Math.min(1, volume.value + 0.1)
      if (videoRef.value) videoRef.value.volume = volume.value
      break
    case 'ArrowDown':
      event.preventDefault()
      volume.value = Math.max(0, volume.value - 0.1)
      if (videoRef.value) videoRef.value.volume = volume.value
      break
    case 'f':
      toggleFullscreen()
      break
    case 'm':
      toggleMute()
      break
  }
}

// Resize handler for the skeleton/overlay canvas
function handleResize() {
  resizeCanvas()
}

onMounted(() => {
  document.addEventListener('fullscreenchange', handleFullscreenChange)
  document.addEventListener('keydown', handleKeydown)
  window.addEventListener('resize', handleResize)
  attachZoomResizeObserver()
})

onUnmounted(() => {
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  document.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('resize', handleResize)
  stopSkeletonAnimation()
  if (controlsTimeout) clearTimeout(controlsTimeout)
  detachZoomResizeObserver()
  // Defensive: ensure window listeners attached during an in-flight
  // pan drag are removed even if the component unmounts mid-drag.
  window.removeEventListener('mousemove', onZoomMouseMove)
  window.removeEventListener('mouseup', onZoomMouseUp)
  isPanning = false
})

watch(() => props.showSkeleton, () => {
  if (!isPlaying.value) {
    drawOverlay()
  }
})

// Re-draw the last frame under the new camera transform when zoom/pan
// changes while the video is paused (the animation loop only runs while
// playing).
watch([() => camera.scale.value, () => camera.tx.value, () => camera.ty.value], () => {
  if (!isPlaying.value) {
    // Bypass drawOverlay's same-frame early-exit — the frame hasn't
    // changed, but the transform has, so we need to re-render.
    lastFrameNumber = -1
    drawOverlay()
  }
})

// Redraw immediately when the follow-target changes while paused so the
// camera snaps onto the newly-selected player without waiting for the next
// video frame.
watch(followedPlayerId, () => {
  if (!isPlaying.value) {
    lastFrameNumber = -1
    drawOverlay()
  }
})

// Reset the camera (and drop any active follow-lock) when leaving court
// view. The skeleton canvas always applies camera.applyToContext before
// drawing, so a leftover zoom/pan would keep the bones scaled up relative
// to the real video, creating a visible mis-registration.
watch(() => props.viewMode, (mode) => {
  if (mode !== 'court') {
    followedPlayerId.value = null
    camera.reset()
    if (!isPlaying.value) {
      lastFrameNumber = -1
      drawOverlay()
    }
  }
})

watch(() => props.showBoundingBoxes, () => {
  if (!isPlaying.value) {
    drawOverlay()
  }
})

watch(() => props.showPlayers, () => {
  if (!isPlaying.value && props.showBoundingBoxes) {
    drawOverlay()
  }
})

watch(() => props.showShuttles, () => {
  if (!isPlaying.value && props.showBoundingBoxes) {
    drawOverlay()
  }
})

watch(() => props.showRackets, () => {
  if (!isPlaying.value && props.showBoundingBoxes) {
    drawOverlay()
  }
})

watch(() => props.showShuttleTracking, () => {
  lastFrameNumber = -1 // Force redraw
  if (!isPlaying.value) {
    drawOverlay()
  }
})

// NOTE: showCourtOverlay watcher removed - automatic court detection disabled

watch(() => props.showHeatmap, (newValue) => {
  // Reset frame cache to force redraw when heatmap toggle changes
  lastFrameNumber = -1
  // Reset progressive heatmap when toggling so it starts fresh
  if (newValue) {
    progressiveHeatmapState = null
  }
  if (!isPlaying.value) {
    drawOverlay()
  }
})

watch(() => props.heatmapFrameRange, () => {
  // Reset progressive heatmap when rally selection changes
  progressiveHeatmapState = null
  lastFrameNumber = -1
  if (!isPlaying.value && props.showHeatmap) {
    drawOverlay()
  }
})

watch(currentSkeletonFrame, () => {
  if (!isPlaying.value && (props.showSkeleton || props.showBoundingBoxes)) {
    drawOverlay()
  }
})

// =============================================================================
// EXPOSE: Methods accessible by parent components via template ref
// =============================================================================

/**
 * Seek the video to a specific time (in seconds)
 */
function seekTo(timeInSeconds: number) {
  if (videoRef.value) {
    videoRef.value.currentTime = timeInSeconds
    currentTime.value = timeInSeconds
  }
}

/**
 * Seek the video to a specific frame number
 */
function seekToFrame(frameNumber: number) {
  if (videoRef.value && props.skeletonData && props.skeletonData.length > 0) {
    // Find the skeleton frame to get its timestamp
    const skFrame = props.skeletonData.find(f => f.frame === frameNumber)
    if (skFrame) {
      seekTo(skFrame.timestamp)
    } else {
      // Fallback: estimate from fps
      const fps = props.skeletonData.length > 1
        ? 1 / ((props.skeletonData[1]?.timestamp ?? 0) - (props.skeletonData[0]?.timestamp ?? 0))
        : 30 // default
      seekTo(frameNumber / fps)
    }
  }
}

// =============================================================================
// VIDEO EXPORT: Bake overlays into downloadable video
// =============================================================================
const {
  isExporting,
  exportProgress,
  startExport,
  cancelExport,
} = useVideoExport({
  videoRef,
  canvasRef,
  findFrameAtOrBeforeFn: (time: number) => {
    const frame = findFrameAtOrBefore(time)
    drawOverlay(frame)
    return frame
  },
  showHeatmap: toRef(props, 'showHeatmap') as Ref<boolean>,
})

function pause() {
  if (videoRef.value && !videoRef.value.paused) {
    videoRef.value.pause()
  }
}

function play() {
  if (videoRef.value && videoRef.value.paused) {
    videoRef.value.play()
  }
}

defineExpose({
  seekTo,
  seekToFrame,
  setPlaybackRate,
  pause,
  play,
  isExporting,
  exportProgress,
  startExport,
  cancelExport,
})
</script>

<template>
  <div
    ref="containerRef"
    class="video-player"
    @mousemove="showControlsTemporarily"
    @mouseleave="isPlaying && (showControls = false)"
  >
    <div class="video-wrapper">
      <video
        ref="videoRef"
        :src="videoUrl"
        crossorigin="anonymous"
        :class="{ 'video-dimmed': showHeatmap, 'video-hidden': viewMode === 'court' }"
        @play="handlePlay"
        @pause="handlePause"
        @timeupdate="handleTimeUpdate"
        @loadedmetadata="handleLoadedMetadata"
        @click="!isExporting && togglePlay()"
      />
      <SyntheticCourtView
        v-if="viewMode === 'court' && manualCourtKeypoints && videoRef?.videoWidth"
        :court-keypoints="manualCourtKeypoints"
        :video-width="videoRef.videoWidth"
        :video-height="videoRef.videoHeight"
        :skeleton-data="skeletonData"
        :current-frame="currentFrame"
        :fps="videoFps"
        :camera="camera"
      />
      <canvas
        v-if="(showSkeleton || showBoundingBoxes || showHeatmap) && skeletonData"
        ref="canvasRef"
        class="skeleton-canvas"
      />
      <div
        v-if="viewMode === 'court'"
        ref="zoomCaptureRef"
        class="zoom-capture"
        @wheel="onZoomWheel"
        @mousedown="onZoomMouseDown"
        @dblclick="onZoomDoubleClick"
      >
        <ViewportControls
          :camera="camera"
          :capture-el="zoomCaptureRef"
          v-model:followed-pid="followedPlayerId"
        />
      </div>
      <PoseOverlay
        :skeleton-frame="currentSkeletonFrame"
        :visible="showPoseOverlay ?? false"
        :pose-source="poseSource ?? 'both'"
      />
      <ShotSummaryOverlay
        v-if="shotSummarySegment && (shotSummaryCountdown ?? 0) > 0"
        :segment="shotSummarySegment"
        :countdown-sec="shotSummaryCountdown ?? 0"
      />
    </div>

    <div class="controls" :class="{ visible: showControls || !isPlaying }">
      <div class="progress-bar" @click="handleSeek">
        <div class="progress-bg">
          <div class="progress-fill" :style="{ width: `${progressPercent}%` }"></div>
        </div>
      </div>

      <div class="controls-row">
        <div class="controls-left">
          <button class="control-btn" @click="togglePlay" :title="isPlaying ? 'Pause' : 'Play'">
            <svg v-if="isPlaying" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          </button>

          <button class="control-btn" @click="skipTime(-10)" title="Rewind 10s">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
              <text x="9" y="15" font-size="7" fill="currentColor" stroke="none">10</text>
            </svg>
          </button>

          <button class="control-btn" @click="skipTime(10)" title="Forward 10s">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 12a9 9 0 1 1-9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
              <text x="8" y="15" font-size="7" fill="currentColor" stroke="none">10</text>
            </svg>
          </button>

          <div class="volume-control">
            <button class="control-btn" @click="toggleMute" :title="isMuted ? 'Unmute' : 'Mute'">
              <svg v-if="isMuted || volume === 0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <line x1="23" y1="9" x2="17" y2="15" />
                <line x1="17" y1="9" x2="23" y2="15" />
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
              </svg>
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              :value="volume"
              @input="handleVolumeChange"
              class="volume-slider"
            />
          </div>

          <span class="time-display">
            {{ formattedCurrentTime }} / {{ formattedDuration }}
          </span>
        </div>

        <div class="controls-right">
          <div class="playback-rates">
            <button
              v-for="rate in [0.1, 0.25, 0.5, 1, 1.5, 2]"
              :key="rate"
              class="rate-btn"
              :class="{ active: playbackRate === rate }"
              @click="setPlaybackRate(rate)"
            >
              {{ rate }}x
            </button>
          </div>

          <!-- Body angle overlay toggle -->
          <div class="angle-menu-wrapper">
            <button
              class="control-btn"
              :class="{ active: enabledOverlays.size > 0 }"
              @click="showAngleMenu = !showAngleMenu"
              title="Body angle overlays"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </button>
            <div v-if="showAngleMenu" class="angle-menu">
              <button
                v-for="key in ALL_ANGLE_OVERLAYS"
                :key="key"
                class="angle-menu-item"
                :class="{ active: enabledOverlays.has(key) }"
                @click="toggleOverlay(key)"
              >
                <span class="angle-menu-check">{{ enabledOverlays.has(key) ? '\u2713' : '' }}</span>
                {{ ANGLE_OVERLAY_LABELS[key] }}
              </button>
            </div>
          </div>

          <button class="control-btn" @click="toggleFullscreen" :title="isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'">
            <svg v-if="!isFullscreen" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2">
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2">
              <polyline points="4 14 10 14 10 20" />
              <polyline points="20 10 14 10 14 4" />
              <line x1="14" y1="10" x2="21" y2="3" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-player {
  position: relative;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 0;
  overflow: hidden;
}

.video-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
}

video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  transition: filter 0.3s ease;
}

video.video-dimmed {
  filter: brightness(0.4);
}

.video-hidden {
  /* Keep video in layout + keep playback state (audio, timing) alive,
     but render it invisible so the synthetic court canvas shows through.
     opacity:0 is chosen over display:none so the video keeps decoding
     frames (the timing events drive skeleton/shuttle sync). */
  opacity: 0;
  pointer-events: none;
}

.video-wrapper:has(.video-hidden) {
  /* Dark backdrop behind the (invisible) video so the synthetic court
     has a canvas-native dark background even before SyntheticCourtView
     paints. */
  background: #0f1419;
}

.skeleton-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  object-fit: contain;
  /* Must sit above the SyntheticCourtView canvas (z-index: 2) so skeletons
     and bounding boxes remain visible in court mode. */
  z-index: 3;
}

.zoom-capture {
  position: absolute;
  inset: 0;
  z-index: 5; /* above .skeleton-canvas (z=3) and synthetic court (z=2) */
  cursor: grab;
  touch-action: none; /* prevent default scroll on trackpad pinch */
}
.zoom-capture:active { cursor: grabbing; }

.controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 16px;
  background: rgba(0, 0, 0, 0.85);
  opacity: 0;
  transition: opacity 0.3s ease;
  /* Must sit above PoseOverlay (z-index: 20), which has pointer-events:auto
     and would otherwise cover the control bar and swallow its clicks. */
  z-index: 25;
}

.controls.visible {
  opacity: 1;
}

.progress-bar {
  width: 100%;
  height: 20px;
  display: flex;
  align-items: center;
  cursor: pointer;
  margin-bottom: 8px;
}

.progress-bg {
  width: 100%;
  height: 4px;
  background: #333;
  border-radius: 0;
  overflow: hidden;
  transition: height 0.2s ease;
}

.progress-bar:hover .progress-bg {
  height: 8px;
}

.progress-fill {
  height: 100%;
  background: var(--color-accent);
  border-radius: 0;
}

.controls-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.controls-left,
.controls-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.control-btn {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: 0;
  color: white;
  cursor: pointer;
  transition: background 0.2s ease;
}

.control-btn:hover {
  background: #222;
}

.control-btn svg {
  width: 20px;
  height: 20px;
}

.volume-control {
  display: flex;
  align-items: center;
  gap: 4px;
}

.volume-slider {
  width: 80px;
  height: 4px;
  appearance: none;
  background: #333;
  border-radius: 0;
  cursor: pointer;
}

.volume-slider::-webkit-slider-thumb {
  appearance: none;
  width: 14px;
  height: 14px;
  background: white;
  border-radius: 0;
  cursor: pointer;
}

.time-display {
  color: white;
  font-size: 0.875rem;
  font-variant-numeric: tabular-nums;
  margin-left: 8px;
}

.playback-rates {
  display: flex;
  gap: 2px;
}

.rate-btn {
  padding: 4px 6px;
  background: transparent;
  border: 1px solid #333;
  border-radius: 0;
  color: #888;
  font-size: 0.7rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.rate-btn:hover {
  background: #222;
  border-color: var(--color-accent);
  color: white;
}

.rate-btn.active {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: white;
}

.angle-menu-wrapper {
  position: relative;
}

.angle-menu {
  position: absolute;
  bottom: 100%;
  right: 0;
  margin-bottom: 8px;
  background: rgba(0, 0, 0, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 6px;
  padding: 4px 0;
  min-width: 140px;
  z-index: 20;
}

.angle-menu-item {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 5px 12px;
  border: none;
  background: none;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.75rem;
  cursor: pointer;
  text-align: left;
}

.angle-menu-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.angle-menu-item.active {
  color: #4ECDC4;
}

.angle-menu-check {
  width: 14px;
  font-size: 0.7rem;
}
</style>
