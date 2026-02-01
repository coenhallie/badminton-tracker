<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, shallowRef, nextTick } from 'vue'
import type { SkeletonFrame, Keypoint, BadmintonDetections, BoundingBoxDetection } from '@/types/analysis'
import { SKELETON_CONNECTIONS, PLAYER_COLORS } from '@/types/analysis'
import PoseOverlay from './PoseOverlay.vue'

// =============================================================================
// PERFORMANCE OPTIMIZATION: Debug mode flag
// =============================================================================
// Set to true only during development to see debug logs
// MUST be false in production to avoid performance overhead
// =============================================================================
const DEBUG_MODE = import.meta.env.DEV && false // Disabled even in dev by default

// NOTE: CourtDetectionResult interface removed - automatic court detection disabled
// Manual court keypoints are now the only method for court calibration

// Manual court keypoint type
interface ManualCourtKeypoint {
  x: number
  y: number
  label: string
}

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
const PROGRESSIVE_HEATMAP_WIDTH = 64  // Low resolution for performance
const PROGRESSIVE_HEATMAP_HEIGHT = 48
const PROGRESSIVE_HEATMAP_RADIUS = 3  // Gaussian radius in heatmap cells

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
  videoHeight: number
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
  heatmapData?: HeatmapData | null
  showSkeleton?: boolean
  showBoundingBoxes?: boolean
  showPlayers?: boolean
  showShuttles?: boolean
  showRackets?: boolean
  showPoseOverlay?: boolean
  poseSource?: 'skeleton' | 'trained' | 'both'
  // NOTE: showCourtOverlay prop removed - no automatic court detection to display
  showHeatmap?: boolean
}>()

// Colors for bounding boxes (matching backend colors)
const PLAYER_BOX_COLOR = '#00FF00'      // Green for players
const SHUTTLE_BOX_COLOR = '#FFA500'     // Orange for shuttlecock
const RACKET_BOX_COLOR = '#FF00FF'      // Magenta for rackets
const OTHER_BOX_COLOR = '#00FFFF'       // Cyan for other detections

// NOTE: COURT_REGION_COLORS removed - automatic court detection disabled

// Extended court keypoints type for 12-point system
interface ExtendedCourtKeypoints {
  // 4 outer corners
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  // Net intersections
  net_left: number[]
  net_right: number[]
  // Service line corners (near court - top half)
  service_near_left: number[]
  service_near_right: number[]
  // Service line corners (far court - bottom half)
  service_far_left: number[]
  service_far_right: number[]
  // Center line endpoints
  center_near: number[]
  center_far: number[]
}

const emit = defineEmits<{
  timeUpdate: [time: number]
  frameUpdate: [frame: number]
  courtKeypointsSet: [keypoints: ExtendedCourtKeypoints]
  keypointsConfirmed: [keypoints: ExtendedCourtKeypoints]
  play: []
  keypointSelectionChange: [isActive: boolean, currentCount: number]
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const keypointCanvasRef = ref<HTMLCanvasElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)

const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const volume = ref(1)
const isMuted = ref(false)
const playbackRate = ref(1)
const isFullscreen = ref(false)
const showControls = ref(true)

// =============================================================================
// MANUAL COURT KEYPOINT SELECTION - 12 POINT SYSTEM
// =============================================================================
// Users can click on 12 court reference points for precise homography:
// 1-4: Outer corners (TL, TR, BR, BL)
// 5-6: Net intersections (NL, NR)
// 7-8: Near service line corners (SSL_NL, SSL_NR)
// 9-10: Far service line corners (SSL_FL, SSL_FR)
// 11-12: Center line endpoints at service lines (CT_N, CT_F)
// =============================================================================
const isKeypointSelectionMode = ref(false)
const manualKeypoints = ref<ManualCourtKeypoint[]>([])
const TOTAL_KEYPOINTS = 12
const KEYPOINT_LABELS = [
  'TL', 'TR', 'BR', 'BL',           // 4 outer corners
  'NL', 'NR',                        // Net intersections with sidelines
  'SNL', 'SNR',                      // Service line near court (top half)
  'SFL', 'SFR',                      // Service line far court (bottom half)
  'CTN', 'CTF'                       // Center line at service lines
] as const
const KEYPOINT_FULL_LABELS = [
  'Top-Left Corner', 'Top-Right Corner', 'Bottom-Right Corner', 'Bottom-Left Corner',
  'Net-Left', 'Net-Right',
  'Service Near-Left', 'Service Near-Right',
  'Service Far-Left', 'Service Far-Right',
  'Center Near', 'Center Far'
] as const
// 12 distinct colors for each keypoint
const keypointColors = [
  '#FF4444', '#44FF44', '#4444FF', '#FFFF44',  // Corners: Red, Green, Blue, Yellow
  '#FF00FF', '#00FFFF',                         // Net: Magenta, Cyan
  '#FF8800', '#88FF00',                         // Service near: Orange, Lime
  '#0088FF', '#FF0088',                         // Service far: Azure, Rose
  '#FFFFFF', '#888888'                          // Center: White, Gray
]

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
let controlsTimeout: number | null = null
let lastRenderTime = 0

// PERFORMANCE OPTIMIZATION: Frame rate limiting for canvas rendering
const TARGET_OVERLAY_FPS = 30  // Limit overlay rendering to 30fps (reduces CPU usage)
const MIN_FRAME_TIME = 1000 / TARGET_OVERLAY_FPS

// PERFORMANCE OPTIMIZATION: Cached canvas context and pre-computed values
let cachedCtx: CanvasRenderingContext2D | null = null
let lastFrameNumber = -1

const formattedCurrentTime = computed(() => formatTime(currentTime.value))
const formattedDuration = computed(() => formatTime(duration.value))
const progressPercent = computed(() =>
  duration.value > 0 ? (currentTime.value / duration.value) * 100 : 0
)

const currentFrame = computed(() => {
  if (!videoRef.value) return 0
  const fps = 30 // Assume 30fps, would be better to get from video metadata
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
  
  newData.forEach((frame, idx) => {
    fIndex.set(frame.frame, idx)
    tIndex.push(frame.timestamp)
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
 * Get skeleton frame for current video time
 * Uses binary search for O(log n) lookup performance
 */
const currentSkeletonFrame = computed(() => {
  if (!props.skeletonData || props.skeletonData.length === 0) return null

  const targetTime = currentTime.value
  
  // Use binary search for fast lookup
  const closestIdx = binarySearchTimestamp(targetTime)
  if (closestIdx < 0) return null
  
  return props.skeletonData[closestIdx] ?? null
})

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function togglePlay() {
  if (!videoRef.value) return

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
  resizeKeypointCanvas()
}

function handleSeek(event: MouseEvent) {
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

// =============================================================================
// MANUAL KEYPOINT SELECTION FUNCTIONS
// =============================================================================

/**
 * Toggle keypoint selection mode on/off
 * When entering mode, pause the video for precise clicking
 */
function toggleKeypointSelectionMode() {
  isKeypointSelectionMode.value = !isKeypointSelectionMode.value
  
  if (isKeypointSelectionMode.value) {
    // Pause video when entering keypoint selection mode
    if (videoRef.value && isPlaying.value) {
      videoRef.value.pause()
    }
    // Clear previous keypoints when starting new selection
    manualKeypoints.value = []
    // Initialize keypoint canvas
    nextTick(() => {
      resizeKeypointCanvas()
      drawKeypointOverlay()
    })
  }
}

/**
 * Clear all manually selected keypoints and exit selection mode
 */
function clearManualKeypoints() {
  manualKeypoints.value = []
  isKeypointSelectionMode.value = false
  drawKeypointOverlay()
}

/**
 * Resize keypoint canvas to match video dimensions
 */
function resizeKeypointCanvas() {
  if (!keypointCanvasRef.value || !videoRef.value) return
  keypointCanvasRef.value.width = videoRef.value.videoWidth || videoRef.value.clientWidth
  keypointCanvasRef.value.height = videoRef.value.videoHeight || videoRef.value.clientHeight
}

/**
 * Handle click on keypoint canvas to add a new keypoint
 */
function handleKeypointCanvasClick(event: MouseEvent) {
  if (!isKeypointSelectionMode.value) return
  if (manualKeypoints.value.length >= TOTAL_KEYPOINTS) return
  
  const canvas = keypointCanvasRef.value
  if (!canvas) return
  
  // Get click position relative to canvas
  const rect = canvas.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const clickY = event.clientY - rect.top
  
  // Scale click position to video coordinates
  const scaleX = (videoRef.value?.videoWidth || canvas.width) / rect.width
  const scaleY = (videoRef.value?.videoHeight || canvas.height) / rect.height
  
  const videoX = clickX * scaleX
  const videoY = clickY * scaleY
  
  // Add keypoint
  const label = KEYPOINT_LABELS[manualKeypoints.value.length] ?? 'Unknown'
  manualKeypoints.value.push({
    x: videoX,
    y: videoY,
    label
  })
  
  if (DEBUG_MODE) {
    console.log(`[Keypoint Selection] Added ${label} at (${videoX.toFixed(1)}, ${videoY.toFixed(1)}) [${manualKeypoints.value.length}/${TOTAL_KEYPOINTS}]`)
  }
  
  // Redraw overlay
  drawKeypointOverlay()
  
  // If we have all 12 keypoints, emit them
  if (manualKeypoints.value.length === TOTAL_KEYPOINTS) {
    const kp = manualKeypoints.value
    const keypointsData: ExtendedCourtKeypoints = {
      // 4 outer corners
      top_left: [kp[0]?.x ?? 0, kp[0]?.y ?? 0],
      top_right: [kp[1]?.x ?? 0, kp[1]?.y ?? 0],
      bottom_right: [kp[2]?.x ?? 0, kp[2]?.y ?? 0],
      bottom_left: [kp[3]?.x ?? 0, kp[3]?.y ?? 0],
      // Net intersections
      net_left: [kp[4]?.x ?? 0, kp[4]?.y ?? 0],
      net_right: [kp[5]?.x ?? 0, kp[5]?.y ?? 0],
      // Service line corners (near court)
      service_near_left: [kp[6]?.x ?? 0, kp[6]?.y ?? 0],
      service_near_right: [kp[7]?.x ?? 0, kp[7]?.y ?? 0],
      // Service line corners (far court)
      service_far_left: [kp[8]?.x ?? 0, kp[8]?.y ?? 0],
      service_far_right: [kp[9]?.x ?? 0, kp[9]?.y ?? 0],
      // Center line endpoints
      center_near: [kp[10]?.x ?? 0, kp[10]?.y ?? 0],
      center_far: [kp[11]?.x ?? 0, kp[11]?.y ?? 0]
    }
    emit('courtKeypointsSet', keypointsData)
    if (DEBUG_MODE) {
      console.log('[Keypoint Selection] All 12 keypoints collected:', keypointsData)
    }
  }
}

/**
 * Undo the last added keypoint
 */
function undoLastKeypoint() {
  if (manualKeypoints.value.length > 0) {
    manualKeypoints.value.pop()
    drawKeypointOverlay()
  }
}

/**
 * Confirm and apply the selected keypoints
 * Called when the "Done" button is clicked
 * This triggers zone coverage recalculation with the new homography
 */
function confirmKeypoints() {
  if (manualKeypoints.value.length !== TOTAL_KEYPOINTS) {
    console.warn('[Keypoint Selection] Cannot confirm - need all 12 keypoints')
    return
  }
  
  const kp = manualKeypoints.value
  const keypointsData: ExtendedCourtKeypoints = {
    // 4 outer corners
    top_left: [kp[0]?.x ?? 0, kp[0]?.y ?? 0],
    top_right: [kp[1]?.x ?? 0, kp[1]?.y ?? 0],
    bottom_right: [kp[2]?.x ?? 0, kp[2]?.y ?? 0],
    bottom_left: [kp[3]?.x ?? 0, kp[3]?.y ?? 0],
    // Net intersections
    net_left: [kp[4]?.x ?? 0, kp[4]?.y ?? 0],
    net_right: [kp[5]?.x ?? 0, kp[5]?.y ?? 0],
    // Service line corners (near court)
    service_near_left: [kp[6]?.x ?? 0, kp[6]?.y ?? 0],
    service_near_right: [kp[7]?.x ?? 0, kp[7]?.y ?? 0],
    // Service line corners (far court)
    service_far_left: [kp[8]?.x ?? 0, kp[8]?.y ?? 0],
    service_far_right: [kp[9]?.x ?? 0, kp[9]?.y ?? 0],
    // Center line endpoints
    center_near: [kp[10]?.x ?? 0, kp[10]?.y ?? 0],
    center_far: [kp[11]?.x ?? 0, kp[11]?.y ?? 0]
  }
  
  console.log('[Keypoint Selection] Keypoints confirmed by user - triggering recalculation')
  
  // Emit the confirmed keypoints - this should trigger zone recalculation
  emit('keypointsConfirmed', keypointsData)
  
  // Also emit courtKeypointsSet for initial setup if not already set
  emit('courtKeypointsSet', keypointsData)
  
  // Exit keypoint selection mode
  isKeypointSelectionMode.value = false
}

/**
 * Draw the keypoint selection overlay with court guide lines
 */
function drawKeypointOverlay() {
  const canvas = keypointCanvasRef.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // If not in selection mode, don't draw anything
  if (!isKeypointSelectionMode.value && manualKeypoints.value.length === 0) return
  
  // Scale factors
  const videoWidth = videoRef.value?.videoWidth || canvas.width
  const videoHeight = videoRef.value?.videoHeight || canvas.height
  const scaleX = canvas.width / videoWidth
  const scaleY = canvas.height / videoHeight
  
  // Draw court structure guide lines based on collected keypoints
  drawCourtGuideLines(ctx, scaleX, scaleY)
  
  // Draw keypoints
  manualKeypoints.value.forEach((kp, idx) => {
    const x = kp.x * scaleX
    const y = kp.y * scaleY
    const color = keypointColors[idx] ?? '#FFFFFF'
    
    // Draw outer glow
    ctx.beginPath()
    ctx.arc(x, y, 18, 0, Math.PI * 2)
    ctx.fillStyle = color + '40'
    ctx.fill()
    
    // Draw circle
    ctx.beginPath()
    ctx.arc(x, y, 12, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 3
    ctx.stroke()
    
    // Draw label
    ctx.font = 'bold 11px Inter, system-ui, sans-serif'
    ctx.fillStyle = '#000000'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(kp.label, x, y)
  })
}

/**
 * Draw guide lines connecting keypoints to show court structure
 */
function drawCourtGuideLines(ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) {
  const kp = manualKeypoints.value
  if (kp.length < 2) return
  
  ctx.setLineDash([8, 4])
  ctx.lineWidth = 2
  
  // Outer boundary (connect corners: 0-1-2-3-0)
  if (kp.length >= 4) {
    ctx.strokeStyle = '#00FF7F'
    ctx.beginPath()
    ctx.moveTo((kp[0]?.x ?? 0) * scaleX, (kp[0]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[1]?.x ?? 0) * scaleX, (kp[1]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[2]?.x ?? 0) * scaleX, (kp[2]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[3]?.x ?? 0) * scaleX, (kp[3]?.y ?? 0) * scaleY)
    ctx.closePath()
    ctx.stroke()
  }
  
  // Net line (connect 4-5)
  if (kp.length >= 6) {
    ctx.strokeStyle = '#FF00FF'
    ctx.beginPath()
    ctx.moveTo((kp[4]?.x ?? 0) * scaleX, (kp[4]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[5]?.x ?? 0) * scaleX, (kp[5]?.y ?? 0) * scaleY)
    ctx.stroke()
  }
  
  // Near service line (connect 6-10-7)
  if (kp.length >= 8 && kp.length >= 11) {
    ctx.strokeStyle = '#FF8800'
    ctx.beginPath()
    ctx.moveTo((kp[6]?.x ?? 0) * scaleX, (kp[6]?.y ?? 0) * scaleY)
    if (kp.length >= 11) {
      ctx.lineTo((kp[10]?.x ?? 0) * scaleX, (kp[10]?.y ?? 0) * scaleY)
    }
    ctx.lineTo((kp[7]?.x ?? 0) * scaleX, (kp[7]?.y ?? 0) * scaleY)
    ctx.stroke()
  } else if (kp.length >= 8) {
    ctx.strokeStyle = '#FF8800'
    ctx.beginPath()
    ctx.moveTo((kp[6]?.x ?? 0) * scaleX, (kp[6]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[7]?.x ?? 0) * scaleX, (kp[7]?.y ?? 0) * scaleY)
    ctx.stroke()
  }
  
  // Far service line (connect 8-11-9)
  if (kp.length >= 10 && kp.length >= 12) {
    ctx.strokeStyle = '#0088FF'
    ctx.beginPath()
    ctx.moveTo((kp[8]?.x ?? 0) * scaleX, (kp[8]?.y ?? 0) * scaleY)
    if (kp.length >= 12) {
      ctx.lineTo((kp[11]?.x ?? 0) * scaleX, (kp[11]?.y ?? 0) * scaleY)
    }
    ctx.lineTo((kp[9]?.x ?? 0) * scaleX, (kp[9]?.y ?? 0) * scaleY)
    ctx.stroke()
  } else if (kp.length >= 10) {
    ctx.strokeStyle = '#0088FF'
    ctx.beginPath()
    ctx.moveTo((kp[8]?.x ?? 0) * scaleX, (kp[8]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[9]?.x ?? 0) * scaleX, (kp[9]?.y ?? 0) * scaleY)
    ctx.stroke()
  }
  
  // Center line (connect 10-11)
  if (kp.length >= 12) {
    ctx.strokeStyle = '#FFFFFF'
    ctx.beginPath()
    ctx.moveTo((kp[10]?.x ?? 0) * scaleX, (kp[10]?.y ?? 0) * scaleY)
    ctx.lineTo((kp[11]?.x ?? 0) * scaleX, (kp[11]?.y ?? 0) * scaleY)
    ctx.stroke()
  }
  
  ctx.setLineDash([])
}

// NOTE: drawKeypointGuide function removed - point order guide now rendered in App.vue as a fixed modal

function skipTime(seconds: number) {
  if (!videoRef.value) return
  videoRef.value.currentTime = Math.max(0, Math.min(duration.value, videoRef.value.currentTime + seconds))
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

// Canvas skeleton drawing
function resizeCanvas() {
  if (!canvasRef.value || !videoRef.value) return

  canvasRef.value.width = videoRef.value.videoWidth || videoRef.value.clientWidth
  canvasRef.value.height = videoRef.value.videoHeight || videoRef.value.clientHeight
}

function drawOverlay() {
  if (!canvasRef.value) {
    if (DEBUG_MODE) console.log('[Overlay Debug] No canvas ref')
    return
  }
  if (!props.showSkeleton && !props.showBoundingBoxes && !props.showHeatmap) {
    if (DEBUG_MODE) console.log('[Overlay Debug] All overlays are disabled')
    return
  }

  const frame = currentSkeletonFrame.value
  
  // Allow heatmap rendering even without skeleton frame
  // Progressive heatmap uses skeleton data, static heatmap uses pre-computed data
  const hasProgressiveHeatmap = props.showHeatmap && props.skeletonData && props.skeletonData.length > 0
  const hasStaticHeatmap = props.showHeatmap && props.heatmapData
  const hasHeatmapToRender = hasProgressiveHeatmap || hasStaticHeatmap
  const hasFrameToRender = frame && (props.showSkeleton || props.showBoundingBoxes)
  
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

  // PERFORMANCE OPTIMIZATION: Skip rendering if same frame (only when frame exists)
  // Always render if we have heatmap to show
  if (frame && !hasHeatmapToRender) {
    if (frame.frame === lastFrameNumber) return
    lastFrameNumber = frame.frame
  } else if (frame) {
    lastFrameNumber = frame.frame
  }
  
  // DEBUG: Log frame info (only in debug mode)
  if (DEBUG_MODE) {
    console.log('[Overlay Debug] Drawing frame:', frame?.frame ?? 'N/A',
      'showSkeleton:', props.showSkeleton,
      'showBoundingBoxes:', props.showBoundingBoxes,
      'showHeatmap:', props.showHeatmap,
      'hasHeatmapData:', !!props.heatmapData,
      'players:', frame?.players?.length ?? 0,
      'badminton_detections:', !!frame?.badminton_detections,
      'canvas:', canvasRef.value.width, 'x', canvasRef.value.height)
  }

  // PERFORMANCE OPTIMIZATION: Cache and reuse canvas context
  if (!cachedCtx) {
    cachedCtx = canvasRef.value.getContext('2d', {
      alpha: true,
      desynchronized: true,  // Reduces latency on supported browsers
    })
  }
  const ctx = cachedCtx
  if (!ctx) return

  // Clear canvas
  ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)

  // Scale factors for canvas (pre-compute once)
  // BUGFIX: Use video dimensions for accurate scaling, fallback to 1 to prevent NaN
  const videoWidth = videoRef.value?.videoWidth || videoRef.value?.clientWidth || 1
  const videoHeight = videoRef.value?.videoHeight || videoRef.value?.clientHeight || 1
  const scaleX = canvasRef.value.width / videoWidth
  const scaleY = canvasRef.value.height / videoHeight

  // NOTE: Court overlay drawing removed - automatic court detection disabled
  // Manual court keypoints are now the only method for court calibration

  // Draw heatmap overlay if enabled (behind skeleton and bounding boxes)
  // PROGRESSIVE HEATMAP: Compute heatmap dynamically based on skeleton data up to current frame
  if (props.showHeatmap) {
    let heatmapToDraw: HeatmapData | null = null
    
    // Prefer progressive heatmap from skeleton data (updates during playback)
    if (props.skeletonData && props.skeletonData.length > 0 && frame) {
      const progressiveState = computeProgressiveHeatmap(
        props.skeletonData,
        frame.frame,
        videoWidth,
        videoHeight
      )
      
      if (progressiveState && progressiveState.maxValue > 0) {
        heatmapToDraw = progressiveHeatmapToRenderFormat(progressiveState)
        if (DEBUG_MODE) {
          console.log('[Progressive Heatmap Debug] Drawing progressive heatmap for frame:', frame.frame,
            'dimensions:', progressiveState.width, 'x', progressiveState.height,
            'maxValue:', progressiveState.maxValue.toFixed(2))
        }
      }
    }
    
    // Fall back to static pre-computed heatmap if no progressive data available
    if (!heatmapToDraw && props.heatmapData) {
      heatmapToDraw = props.heatmapData
      if (DEBUG_MODE) {
        console.log('[Heatmap Debug] Drawing static heatmap overlay, data size:',
          props.heatmapData.combined_heatmap?.length ?? 0, 'x',
          props.heatmapData.combined_heatmap?.[0]?.length ?? 0)
      }
    }
    
    // Draw the heatmap
    if (heatmapToDraw) {
      drawHeatmap(ctx, heatmapToDraw, canvasRef.value.width, canvasRef.value.height, 0.6)
    }
  }

  // Draw bounding boxes if enabled (requires frame data)
  if (props.showBoundingBoxes && frame?.badminton_detections) {
    drawBoundingBoxes(ctx, frame.badminton_detections, scaleX, scaleY)
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
  scaleY: number
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
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, width, height)

    // Draw label background
    ctx.font = 'bold 12px Inter, system-ui, sans-serif'
    const labelText = `${label}: ${(det.confidence * 100).toFixed(0)}%`
    const textMetrics = ctx.measureText(labelText)
    const textHeight = 16
    const padding = 4

    ctx.fillStyle = color
    ctx.fillRect(x, y - textHeight - padding, textMetrics.width + padding * 2, textHeight + padding)

    // Draw label text
    ctx.fillStyle = '#000000'
    ctx.fillText(labelText, x + padding, y - padding - 2)
  }

  // Draw players (green) - only if showPlayers is true
  if (props.showPlayers !== false) {
    detections.players?.forEach((player, i) => {
      drawBox(player, PLAYER_BOX_COLOR, `Player ${i + 1}`)
    })
  }

  // Draw shuttlecocks (orange) - only if showShuttles is true
  if (props.showShuttles !== false) {
    detections.shuttlecocks?.forEach(shuttle => {
      drawBox(shuttle, SHUTTLE_BOX_COLOR, 'Shuttle')
      
      // Draw center marker for shuttle
      const cx = shuttle.x * scaleX
      const cy = shuttle.y * scaleY
      ctx.beginPath()
      ctx.arc(cx, cy, 6, 0, Math.PI * 2)
      ctx.fillStyle = SHUTTLE_BOX_COLOR
      ctx.fill()
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()
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
  
  // Draw each player's skeleton
  frame.players.forEach((player, playerIndex) => {
    const color = PLAYER_COLORS[playerIndex % PLAYER_COLORS.length] ?? '#FF6B6B'
    const keypoints = player.keypoints
    
    // Skip if no keypoints
    if (!keypoints || keypoints.length === 0) {
      console.warn('[Skeleton Debug] Player', player.player_id, 'has no keypoints')
      return
    }

    // Draw skeleton connections
    ctx.strokeStyle = color
    ctx.lineWidth = 3
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
        ctx.arc(kp.x * scaleX, kp.y * scaleY, 5, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
        keypointsDrawn++
      }
    }
    
    // DEBUG: Log drawing stats
    if (frame.frame <= 5 || frame.frame % 100 === 0) {
      console.log('[Skeleton Debug] Player', player.player_id,
        'drew', keypointsDrawn, 'keypoints,', connectionsDrawn, 'connections')
    }

    // Draw player label and speed (always draw if center exists)
    if (player.center) {
      ctx.font = 'bold 14px Inter, system-ui, sans-serif'
      ctx.fillStyle = color
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 3

      // player_id is 0-indexed, display as 1-indexed (Player 1, Player 2)
      const label = `P${player.player_id + 1}: ${player.current_speed?.toFixed(1) ?? 0} km/h`
      const x = player.center.x * scaleX
      const y = player.center.y * scaleY - 30

      ctx.strokeText(label, x - 30, y)
      ctx.fillText(label, x - 30, y)
    }
  })
}

function startSkeletonAnimation() {
  const animate = (timestamp: number) => {
    // PERFORMANCE OPTIMIZATION: Frame rate limiting
    // Only render at TARGET_OVERLAY_FPS to reduce CPU usage
    const elapsed = timestamp - lastRenderTime
    
    if (elapsed >= MIN_FRAME_TIME) {
      lastRenderTime = timestamp - (elapsed % MIN_FRAME_TIME)
      drawOverlay()
    }
    
    animationFrameId = requestAnimationFrame(animate)
  }
  // Reset timing on start
  lastRenderTime = performance.now()
  animationFrameId = requestAnimationFrame(animate)
}

function stopSkeletonAnimation() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
  // Draw one final frame
  drawOverlay()
}

// Keyboard shortcuts
function handleKeydown(event: KeyboardEvent) {
  if (event.target instanceof HTMLInputElement) return

  switch (event.key) {
    case ' ':
      event.preventDefault()
      togglePlay()
      break
    case 'ArrowLeft':
      skipTime(-5)
      break
    case 'ArrowRight':
      skipTime(5)
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

// Resize handler that resizes both canvases
function handleResize() {
  resizeCanvas()
  resizeKeypointCanvas()
  drawKeypointOverlay()
}

onMounted(() => {
  document.addEventListener('fullscreenchange', handleFullscreenChange)
  document.addEventListener('keydown', handleKeydown)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  document.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('resize', handleResize)
  stopSkeletonAnimation()
  if (controlsTimeout) clearTimeout(controlsTimeout)
})

watch(() => props.showSkeleton, () => {
  if (!isPlaying.value) {
    drawOverlay()
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

watch(() => props.heatmapData, () => {
  // Reset frame cache to force redraw when heatmap data arrives
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

// Emit keypoint selection state changes to parent
watch([isKeypointSelectionMode, () => manualKeypoints.value.length], ([isActive, count]) => {
  emit('keypointSelectionChange', isActive as boolean, count as number)
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
        :class="{ 'video-dimmed': showHeatmap }"
        @play="handlePlay"
        @pause="handlePause"
        @timeupdate="handleTimeUpdate"
        @loadedmetadata="handleLoadedMetadata"
        @click="!isKeypointSelectionMode && togglePlay()"
      />
      <canvas
        v-if="(showSkeleton || showBoundingBoxes || showHeatmap) && (skeletonData || heatmapData)"
        ref="canvasRef"
        class="skeleton-canvas"
      />
      <!-- Keypoint selection canvas - clickable when in selection mode -->
      <canvas
        ref="keypointCanvasRef"
        class="keypoint-canvas"
        :class="{ 'selection-mode': isKeypointSelectionMode }"
        @click="handleKeypointCanvasClick"
      />
      <PoseOverlay
        :skeleton-frame="currentSkeletonFrame"
        :visible="showPoseOverlay ?? false"
        :pose-source="poseSource ?? 'both'"
      />
      <div v-if="!isPlaying && !isKeypointSelectionMode" class="play-overlay" @click="togglePlay">
        <div class="play-button">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
        </div>
      </div>
      
      <!-- Keypoint selection mode controls -->
      <div v-if="isKeypointSelectionMode" class="keypoint-controls">
        <div class="keypoint-info">
          <span class="keypoint-title">🎯 12-Point Court Mapping</span>
          <span class="keypoint-count">{{ manualKeypoints.length }} / 12 points set</span>
        </div>
        <div class="keypoint-buttons">
          <button
            class="keypoint-btn undo"
            @click="undoLastKeypoint"
            :disabled="manualKeypoints.length === 0"
            title="Undo last keypoint"
          >
            ↩ Undo
          </button>
          <button
            class="keypoint-btn cancel"
            @click="clearManualKeypoints"
            title="Cancel keypoint selection"
          >
            ✕ Cancel
          </button>
          <button
            v-if="manualKeypoints.length === 12"
            class="keypoint-btn apply"
            @click="confirmKeypoints"
            title="Confirm keypoints and recalculate zone coverage"
          >
            ✓ Done
          </button>
        </div>
      </div>
      
    </div>

    <div class="controls" :class="{ visible: (showControls || !isPlaying) && !isKeypointSelectionMode }">
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
              v-for="rate in [0.5, 1, 1.5, 2]"
              :key="rate"
              class="rate-btn"
              :class="{ active: playbackRate === rate }"
              @click="setPlaybackRate(rate)"
            >
              {{ rate }}x
            </button>
          </div>

          <!-- Manual keypoint selection button -->
          <button
            class="control-btn keypoint-toggle"
            :class="{ active: isKeypointSelectionMode }"
            @click="toggleKeypointSelectionMode"
            title="Set court corners manually"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              <path d="M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </button>

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
  background: #000;
  border: 1px solid #222;
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

.skeleton-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.play-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.3);
  cursor: pointer;
}

.play-button {
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #22c55e;
  border-radius: 0;
  color: white;
  transition: transform 0.2s ease, background 0.2s ease;
}

.play-button:hover {
  transform: scale(1.05);
  background: #16a34a;
}

.play-button svg {
  width: 32px;
  height: 32px;
  margin-left: 4px;
}

.controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 16px;
  background: rgba(0, 0, 0, 0.85);
  opacity: 0;
  transition: opacity 0.3s ease;
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
  background: #22c55e;
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
  gap: 4px;
}

.rate-btn {
  padding: 4px 8px;
  background: transparent;
  border: 1px solid #333;
  border-radius: 0;
  color: #888;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.rate-btn:hover {
  background: #222;
  border-color: #22c55e;
  color: white;
}

.rate-btn.active {
  background: #22c55e;
  border-color: #22c55e;
  color: white;
}

/* ============================ */
/* Keypoint Selection Styles    */
/* ============================ */

.keypoint-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
}

.keypoint-canvas.selection-mode {
  pointer-events: auto;
  cursor: crosshair;
}

.keypoint-controls {
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  background: #0d0d0d;
  border: 2px solid #22c55e;
  border-radius: 0;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  z-index: 20;
}

.keypoint-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.keypoint-title {
  color: #22c55e;
  font-weight: bold;
  font-size: 1rem;
}

.keypoint-count {
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.875rem;
}

.keypoint-buttons {
  display: flex;
  gap: 8px;
}

.keypoint-btn {
  padding: 8px 16px;
  border: 1px solid #333;
  border-radius: 0;
  font-weight: bold;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.keypoint-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.keypoint-btn.undo {
  background: #1a1a1a;
  border-color: #333;
  color: white;
}

.keypoint-btn.undo:hover:not(:disabled) {
  background: #222;
  border-color: #22c55e;
}

.keypoint-btn.cancel {
  background: #1a0000;
  border-color: #ef4444;
  color: #ef4444;
}

.keypoint-btn.cancel:hover {
  background: #2a0000;
}

.keypoint-btn.apply {
  background: #001a00;
  border-color: #22c55e;
  color: #22c55e;
}

.keypoint-btn.apply:hover {
  background: #002a00;
}

.control-btn.keypoint-toggle.active {
  background: #001a00;
  border: 1px solid #22c55e;
  color: #22c55e;
}

.control-btn.keypoint-toggle.active:hover {
  background: #002a00;
}
</style>
