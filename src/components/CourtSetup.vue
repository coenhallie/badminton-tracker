<script setup lang="ts">
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { Id } from '../../convex/_generated/dataModel'
import { fetchVideoUrl } from '@/services/api'
import MiniCourt from './MiniCourt.vue'

const props = defineProps<{
  videoId: string
  filename: string
}>()

const emit = defineEmits<{
  complete: [keypoints: ExtendedCourtKeypoints]
  skip: []
  error: [message: string]
}>()

const client = useConvexClient()

// =============================================================================
// MANUAL COURT KEYPOINT SELECTION - 12 POINT SYSTEM
// =============================================================================
// Same system used in VideoPlayer for precise homography:
// 1-4: Outer corners (TL, TR, BR, BL)
// 5-6: Net intersections (NL, NR)
// 7-8: Near service line corners (SSL_NL, SSL_NR)
// 9-10: Far service line corners (SSL_FL, SSL_FR)
// 11-12: Center line endpoints at service lines (CT_N, CT_F)
// =============================================================================

// Must match the Convex validator in videos.ts (12-point system)
interface ExtendedCourtKeypoints {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  net_left?: number[]
  net_right?: number[]
  service_line_near_left?: number[]
  service_line_near_right?: number[]
  service_line_far_left?: number[]
  service_line_far_right?: number[]
  center_near?: number[]
  center_far?: number[]
}

interface ManualCourtKeypoint {
  x: number
  y: number
  label: string
}

// 12-point system (matching VideoPlayer for precise homography)
const TOTAL_KEYPOINTS = 12
const KEYPOINT_LABELS = [
  'TL', 'TR', 'BR', 'BL',           // 4 outer corners
  'NL', 'NR',                        // Net intersections with sidelines
  'SNL', 'SNR',                      // Service line near court (top half)
  'SFL', 'SFR',                      // Service line far court (bottom half)
  'CTN', 'CTF'                       // Center line endpoints (near and far)
] as const

const KEYPOINT_FULL_LABELS = [
  'Top-Left Corner', 'Top-Right Corner', 'Bottom-Right Corner', 'Bottom-Left Corner',
  'Net-Left', 'Net-Right',
  'Service Near-Left', 'Service Near-Right',
  'Service Far-Left', 'Service Far-Right',
  'Center-Near', 'Center-Far'
] as const

// 12 distinct colors for each keypoint (same as VideoPlayer)
const keypointColors = [
  '#FF4444', '#44FF44', '#4444FF', '#FFFF44',  // Corners: Red, Green, Blue, Yellow
  '#FF00FF', '#00FFFF',                         // Net: Magenta, Cyan
  '#FF8800', '#88FF00',                         // Service near: Orange, Lime
  '#0088FF', '#FF0088',                         // Service far: Azure, Rose
  '#FFFFFF', '#888888'                          // Center line: White, Gray
]

// State
const isLoading = ref(true)
const videoUrl = ref('')
const canvasRef = ref<HTMLCanvasElement | null>(null)
const containerRef = ref<HTMLElement | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)
const manualKeypoints = ref<ManualCourtKeypoint[]>([])
const isSaving = ref(false)
const videoDimensions = ref({ width: 0, height: 0 })

// Computed
const currentPointIndex = computed(() => manualKeypoints.value.length)
const currentPointLabel = computed(() => {
  if (currentPointIndex.value >= TOTAL_KEYPOINTS) return 'Complete'
  return KEYPOINT_FULL_LABELS[currentPointIndex.value] ?? 'Unknown'
})
const isComplete = computed(() => manualKeypoints.value.length === TOTAL_KEYPOINTS)

// Load video URL and set up first frame
onMounted(async () => {
  try {
    videoUrl.value = await fetchVideoUrl(props.videoId)
    await loadVideo()
  } catch (error) {
    console.error('Failed to load video:', error)
    emit('error', 'Failed to load video for court setup')
  }
})

async function loadVideo() {
  return new Promise<void>((resolve, reject) => {
    const video = document.createElement('video')
    video.crossOrigin = 'anonymous'
    video.muted = true
    video.preload = 'metadata'
    
    video.onloadedmetadata = () => {
      videoDimensions.value = {
        width: video.videoWidth,
        height: video.videoHeight
      }
      // Seek to first frame (0.1s to avoid black frame)
      video.currentTime = 0.1
    }
    
    video.onseeked = () => {
      videoRef.value = video
      isLoading.value = false
      nextTick(() => {
        resizeCanvas()
        drawOverlay()
      })
      resolve()
    }
    
    video.onerror = () => {
      reject(new Error('Failed to load video'))
    }
    
    video.src = videoUrl.value
    video.load()
  })
}

function resizeCanvas() {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container || !videoRef.value) return
  
  // Calculate display size maintaining aspect ratio
  // Layout: minicourt 280px + 8px gap
  const aspectRatio = videoDimensions.value.height / videoDimensions.value.width
  const availableWidth = container.clientWidth // Full container width
  const minicourtWidth = 288 // Minicourt (280px) + gap (8px)
  const maxWidth = availableWidth - minicourtWidth
  const width = Math.max(maxWidth, 400) // Minimum 400px
  const height = Math.round(width * aspectRatio)
  
  canvas.width = videoDimensions.value.width
  canvas.height = videoDimensions.value.height
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
}

function handleCanvasClick(event: MouseEvent) {
  if (manualKeypoints.value.length >= TOTAL_KEYPOINTS) return
  
  const canvas = canvasRef.value
  if (!canvas) return
  
  const rect = canvas.getBoundingClientRect()
  const scaleX = videoDimensions.value.width / rect.width
  const scaleY = videoDimensions.value.height / rect.height
  
  const videoX = (event.clientX - rect.left) * scaleX
  const videoY = (event.clientY - rect.top) * scaleY
  
  const label = KEYPOINT_LABELS[manualKeypoints.value.length] ?? 'Unknown'
  manualKeypoints.value.push({
    x: videoX,
    y: videoY,
    label
  })
  
  drawOverlay()
}

function undoLastKeypoint() {
  if (manualKeypoints.value.length > 0) {
    manualKeypoints.value.pop()
    drawOverlay()
  }
}

function clearKeypoints() {
  manualKeypoints.value = []
  drawOverlay()
}

function drawOverlay() {
  const canvas = canvasRef.value
  const video = videoRef.value
  if (!canvas || !video) return
  
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // Clear and draw video frame
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  
  // Draw court guide lines (semi-transparent)
  drawCourtGuide(ctx)
  
  // Draw existing keypoints
  manualKeypoints.value.forEach((kp, index) => {
    const color = keypointColors[index] ?? '#FFFFFF'
    
    // Point circle
    ctx.beginPath()
    ctx.arc(kp.x, kp.y, 10, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 2
    ctx.stroke()
    
    // Label
    ctx.fillStyle = '#000000'
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(kp.label, kp.x, kp.y)
  })
  
  // Draw next point indicator
  if (manualKeypoints.value.length < TOTAL_KEYPOINTS) {
    const nextColor = keypointColors[manualKeypoints.value.length] ?? '#FFFFFF'
    const nextLabel = KEYPOINT_LABELS[manualKeypoints.value.length] ?? ''
    
    // Instruction box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
    ctx.fillRect(10, 10, 300, 35)
    ctx.strokeStyle = nextColor
    ctx.lineWidth = 2
    ctx.strokeRect(10, 10, 300, 35)
    
    ctx.fillStyle = nextColor
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(`Click: ${KEYPOINT_FULL_LABELS[manualKeypoints.value.length]}`, 20, 28)
  }
  
  // Connect points with lines
  if (manualKeypoints.value.length >= 2) {
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.6)'
    ctx.lineWidth = 2
    ctx.setLineDash([8, 4])
    
    // Draw outer rectangle (first 4 points)
    if (manualKeypoints.value.length >= 4) {
      ctx.beginPath()
      const [p0, p1, p2, p3] = manualKeypoints.value
      if (p0 && p1 && p2 && p3) {
        ctx.moveTo(p0.x, p0.y)
        ctx.lineTo(p1.x, p1.y)
        ctx.lineTo(p2.x, p2.y)
        ctx.lineTo(p3.x, p3.y)
        ctx.closePath()
        ctx.stroke()
      }
    }
    
    ctx.setLineDash([])
  }
}

function drawCourtGuide(ctx: CanvasRenderingContext2D) {
  // Draw a semi-transparent court guide showing expected positions
  const canvas = canvasRef.value
  if (!canvas) return
  const w = canvas.width
  const h = canvas.height
  
  ctx.strokeStyle = 'rgba(34, 197, 94, 0.2)'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 4])
  
  // Draw approximate court rectangle (middle 70% of frame)
  const margin = 0.15
  const x1 = w * margin
  const y1 = h * margin
  const x2 = w * (1 - margin)
  const y2 = h * (1 - margin)
  
  ctx.beginPath()
  ctx.rect(x1, y1, x2 - x1, y2 - y1)
  ctx.stroke()
  
  // Draw net line (middle horizontal)
  ctx.beginPath()
  ctx.moveTo(x1, h / 2)
  ctx.lineTo(x2, h / 2)
  ctx.stroke()
  
  // Draw service lines
  const serviceY1 = y1 + (h / 2 - y1) * 0.6
  const serviceY2 = y2 - (y2 - h / 2) * 0.6
  ctx.beginPath()
  ctx.moveTo(x1, serviceY1)
  ctx.lineTo(x2, serviceY1)
  ctx.moveTo(x1, serviceY2)
  ctx.lineTo(x2, serviceY2)
  ctx.stroke()
  
  // Draw center line
  ctx.beginPath()
  ctx.moveTo(w / 2, serviceY1)
  ctx.lineTo(w / 2, serviceY2)
  ctx.stroke()
  
  ctx.setLineDash([])
}

async function saveAndProceed() {
  if (!isComplete.value) return
  
  isSaving.value = true
  
  try {
    const kp = manualKeypoints.value
    // Use field names that match Convex validator (12-point system)
    const keypoints: ExtendedCourtKeypoints = {
      // 4 outer corners
      top_left: [kp[0]?.x ?? 0, kp[0]?.y ?? 0],
      top_right: [kp[1]?.x ?? 0, kp[1]?.y ?? 0],
      bottom_right: [kp[2]?.x ?? 0, kp[2]?.y ?? 0],
      bottom_left: [kp[3]?.x ?? 0, kp[3]?.y ?? 0],
      // Net intersections
      net_left: [kp[4]?.x ?? 0, kp[4]?.y ?? 0],
      net_right: [kp[5]?.x ?? 0, kp[5]?.y ?? 0],
      // Service line corners
      service_line_near_left: [kp[6]?.x ?? 0, kp[6]?.y ?? 0],
      service_line_near_right: [kp[7]?.x ?? 0, kp[7]?.y ?? 0],
      service_line_far_left: [kp[8]?.x ?? 0, kp[8]?.y ?? 0],
      service_line_far_right: [kp[9]?.x ?? 0, kp[9]?.y ?? 0],
      // Center line endpoints (for precise homography)
      center_near: [kp[10]?.x ?? 0, kp[10]?.y ?? 0],
      center_far: [kp[11]?.x ?? 0, kp[11]?.y ?? 0],
    }
    
    // Save to Convex database
    await client.mutation(api.videos.setManualCourtKeypoints, {
      videoId: props.videoId as Id<'videos'>,
      keypoints
    })
    
    console.log('[CourtSetup] 12-point keypoints saved:', keypoints)
    emit('complete', keypoints)
  } catch (error) {
    console.error('[CourtSetup] Failed to save keypoints:', error)
    emit('error', 'Failed to save court keypoints')
  } finally {
    isSaving.value = false
  }
}

function skipSetup() {
  emit('skip')
}

// Watch for resize
watch(videoDimensions, () => {
  nextTick(() => {
    resizeCanvas()
    drawOverlay()
  })
})
</script>

<template>
  <div class="court-setup">
    <div class="setup-header">
      <h2>Court Mapping</h2>
      <p class="subtitle">{{ props.filename }}</p>
    </div>

    <div class="setup-content" ref="containerRef">
      <!-- Loading state -->
      <div v-if="isLoading" class="loading-state">
        <div class="spinner"></div>
        <p>Loading video frame...</p>
      </div>

      <!-- Horizontal layout: Canvas + MiniCourt keypoint guide -->
      <div v-else class="setup-layout">
        <!-- Canvas for keypoint selection -->
        <div class="canvas-wrapper">
          <canvas
            ref="canvasRef"
            @click="handleCanvasClick"
            class="selection-canvas"
            :class="{ complete: isComplete }"
          ></canvas>
          
          <!-- Keypoint controls (same style as VideoPlayer) -->
          <div class="keypoint-controls">
            <div class="keypoint-info">
              <span class="keypoint-title">🎯 Court Mapping</span>
              <span class="keypoint-count">{{ manualKeypoints.length }} / {{ TOTAL_KEYPOINTS }} points</span>
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
                @click="skipSetup"
                title="Skip court setup"
              >
                ✕ Skip
              </button>
              <button
                v-if="isComplete"
                class="keypoint-btn apply"
                @click="saveAndProceed"
                :disabled="isSaving"
                title="Confirm keypoints and start analysis"
              >
                {{ isSaving ? '...' : '✓ Start Analysis' }}
              </button>
            </div>
          </div>
        </div>

        <!-- MiniCourt keypoint order guide - slightly larger for better visibility -->
        <div class="minicourt-guide">
          <MiniCourt
            :width="280"
            :height="500"
            :show-grid="true"
            :show-labels="true"
            :show-shuttle="false"
            :show-trails="false"
            :is-keypoint-selection-mode="true"
            :keypoint-selection-count="manualKeypoints.length"
          />
        </div>
        
        <!-- Info box - spans full width of layout -->
        <div class="info-box">
          <strong>Why 12 points?</strong>
          <p>
            The 12-point court mapping provides precise homography for player tracking,
            speed calculations, and zone coverage analysis. Click each court landmark in the order shown.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.court-setup {
  max-width: 100%;
  margin: 0;
  padding: 0;
}

.setup-header {
  text-align: center;
  margin-bottom: 24px;
}

.setup-header h2 {
  color: #22c55e;
  font-size: 1.5rem;
  margin: 0 0 8px 0;
}

.setup-header .subtitle {
  color: rgba(255, 255, 255, 0.6);
  margin: 0;
}

.setup-content {
  background: #0d0d0d;
  width: 100%;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  gap: 16px;
  color: rgba(255, 255, 255, 0.6);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #333;
  border-top-color: #22c55e;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Horizontal layout container - matches dashboard video-with-minicourt */
.setup-layout {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 8px;
  align-items: flex-start;
  justify-content: flex-start;
  width: 100%;
}

.canvas-wrapper {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  min-width: 0;
}

.selection-canvas {
  display: block;
  cursor: crosshair;
  border: 2px solid #333;
}

.selection-canvas.complete {
  cursor: default;
  border-color: #22c55e;
}

/* MiniCourt keypoint guide - slightly larger for better visibility */
.minicourt-guide {
  display: flex;
  flex-direction: column;
  background: transparent;
  padding: 0;
  flex-shrink: 0;
  width: 280px;
}

/* Keypoint controls - exact match with VideoPlayer */
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

.keypoint-btn.apply:hover:not(:disabled) {
  background: #002a00;
}

/* Info box - spans full width of layout */
.info-box {
  width: 100%;
  margin-top: 12px;
  padding: 16px;
  background: rgba(34, 197, 94, 0.05);
  border: 1px solid rgba(34, 197, 94, 0.2);
}

.info-box strong {
  display: block;
  color: #22c55e;
  margin-bottom: 6px;
}

.info-box p {
  color: rgba(255, 255, 255, 0.6);
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
}

/* Responsive layout */
@media (max-width: 900px) {
  .setup-layout {
    flex-direction: column;
    align-items: center;
  }

  .minicourt-guide {
    width: 100%;
    max-width: 400px;
  }

  .guide-legend {
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
  }

  .legend-row {
    flex-wrap: wrap;
  }
}
</style>
