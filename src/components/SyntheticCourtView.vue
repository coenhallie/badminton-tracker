<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { calculateHomography, applyHomography } from '@/utils/homography'
import { COURT_DIMENSIONS } from '@/types/analysis'

interface ExtendedCourtKeypoints {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  net_left: number[]
  net_right: number[]
  service_near_left: number[]
  service_near_right: number[]
  service_far_left: number[]
  service_far_right: number[]
  center_near: number[]
  center_far: number[]
}

const props = defineProps<{
  courtKeypoints: ExtendedCourtKeypoints
  videoWidth: number
  videoHeight: number
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)

// Court geometry in meters (origin top-left, y increasing toward back court).
const COURT_LEN = COURT_DIMENSIONS.length          // 13.4
const COURT_WID_D = COURT_DIMENSIONS.width_doubles // 6.1
const COURT_WID_S = COURT_DIMENSIONS.width_singles // 5.18
const NET_Y = COURT_LEN / 2                        // 6.7
const SERVICE = COURT_DIMENSIONS.service_line      // 1.98 from net
const DOUBLES_BACK_LINE_OFFSET = COURT_DIMENSIONS.back_boundary_service // 0.76

// Meters→pixels homography, recomputed when keypoints change.
const metersToPixels = computed((): number[][] | null => {
  const kp = props.courtKeypoints
  // Order must match COURT_KEYPOINT_POSITIONS in @/utils/homography.
  const videoPts = [
    kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left,
    kp.net_left, kp.net_right,
    kp.service_near_left, kp.service_near_right,
    kp.service_far_left, kp.service_far_right,
    kp.center_near, kp.center_far,
  ]
  // Court-meter positions matching the 12 keypoints above.
  const courtPts: number[][] = [
    [0, 0], [COURT_WID_D, 0], [COURT_WID_D, COURT_LEN], [0, COURT_LEN],
    [0, NET_Y], [COURT_WID_D, NET_Y],
    [0, NET_Y - SERVICE], [COURT_WID_D, NET_Y - SERVICE],
    [0, NET_Y + SERVICE], [COURT_WID_D, NET_Y + SERVICE],
    [COURT_WID_D / 2, NET_Y - SERVICE], [COURT_WID_D / 2, NET_Y + SERVICE],
  ]
  // Validate all 12 points are present (length >= 2, numeric).
  for (const p of videoPts) {
    if (!p || p.length < 2 || typeof p[0] !== 'number' || typeof p[1] !== 'number') {
      console.warn('[SyntheticCourtView] Incomplete keypoints; skipping homography.')
      return null
    }
  }
  const H = calculateHomography(courtPts, videoPts)
  if (!H) {
    console.warn('[SyntheticCourtView] Degenerate homography; court lines will not render.')
  }
  return H
})

// Helper: project a court-meter point to video-pixel coords.
function m2p(H: number[][], xm: number, ym: number): [number, number] | null {
  const p = applyHomography(H, xm, ym)
  return p ? [p.x, p.y] : null
}

// All standard badminton court lines, as pairs of meter endpoints.
function courtLineSegments(): Array<[[number, number], [number, number], 'normal' | 'net']> {
  const singlesOffset = (COURT_WID_D - COURT_WID_S) / 2
  const leftSingles = singlesOffset
  const rightSingles = COURT_WID_D - singlesOffset
  const shortSvcNear = NET_Y - SERVICE
  const shortSvcFar = NET_Y + SERVICE
  const longSvcNear = DOUBLES_BACK_LINE_OFFSET
  const longSvcFar = COURT_LEN - DOUBLES_BACK_LINE_OFFSET

  return [
    // Outer boundary (doubles)
    [[0, 0], [COURT_WID_D, 0], 'normal'],
    [[COURT_WID_D, 0], [COURT_WID_D, COURT_LEN], 'normal'],
    [[COURT_WID_D, COURT_LEN], [0, COURT_LEN], 'normal'],
    [[0, COURT_LEN], [0, 0], 'normal'],
    // Singles sidelines (full length)
    [[leftSingles, 0], [leftSingles, COURT_LEN], 'normal'],
    [[rightSingles, 0], [rightSingles, COURT_LEN], 'normal'],
    // Short service lines
    [[0, shortSvcNear], [COURT_WID_D, shortSvcNear], 'normal'],
    [[0, shortSvcFar], [COURT_WID_D, shortSvcFar], 'normal'],
    // Long service line (doubles)
    [[0, longSvcNear], [COURT_WID_D, longSvcNear], 'normal'],
    [[0, longSvcFar], [COURT_WID_D, longSvcFar], 'normal'],
    // Center line (does NOT cross the net zone)
    [[COURT_WID_D / 2, 0], [COURT_WID_D / 2, shortSvcNear], 'normal'],
    [[COURT_WID_D / 2, shortSvcFar], [COURT_WID_D / 2, COURT_LEN], 'normal'],
    // Net
    [[0, NET_Y], [COURT_WID_D, NET_Y], 'net'],
  ]
}

// Draw court lines to an offscreen canvas; re-created whenever homography changes.
const offscreenCourt = ref<HTMLCanvasElement | null>(null)

function buildOffscreenCourt() {
  const H = metersToPixels.value
  if (!H) { offscreenCourt.value = null; return }

  const off = document.createElement('canvas')
  off.width = props.videoWidth
  off.height = props.videoHeight
  const ctx = off.getContext('2d')
  if (!ctx) return

  ctx.fillStyle = '#0f1419'
  ctx.fillRect(0, 0, off.width, off.height)

  ctx.strokeStyle = '#f5f5f5'
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  for (const [[x1, y1], [x2, y2], kind] of courtLineSegments()) {
    const a = m2p(H, x1, y1)
    const b = m2p(H, x2, y2)
    if (!a || !b) continue
    ctx.lineWidth = kind === 'net' ? 3 : 2
    ctx.beginPath()
    ctx.moveTo(a[0], a[1])
    ctx.lineTo(b[0], b[1])
    ctx.stroke()
  }

  offscreenCourt.value = off
}

function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.width = props.videoWidth
  canvas.height = props.videoHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (offscreenCourt.value) {
    ctx.drawImage(offscreenCourt.value, 0, 0)
  } else {
    // Fallback: solid dark background when homography is unavailable.
    ctx.fillStyle = '#0f1419'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
}

onMounted(() => {
  buildOffscreenCourt()
  render()
})

watch([() => props.courtKeypoints, () => props.videoWidth, () => props.videoHeight], () => {
  buildOffscreenCourt()
  render()
}, { deep: true })

onUnmounted(() => {
  offscreenCourt.value = null
})
</script>

<template>
  <canvas ref="canvasRef" class="synthetic-court-canvas" />
</template>

<style scoped>
.synthetic-court-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 2; /* Above <video>, below PoseOverlay (which sits at higher z). */
}
</style>
