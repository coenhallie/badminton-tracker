<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { calculateHomography, applyHomography } from '@/utils/homography'
import { COURT_DIMENSIONS } from '@/types/analysis'
import type { ExtendedCourtKeypoints } from '@/types/analysis'
import type { ViewportCamera } from '@/composables/useViewportCamera'

const props = defineProps<{
  courtKeypoints: ExtendedCourtKeypoints
  videoWidth: number
  videoHeight: number
  skeletonData?: Array<{ frame: number; shuttle_position?: { x: number; y: number; visible?: boolean } | null }>
  currentFrame?: number
  fps?: number
  camera: ViewportCamera
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)

// Court geometry in meters (origin top-left, y increasing toward back court).
const COURT_LEN = COURT_DIMENSIONS.length          // 13.4
const COURT_WID_D = COURT_DIMENSIONS.width_doubles // 6.1
const COURT_WID_S = COURT_DIMENSIONS.width_singles // 5.18
const NET_Y = COURT_LEN / 2                        // 6.7
const SERVICE = COURT_DIMENSIONS.service_line      // 1.98 from net
const DOUBLES_BACK_LINE_OFFSET = COURT_DIMENSIONS.back_boundary_service // 0.76
// BWF regulation: 1.524m at the centre, 1.55m at the posts — the 2.6cm
// dip in the middle is what gives a real net its subtle curve.
const NET_HEIGHT_CENTER = COURT_DIMENSIONS.net_height_center // 1.524
const NET_HEIGHT_POST = COURT_DIMENSIONS.net_height_posts    // 1.55

// Net visual tuning. The homography maps ground→pixels only, so we can't
// project true vertical heights. We estimate the net's on-screen height
// from the pixel length of the ground net line (known 6.1m at net depth)
// and fold in a factor that accounts for camera pitch — vertical world
// distances foreshorten relative to horizontal ones at the same depth.
// 0.85 ≈ cos(32°), which matches typical broadcast badminton camera tilt.
const NET_VERTICAL_FORESHORTEN = 0.85
const NET_COLOR = '#ff9500'   // Distinct orange so the net reads against white court lines

// Meters→pixels homography, recomputed when keypoints change.
const metersToPixels = computed((): number[][] | null => {
  const kp = props.courtKeypoints
  // Order must match COURT_KEYPOINT_POSITIONS in @/utils/homography.
  const videoPts = [
    kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left,
    kp.net_left, kp.net_right,
    kp.service_line_near_left, kp.service_line_near_right,
    kp.service_line_far_left, kp.service_line_far_right,
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

// Render the net as a vertical rectangle standing on the ground net line.
// Posts drawn at the taller 1.55m post height; top edge curves down to the
// 1.524m center as a quadratic Bezier, matching a real taut net's sag.
// Interior is translucent so skeletons behind the net remain visible.
function drawNet(ctx: CanvasRenderingContext2D, H: number[][]) {
  const bL = m2p(H, 0, NET_Y)
  const bR = m2p(H, COURT_WID_D, NET_Y)
  if (!bL || !bR) return

  // Pixel length of the ground net line = 6.1m of world distance at the net
  // depth. Net height pixels = (worldHeight / 6.1) × that length × foreshorten.
  const netWidthPx = Math.hypot(bR[0] - bL[0], bR[1] - bL[1])
  const mPerPx = COURT_WID_D / netWidthPx
  if (!isFinite(mPerPx) || mPerPx <= 0) return
  const postHeightPx = (NET_HEIGHT_POST / mPerPx) * NET_VERTICAL_FORESHORTEN
  const centerHeightPx = (NET_HEIGHT_CENTER / mPerPx) * NET_VERTICAL_FORESHORTEN
  if (postHeightPx <= 0) return

  const tL: [number, number] = [bL[0], bL[1] - postHeightPx]
  const tR: [number, number] = [bR[0], bR[1] - postHeightPx]
  const midBottom: [number, number] = [(bL[0] + bR[0]) / 2, (bL[1] + bR[1]) / 2]
  const midTop: [number, number] = [midBottom[0], midBottom[1] - centerHeightPx]

  // Quadratic Bezier control point solved so the curve passes through midTop
  // at t = 0.5:  B(0.5) = 0.25·tL + 0.5·C + 0.25·tR  →  C = 2·midTop − 0.5·(tL + tR)
  const ctrl: [number, number] = [
    2 * midTop[0] - 0.5 * (tL[0] + tR[0]),
    2 * midTop[1] - 0.5 * (tL[1] + tR[1]),
  ]

  // Translucent fill — follows the curved top edge.
  ctx.save()
  ctx.fillStyle = NET_COLOR
  ctx.globalAlpha = 0.12
  ctx.beginPath()
  ctx.moveTo(bL[0], bL[1])
  ctx.lineTo(bR[0], bR[1])
  ctx.lineTo(tR[0], tR[1])
  ctx.quadraticCurveTo(ctrl[0], ctrl[1], tL[0], tL[1])
  ctx.closePath()
  ctx.fill()
  ctx.restore()

  // Posts + curved top tape in solid color.
  ctx.save()
  ctx.strokeStyle = NET_COLOR
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  ctx.lineWidth = props.camera.pixelSize(3)
  ctx.beginPath()
  ctx.moveTo(tL[0], tL[1])
  ctx.quadraticCurveTo(ctrl[0], ctrl[1], tR[0], tR[1])
  ctx.stroke()

  ctx.lineWidth = props.camera.pixelSize(2)
  ctx.beginPath()
  ctx.moveTo(bL[0], bL[1])
  ctx.lineTo(tL[0], tL[1])
  ctx.moveTo(bR[0], bR[1])
  ctx.lineTo(tR[0], tR[1])
  ctx.stroke()
  ctx.restore()
}

function render() {
  const canvas = canvasRef.value
  if (!canvas) return
  canvas.width = props.videoWidth
  canvas.height = props.videoHeight

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Identity for clear + background fill, then apply camera.
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#0f1419'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  props.camera.applyToContext(ctx)

  const H = metersToPixels.value
  if (H) {
    ctx.strokeStyle = '#f5f5f5'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    for (const [[x1, y1], [x2, y2], kind] of courtLineSegments()) {
      const a = m2p(H, x1, y1)
      const b = m2p(H, x2, y2)
      if (!a || !b) continue
      ctx.lineWidth = props.camera.pixelSize(kind === 'net' ? 3 : 2)
      ctx.beginPath()
      ctx.moveTo(a[0], a[1])
      ctx.lineTo(b[0], b[1])
      ctx.stroke()
    }
    drawNet(ctx, H)
  }

  drawShuttle(ctx)
}

// Trail = last ~0.5s of shuttle positions. At 30fps = 15 frames.
const TRAIL_SECONDS = 0.5

function shuttleFrames(): Array<{ x: number; y: number; age: number }> {
  const cf = props.currentFrame ?? 0
  const fps = props.fps && props.fps > 0 ? props.fps : 30
  const window = Math.max(1, Math.round(TRAIL_SECONDS * fps))
  const frames = props.skeletonData
  if (!frames || frames.length === 0) return []

  // skeletonData is frame-indexed but may skip frames. Walk backward from
  // currentFrame, take up to `window` visible shuttle samples.
  const result: Array<{ x: number; y: number; age: number }> = []
  // Binary search would be nicer but frames are small + we only look at a window.
  for (let i = frames.length - 1; i >= 0 && result.length < window; i--) {
    const f = frames[i]
    if (!f || f.frame > cf) continue
    const sp = f.shuttle_position
    if (!sp || sp.visible === false) continue
    if (typeof sp.x !== 'number' || typeof sp.y !== 'number') continue
    const age = cf - f.frame
    if (age >= window) break
    result.push({ x: sp.x, y: sp.y, age })
  }
  return result // result[0] is newest
}

function drawShuttle(ctx: CanvasRenderingContext2D) {
  const pts = shuttleFrames()
  if (pts.length === 0) return
  const window = Math.max(1, Math.round(TRAIL_SECONDS * (props.fps && props.fps > 0 ? props.fps : 30)))

  // Trail (oldest → newest) so newer points paint over older.
  for (let i = pts.length - 1; i >= 0; i--) {
    const { x, y, age } = pts[i]!
    const alpha = Math.max(0.05, 1 - age / window)
    ctx.globalAlpha = alpha * 0.9
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(x, y, props.camera.pixelSize(3), 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1

  // Current position: bright dot + glow.
  const head = pts[0]!
  ctx.shadowColor = '#ffffff'
  // shadowBlur is defined in screen pixels per the Canvas spec and is
  // unaffected by the current transform, so it must NOT be divided by
  // camera.scale — use the literal pixel value.
  ctx.shadowBlur = 12
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  ctx.arc(head.x, head.y, props.camera.pixelSize(6), 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
}

let rafId: number | null = null
function tick() {
  render()
  rafId = requestAnimationFrame(tick)
}

onMounted(() => {
  rafId = requestAnimationFrame(tick)
})

onUnmounted(() => {
  if (rafId !== null) cancelAnimationFrame(rafId)
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
