import { ref, readonly, type Ref } from 'vue'

export interface ViewportCameraOptions {
  /** Inclusive [min, max] zoom range. Defaults to [1, 8]. */
  scaleRange?: [number, number]
  /** Fraction of the viewport that the anchored bbox must keep on-screen
   *  when clamping pan. 0.2 means at least 20% visible. Defaults to 0.2. */
  minVisibleFraction?: number
}

export interface ViewportCamera {
  scale: Readonly<Ref<number>>
  tx: Readonly<Ref<number>>
  ty: Readonly<Ref<number>>
  /** Reset to scale=1, tx=0, ty=0. */
  reset(): void
  /** Zoom toward (clientX, clientY) on the canvas element `el`. Delta is
   *  a multiplicative scale factor (e.g. 1.15 = zoom in one notch). */
  zoomAt(el: HTMLElement, clientX: number, clientY: number, delta: number): void
  /** Pan by dx, dy screen pixels. */
  panBy(el: HTMLElement, dx: number, dy: number): void
  /** ctx.setTransform(scale, 0, 0, scale, tx, ty). */
  applyToContext(ctx: CanvasRenderingContext2D): void
  /** Convert a "how many on-screen pixels do I want?" value into the
   *  pre-transform world-space value that yields that many on-screen px. */
  pixelSize(screenPx: number): number
  /** Screen-space (x, y) → world-space (pre-transform). */
  screenToWorld(x: number, y: number): { x: number; y: number }
  /** World-space (x, y) → screen-space (post-transform). */
  worldToScreen(x: number, y: number): { x: number; y: number }
  /** Re-clamp pan against the current viewport size. Call after canvas resize. */
  reclamp(el: HTMLElement): void
}

export function useViewportCamera(options: ViewportCameraOptions = {}): ViewportCamera {
  const [MIN_SCALE, MAX_SCALE] = options.scaleRange ?? [1, 8]
  const MIN_VISIBLE = options.minVisibleFraction ?? 0.2

  const scale = ref(1)
  const tx = ref(0)
  const ty = ref(0)

  function clampScale(s: number): number {
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s))
  }

  /** Clamp tx/ty so at least MIN_VISIBLE of the viewport's bbox stays on-screen. */
  function clampPan(el: HTMLElement): void {
    const w = el.clientWidth
    const h = el.clientHeight
    if (w <= 0 || h <= 0) return
    // The transformed viewport covers world-rect [−tx/s, (w−tx)/s] × [−ty/s, (h−ty)/s].
    // Equivalent: the world's [0,w] × [0,h] rect maps to screen [tx, tx + w*s] × [ty, ty + h*s].
    // Require at least MIN_VISIBLE * w (resp. h) of that rect to stay inside [0, w] (resp. [0, h]).
    const s = scale.value
    const minOverlap = MIN_VISIBLE
    const maxTx = w * (1 - minOverlap)
    const minTx = w * minOverlap - w * s
    tx.value = Math.min(maxTx, Math.max(minTx, tx.value))
    const maxTy = h * (1 - minOverlap)
    const minTy = h * minOverlap - h * s
    ty.value = Math.min(maxTy, Math.max(minTy, ty.value))
  }

  function reset(): void {
    scale.value = 1
    tx.value = 0
    ty.value = 0
  }

  function zoomAt(el: HTMLElement, clientX: number, clientY: number, delta: number): void {
    const rect = el.getBoundingClientRect()
    // Screen-space point on the canvas, in canvas pixels.
    const sx = clientX - rect.left
    const sy = clientY - rect.top
    const oldScale = scale.value
    const newScale = clampScale(oldScale * delta)
    if (newScale === oldScale) return
    // Keep the world point under the cursor fixed:
    //   screen = world * s + t   =>   world = (screen − t) / s
    // After zoom, want screen' = screen, so: t' = screen − world * s'
    const worldX = (sx - tx.value) / oldScale
    const worldY = (sy - ty.value) / oldScale
    scale.value = newScale
    tx.value = sx - worldX * newScale
    ty.value = sy - worldY * newScale
    clampPan(el)
  }

  function panBy(el: HTMLElement, dx: number, dy: number): void {
    tx.value += dx
    ty.value += dy
    clampPan(el)
  }

  function applyToContext(ctx: CanvasRenderingContext2D): void {
    ctx.setTransform(scale.value, 0, 0, scale.value, tx.value, ty.value)
  }

  function pixelSize(screenPx: number): number {
    return screenPx / scale.value
  }

  function screenToWorld(x: number, y: number): { x: number; y: number } {
    return { x: (x - tx.value) / scale.value, y: (y - ty.value) / scale.value }
  }

  function worldToScreen(x: number, y: number): { x: number; y: number } {
    return { x: x * scale.value + tx.value, y: y * scale.value + ty.value }
  }

  function reclamp(el: HTMLElement): void {
    clampPan(el)
  }

  return {
    scale: readonly(scale),
    tx: readonly(tx),
    ty: readonly(ty),
    reset,
    zoomAt,
    panBy,
    applyToContext,
    pixelSize,
    screenToWorld,
    worldToScreen,
    reclamp,
  }
}
