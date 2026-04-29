# Body Angle & Leg Stretch Overlay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add toggle-able body angle and leg stretch distance overlays drawn on player skeletons in real-time during video playback.

**Architecture:** A reactive `Set<string>` tracks which overlays are enabled. The existing `drawSkeleton()` in VideoPlayer.vue is extended to draw angle arcs and labels at joint positions when enabled. Leg stretch computes ankle distance in meters via the existing homography utilities. A dropdown menu next to the existing controls toggles overlays on/off.

**Tech Stack:** Vue 3 + Canvas 2D API, existing homography utils (`src/utils/homography.ts`)

---

### Task 1: Add overlay state and dropdown toggle UI

**Files:**
- Modify: `src/components/VideoPlayer.vue`

**Step 1: Add overlay state and dropdown refs**

After the existing `KEYPOINT_CONFIDENCE_THRESHOLD` constant (line 1571), add:

```typescript
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
```

**Step 2: Add the dropdown toggle button and menu in the template**

In the template, insert before the fullscreen button (before line 2113 — `<button class="control-btn" @click="toggleFullscreen"`):

```html
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
                <span class="angle-menu-check">{{ enabledOverlays.has(key) ? '✓' : '' }}</span>
                {{ ANGLE_OVERLAY_LABELS[key] }}
              </button>
            </div>
          </div>
```

**Step 3: Add CSS for the dropdown**

In the `<style scoped>` section, add:

```css
.angle-menu-wrapper {
  position: relative;
}

.angle-menu {
  position: absolute;
  bottom: 100%;
  right: 0;
  margin-bottom: 8px;
  background: rgba(0, 0, 0, 0.9);
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
```

**Step 4: Verify**

Run: `npx vue-tsc --noEmit`
Expected: PASS — no type errors.

Visual check: The dropdown button appears next to the keypoint/fullscreen buttons. Clicking opens the menu, toggling items updates the Set.

**Step 5: Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "add body angle overlay toggle dropdown UI"
```

---

### Task 2: Add homography prop to VideoPlayer

**Files:**
- Modify: `src/components/VideoPlayer.vue`
- Modify: `src/App.vue` (pass courtKeypoints to VideoPlayer)

**Step 1: Add courtKeypoints prop**

In VideoPlayer.vue props (line 264), add:

```typescript
  courtKeypoints?: number[][] | null
```

**Step 2: Import homography utils**

At the top of `<script setup>` (after line 6), add:

```typescript
import { computeHomographyFromKeypoints, applyHomography } from '@/utils/homography'
```

**Step 3: Compute homography matrix**

After the `enabledOverlays` state block, add:

```typescript
const homographyMatrix = computed(() => {
  const kp = props.courtKeypoints
  if (!kp || kp.length < 4) return null
  return computeHomographyFromKeypoints(kp)
})
```

**Step 4: Pass courtKeypoints from App.vue**

Find where `<VideoPlayer>` is used in App.vue and add the prop. Search for the `VideoPlayer` component usage and add:

```html
:court-keypoints="courtKeypointsArray"
```

Where `courtKeypointsArray` is the existing court keypoints data passed to useAdvancedAnalytics. (Check the existing `courtKeypoints` computed/ref in App.vue.)

**Step 5: Verify**

Run: `npx vue-tsc --noEmit`
Expected: PASS

**Step 6: Commit**

```bash
git add src/components/VideoPlayer.vue src/App.vue
git commit -m "add court keypoints prop and homography to VideoPlayer"
```

---

### Task 3: Draw angle arcs and labels on skeleton

**Files:**
- Modify: `src/components/VideoPlayer.vue`

**Step 1: Add angle drawing helper function**

Before `drawSkeleton()`, add:

```typescript
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
 * Map each angle overlay to the 3 keypoint indices that form the angle.
 * The angle is measured at the MIDDLE keypoint (index 1).
 * Format: [point_a, vertex, point_b]
 */
const ANGLE_JOINTS: Record<string, [number, number, number]> = {
  left_elbow:    [KP.left_shoulder,  KP.left_elbow,    KP.left_wrist],
  right_elbow:   [KP.right_shoulder, KP.right_elbow,   KP.right_wrist],
  left_shoulder:  [KP.left_elbow,    KP.left_shoulder,  KP.left_hip],
  right_shoulder: [KP.right_elbow,   KP.right_shoulder, KP.right_hip],
  left_knee:     [KP.left_hip,       KP.left_knee,     KP.left_ankle],
  right_knee:    [KP.right_hip,      KP.right_knee,    KP.right_ankle],
  left_hip:      [KP.left_shoulder,  KP.left_hip,      KP.left_knee],
  right_hip:     [KP.right_shoulder, KP.right_hip,     KP.right_knee],
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

  // Compute angles of both arms from the vertex
  const angle1 = Math.atan2(ay - vy, ax - vx)
  const angle2 = Math.atan2(by - vy, bx - vx)

  // Draw arc
  const radius = 20
  ctx.beginPath()
  ctx.arc(vx, vy, radius, Math.min(angle1, angle2), Math.max(angle1, angle2))
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.globalAlpha = 0.8
  ctx.stroke()
  ctx.globalAlpha = 1.0

  // Draw label
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
```

**Step 2: Add leg stretch drawing function**

```typescript
function drawLegStretch(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  scaleX: number,
  scaleY: number,
  color: string,
  H: number[][] | null,
) {
  const la = keypoints[KP.left_ankle], ra = keypoints[KP.right_ankle]
  if (!la?.x || !la?.y || !ra?.x || !ra?.y) return
  if (la.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      ra.confidence < KEYPOINT_CONFIDENCE_THRESHOLD) return

  // Compute distance in meters via homography
  if (!H) return
  const leftM = applyHomography(H, la.x, la.y)
  const rightM = applyHomography(H, ra.x, ra.y)
  if (!leftM || !rightM) return

  const distMeters = Math.sqrt((leftM.x - rightM.x) ** 2 + (leftM.y - rightM.y) ** 2)

  const lax = la.x * scaleX, lay = la.y * scaleY
  const rax = ra.x * scaleX, ray = ra.y * scaleY

  // Dashed line between ankles
  ctx.beginPath()
  ctx.setLineDash([6, 4])
  ctx.moveTo(lax, lay)
  ctx.lineTo(rax, ray)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.globalAlpha = 0.7
  ctx.stroke()
  ctx.setLineDash([])
  ctx.globalAlpha = 1.0

  // Distance label at midpoint
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
```

**Step 3: Call the drawing functions from drawSkeleton()**

At the end of the `sortedPlayers.forEach` callback (after the player label drawing block, around line 1669), insert before the closing `})`:

```typescript
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

      // Torso lean — draw as label near shoulder midpoint
      if (enabledOverlays.value.has('torso_lean') && angles.torso_lean != null) {
        const ls = keypoints[KP.left_shoulder], rs = keypoints[KP.right_shoulder]
        if (ls?.x && ls?.y && rs?.x && rs?.y) {
          const mx = ((ls.x + rs.x) / 2) * scaleX
          const my = ((ls.y + rs.y) / 2) * scaleY
          ctx.font = 'bold 11px Inter, system-ui, sans-serif'
          ctx.fillStyle = '#ffffff'
          ctx.strokeStyle = '#000000'
          ctx.lineWidth = 2.5
          const label = `${Math.round(angles.torso_lean)}°`
          ctx.strokeText(label, mx + 15, my)
          ctx.fillText(label, mx + 15, my)
        }
      }

      // Leg stretch
      if (enabledOverlays.value.has('leg_stretch')) {
        drawLegStretch(ctx, keypoints, scaleX, scaleY, color, homographyMatrix.value)
      }
    }
```

**Step 4: Verify**

Run: `npx vue-tsc --noEmit`
Expected: PASS

Visual check: Enable overlays in dropdown, play video — arcs and labels appear on joints, dashed line between ankles with distance.

**Step 5: Commit**

```bash
git add src/components/VideoPlayer.vue
git commit -m "draw body angle arcs and leg stretch distance on skeleton overlay"
```
