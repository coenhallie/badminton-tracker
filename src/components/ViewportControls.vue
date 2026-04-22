<script setup lang="ts">
import { computed } from 'vue'
import type { ViewportCamera } from '@/composables/useViewportCamera'

const props = defineProps<{ camera: ViewportCamera; captureEl?: HTMLElement | null }>()

const zoomPercent = computed(() => Math.round(props.camera.scale.value * 100))

function stepZoom(factor: number, event: MouseEvent) {
  event.stopPropagation()
  const el = props.captureEl ?? (event.currentTarget as HTMLElement).parentElement
  if (!el) return
  // Zoom toward the center of the capture element.
  const rect = el.getBoundingClientRect()
  const cx = rect.left + rect.width / 2
  const cy = rect.top + rect.height / 2
  props.camera.zoomAt(el, cx, cy, factor)
}

function reset(event: MouseEvent) {
  event.stopPropagation()
  props.camera.reset()
}
</script>

<template>
  <div class="viewport-controls" @mousedown.stop @dblclick.stop>
    <button type="button" title="Zoom out" @click="stepZoom(1 / 1.5, $event)">−</button>
    <span class="zoom-readout">{{ zoomPercent }}%</span>
    <button type="button" title="Zoom in" @click="stepZoom(1.5, $event)">+</button>
    <button type="button" title="Reset view" class="reset" @click="reset">⟲</button>
  </div>
</template>

<style scoped>
.viewport-controls {
  position: absolute;
  right: 12px;
  bottom: 12px;
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 4px 6px;
  background: rgba(15, 20, 25, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  color: #f5f5f5;
  font: 500 12px Inter, system-ui, sans-serif;
  user-select: none;
  pointer-events: auto;
}
.viewport-controls button {
  width: 24px;
  height: 24px;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: inherit;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
}
.viewport-controls button:hover { background: rgba(255, 255, 255, 0.08); }
.viewport-controls .zoom-readout {
  min-width: 42px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.viewport-controls .reset { font-size: 12px; }
</style>
