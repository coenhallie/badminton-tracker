<script setup lang="ts">
import { ref, computed } from 'vue'
import type { AnalysisResult } from '@/types/analysis'

const props = defineProps<{
  result: AnalysisResult
  currentTime: number
}>()

const emit = defineEmits<{
  seekToTime: [time: number]
}>()

// Rallies from backend
const rallies = computed(() => props.result.rallies ?? [])

const videoTimeline = computed(() => {
  const duration = props.result.duration
  if (!duration || duration <= 0 || rallies.value.length === 0) return []
  return rallies.value.map((r, i) => ({
    id: r.id ?? i + 1,
    leftPct: (r.start_timestamp / duration) * 100,
    widthPct: (r.duration_seconds / duration) * 100,
    startTimestamp: r.start_timestamp,
    durationSeconds: r.duration_seconds,
  }))
})

const playbackPct = computed(() => {
  const duration = props.result.duration
  if (!duration || duration <= 0) return 0
  return (props.currentTime / duration) * 100
})

const selectedRallyId = ref<number | null>(null)

function selectRally(seg: typeof videoTimeline.value[0]) {
  if (selectedRallyId.value === seg.id) {
    selectedRallyId.value = null
  } else {
    selectedRallyId.value = seg.id
    emit('seekToTime', seg.startTimestamp)
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Find which rally is currently playing
const currentRallyId = computed(() => {
  const t = props.currentTime
  for (const seg of videoTimeline.value) {
    const end = seg.startTimestamp + seg.durationSeconds
    if (t >= seg.startTimestamp && t <= end) return seg.id
  }
  return null
})
</script>

<template>
  <div class="rally-tl" v-if="rallies.length > 0">
    <div class="rally-tl-header">
      <span class="rally-tl-label">Rallies</span>
      <span class="rally-tl-count">{{ rallies.length }} rallies detected</span>
      <button
        v-if="selectedRallyId !== null"
        class="rally-tl-clear"
        @click="selectedRallyId = null"
      >
        Clear
      </button>
    </div>
    <div class="rally-tl-bar" @click.self="selectedRallyId = null">
      <div
        v-for="seg in videoTimeline"
        :key="seg.id"
        class="rally-tl-segment"
        :class="{
          active: currentRallyId === seg.id,
          selected: selectedRallyId === seg.id,
          dimmed: selectedRallyId !== null && selectedRallyId !== seg.id,
        }"
        :style="{
          left: seg.leftPct + '%',
          width: Math.max(seg.widthPct, 0.4) + '%',
        }"
        :title="`Rally #${seg.id} — ${formatTime(seg.startTimestamp)} (${seg.durationSeconds.toFixed(1)}s)`"
        @click="selectRally(seg)"
      />
      <div class="rally-tl-playhead" :style="{ left: playbackPct + '%' }" />
    </div>
    <div class="rally-tl-labels">
      <span>0:00</span>
      <span>{{ formatTime(result.duration / 2) }}</span>
      <span>{{ formatTime(result.duration) }}</span>
    </div>
  </div>
</template>

<style scoped>
.rally-tl {
  padding: 8px 0 4px;
}

.rally-tl-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.rally-tl-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-heading);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.rally-tl-count {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}

.rally-tl-clear {
  margin-left: auto;
  padding: 2px 8px;
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
}

.rally-tl-clear:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.rally-tl-bar {
  position: relative;
  height: 24px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  overflow: hidden;
  cursor: default;
}

.rally-tl-segment {
  position: absolute;
  top: 3px;
  bottom: 3px;
  background: var(--color-accent);
  opacity: 0.55;
  cursor: pointer;
  transition: opacity 0.15s, background 0.15s;
  min-width: 3px;
}

.rally-tl-segment:hover {
  opacity: 1;
}

.rally-tl-segment.active {
  opacity: 0.85;
  background: var(--color-accent);
}

.rally-tl-segment.selected {
  opacity: 1;
  background: var(--color-accent);
  box-shadow: 0 0 0 1px var(--color-bg), 0 0 0 2px var(--color-accent);
}

.rally-tl-segment.dimmed {
  opacity: 0.2;
}

.rally-tl-playhead {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--color-text-heading);
  pointer-events: none;
  z-index: 2;
  transition: left 0.1s linear;
}

.rally-tl-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 3px;
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
  font-variant-numeric: tabular-nums;
}
</style>
