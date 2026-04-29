<script setup lang="ts">
import { ref, computed, inject } from 'vue'
import type { Rally, RallySpeedStats } from '@/types/analysis'
import { PLAYER_COLORS } from '@/types/analysis'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'

const playerLabelsRef = inject(PLAYER_LABELS_KEY)
const pidDisplayFor = (canonical: number): number =>
  playerLabelsRef?.value?.displayId(canonical) ?? canonical

const props = defineProps<{
  rallies: Rally[]
  backendRallies: { startTimestamp: number; endTimestamp: number; durationSeconds: number }[]
  duration: number
  rallySource: 'client' | 'backend' | 'both' | null
  currentTime: number
  pauseBetweenRallies: boolean
  rallyPauseCountdown: number
  pausedAfterRallyId: number | null
  rallySpeedStats?: RallySpeedStats[]
  pauseBetweenShots: boolean
  shotPauseDurationSec: 1 | 1.5 | 2 | 3
}>()

const emit = defineEmits<{
  seekToTime: [time: number]
  'update:pauseBetweenRallies': [value: boolean]
  skipToNextRally: []
  resumeFromPause: []
  selectRally: [rallyId: number | null]
  'update:pauseBetweenShots': [value: boolean]
  'update:shotPauseDurationSec': [value: 1 | 1.5 | 2 | 3]
}>()

const videoTimeline = computed(() => {
  if (!props.duration || props.duration <= 0 || props.rallies.length === 0) return []
  return props.rallies.map(r => ({
    id: r.id,
    leftPct: (r.startTimestamp / props.duration) * 100,
    widthPct: (r.durationSeconds / props.duration) * 100,
    startTimestamp: r.startTimestamp,
    durationSeconds: r.durationSeconds,
  }))
})

const playbackPct = computed(() => {
  if (!props.duration || props.duration <= 0) return 0
  return (props.currentTime / props.duration) * 100
})

/** Build combined timeline segments color-coded by source agreement */
const combinedTimeline = computed(() => {
  if (!props.duration || props.duration <= 0) return []
  if (props.rallies.length === 0 && props.backendRallies.length === 0) return []

  type Seg = { start: number; end: number; source: 'client' | 'backend' }
  const segments: Seg[] = []

  for (const r of props.rallies) {
    segments.push({ start: r.startTimestamp, end: r.endTimestamp, source: 'client' })
  }
  for (const r of props.backendRallies) {
    segments.push({ start: r.startTimestamp, end: r.endTimestamp, source: 'backend' })
  }

  type Event = { time: number; type: 'start' | 'end'; source: 'client' | 'backend' }
  const events: Event[] = []
  for (const s of segments) {
    events.push({ time: s.start, type: 'start', source: s.source })
    events.push({ time: s.end, type: 'end', source: s.source })
  }
  events.sort((a, b) => a.time - b.time || (a.type === 'start' ? -1 : 1))

  const result: { leftPct: number; widthPct: number; kind: 'both' | 'client' | 'backend' }[] = []
  let clientActive = 0
  let backendActive = 0
  let prevTime = 0

  for (const ev of events) {
    if ((clientActive > 0 || backendActive > 0) && ev.time > prevTime) {
      const kind = (clientActive > 0 && backendActive > 0) ? 'both'
        : clientActive > 0 ? 'client' : 'backend'
      const leftPct = (prevTime / props.duration) * 100
      const widthPct = ((ev.time - prevTime) / props.duration) * 100
      if (widthPct > 0.01) {
        result.push({ leftPct, widthPct, kind })
      }
    }
    prevTime = ev.time
    if (ev.type === 'start') {
      if (ev.source === 'client') clientActive++
      else backendActive++
    } else {
      if (ev.source === 'client') clientActive--
      else backendActive--
    }
  }

  return result
})

const selectedRallyId = ref<number | null>(null)

function selectRally(seg: typeof videoTimeline.value[0]) {
  if (selectedRallyId.value === seg.id) {
    selectedRallyId.value = null
    emit('selectRally', null)
  } else {
    selectedRallyId.value = seg.id
    emit('selectRally', seg.id)
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

// Index of the rally that just ended (for the pause banner)
const pausedRallyIndex = computed(() => {
  if (props.pausedAfterRallyId === null) return -1
  return props.rallies.findIndex(r => r.id === props.pausedAfterRallyId)
})

// Hover tooltip state
const hoveredRallyId = ref<number | null>(null)
const tooltipX = ref(0)

const hoveredRally = computed(() => {
  if (hoveredRallyId.value === null) return null
  return props.rallies.find(r => r.id === hoveredRallyId.value) ?? null
})

const hoveredRallySpeed = computed(() => {
  if (hoveredRallyId.value === null || !props.rallySpeedStats) return null
  const stats = props.rallySpeedStats.find(s => s.rallyId === hoveredRallyId.value)
  if (!stats || !stats.reliable) return null
  return stats
})

function onSegmentMouseEnter(seg: typeof videoTimeline.value[0], event: MouseEvent) {
  hoveredRallyId.value = seg.id
  updateTooltipPosition(event)
}

function onSegmentMouseMove(event: MouseEvent) {
  updateTooltipPosition(event)
}

function onSegmentMouseLeave() {
  hoveredRallyId.value = null
}

function updateTooltipPosition(event: MouseEvent) {
  const bar = (event.currentTarget as HTMLElement)?.closest('.rally-tl-bar')
  if (!bar) return
  const rect = bar.getBoundingClientRect()
  tooltipX.value = event.clientX - rect.left
}

function getPlayerColor(index: number): string {
  return PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B'
}
</script>

<template>
  <div class="rally-tl">
    <div class="rally-tl-header">
      <span class="rally-tl-label">Rallies</span>
      <span class="rally-tl-count">{{ rallies.length }} rallies detected</span>
      <span
        class="rally-tl-source client"
        title="Rallies detected from shuttle direction changes"
      >Auto</span>
      <button
        v-if="selectedRallyId !== null"
        class="rally-tl-clear"
        @click="selectedRallyId = null; emit('selectRally', null)"
      >
        Clear
      </button>

      <label class="rally-tl-autopause" :title="pauseBetweenRallies ? 'Auto-pause between rallies is ON' : 'Auto-pause between rallies is OFF'">
        <input
          type="checkbox"
          :checked="pauseBetweenRallies"
          @change="emit('update:pauseBetweenRallies', ($event.target as HTMLInputElement).checked)"
        />
        <span class="rally-tl-autopause-slider" />
        <span class="rally-tl-autopause-label">Auto-pause</span>
      </label>

      <label class="rally-tl-autopause" :title="pauseBetweenShots ? 'Auto-pause between shots is ON' : 'Auto-pause between shots is OFF'">
        <input
          type="checkbox"
          :checked="pauseBetweenShots"
          @change="emit('update:pauseBetweenShots', ($event.target as HTMLInputElement).checked)"
        />
        <span class="rally-tl-autopause-slider" />
        <span class="rally-tl-autopause-label">Pause Between Shots</span>
      </label>

      <select
        v-if="pauseBetweenShots"
        class="rally-tl-shot-duration"
        :value="shotPauseDurationSec"
        @change="emit('update:shotPauseDurationSec', Number(($event.target as HTMLSelectElement).value) as 1 | 1.5 | 2 | 3)"
        :title="'Pause duration after each shot'"
      >
        <option :value="1">1s</option>
        <option :value="1.5">1.5s</option>
        <option :value="2">2s</option>
        <option :value="3">3s</option>
      </select>
    </div>

    <!-- Pause banner -->
    <Transition name="rally-pause-fade">
      <div v-if="rallyPauseCountdown > 0 && pausedRallyIndex >= 0" class="rally-pause-banner">
        <span class="rally-pause-text">
          Rally #{{ pausedAfterRallyId }} ended
          <span class="rally-pause-dot" />
          Next rally in {{ rallyPauseCountdown }}s
        </span>
        <div class="rally-pause-actions">
          <button class="rally-pause-btn resume" @click="emit('resumeFromPause')">
            Resume
          </button>
          <button class="rally-pause-btn skip" @click="emit('skipToNextRally')">
            Skip
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">
              <polygon points="5 4 15 12 5 20 5 4" />
              <line x1="19" y1="5" x2="19" y2="19" />
            </svg>
          </button>
        </div>
      </div>
    </Transition>

    <div class="rally-tl-bar" @click.self="selectedRallyId = null">
      <div
        v-for="seg in videoTimeline"
        :key="seg.id"
        class="rally-tl-segment"
        :class="{
          active: currentRallyId === seg.id,
          selected: selectedRallyId === seg.id,
          dimmed: selectedRallyId !== null && selectedRallyId !== seg.id,
          'just-ended': pausedAfterRallyId === seg.id && rallyPauseCountdown > 0,
        }"
        :style="{
          left: seg.leftPct + '%',
          width: Math.max(seg.widthPct, 0.4) + '%',
        }"
        @click="selectRally(seg)"
        @mouseenter="onSegmentMouseEnter(seg, $event)"
        @mousemove="onSegmentMouseMove"
        @mouseleave="onSegmentMouseLeave"
      />
      <div class="rally-tl-playhead" :style="{ left: playbackPct + '%' }" />
    </div>

    <!-- Hover tooltip (outside bar to avoid overflow:hidden clipping) -->
    <div class="rally-tooltip-anchor">
      <Transition name="tooltip-fade">
        <div
          v-if="hoveredRallyId !== null"
          class="rally-tooltip"
          :style="{ left: tooltipX + 'px' }"
        >
          <div class="rally-tooltip-header">
            <span class="rally-tooltip-title">Rally #{{ hoveredRallyId }}</span>
            <span class="rally-tooltip-time">{{ formatTime(hoveredRally?.startTimestamp ?? 0) }}</span>
          </div>
          <div class="rally-tooltip-stats">
            <div class="rally-tooltip-row">
              <span class="rally-tooltip-label">Duration</span>
              <span class="rally-tooltip-value">{{ (hoveredRally?.durationSeconds ?? 0).toFixed(1) }}s</span>
            </div>
            <div class="rally-tooltip-row">
              <span class="rally-tooltip-label">Shots</span>
              <span class="rally-tooltip-value">{{ hoveredRally?.shotCount ?? 0 }}</span>
            </div>
          </div>
          <template v-if="hoveredRallySpeed">
            <div class="rally-tooltip-divider" />
            <div
              v-for="p in hoveredRallySpeed.players"
              :key="p.playerId"
              class="rally-tooltip-player"
            >
              <span class="rally-tooltip-player-dot" :style="{ background: getPlayerColor(p.playerId) }" />
              <span class="rally-tooltip-player-label">P{{ pidDisplayFor(p.playerId) + 1 }}</span>
              <div class="rally-tooltip-player-stats">
                <span>{{ p.avgSpeed.toFixed(1) }} <small>avg km/h</small></span>
                <span>{{ p.maxSpeed.toFixed(1) }} <small>max km/h</small></span>
                <span>{{ p.distanceCovered.toFixed(1) }}<small>m</small></span>
              </div>
            </div>
          </template>
        </div>
      </Transition>
    </div>
    <div class="rally-tl-labels">
      <span>0:00</span>
      <span>{{ formatTime(duration / 2) }}</span>
      <span>{{ formatTime(duration) }}</span>
    </div>

    <!-- Combined comparison bar -->
    <div v-if="backendRallies.length > 0" class="rally-tl-combined">
      <div class="rally-tl-combined-header">
        <span class="rally-tl-label">Combined</span>
        <span class="rally-tl-count">client + backend</span>
        <div class="rally-tl-legend">
          <span class="rally-tl-legend-item both">Both</span>
          <span class="rally-tl-legend-item client-only">Client</span>
          <span class="rally-tl-legend-item backend-only">Backend</span>
        </div>
      </div>
      <div class="rally-tl-bar combined-bar">
        <div
          v-for="(seg, i) in combinedTimeline"
          :key="i"
          class="rally-tl-combined-segment"
          :class="seg.kind"
          :style="{
            left: seg.leftPct + '%',
            width: Math.max(seg.widthPct, 0.4) + '%',
          }"
        />
        <div class="rally-tl-playhead" :style="{ left: playbackPct + '%' }" />
      </div>
      <div class="rally-tl-labels">
        <span>0:00</span>
        <span>{{ formatTime(duration / 2) }}</span>
        <span>{{ formatTime(duration) }}</span>
      </div>
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

.rally-tl-source {
  font-size: 0.6rem;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.rally-tl-source.client {
  color: var(--color-warning, #f59e0b);
  background: color-mix(in srgb, var(--color-warning, #f59e0b) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--color-warning, #f59e0b) 25%, transparent);
}

.rally-tl-clear {
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

/* Auto-pause toggle in header */
.rally-tl-autopause {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  user-select: none;
}

.rally-tl-autopause input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}

.rally-tl-autopause-slider {
  position: relative;
  width: 26px;
  height: 14px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 7px;
  transition: all 0.2s;
  flex-shrink: 0;
}

.rally-tl-autopause-slider::after {
  content: '';
  position: absolute;
  top: 1px;
  left: 1px;
  width: 10px;
  height: 10px;
  background: var(--color-text-tertiary);
  border-radius: 50%;
  transition: all 0.2s;
}

.rally-tl-autopause input:checked + .rally-tl-autopause-slider {
  background: var(--color-accent);
  border-color: var(--color-accent);
}

.rally-tl-autopause input:checked + .rally-tl-autopause-slider::after {
  left: 13px;
  background: #fff;
}

.rally-tl-autopause-label {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
  transition: color 0.15s;
}

.rally-tl-autopause:hover .rally-tl-autopause-label {
  color: var(--color-text-secondary);
}

.rally-tl-autopause input:checked ~ .rally-tl-autopause-label {
  color: var(--color-accent);
}

.rally-tl-shot-duration {
  padding: 2px 6px;
  margin-left: 6px;
  background: rgba(255, 255, 255, 0.06);
  color: inherit;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}

/* Pause banner between header and timeline bar */
.rally-pause-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 10px;
  margin-bottom: 6px;
  background: color-mix(in srgb, var(--color-accent) 10%, var(--color-bg-secondary));
  border: 1px solid color-mix(in srgb, var(--color-accent) 30%, var(--color-border));
  border-radius: 4px;
}

.rally-pause-text {
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.rally-pause-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--color-accent);
  animation: rally-pause-pulse 1s ease-in-out infinite;
}

@keyframes rally-pause-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

.rally-pause-actions {
  display: flex;
  gap: 6px;
}

.rally-pause-btn {
  padding: 3px 10px;
  font-size: 0.65rem;
  font-weight: 500;
  border: 1px solid var(--color-border-secondary);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  gap: 3px;
  border-radius: 3px;
}

.rally-pause-btn:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.rally-pause-btn.skip {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: #fff;
}

.rally-pause-btn.skip:hover {
  filter: brightness(1.1);
  color: #fff;
}

/* Transition for pause banner */
.rally-pause-fade-enter-active,
.rally-pause-fade-leave-active {
  transition: all 0.2s ease;
}

.rally-pause-fade-enter-from,
.rally-pause-fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

/* Timeline bar */
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

.rally-tl-segment.just-ended {
  opacity: 1;
  background: color-mix(in srgb, var(--color-accent) 70%, var(--color-text-heading));
  animation: rally-ended-glow 1s ease-in-out infinite;
}

@keyframes rally-ended-glow {
  0%, 100% { box-shadow: 0 0 0 0 transparent; }
  50% { box-shadow: 0 0 4px 1px color-mix(in srgb, var(--color-accent) 40%, transparent); }
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

/* Tooltip */
.rally-tooltip-anchor {
  position: relative;
  height: 0;
}

.rally-tooltip {
  position: absolute;
  bottom: 0;
  transform: translateX(-50%);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-hover);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  padding: 8px 10px;
  pointer-events: none;
  z-index: 10;
  min-width: 160px;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.rally-tooltip-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 6px;
}

.rally-tooltip-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-heading);
}

.rally-tooltip-time {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
}

.rally-tooltip-stats {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.rally-tooltip-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}

.rally-tooltip-label {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
}

.rally-tooltip-value {
  font-size: 0.65rem;
  color: var(--color-text-secondary);
  font-weight: 500;
}

.rally-tooltip-divider {
  height: 1px;
  background: var(--color-border);
  margin: 5px 0;
}

.rally-tooltip-player {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 3px;
}

.rally-tooltip-player-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.rally-tooltip-player-label {
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  min-width: 18px;
}

.rally-tooltip-player-stats {
  display: flex;
  gap: 8px;
  font-size: 0.6rem;
  color: var(--color-text-secondary);
}

.rally-tooltip-player-stats small {
  color: var(--color-text-tertiary);
  margin-left: 1px;
}

/* Tooltip transitions */
.tooltip-fade-enter-active {
  transition: opacity 0.1s ease;
}

.tooltip-fade-leave-active {
  transition: opacity 0.05s ease;
}

.tooltip-fade-enter-from,
.tooltip-fade-leave-to {
  opacity: 0;
}

/* Combined comparison timeline */
.rally-tl-combined {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
}

.rally-tl-combined-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.rally-tl-legend {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.rally-tl-legend-item {
  font-size: 0.6rem;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 500;
}

.rally-tl-legend-item.both {
  color: #10b981;
  background: color-mix(in srgb, #10b981 12%, transparent);
  border: 1px solid color-mix(in srgb, #10b981 25%, transparent);
}

.rally-tl-legend-item.client-only {
  color: #f59e0b;
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  border: 1px solid color-mix(in srgb, #f59e0b 25%, transparent);
}

.rally-tl-legend-item.backend-only {
  color: #3b82f6;
  background: color-mix(in srgb, #3b82f6 12%, transparent);
  border: 1px solid color-mix(in srgb, #3b82f6 25%, transparent);
}

.combined-bar {
  height: 20px;
}

.rally-tl-combined-segment {
  position: absolute;
  top: 3px;
  bottom: 3px;
  min-width: 3px;
  opacity: 0.75;
  transition: opacity 0.15s;
}

.rally-tl-combined-segment:hover {
  opacity: 1;
}

.rally-tl-combined-segment.both {
  background: #10b981;
}

.rally-tl-combined-segment.client {
  background: #f59e0b;
}

.rally-tl-combined-segment.backend {
  background: #3b82f6;
}
</style>
