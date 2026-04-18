<script setup lang="ts">
/**
 * ShotSpeedList Component
 * 
 * Analyzes skeleton data to detect shot events and calculate player movement
 * speed between shots. Shows a clickable list of shots in each rally with
 * max and average speed of the responding player's movement.
 * 
 * The key insight from the client: the most valuable speed metric is the
 * movement speed between when the opponent hits the shuttle and when the
 * player hits the shuttle ("reaction burst speed").
 * 
 * Shot detection uses shuttle position trajectory analysis:
 * 1. Track shuttle positions across frames
 * 2. Detect direction changes (indicating a hit)
 * 3. Associate each hit with the nearest player
 * 4. Calculate movement speeds between consecutive hits
 */

import { ref, computed, watch } from 'vue'
import type {
  SkeletonFrame,
  SpeedZone,
} from '@/types/analysis'
import {
  SPEED_ZONE_COLORS,
  SPEED_ZONE_NAMES,
  PLAYER_SPEED_COLORS,
} from '@/types/analysis'
import {
  useShotSegments,
  type ShotEvent,
  type ShotMovementSegment,
} from '@/composables/useShotSegments'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  skeletonData: SkeletonFrame[]
  fps: number
  visible?: boolean
  courtKeypointsSet?: boolean
  speedCalculated?: boolean
}>()

const emit = defineEmits<{
  (e: 'seekToSegment', startTime: number, endTime: number): void
  (e: 'close'): void
}>()

// =============================================================================
// STATE
// =============================================================================

const useKmh = ref(true)
const selectedSegmentId = ref<number | null>(null)
const filterPlayer = ref<number | null>(null) // null = show all players
const sortBy = ref<'time' | 'maxSpeed' | 'avgSpeed'>('time')
const sortDesc = ref(false)

// =============================================================================
// SHOT EVENTS & MOVEMENT SEGMENTS (via composable)
// =============================================================================

const skeletonDataRef = computed(() => props.skeletonData)
const { shotEvents, segments: movementSegments } = useShotSegments(skeletonDataRef)

// =============================================================================
// COMPUTED: Filtered & sorted segments
// =============================================================================

const filteredSegments = computed(() => {
  let result = movementSegments.value
  
  // Filter by player
  if (filterPlayer.value !== null) {
    result = result.filter(s => s.movingPlayerId === filterPlayer.value)
  }
  
  // Sort
  const sorted = [...result]
  switch (sortBy.value) {
    case 'time':
      sorted.sort((a, b) => a.startTimestamp - b.startTimestamp)
      break
    case 'maxSpeed':
      sorted.sort((a, b) => b.maxSpeedKmh - a.maxSpeedKmh)
      break
    case 'avgSpeed':
      sorted.sort((a, b) => b.avgSpeedKmh - a.avgSpeedKmh)
      break
  }
  
  if (sortDesc.value && sortBy.value === 'time') {
    sorted.reverse()
  }
  
  return sorted
})

// Available player IDs
const playerIds = computed(() => {
  const ids = new Set<number>()
  for (const segment of movementSegments.value) {
    ids.add(segment.movingPlayerId)
  }
  return Array.from(ids).sort()
})

// Summary statistics
const summaryStats = computed(() => {
  const segments = filteredSegments.value
  if (segments.length === 0) return null
  
  return {
    totalShots: segments.length,
    overallMaxSpeed: Math.max(...segments.map(s => s.maxSpeedKmh)),
    overallAvgSpeed: segments.reduce((sum, s) => sum + s.avgSpeedKmh, 0) / segments.length,
    avgReactionTime: segments.reduce((sum, s) => sum + s.durationSeconds, 0) / segments.length,
    fastestSegment: segments.reduce((max, s) => s.maxSpeedKmh > max.maxSpeedKmh ? s : max, segments[0]!),
  }
})

// =============================================================================
// METHODS
// =============================================================================

function handleSegmentClick(segment: ShotMovementSegment) {
  selectedSegmentId.value = segment.id
  // Seek to 0.5s before the segment starts for context
  const startTime = Math.max(0, segment.startTimestamp - 0.5)
  emit('seekToSegment', startTime, segment.endTimestamp + 0.5)
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  const tenths = Math.floor((seconds % 1) * 10)
  return `${mins}:${secs.toString().padStart(2, '0')}.${tenths}`
}

function formatSpeed(kmh: number): string {
  if (useKmh.value) {
    return kmh.toFixed(1)
  }
  return (kmh / 3.6).toFixed(2)
}

function getSpeedUnit(): string {
  return useKmh.value ? 'km/h' : 'm/s'
}

function getPlayerColor(playerId: number): string {
  const idx = Math.max(0, playerId - 1) % PLAYER_SPEED_COLORS.length
  return PLAYER_SPEED_COLORS[idx] ?? '#3B82F6'
}

function getZoneColor(zone: SpeedZone): string {
  return SPEED_ZONE_COLORS[zone] ?? '#94A3B8'
}

function getZoneName(zone: SpeedZone): string {
  return SPEED_ZONE_NAMES[zone] ?? 'Unknown'
}

function toggleSort(field: 'time' | 'maxSpeed' | 'avgSpeed') {
  if (sortBy.value === field) {
    sortDesc.value = !sortDesc.value
  } else {
    sortBy.value = field
    sortDesc.value = false
  }
}

/**
 * Generate a mini sparkline SVG path for the speed profile.
 */
function getSparklinePath(speedProfile: number[], width: number = 60, height: number = 20): string {
  if (speedProfile.length < 2) return ''
  
  const maxSpeed = Math.max(...speedProfile, 1)
  const step = width / (speedProfile.length - 1)
  
  let path = `M 0 ${height - (speedProfile[0]! / maxSpeed) * height}`
  for (let i = 1; i < speedProfile.length; i++) {
    const x = i * step
    const y = height - (speedProfile[i]! / maxSpeed) * height
    path += ` L ${x} ${y}`
  }
  
  return path
}
</script>

<template>
  <div v-if="visible !== false" class="shot-speed-container">
    <!-- Header -->
    <div class="shot-speed-header">
      <div class="header-title">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        <span>Shot-by-Shot Movement Speed</span>
      </div>
      
      <div class="header-controls">
        <!-- Unit toggle -->
        <button 
          class="unit-toggle"
          :class="{ active: useKmh }"
          @click="useKmh = !useKmh"
          title="Toggle speed unit"
        >
          {{ useKmh ? 'km/h' : 'm/s' }}
        </button>
        
        <!-- Player filter -->
        <select 
          v-model="filterPlayer" 
          class="filter-select"
          title="Filter by player"
        >
          <option :value="null">All Players</option>
          <option v-for="pid in playerIds" :key="pid" :value="pid">
            Player {{ pid }}
          </option>
        </select>
        
        <!-- Close button -->
        <button 
          class="close-btn"
          @click="emit('close')"
          title="Close shot speed analysis"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>
    
    <!-- Summary Stats -->
    <div v-if="summaryStats" class="summary-stats">
      <div class="stat-card">
        <span class="stat-label">Shots</span>
        <span class="stat-value">{{ summaryStats.totalShots }}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Top Speed</span>
        <span class="stat-value highlight">
          {{ formatSpeed(summaryStats.overallMaxSpeed) }}
          <span class="stat-unit">{{ getSpeedUnit() }}</span>
        </span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Speed</span>
        <span class="stat-value">
          {{ formatSpeed(summaryStats.overallAvgSpeed) }}
          <span class="stat-unit">{{ getSpeedUnit() }}</span>
        </span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Avg Reaction</span>
        <span class="stat-value">
          {{ summaryStats.avgReactionTime.toFixed(1) }}
          <span class="stat-unit">s</span>
        </span>
      </div>
    </div>
    
    <!-- Info banner explaining the metric -->
    <div class="info-banner">
      <svg xmlns="http://www.w3.org/2000/svg" class="icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 16v-4M12 8h.01"/>
      </svg>
      <span>
        Speed measured between opponent's hit and player's response hit.
        <strong>Click any row to replay that segment.</strong>
      </span>
    </div>
    
    <!-- Column Headers (sortable) -->
    <div class="list-header">
      <span class="col-num">#</span>
      <span class="col-time sortable" @click="toggleSort('time')">
        Time
        <span v-if="sortBy === 'time'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
      <span class="col-player">Player</span>
      <span class="col-duration">Duration</span>
      <span class="col-sparkline">Profile</span>
      <span class="col-speed sortable" @click="toggleSort('maxSpeed')">
        Max
        <span v-if="sortBy === 'maxSpeed'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
      <span class="col-speed sortable" @click="toggleSort('avgSpeed')">
        Avg
        <span v-if="sortBy === 'avgSpeed'" class="sort-indicator">{{ sortDesc ? '↓' : '↑' }}</span>
      </span>
    </div>
    
    <!-- Shot List -->
    <div class="shot-list" v-if="filteredSegments.length > 0">
      <div
        v-for="segment in filteredSegments"
        :key="segment.id"
        class="shot-row"
        :class="{ selected: selectedSegmentId === segment.id }"
        @click="handleSegmentClick(segment)"
      >
        <span class="col-num">{{ segment.id + 1 }}</span>
        
        <span class="col-time">
          {{ formatTime(segment.startTimestamp) }}
        </span>
        
        <span class="col-player">
          <span 
            class="player-badge"
            :style="{ backgroundColor: getPlayerColor(segment.movingPlayerId) }"
          >
            P{{ segment.movingPlayerId }}
          </span>
        </span>
        
        <span class="col-duration">
          {{ segment.durationSeconds.toFixed(1) }}s
        </span>
        
        <span class="col-sparkline">
          <svg 
            v-if="segment.speedProfile.length > 1"
            :width="60" 
            :height="20" 
            class="sparkline"
          >
            <path 
              :d="getSparklinePath(segment.speedProfile)" 
              fill="none" 
              :stroke="getZoneColor(segment.maxSpeedZone)" 
              stroke-width="1.5"
            />
          </svg>
        </span>
        
        <span class="col-speed">
          <span 
            class="speed-badge"
            :style="{ backgroundColor: getZoneColor(segment.maxSpeedZone) + '33', color: getZoneColor(segment.maxSpeedZone) }"
          >
            {{ formatSpeed(segment.maxSpeedKmh) }}
          </span>
        </span>
        
        <span class="col-speed">
          <span class="speed-value">
            {{ formatSpeed(segment.avgSpeedKmh) }}
          </span>
        </span>
      </div>
    </div>
    
    <!-- Empty State -->
    <div v-else class="empty-state">
      <div v-if="!speedCalculated && !courtKeypointsSet" class="no-data pending">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
        <p>Set court keypoints and start video to enable shot analysis</p>
        <p class="hint">Speed calibration is needed for accurate measurements</p>
      </div>
      <div v-else class="no-data">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        <p>No shots detected</p>
        <p class="hint">Shot detection requires shuttle tracking data or pose classification</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.shot-speed-container {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 100%;
  overflow: hidden;
}

/* Header */
.shot-speed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border);
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-heading);
  font-weight: 600;
  font-size: 14px;
}

.header-title .icon {
  width: 18px;
  height: 18px;
  color: #f59e0b;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.unit-toggle {
  background: var(--color-bg-tertiary);
  color: #f59e0b;
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.unit-toggle:hover,
.unit-toggle.active {
  background: var(--color-border);
  border-color: #f59e0b;
}

.filter-select {
  background: var(--color-bg-tertiary);
  color: var(--color-text-heading);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.close-btn {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  padding: 4px;
  cursor: pointer;
  color: var(--color-text-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.close-btn:hover {
  background: var(--color-border);
  color: var(--color-text-heading);
  border-color: var(--color-border-hover);
}

.icon-sm {
  width: 14px;
  height: 14px;
}

.icon-xs {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

.icon-large {
  width: 32px;
  height: 32px;
  color: var(--color-text-tertiary);
  margin-bottom: 8px;
}

/* Summary Stats */
.summary-stats {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.stat-card {
  flex: 1;
  min-width: 70px;
  background: var(--color-bg-tertiary);
  padding: 8px 10px;
  text-align: center;
}

.stat-label {
  display: block;
  color: var(--color-text-tertiary);
  font-size: 10px;
  text-transform: uppercase;
  margin-bottom: 2px;
}

.stat-value {
  display: block;
  color: var(--color-text-heading);
  font-size: 14px;
  font-weight: 600;
}

.stat-value.highlight {
  color: #f59e0b;
}

.stat-unit {
  font-size: 10px;
  opacity: 0.7;
  font-weight: normal;
}

/* Info Banner */
.info-banner {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  font-size: 11px;
  color: var(--color-text-secondary);
}

.info-banner strong {
  color: var(--color-text);
}

/* Column Headers */
.list-header {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  background: var(--color-bg-tertiary);
  border-bottom: 1px solid var(--color-border-secondary);
  font-size: 10px;
  text-transform: uppercase;
  color: var(--color-text-tertiary);
  font-weight: 600;
  gap: 4px;
}

.sortable {
  cursor: pointer;
  user-select: none;
}

.sortable:hover {
  color: #f59e0b;
}

.sort-indicator {
  font-size: 10px;
  margin-left: 2px;
}

.col-num {
  width: 28px;
  text-align: center;
  flex-shrink: 0;
}

.col-time {
  width: 65px;
  flex-shrink: 0;
}

.col-player {
  width: 40px;
  flex-shrink: 0;
}

.col-duration {
  width: 50px;
  flex-shrink: 0;
  text-align: center;
}

.col-sparkline {
  flex: 1;
  min-width: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.col-speed {
  width: 52px;
  flex-shrink: 0;
  text-align: right;
}

/* Shot List */
.shot-list {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.shot-row {
  display: flex;
  align-items: center;
  padding: 6px 8px;
  gap: 4px;
  cursor: pointer;
  transition: background 0.15s;
  border-bottom: 1px solid var(--color-bg-tertiary);
  font-size: 12px;
  color: #ccc;
}

.shot-row:hover {
  background: #1a1a2e;
}

.shot-row.selected {
  background: #1a2a1a;
  border-left: 2px solid var(--color-accent);
  padding-left: 6px;
}

.player-badge {
  display: inline-block;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  color: var(--color-text-heading);
  border-radius: 2px;
}

.speed-badge {
  display: inline-block;
  padding: 2px 6px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 2px;
}

.speed-value {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.sparkline {
  display: block;
}

/* Empty state */
.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.no-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  color: var(--color-text-tertiary);
  gap: 4px;
  text-align: center;
}

.no-data p {
  margin: 2px 0;
  font-size: 12px;
}

.no-data .hint {
  font-size: 11px;
  opacity: 0.7;
}

/* Scrollbar styling */
.shot-list::-webkit-scrollbar {
  width: 6px;
}

.shot-list::-webkit-scrollbar-track {
  background: var(--color-bg-input);
}

.shot-list::-webkit-scrollbar-thumb {
  background: var(--color-border-secondary);
  border-radius: 3px;
}

.shot-list::-webkit-scrollbar-thumb:hover {
  background: var(--color-border-hover);
}
</style>
