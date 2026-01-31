<script setup lang="ts">
/**
 * SpeedGraph Component
 * 
 * Real-time speed visualization for badminton player movement analysis.
 * Displays a dual-line chart showing Player 1 and Player 2 speeds over time,
 * with speed zone color coding and statistics.
 * 
 * Features:
 * - Dual-line chart with distinct player colors
 * - Speed zone background bands for context
 * - Current, max, and average speed statistics
 * - Toggle between m/s and km/h units
 * - Sliding window for real-time display
 * - Speed zone indicators with color coding
 */

import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  type ChartData,
  type ChartOptions
} from 'chart.js'
import annotationPlugin from 'chartjs-plugin-annotation'
import type {
  SkeletonFrame,
  SpeedZone,
  PlayerSpeedStats
} from '@/types/analysis'
import {
  SPEED_ZONE_COLORS,
  SPEED_ZONE_NAMES,
  SPEED_ZONE_THRESHOLDS,
  PLAYER_SPEED_COLORS
} from '@/types/analysis'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
)

// Props
const props = defineProps<{
  skeletonData: SkeletonFrame[]
  fps: number
  currentFrame: number
  visible?: boolean
  isSpeedCalculating?: boolean
  speedCalculated?: boolean
  courtKeypointsSet?: boolean
  videoStarted?: boolean
}>()

// Emits
const emit = defineEmits<{
  (e: 'close'): void
}>()

// State
const useKmh = ref(true) // Toggle between km/h and m/s
const windowSeconds = ref(30) // Sliding window size
const showZoneBands = ref(true) // Show speed zone background bands
const showStatistics = ref(true) // Show statistics panel

// Computed: Convert skeleton data to speed timeline
const speedTimeline = computed(() => {
  const players: Record<number, {
    timestamps: number[]
    speeds: number[]  // Always stored in m/s internally
    zones: SpeedZone[]
  }> = {}
  
  // Guard against undefined/null/empty skeleton data
  if (!props.skeletonData || props.skeletonData.length === 0) {
    console.log('[SpeedGraph] No skeleton data available')
    return players
  }
  
  console.log('[SpeedGraph] Processing', props.skeletonData.length, 'frames')
  
  // Debug: Sample some frames to check speed values
  let nonZeroSpeedCount = 0
  let totalSpeedSamples = 0
  let maxSpeedSeen = 0
  
  for (const frame of props.skeletonData) {
    if (!frame.players || frame.players.length === 0) continue
    
    for (const player of frame.players) {
      const playerId = player.player_id
      
      if (!players[playerId]) {
        players[playerId] = {
          timestamps: [],
          speeds: [],
          zones: []
        }
      }
      
      // Use current_speed from skeleton data (already in km/h)
      const speedKmh = player.current_speed ?? 0
      const speedMps = speedKmh / 3.6
      
      // Debug tracking
      totalSpeedSamples++
      if (speedKmh > 0) nonZeroSpeedCount++
      if (speedKmh > maxSpeedSeen) maxSpeedSeen = speedKmh
      
      // Classify into zone
      let zone: SpeedZone = 'standing'
      for (const [zoneName, thresholds] of Object.entries(SPEED_ZONE_THRESHOLDS)) {
        const maxSpeed = thresholds.max ?? Infinity
        if (speedMps >= thresholds.min && speedMps < maxSpeed) {
          zone = zoneName as SpeedZone
          break
        }
      }
      
      players[playerId].timestamps.push(frame.timestamp)
      players[playerId].speeds.push(speedMps)
      players[playerId].zones.push(zone)
    }
  }
  
  // Debug: Log speed data summary
  console.log('[SpeedGraph] Speed data summary:')
  console.log(`  - Players found: ${Object.keys(players).length}`)
  console.log(`  - Total speed samples: ${totalSpeedSamples}`)
  console.log(`  - Non-zero speeds: ${nonZeroSpeedCount} (${(nonZeroSpeedCount/totalSpeedSamples*100).toFixed(1)}%)`)
  console.log(`  - Max speed seen: ${maxSpeedSeen.toFixed(1)} km/h`)
  
  // Debug: Log first few non-zero speeds
  if (nonZeroSpeedCount === 0) {
    console.warn('[SpeedGraph] WARNING: All speeds are 0! Check if speed calculation was triggered.')
    console.log('[SpeedGraph] Speed recalculation props:', {
      isSpeedCalculating: props.isSpeedCalculating,
      speedCalculated: props.speedCalculated,
      courtKeypointsSet: props.courtKeypointsSet,
      videoStarted: props.videoStarted
    })
  }
  
  return players
})

// Computed: Statistics for each player
const playerStats = computed(() => {
  const stats: Record<number, {
    current_mps: number
    current_kmh: number
    max_mps: number
    max_kmh: number
    avg_mps: number
    avg_kmh: number
    currentZone: SpeedZone
  }> = {}
  
  for (const [playerIdStr, data] of Object.entries(speedTimeline.value)) {
    const playerId = parseInt(playerIdStr)
    const speeds = data.speeds
    const timestamps = data.timestamps
    
    if (speeds.length === 0 || timestamps.length === 0) continue
    
    // Find current speed (closest to current frame timestamp)
    const currentTime = props.currentFrame / props.fps
    let closestIdx = 0
    let minDiff = Infinity
    
    for (let i = 0; i < timestamps.length; i++) {
      const timestamp = timestamps[i]
      if (timestamp !== undefined) {
        const diff = Math.abs(timestamp - currentTime)
        if (diff < minDiff) {
          minDiff = diff
          closestIdx = i
        }
      }
    }
    
    const currentMps = speeds[closestIdx] ?? 0
    const maxMps = Math.max(...speeds)
    const avgMps = speeds.reduce((a, b) => a + b, 0) / speeds.length
    
    stats[playerId] = {
      current_mps: currentMps,
      current_kmh: currentMps * 3.6,
      max_mps: maxMps,
      max_kmh: maxMps * 3.6,
      avg_mps: avgMps,
      avg_kmh: avgMps * 3.6,
      currentZone: data.zones[closestIdx] ?? 'standing'
    }
  }
  
  return stats
})

// Computed: Chart data with sliding window
const chartData = computed<ChartData<'line'>>(() => {
  const currentTime = props.currentFrame / props.fps
  
  // Handle edge case: when video hasn't started (currentFrame = 0),
  // show data from start up to windowSeconds
  const effectiveCurrentTime = currentTime <= 0.1 ? windowSeconds.value : currentTime
  const windowStart = Math.max(0, effectiveCurrentTime - windowSeconds.value)
  const windowEnd = effectiveCurrentTime
  
  const datasets: any[] = []
  const allTimestamps = new Set<number>()
  
  // Check if there's any data in speedTimeline
  const playerEntries = Object.entries(speedTimeline.value)
  if (playerEntries.length === 0) {
    console.log('[SpeedGraph] No players in speedTimeline')
    return { labels: [], datasets: [] }
  }
  
  // Collect all timestamps within window
  for (const [, data] of playerEntries) {
    for (let i = 0; i < data.timestamps.length; i++) {
      const time = data.timestamps[i]
      if (time !== undefined && time >= windowStart && time <= windowEnd) {
        allTimestamps.add(time)
      }
    }
  }
  
  // If no timestamps in window, try to show all available data (up to first windowSeconds)
  if (allTimestamps.size === 0) {
    console.log('[SpeedGraph] No timestamps in window, showing all available data')
    for (const [, data] of playerEntries) {
      for (let i = 0; i < Math.min(data.timestamps.length, 300); i++) { // Limit to 300 points
        const time = data.timestamps[i]
        if (time !== undefined) {
          allTimestamps.add(time)
        }
      }
    }
  }
  
  // Sort timestamps for labels
  const labels = Array.from(allTimestamps).sort((a, b) => a - b)
  
  console.log('[SpeedGraph] Chart has', labels.length, 'data points')
  
  // Create dataset for each player
  for (const [playerIdStr, data] of playerEntries) {
    const playerId = parseInt(playerIdStr)
    const colorIndex = Math.max(0, playerId - 1) % PLAYER_SPEED_COLORS.length
    const color = PLAYER_SPEED_COLORS[colorIndex]
    
    // Map data to labels
    const speedData = labels.map(time => {
      const idx = data.timestamps.findIndex(t => Math.abs(t - time) < 0.001)
      if (idx >= 0) {
        const speed = data.speeds[idx]
        if (speed !== undefined) {
          return useKmh.value ? speed * 3.6 : speed
        }
      }
      return null
    })
    
    datasets.push({
      label: `Player ${playerId}`,
      data: speedData,
      borderColor: color,
      backgroundColor: color + '33', // 20% opacity
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.3, // Smooth curves
      fill: false
    })
  }
  
  return {
    labels: labels.map(t => t.toFixed(1) + 's'),
    datasets
  }
})

// Computed: Find the maximum speed in the data for dynamic Y-axis scaling
const maxSpeedInData = computed(() => {
  let maxSpeed = 0
  for (const [, data] of Object.entries(speedTimeline.value)) {
    for (const speed of data.speeds) {
      if (speed !== undefined && speed > maxSpeed) {
        maxSpeed = speed
      }
    }
  }
  return maxSpeed
})

// Computed: Chart options
const chartOptions = computed<ChartOptions<'line'>>(() => {
  const unit = useKmh.value ? 'km/h' : 'm/s'
  
  // Dynamic Y-axis max: scale based on actual data for better visibility
  // Use 20% above max recorded speed, but at least show up to jogging zone
  const maxRecordedSpeed = maxSpeedInData.value
  const minReasonableMax = 5.0 // m/s (~18 km/h) - at least show jogging zone
  const dynamicMax = Math.max(minReasonableMax, maxRecordedSpeed * 1.3) // 30% headroom above max
  const maxSpeed = useKmh.value ? dynamicMax * 3.6 : dynamicMax
  
  // Build annotation lines for zones if enabled
  const annotations: any = {}
  
  if (showZoneBands.value) {
    const zones: SpeedZone[] = ['walking', 'jogging', 'running', 'sprinting', 'explosive']
    
    for (const zone of zones) {
      const thresholds = SPEED_ZONE_THRESHOLDS[zone]
      const yMin = useKmh.value ? thresholds.min * 3.6 : thresholds.min
      const yMax = thresholds.max 
        ? (useKmh.value ? thresholds.max * 3.6 : thresholds.max) 
        : maxSpeed
      
      annotations[`zone_${zone}`] = {
        type: 'box',
        yMin: yMin,
        yMax: yMax,
        backgroundColor: SPEED_ZONE_COLORS[zone] + '15', // Very transparent
        borderWidth: 0
      }
    }
  }
  
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Disable animation for real-time updates
    },
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#E2E8F0',
          font: {
            size: 11
          },
          boxWidth: 12,
          padding: 8
        }
      },
      tooltip: {
        backgroundColor: 'rgba(30, 41, 59, 0.95)',
        titleColor: '#F1F5F9',
        bodyColor: '#E2E8F0',
        padding: 10,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const value = context.raw as number
            if (value === null) return ''
            return `${context.dataset.label}: ${value.toFixed(1)} ${unit}`
          }
        }
      },
      annotation: {
        annotations
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time',
          color: '#94A3B8',
          font: { size: 11 }
        },
        ticks: {
          color: '#94A3B8',
          maxTicksLimit: 8,
          font: { size: 10 }
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: `Speed (${unit})`,
          color: '#94A3B8',
          font: { size: 11 }
        },
        min: 0,
        max: maxSpeed,
        ticks: {
          color: '#94A3B8',
          font: { size: 10 },
          callback: (value) => `${value}`
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.15)'
        }
      }
    }
  }
})

// Format speed value
const formatSpeed = (mps: number): string => {
  if (useKmh.value) {
    return (mps * 3.6).toFixed(1)
  }
  return mps.toFixed(2)
}

// Get zone color
const getZoneColor = (zone: SpeedZone): string => {
  return SPEED_ZONE_COLORS[zone] || '#94A3B8'
}

// Get zone name
const getZoneName = (zone: SpeedZone): string => {
  return SPEED_ZONE_NAMES[zone] || 'Unknown'
}
</script>

<template>
  <div 
    v-if="visible !== false"
    class="speed-graph-container"
  >
    <!-- Header -->
    <div class="speed-graph-header">
      <div class="header-title">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
        <span>Player Speed</span>
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
        
        <!-- Window size selector -->
        <select 
          v-model="windowSeconds" 
          class="window-select"
          title="Time window"
        >
          <option :value="15">15s</option>
          <option :value="30">30s</option>
          <option :value="60">60s</option>
        </select>
        
        <!-- Zone bands toggle -->
        <button 
          class="toggle-btn"
          :class="{ active: showZoneBands }"
          @click="showZoneBands = !showZoneBands"
          title="Toggle zone bands"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 6h18M3 12h18M3 18h18"/>
          </svg>
        </button>
        
        <!-- Close button -->
        <button 
          class="close-btn"
          @click="emit('close')"
          title="Close speed graph"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>
    
    <!-- Statistics Panel -->
    <div v-if="showStatistics && Object.keys(playerStats).length > 0" class="stats-panel">
      <div
        v-for="(stats, playerId) in playerStats"
        :key="playerId"
        class="player-stats"
        :style="{ borderColor: PLAYER_SPEED_COLORS[Math.max(0, Number(playerId) - 1) % PLAYER_SPEED_COLORS.length] }"
      >
        <div class="player-label">
          <span
            class="player-dot"
            :style="{ backgroundColor: PLAYER_SPEED_COLORS[Math.max(0, Number(playerId) - 1) % PLAYER_SPEED_COLORS.length] }"
          ></span>
          Player {{ playerId }}
        </div>
        
        <div class="stats-grid">
          <!-- Current Speed with Zone -->
          <div class="stat-item current">
            <span class="stat-label">Current</span>
            <span 
              class="stat-value zone-badge"
              :style="{ backgroundColor: getZoneColor(stats.currentZone) }"
            >
              {{ formatSpeed(stats.current_mps) }}
              <span class="stat-unit">{{ useKmh ? 'km/h' : 'm/s' }}</span>
            </span>
            <span class="zone-name">{{ getZoneName(stats.currentZone) }}</span>
          </div>
          
          <!-- Max Speed -->
          <div class="stat-item">
            <span class="stat-label">Max</span>
            <span class="stat-value">
              {{ formatSpeed(stats.max_mps) }}
              <span class="stat-unit">{{ useKmh ? 'km/h' : 'm/s' }}</span>
            </span>
          </div>
          
          <!-- Average Speed -->
          <div class="stat-item">
            <span class="stat-label">Avg</span>
            <span class="stat-value">
              {{ formatSpeed(stats.avg_mps) }}
              <span class="stat-unit">{{ useKmh ? 'km/h' : 'm/s' }}</span>
            </span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Chart -->
    <div class="chart-container">
      <!-- Loading state -->
      <div v-if="isSpeedCalculating" class="no-data calculating">
        <svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-linecap="round" />
        </svg>
        <p>Calculating player speeds...</p>
        <p class="hint">Using court calibration for accurate measurements</p>
      </div>
      
      <!-- Waiting for prerequisites -->
      <div v-else-if="!speedCalculated && !isSpeedCalculating" class="no-data pending">
        <svg xmlns="http://www.w3.org/2000/svg" class="icon-large" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
        <p>Speed calculation pending</p>
        <div class="prerequisites">
          <div class="prereq-item" :class="{ done: courtKeypointsSet }">
            <svg v-if="courtKeypointsSet" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
            </svg>
            <span>Set court keypoints (click "Set Court" on video)</span>
          </div>
          <div class="prereq-item" :class="{ done: videoStarted }">
            <svg v-if="videoStarted" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
            </svg>
            <span>Start video playback</span>
          </div>
        </div>
        <p class="hint">Both steps are required for accurate speed measurement</p>
      </div>
      
      <!-- Data available, show chart -->
      <Line
        v-else-if="chartData.datasets.length > 0"
        :data="chartData"
        :options="chartOptions"
      />
      
      <!-- Speed calculated but no data points -->
      <div v-else class="no-data">
        <p>No speed data available</p>
        <p class="hint">Speed is calculated from player position changes</p>
      </div>
    </div>
    
    <!-- Zone Legend -->
    <div class="zone-legend">
      <div 
        v-for="(thresholds, zone) in SPEED_ZONE_THRESHOLDS" 
        :key="zone"
        class="zone-item"
      >
        <span 
          class="zone-color" 
          :style="{ backgroundColor: SPEED_ZONE_COLORS[zone as SpeedZone] }"
        ></span>
        <span class="zone-label">{{ SPEED_ZONE_NAMES[zone as SpeedZone] }}</span>
        <span class="zone-range">
          {{ (useKmh ? thresholds.min * 3.6 : thresholds.min).toFixed(1) }}
          -
          {{ thresholds.max ? (useKmh ? thresholds.max * 3.6 : thresholds.max).toFixed(1) : '∞' }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.speed-graph-container {
  background: #141414;
  border-radius: 0;
  border: 1px solid #222;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 100%;
}

/* Header */
.speed-graph-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid #222;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
  font-weight: 600;
  font-size: 14px;
}

.header-title .icon {
  width: 18px;
  height: 18px;
  color: #22c55e;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.unit-toggle {
  background: #1a1a1a;
  color: #22c55e;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.unit-toggle:hover,
.unit-toggle.active {
  background: #222;
  border-color: #22c55e;
}

.window-select {
  background: #1a1a1a;
  color: #fff;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.toggle-btn,
.close-btn {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  padding: 4px;
  cursor: pointer;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.toggle-btn:hover,
.close-btn:hover {
  background: #222;
  color: #fff;
  border-color: #444;
}

.toggle-btn.active {
  background: #1a1a1a;
  color: #22c55e;
  border-color: #22c55e;
}

.icon-sm {
  width: 14px;
  height: 14px;
}

/* Statistics Panel */
.stats-panel {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.player-stats {
  flex: 1;
  min-width: 180px;
  background: #1a1a1a;
  border-radius: 0;
  padding: 10px;
  border-left: 3px solid;
}

.player-label {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #fff;
  font-weight: 600;
  font-size: 12px;
  margin-bottom: 8px;
}

.player-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.stat-item {
  text-align: center;
}

.stat-label {
  display: block;
  color: #666;
  font-size: 10px;
  text-transform: uppercase;
  margin-bottom: 2px;
}

.stat-value {
  display: block;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
}

.stat-value.zone-badge {
  border-radius: 0;
  padding: 2px 6px;
  display: inline-block;
}

.stat-unit {
  font-size: 10px;
  opacity: 0.7;
  font-weight: normal;
}

.zone-name {
  display: block;
  font-size: 9px;
  color: #666;
  margin-top: 2px;
}

/* Chart Container */
.chart-container {
  flex: 1;
  min-height: 250px;
  max-height: 350px;
  position: relative;
}

.no-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #666;
  gap: 8px;
}

.no-data p {
  margin: 4px 0;
}

.no-data .hint {
  font-size: 11px;
  opacity: 0.7;
}

.no-data.calculating .spinner {
  width: 32px;
  height: 32px;
  animation: spin 1s linear infinite;
  color: #22c55e;
  margin-bottom: 8px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.no-data.pending .icon-large {
  width: 40px;
  height: 40px;
  color: #f59e0b;
  margin-bottom: 8px;
}

.prerequisites {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 12px 0;
  padding: 12px 16px;
  background: #1a1a1a;
  border-radius: 0;
  border: 1px solid #333;
}

.prereq-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #666;
}

.prereq-item svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.prereq-item.done {
  color: #22c55e;
}

.prereq-item.done svg {
  color: #22c55e;
}

/* Zone Legend */
.zone-legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
  padding-top: 8px;
  border-top: 1px solid #222;
}

.zone-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: #666;
}

.zone-color {
  width: 10px;
  height: 10px;
  border-radius: 0;
}

.zone-label {
  font-weight: 500;
}

.zone-range {
  opacity: 0.7;
  font-size: 9px;
}

/* Responsive adjustments */
@media (max-width: 600px) {
  .stats-panel {
    flex-direction: column;
  }
  
  .player-stats {
    min-width: 100%;
  }
  
  .zone-legend {
    flex-wrap: wrap;
    justify-content: flex-start;
  }
}
</style>
