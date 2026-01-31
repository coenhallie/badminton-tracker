<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue'
import type { AnalysisResult, PlayerMetrics, ShuttleMetrics, CourtDetection, PlayersZoneAnalytics, ZoneCoverage } from '@/types/analysis'
import { PLAYER_COLORS } from '@/types/analysis'
import { exportPDFWithFrontendData, getRecalculatedZoneAnalytics, clearZoneAnalyticsCache, type PlayerZoneData } from '@/services/api'
import CourtZoneTooltip from './CourtZoneTooltip.vue'

type ZoneType = 'front' | 'mid' | 'back' | 'left' | 'center' | 'right'

const props = defineProps<{
  result: AnalysisResult
  // Indicates whether manual court keypoints have been set (triggers zone analytics reload)
  manualKeypointsSet?: boolean
  // Counter that increments when keypoints are confirmed (triggers zone recalculation)
  zoneRecalculationTrigger?: number
}>()

// PDF Export state
const isExporting = ref(false)
const exportError = ref<string | null>(null)

// Zone analytics state
const zoneAnalytics = ref<Record<string, PlayerZoneData> | null>(null)
const zoneAnalyticsLoading = ref(false)
const zoneAnalyticsError = ref<string | null>(null)

// Tooltip hover state
const hoveredZone = ref<{ playerId: string; zone: ZoneType } | null>(null)

function showZoneTooltip(playerId: string, zone: ZoneType) {
  hoveredZone.value = { playerId, zone }
}

function hideZoneTooltip() {
  hoveredZone.value = null
}

function isZoneHovered(playerId: string, zone: ZoneType): boolean {
  return hoveredZone.value?.playerId === playerId && hoveredZone.value?.zone === zone
}

async function handleExportPDF() {
  if (isExporting.value) return
  
  isExporting.value = true
  exportError.value = null
  
  try {
    // Send frontend data to PDF endpoint to ensure accurate display
    // This uses the exact data shown in the dashboard
    
    // Get video dimensions from result (may not be in type but could be in data)
    const result = props.result as AnalysisResult & { video_width?: number; video_height?: number }
    
    await exportPDFWithFrontendData(props.result.video_id, {
      include_heatmap: true,
      heatmap_colormap: 'turbo',
      heatmap_alpha: 0.6,
      title: 'Badminton Video Analysis Report',
      // Pass all data from the frontend result
      duration: props.result.duration,
      fps: props.result.fps,
      total_frames: props.result.total_frames,
      processed_frames: props.result.processed_frames,
      video_width: result.video_width ?? 1920,
      video_height: result.video_height ?? 1080,
      players: props.result.players.map(p => ({
        player_id: p.player_id,
        total_distance: p.total_distance,
        avg_speed: p.avg_speed,
        max_speed: p.max_speed,
        positions: p.positions
      })),
      shuttle: props.result.shuttle ?? null,
      court_detection: props.result.court_detection ? {
        detected: props.result.court_detection.detected,
        confidence: props.result.court_detection.confidence,
        court_dimensions: props.result.court_detection.court_dimensions
      } : null,
      shuttle_analytics: props.result.shuttle_analytics as Record<string, unknown> | null ?? null,
      player_zone_analytics: props.result.player_zone_analytics as Record<string, unknown> | null ?? null
    })
  } catch (error) {
    console.error('PDF export failed:', error)
    exportError.value = error instanceof Error ? error.message : 'Export failed'
  } finally {
    isExporting.value = false
  }
}

const courtDetection = computed(() => props.result.court_detection)
const courtConfidencePercent = computed(() => {
  if (!courtDetection.value) return 0
  return Math.round(courtDetection.value.confidence * 100)
})

const formattedDuration = computed(() => {
  const mins = Math.floor(props.result.duration / 60)
  const secs = Math.floor(props.result.duration % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
})

const totalDistance = computed(() => {
  return props.result.players.reduce((sum, p) => sum + p.total_distance, 0)
})

const avgSpeed = computed(() => {
  if (props.result.players.length === 0) return 0
  return props.result.players.reduce((sum, p) => sum + p.avg_speed, 0) / props.result.players.length
})

const maxSpeed = computed(() => {
  return Math.max(...props.result.players.map(p => p.max_speed), 0)
})

function getPlayerColor(index: number): string {
  return PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B'
}

function formatSpeed(speed: number): string {
  return speed.toFixed(1)
}

function formatDistance(distance: number): string {
  if (distance >= 1000) {
    return `${(distance / 1000).toFixed(2)} km`
  }
  return `${distance.toFixed(1)} m`
}

// Zone analytics functions
async function loadZoneAnalytics(forceRefresh: boolean = false) {
  if (!props.result.video_id) return
  
  zoneAnalyticsLoading.value = true
  zoneAnalyticsError.value = null
  
  try {
    // Try to get recalculated zone analytics (uses manual keypoints if set)
    const response = await getRecalculatedZoneAnalytics(props.result.video_id, forceRefresh)
    zoneAnalytics.value = response.player_zone_analytics
    console.log('[ResultsDashboard] Zone analytics loaded:', {
      forceRefresh,
      manualKeypointsUsed: response.manual_keypoints_used,
      playerCount: Object.keys(response.player_zone_analytics).length
    })
  } catch (error) {
    console.warn('Failed to fetch recalculated zone analytics:', error)
    // Fall back to stored zone analytics from result
    if (props.result.player_zone_analytics) {
      zoneAnalytics.value = props.result.player_zone_analytics as unknown as Record<string, PlayerZoneData>
    } else {
      zoneAnalyticsError.value = 'Zone analytics not available'
    }
  } finally {
    zoneAnalyticsLoading.value = false
  }
}

function hasValidZoneData(): boolean {
  if (!zoneAnalytics.value) return false
  
  // Check if any player has non-zero zone coverage
  for (const playerData of Object.values(zoneAnalytics.value)) {
    const coverage = playerData.zone_coverage
    if (coverage.front > 0 || coverage.mid > 0 || coverage.back > 0 ||
        coverage.left > 0 || coverage.center > 0 || coverage.right > 0) {
      return true
    }
  }
  return false
}

function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`
}

function getZoneBarWidth(value: number): string {
  // Scale percentages for visual representation
  return `${Math.min(value, 100)}%`
}

// Load zone analytics on mount and when result changes
onMounted(() => {
  loadZoneAnalytics()
})

watch(() => props.result.video_id, () => {
  loadZoneAnalytics()
})

// Watch for manual keypoints being set - reload zone analytics with force refresh
watch(() => props.manualKeypointsSet, (newValue, oldValue) => {
  // Only reload when transitioning from false/undefined to true
  if (newValue && !oldValue && props.result.video_id) {
    console.log('[ResultsDashboard] Manual keypoints set - reloading zone analytics with forceRefresh')
    // Clear cache first to ensure fresh data
    clearZoneAnalyticsCache(props.result.video_id)
    // Reload with fresh data - pass forceRefresh=true to bypass any remaining cache
    loadZoneAnalytics(true)
  }
})

// Watch for zone recalculation trigger - reloads zone analytics when keypoints are confirmed via Done button
// This triggers every time the trigger counter increments, ensuring recalculation happens
// even when keypoints are adjusted and Done is clicked multiple times
watch(() => props.zoneRecalculationTrigger, (newValue, oldValue) => {
  // Only trigger when counter increases (and is a valid number)
  if (newValue && newValue > 0 && newValue !== oldValue && props.result.video_id) {
    console.log(`[ResultsDashboard] Zone recalculation triggered (counter: ${newValue})`)
    console.log('[ResultsDashboard] Clearing cache and reloading zone analytics...')
    
    // Clear cache to force fresh calculation from backend
    clearZoneAnalyticsCache(props.result.video_id)
    
    // Set loading state for UI feedback
    zoneAnalyticsLoading.value = true
    
    // Reload with force refresh to get recalculated data using new homography
    loadZoneAnalytics(true)
  }
})
</script>

<template>
  <div class="results-dashboard">
    <header class="dashboard-header">
      <div class="header-left">
        <h2>Analysis Results</h2>
        <span class="video-id">Video ID: {{ result.video_id.slice(0, 8) }}...</span>
      </div>
      <div class="header-actions">
        <button
          class="export-btn"
          :class="{ 'is-loading': isExporting }"
          :disabled="isExporting"
          @click="handleExportPDF"
        >
          <svg v-if="!isExporting" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <line x1="9" y1="15" x2="12" y2="12" />
            <line x1="15" y1="15" x2="12" y2="12" />
          </svg>
          <svg v-else class="spinner" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="32">
              <animate attributeName="stroke-dashoffset" values="32;0" dur="1s" repeatCount="indefinite" />
            </circle>
          </svg>
          <span>{{ isExporting ? 'Generating...' : 'Export PDF' }}</span>
        </button>
        <p v-if="exportError" class="export-error">{{ exportError }}</p>
      </div>
    </header>

    <!-- Summary Stats -->
    <section class="summary-stats">
      <div class="stat-card">
        <div class="stat-icon duration-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ formattedDuration }}</span>
          <span class="stat-label">Duration</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon frames-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
            <line x1="7" y1="2" x2="7" y2="22" />
            <line x1="17" y1="2" x2="17" y2="22" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <line x1="2" y1="7" x2="7" y2="7" />
            <line x1="2" y1="17" x2="7" y2="17" />
            <line x1="17" y1="17" x2="22" y2="17" />
            <line x1="17" y1="7" x2="22" y2="7" />
          </svg>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ result.processed_frames }}</span>
          <span class="stat-label">Frames Analyzed</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon players-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
            <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
            <path d="M16 3.13a4 4 0 0 1 0 7.75" />
          </svg>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ result.players.length }}</span>
          <span class="stat-label">Players Detected</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon fps-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ result.fps.toFixed(0) }}</span>
          <span class="stat-label">FPS</span>
        </div>
      </div>
    </section>

    <!-- Movement Metrics -->
    <section class="metrics-section">
      <h3>Movement Analysis</h3>
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-label">Total Distance Covered</span>
          <span class="metric-value">{{ formatDistance(totalDistance) }}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Average Speed</span>
          <span class="metric-value">{{ formatSpeed(avgSpeed) }} <small>km/h</small></span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Max Speed</span>
          <span class="metric-value">{{ formatSpeed(maxSpeed) }} <small>km/h</small></span>
        </div>
      </div>
    </section>

    <!-- Player Stats -->
    <section class="players-section">
      <h3>Player Statistics</h3>
      <div class="players-grid">
        <div
          v-for="(player, index) in result.players"
          :key="player.player_id"
          class="player-card"
          :style="{ '--player-color': getPlayerColor(index) }"
        >
          <div class="player-header">
            <div class="player-avatar" :style="{ background: getPlayerColor(index) }">
              P{{ player.player_id }}
            </div>
            <h4>Player {{ player.player_id }}</h4>
          </div>
          <div class="player-stats">
            <div class="player-stat">
              <span class="label">Distance</span>
              <span class="value">{{ formatDistance(player.total_distance) }}</span>
            </div>
            <div class="player-stat">
              <span class="label">Avg Speed</span>
              <span class="value">{{ formatSpeed(player.avg_speed) }} km/h</span>
            </div>
            <div class="player-stat">
              <span class="label">Max Speed</span>
              <span class="value">{{ formatSpeed(player.max_speed) }} km/h</span>
            </div>
            <div class="player-stat">
              <span class="label">Tracked Frames</span>
              <span class="value">{{ player.positions.length }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Court Coverage Section -->
    <section class="coverage-section">
      <h3>
        <svg class="section-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="2" y="4" width="20" height="16" rx="1" />
          <line x1="12" y1="4" x2="12" y2="20" />
          <line x1="2" y1="12" x2="22" y2="12" />
        </svg>
        Court Coverage
        <span v-if="zoneAnalyticsLoading" class="loading-indicator">Loading...</span>
      </h3>
      
      <div v-if="zoneAnalyticsError" class="coverage-error">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <span>{{ zoneAnalyticsError }}</span>
      </div>
      
      <div v-else-if="!hasValidZoneData() && !zoneAnalyticsLoading" class="coverage-notice">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <span>Set manual court keypoints for accurate zone coverage analysis</span>
      </div>
      
      <div v-else-if="zoneAnalytics" class="coverage-grid">
        <div
          v-for="(playerData, playerId) in zoneAnalytics"
          :key="playerId"
          class="coverage-card"
          :style="{ '--player-color': getPlayerColor(parseInt(playerId as string) - 1) }"
        >
          <div class="coverage-header">
            <div class="player-avatar" :style="{ background: getPlayerColor(parseInt(playerId as string) - 1) }">
              P{{ playerId }}
            </div>
            <h4>Player {{ playerId }}</h4>
            <span class="position-count">{{ playerData.position_count }} positions</span>
          </div>
          
          <!-- Vertical zones (front/mid/back) -->
          <div class="zone-group">
            <h5>Court Depth</h5>
            <div class="zone-bars">
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'front')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Front
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'front')" zone="front" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar front-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.front) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.front) }}</span>
              </div>
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'mid')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Mid
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'mid')" zone="mid" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar mid-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.mid) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.mid) }}</span>
              </div>
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'back')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Back
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'back')" zone="back" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar back-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.back) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.back) }}</span>
              </div>
            </div>
          </div>
          
          <!-- Horizontal zones (left/center/right) -->
          <div class="zone-group">
            <h5>Court Width</h5>
            <div class="zone-bars">
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'left')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Left
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'left')" zone="left" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar left-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.left) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.left) }}</span>
              </div>
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'center')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Center
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'center')" zone="center" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar center-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.center) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.center) }}</span>
              </div>
              <div class="zone-item">
                <span
                  class="zone-label with-tooltip"
                  @mouseenter="showZoneTooltip(playerId as string, 'right')"
                  @mouseleave="hideZoneTooltip()"
                >
                  Right
                  <CourtZoneTooltip v-if="isZoneHovered(playerId as string, 'right')" zone="right" />
                </span>
                <div class="zone-bar-container">
                  <div class="zone-bar right-zone" :style="{ width: getZoneBarWidth(playerData.zone_coverage.right) }"></div>
                </div>
                <span class="zone-value">{{ formatPercentage(playerData.zone_coverage.right) }}</span>
              </div>
            </div>
          </div>
          
          <!-- Average distance to net -->
          <div class="net-distance" v-if="playerData.avg_distance_to_net_m > 0">
            <span class="distance-label">Avg Distance to Net</span>
            <span class="distance-value">{{ playerData.avg_distance_to_net_m.toFixed(2) }} m</span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.results-dashboard {
  width: 100%;
  padding: 24px;
  background: #141414;
  border: 1px solid #222;
  border-radius: 0;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #222;
  gap: 16px;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.dashboard-header h2 {
  color: #fff;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

.video-id {
  color: #666;
  font-size: 0.875rem;
  font-family: monospace;
}

.header-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.export-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: #22c55e;
  color: #000;
  border: none;
  border-radius: 0;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.export-btn:hover:not(:disabled) {
  background: #16a34a;
}

.export-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.export-btn.is-loading {
  background: #1a1a1a;
  border: 1px solid #22c55e;
  color: #22c55e;
}

.export-btn svg {
  width: 18px;
  height: 18px;
}

.export-btn .spinner {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.export-error {
  color: #ef4444;
  font-size: 0.75rem;
  margin: 0;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #1a1a1a;
  border: 1px solid #222;
  border-radius: 0;
}

.stat-icon {
  width: 48px;
  height: 48px;
  padding: 12px;
  border-radius: 0;
  border: 1px solid #333;
}

.stat-icon svg {
  width: 100%;
  height: 100%;
}

.duration-icon {
  background: #1a1a1a;
  color: #22c55e;
}

.frames-icon {
  background: #1a1a1a;
  color: #22c55e;
}

.players-icon {
  background: #1a1a1a;
  color: #22c55e;
}

.fps-icon {
  background: #1a1a1a;
  color: #22c55e;
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-value {
  color: #fff;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
}

.stat-label {
  color: #666;
  font-size: 0.875rem;
}

.metrics-section,
.players-section,
.shuttle-section {
  margin-bottom: 32px;
}

.metrics-section h3,
.players-section h3,
.shuttle-section h3 {
  color: #fff;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 16px;
}

.metrics-grid,
.shuttle-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.metric-card {
  padding: 20px;
  background: #1a1a1a;
  border: 1px solid #222;
  border-radius: 0;
  text-align: center;
}

.metric-card.highlight {
  background: #1a1a1a;
  border: 1px solid #22c55e;
}

.metric-label {
  display: block;
  color: #888;
  font-size: 0.875rem;
  margin-bottom: 8px;
}

.metric-value {
  display: block;
  color: #fff;
  font-size: 2rem;
  font-weight: 700;
}

.metric-value small {
  font-size: 1rem;
  font-weight: 400;
  color: #888;
}

.shuttle-value {
  color: #22c55e;
}

.players-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

.player-card {
  padding: 20px;
  background: #1a1a1a;
  border: 1px solid #222;
  border-radius: 0;
  border-left: 4px solid var(--player-color);
}

.player-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.player-avatar {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0;
  color: white;
  font-weight: 700;
  font-size: 0.875rem;
}

.player-header h4 {
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
}

.player-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.player-stat {
  display: flex;
  flex-direction: column;
}

.player-stat .label {
  color: #666;
  font-size: 0.75rem;
  margin-bottom: 2px;
}

.player-stat .value {
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
}

.shot-speeds {
  margin-top: 24px;
  padding: 20px;
  background: #0d0d0d;
  border: 1px solid #222;
  border-radius: 0;
}

.shot-speeds h4 {
  color: #888;
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 16px;
}

.speed-bars {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 120px;
  padding: 0 8px;
}

.speed-bar {
  flex: 1;
  min-width: 12px;
  background: #22c55e;
  border-radius: 0;
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.speed-bar:hover {
  opacity: 0.8;
}

.speed-labels {
  display: flex;
  justify-content: space-between;
  padding: 8px 8px 0;
  color: #666;
  font-size: 0.625rem;
}

.no-data {
  opacity: 0.7;
}

.no-data-message {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #888;
  font-size: 0.875rem;
  padding: 16px;
  background: #0d0d0d;
  border: 1px solid #222;
  border-radius: 0;
}

.no-data-message svg {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
}

/* Court Detection Section */
.court-section {
  margin-bottom: 32px;
}

.court-section h3 {
  color: #fff;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-icon {
  width: 20px;
  height: 20px;
  color: #22c55e;
}

.court-badge {
  font-size: 0.75rem;
  font-weight: 500;
  padding: 4px 10px;
  border-radius: 0;
  background: #1a0000;
  border: 1px solid #ef4444;
  color: #ef4444;
  margin-left: auto;
}

.court-badge.court-detected {
  background: #001a00;
  border-color: #22c55e;
  color: #22c55e;
}

.court-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.court-card {
  border: 1px solid #22c55e;
}

.court-value {
  color: #22c55e;
}

.confidence-bar {
  margin-top: 12px;
  height: 6px;
  background: #222;
  border-radius: 0;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: #22c55e;
  border-radius: 0;
  transition: width 0.3s ease;
}

.court-note {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: #001a00;
  border: 1px solid #22c55e;
  border-radius: 0;
}

.court-note svg {
  width: 20px;
  height: 20px;
  color: #22c55e;
  flex-shrink: 0;
  margin-top: 2px;
}

.court-note span {
  color: #888;
  font-size: 0.875rem;
  line-height: 1.5;
}

.court-not-detected {
  opacity: 0.8;
}

/* Court Coverage Section */
.coverage-section {
  margin-bottom: 32px;
}

.coverage-section h3 {
  color: #fff;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.loading-indicator {
  font-size: 0.75rem;
  color: #666;
  font-weight: 400;
  margin-left: auto;
}

.coverage-error,
.coverage-notice {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
}

.coverage-error {
  border-color: #ef4444;
  background: #1a0000;
}

.coverage-error svg,
.coverage-notice svg {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.coverage-error svg {
  color: #ef4444;
}

.coverage-notice svg {
  color: #f59e0b;
}

.coverage-error span,
.coverage-notice span {
  color: #888;
  font-size: 0.875rem;
}

.coverage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}

.coverage-card {
  padding: 20px;
  background: #1a1a1a;
  border: 1px solid #222;
  border-radius: 0;
  border-left: 4px solid var(--player-color);
}

.coverage-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.coverage-header h4 {
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  flex: 1;
}

.position-count {
  color: #666;
  font-size: 0.75rem;
}

.zone-group {
  margin-bottom: 16px;
}

.zone-group:last-of-type {
  margin-bottom: 12px;
}

.zone-group h5 {
  color: #888;
  font-size: 0.75rem;
  font-weight: 500;
  margin: 0 0 8px 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.zone-bars {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.zone-item {
  display: grid;
  grid-template-columns: 50px 1fr 50px;
  align-items: center;
  gap: 8px;
}

.zone-label {
  color: #666;
  font-size: 0.75rem;
}

.zone-label.with-tooltip {
  position: relative;
  cursor: help;
  border-bottom: 1px dotted #444;
  padding-bottom: 1px;
  transition: color 0.2s ease, border-color 0.2s ease;
}

.zone-label.with-tooltip:hover {
  color: #fff;
  border-color: #666;
}

.zone-bar-container {
  height: 8px;
  background: #222;
  border-radius: 0;
  overflow: hidden;
}

.zone-bar {
  height: 100%;
  border-radius: 0;
  transition: width 0.3s ease;
}

.front-zone {
  background: linear-gradient(90deg, #22c55e, #4ade80);
}

.mid-zone {
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
}

.back-zone {
  background: linear-gradient(90deg, #f59e0b, #fbbf24);
}

.left-zone {
  background: linear-gradient(90deg, #ec4899, #f472b6);
}

.center-zone {
  background: linear-gradient(90deg, #8b5cf6, #a78bfa);
}

.right-zone {
  background: linear-gradient(90deg, #06b6d4, #22d3ee);
}

.zone-value {
  color: #fff;
  font-size: 0.75rem;
  font-weight: 600;
  text-align: right;
}

.net-distance {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 12px;
  margin-top: 4px;
  border-top: 1px solid #222;
}

.distance-label {
  color: #666;
  font-size: 0.75rem;
}

.distance-value {
  color: #22c55e;
  font-size: 0.875rem;
  font-weight: 600;
}
</style>
