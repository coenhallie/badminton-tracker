<script setup lang="ts">
import { computed } from 'vue'
import type { AnalysisResult, PlayerMetrics, ShuttleMetrics, CourtDetection } from '@/types/analysis'
import { PLAYER_COLORS } from '@/types/analysis'

const props = defineProps<{
  result: AnalysisResult
}>()

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
</script>

<template>
  <div class="results-dashboard">
    <header class="dashboard-header">
      <h2>Analysis Results</h2>
      <span class="video-id">Video ID: {{ result.video_id.slice(0, 8) }}...</span>
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
        <div class="metric-card highlight">
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
  </div>
</template>

<style scoped>
.results-dashboard {
  width: 100%;
  padding: 24px;
  background: linear-gradient(145deg, #1a1f2e, #242b3d);
  border-radius: 16px;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.dashboard-header h2 {
  color: #e2e8f0;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

.video-id {
  color: #718096;
  font-size: 0.875rem;
  font-family: monospace;
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
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  padding: 12px;
  border-radius: 10px;
}

.stat-icon svg {
  width: 100%;
  height: 100%;
}

.duration-icon {
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
}

.frames-icon {
  background: rgba(78, 205, 196, 0.2);
  color: #4ecdc4;
}

.players-icon {
  background: rgba(150, 206, 180, 0.2);
  color: #96ceb4;
}

.fps-icon {
  background: rgba(255, 234, 167, 0.2);
  color: #ffeaa7;
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-value {
  color: #e2e8f0;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
}

.stat-label {
  color: #718096;
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
  color: #e2e8f0;
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
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  text-align: center;
}

.metric-card.highlight {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
  border: 1px solid rgba(102, 126, 234, 0.3);
}

.metric-label {
  display: block;
  color: #a0aec0;
  font-size: 0.875rem;
  margin-bottom: 8px;
}

.metric-value {
  display: block;
  color: #e2e8f0;
  font-size: 2rem;
  font-weight: 700;
}

.metric-value small {
  font-size: 1rem;
  font-weight: 400;
  color: #a0aec0;
}

.shuttle-value {
  color: #ffeaa7;
}

.players-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

.player-card {
  padding: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
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
  border-radius: 50%;
  color: white;
  font-weight: 700;
  font-size: 0.875rem;
}

.player-header h4 {
  color: #e2e8f0;
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
  color: #718096;
  font-size: 0.75rem;
  margin-bottom: 2px;
}

.player-stat .value {
  color: #e2e8f0;
  font-size: 1rem;
  font-weight: 600;
}

.shot-speeds {
  margin-top: 24px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 12px;
}

.shot-speeds h4 {
  color: #a0aec0;
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
  background: linear-gradient(180deg, #ffeaa7, #fdcb6e);
  border-radius: 4px 4px 0 0;
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
  color: #718096;
  font-size: 0.625rem;
}

.no-data {
  opacity: 0.7;
}

.no-data-message {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #a0aec0;
  font-size: 0.875rem;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
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
  color: #e2e8f0;
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
  color: #4ecdc4;
}

.court-badge {
  font-size: 0.75rem;
  font-weight: 500;
  padding: 4px 10px;
  border-radius: 20px;
  background: rgba(255, 107, 107, 0.2);
  color: #ff6b6b;
  margin-left: auto;
}

.court-badge.court-detected {
  background: rgba(78, 205, 196, 0.2);
  color: #4ecdc4;
}

.court-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.court-card {
  border: 1px solid rgba(78, 205, 196, 0.2);
}

.court-value {
  color: #4ecdc4;
}

.confidence-bar {
  margin-top: 12px;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #4ecdc4, #45b7d1);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.court-note {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: rgba(78, 205, 196, 0.1);
  border-radius: 12px;
  border: 1px solid rgba(78, 205, 196, 0.2);
}

.court-note svg {
  width: 20px;
  height: 20px;
  color: #4ecdc4;
  flex-shrink: 0;
  margin-top: 2px;
}

.court-note span {
  color: #a0aec0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.court-not-detected {
  opacity: 0.8;
}
</style>
