<script setup lang="ts">
import { ref, computed } from 'vue'
import type { AnalysisResult } from '@/types/analysis'
import { PLAYER_COLORS } from '@/types/analysis'
import { useAdvancedAnalytics } from '@/composables/useAdvancedAnalytics'

const props = defineProps<{
  result: AnalysisResult
  currentFrame: number
  courtKeypoints?: number[][] | null
}>()

const {
  rallies,
  rallyLengthDistribution,
  currentRally,
  shotPlacementsByType,
  recoveryStats,
  reactionStats,
  movementEfficiency,
} = useAdvancedAnalytics(
  computed(() => props.result),
  computed(() => props.currentFrame),
  computed(() => props.courtKeypoints ?? null)
)

type Tab = 'rallies' | 'shots' | 'movement'
const activeTab = ref<Tab>('rallies')

const tabs: { id: Tab; label: string }[] = [
  { id: 'rallies', label: 'Rallies' },
  { id: 'shots', label: 'Shots' },
  { id: 'movement', label: 'Movement' },
]

// Selected shot type filter for placement heatmap
const selectedShotType = ref('all')
const availableShotTypes = computed(() =>
  shotPlacementsByType.value.map(s => s.shotType)
)
const activePlacementHeatmap = computed(() =>
  shotPlacementsByType.value.find(s => s.shotType === selectedShotType.value)
)

function getPlayerColor(index: number): string {
  return PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B'
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function getHeatmapCellColor(value: number, maxValue: number): string {
  if (value === 0 || maxValue === 0) return 'transparent'
  const intensity = value / maxValue
  const r = Math.round(34 + intensity * 221)
  const g = Math.round(197 - intensity * 140)
  const b = Math.round(94 - intensity * 60)
  const a = 0.3 + intensity * 0.7
  return `rgba(${r}, ${g}, ${b}, ${a})`
}

function getQualityColor(quality: string): string {
  switch (quality) {
    case 'excellent': return '#22c55e'
    case 'good': return '#3b82f6'
    case 'fair': return '#f59e0b'
    case 'poor': return '#ef4444'
    default: return '#666'
  }
}
</script>

<template>
  <div class="advanced-analytics">
    <header class="aa-header">
      <h2>Advanced Analytics</h2>
      <p class="aa-subtitle">Deep performance insights computed from your match data</p>
    </header>

    <!-- Tab Navigation -->
    <nav class="aa-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="aa-tab"
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </nav>

    <!-- Tab Content -->
    <div class="aa-content">

      <!-- ============================================================ -->
      <!-- RALLIES TAB -->
      <!-- ============================================================ -->
      <div v-if="activeTab === 'rallies'" class="tab-panel">

        <!-- Rally Summary -->
        <section class="aa-section">
          <h3>Rally Overview</h3>
          <div class="aa-stat-row">
            <div class="aa-stat-card">
              <span class="aa-stat-value">{{ rallies.length }}</span>
              <span class="aa-stat-label">Total Rallies</span>
            </div>
            <div class="aa-stat-card">
              <span class="aa-stat-value">{{ rallyLengthDistribution.avgShots.toFixed(1) }}</span>
              <span class="aa-stat-label">Avg Shots/Rally</span>
            </div>
            <div class="aa-stat-card">
              <span class="aa-stat-value">{{ rallyLengthDistribution.avgDuration.toFixed(1) }}s</span>
              <span class="aa-stat-label">Avg Rally Duration</span>
            </div>
            <div class="aa-stat-card" v-if="rallies.length > 0">
              <span class="aa-stat-value">{{ Math.max(...rallies.map(r => r.shotCount)) }}</span>
              <span class="aa-stat-label">Longest Rally (shots)</span>
            </div>
          </div>
        </section>

        <!-- Rally Length Distribution -->
        <section class="aa-section" v-if="rallyLengthDistribution.bins.length > 0">
          <h3>Rally Length Distribution</h3>
          <div class="bar-chart">
            <div
              v-for="bin in rallyLengthDistribution.bins"
              :key="bin.label"
              class="bar-item"
            >
              <div class="bar-fill-container">
                <div
                  class="bar-fill"
                  :style="{
                    height: `${rallies.length > 0 ? (bin.count / rallies.length) * 100 : 0}%`
                  }"
                />
              </div>
              <span class="bar-count">{{ bin.count }}</span>
              <span class="bar-label">{{ bin.label }}</span>
            </div>
          </div>
          <p class="chart-axis-label">Shots per rally</p>
        </section>

        <!-- Rally List -->
        <section class="aa-section" v-if="rallies.length > 0">
          <h3>Rally Details</h3>
          <div class="rally-list">
            <div
              v-for="rally in rallies.slice(0, 20)"
              :key="rally.id"
              class="rally-item"
              :class="{ 'is-current': currentRally?.id === rally.id }"
            >
              <div class="rally-meta">
                <span class="rally-number">#{{ rally.id }}</span>
                <span class="rally-time">{{ formatTime(rally.startTimestamp) }}</span>
              </div>
              <div class="rally-stats">
                <span class="rally-shots">{{ rally.shotCount }} shots</span>
                <span class="rally-duration">{{ rally.durationSeconds.toFixed(1) }}s</span>
              </div>
              <div class="rally-shot-types">
                <span
                  v-for="(shot, i) in rally.shots.slice(0, 8)"
                  :key="i"
                  class="shot-dot"
                  :style="{ background: getPlayerColor(shot.playerId) }"
                  :title="`P${shot.playerId + 1}: ${shot.shotType}`"
                />
                <span v-if="rally.shots.length > 8" class="shot-more">+{{ rally.shots.length - 8 }}</span>
              </div>
            </div>
          </div>
        </section>

        <div v-if="rallies.length === 0" class="aa-empty">
          <p>No rallies detected. At least 2 shots must be detected via shuttle trajectory, player pose, or movement deceleration.</p>
        </div>
      </div>

      <!-- ============================================================ -->
      <!-- SHOTS TAB -->
      <!-- ============================================================ -->
      <div v-if="activeTab === 'shots'" class="tab-panel">

        <!-- Shot Placement Heatmap -->
        <section class="aa-section" v-if="shotPlacementsByType.length > 0">
          <h3>Shot Placement Heatmap</h3>
          <div class="shot-type-filter">
            <button
              v-for="type in availableShotTypes"
              :key="type"
              class="filter-btn"
              :class="{ active: selectedShotType === type }"
              @click="selectedShotType = type"
            >
              {{ type }}
            </button>
          </div>
          <div class="placement-heatmap" v-if="activePlacementHeatmap">
            <div class="heatmap-court">
              <div
                v-for="(row, ry) in activePlacementHeatmap.grid"
                :key="ry"
                class="heatmap-row"
              >
                <div
                  v-for="(val, cx) in row"
                  :key="cx"
                  class="heatmap-cell"
                  :style="{
                    background: getHeatmapCellColor(val as number, Math.max(...activePlacementHeatmap.grid.flatMap(r => r as number[])))
                  }"
                  :title="`${val} shots`"
                />
              </div>
              <!-- Court lines overlay -->
              <div class="court-overlay">
                <div class="net-line" />
                <div class="center-line" />
              </div>
            </div>
            <p class="heatmap-total">{{ activePlacementHeatmap.total }} total shots</p>
          </div>
        </section>

        <div v-if="shotPlacementsByType.length === 0" class="aa-empty">
          <p>Insufficient shuttle tracking data for shot placement heatmap. Shots must have detected shuttle positions.</p>
        </div>
      </div>

      <!-- ============================================================ -->
      <!-- MOVEMENT TAB -->
      <!-- ============================================================ -->
      <div v-if="activeTab === 'movement'" class="tab-panel">

        <!-- Reaction Time -->
        <section class="aa-section" v-if="reactionStats && reactionStats.length > 0">
          <h3>Reaction Time</h3>
          <div class="aa-stat-row">
            <div
              v-for="stat in reactionStats"
              :key="stat.playerId"
              class="aa-stat-card wide"
            >
              <div class="player-badge" :style="{ background: getPlayerColor(stat.playerId) }">
                P{{ stat.playerId + 1 }}
              </div>
              <div class="stat-details">
                <div class="stat-main">
                  <span class="aa-stat-value">{{ stat.avgReactionMs.toFixed(0) }}ms</span>
                  <span class="aa-stat-label">Avg Reaction</span>
                </div>
                <div class="stat-extra">
                  <span>Min: {{ stat.minReactionMs.toFixed(0) }}ms</span>
                  <span>Max: {{ stat.maxReactionMs.toFixed(0) }}ms</span>
                  <span>{{ stat.totalMeasured }} samples</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Recovery Analysis -->
        <section class="aa-section" v-if="recoveryStats && recoveryStats.perPlayer.length > 0">
          <h3>Recovery Position Analysis</h3>
          <div class="recovery-grid">
            <div
              v-for="stat in recoveryStats.perPlayer"
              :key="stat.playerId"
              class="recovery-card"
            >
              <div class="recovery-header">
                <div class="player-badge" :style="{ background: getPlayerColor(stat.playerId) }">
                  P{{ stat.playerId + 1 }}
                </div>
                <span class="recovery-avg">{{ stat.avgRecoveryTime.toFixed(2) }}s avg</span>
              </div>
              <div class="recovery-quality">
                <div
                  v-for="(count, quality) in stat.qualityDistribution"
                  :key="quality"
                  class="quality-bar-row"
                >
                  <span class="quality-label" :style="{ color: getQualityColor(quality) }">{{ quality }}</span>
                  <div class="quality-bar-bg">
                    <div
                      class="quality-bar-fill"
                      :style="{
                        width: `${stat.totalRecoveries > 0 ? (count / stat.totalRecoveries) * 100 : 0}%`,
                        background: getQualityColor(quality)
                      }"
                    />
                  </div>
                  <span class="quality-count">{{ count }}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Movement Efficiency -->
        <section class="aa-section" v-if="movementEfficiency.length > 0">
          <h3>Movement Efficiency</h3>
          <p class="section-desc">Ratio of direct movement to total distance — higher is more efficient</p>
          <div class="efficiency-grid">
            <div
              v-for="eff in movementEfficiency"
              :key="eff.playerId"
              class="efficiency-card"
            >
              <div class="efficiency-header">
                <div class="player-badge" :style="{ background: getPlayerColor(eff.playerId) }">
                  P{{ eff.playerId + 1 }}
                </div>
                <div class="efficiency-score">
                  <svg viewBox="0 0 36 36" class="circular-progress">
                    <path
                      class="circle-bg"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path
                      class="circle-fill"
                      :stroke-dasharray="`${eff.efficiencyScore}, 100`"
                      :style="{ stroke: eff.efficiencyScore >= 60 ? '#22c55e' : eff.efficiencyScore >= 40 ? '#f59e0b' : '#ef4444' }"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                  </svg>
                  <span class="score-text">{{ eff.efficiencyScore.toFixed(0) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div v-if="!reactionStats && !recoveryStats && movementEfficiency.length === 0" class="aa-empty">
          <p>Movement analytics require player tracking data with speed calculations. Minimum sample thresholds must be met.</p>
        </div>
      </div>

    </div>
  </div>
</template>

<style scoped>
.advanced-analytics {
  width: 100%;
  padding: 24px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
}

.aa-header {
  margin-bottom: 20px;
}

.aa-header h2 {
  color: var(--color-text-heading);
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 4px 0;
}

.aa-subtitle {
  color: var(--color-text-tertiary);
  font-size: 0.875rem;
  margin: 0;
}

/* Tabs */
.aa-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--color-border-secondary);
  margin-bottom: 24px;
  overflow-x: auto;
}

.aa-tab {
  padding: 10px 20px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.aa-tab:hover {
  color: #ccc;
}

.aa-tab.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}

/* Content */
.aa-content {
  min-height: 200px;
}

.tab-panel {
  display: flex;
  flex-direction: column;
  gap: 28px;
}

/* Sections */
.aa-section h3 {
  color: var(--color-text-heading);
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 12px 0;
}

.section-desc {
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  margin: -8px 0 12px 0;
}

/* Stat Cards */
.aa-stat-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.aa-stat-card {
  padding: 16px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  text-align: center;
}

.aa-stat-card.wide {
  display: flex;
  align-items: center;
  gap: 14px;
  text-align: left;
}

.aa-stat-value {
  display: block;
  color: var(--color-accent);
  font-size: 1.5rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.aa-stat-label {
  display: block;
  color: var(--color-text-secondary);
  font-size: 0.75rem;
  margin-top: 2px;
}

.player-badge {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  color: #000;
  flex-shrink: 0;
}

.stat-details {
  flex: 1;
}

.stat-main {
  display: flex;
  flex-direction: column;
}

.stat-extra {
  display: flex;
  gap: 12px;
  margin-top: 4px;
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}

/* Bar Chart */
.bar-chart {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  height: 140px;
  padding: 0 4px;
}

.bar-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
}

.bar-fill-container {
  flex: 1;
  width: 100%;
  display: flex;
  align-items: flex-end;
}

.bar-fill {
  width: 100%;
  background: var(--color-accent);
  min-height: 2px;
  transition: height 0.3s ease;
  opacity: 0.8;
}

.bar-count {
  font-size: 0.7rem;
  color: var(--color-accent);
  margin-top: 4px;
  font-weight: 600;
}

.bar-label {
  font-size: 0.65rem;
  color: var(--color-text-secondary);
  margin-top: 2px;
}

.chart-axis-label {
  text-align: center;
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  margin-top: 8px;
}

/* Rally List */
.rally-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 300px;
  overflow-y: auto;
}

.rally-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  transition: border-color 0.2s;
}

.rally-item.is-current {
  border-color: var(--color-accent);
  background: #1a2a1a;
}

.rally-meta {
  display: flex;
  flex-direction: column;
  min-width: 60px;
}

.rally-number {
  font-weight: 600;
  color: var(--color-text-heading);
  font-size: 0.875rem;
}

.rally-time {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  font-variant-numeric: tabular-nums;
}

.rally-stats {
  display: flex;
  flex-direction: column;
  min-width: 70px;
}

.rally-shots {
  font-size: 0.8rem;
  color: #ccc;
}

.rally-duration {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}

.rally-shot-types {
  display: flex;
  gap: 3px;
  align-items: center;
  flex-wrap: wrap;
}

.shot-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.shot-more {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
}

/* Shot Placement Heatmap */
.shot-type-filter {
  display: flex;
  gap: 6px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.filter-btn {
  padding: 4px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  color: var(--color-text-secondary);
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
  text-transform: capitalize;
}

.filter-btn:hover {
  border-color: var(--color-text-tertiary);
  color: #ccc;
}

.filter-btn.active {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: #000;
  font-weight: 600;
}

.placement-heatmap {
  max-width: 320px;
}

.heatmap-court {
  position: relative;
  display: flex;
  flex-direction: column;
  border: 2px solid var(--color-border-hover);
  background: var(--color-bg-input);
  aspect-ratio: 6.1 / 13.4;
  max-height: 280px;
}

.heatmap-row {
  display: flex;
  flex: 1;
}

.heatmap-cell {
  flex: 1;
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: background 0.3s;
}

.court-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.net-line {
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 2px;
  background: #555;
}

.center-line {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(255, 255, 255, 0.1);
}

.heatmap-total {
  font-size: 0.75rem;
  color: var(--color-text-tertiary);
  margin-top: 6px;
}

/* Recovery */
.recovery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}

.recovery-card {
  padding: 16px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
}

.recovery-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.recovery-avg {
  font-size: 0.9rem;
  color: #ccc;
  font-weight: 600;
}

.recovery-quality {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.quality-bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.quality-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: capitalize;
  width: 65px;
}

.quality-bar-bg {
  flex: 1;
  height: 8px;
  background: var(--color-border);
  overflow: hidden;
}

.quality-bar-fill {
  height: 100%;
  transition: width 0.3s;
}

.quality-count {
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  width: 24px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Efficiency */
.efficiency-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.efficiency-card {
  padding: 16px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  text-align: center;
}

.efficiency-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.efficiency-score {
  position: relative;
  width: 80px;
  height: 80px;
}

.circular-progress {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.circle-bg {
  fill: none;
  stroke: var(--color-border);
  stroke-width: 3;
}

.circle-fill {
  fill: none;
  stroke-width: 3;
  stroke-linecap: butt;
  transition: stroke-dasharray 0.3s;
}

.score-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-text-heading);
}

/* Empty state */
.aa-empty {
  padding: 32px;
  text-align: center;
  color: var(--color-text-tertiary);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
}

.aa-empty p {
  font-size: 0.875rem;
}
</style>
