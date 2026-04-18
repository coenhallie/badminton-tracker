<script setup lang="ts">
import type { ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'

const props = defineProps<{
  segment: ShotMovementSegmentWithPeaks
  countdownSec: number
}>()

function fmtNumber(v: number | null, digits: number, unit: string): string {
  if (v == null || !isFinite(v)) return '—'
  return `${v.toFixed(digits)} ${unit}`
}
</script>

<template>
  <div class="shot-summary-overlay">
    <div class="shot-summary-header">
      <span class="shot-summary-player">Player {{ segment.movingPlayerId + 1 }} responded</span>
      <span class="shot-summary-countdown">⏱ {{ countdownSec.toFixed(1) }}s</span>
    </div>

    <div class="shot-summary-grid">
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Max speed</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.maxSpeedKmh, 1, 'km/h') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Avg speed</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.avgSpeedKmh, 1, 'km/h') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Distance</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.distanceCoveredM, 2, 'm') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Duration</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.durationSeconds, 2, 's') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Leg stretch</span>
        <span class="shot-summary-value">{{ fmtNumber(segment.peaks.peakLegStretchM, 2, 'm') }}</span>
      </div>
      <div class="shot-summary-cell">
        <span class="shot-summary-label">Knee flex</span>
        <span class="shot-summary-value">{{ segment.peaks.peakKneeFlexDeg != null ? Math.round(segment.peaks.peakKneeFlexDeg) + '°' : '—' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.shot-summary-overlay {
  position: absolute;
  left: 50%;
  bottom: 64px;
  transform: translateX(-50%);
  width: 320px;
  padding: 12px 14px;
  background: rgba(0, 0, 0, 0.88);
  border: 1px solid var(--color-accent);
  color: #f5f5f5;
  font: 500 12px/1.2 Inter, system-ui, sans-serif;
  /* Above SyntheticCourtView (2), skeleton-canvas (3), PoseOverlay (20),
     controls (25). See VideoPlayer.vue z-index table. */
  z-index: 26;
  pointer-events: none;
}

.shot-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}

.shot-summary-player {
  font-weight: 600;
  font-size: 13px;
}

.shot-summary-countdown {
  font-variant-numeric: tabular-nums;
  color: var(--color-accent);
}

.shot-summary-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 12px;
}

.shot-summary-cell {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.shot-summary-label {
  color: rgba(255, 255, 255, 0.6);
  font-size: 11px;
}

.shot-summary-value {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
</style>
