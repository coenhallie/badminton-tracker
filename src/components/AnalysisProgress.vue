<script setup lang="ts">
import { ref, onMounted, computed, nextTick, watch } from 'vue'
import { useConvexQuery, useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { AnalysisResult, ProcessingLog, LogLevel, LogCategory } from '@/types/analysis'
import type { Id } from '../../convex/_generated/dataModel'

const props = defineProps<{
  videoId: string
  filename: string
}>()

const emit = defineEmits<{
  complete: [result: AnalysisResult]
  error: [message: string]
  cancel: []
}>()

const client = useConvexClient()

// Convert string videoId to Convex Id type
const convexVideoId = computed(() => props.videoId as Id<'videos'>)

// Real-time video query - automatically updates when database changes
const { data: videoData } = useConvexQuery(
  api.videos.getVideo,
  computed(() => ({ videoId: convexVideoId.value }))
)

// Real-time logs query
const { data: logsData } = useConvexQuery(
  api.videos.getProcessingLogs,
  computed(() => ({ videoId: convexVideoId.value }))
)

// Local state
const startTime = ref<number>(0)
const eta = ref<string>('')
const logsContainerRef = ref<HTMLElement | null>(null)
const showLogs = ref(true)
const analysisStarted = ref(false)

// Computed values from Convex data
const progress = computed(() => videoData.value?.progress ?? 0)
const currentFrame = computed(() => videoData.value?.currentFrame ?? 0)
const totalFrames = computed(() => videoData.value?.totalFrames ?? 0)
const videoStatus = computed(() => videoData.value?.status ?? 'uploaded')
const errorMessage = computed(() => videoData.value?.error ?? '')

// Map Convex status to component status
const status = computed(() => {
  switch (videoStatus.value) {
    case 'uploaded':
      return 'connecting'
    case 'processing':
      return 'analyzing'
    case 'completed':
      return 'complete'
    case 'failed':
      return 'error'
    default:
      return 'processing'
  }
})

const progressPercent = computed(() => Math.min(100, Math.round(progress.value)))

const statusMessage = computed(() => {
  switch (status.value) {
    case 'connecting':
      return 'Starting analysis...'
    case 'analyzing':
      return 'Analyzing video frames...'
    case 'processing':
      return 'Processing pose estimation...'
    case 'complete':
      return 'Analysis complete!'
    case 'error':
      return errorMessage.value || 'An error occurred'
    default:
      return 'Initializing...'
  }
})

// Convert Convex logs to ProcessingLog format
const processingLogs = computed<ProcessingLog[]>(() => {
  if (!logsData.value) return []
  return logsData.value.map((log, index) => ({
    id: index,
    message: log.message,
    level: log.level as LogLevel,
    category: log.category as LogCategory,
    timestamp: log.timestamp / 1000, // Convert to seconds
  }))
})

// Watch for completion or error
watch(videoStatus, async (newStatus) => {
  if (newStatus === 'completed' && videoData.value) {
    // Fetch results from storage URL
    const resultsUrl = videoData.value.resultsUrl
    if (resultsUrl) {
      try {
        const response = await fetch(resultsUrl)
        const results = await response.json() as Record<string, unknown>
        
        const analysisResult: AnalysisResult = {
          video_id: props.videoId,
          duration: (results.duration as number) ?? 0,
          fps: (results.fps as number) ?? 0,
          total_frames: (results.total_frames as number) ?? 0,
          processed_frames: (results.processed_frames as number) ?? 0,
          players: (results.players as AnalysisResult['players']) ?? [],
          shuttle: (results.shuttle as AnalysisResult['shuttle']) ?? null,
          skeleton_data: (results.skeleton_data as AnalysisResult['skeleton_data']) ?? [],
          court_detection: (results.court_detection as AnalysisResult['court_detection']) ?? null,
          shuttle_analytics: (results.shuttle_analytics as AnalysisResult['shuttle_analytics']) ?? null,
          player_zone_analytics: (results.player_zone_analytics as AnalysisResult['player_zone_analytics']) ?? null,
        }
        emit('complete', analysisResult)
      } catch (err) {
        console.error('Failed to fetch results:', err)
        emit('error', 'Failed to load analysis results')
      }
    } else {
      // Fallback: use metadata if no results URL (shouldn't happen normally)
      const meta = videoData.value.resultsMeta
      if (meta) {
        const analysisResult: AnalysisResult = {
          video_id: props.videoId,
          duration: meta.duration ?? 0,
          fps: meta.fps ?? 0,
          total_frames: meta.total_frames ?? 0,
          processed_frames: meta.processed_frames ?? 0,
          players: [],
          shuttle: null,
          skeleton_data: [],
          court_detection: null,
          shuttle_analytics: null,
          player_zone_analytics: null,
        }
        emit('complete', analysisResult)
      }
    }
  } else if (newStatus === 'failed') {
    emit('error', errorMessage.value || 'Analysis failed')
  }
})

// Watch for progress changes to update ETA
watch(progress, () => {
  calculateETA()
})

// Auto-scroll logs
watch(processingLogs, () => {
  nextTick(() => {
    if (logsContainerRef.value) {
      logsContainerRef.value.scrollTop = logsContainerRef.value.scrollHeight
    }
  })
})

// Get icon for log level
function getLogIcon(level: LogLevel): string {
  switch (level) {
    case 'success': return '✓'
    case 'warning': return '⚠'
    case 'error': return '✗'
    case 'debug': return '🔧'
    case 'info':
    default: return '→'
  }
}

// Get category icon
function getCategoryIcon(category: LogCategory): string {
  switch (category) {
    case 'modal': return '☁️'
    case 'model': return '🧠'
    case 'detection': return '👁️'
    case 'court': return '🏸'
    case 'processing':
    default: return '⚙️'
  }
}

// Format timestamp
function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000)
  return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function calculateETA() {
  if (progress.value === 0 || !startTime.value) {
    eta.value = 'Calculating...'
    return
  }

  const elapsed = Date.now() - startTime.value
  const estimatedTotal = elapsed / (progress.value / 100)
  const remaining = estimatedTotal - elapsed

  if (remaining < 60000) {
    eta.value = `${Math.round(remaining / 1000)}s remaining`
  } else {
    eta.value = `${Math.round(remaining / 60000)}m remaining`
  }
}

async function startAnalysis() {
  if (analysisStarted.value) return
  analysisStarted.value = true
  startTime.value = Date.now()

  try {
    // Trigger Modal processing via Convex action
    await client.action(api.videos.processVideo, {
      videoId: convexVideoId.value,
    })
  } catch (error) {
    // Error handling is done via the real-time status updates
    console.error('Failed to start analysis:', error)
  }
}

function cancelAnalysis() {
  emit('cancel')
}

onMounted(() => {
  startAnalysis()
})
</script>

<template>
  <div class="analysis-progress">
    <div class="progress-header">
      <div class="file-info">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="23 7 16 12 23 17 23 7" />
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
        </svg>
        <span class="filename">{{ filename }}</span>
      </div>
      <button
        v-if="status !== 'complete' && status !== 'error'"
        class="cancel-btn"
        @click="cancelAnalysis"
      >
        Cancel
      </button>
    </div>

    <div class="progress-content">
      <div class="progress-visual">
        <svg class="progress-ring" viewBox="0 0 120 120">
          <circle
            class="progress-ring-bg"
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke-width="8"
          />
          <circle
            class="progress-ring-fill"
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke-width="8"
            :stroke-dasharray="327"
            :stroke-dashoffset="327 - (327 * progressPercent / 100)"
          />
        </svg>
        <div class="progress-text">
          <span class="progress-value">{{ progressPercent }}%</span>
          <span class="progress-label">{{ status === 'complete' ? 'Done' : 'Progress' }}</span>
        </div>
      </div>

      <div class="progress-details">
        <div class="status-row">
          <div class="status-indicator" :class="status">
            <div v-if="status === 'analyzing' || status === 'processing' || status === 'connecting'" class="pulse" />
            <svg v-if="status === 'complete'" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <svg v-if="status === 'error'" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </div>
          <span class="status-text">{{ statusMessage }}</span>
        </div>

        <div v-if="status === 'analyzing'" class="progress-stats">
          <div class="stat">
            <span class="stat-label">Current Frame</span>
            <span class="stat-value">{{ currentFrame }} / {{ totalFrames }}</span>
          </div>
          <div class="stat">
            <span class="stat-label">ETA</span>
            <span class="stat-value">{{ eta }}</span>
          </div>
        </div>

        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: `${progressPercent}%` }" />
          </div>
        </div>

        <div class="analysis-steps">
          <div class="step" :class="{ active: status !== 'connecting', complete: progress > 0 }">
            <div class="step-icon">1</div>
            <span>Initialize</span>
          </div>
          <div class="step-connector" :class="{ active: progress > 0 }" />
          <div class="step" :class="{ active: progress > 0, complete: progress > 30 }">
            <div class="step-icon">2</div>
            <span>Pose Detection</span>
          </div>
          <div class="step-connector" :class="{ active: progress > 30 }" />
          <div class="step" :class="{ active: progress > 30, complete: progress > 60 }">
            <div class="step-icon">3</div>
            <span>Speed Calculation</span>
          </div>
          <div class="step-connector" :class="{ active: progress > 60 }" />
          <div class="step" :class="{ active: progress > 60, complete: progress >= 100 }">
            <div class="step-icon">4</div>
            <span>Finalize</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Processing Logs Container -->
    <div class="logs-section">
      <div class="logs-header" @click="showLogs = !showLogs">
        <div class="logs-title">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <span>Processing Logs</span>
          <span class="log-count">{{ processingLogs.length }}</span>
        </div>
        <svg
          class="chevron"
          :class="{ rotated: !showLogs }"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>
      
      <div v-show="showLogs" ref="logsContainerRef" class="logs-container">
        <div v-if="processingLogs.length === 0" class="logs-empty">
          Waiting for processing logs...
        </div>
        <div
          v-for="log in processingLogs"
          :key="log.id"
          class="log-entry"
          :class="[`log-${log.level}`, `category-${log.category}`]"
        >
          <span class="log-category-icon" :title="log.category">{{ getCategoryIcon(log.category) }}</span>
          <span class="log-icon">{{ getLogIcon(log.level) }}</span>
          <span class="log-message">{{ log.message }}</span>
          <span class="log-time">{{ formatTimestamp(log.timestamp) }}</span>
        </div>
      </div>
    </div>

    <div class="progress-footer">
      <p class="tip">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        Tip: Analysis time depends on video length and resolution. Shorter clips process faster.
      </p>
    </div>
  </div>
</template>

<style scoped>
.analysis-progress {
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  padding: 24px;
  background: #141414;
  border-radius: 0;
  border: 1px solid #222;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #fff;
}

.file-info svg {
  width: 24px;
  height: 24px;
  color: #22c55e;
}

.filename {
  font-weight: 500;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cancel-btn {
  padding: 8px 16px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  color: #ef4444;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn:hover {
  background: #222;
  border-color: #ef4444;
}

.progress-content {
  display: flex;
  gap: 32px;
  margin-bottom: 24px;
}

.progress-visual {
  position: relative;
  width: 120px;
  height: 120px;
  flex-shrink: 0;
}

.progress-ring {
  transform: rotate(-90deg);
}

.progress-ring-bg {
  stroke: #222;
}

.progress-ring-fill {
  stroke: #22c55e;
  stroke-linecap: square;
  transition: stroke-dashoffset 0.3s ease;
}

.progress-text {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.progress-value {
  color: #fff;
  font-size: 1.75rem;
  font-weight: 700;
}

.progress-label {
  color: #666;
  font-size: 0.75rem;
}

.progress-details {
  flex: 1;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.status-indicator {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: #1a1a1a;
}

.status-indicator.analyzing,
.status-indicator.processing,
.status-indicator.connecting {
  background: #1a1a1a;
}

.status-indicator.complete {
  background: #1a1a1a;
  color: #22c55e;
}

.status-indicator.error {
  background: #1a1a1a;
  color: #ef4444;
}

.status-indicator svg {
  width: 14px;
  height: 14px;
}

.pulse {
  width: 10px;
  height: 10px;
  background: #22c55e;
  border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

.status-text {
  color: #888;
  font-size: 0.875rem;
}

.progress-stats {
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
}

.stat {
  display: flex;
  flex-direction: column;
}

.stat-label {
  color: #666;
  font-size: 0.75rem;
}

.stat-value {
  color: #fff;
  font-size: 0.875rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.progress-bar-container {
  margin-bottom: 20px;
}

.progress-bar {
  height: 6px;
  background: #1a1a1a;
  border-radius: 0;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #22c55e;
  border-radius: 0;
  transition: width 0.3s ease;
}

.analysis-steps {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  opacity: 0.4;
  transition: opacity 0.3s ease;
}

.step.active {
  opacity: 0.7;
}

.step.complete {
  opacity: 1;
}

.step-icon {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a1a;
  border-radius: 50%;
  color: #888;
  font-size: 0.75rem;
  font-weight: 600;
}

.step.complete .step-icon {
  background: #1a1a1a;
  color: #22c55e;
}

.step span {
  color: #666;
  font-size: 0.625rem;
  text-align: center;
}

.step-connector {
  flex: 1;
  height: 2px;
  background: #222;
  margin: 0 4px;
  margin-bottom: 18px;
  transition: background 0.3s ease;
}

.step-connector.active {
  background: #22c55e;
}

.progress-footer {
  padding-top: 16px;
  border-top: 1px solid #222;
}

.tip {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
  font-size: 0.75rem;
  margin: 0;
}

.tip svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

/* Processing Logs Styles */
.logs-section {
  margin-top: 20px;
  margin-bottom: 16px;
  border: 1px solid #222;
  border-radius: 0;
  overflow: hidden;
  background: #0d0d0d;
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #1a1a1a;
  cursor: pointer;
  transition: background 0.2s ease;
}

.logs-header:hover {
  background: #222;
}

.logs-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #888;
  font-size: 0.875rem;
  font-weight: 500;
}

.logs-title svg {
  width: 16px;
  height: 16px;
  color: #22c55e;
}

.log-count {
  background: #1a1a1a;
  color: #22c55e;
  padding: 2px 8px;
  border-radius: 0;
  font-size: 0.75rem;
  border: 1px solid #333;
}

.chevron {
  width: 18px;
  height: 18px;
  color: #666;
  transition: transform 0.2s ease;
}

.chevron.rotated {
  transform: rotate(-90deg);
}

.logs-container {
  max-height: 200px;
  overflow-y: auto;
  padding: 8px;
  font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
  font-size: 0.75rem;
}

.logs-container::-webkit-scrollbar {
  width: 6px;
}

.logs-container::-webkit-scrollbar-track {
  background: #0d0d0d;
}

.logs-container::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 0;
}

.logs-container::-webkit-scrollbar-thumb:hover {
  background: #444;
}

.logs-empty {
  color: #666;
  text-align: center;
  padding: 16px;
  font-style: italic;
}

.log-entry {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 0;
  margin-bottom: 2px;
  background: #141414;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.log-entry:last-child {
  margin-bottom: 0;
}

.log-category-icon {
  font-size: 0.875rem;
  flex-shrink: 0;
  opacity: 0.7;
}

.log-icon {
  flex-shrink: 0;
  width: 14px;
  text-align: center;
}

.log-message {
  flex: 1;
  color: #ccc;
  word-break: break-word;
}

.log-time {
  flex-shrink: 0;
  color: #444;
  font-size: 0.625rem;
}

/* Log level colors */
.log-info .log-icon {
  color: #3b82f6;
}

.log-success .log-icon {
  color: #22c55e;
}

.log-warning .log-icon {
  color: #f59e0b;
}

.log-error .log-icon {
  color: #ef4444;
}

.log-debug .log-icon {
  color: #a855f7;
}

/* Log level backgrounds */
.log-success {
  background: #141414;
  border-left: 2px solid #22c55e;
}

.log-warning {
  background: #141414;
  border-left: 2px solid #f59e0b;
}

.log-error {
  background: #141414;
  border-left: 2px solid #ef4444;
}

/* Category indicator */
.category-modal .log-category-icon {
  color: #a855f7;
}

.category-model .log-category-icon {
  color: #06b6d4;
}

.category-detection .log-category-icon {
  color: #3b82f6;
}

.category-court .log-category-icon {
  color: #f59e0b;
}

.category-processing .log-category-icon {
  color: #888;
}
</style>
