<script setup lang="ts">
import { ref, onMounted, computed, nextTick, watch, toRef } from 'vue'
import { supabase } from '@/lib/supabase'
import { useReactiveRow } from '@/composables/useReactiveRow'
import { useReactiveList } from '@/composables/useReactiveList'
import type { AnalysisResult, ProcessingLog, LogLevel, LogCategory } from '@/types/analysis'

interface Video {
  id: string
  owner_id: string
  filename: string
  size: number
  storage_path: string
  status: 'uploaded' | 'processing' | 'completed' | 'failed'
  progress: number | null
  current_frame: number | null
  total_frames: number | null
  error: string | null
  results_meta: Record<string, any> | null
  results_storage_path: string | null
  processed_video_path: string | null
  skeleton_data_path: string | null
  manual_court_keypoints: Record<string, any> | null
  player_labels: Record<string, any> | null
  created_at: string
  processing_started_at: string | null
  completed_at: string | null
}

interface ProcessingLogRow {
  id: number
  video_id: string
  owner_id: string
  message: string
  level: LogLevel
  category: LogCategory
  timestamp: string
}

const props = defineProps<{
  videoId: string
  filename: string
}>()

const emit = defineEmits<{
  complete: [result: AnalysisResult]
  error: [message: string]
  cancel: []
}>()

const videoIdRef = toRef(props, 'videoId')

// Real-time video query - automatically updates when database changes
const { row: video } = useReactiveRow<Video>('videos', videoIdRef)

// Real-time logs query
const logsFilter = computed(() =>
  videoIdRef.value ? { column: 'video_id', value: videoIdRef.value } : null
)
const { items: logs } = useReactiveList<ProcessingLogRow>(
  'processing_logs',
  logsFilter,
  { orderBy: 'timestamp', ascending: true }
)

// Local state
const startTime = ref<number>(0)
const eta = ref<string>('')
const logsContainerRef = ref<HTMLElement | null>(null)
const showLogs = ref(true)
const analysisStarted = ref(false)

// Computed values from Supabase data
const progress = computed(() => video.value?.progress ?? 0)
const current_frame = computed(() => video.value?.current_frame ?? 0)
const total_frames = computed(() => video.value?.total_frames ?? 0)
const videoStatus = computed(() => video.value?.status ?? 'uploaded')
const errorMessage = computed(() => video.value?.error ?? '')

// Map db status to component status
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

// Convert raw log rows to ProcessingLog format expected by template
const processingLogs = computed<ProcessingLog[]>(() => {
  if (!logs.value) return []
  return logs.value.map((log, index) => ({
    id: index,
    message: log.message,
    level: log.level as LogLevel,
    category: log.category as LogCategory,
    // Postgres timestamptz returned as ISO string -> seconds since epoch
    timestamp: new Date(log.timestamp).getTime() / 1000,
  }))
})

async function fetchResultsJson(): Promise<any | null> {
  if (!video.value?.results_storage_path) return null
  const { data: signed, error } = await supabase
    .storage.from('results')
    .createSignedUrl(video.value.results_storage_path, 3600)
  if (error || !signed) throw error ?? new Error('Could not sign results URL')
  const res = await fetch(signed.signedUrl)
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`)
  return res.json()
}

// Watch for completion or error
watch(videoStatus, async (newStatus) => {
  if (newStatus === 'completed' && video.value) {
    if (video.value.results_storage_path) {
      try {
        let results: Record<string, unknown> | null = null
        let lastErr: unknown = null
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            results = await fetchResultsJson()
            if (results) break
          } catch (err) {
            lastErr = err
            if (attempt < 2) await new Promise(r => setTimeout(r, 1000 * (attempt + 1)))
          }
        }

        if (!results) {
          throw lastErr instanceof Error
            ? lastErr
            : new Error('Failed to fetch results after 3 attempts')
        }

        if (typeof results !== 'object') {
          throw new Error('Results data has unexpected format')
        }

        const analysisResult: AnalysisResult = {
          video_id: props.videoId,
          duration: (results.duration as number) ?? 0,
          fps: (results.fps as number) ?? 0,
          total_frames: (results.total_frames as number) ?? 0,
          processed_frames: (results.processed_frames as number) ?? 0,
          players: (results.players as AnalysisResult['players']) ?? [],
          shuttle: (results.shuttle as AnalysisResult['shuttle']) ?? null,
          skeleton_data: Array.isArray(results.skeleton_data) ? (results.skeleton_data as AnalysisResult['skeleton_data']) : [],
          court_detection: (results.court_detection as AnalysisResult['court_detection']) ?? null,
          player_zone_analytics: (results.player_zone_analytics as AnalysisResult['player_zone_analytics']) ?? null,
          rallies: (results.rallies as AnalysisResult['rallies']) ?? null,
        }
        emit('complete', analysisResult)
      } catch (err) {
        console.error('Failed to fetch results:', err)
        emit('error', err instanceof Error ? err.message : 'Failed to load analysis results')
      }
    } else {
      // Fallback: use metadata if no results path (shouldn't happen normally)
      const meta = video.value.results_meta
      if (meta) {
        const analysisResult: AnalysisResult = {
          video_id: props.videoId,
          duration: (meta.duration as number) ?? 0,
          fps: (meta.fps as number) ?? 0,
          total_frames: (meta.total_frames as number) ?? 0,
          processed_frames: (meta.processed_frames as number) ?? 0,
          players: [],
          shuttle: null,
          skeleton_data: [],
          court_detection: null,
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
    // Trigger backend processing via Supabase Edge Function
    const { error } = await supabase.functions.invoke('process-video', {
      body: { video_id: videoIdRef.value },
    })
    if (error) throw error
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
            <span class="stat-value">{{ current_frame }} / {{ total_frames }}</span>
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
  background: var(--color-bg-secondary);
  border-radius: 0;
  border: 1px solid var(--color-border);
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
  color: var(--color-text-heading);
}

.file-info svg {
  width: 24px;
  height: 24px;
  color: var(--color-accent);
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
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-error);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn:hover {
  background: var(--color-border);
  border-color: var(--color-error);
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
  stroke: var(--color-border);
}

.progress-ring-fill {
  stroke: var(--color-accent);
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
  color: var(--color-text-heading);
  font-size: 1.75rem;
  font-weight: 700;
}

.progress-label {
  color: var(--color-text-tertiary);
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
  background: var(--color-bg-tertiary);
}

.status-indicator.analyzing,
.status-indicator.processing,
.status-indicator.connecting {
  background: var(--color-bg-tertiary);
}

.status-indicator.complete {
  background: var(--color-bg-tertiary);
  color: var(--color-accent);
}

.status-indicator.error {
  background: var(--color-bg-tertiary);
  color: var(--color-error);
}

.status-indicator svg {
  width: 14px;
  height: 14px;
}

.pulse {
  width: 10px;
  height: 10px;
  background: var(--color-accent);
  border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

.status-text {
  color: var(--color-text-secondary);
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
  color: var(--color-text-tertiary);
  font-size: 0.75rem;
}

.stat-value {
  color: var(--color-text-heading);
  font-size: 0.875rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.progress-bar-container {
  margin-bottom: 20px;
}

.progress-bar {
  height: 6px;
  background: var(--color-bg-tertiary);
  border-radius: 0;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--color-accent);
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
  background: var(--color-bg-tertiary);
  border-radius: 50%;
  color: var(--color-text-secondary);
  font-size: 0.75rem;
  font-weight: 600;
}

.step.complete .step-icon {
  background: var(--color-bg-tertiary);
  color: var(--color-accent);
}

.step span {
  color: var(--color-text-tertiary);
  font-size: 0.625rem;
  text-align: center;
}

.step-connector {
  flex: 1;
  height: 2px;
  background: var(--color-border);
  margin: 0 4px;
  margin-bottom: 18px;
  transition: background 0.3s ease;
}

.step-connector.active {
  background: var(--color-accent);
}

.progress-footer {
  padding-top: 16px;
  border-top: 1px solid var(--color-border);
}

.tip {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-tertiary);
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
  border: 1px solid var(--color-border);
  border-radius: 0;
  overflow: hidden;
  background: var(--color-bg);
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--color-bg-tertiary);
  cursor: pointer;
  transition: background 0.2s ease;
}

.logs-header:hover {
  background: var(--color-border);
}

.logs-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  font-weight: 500;
}

.logs-title svg {
  width: 16px;
  height: 16px;
  color: var(--color-accent);
}

.log-count {
  background: var(--color-bg-tertiary);
  color: var(--color-accent);
  padding: 2px 8px;
  border-radius: 0;
  font-size: 0.75rem;
  border: 1px solid var(--color-border-secondary);
}

.chevron {
  width: 18px;
  height: 18px;
  color: var(--color-text-tertiary);
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
  background: var(--color-bg);
}

.logs-container::-webkit-scrollbar-thumb {
  background: var(--color-border-secondary);
  border-radius: 0;
}

.logs-container::-webkit-scrollbar-thumb:hover {
  background: var(--color-border-hover);
}

.logs-empty {
  color: var(--color-text-tertiary);
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
  background: var(--color-bg-secondary);
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
  color: var(--color-border-hover);
  font-size: 0.625rem;
}

/* Log level colors */
.log-info .log-icon {
  color: var(--color-info);
}

.log-success .log-icon {
  color: var(--color-accent);
}

.log-warning .log-icon {
  color: var(--color-warning);
}

.log-error .log-icon {
  color: var(--color-error);
}

.log-debug .log-icon {
  color: #a855f7;
}

/* Log level backgrounds */
.log-success {
  background: var(--color-bg-secondary);
  border-left: 2px solid var(--color-accent);
}

.log-warning {
  background: var(--color-bg-secondary);
  border-left: 2px solid var(--color-warning);
}

.log-error {
  background: var(--color-bg-secondary);
  border-left: 2px solid var(--color-error);
}

/* Category indicator */
.category-modal .log-category-icon {
  color: #a855f7;
}

.category-model .log-category-icon {
  color: #06b6d4;
}

.category-detection .log-category-icon {
  color: var(--color-info);
}

.category-court .log-category-icon {
  color: var(--color-warning);
}

.category-processing .log-category-icon {
  color: var(--color-text-secondary);
}
</style>
