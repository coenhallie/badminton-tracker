<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, nextTick } from 'vue'
import { analyzeVideo, AnalysisProgressSocket } from '@/services/api'
import type { AnalysisResult, ProgressUpdate, AnalysisConfig, ProcessingLog, LogLevel, LogCategory } from '@/types/analysis'

const props = defineProps<{
  videoId: string
  filename: string
}>()

const emit = defineEmits<{
  complete: [result: AnalysisResult]
  error: [message: string]
  cancel: []
}>()

const progress = ref(0)
const currentFrame = ref(0)
const totalFrames = ref(0)
const status = ref<'connecting' | 'analyzing' | 'processing' | 'complete' | 'error'>('connecting')
const errorMessage = ref('')
const startTime = ref<number>(0)
const eta = ref<string>('')

// Processing logs
const processingLogs = ref<ProcessingLog[]>([])
const logIdCounter = ref(0)
const logsContainerRef = ref<HTMLElement | null>(null)
const showLogs = ref(true)

let socket: AnalysisProgressSocket | null = null

const progressPercent = computed(() => Math.min(100, Math.round(progress.value)))

const statusMessage = computed(() => {
  switch (status.value) {
    case 'connecting':
      return 'Connecting to analysis server...'
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

// Add a log entry
function addLog(message: string, level: LogLevel = 'info', category: LogCategory = 'processing', timestamp?: number) {
  const log: ProcessingLog = {
    id: ++logIdCounter.value,
    message,
    level,
    category,
    timestamp: timestamp || Date.now() / 1000
  }
  processingLogs.value.push(log)
  
  // Auto-scroll to bottom
  nextTick(() => {
    if (logsContainerRef.value) {
      logsContainerRef.value.scrollTop = logsContainerRef.value.scrollHeight
    }
  })
  
  // Keep max 50 logs to prevent memory issues
  if (processingLogs.value.length > 50) {
    processingLogs.value.shift()
  }
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

function handleProgress(update: ProgressUpdate) {
  if (update.type === 'progress' && update.progress !== undefined) {
    progress.value = update.progress
    currentFrame.value = update.frame || 0
    totalFrames.value = update.total_frames || 0
    status.value = 'analyzing'
    calculateETA()
  } else if (update.type === 'log' && update.message) {
    // Handle log messages from backend
    addLog(
      update.message,
      update.level || 'info',
      update.category || 'processing',
      update.timestamp
    )
  }
}

function handleError(error: Error) {
  status.value = 'error'
  errorMessage.value = error.message
  emit('error', error.message)
}

function handleComplete() {
  status.value = 'complete'
}

async function startAnalysis() {
  startTime.value = Date.now()

  // Connect WebSocket for progress updates
  socket = new AnalysisProgressSocket(
    props.videoId,
    handleProgress,
    handleError,
    handleComplete
  )

  socket.connect()

  // Start the analysis
  try {
    status.value = 'processing'

    const config: Partial<AnalysisConfig> = {
      fps_sample_rate: 1,
      confidence_threshold: 0.5,
      track_shuttle: true,
      calculate_speeds: true
      // Note: Court detection model selection removed - using manual keypoints only
    }
    
    // Log that manual keypoints should be used for court calibration
    addLog('Court calibration: Use manual keypoints for accurate measurements', 'info', 'court')

    const response = await analyzeVideo(props.videoId, config)

    if (response.status === 'completed' && response.result) {
      status.value = 'complete'
      progress.value = 100
      emit('complete', response.result)
    }
  } catch (error) {
    status.value = 'error'
    errorMessage.value = error instanceof Error ? error.message : 'Analysis failed'
    emit('error', errorMessage.value)
  }
}

function cancelAnalysis() {
  if (socket) {
    socket.disconnect()
    socket = null
  }
  emit('cancel')
}

onMounted(() => {
  startAnalysis()
})

onUnmounted(() => {
  if (socket) {
    socket.disconnect()
  }
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
  background: linear-gradient(145deg, #1a1f2e, #242b3d);
  border-radius: 16px;
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
  color: #e2e8f0;
}

.file-info svg {
  width: 24px;
  height: 24px;
  color: #667eea;
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
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 6px;
  color: #ef4444;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn:hover {
  background: rgba(239, 68, 68, 0.3);
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
  stroke: rgba(255, 255, 255, 0.1);
}

.progress-ring-fill {
  stroke: url(#gradient);
  stroke-linecap: round;
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
  color: #e2e8f0;
  font-size: 1.75rem;
  font-weight: 700;
}

.progress-label {
  color: #718096;
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
  background: rgba(102, 126, 234, 0.2);
}

.status-indicator.analyzing,
.status-indicator.processing,
.status-indicator.connecting {
  background: rgba(102, 126, 234, 0.2);
}

.status-indicator.complete {
  background: rgba(72, 187, 120, 0.2);
  color: #48bb78;
}

.status-indicator.error {
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
}

.status-indicator svg {
  width: 14px;
  height: 14px;
}

.pulse {
  width: 10px;
  height: 10px;
  background: #667eea;
  border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

.status-text {
  color: #a0aec0;
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
  color: #718096;
  font-size: 0.75rem;
}

.stat-value {
  color: #e2e8f0;
  font-size: 0.875rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.progress-bar-container {
  margin-bottom: 20px;
}

.progress-bar {
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 3px;
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
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  color: #a0aec0;
  font-size: 0.75rem;
  font-weight: 600;
}

.step.complete .step-icon {
  background: rgba(102, 126, 234, 0.3);
  color: #667eea;
}

.step span {
  color: #718096;
  font-size: 0.625rem;
  text-align: center;
}

.step-connector {
  flex: 1;
  height: 2px;
  background: rgba(255, 255, 255, 0.1);
  margin: 0 4px;
  margin-bottom: 18px;
  transition: background 0.3s ease;
}

.step-connector.active {
  background: rgba(102, 126, 234, 0.5);
}

.progress-footer {
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.tip {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #718096;
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
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.2);
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: background 0.2s ease;
}

.logs-header:hover {
  background: rgba(255, 255, 255, 0.08);
}

.logs-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #a0aec0;
  font-size: 0.875rem;
  font-weight: 500;
}

.logs-title svg {
  width: 16px;
  height: 16px;
  color: #667eea;
}

.log-count {
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.75rem;
}

.chevron {
  width: 18px;
  height: 18px;
  color: #718096;
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
  background: rgba(0, 0, 0, 0.2);
}

.logs-container::-webkit-scrollbar-thumb {
  background: rgba(102, 126, 234, 0.3);
  border-radius: 3px;
}

.logs-container::-webkit-scrollbar-thumb:hover {
  background: rgba(102, 126, 234, 0.5);
}

.logs-empty {
  color: #718096;
  text-align: center;
  padding: 16px;
  font-style: italic;
}

.log-entry {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 4px;
  margin-bottom: 2px;
  background: rgba(255, 255, 255, 0.02);
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
  color: #cbd5e0;
  word-break: break-word;
}

.log-time {
  flex-shrink: 0;
  color: #4a5568;
  font-size: 0.625rem;
}

/* Log level colors */
.log-info .log-icon {
  color: #63b3ed;
}

.log-success .log-icon {
  color: #68d391;
}

.log-warning .log-icon {
  color: #f6ad55;
}

.log-error .log-icon {
  color: #fc8181;
}

.log-debug .log-icon {
  color: #b794f4;
}

/* Log level backgrounds */
.log-success {
  background: rgba(104, 211, 145, 0.1);
}

.log-warning {
  background: rgba(246, 173, 85, 0.1);
}

.log-error {
  background: rgba(252, 129, 129, 0.1);
}

/* Category indicator */
.category-modal .log-category-icon {
  color: #b794f4;
}

.category-model .log-category-icon {
  color: #4fd1c5;
}

.category-detection .log-category-icon {
  color: #63b3ed;
}

.category-court .log-category-icon {
  color: #f6ad55;
}

.category-processing .log-category-icon {
  color: #a0aec0;
}
</style>
