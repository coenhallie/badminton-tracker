<script setup lang="ts">
import { ref, computed } from 'vue'
import { useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { Id } from '../../convex/_generated/dataModel'
import type { UploadResponse } from '@/types/analysis'

const client = useConvexClient()

const emit = defineEmits<{
  uploaded: [response: UploadResponse]
  error: [message: string]
}>()

const isDragging = ref(false)
const isUploading = ref(false)
const uploadProgress = ref(0)
const uploadSpeed = ref('')
const selectedFile = ref<File | null>(null)
const activeXhr = ref<XMLHttpRequest | null>(null)
const retryCount = ref(0)
const MAX_RETRIES = 2
const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000 // 10 minutes

// Monotonic counter used to invalidate stale XHR progress events. Each
// uploadFileWithProgress call captures the current generation; if a later
// upload supersedes it (retry, or double-click before Vue hides the button)
// the captured generation no longer matches the module-level one and that
// XHR's listeners stop writing to uploadProgress. Without this, browsers
// can flush queued `progress` events from an aborted XHR after the new
// XHR has already started, producing the interleaved "53 → 21 → 55 → 24"
// pattern where two upload trajectories alternate in the UI.
let uploadGeneration = 0

const allowedTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'video/webm']

const formattedSize = computed(() => {
  if (!selectedFile.value) return ''
  const bytes = selectedFile.value.size
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
})

function handleDragOver(event: DragEvent) {
  event.preventDefault()
  isDragging.value = true
}

function handleDragLeave() {
  isDragging.value = false
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false

  const files = event.dataTransfer?.files
  if (files && files.length > 0) {
    const file = files[0]
    if (file) validateAndSetFile(file)
  }
}

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (input.files && input.files.length > 0) {
    const file = input.files[0]
    if (file) validateAndSetFile(file)
  }
}

function validateAndSetFile(file: File) {
  if (!allowedTypes.includes(file.type)) {
    emit('error', `Invalid file type. Please upload a video file (MP4, MPEG, MOV, AVI, or WebM).`)
    return
  }

  // Max file size: 1GB (Convex limit)
  const maxSize = 1024 * 1024 * 1024
  if (file.size > maxSize) {
    emit('error', 'File is too large. Maximum size is 1GB.')
    return
  }

  const minSize = 1024 * 100
  if (file.size < minSize) {
    emit('error', 'File is too small. Please upload a valid video file.')
    return
  }

  selectedFile.value = file
}

function removeFile() {
  selectedFile.value = null
  uploadProgress.value = 0
  uploadSpeed.value = ''
}

function cancelUpload() {
  if (activeXhr.value) {
    activeXhr.value.abort()
    activeXhr.value = null
  }
  isUploading.value = false
  uploadProgress.value = 0
  uploadSpeed.value = ''
  retryCount.value = 0
}

function uploadFileWithProgress(url: string, file: File): Promise<{ storageId: string }> {
  // Capture this upload's generation. If a newer upload starts (retry,
  // double-click, etc.) it bumps uploadGeneration and our isCurrent()
  // check will return false, causing this XHR's queued progress/speed
  // events to stop writing to shared UI state.
  const generation = ++uploadGeneration
  const isCurrent = () => generation === uploadGeneration

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    activeXhr.value = xhr
    let lastLoaded = 0
    let lastTime = Date.now()

    xhr.upload.addEventListener('progress', (e) => {
      if (!isCurrent()) return
      if (e.lengthComputable) {
        uploadProgress.value = (e.loaded / e.total) * 100

        const now = Date.now()
        const elapsed = (now - lastTime) / 1000
        if (elapsed >= 0.5) {
          const bytesPerSec = (e.loaded - lastLoaded) / elapsed
          lastLoaded = e.loaded
          lastTime = now
          if (bytesPerSec > 0) {
            if (bytesPerSec >= 1024 * 1024) {
              uploadSpeed.value = `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`
            } else {
              uploadSpeed.value = `${(bytesPerSec / 1024).toFixed(0)} KB/s`
            }
          }
        }
      }
    })

    xhr.addEventListener('load', () => {
      if (activeXhr.value === xhr) activeXhr.value = null
      if (!isCurrent()) {
        // We were superseded mid-flight — our upload happened but the
        // storageId we got belongs to a stale attempt. Reject so the
        // caller's await throws, but don't pollute the UI state.
        reject(new Error('Upload superseded'))
        return
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(new Error('Invalid response from storage'))
        }
      } else {
        reject(new Error(`Upload failed (HTTP ${xhr.status})`))
      }
    })

    xhr.addEventListener('error', () => {
      if (activeXhr.value === xhr) activeXhr.value = null
      reject(new Error('Network error during upload'))
    })

    xhr.addEventListener('abort', () => {
      if (activeXhr.value === xhr) activeXhr.value = null
      reject(new Error('Upload cancelled'))
    })

    xhr.addEventListener('timeout', () => {
      if (activeXhr.value === xhr) activeXhr.value = null
      reject(new Error('Upload timed out'))
    })

    xhr.timeout = UPLOAD_TIMEOUT_MS
    xhr.open('POST', url)
    xhr.setRequestHeader('Content-Type', file.type)
    xhr.send(file)
  })
}

async function startUpload() {
  if (!selectedFile.value) return

  // Guard: second click arriving before Vue hides the button (the click
  // handler fires synchronously, DOM updates on the next tick). Without
  // this, two startUpload calls overlap and their XHRs race.
  if (isUploading.value) return

  // If any XHR from an earlier attempt is somehow still live, abort it
  // before starting a new one. Combined with the generation guard in
  // uploadFileWithProgress, this prevents interleaved progress updates.
  if (activeXhr.value) {
    activeXhr.value.abort()
    activeXhr.value = null
  }

  isUploading.value = true
  uploadProgress.value = 0
  uploadSpeed.value = ''
  retryCount.value = 0

  await attemptUpload()
}

async function attemptUpload() {
  if (!selectedFile.value) return

  try {
    // Step 1: Generate upload URL from Convex
    const uploadUrl = await client.mutation(api.videos.generateUploadUrl, {})

    // Step 2: Upload file with real progress tracking
    const { storageId } = await uploadFileWithProgress(uploadUrl, selectedFile.value)

    // Step 3: Create video record in Convex database
    const videoId = await client.mutation(api.videos.createVideo, {
      storageId: storageId as Id<'_storage'>,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
    })

    uploadProgress.value = 100
    uploadSpeed.value = ''

    emit('uploaded', {
      video_id: videoId,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
      status: 'uploaded',
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Upload failed'

    // Don't retry if user cancelled
    if (message === 'Upload cancelled') {
      return
    }

    // Auto-retry on network errors / timeouts
    if (retryCount.value < MAX_RETRIES && (message.includes('Network') || message.includes('timed out'))) {
      retryCount.value++
      uploadProgress.value = 0
      uploadSpeed.value = `Retrying (${retryCount.value}/${MAX_RETRIES})...`
      await new Promise(r => setTimeout(r, 1000 * retryCount.value))
      await attemptUpload()
      return
    }

    emit('error', message)
  } finally {
    if (uploadProgress.value === 100 || !activeXhr.value) {
      isUploading.value = false
      uploadSpeed.value = ''
    }
  }
}
</script>

<template>
  <div class="video-upload">
    <div
      v-if="!selectedFile"
      class="upload-zone"
      :class="{ 'dragging': isDragging }"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
    >
      <div class="upload-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <h3>Upload Badminton Match Video</h3>
      <p>Drag and drop your video file here, or click to browse</p>
      <p class="file-types">Supported formats: MP4, MPEG, MOV, AVI, WebM (max 1GB)</p>
      <input
        type="file"
        accept="video/*"
        @change="handleFileSelect"
        class="file-input"
      />
    </div>

    <div v-else class="file-preview">
      <div class="file-info">
        <div class="file-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7" />
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
          </svg>
        </div>
        <div class="file-details">
          <span class="file-name">{{ selectedFile.name }}</span>
          <span class="file-size">{{ formattedSize }}</span>
        </div>
        <button
          v-if="!isUploading"
          class="remove-btn"
          @click="removeFile"
          title="Remove file"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div v-if="isUploading" class="progress-bar">
        <div class="progress-fill" :style="{ width: `${uploadProgress}%` }"></div>
      </div>

      <button
        v-if="!isUploading"
        class="upload-btn"
        @click="startUpload"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        Upload & Analyze
      </button>

      <div v-else class="uploading-status">
        <div class="upload-status-row">
          <div class="spinner"></div>
          <span>Uploading... {{ Math.round(uploadProgress) }}%</span>
          <span v-if="uploadSpeed" class="upload-speed">{{ uploadSpeed }}</span>
        </div>
        <button class="cancel-btn" @click="cancelUpload">Cancel</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-upload {
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
}

.upload-zone {
  position: relative;
  border: 2px dashed var(--color-border-secondary);
  border-radius: 0;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: var(--color-bg-secondary);
}

.upload-zone:hover,
.upload-zone.dragging {
  border-color: var(--color-accent);
  background: var(--color-bg-tertiary);
}

.upload-zone.dragging {
  transform: scale(1.02);
}

.upload-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  color: var(--color-accent);
}

.upload-icon svg {
  width: 100%;
  height: 100%;
}

.upload-zone h3 {
  color: var(--color-text-heading);
  font-size: 1.25rem;
  margin-bottom: 8px;
}

.upload-zone p {
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

.file-types {
  font-size: 0.875rem;
  color: var(--color-text-tertiary);
}

.file-input {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.file-preview {
  background: var(--color-bg-secondary);
  border-radius: 0;
  padding: 24px;
  border: 1px solid var(--color-border);
}

.file-info {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}

.file-icon {
  width: 48px;
  height: 48px;
  padding: 10px;
  background: var(--color-bg-tertiary);
  border-radius: 0;
  color: var(--color-accent);
}

.file-icon svg {
  width: 100%;
  height: 100%;
}

.file-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.file-name {
  color: var(--color-text-heading);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-size {
  color: var(--color-text-tertiary);
  font-size: 0.875rem;
}

.remove-btn {
  width: 36px;
  height: 36px;
  padding: 8px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-error);
  cursor: pointer;
  transition: all 0.2s ease;
}

.remove-btn:hover {
  background: var(--color-border);
  border-color: var(--color-error);
}

.progress-bar {
  height: 8px;
  background: var(--color-bg-tertiary);
  border-radius: 0;
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-fill {
  height: 100%;
  background: var(--color-accent);
  border-radius: 0;
  transition: width 0.3s ease;
}

.upload-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 24px;
  background: var(--color-accent);
  border: none;
  border-radius: 0;
  color: #000;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-btn:hover {
  background: var(--color-accent-dark);
}

.upload-btn svg {
  width: 20px;
  height: 20px;
}

.uploading-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: var(--color-text-secondary);
}

.upload-status-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.upload-speed {
  font-size: 0.8rem;
  color: var(--color-text-tertiary);
}

.cancel-btn {
  padding: 6px 16px;
  background: transparent;
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn:hover {
  border-color: var(--color-error);
  color: var(--color-error);
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--color-border);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.mode-selector {
  margin-bottom: 20px;
}

.mode-label {
  display: block;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  font-weight: 500;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.mode-options {
  display: flex;
  gap: 8px;
}

.mode-option {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 16px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-border-secondary);
  border-radius: 0;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s ease;
}

.mode-option:hover {
  border-color: var(--color-text-tertiary);
}

.mode-option.active {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
}

.mode-title {
  color: var(--color-text-heading);
  font-weight: 600;
  font-size: 0.9rem;
}

.mode-desc {
  color: var(--color-text-tertiary);
  font-size: 0.75rem;
}

.mode-option.active .mode-title {
  color: var(--color-accent);
}
</style>
