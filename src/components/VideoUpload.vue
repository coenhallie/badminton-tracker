<script setup lang="ts">
import { ref, computed } from 'vue'
import { supabase } from '@/lib/supabase'
import { useSession } from '@/composables/useSession'
import type { UploadResponse } from '@/types/analysis'

const { user } = useSession()

const emit = defineEmits<{
  uploaded: [response: UploadResponse]
  error: [message: string]
}>()

const isDragging = ref(false)
const isUploading = ref(false)
const selectedFile = ref<File | null>(null)
const analysisMode = ref<'rally_only' | 'full'>('full')

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

  // Max file size: 1GB
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
}

async function uploadAndCreate(file: File): Promise<string> {
  if (!user.value) throw new Error('Not signed in')

  const videoId = crypto.randomUUID()
  const path = `${user.value.id}/${videoId}.mp4`

  // 1. Upload bytes to Storage
  const { error: upErr } = await supabase.storage
    .from('videos')
    .upload(path, file, { contentType: file.type, upsert: false })
  if (upErr) throw upErr

  // 2. Insert row (RLS allows because owner_id = auth.uid())
  const { data: row, error: insErr } = await supabase
    .from('videos')
    .insert({
      id: videoId,
      owner_id: user.value.id,
      filename: file.name,
      size: file.size,
      storage_path: path,
      status: 'uploaded',
    })
    .select()
    .single()
  if (insErr) throw insErr

  return row.id as string
}

async function startUpload() {
  if (!selectedFile.value) return

  isUploading.value = true

  try {
    const file = selectedFile.value
    const videoId = await uploadAndCreate(file)

    emit('uploaded', {
      video_id: videoId,
      filename: file.name,
      size: file.size,
      status: 'uploaded',
      analysisMode: analysisMode.value,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Upload failed'
    emit('error', message)
  } finally {
    isUploading.value = false
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

      <!-- Analysis Mode Selector -->
      <div v-if="!isUploading" class="mode-selector">
        <span class="mode-label">Analysis Mode</span>
        <div class="mode-options">
          <button
            class="mode-option"
            :class="{ active: analysisMode === 'rally_only' }"
            @click="analysisMode = 'rally_only'"
            type="button"
          >
            <span class="mode-title">Rally Separation</span>
            <span class="mode-desc">Detect rally boundaries only (faster)</span>
          </button>
          <button
            class="mode-option"
            :class="{ active: analysisMode === 'full' }"
            @click="analysisMode = 'full'"
            type="button"
          >
            <span class="mode-title">Full Analysis</span>
            <span class="mode-desc">Player tracking, poses, speed + rallies</span>
          </button>
        </div>
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
          <span>Uploading...</span>
        </div>
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
