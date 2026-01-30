<script setup lang="ts">
import { ref, computed } from 'vue'
import { uploadVideo } from '@/services/api'
import type { UploadResponse } from '@/types/analysis'

const emit = defineEmits<{
  uploaded: [response: UploadResponse]
  error: [message: string]
}>()

const isDragging = ref(false)
const isUploading = ref(false)
const uploadProgress = ref(0)
const selectedFile = ref<File | null>(null)

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

  // Max file size: 500MB
  const maxSize = 500 * 1024 * 1024
  if (file.size > maxSize) {
    emit('error', 'File is too large. Maximum size is 500MB.')
    return
  }

  selectedFile.value = file
}

function removeFile() {
  selectedFile.value = null
  uploadProgress.value = 0
}

async function startUpload() {
  if (!selectedFile.value) return

  isUploading.value = true
  uploadProgress.value = 0

  try {
    // Simulate progress for UX (actual progress would require XHR)
    const progressInterval = setInterval(() => {
      if (uploadProgress.value < 90) {
        uploadProgress.value += Math.random() * 10
      }
    }, 200)

    const response = await uploadVideo(selectedFile.value)

    clearInterval(progressInterval)
    uploadProgress.value = 100

    emit('uploaded', response)
  } catch (error) {
    emit('error', error instanceof Error ? error.message : 'Upload failed')
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
      <p class="file-types">Supported formats: MP4, MPEG, MOV, AVI, WebM (max 500MB)</p>
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
        <div class="spinner"></div>
        <span>Uploading... {{ Math.round(uploadProgress) }}%</span>
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
  border: 2px dashed #4a5568;
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: linear-gradient(145deg, #1a1f2e, #242b3d);
}

.upload-zone:hover,
.upload-zone.dragging {
  border-color: #667eea;
  background: linear-gradient(145deg, #1e2436, #2a3349);
}

.upload-zone.dragging {
  transform: scale(1.02);
}

.upload-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  color: #667eea;
}

.upload-icon svg {
  width: 100%;
  height: 100%;
}

.upload-zone h3 {
  color: #e2e8f0;
  font-size: 1.25rem;
  margin-bottom: 8px;
}

.upload-zone p {
  color: #a0aec0;
  margin-bottom: 8px;
}

.file-types {
  font-size: 0.875rem;
  color: #718096;
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
  background: linear-gradient(145deg, #1a1f2e, #242b3d);
  border-radius: 12px;
  padding: 24px;
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
  background: rgba(102, 126, 234, 0.2);
  border-radius: 10px;
  color: #667eea;
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
  color: #e2e8f0;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-size {
  color: #718096;
  font-size: 0.875rem;
}

.remove-btn {
  width: 36px;
  height: 36px;
  padding: 8px;
  background: rgba(239, 68, 68, 0.2);
  border: none;
  border-radius: 8px;
  color: #ef4444;
  cursor: pointer;
  transition: all 0.2s ease;
}

.remove-btn:hover {
  background: rgba(239, 68, 68, 0.3);
}

.progress-bar {
  height: 8px;
  background: #2d3748;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.upload-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 24px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
}

.upload-btn svg {
  width: 20px;
  height: 20px;
}

.uploading-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #a0aec0;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #2d3748;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
