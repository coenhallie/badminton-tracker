<script setup lang="ts">
import { ref, computed } from 'vue'
import { useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { UploadResponse } from '@/types/analysis'

const client = useConvexClient()

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

  // Max file size: 1GB (Convex limit)
  const maxSize = 1024 * 1024 * 1024
  if (file.size > maxSize) {
    emit('error', 'File is too large. Maximum size is 1GB.')
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
    // Simulate progress for UX
    const progressInterval = setInterval(() => {
      if (uploadProgress.value < 90) {
        uploadProgress.value += Math.random() * 10
      }
    }, 200)

    // Step 1: Generate upload URL from Convex
    const uploadUrl = await client.mutation(api.videos.generateUploadUrl, {})
    
    // Step 2: Upload file directly to Convex storage
    const uploadResponse = await fetch(uploadUrl, {
      method: 'POST',
      headers: { 'Content-Type': selectedFile.value.type },
      body: selectedFile.value,
    })
    
    if (!uploadResponse.ok) {
      throw new Error('Failed to upload file to storage')
    }
    
    const { storageId } = await uploadResponse.json()
    
    // Step 3: Create video record in Convex database
    const videoId = await client.mutation(api.videos.createVideo, {
      storageId,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
    })

    clearInterval(progressInterval)
    uploadProgress.value = 100

    // Emit response in the same format as before for compatibility
    emit('uploaded', {
      video_id: videoId,
      filename: selectedFile.value.name,
      size: selectedFile.value.size,
      status: 'uploaded'
    })
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
  border: 2px dashed #333;
  border-radius: 0;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #141414;
}

.upload-zone:hover,
.upload-zone.dragging {
  border-color: #22c55e;
  background: #1a1a1a;
}

.upload-zone.dragging {
  transform: scale(1.02);
}

.upload-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  color: #22c55e;
}

.upload-icon svg {
  width: 100%;
  height: 100%;
}

.upload-zone h3 {
  color: #fff;
  font-size: 1.25rem;
  margin-bottom: 8px;
}

.upload-zone p {
  color: #888;
  margin-bottom: 8px;
}

.file-types {
  font-size: 0.875rem;
  color: #666;
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
  background: #141414;
  border-radius: 0;
  padding: 24px;
  border: 1px solid #222;
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
  background: #1a1a1a;
  border-radius: 0;
  color: #22c55e;
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
  color: #fff;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-size {
  color: #666;
  font-size: 0.875rem;
}

.remove-btn {
  width: 36px;
  height: 36px;
  padding: 8px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  color: #ef4444;
  cursor: pointer;
  transition: all 0.2s ease;
}

.remove-btn:hover {
  background: #222;
  border-color: #ef4444;
}

.progress-bar {
  height: 8px;
  background: #1a1a1a;
  border-radius: 0;
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-fill {
  height: 100%;
  background: #22c55e;
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
  background: #22c55e;
  border: none;
  border-radius: 0;
  color: #000;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-btn:hover {
  background: #16a34a;
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
  color: #888;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #222;
  border-top-color: #22c55e;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
