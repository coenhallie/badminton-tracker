<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { supabase } from '@/lib/supabase'

const props = defineProps<{
  videoId: string
}>()

const emit = defineEmits<{
  continue: []
  done: []
}>()

// Matches the actual rally_clips schema from
// supabase/migrations/0001_initial_schema.sql plus extensions in 0004_*.sql.
interface RallyClipRow {
  id: string
  video_id: string
  owner_id: string
  rally_index: number
  start_timestamp: number
  end_timestamp: number
  duration_seconds: number
  clip_storage_path: string
  thumbnail_storage_path: string | null
  title: string | null
  annotation_count: number
  created_at: string
}

// Row enriched with the signed URL we resolved for the thumbnail or clip,
// plus a flag that tells the template which element to render.
interface RallyClipView extends RallyClipRow {
  previewUrl: string | null
  previewKind: 'thumbnail' | 'video' | 'none'
}

const loading = ref(true)
const errorMsg = ref<string | null>(null)
const rallies = ref<RallyClipView[]>([])
const matchDurationSeconds = ref<number | null>(null)
const continuing = ref(false)

function formatDuration(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}m ${sec.toString().padStart(2, '0')}s`
}

const summaryLine = computed(() => {
  const n = rallies.value.length
  if (matchDurationSeconds.value == null) {
    return `${n} ${n === 1 ? 'rally' : 'rallies'} detected`
  }
  return `${n} ${n === 1 ? 'rally' : 'rallies'} detected · total match duration ${formatDuration(matchDurationSeconds.value)}`
})

async function signPreview(row: RallyClipRow): Promise<RallyClipView> {
  // Prefer the cheap thumbnail when present; otherwise fall back to the
  // clip itself (rendered as a <video preload="metadata"> element).
  if (row.thumbnail_storage_path) {
    const { data, error } = await supabase.storage
      .from('thumbnails')
      .createSignedUrl(row.thumbnail_storage_path, 3600)
    if (!error && data?.signedUrl) {
      return { ...row, previewUrl: data.signedUrl, previewKind: 'thumbnail' }
    }
    // fall through to clip fallback if thumbnail signing failed
  }
  const { data, error } = await supabase.storage
    .from('clips')
    .createSignedUrl(row.clip_storage_path, 3600)
  if (!error && data?.signedUrl) {
    return { ...row, previewUrl: data.signedUrl, previewKind: 'video' }
  }
  return { ...row, previewUrl: null, previewKind: 'none' }
}

async function loadMatchDuration(videoId: string): Promise<number | null> {
  // Pull the phase-1 results JSON for total match duration. A failure here
  // is non-fatal — the summary header just omits the duration.
  try {
    const { data: row, error } = await supabase
      .from('videos')
      .select('results_storage_path')
      .eq('id', videoId)
      .single()
    if (error || !row?.results_storage_path) return null

    const { data: signed, error: e2 } = await supabase.storage
      .from('results')
      .createSignedUrl(row.results_storage_path, 3600)
    if (e2 || !signed?.signedUrl) return null

    const res = await fetch(signed.signedUrl)
    if (!res.ok) return null
    const json = (await res.json()) as {
      video_metadata?: { duration_seconds?: number | null }
    }
    return json.video_metadata?.duration_seconds ?? null
  } catch {
    return null
  }
}

async function loadRallies() {
  loading.value = true
  errorMsg.value = null
  try {
    const { data, error } = await supabase
      .from('rally_clips')
      .select(
        'id, video_id, owner_id, rally_index, start_timestamp, end_timestamp, duration_seconds, clip_storage_path, thumbnail_storage_path, title, annotation_count, created_at',
      )
      .eq('video_id', props.videoId)
      .order('rally_index', { ascending: true })

    if (error) throw error
    const rows = (data ?? []) as RallyClipRow[]

    const [signed, duration] = await Promise.all([
      Promise.all(rows.map(signPreview)),
      loadMatchDuration(props.videoId),
    ])

    rallies.value = signed
    matchDurationSeconds.value = duration
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : 'Failed to load rallies'
  } finally {
    loading.value = false
  }
}

function onContinue() {
  if (continuing.value) return
  continuing.value = true
  emit('continue')
  // Parent normally unmounts this component immediately on transition. If
  // that doesn't happen (e.g., parent's start-analytics request failed
  // before re-rendering), unlock the button so the user can retry rather
  // than staying stranded on "Starting…".
  window.setTimeout(() => { continuing.value = false }, 2000)
}

function onDone() {
  emit('done')
}

onMounted(loadRallies)
</script>

<template>
  <div class="rally-review">
    <header class="review-header">
      <h2>Rally Review</h2>
      <p class="subtitle">{{ summaryLine }}</p>
    </header>

    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <p>Loading rallies…</p>
    </div>

    <div v-else-if="errorMsg" class="error-state">
      <p>{{ errorMsg }}</p>
      <button class="secondary" @click="loadRallies">Retry</button>
    </div>

    <div v-else-if="rallies.length === 0" class="empty-state">
      <p>No rallies were detected for this video.</p>
    </div>

    <ul v-else class="rally-list">
      <li v-for="rally in rallies" :key="rally.id" class="rally-row">
        <div class="rally-preview">
          <img
            v-if="rally.previewKind === 'thumbnail' && rally.previewUrl"
            :src="rally.previewUrl"
            :alt="`Rally ${rally.rally_index} thumbnail`"
            loading="lazy"
          />
          <video
            v-else-if="rally.previewKind === 'video' && rally.previewUrl"
            :src="rally.previewUrl"
            preload="metadata"
            muted
            playsinline
          ></video>
          <div v-else class="preview-fallback">No preview</div>
        </div>
        <div class="rally-meta">
          <div class="rally-title">
            <span class="rally-number">#{{ rally.rally_index }}</span>
            <span class="rally-name">{{ rally.title ?? `Rally #${rally.rally_index}` }}</span>
          </div>
          <div class="rally-range">
            {{ formatDuration(rally.start_timestamp) }} — {{ formatDuration(rally.end_timestamp) }}
          </div>
        </div>
      </li>
    </ul>

    <footer class="review-footer">
      <button class="secondary" :disabled="continuing" @click="onDone">
        Done for now
      </button>
      <button class="primary" :disabled="continuing" @click="onContinue">
        {{ continuing ? 'Starting…' : 'Continue with full analytics' }}
      </button>
    </footer>
  </div>
</template>

<style scoped>
.rally-review {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px 16px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.review-header {
  text-align: center;
}

.review-header h2 {
  color: var(--color-accent);
  font-size: 1.5rem;
  margin: 0 0 8px 0;
}

.review-header .subtitle {
  color: rgba(255, 255, 255, 0.6);
  margin: 0;
}

.loading-state,
.error-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  gap: 16px;
  color: rgba(255, 255, 255, 0.7);
}

.error-state {
  color: var(--color-error);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--color-border-secondary);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.rally-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rally-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  background: var(--color-bg);
  border: 1px solid var(--color-border-secondary);
}

.rally-preview {
  width: 160px;
  height: 90px;
  flex-shrink: 0;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.rally-preview img,
.rally-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.preview-fallback {
  color: rgba(255, 255, 255, 0.4);
  font-size: 0.8rem;
}

.rally-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 1;
}

.rally-title {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.rally-number {
  color: var(--color-accent);
  font-weight: bold;
  font-size: 0.9rem;
}

.rally-name {
  color: rgba(255, 255, 255, 0.9);
  font-size: 1rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rally-range {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.85rem;
  font-variant-numeric: tabular-nums;
}

.review-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-secondary);
}

.review-footer button {
  padding: 10px 20px;
  font-weight: bold;
  font-size: 0.95rem;
  cursor: pointer;
  border: 1px solid var(--color-border-secondary);
  transition: all 0.2s ease;
  background: var(--color-bg-tertiary);
  color: rgba(255, 255, 255, 0.9);
}

.review-footer button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.review-footer button.primary {
  background: #001a00;
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.review-footer button.primary:hover:not(:disabled) {
  background: #002a00;
}

.review-footer button.secondary:hover:not(:disabled) {
  background: var(--color-bg-hover);
  border-color: var(--color-accent);
}

@media (max-width: 600px) {
  .rally-row {
    flex-direction: column;
    align-items: stretch;
  }

  .rally-preview {
    width: 100%;
    height: 180px;
  }

  .review-footer {
    flex-direction: column-reverse;
  }
}
</style>
