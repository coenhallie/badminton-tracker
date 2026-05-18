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
      <div class="android-banner" role="note">
        <svg
          class="android-banner-icon"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <rect x="5" y="2" width="14" height="20" rx="2" ry="2" />
          <line x1="12" y1="18" x2="12.01" y2="18" />
        </svg>
        <span>These rallies are synced to your account — open the Android app to view them.</span>
      </div>
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

    <ul v-else class="rally-grid">
      <li v-for="rally in rallies" :key="rally.id" class="rally-card">
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
          <span class="rally-badge">#{{ rally.rally_index }}</span>
        </div>
        <div class="rally-meta">
          <span class="rally-duration">{{ formatDuration(rally.duration_seconds) }}</span>
          <span class="rally-range">
            {{ formatDuration(rally.start_timestamp) }}–{{ formatDuration(rally.end_timestamp) }}
          </span>
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

.android-banner {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 12px auto 0;
  padding: 8px 14px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-left: 2px solid var(--color-accent);
  color: rgba(255, 255, 255, 0.85);
  font-size: 0.85rem;
  line-height: 1.3;
  text-align: left;
}

.android-banner-icon {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  color: var(--color-accent);
}

@media (max-width: 600px) {
  .android-banner {
    display: flex;
    width: 100%;
  }
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

.rally-grid {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}

.rally-card {
  display: flex;
  flex-direction: column;
  background: var(--color-bg);
  border: 1px solid var(--color-border-secondary);
  overflow: hidden;
  transition: border-color 0.15s ease, transform 0.15s ease;
}

.rally-card:hover {
  border-color: var(--color-accent);
  transform: translateY(-1px);
}

.rally-preview {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
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

.rally-badge {
  position: absolute;
  top: 6px;
  left: 6px;
  padding: 2px 6px;
  background: rgba(0, 0, 0, 0.75);
  color: var(--color-accent);
  font-size: 0.75rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
  border: 1px solid var(--color-border-secondary);
}

.rally-meta {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  min-width: 0;
}

.rally-duration {
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.85rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.rally-range {
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
  .review-footer {
    flex-direction: column-reverse;
  }
}
</style>
