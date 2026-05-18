<script setup lang="ts">
import { computed, toRef } from 'vue'
import { useReactiveRow } from '@/composables/useReactiveRow'
import { useReactiveList } from '@/composables/useReactiveList'
import type { VideoStatus } from '@/types/analysis'

interface VideoRow { id: string; status: VideoStatus }
interface RallyClipRow { id: string }

type StepState = 'pending' | 'active' | 'done' | 'failed'

const props = defineProps<{
  videoId: string
  activePhase: 'phase1' | 'phase2'
}>()

const videoIdRef = toRef(props, 'videoId')
const { row: video } = useReactiveRow<VideoRow>('videos', videoIdRef)

const rallyFilter = computed(() =>
  videoIdRef.value ? { column: 'video_id', value: videoIdRef.value } : null,
)
const { items: rallyClips } = useReactiveList<RallyClipRow>(
  'rally_clips',
  rallyFilter,
)

const status = computed<VideoStatus>(() => video.value?.status ?? 'pending')

// Status → (phase1State, phase2State). Exhaustive over VideoStatus including
// legacy values ('processing', 'failed'). Legacy 'failed' maps to phase-1
// failure (most legacy rows that failed never produced rally clips); legacy
// 'processing' maps to phase-1 active.
const phaseStates = computed<{ phase1: StepState; phase2: StepState }>(() => {
  switch (status.value) {
    case 'pending':
    case 'uploaded':
      return { phase1: 'pending', phase2: 'pending' }
    case 'processing_phase1':
    case 'processing':
      return { phase1: 'active', phase2: 'pending' }
    case 'phase1_complete':
      return { phase1: 'done', phase2: 'pending' }
    case 'processing_phase2':
      return { phase1: 'done', phase2: 'active' }
    case 'completed':
      return { phase1: 'done', phase2: 'done' }
    case 'failed_phase1':
    case 'failed':
      return { phase1: 'failed', phase2: 'pending' }
    case 'failed_phase2':
      return { phase1: 'done', phase2: 'failed' }
  }
})

const phase1State = computed(() => phaseStates.value.phase1)
const phase2State = computed(() => phaseStates.value.phase2)

const rallyCount = computed(() => rallyClips.value?.length ?? 0)

// Show rally count chip only once Phase 1 is done AND we have a count.
const showRallyCount = computed(
  () => phase1State.value === 'done' && rallyCount.value > 0,
)

const connectorActive = computed(() => phase1State.value === 'done')
</script>

<template>
  <div class="phase-header" :data-active="activePhase">
    <div class="step" :data-state="phase1State">
      <div class="marker" aria-hidden="true">
        <span v-if="phase1State === 'done'" class="check">✓</span>
        <span v-else-if="phase1State === 'failed'" class="cross">✕</span>
        <span v-else-if="phase1State === 'active'" class="pulse" />
        <span v-else class="hollow" />
      </div>
      <div class="label">
        <span class="title">Rally Detection</span>
        <span v-if="showRallyCount" class="meta">· {{ rallyCount }} {{ rallyCount === 1 ? 'rally' : 'rallies' }}</span>
      </div>
    </div>

    <div class="connector" :class="{ active: connectorActive }" aria-hidden="true" />

    <div class="step" :data-state="phase2State">
      <div class="marker" aria-hidden="true">
        <span v-if="phase2State === 'done'" class="check">✓</span>
        <span v-else-if="phase2State === 'failed'" class="cross">✕</span>
        <span v-else-if="phase2State === 'active'" class="pulse" />
        <span v-else class="hollow" />
      </div>
      <div class="label">
        <span class="title">Full Analytics</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.phase-header {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 600px;
  margin: 0 auto 16px;
  padding: 12px 16px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
}

.step {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.marker {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--color-bg-tertiary);
  font-size: 0.75rem;
  font-weight: 600;
  line-height: 1;
}

.step[data-state='done'] .marker {
  color: var(--color-accent);
}

.step[data-state='failed'] .marker {
  color: var(--color-error);
}

.hollow {
  width: 8px;
  height: 8px;
  border: 1.5px solid var(--color-text-tertiary);
  border-radius: 50%;
}

.pulse {
  width: 8px;
  height: 8px;
  background: var(--color-accent);
  border-radius: 50%;
  animation: phase-pulse 1.5s ease-in-out infinite;
}

@keyframes phase-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.3); opacity: 0.6; }
}

.label {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-size: 0.875rem;
}

.title {
  color: var(--color-text-secondary);
  font-weight: 500;
}

.step[data-state='active'] .title,
.step[data-state='done'] .title {
  color: var(--color-text-heading);
}

.step[data-state='failed'] .title {
  color: var(--color-error);
}

.step[data-state='pending'] .title {
  color: var(--color-text-tertiary);
}

.meta {
  color: var(--color-text-tertiary);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}

.connector {
  flex: 1;
  height: 2px;
  min-width: 24px;
  background: var(--color-border);
  transition: background 0.3s ease;
}

.connector.active {
  background: var(--color-accent);
}

@media (max-width: 480px) {
  .phase-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .connector {
    width: 2px;
    height: 16px;
    min-width: 0;
    margin-left: 9px;
  }
}
</style>
