<script setup lang="ts">
import { ref, computed, inject, watch } from 'vue'
import { useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { Id } from '../../convex/_generated/dataModel'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'

const props = defineProps<{
  videoId: string
}>()

// PLAYER_LABELS_KEY is typed as InjectionKey<Ref<PlayerLabelsHelper | null>>,
// so `playerLabels` is a Ref. Read the helper via `.value?.` before touching
// `labels`, `displayId`, or `labelFor`. See ResultsDashboard.vue for the same
// pattern.
const playerLabels = inject(PLAYER_LABELS_KEY)
const convex = useConvexClient()

const labels = computed(() => playerLabels?.value?.labels.value ?? null)
const swapped = computed(() => labels.value?.swapped ?? false)

// Reactive thumbnail URLs via Convex storage. Refetch when ids change.
const thumb0Url = ref<string | null>(null)
const thumb1Url = ref<string | null>(null)

async function resolveThumbnail(
  storageId: Id<'_storage'> | undefined,
): Promise<string | null> {
  if (!storageId) return null
  try {
    const url = await convex.query(api.videos.getStorageUrl, { storageId })
    return url ?? null
  } catch {
    return null
  }
}

watch(
  () => [labels.value?.player_0_thumbnail, labels.value?.player_1_thumbnail],
  async ([id0, id1]) => {
    thumb0Url.value = await resolveThumbnail(id0 as Id<'_storage'> | undefined)
    thumb1Url.value = await resolveThumbnail(id1 as Id<'_storage'> | undefined)
  },
  { immediate: true },
)

// Local editable name state, debounced before persisting.
const name0 = ref('')
const name1 = ref('')
watch(
  labels,
  (l) => {
    name0.value = l?.player_0_name ?? ''
    name1.value = l?.player_1_name ?? ''
  },
  { immediate: true },
)

// Display order: if swapped, show canonical 1 first then canonical 0.
// Each slot renders as the DISPLAY label, so when the user clicks swap
// the left card becomes "Player 2" and the right becomes "Player 1".
const slots = computed(() => {
  const mk = (canonical: 0 | 1) => ({
    canonical,
    thumb: canonical === 0 ? thumb0Url.value : thumb1Url.value,
    nameModel: canonical === 0 ? name0 : name1,
    displayIndex: playerLabels?.value?.displayId(canonical) ?? canonical,
    displayLabel:
      playerLabels?.value?.labelFor(canonical) ?? `Player ${canonical + 1}`,
  })
  const a = mk(0)
  const b = mk(1)
  return a.displayIndex < b.displayIndex ? [a, b] : [b, a]
})

async function toggleSwap() {
  await convex.mutation(api.videos.updatePlayerLabels, {
    videoId: props.videoId as Id<'videos'>,
    swapped: !swapped.value,
  })
}

// Single module-level timer so every keystroke resets the same debounce.
let nameDebounce: ReturnType<typeof setTimeout> | null = null
function scheduleNameSave(canonical: 0 | 1, value: string) {
  if (nameDebounce) clearTimeout(nameDebounce)
  nameDebounce = setTimeout(async () => {
    const payload: Record<string, unknown> = {
      videoId: props.videoId as Id<'videos'>,
    }
    if (canonical === 0) payload.player_0_name = value
    else payload.player_1_name = value
    await convex.mutation(api.videos.updatePlayerLabels, payload as never)
  }, 500)
}

function onNameInput(canonical: 0 | 1, event: Event) {
  const value = (event.target as HTMLInputElement).value
  if (canonical === 0) name0.value = value
  else name1.value = value
  scheduleNameSave(canonical, value)
}
</script>

<template>
  <div class="player-identity-panel">
    <div class="pip-header">
      <span class="pip-title">Players</span>
      <button type="button" class="pip-swap-btn" @click="toggleSwap">
        &#8646; Swap
      </button>
    </div>

    <div class="pip-slots">
      <div v-for="slot in slots" :key="slot.canonical" class="pip-slot">
        <div class="pip-thumb">
          <img v-if="slot.thumb" :src="slot.thumb" :alt="slot.displayLabel" />
          <div v-else class="pip-thumb-placeholder">No thumbnail</div>
        </div>
        <div class="pip-slot-body">
          <div class="pip-slot-label">Player {{ slot.displayIndex + 1 }}</div>
          <input
            class="pip-name-input"
            type="text"
            :placeholder="`Player ${slot.displayIndex + 1}`"
            :value="slot.nameModel.value"
            @input="onNameInput(slot.canonical, $event)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.player-identity-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  margin-bottom: 14px;
}
.pip-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.pip-title {
  font-weight: 600;
  font-size: 13px;
  color: rgba(255, 255, 255, 0.9);
}
.pip-swap-btn {
  background: var(--color-accent, #ff9500);
  color: #111;
  border: 0;
  padding: 6px 12px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
}
.pip-slots {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.pip-slot {
  display: flex;
  gap: 10px;
  align-items: center;
}
.pip-thumb {
  width: 80px;
  height: 120px;
  background: rgba(0, 0, 0, 0.4);
  border-radius: 6px;
  overflow: hidden;
  flex-shrink: 0;
}
.pip-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.pip-thumb-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.4);
  text-align: center;
  padding: 4px;
}
.pip-slot-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.pip-slot-label {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.55);
}
.pip-name-input {
  background: rgba(0, 0, 0, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 13px;
  min-width: 0;
  width: 100%;
}
.pip-name-input:focus {
  outline: 1px solid var(--color-accent, #ff9500);
  outline-offset: 0;
}
@media (max-width: 600px) {
  .pip-slots {
    grid-template-columns: 1fr;
  }
}
</style>
