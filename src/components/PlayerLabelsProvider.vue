<script setup lang="ts">
import { computed, shallowRef, type Ref } from 'vue'
import { provide } from 'vue'
import {
  usePlayerLabels,
  PLAYER_LABELS_KEY,
  type PlayerLabelsHelper,
} from '@/composables/usePlayerLabels'

// Renderless provider: instantiates `usePlayerLabels` synchronously in its own
// setup. The parent must mount this component via `v-if` only once `videoId`
// is resolved to a real id.
const props = defineProps<{ videoId: string }>()

const videoIdRef = computed(() => props.videoId)
const helper = usePlayerLabels(videoIdRef)

// shallowRef prevents Vue from deep-unwrapping the nested ComputedRef<labels>
// inside the helper, which would otherwise clash with the
// `Ref<PlayerLabelsHelper | null>` inject-key type.
const helperRef: Ref<PlayerLabelsHelper | null> = shallowRef(helper)
provide(PLAYER_LABELS_KEY, helperRef)
</script>

<template>
  <slot />
</template>
