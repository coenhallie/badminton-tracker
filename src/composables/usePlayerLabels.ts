import { computed, type Ref, type InjectionKey } from 'vue'
import { useConvexQuery } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { Id } from '../../convex/_generated/dataModel'

/**
 * Reactive player-label helper. Reads the video's `playerLabels` sub-object
 * from Convex and exposes display-layer accessors.
 *
 * Stored analysis data (skeleton_data[].players[].player_id) is authoritative
 * and never rewritten — these helpers translate canonical 0/1 ids into what
 * the UI should show, applying the user's swap + any custom names.
 *
 * Caller contract: `videoId` must already be resolved to a real video id.
 * This composable does not have a skip mode — convex-vue's useConvexQuery
 * subscribes unconditionally. Mount the consumer only once you have a videoId.
 */
export function usePlayerLabels(videoId: Ref<Id<'videos'>>) {
  const { data } = useConvexQuery(
    api.videos.getPlayerLabels,
    computed(() => ({ videoId: videoId.value })),
  )

  const labels = computed(() => data.value)

  /** Canonical id (0/1) → display id (0/1). Identity unless swapped. */
  function displayId(canonical: number): number {
    if (!labels.value?.swapped) return canonical
    if (canonical === 0) return 1
    if (canonical === 1) return 0
    return canonical
  }

  /** Display id → canonical id. Same function — swap is involution. */
  const canonicalId = displayId

  /** Human-readable label for a canonical player id. */
  function labelFor(canonical: number): string {
    const d = displayId(canonical)
    const name = d === 0 ? labels.value?.player_0_name : labels.value?.player_1_name
    return name && name.trim() !== '' ? name : `Player ${d + 1}`
  }

  return { labels, displayId, canonicalId, labelFor }
}

export type PlayerLabelsHelper = ReturnType<typeof usePlayerLabels>
export const PLAYER_LABELS_KEY: InjectionKey<PlayerLabelsHelper> = Symbol('playerLabels')
