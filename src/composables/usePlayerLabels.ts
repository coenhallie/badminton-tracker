import { computed, type Ref, type InjectionKey } from 'vue'
import { useReactiveRow } from '@/composables/useReactiveRow'

/**
 * Reactive player-label helper. Subscribes to the video row's
 * `player_labels` JSONB column and exposes display-layer accessors.
 *
 * Stored analysis data (skeleton_data[].players[].player_id) is authoritative
 * and never rewritten — these helpers translate canonical 0/1 ids into what
 * the UI should show, applying the user's swap + any custom names.
 *
 * Shape of `player_labels` (set partly by the Modal worker — thumbnail paths —
 * and partly by the user via PlayerIdentityPanel — swapped flag + names):
 *   {
 *     swapped?: boolean,
 *     player_0_name?: string,
 *     player_1_name?: string,
 *     player_0_thumbnail_path?: string,
 *     player_1_thumbnail_path?: string,
 *   }
 *
 * Caller contract: `videoId` must already be resolved to a real video id.
 * Mount the consumer only once you have a videoId.
 */
export interface PlayerLabels {
  swapped?: boolean
  player_0_name?: string
  player_1_name?: string
  player_0_thumbnail_path?: string
  player_1_thumbnail_path?: string
}

interface VideoRowSlim {
  id: string
  player_labels: PlayerLabels | null
}

export function usePlayerLabels(videoId: Ref<string>) {
  const { row } = useReactiveRow<VideoRowSlim>('videos', videoId)

  const labels = computed<PlayerLabels | null>(() => row.value?.player_labels ?? null)

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
export const PLAYER_LABELS_KEY: InjectionKey<Ref<PlayerLabelsHelper | null>> = Symbol('playerLabels')
