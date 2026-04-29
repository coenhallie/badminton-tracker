# Post-Analysis Player Identity Correction Design

**Date:** 2026-04-21
**Status:** Approved
**Goal:** After analysis completes, the dashboard shows a panel with cropped thumbnails of both detected players. User can swap them (`Player 1 ↔ Player 2`) with one click, and optionally rename them. Stored analysis data stays authoritative — swap is a display-layer transform.

## Context

The current `PlayerIdentityTracker` assigns `player_id` 0 or 1 based on ankle-midpoint Y vs the net line. When the manual `net_left`/`net_right` keypoints are set (shipped 2026-04-21 morning), the tracker is robust within the pixel-space heuristic. But it still fails when:
- The initial 2-player frame is ambiguous (crossed players at serve setup, spectators near court, etc.) — the Y-sort calibration locks the wrong side to the wrong player.
- Players genuinely swap sides per badminton rules (mid-game) — the tracker refuses to let them cross the net.
- The labels `Player 1` / `Player 2` are arbitrary — coaches want real names.

Plan B is the minimal, low-risk correction layer: let the user see who each detected player is and flip labels or add names. No change to the tracker or stored skeleton data.

## View Geometry / UI Placement

**Panel location:** top of `ResultsDashboard.vue`, above the existing per-player summary cards. Collapsible so it doesn't take permanent vertical space.

**Panel contents:**
- Two rows, one per player.
- Each row: thumbnail (cropped body bbox, ~128×256 JPEG) + editable name input.
- A single `Swap` button flips `Player 1 ↔ Player 2` assignments globally.
- If a video lacks thumbnails (pre-feature analysis), show placeholder avatars — swap + rename still work.

**Responsive:** on narrow viewports, thumbnails shrink and names stack vertically.

Rejected alternatives:
- Modal dialog — too heavy-weight for an edit that a user will rarely make.
- Dropdown per stats card — repetitive, inconsistent across views.

## Data Model

### `convex/schema.ts` — `videos` table
Add `playerLabels` (optional, nested) with:
- `player_0_thumbnail?: Id<"_storage">` — storage blob written by the Modal worker.
- `player_1_thumbnail?: Id<"_storage">` — same.
- `swapped?: boolean` — user flip. Defaults to `false` (identity mapping).
- `player_0_name?: string` — user-entered display name.
- `player_1_name?: string` — same.

### Mutations / queries
New in `convex/videos.ts`:
- `updatePlayerLabels(videoId, swapped?, player_0_name?, player_1_name?)` — partial update, writes only the provided fields.
- `getPlayerLabels(videoId)` — reads the sub-object. Returns `null` when absent (pre-feature videos).
- `setPlayerThumbnails(videoId, player_0_thumbnail, player_1_thumbnail)` — called by the Modal worker at the end of analysis.

`deleteVideo` already iterates all optional storageIds for cleanup, but needs extending to cover the two thumbnail blobs so deletion doesn't orphan them.

## Pipeline — Capture Thumbnails in Modal Worker

In `backend/modal_convex_processor.py`, after the main frame loop and before writing `updateResults`:

1. **Select a representative frame** — scan the already-collected per-frame skeletons for the frame with:
   - Exactly 2 players detected.
   - Both players' keypoint confidence sum maximal.
   - Both players' ankle-midpoint Y-coordinates are well-separated (≥25% of video height) so their bounding boxes don't overlap.
   - Prefer a frame in the first 10s of the video; fall back to "best anywhere" if the early window has no qualifying frame.

2. **Re-decode that single frame** — `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, n)` + one `read()`. Sub-100ms.

3. **Crop each player's bounding box** — bbox from that frame's skeleton data, pad by 15%, clamp to image bounds.

4. **JPEG-encode** each crop at ~85% quality (~20 KB each).

5. **Upload to Convex storage** — the worker already does this for results JSON; reuse the same HTTP callback pattern (POST to `/storage/upload` → storageId).

6. **Patch the video record** — new callback `/setPlayerThumbnails` that writes the two storage IDs.

Cost: one `cv2.seek` + two `cv2.imencode`s per video. Negligible vs. the minute-scale analysis cost.

## Swap Logic — Display-Layer Mapping

Stored `skeleton_data[].players[].player_id` stays authoritative. The swap is a reversible bijection applied **only when labels or grouped display order are rendered**.

### Shared composable `src/composables/usePlayerLabels.ts` (new)

```typescript
export function usePlayerLabels(videoId: Ref<Id<"videos"> | null>) {
  const labelsQuery = useQuery(api.videos.getPlayerLabels, { videoId })
  const labels = computed(() => labelsQuery.data)

  /** Canonical id (0/1) → display id (0/1). Identity unless swapped. */
  function displayId(canonical: number): number {
    if (!labels.value?.swapped) return canonical
    return canonical === 0 ? 1 : canonical === 1 ? 0 : canonical
  }
  /** Display id → canonical id. Same as displayId (involution). */
  const canonicalId = displayId

  function labelFor(canonical: number): string {
    const d = displayId(canonical)
    const nameField = d === 0 ? 'player_0_name' : 'player_1_name'
    return labels.value?.[nameField] ?? `Player ${d + 1}`
  }

  return { labels, displayId, canonicalId, labelFor }
}
```

### Display sites — minimal, targeted edits

| File | Change |
|---|---|
| `src/components/VideoPlayer.vue:1827` | `P${player_id + 1}:` → `labelFor(player_id) + ':'` |
| `VideoPlayer.vue` color lookup | `PLAYER_COLORS[player_id]` → `PLAYER_COLORS[displayId(player_id)]` |
| `src/components/ShotSummaryOverlay.vue` | `Player ${movingPlayerId + 1} responded` → `labelFor(movingPlayerId) + ' responded'` |
| `src/components/ShotSpeedList.vue` | player filter & row labels use `labelFor` + sort by `displayId` |
| `src/components/SpeedGraph.vue` | legend + series colors via `labelFor` + `displayId` |
| `src/components/MiniCourt.vue` | dot label + color via `labelFor` + `displayId` |
| `src/components/AdvancedAnalytics.vue` | per-player card headings via `labelFor` |
| `src/components/ResultsDashboard.vue` | player summary cards headings via `labelFor` |

**Aggregations stay keyed on canonical `player_id`.** Zone analytics, per-player speed stats, rally speed stats — none of them change. Swap affects the label + the order of the two cards when rendered side by side.

Rejected alternatives:
- Rewrite `skeleton_data` in place when swap is toggled — tens of MB rewrite per toggle, server round-trip, and any cached derived data goes stale.
- Propagate swap as a prop through every component — tedious and easy to miss a site.

## Error Handling / Edge Cases

- **No thumbnails on old videos** — panel renders placeholder silhouettes; swap + rename still work.
- **Worker thumbnail capture fails** — analysis continues; the mutation call is wrapped in try/except in the worker. No thumbnails get saved; panel falls back to placeholders.
- **Only one player detected throughout the video** (rare) — use a synthetic placeholder for the missing side. Thumbnail selection requires 2 players.
- **Swap while mid-rally viewing** — reactive; labels update live on the video overlay and any open panels within the next frame tick.
- **Delete video** — `deleteVideo` must also unlink `player_0_thumbnail` and `player_1_thumbnail` from storage. Covered by extending its `blobIds` array.

## Performance

- Thumbnail capture: one-time cost per analysis, <100ms.
- Thumbnail size: ~20 KB each × 2 per video; negligible Convex storage impact.
- `usePlayerLabels` is one query per `ResultsDashboard` mount; cached by Convex reactivity.
- Swap toggle: flips a single boolean; display sites re-render from the reactive `labels` computed. No data rewriting.

## Testing

No unit framework in the project. Verification:
- `npm run type-check` + `npm run build` pass with zero new errors.
- Manual: upload + analyze a new video → thumbnails appear in the panel → click Swap → labels flip everywhere (skeleton overlay, stats cards, shot overlay, mini court). Click Swap again → labels return to original. Type a name → persists across page reloads.
- Manual: open a video analyzed before the feature shipped → panel shows placeholders, swap still works.
- Manual: delete a video with thumbnails → verify no orphan blobs in Convex storage dashboard.

## Non-Goals (Explicit)

- Doubles (>2 players). Current 2-player assumption is preserved.
- Appearance/re-ID models for automatic long-term identity. (Option C from the investigation.)
- Persistent known-players library across videos. Names are per-video.
- Auto-backfill of thumbnails on pre-feature videos.
- Mid-video per-rally identity corrections (if tracker swaps labels mid-match, this feature doesn't address it — that's the tracker's problem, already addressed by the net-line fix).
