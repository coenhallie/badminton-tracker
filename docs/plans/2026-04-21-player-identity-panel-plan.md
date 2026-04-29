# Player Identity Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a post-analysis UI panel where the user can see cropped thumbnails of both detected players, swap their labels (`Player 1 ↔ Player 2`), and rename them. The swap is a display-layer transform — stored skeleton data stays authoritative.

**Architecture:** New `playerLabels` sub-object on the Convex `videos` table holds thumbnail storage IDs, a `swapped` flag, and optional names. Modal worker captures two player thumbnails from a representative frame after analysis and writes them. Frontend `usePlayerLabels` composable exposes `displayId(pid)` and `labelFor(pid)` helpers; ~8 display sites swap `P${id + 1}` strings for `labelFor(id)` calls. A new `<PlayerIdentityPanel>` component in `ResultsDashboard` drives toggling + renaming via a new `updatePlayerLabels` mutation.

**Tech Stack:** Vue 3 `<script setup>` + TypeScript, Convex (schema + mutations + storage), Python (Modal worker), `cv2` for frame-decode + crop.

**Design doc:** `docs/plans/2026-04-21-player-identity-panel-design.md`

---

### Verification primitives used throughout

No unit-test framework in this project. Each task uses:
- `npm run type-check` — must pass with zero NEW errors. The 8 pre-existing errors to ignore: `convex/http.ts:1267` (`process`), `convex/videos.ts:461,472` (`process`), `src/App.vue:389,392` (`rally` possibly undefined — line numbers may drift as edits land; same error text), `src/components/VideoUpload.vue:191`, `src/composables/useVideoExport.ts:131,142`.
- `npm run build` — clean build.
- `python3 -c "import ast; ast.parse(open('backend/modal_convex_processor.py').read()); print('ok')"` — backend syntax check.
- Manual browser check — called out per task where relevant.

Sequencing is deliberately incremental: the schema lands first and is optional, then the display sites switch to `labelFor(id)` (a no-op while `swapped=false` and no names are set), then the backend starts emitting thumbnails, and finally the panel UI ships. Each commit leaves the app fully working.

---

### Task 1: Convex schema + mutations + query + deleteVideo cleanup

**Goal:** Add the `playerLabels` sub-object to the schema, the mutations/queries the frontend and worker will call, and extend `deleteVideo` to clean up thumbnail blobs.

**Files:**
- Modify: `convex/schema.ts`
- Modify: `convex/videos.ts`

**Step 1 — Schema field**

In `convex/schema.ts`, inside the `videos` table `defineTable` object, add a new optional field (place it next to the other per-video metadata, e.g. after `manualCourtKeypoints`):

```typescript
playerLabels: v.optional(v.object({
  // Storage IDs of cropped thumbnails written by the Modal worker.
  player_0_thumbnail: v.optional(v.id("_storage")),
  player_1_thumbnail: v.optional(v.id("_storage")),
  // User-controlled display state. Defaults to identity mapping
  // (no swap, no custom names) when absent.
  swapped: v.optional(v.boolean()),
  player_0_name: v.optional(v.string()),
  player_1_name: v.optional(v.string()),
})),
```

**Step 2 — Query: `getPlayerLabels`**

In `convex/videos.ts`, near `getManualCourtKeypoints`, add:

```typescript
export const getPlayerLabels = query({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return null
    return video.playerLabels ?? null
  },
})
```

**Step 3 — Mutation: `updatePlayerLabels` (user toggle + rename)**

Add:

```typescript
export const updatePlayerLabels = mutation({
  args: {
    videoId: v.id("videos"),
    swapped: v.optional(v.boolean()),
    player_0_name: v.optional(v.string()),
    player_1_name: v.optional(v.string()),
  },
  handler: async (ctx, { videoId, swapped, player_0_name, player_1_name }) => {
    const video = await ctx.db.get(videoId)
    if (!video) throw new Error("Video not found")

    const next = { ...(video.playerLabels ?? {}) }
    if (swapped !== undefined) next.swapped = swapped
    if (player_0_name !== undefined) next.player_0_name = player_0_name
    if (player_1_name !== undefined) next.player_1_name = player_1_name

    await ctx.db.patch(videoId, { playerLabels: next })
    return { success: true, playerLabels: next }
  },
})
```

**Step 4 — Mutation: `setPlayerThumbnails` (worker callback)**

Add:

```typescript
export const setPlayerThumbnails = mutation({
  args: {
    videoId: v.id("videos"),
    player_0_thumbnail: v.id("_storage"),
    player_1_thumbnail: v.id("_storage"),
  },
  handler: async (ctx, { videoId, player_0_thumbnail, player_1_thumbnail }) => {
    const video = await ctx.db.get(videoId)
    if (!video) throw new Error("Video not found")

    const next = {
      ...(video.playerLabels ?? {}),
      player_0_thumbnail,
      player_1_thumbnail,
    }
    await ctx.db.patch(videoId, { playerLabels: next })
    return { success: true }
  },
})
```

**Step 5 — Extend `deleteVideo` blob sweep**

Find the existing `deleteVideo` mutation. Its `blobIds` array currently lists:
```typescript
const blobIds = [
  video.storageId,
  video.resultsStorageId,
  video.processedVideoStorageId,
  video.skeletonDataStorageId,
].filter((id): id is Id<"_storage"> => id != null)
```

Extend to include the two new thumbnail blobs:

```typescript
const blobIds = [
  video.storageId,
  video.resultsStorageId,
  video.processedVideoStorageId,
  video.skeletonDataStorageId,
  video.playerLabels?.player_0_thumbnail,
  video.playerLabels?.player_1_thumbnail,
].filter((id): id is Id<"_storage"> => id != null)
```

**Step 6 — Push + verify**

```bash
npx convex dev --once
```
Expected: `Convex functions ready!` with no errors.

```bash
npm run type-check
```
Expected: only the 8 pre-existing errors.

**Step 7 — Commit**

```bash
git add convex/schema.ts convex/videos.ts
git commit -m "feat(schema): add playerLabels with thumbnail IDs + swap + names; extend deleteVideo cleanup"
```

---

### Task 2: Shared `usePlayerLabels` composable

**Goal:** One helper that exposes `displayId`, `canonicalId`, and `labelFor`. Everything downstream reads from this.

**Files:**
- Create: `src/composables/usePlayerLabels.ts`

**Step 1 — Write the composable**

`src/composables/usePlayerLabels.ts`:

```typescript
import { computed, type Ref } from 'vue'
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
 */
export function usePlayerLabels(videoId: Ref<Id<'videos'> | null | undefined>) {
  const { data } = useConvexQuery(
    api.videos.getPlayerLabels,
    // convex-vue accepts a reactive arg; null videoId means "don't fetch"
    computed(() => (videoId.value ? { videoId: videoId.value } : 'skip')),
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
```

IMPORTANT — verify the `'skip'` sentinel is what `convex-vue`'s `useConvexQuery` uses. Check the existing usage pattern in `src/components/AnalysisProgress.vue:26-32` first. If this project's convex-vue version uses a different "don't-fetch" value (e.g. passing `undefined` or a separate `enabled` flag), adjust accordingly. The key requirement: the composable must not crash when `videoId.value` is null.

**Step 2 — Verify**

```bash
npm run type-check
```
Expected: no new errors. File is isolated; nothing imports it yet.

**Step 3 — Commit**

```bash
git add src/composables/usePlayerLabels.ts
git commit -m "feat: add usePlayerLabels composable for display-layer player label mapping"
```

---

### Task 3: Provide labels helper from App.vue + adopt in VideoPlayer skeleton labels

**Goal:** Wire the composable in one place (`App.vue`), provide it to descendants via `provide`/`inject`, and flip the first display site — VideoPlayer's skeleton label + color.

**Files:**
- Modify: `src/App.vue`
- Modify: `src/components/VideoPlayer.vue`

**Step 1 — Create the provider in App.vue**

In `src/App.vue`, after the `analysisResult` ref and `useAdvancedAnalytics` call (near where `manualCourtKeypoints` etc are declared), add:

```typescript
import { provide } from 'vue'
import { usePlayerLabels } from '@/composables/usePlayerLabels'
import type { Id } from '../convex/_generated/dataModel'

const videoIdRef = computed(() =>
  (analysisResult.value?.video_id ?? null) as Id<'videos'> | null
)
const playerLabelsHelper = usePlayerLabels(videoIdRef)
provide('playerLabels', playerLabelsHelper)
```

`provide` should be called at the top level of the `<script setup>`, NOT inside a lifecycle hook or conditional. `vue` may already be imported — extend the existing import rather than duplicating.

Define and export the inject key for type-safe access. Add to `src/composables/usePlayerLabels.ts`:

```typescript
import type { InjectionKey } from 'vue'

export type PlayerLabelsHelper = ReturnType<typeof usePlayerLabels>
export const PLAYER_LABELS_KEY: InjectionKey<PlayerLabelsHelper> = Symbol('playerLabels')
```

And use it in `App.vue`:
```typescript
import { usePlayerLabels, PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'
// ...
provide(PLAYER_LABELS_KEY, playerLabelsHelper)
```

**Step 2 — Inject + use in VideoPlayer skeleton label**

In `src/components/VideoPlayer.vue`, add near the existing imports:

```typescript
import { inject } from 'vue'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'

const playerLabels = inject(PLAYER_LABELS_KEY)
```

`inject` may return `undefined` if no provider is present (e.g. in tests or if rendered outside App). Handle gracefully with a fallback: `playerLabels?.labelFor(id) ?? `Player ${id + 1}``.

Find the skeleton label construction at **`src/components/VideoPlayer.vue:1460`**:

```typescript
const label = `P${player.player_id + 1}: ${player.current_speed?.toFixed(1) ?? 0} km/h`
```

Change to:
```typescript
const labelName = playerLabels?.labelFor(player.player_id) ?? `Player ${player.player_id + 1}`
// Shorten "Player 1" → "P1" for the compact on-skeleton label; keep custom
// names as-is if the user set one (they might prefer "John: 9.2 km/h").
const labelPrefix = labelName.startsWith('Player ') ? `P${labelName.slice(7)}` : labelName
const label = `${labelPrefix}: ${player.current_speed?.toFixed(1) ?? 0} km/h`
```

Find the color pick at **`src/components/VideoPlayer.vue:1398`**:

```typescript
const color = PLAYER_COLORS[player.player_id % PLAYER_COLORS.length] ?? '#FF6B6B'
```

Change to:
```typescript
const displayPid = playerLabels?.displayId(player.player_id) ?? player.player_id
const color = PLAYER_COLORS[displayPid % PLAYER_COLORS.length] ?? '#FF6B6B'
```

**Step 3 — Verify**

```bash
npm run type-check
```
Expected: no new errors. With `swapped=false` everywhere and no custom names, behaviour is bit-for-bit unchanged (labelFor returns `Player ${displayId + 1}` = `Player ${canonical + 1}` when no swap; labelPrefix becomes `P${canonical + 1}`).

**Step 4 — Commit**

```bash
git add src/App.vue src/components/VideoPlayer.vue src/composables/usePlayerLabels.ts
git commit -m "feat: provide playerLabels helper from App.vue; VideoPlayer skeleton labels use labelFor"
```

---

### Task 4: ResultsDashboard player cards

**Goal:** Flip the per-player summary cards (top of the results area) to `labelFor`.

**Files:**
- Modify: `src/components/ResultsDashboard.vue`

**Step 1 — Inject + use**

Add at the top of the `<script setup>` if not already present:
```typescript
import { inject } from 'vue'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'
const playerLabels = inject(PLAYER_LABELS_KEY)
```

Find lines **`src/components/ResultsDashboard.vue:375`** and **`:377`**:

```vue
<!-- line 375 -->
P{{ player.player_id + 1 }}
...
<!-- line 377 -->
<h4>Player {{ player.player_id + 1 }}</h4>
```

Change to:
```vue
<!-- line 375 — the badge/pill -->
{{ playerLabels ? `P${playerLabels.displayId(player.player_id) + 1}` : `P${player.player_id + 1}` }}
...
<!-- line 377 — the heading -->
<h4>{{ playerLabels?.labelFor(player.player_id) ?? `Player ${player.player_id + 1}` }}</h4>
```

If the iteration order of the cards should also flip on swap, sort the `players` list by `displayId` where it's constructed. Search the file for the `v-for` that produces these cards (around the same area) and add a sort. Concrete pattern:

```vue
<div v-for="player in sortedPlayers" :key="player.player_id">
```

with:
```typescript
const sortedPlayers = computed(() => {
  const arr = [...(/* existing source */)]
  return arr.sort((a, b) => {
    const da = playerLabels?.displayId(a.player_id) ?? a.player_id
    const db = playerLabels?.displayId(b.player_id) ?? b.player_id
    return da - db
  })
})
```

If the cards are rendered via a different data structure, adapt to sort by canonical player_id mapped through displayId. Keep the original if making this a separate task is cleaner — but label-only is the minimum.

**Step 2 — Verify**

```bash
npm run type-check
```

**Step 3 — Commit**

```bash
git add src/components/ResultsDashboard.vue
git commit -m "feat: ResultsDashboard player cards use labelFor (respects swap + custom names)"
```

---

### Task 5: ShotSummaryOverlay + MiniCourt dot labels

**Goal:** Two small edits in two files.

**Files:**
- Modify: `src/components/ShotSummaryOverlay.vue`
- Modify: `src/components/MiniCourt.vue`

**Step 1 — ShotSummaryOverlay.vue line 18**

Add the inject at the top of `<script setup>`:
```typescript
import { inject } from 'vue'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'
const playerLabels = inject(PLAYER_LABELS_KEY)
```

Change the template:
```vue
<!-- was -->
<span class="shot-summary-player">Player {{ segment.movingPlayerId + 1 }} responded</span>
<!-- becomes -->
<span class="shot-summary-player">
  {{ playerLabels?.labelFor(segment.movingPlayerId) ?? `Player ${segment.movingPlayerId + 1}` }} responded
</span>
```

**Step 2 — MiniCourt.vue line 1058**

Add the inject similarly. Change:
```typescript
ctx.fillText(`P${playerId + 1}`, canvasPos.x, canvasPos.y)
```
to:
```typescript
const labelName = playerLabels?.labelFor(playerId) ?? `Player ${playerId + 1}`
const labelText = labelName.startsWith('Player ') ? `P${labelName.slice(7)}` : labelName
ctx.fillText(labelText, canvasPos.x, canvasPos.y)
```

Find the color lookups at **`:807`** and **`:1020`**:
```typescript
const color = PLAYER_COLORS[playerId % PLAYER_COLORS.length] ?? '#FF6B6B'
```
Change to:
```typescript
const displayPid = playerLabels?.displayId(playerId) ?? playerId
const color = PLAYER_COLORS[displayPid % PLAYER_COLORS.length] ?? '#FF6B6B'
```
(apply the same two-line transformation to both sites).

**Step 3 — Verify + commit**

```bash
npm run type-check
git add src/components/ShotSummaryOverlay.vue src/components/MiniCourt.vue
git commit -m "feat: ShotSummaryOverlay + MiniCourt dot labels respect player swap/names"
```

---

### Task 6: RallyTimeline + AdvancedAnalytics

**Goal:** 5 `P{{ ... + 1 }}` sites across these two files. All identical pattern.

**Files:**
- Modify: `src/components/RallyTimeline.vue`
- Modify: `src/components/AdvancedAnalytics.vue`

**Step 1 — Add inject to both files**

Same pattern:
```typescript
import { inject } from 'vue'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'
const playerLabels = inject(PLAYER_LABELS_KEY)
```

**Step 2 — Replace label sites**

| File:line | From | To |
|---|---|---|
| `RallyTimeline.vue:300` | `` `P{{ p.playerId + 1 }}` `` | `` `{{ playerLabels ? \`P${playerLabels.displayId(p.playerId) + 1}\` : \`P${p.playerId + 1}\` }}` `` |
| `AdvancedAnalytics.vue:316` | `` `:title="`P${shot.playerId + 1}: ${shot.shotType}`"` `` | `` `:title="\`${playerLabels?.labelFor(shot.playerId) ?? 'Player ' + (shot.playerId + 1)}: ${shot.shotType}\`"` `` |
| `AdvancedAnalytics.vue:395` | `` `P{{ stat.playerId + 1 }}` `` | Same pattern as RallyTimeline |
| `AdvancedAnalytics.vue:423` | `` `P{{ stat.playerId + 1 }}` `` | Same |
| `AdvancedAnalytics.vue:462` | `` `P{{ eff.playerId + 1 }}` `` | Same |

Keep the expressions simple. If the template interpolation gets ugly, extract a helper method on the script side:
```typescript
function pidBadge(canonical: number): string {
  const d = playerLabels?.displayId(canonical) ?? canonical
  return `P${d + 1}`
}
```
Use `{{ pidBadge(p.playerId) }}` in the template.

**Step 3 — Verify + commit**

```bash
npm run type-check
git add src/components/RallyTimeline.vue src/components/AdvancedAnalytics.vue
git commit -m "feat: RallyTimeline + AdvancedAnalytics player badges respect swap"
```

---

### Task 7: ShotSpeedList + SpeedGraph

**Goal:** Last two display sites. Note: these files have slightly different conventions — they currently use `playerId` that may be 1-indexed or 0-indexed; verify per file before editing.

**Files:**
- Modify: `src/components/ShotSpeedList.vue`
- Modify: `src/components/SpeedGraph.vue`

**Step 1 — ShotSpeedList.vue**

Add the inject. Update line **`:325`**: the badge currently shows `P{{ segment.movingPlayerId }}` (no `+1` — check whether this is intentional or a bug). If `movingPlayerId` is 0-indexed as it is everywhere else, this was a pre-existing display bug — show it as 1-indexed via `labelFor`. Change to:
```vue
{{ playerLabels?.labelFor(segment.movingPlayerId) ?? `Player ${segment.movingPlayerId + 1}` }}
```

Update line **`:155`** `getPlayerColor(playerId)`: the function does `playerId - 1`, implying its input is 1-indexed. Since we're reading canonical 0-indexed `movingPlayerId`, change the callsites to pass `displayId(canonical) + 1` instead:

```typescript
function getPlayerColor(displayIdx1: number): string {
  // displayIdx1 is expected to be 1-indexed as before
  const idx = Math.max(0, displayIdx1 - 1) % PLAYER_SPEED_COLORS.length
  return PLAYER_SPEED_COLORS[idx] ?? '#FF6B6B'
}
```

Callsites (e.g. line 323): pass `(playerLabels?.displayId(segment.movingPlayerId) ?? segment.movingPlayerId) + 1`.

If this is too fiddly, extract a tiny helper `colorForCanonical(canonical: number)` that does the conversion.

**Step 2 — SpeedGraph.vue**

Line **`:293`** currently: `` `label: `Player ${playerId}`` ``. `playerId` here — check whether it's 0-indexed or 1-indexed by reading 20 lines above the match. If 0-indexed, the current label shows "Player 0" / "Player 1" (possible bug). If 1-indexed, "Player 1" / "Player 2".

Apply:
```typescript
// If playerId is 0-indexed at this site:
label: playerLabels?.labelFor(playerId) ?? `Player ${playerId + 1}`,
// If playerId is 1-indexed at this site:
label: playerLabels?.labelFor(playerId - 1) ?? `Player ${playerId}`,
```

Pick the correct form based on the grep context.

**Step 3 — Verify + commit**

```bash
npm run type-check && npm run build
git add src/components/ShotSpeedList.vue src/components/SpeedGraph.vue
git commit -m "feat: ShotSpeedList + SpeedGraph labels + colors respect player swap"
```

After this task, every display site uses `labelFor` / `displayId`. With `swapped=false` and no names set, UI is byte-for-byte unchanged from before the feature. Safe to ship the next chunk independently.

---

### Task 8: Backend — capture two player thumbnails during analysis

**Goal:** At the end of `_process_video_worker` in `backend/modal_convex_processor.py`, pick a good 2-player frame, crop both bounding boxes, upload the crops to Convex storage, and call `setPlayerThumbnails`.

**Files:**
- Modify: `backend/modal_convex_processor.py`

**Step 1 — Add the selection + crop + upload helper**

Near the other module-level helper functions in `modal_convex_processor.py` (e.g. near `skeleton_center_from_keypoints`), add:

```python
def _pick_best_two_player_frame(
    skeleton_frames: List[Dict],
    video_height: int,
    max_search_seconds: float,
    fps: float,
) -> Optional[Dict]:
    """
    Pick the frame best suited for player-identity thumbnails.

    Criteria (in priority order):
      1. Exactly 2 players detected in the frame.
      2. Both have valid bounding boxes.
      3. Their ankle-midpoint Y-coordinates are separated by ≥25% of
         video_height (so the bounding boxes are not overlapping).
      4. Sum of keypoint confidences across both players is maximal.

    Prefer frames within the first max_search_seconds; fall back to
    "best anywhere" if no frame in the early window qualifies.

    Returns the chosen frame dict (the in-memory skeleton frame with
    its "players" list) or None if no qualifying frame exists.
    """
    min_sep_px = video_height * 0.25
    early_cutoff_frame = int(max_search_seconds * fps)

    def score(frame: Dict) -> float:
        players = frame.get("players", [])
        if len(players) != 2:
            return -1.0
        c0 = players[0].get("center")
        c1 = players[1].get("center")
        if not c0 or not c1:
            return -1.0
        # Access center as dict {"x":..,"y":..} or list [x,y] depending on serialization;
        # the in-memory shape at this point uses dicts.
        y0 = c0.get("y") if isinstance(c0, dict) else c0[1]
        y1 = c1.get("y") if isinstance(c1, dict) else c1[1]
        if abs(y0 - y1) < min_sep_px:
            return -1.0
        conf_sum = 0.0
        for p in players:
            for kp in p.get("keypoints", []):
                # kpts in memory are [x, y] — if confidence is available in a
                # sibling array (conf), use that. Otherwise approximate by
                # counting non-zero keypoints.
                if len(kp) >= 3:
                    conf_sum += float(kp[2])
                elif kp[0] > 0 and kp[1] > 0:
                    conf_sum += 1.0
        return conf_sum

    best_early: Optional[Dict] = None
    best_early_score = -1.0
    best_any: Optional[Dict] = None
    best_any_score = -1.0
    for frame in skeleton_frames:
        s = score(frame)
        if s <= 0:
            continue
        frame_num = frame.get("frame", 0)
        if frame_num <= early_cutoff_frame and s > best_early_score:
            best_early = frame
            best_early_score = s
        if s > best_any_score:
            best_any = frame
            best_any_score = s

    return best_early or best_any
```

Read the actual shape of `frame["players"][i]` written elsewhere in the worker (around line 1915 where `"center": center` is set) to confirm whether `center` is a `{"x": , "y": }` dict, a `(x, y)` tuple, or a `[x, y]` list at this point. Adjust the score function accordingly.

**Step 2 — Add the capture routine**

Add another helper that, given a selected frame + video path, decodes that frame and returns two JPEG byte strings:

```python
def _capture_player_thumbnails(
    video_path: Path,
    frame_number: int,
    players: List[Dict],
    padding_ratio: float = 0.15,
) -> Optional[Tuple[bytes, bytes]]:
    """
    Decode the given frame from disk, crop each player's bounding box
    with padding, JPEG-encode. Returns (player_0_jpeg, player_1_jpeg)
    ordered by the incoming players list index — caller is responsible
    for the ordering matching skeleton_data[].players[].player_id.

    Returns None if anything fails (disk read, empty bbox, encode error).
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, img = cap.read()
        if not ok or img is None:
            return None
    finally:
        cap.release()

    h, w = img.shape[:2]
    out: List[bytes] = []
    for p in players:
        bbox = p.get("bbox")
        if not bbox:
            return None
        # bbox shape used by this worker at save time is
        # {"x1":, "y1":, "x2":, "y2":} — confirm at the callsite.
        x1 = bbox["x1"] if isinstance(bbox, dict) else bbox[0]
        y1 = bbox["y1"] if isinstance(bbox, dict) else bbox[1]
        x2 = bbox["x2"] if isinstance(bbox, dict) else bbox[2]
        y2 = bbox["y2"] if isinstance(bbox, dict) else bbox[3]
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        ix1 = max(0, int(x1 - pad_x))
        iy1 = max(0, int(y1 - pad_y))
        ix2 = min(w, int(x2 + pad_x))
        iy2 = min(h, int(y2 + pad_y))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = img[iy1:iy2, ix1:ix2]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        out.append(buf.tobytes())
    if len(out) != 2:
        return None
    return out[0], out[1]
```

Confirm the exact `bbox` shape stored on the in-memory skeleton frame by reading the existing code (around the `"bbox": bbox,` assignment in the main frame loop — grep for `"bbox":` in the worker).

**Step 3 — Wire the callback into `_process_video_worker`**

At the end of `_process_video_worker`, after the results JSON has been uploaded but before the final return, add:

```python
# Player-identity thumbnails — best-effort; never fail analysis on this.
try:
    chosen = _pick_best_two_player_frame(
        skeleton_frames,  # the in-memory list built during the loop
        video_height=int(height),
        max_search_seconds=10.0,
        fps=fps,
    )
    if chosen is None:
        await send_log(
            "No qualifying 2-player frame for player thumbnails; skipping",
            "info", "processing",
        )
    else:
        thumbs = _capture_player_thumbnails(
            video_path,
            frame_number=int(chosen["frame"]),
            players=chosen["players"][:2],
        )
        if thumbs is None:
            await send_log(
                "Player thumbnail capture failed; skipping",
                "warning", "processing",
            )
        else:
            thumb0_id = await _upload_blob_to_convex(
                http_client, callback_url, thumbs[0], "image/jpeg",
            )
            thumb1_id = await _upload_blob_to_convex(
                http_client, callback_url, thumbs[1], "image/jpeg",
            )
            await http_client.post(
                f"{callback_url}/setPlayerThumbnails",
                json={
                    "videoId": video_id,
                    "player_0_thumbnail": thumb0_id,
                    "player_1_thumbnail": thumb1_id,
                },
            )
            await send_log(
                "Player thumbnails uploaded", "success", "processing",
            )
except Exception as thumb_err:
    # Thumbnails are cosmetic — swallow any error so analysis still returns OK.
    print(f"[MODAL] Player thumbnail capture error: {thumb_err}")
    await send_log(
        f"Player thumbnail capture error (non-fatal): {thumb_err}",
        "warning", "processing",
    )
```

`_upload_blob_to_convex` may already exist from the results JSON upload path — search for the existing code that POSTs the results JSON to a Convex storage endpoint and refactor into a small helper if one isn't already broken out. If the codebase already has an equivalent helper used for results JSON, use it directly.

If `_upload_blob_to_convex` doesn't exist as a standalone helper, extract it during this task: find the existing upload code near the results-JSON upload and make it a function that takes `(http_client, callback_url, blob_bytes, content_type) -> storage_id`.

Add the corresponding HTTP route on the Convex side:

```typescript
// convex/http.ts — add near other POST routes
http.route({
  path: "/setPlayerThumbnails",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      const { videoId, player_0_thumbnail, player_1_thumbnail } = body
      if (!videoId || !player_0_thumbnail || !player_1_thumbnail) {
        return new Response(
          JSON.stringify({ error: "Missing required fields" }),
          { status: 400, headers: corsHeaders("application/json") },
        )
      }
      await ctx.runMutation(api.videos.setPlayerThumbnails, {
        videoId: videoId as Id<"videos">,
        player_0_thumbnail: player_0_thumbnail as Id<"_storage">,
        player_1_thumbnail: player_1_thumbnail as Id<"_storage">,
      })
      return new Response(JSON.stringify({ status: "success" }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (err) {
      console.error("setPlayerThumbnails error:", err)
      return new Response(
        JSON.stringify({ error: err instanceof Error ? err.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") },
      )
    }
  }),
})
```

Reuse the existing `corsHeaders` helper in `convex/http.ts`.

**Step 4 — Verify**

```bash
python3 -c "import ast; ast.parse(open('backend/modal_convex_processor.py').read()); print('ok')"
npx convex dev --once
```
Both should succeed.

**Step 5 — Commit (no deploy yet — the UI can't use the thumbnails until Task 9)**

```bash
git add backend/modal_convex_processor.py convex/http.ts
git commit -m "feat(backend): capture player thumbnails + setPlayerThumbnails HTTP route"
```

**Step 6 — Deploy Modal**

```bash
python3 -m modal deploy backend/modal_convex_processor.py
```

Expected: deploys successfully. From this point forward, new video analyses produce two player thumbnail blobs per video.

---

### Task 9: `<PlayerIdentityPanel>` component + integration

**Goal:** The actual UI the user sees — two thumbnails, a swap button, editable name inputs.

**Files:**
- Create: `src/components/PlayerIdentityPanel.vue`
- Modify: `src/components/ResultsDashboard.vue`

**Step 1 — Component**

`src/components/PlayerIdentityPanel.vue`:

```vue
<script setup lang="ts">
import { ref, computed, inject, watch } from 'vue'
import { useConvexClient } from 'convex-vue'
import { api } from '../../convex/_generated/api'
import type { Id } from '../../convex/_generated/dataModel'
import { PLAYER_LABELS_KEY } from '@/composables/usePlayerLabels'

const props = defineProps<{
  videoId: string
}>()

const playerLabels = inject(PLAYER_LABELS_KEY)
const convex = useConvexClient()

const labels = computed(() => playerLabels?.labels.value ?? null)
const swapped = computed(() => labels.value?.swapped ?? false)

// Reactive thumbnail URLs via Convex storage. Fetch once on id change.
const thumb0Url = ref<string | null>(null)
const thumb1Url = ref<string | null>(null)

async function resolveThumbnail(storageId: Id<'_storage'> | undefined): Promise<string | null> {
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
watch(labels, (l) => {
  name0.value = l?.player_0_name ?? ''
  name1.value = l?.player_1_name ?? ''
}, { immediate: true })

// Display order: if swapped, show canonical 1 first then canonical 0.
// Each slot renders as the DISPLAY label, so when the user clicks swap
// the left card becomes "Player 2" and the right becomes "Player 1".
const slots = computed(() => {
  const mk = (canonical: 0 | 1) => ({
    canonical,
    thumb: canonical === 0 ? thumb0Url.value : thumb1Url.value,
    nameModel: canonical === 0 ? name0 : name1,
    displayIndex: playerLabels?.displayId(canonical) ?? canonical,
    displayLabel: playerLabels?.labelFor(canonical) ?? `Player ${canonical + 1}`,
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
</script>

<template>
  <div class="player-identity-panel">
    <div class="pip-header">
      <span class="pip-title">Players</span>
      <button type="button" class="pip-swap-btn" @click="toggleSwap">
        ⇄ Swap
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
            @input="slot.nameModel.value = ($event.target as HTMLInputElement).value"
            @change="scheduleNameSave(slot.canonical, slot.nameModel.value)"
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
```

**Step 2 — `getStorageUrl` query (if not already present)**

Check `convex/videos.ts` / other Convex files for a query that returns a signed URL for a storage blob. If one doesn't exist, add:

```typescript
// convex/videos.ts
export const getStorageUrl = query({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, { storageId }) => {
    return await ctx.storage.getUrl(storageId)
  },
})
```

**Step 3 — Mount the panel in `ResultsDashboard.vue`**

Near the top of the results area (where per-player summary cards currently render), add:

```vue
<script setup lang="ts">
import PlayerIdentityPanel from './PlayerIdentityPanel.vue'
// ...existing imports and defineProps
</script>

<template>
  <PlayerIdentityPanel
    v-if="videoId"
    :video-id="videoId"
  />
  <!-- existing player cards below -->
</template>
```

`videoId` may already be available as a prop or from `analysisResult` in this component; use whatever is already there.

**Step 4 — Verify**

```bash
npm run type-check && npm run build
```

**Step 5 — Manual browser check**

1. Upload a fresh video → run analysis (Modal now captures thumbnails).
2. Open the results dashboard — the Player Identity panel appears at the top with two thumbnails.
3. Click Swap — every labelled site in the UI flips: skeleton overlay on the video, per-player cards, shot summary overlay, mini court, rally timeline, etc. Click Swap again — returns to original.
4. Type a name in one of the inputs — 500ms after you stop typing, it persists; reload the page to confirm it survives.
5. Open a video analyzed BEFORE Task 8 was deployed — panel shows placeholder thumbnails but Swap + names still work.

**Step 6 — Commit**

```bash
git add src/components/PlayerIdentityPanel.vue src/components/ResultsDashboard.vue convex/videos.ts
git commit -m "feat: PlayerIdentityPanel with thumbnails, swap, and rename"
```

---

### Task 10: Final verification + scope audit

**Step 1 — Type-check + build**

```bash
npm run type-check && npm run build
```

Expected: only the 8 pre-existing errors; clean build.

**Step 2 — Scope audit**

```bash
git log --oneline <plan-commit-sha>..HEAD
git diff --name-only <plan-commit-sha>..HEAD
```

Files expected to appear:
- `convex/schema.ts`, `convex/videos.ts`, `convex/http.ts`
- `backend/modal_convex_processor.py`
- `src/App.vue`
- `src/composables/usePlayerLabels.ts`
- `src/components/PlayerIdentityPanel.vue` (new)
- `src/components/VideoPlayer.vue`
- `src/components/ResultsDashboard.vue`
- `src/components/ShotSummaryOverlay.vue`
- `src/components/MiniCourt.vue`
- `src/components/RallyTimeline.vue`
- `src/components/AdvancedAnalytics.vue`
- `src/components/ShotSpeedList.vue`
- `src/components/SpeedGraph.vue`
- `docs/plans/2026-04-21-player-identity-panel-design.md`
- `docs/plans/2026-04-21-player-identity-panel-plan.md`

Nothing else should appear.

**Step 3 — Full manual pass**

- Upload → analyze → thumbnails appear → swap → relabel → reload → state persists.
- Delete the video → thumbnails gone from Convex storage (verify via dashboard).
- Old video (no thumbnails) → panel still works with placeholders.

---

## Out of scope (do NOT attempt)

- Doubles / >2 player support.
- Persistent known-players library across videos.
- Appearance-based auto-identification or re-ID models.
- Auto-backfilling thumbnails on pre-feature videos.
- Mid-video per-rally identity correction (tracker's problem, already addressed separately).
- Custom avatar upload (thumbnails are auto-selected only).
