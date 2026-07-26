# Rally detection & clipping accuracy audit — 2026-07-25

Scope: `backend/modal_supabase_processor.py`, `rally_detection.py`,
`rally_detection_shot_gap.py`, `shot_detection.py`, `src/utils/shotDetection.ts`,
`src/composables/useAdvancedAnalytics.ts`, `src/components/RallyReview.vue`.

**[WIP]** = introduced by the uncommitted diff on `feat/supabase-migration`.
**[SHIPPING]** = pre-existing, already affecting produced clips.

Findings §1(cut path), §2, §4, §5 were verified by executing the code offline;
the rest are read from source and marked as such.

> **Status 2026-07-25**: §1 (pre/post roll) and §4 (weld bug, Python + TS) are
> **FIXED** — see the notes in those sections. §2, §3, §5, §6, §7, §8, §9 are
> still open. Read the "Decide this before working §5 and §6" box before
> continuing.

---

## 1. Clips systematically miss the serve and the landing — [SHIPPING] — ✅ FIXED

The shot-gap detector defines a shot as a shuttle **direction reversal**
(`shot_detection.py:210-217`, mirrored in `shotDetection.ts`). A serve is not a
reversal — the shuttle goes from static to moving in one direction, so `dot > 0`
and no shot fires. **The first reversal in any rally is the return of serve.**

Rally bounds are then first-shot → last-shot (`rally_detection_shot_gap.py:88-91`),
and `cut_and_upload_rally_clips` cuts exactly those with no padding
(`modal_supabase_processor.py:81-82`).

Net effect on every clip:
- **start**: at the return of serve — serve, serve prep, ready position all cut (~1–2.5s);
- **end**: at the last racket contact — shuttle still airborne, so in/out and the
  winning moment are never visible (~0.5–1.5s).

The gradient detector already compensates for the tail (`rally_detection.py:228`:
`end_frame += max(1, int(0.5*fps))`). The shot-gap path, which now drives clip
cutting, adds nothing.

**Fixed** in `modal_supabase_processor.py`: `CLIP_PRE_ROLL_S = 2.0`,
`CLIP_POST_ROLL_S = 1.5`, applied by `pad_rally_windows()` at cut time only — so
`results.json` rally bounds stay the raw analytical window and no player/shot
metric moves. Padding may only consume the dead air *between* rallies, never
another rally's detected window; it never shrinks a window (relevant while §5 is
open and rallies can overlap); it clamps to `[0, ffprobe duration]`. The
`rally_clips` row now stores the padded bounds, because those describe the file
the apps play — the schema has no column for the unpadded window, which remains
available in `results.json`. Thumbnails are grabbed at the first detected shot
rather than 0.5s in, so they show play instead of the pre-serve pause.

Applies to both the Phase 1 cut and the Phase 2 re-cut, which share this function.

### The ffmpeg cut itself is correct — verified

Both `-ss` and `-to` sit before `-i`, making them input options, where `-to`'s
absolute-vs-relative semantics are a known footgun. Executed the exact production
command against a 60s source:

```
ffmpeg -y -ss 10 -to 20 -i src60.mp4 -c:v libx264 -preset ultrafast -crf 23 ...
→ output duration 10.000000   (absolute — correct)
```

Verified on ffmpeg 8.1 locally. The Modal image installs ffmpeg via
`debian_slim().apt_install("ffmpeg")` (`modal_supabase_processor.py:3468-3470`),
which pins a different version — worth one confirming run in the deployed image,
since if `-to` were relative every clip would be double length.

**Also**: the `cut_and_upload_rally_clips` docstring (lines 54-57) still describes
`-c copy` and keyframe-aligned starts, while the body re-encodes with libx264. Fix
the comment so nobody "optimises" it back to stream copy and silently reintroduces
±2s keyframe drift on every clip boundary.

---

## 2. The static-cluster shuttle filter is dead in both per-frame loops — and the naive fix makes things worse — [SHIPPING]

**Do not apply the one-line fix.** Read this whole section before touching it.

### 2a. The mechanism is dead

`_run_detection_only_loop` (Phase 1) and `_run_full_yolo_loop` (Phase 2) build
"static clusters" to reject fixed false positives, then prune to `count >= 3`
**unconditionally every frame** (`modal_supabase_processor.py:2261-2263`, `2866-2869`).
A cluster is created with `count = 1` and pruned in the same iteration, so it can
never reach the threshold. Verified by replaying the loop logic:

```
prune EVERY frame (lines 2261 / 2866):      0/200 frames suppressed, 0 clusters survive
prune only on accepted frame (line 2379): 198/200 frames suppressed, 1 cluster survives
```

`_build_shuttle_positions_dict:2379` places the identical prune **inside the
accepted-position branch only** (the `continue` paths skip it), which is why the
filtered track's filter works and the raw track's does not.

### 2b. But activating it propagates a worse failure

The per-frame movement gate at `2222-2229` already fires independently of the
cluster list: `movement < SHUTTLE_MIN_MOVE` → position dropped. Low-movement frames
are **already** excluded from the raw track today. The only behaviour the cluster
memory adds is *"once a location is known-static, suppress everything within ~24px
of it forever."*

And clusters are never aged out. Once one reaches `count >= 3` it survives every
subsequent prune. Any location where the shuttle appears to move `< _RALLY_MIN_MOVE`
(≈13px at 1080p) for 3 frames — a clear's apex, a slow drop shot, a net exchange —
becomes a **permanent ≈24px blind spot** for the rest of the match. Over a
50k-frame match these accumulate monotonically. This is the actual cause of the
"over-filtering trims rally tails" symptom the WIP comments describe.

So activating the prune as-is would propagate that blind-spot behaviour from the
filtered track into `skeleton_frames` → `skeleton_data` → the client timeline.

**Fix**: treat this as one change, not two. Either
(a) activate the prune **together with** cluster ageing (decay `count` when the
shuttle is seen moving elsewhere, or scope clusters to a sliding window of a few
hundred frames) and a temporal-contiguity requirement, so a cluster reflects a
genuinely persistent false positive rather than three co-located slow frames; or
(b) leave the cluster memory dead and rely on the per-frame movement gate, deleting
the dead code so it stops reading as active protection.

While there: the Phase 1 loop at line 2225 appends a fresh cluster unconditionally,
unlike `2369-2375` which looks for an existing one first — so repeat observations
of the same static point create duplicate `count=1` clusters instead of incrementing one.

---

## 3. ROI divergence between the two shuttle tracks — [SHIPPING]

Two different court polygons filter the same shuttle data:

| Track | Polygon | Effective size |
|---|---|---|
| raw (`_run_detection_only_loop:2098-2108`) | 1.02 × court, then **1.40 horizontal**, top clamped to `y=0` | very permissive |
| filtered (`_build_shuttle_positions_dict:2337`) | 1.02 × court, then **1.15 uniform** | ≈1.17 × court |

The filtered track discards any shuttle beyond ~1.17× the court footprint — high
clears and deep lifts near the baseline. Together with §2b this is the second half
of the trimmed-tails problem. Pick one polygon and use it for both tracks.

---

## 4. A trailing isolated shot welds dead air into the last rally — [SHIPPING, Python *and* TS] — ✅ FIXED

`rally_detection_shot_gap.py:76-78`:

```python
if gap > RALLY_GAP_SECONDS or is_last:
    end_idx = i + 1 if is_last else i
```

When the final index is **both** `is_last` and `gap > RALLY_GAP_SECONDS`, the
`is_last` branch wins and the isolated trailing shot is absorbed into the previous
rally. Verified:

```
input:  rally of 3 shots (1.0–3.0s), then ONE lone shot at 30.0s
output: [(1.0, 30.0, 4 shots)]   ← 27s of dead air welded into the clip
```

`rally_detection.py:221` handles this correctly
(`end_idx = i if (is_last and gap <= rally_gap_frames) else i - 1`). The shot-gap
port does not, and neither does the TS original
(`useAdvancedAnalytics.ts`: `shots.slice(rallyStart, i === shots.length - 1 ? i + 1 : i)`).

**Fixed** in both `rally_detection_shot_gap.py` and `useAdvancedAnalytics.ts`
(applied together to preserve the parity contract):

```python
end_idx = i + 1 if (is_last and gap <= RALLY_GAP_SECONDS) else i
```

Side effect worth knowing: two shots separated by more than `RALLY_GAP_SECONDS`
previously produced a bogus rally spanning the whole gap. Those now correctly
produce no rally, so rally *counts* may drop slightly on videos that had them.

**Corpus-level consequence**: the TS side is a `computed` over stored
`skeleton_data`, so fixing it retroactively changes the client timeline for
**every already-processed video**, while those videos' clips stay as originally
cut. The Python side only affects new runs. Decide whether existing videos get a
Phase 2 re-cut, or accept that old videos show a timeline that disagrees with their
clips.

---

## 5. `refine_rallies` produces overlapping clips and drags neighbours' bounds — [WIP]

Two distinct defects, both verified.

**(a) Clamps against un-refined neighbours.** `rally_detection_shot_gap.py:130-133`
clamps rally *i*'s start against `filtered[i-1]["end_timestamp"]` — the **original**
end — but rally *i-1*'s end was already widened by up to `max_extension_sec`.

**(b) `overlapping` collects raw rallies belonging to *other* filtered rallies.**
The overlap test at lines 121-124 matches any raw rally intersecting the current
one; taking `min(raw_start)` / `max(raw_end)` across all of them lets a welded or
long raw rally drag a neighbour's bounds far past its own detection.

Verified — filtered rallies `[10,20]` and `[23.5,33]`, raw `[9.5,25]` and `[21,34]`:

```
rally 1:  9.50 -> 23.10
rally 2: 20.40 -> 34.00
overlap = 2.70s
```

Rally 2 now starts **3.1s before its own detected start**, pulled back by raw
rally 1. Overlap is bounded by `max_extension_sec` (3.1s); 2.7s is reachable with
realistic inputs.

**Fix**: clamp against the already-refined neighbour, and restrict `overlapping` to
raw rallies whose midpoint falls inside this filtered rally:

```python
if refined:
    start = max(start, refined[-1]["end_timestamp"] + 0.05)
```

---

## 6. `max_extension_sec` defaults to the rally-gap threshold — [WIP]

`refine_rallies(..., max_extension_sec=RALLY_GAP_SECONDS)` = 3.1s **per side**, so an
isolated rally can gain 6.2s. 3.1s is the threshold that *separates* rallies; using
it as a bound-widening budget means a widened rally can swallow most of an
inter-rally gap. A value tied to actual shuttle flight time (~0.5–1.0s) is
defensible; 3.1s is not.

---

## ⚠️ Decide this before working §5 and §6

`refine_rallies` exists **because** the raw track is unfiltered and the filtered
track over-filters. That asymmetry *is* §2 + §2b + §3. Fix those and the two tracks
converge — at which point `refine_rallies` is close to a no-op and §5/§6 are fixes
to compensation code with nothing left to compensate for.

So: settle §2/§3 first, then decide whether `refine_rallies` survives at all.
Fixing §5 and §6 before that decision risks polishing code that should be deleted.

---

## 7. TrackNet frame indices are 0-based, the loops index 1-based — [SHIPPING]

- `tracknet/inference.py:248-267`: `frame_idx = 0` for the first decoded frame → **0-based**.
- `_run_detection_only_loop:2157-2161`: `frame_count += 1` *before* use → **1-based**.
- Both loops then do `tracknet_positions[frame_count]` (lines 2206, 2807).

Skeleton frame *N* — whose `timestamp` is the PTS of decoded frame *N-1* — is
assigned TrackNet's position for decoded frame *N*. Every shuttle position sits one
frame ahead of the image and timestamp it is attached to. `tracknet[0]` is never
read; the last frame gets no shuttle.

Impact on rally bounds is small (~33ms) and uniform, and both tracks share it — so
the WIP `refine_rallies` path is **not** dead; the two frame-index spaces do
coincide. But it is a real misalignment for anything correlating shuttle position
with the frame image or with player positions: shot attribution to a player,
shuttle speed, overlays.

**Fix**: index `tracknet_positions[frame_count - 1]`, or start `frame_count` at 0.

---

## 8. Gradient and shot-gap rallies are unioned across different time bases — [SHIPPING]

`rally_detection.detect_rallies` computes timestamps as `frame / fps`
(`rally_detection.py:108-109`) — synthetic and uniform. The shot-gap detector uses
`CAP_PROP_POS_MSEC` PTS values carried on `skeleton_frames`. `union_rallies` merges
the two directly. For VFR sources, or any file whose PTS drift from `n/fps`, the
overlap test compares two different time bases. Normalise to PTS, or assert CFR.

Minor: `detect_rallies` is called with its default `min_gap_duration_s=3.0`
(`modal_supabase_processor.py:3930`) while the shot-gap detector uses
`RALLY_GAP_SECONDS = 3.1`. Harmless today; should be one shared constant.

---

## 9. Phase 2 re-cut will silently rewrite annotated clips — [WIP, latent]

The re-cut (`modal_supabase_processor.py:4593-4630`) upserts on
`(video_id, rally_index)`, so `rally_clips.id` is preserved while
`start_timestamp` / `end_timestamp` / the stored MP4 all change.
`rally_annotations.clip_id` keeps pointing at the same row and `timestamp_seconds`
becomes an offset into footage that no longer matches.

**Latent, not live**: no annotation writer exists yet — `rally_annotations` appears
only in `supabase/migrations/0004`, its test, and four design docs
(`docs/plans/2026-05-02-rally-annotations-*`, `docs/plans/2026-05-04-kmp-*`). No
insert path in `src/`, and the KMP Android app is designed but unbuilt. This becomes
a live data-corruption bug the moment an annotation writer ships.

**Fix before that ships**: when a re-cut moves a clip's bounds beyond a small
epsilon, shift `timestamp_seconds` by the boundary delta or flag the annotation
stale. `delete_stale_rally_clips` is correct in shape but deletes annotations for
removed clips without warning.

---

## 10. Parity claims

**Holds** — the Phase 2 re-cut's "byte-for-byte browser parity" claim checks out.
`detect_rallies_from_shots(skeleton_frames, fps)` with `require_players=True` runs on
the same array serialised as `skeleton_data` (`modal_supabase_processor.py:4517`),
which is exactly what the client's `rallies` computed reads. The TS gate
(`shotDetection.ts:375`: `if (!skFrame || skFrame.players.length === 0) continue`)
matches the Python `require_players` gate. The re-cut runs before
`del skeleton_frames` (4614 < 4650). ✅

**Overstated** — the Phase 1 clip set and the Phase 2 clip set are produced by
different detectors on different data, so clip *boundaries*, not just clip *count*,
change under the user. `RallyReview.vue`'s new "preliminary" labelling is the right
mitigation; it should say the boundaries will change too.

---

## Recommended order

1. **§1 pre/post roll** — independent of everything else, largest visible win, low risk.
2. **§4 trailing-shot weld** — one line, Python + TS together; decide the corpus question.
3. **§2 + §3 together** — the root-cause work. Then decide whether `refine_rallies`
   (and therefore §5, §6) still has a job.
4. **§5, §6** — only if `refine_rallies` survives step 3.
5. **§7, §8, §9** — correctness cleanup; §9 before any annotation writer ships.

## Regression suite

`backend/scripts/verify_rally_bounds.py` — pure-function, no GPU/video/network,
runs in milliseconds. Guards the §1 and §4 fixes: rally grouping (weld case,
final-shot-inside-gap, the 2-shot boundary, genuine splits, bogus-gap rejection)
and clip padding (pre/post roll, video start, EOF clamp, neighbour clamp,
overlapping input, no mutation, unsorted input, missing duration probe).

Run it after any change to `rally_detection_shot_gap.py`, `pad_rally_windows`, or
the TS twin in `useAdvancedAnalytics.ts` — the TS grouping loop has no automated
guard and must be kept in sync by hand.

## Verification gap

No saved `results.json` or TrackNet dump exists anywhere in the repo — `runs/`,
`backend/processed/`, `backend/uploads/` and `tracker_metrics/` hold only YOLO
validation artifacts and tracker summaries. Every threshold above is currently
unfalsifiable on real footage.

Before tuning §2b/§6, persist one full `skeleton_frames` + `tracknet_positions` dump
from a real match and build an offline harness printing gradient / raw / filtered /
refined / union bounds side by side, plus per-stage shuttle-position counts
(raw-visible, dropped-by-ROI, dropped-by-static, kept). That turns "3.1s seems large"
and "static filtering may be over-aggressive" into measurements. Given there is no
test framework in this project, that harness is also the only regression guard the
rally pipeline would have.
