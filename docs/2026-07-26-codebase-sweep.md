# Codebase sweep — 2026-07-26

Scope note: the two audits dated 2026-07-25 cover rally segmentation/clipping and the
metric pipeline, and both state that **security, infra and UI were not audited**. This
sweep deliberately fills that gap. Everything below was traced to a concrete code path;
findings already documented elsewhere are listed at the end, not re-reported.

Mechanical gates first: `npm run type-check` (`vue-tsc --build`) passes clean.

---

## 1. Realtime is never enabled by a migration, and isn't in SETUP.md — ✅ FIXED

The entire progress UI is realtime-only: `AnalysisProgress.vue` reads the video row via
`useReactiveRow('videos', …)` and the log stream via `useReactiveList('processing_logs', …)`,
both of which do one `select` and then subscribe to `postgres_changes`.

Postgres only emits those events for tables in the `supabase_realtime` publication. No
migration adds them:

```
grep -rn "publication\|realtime" supabase/ --include=*.sql   -> no matches
```

and `supabase/SETUP.md` — which exists precisely to record the manual dashboard steps —
never mentions it either.

**The finding is reproducibility, not breakage.** Realtime demonstrably works in the live
project — so the enabling step happened, in a dashboard, and was never written down. The
repo therefore cannot reproduce a working environment: rebuild from these migrations plus
SETUP.md and the load-bearing publication membership is simply absent, with no error to
point at it. The progress screen's dependency on it is total (status transitions *and* the
log pane), so this is the single highest-value line of SQL missing from the repo.

**Fixed** — `supabase/migrations/0009_realtime_publication.sql` adds `public.videos` and
`public.processing_logs` to the publication, guarded by `pg_publication_tables` existence
checks so it applies cleanly to both a fresh project and the current one (where a bare
`alter publication … add table` would raise `duplicate_object` and fail the migration).
It also creates the publication if absent, for self-hosted/bare-Postgres targets.

Replica identity is deliberately left at the default; the reasoning is in the migration's
trailing comment. Short version: RLS-checked DELETE events would need `REPLICA IDENTITY
FULL`, which is not worth the WAL cost on `processing_logs` to buy only the
`useReactiveRow`/`useReactiveList` DELETE handlers.

`rally_clips` was left out on purpose — `RallyReview` polls and refetches, and the mobile
app fetches on appearance; neither holds a `postgres_changes` subscription.

`SETUP.md` §3 now states that Realtime needs **no** dashboard step (the migration owns it),
and gives the one-line query to check the publication if the progress screen ever stalls.
`supabase/tests/0009_realtime_publication.sql` asserts both memberships and re-runs the
migration body to prove idempotency.

**Not executed.** There is no `psql` on this machine and no `supabase/config.toml`, so no
local DB to apply it against — the SQL is unverified by execution. Run
`supabase db push`, then `psql "$DB_URL" -f supabase/tests/0009_realtime_publication.sql`.

---

## 2. No polling fallback when the realtime socket drops — the progress screen can hang forever — ✅ FIXED

`useReactiveRow` fetches once, then relies exclusively on the websocket. There is no
refetch on reconnect, no `SUBSCRIBED`-callback re-read, and no interval fallback.

Two concrete ways this strands the user:

- **Mount race.** The `await supabase.from(table).select(...)` completes *before*
  `.subscribe()` is called. Any `UPDATE` landing in that window (hundreds of ms) is lost
  forever. Narrow, but real on the fast transitions — e.g. `processing_phase2` →
  `completed` while the results page is hydrating.
- **Socket drop.** `AnalysisProgress.vue:107` tells the user *"You can leave this tab;
  results will be saved."* Backgrounded tabs, laptop sleep, and network changes are
  exactly when Supabase realtime disconnects. On reconnect the client resubscribes but
  never re-reads the row, so a `phase1_complete` / `completed` flip that happened while
  disconnected is never observed: the bar stays at whatever percent it last saw, and
  `watch(videoStatus)` never fires, so `phase1Complete` / `phase2Complete` never emit.
  The analysis is finished in the DB; the UI waits forever.

`RallyReview.vue` already solved this for itself — the new `checkVideoStatus` 8s poll
(`RallyReview.vue:150-175`) exists for precisely this reason. `AnalysisProgress` needs the
same treatment.

**Fixed** in both composables, so every consumer benefits and no component needed changing.
Four parts:

1. **Refetch on `SUBSCRIBED`.** The subscribe callback fires on the initial join *and* on
   every automatic rejoin, so one hook closes both the mount race and the reconnect gap.
   Verified against the vendored source rather than assumed — `channel.js:296` `rejoin()`
   → `joinPush.resend()` → `push.js:73` `reset()`, which clears `ref`/`receivedResp`/`sent`
   but leaves `recHooks` intact, so `matchReceive` (`push.js:88`) re-runs the `'ok'` hook
   and `RealtimeChannel.js:151/192` re-emits `SUBSCRIBED`. `channel.js:47,52,64,76,84` route
   socket-open, channel-error and timeout into that same `rejoin()`, so a half-open socket
   is covered too.
2. **Refetch on `visibilitychange` → visible.** Background tabs get timers throttled, so
   the heartbeat can lapse without a prompt rejoin. `AnalysisProgress` explicitly invites
   the user to leave the tab, so returning to it is now a refresh — covering the case where
   the rejoin is slow or never comes.
3. **Monotonic write ordering.** A refetch and a realtime push can be in flight together,
   and a `SELECT` issued earlier can resolve later — applying it would overwrite newer data
   with older. Every write now takes a ticket and only the newest wins. Without this, the
   fix would have introduced a fresh race in the course of removing one.
4. **The initial read stays eager**, before the handshake. Fetching *only* on `SUBSCRIBED`
   would have made first paint wait on the websocket and shown nothing at all if realtime
   were unavailable — i.e. strictly worse than the bug. Background refetches are quiet:
   they never raise the spinner, and a failed one never replaces good data with an error
   banner.

Two related defects fixed in passing, both surfaced by the above:

- Both composables were `async` `watchEffect`s registering `onCleanup` **after** an `await`.
  Vue only reliably attaches a cleanup registered before the first await, so a rapid `id`
  change could leak a channel. Both effects are now synchronous, with the awaiting moved
  inside `load()`.
- `useReactiveList`'s INSERT handler appended blind, and `load()` replaced the array
  wholesale. A push arriving during an in-flight `SELECT` would be dropped by the replace,
  and a row delivered by both paths would appear twice. Now: dedupe by id on insert, and
  `load()` merges rather than replaces.

No deliberate polling. `SUBSCRIBED` + visibility covers the realistic failures without
adding steady-state query load; `RallyReview`'s own 8s poll stays, since it keys on a
terminal condition the generic composable cannot know.

---

## 3. A Phase-1 failure is unrecoverable without re-uploading the whole video

`App.vue:1459` renders the retry button only for phase 2:

```html
<button v-if="lastFailedPhase === 'phase2'" @click="handleContinueAnalytics">Retry analytics</button>
```

For `failed_phase1` the only affordance is "start new analysis", which resets to the
upload screen. And there is no way back even in principle from the client:

- `process-video` requires `video.status !== 'uploaded'` → 409;
- `0002_rls_policies.sql` does `revoke update on public.videos from authenticated` and
  re-grants only `(filename, manual_court_keypoints, player_labels)` (+ `pipeline_variant`
  in 0007) — `status` is deliberately not client-writable.

**Failure scenario.** A 2 GB match video uploads fine, Phase 1 dies on a transient Modal
error (OOM, GPU eviction, the `fps` edge cases the metric audit catalogued). The bytes are
sitting in the `videos` bucket, the keypoints are set — and the user's only option is to
upload the whole file again.

**Fix.** Either let `process-video` also accept `failed_phase1` (resetting
`error`/`progress` on the flip, mirroring `start-analytics` step 4), or add a tiny
`retry-phase1` edge function. The service-role client in the function is already the right
place to flip the status.

---

## 4. Edge functions authorize the row, not the path — service-role reads follow client-written path columns

`0002` restricts *UPDATE* on `videos` column-by-column. **INSERT is unrestricted** — and
`VideoUpload.vue:110-122` shows the client is the one who writes `storage_path`. So a
client can insert a perfectly valid, RLS-passing row (`owner_id = auth.uid()`) whose
`storage_path` / `results_storage_path` point at *someone else's* object.

The functions then read the row with a user-scoped client (correctly proving ownership of
the **row**) and hand the path straight to a **service-role** client, which bypasses
storage RLS:

- `duplicate-video/index.ts:44-46` — `adminClient.storage.from("videos").copy(video.storage_path, newPath)`
  copies the referenced object into `${user.id}/…`, where the attacker can then read it
  under the ordinary owner-read policy.
- `export-pdf/index.ts:36-45` — passes `results_storage_path` to Modal, which renders it
  and returns the PDF to the caller.

**Why this is hardening, not a live breach:** cross-tenant path confusion needs a second
tenant, and `SETUP.md:52` specifies *"Disable signups: ON (so only you can create users
from the dashboard)"* — today there is only one. It is unreachable as deployed. It becomes
live the moment signups open, and at that point the only thing between two tenants is UUID
entropy (`<owner_id>/<video_id>.mp4` — infeasible to guess, but entropy is not
authorization) plus any future feature that surfaces a path: an admin view, a support
tool, a shared link, a log export.

Worth fixing before signups open rather than after.

**Fix.** Both functions already know `user.id`. Before using a path, assert
`video.storage_path.startsWith(`${user.id}/`)` (same for `results_storage_path`) and 403
otherwise. Two lines each. Longer term, restrict `videos` INSERT to the columns a client
should own, the way UPDATE already is.

---

## 5. `progress` is written on two different scales — the bar snaps to 1% at each phase end — ✅ FIXED

In-flight updates write a percentage:

- `modal_supabase_processor.py:2386` — `progress = (frame_count / total_frames) * 100`
- `modal_supabase_processor.py:3424` — same

The two completion calls write a fraction:

- `modal_supabase_processor.py:4298` — `send_status_update("phase1_complete", progress=1.0, …)`
- `modal_supabase_processor.py:4865` — `send_status_update("completed", progress=1.0, …)`

`send_status_update` stores the value verbatim (`"progress": progress`, lines 3860 / 4449),
and `AnalysisProgress.vue:100` renders `Math.min(100, Math.round(progress))`.

**Failure scenario.** At the end of each phase the progress bar jumps from ~99% down to
**1%** for the moment before the status watcher swaps the view — reading as a crash right
at the finish line. `calculateETA()` also divides by `progress/100`, so it prints a wildly
inflated "remaining" on that same tick.

**Fixed** — both call sites now pass `progress=100.0`, with a comment at the phase-1 site
recording that the column is percent-scaled and why (the next person editing a
`send_status_update` call is the one who needs to know). Both frontend consumers were
re-checked against the new value: `progressPercent`'s `Math.min(100, …)`
(`AnalysisProgress.vue:107`) and the ETA divide (`:326`) are both correct at 100.

---

## 6. `getSpeedTimeline` hardcodes 30 fps in a response typed as carrying real fps

`src/services/api.ts:533` — `players[pid].timestamps.push(frame.frame / 30)`
`src/services/api.ts:551` — `fps: 30`
(and `api.ts:383`, the cached branch of `getSpeedData`, likewise returns `fps: 30`)

`SpeedTimelineResponse` declares `fps: number` and `timestamps: number[]`, so consumers
have every reason to trust them.

**Failure scenario.** Latent today — the only caller (`App.vue:547`) reads just `frames`
and `speeds_kmh`, so nothing is currently wrong on screen. But 60 fps phone footage is
common and nothing in the upload path normalises frame rate; the first component to plot
against `timestamps` will show a timeline stretched 2× on 60 fps video and compressed on
25 fps, with no error anywhere. This is the same class of defect the metric audit's §6
just removed from the backend.

**Fix.** Thread the real fps (it's in `results.json` / `results_meta`, and
`recalculate-speeds` already returns one) instead of the literal.

---

## 7. Minor: orphaned storage object when the `videos` insert fails

`VideoUpload.vue:104-125` uploads the bytes, then inserts the row. If the insert fails the
object is left in the bucket with no row pointing at it — invisible to the app, counted
against storage, never cleaned up. `duplicate-video` already does exactly the right thing
here (`index.ts:66-71` removes the copied object when its insert fails); the upload path
should mirror it.

---

## 8. `0008_video_title.sql` shipped with no DDL — the schema lives in the dashboard, not the repo — ✅ FIXED

Found while pushing the 0009 fix. The file was **15 lines of comments and nothing else** —
the `alter table public.videos add column title …` statement specified in
`docs/plans/2026-07-25-match-title-design.md` §5.1 never made it in. Only the *rationale*
for the accompanying grant decision was written.

The feature works today because the column was added by hand in the SQL Editor. So this is
finding #1 again, in a second place: **the repo cannot rebuild this database.** Two of the
last two schema changes exist only as dashboard actions. That's now a pattern, not an
incident — worth a habit of `supabase db reset` (or a scratch project) before calling a
migration done, since a comment-only migration file "succeeds" loudly and silently.

**Fixed** — the statement is in, split into a guarded `add column if not exists` plus a
separately guarded named constraint (`videos_title_length_check`). Both guards are load
bearing: a bare `add column` aborts `db push` against the live project, and
`add column if not exists` on its own would skip the CHECK there while applying it to a
fresh build, leaving the two schemas permanently divergent.

**Worth checking on the live DB**, since the manual column was created outside this file:

```sql
select data_type, character_maximum_length
from information_schema.columns
where table_schema='public' and table_name='videos' and column_name='title';
```

`add column if not exists` skips on name alone — if the hand-made column isn't plain
`text`, the type divergence survives this migration silently.

---

## Already documented, still open (not re-reported above)

For completeness, so these aren't mistaken for new ground:

- `docs/2026-07-25-rally-detection-clipping-audit.md` §2 (static-cluster filter dead in
  both loops), §5 (`refine_rallies` overlapping clips — it clamps against the *unrefined*
  previous bound, `rally_detection_shot_gap.py:135-138`), §6 (`max_extension_sec` default),
  §9 (Phase-2 re-cut rewrites annotated clips, since `rally_index` is the upsert key and
  means a different rally after a re-cut).
- `docs/2026-07-25-metric-pipeline-audit.md` §7 (distance as an unsmoothed noise integral —
  needs a real skeleton dump to size), plus the standing "validate on real footage" item.

## Stale comment worth a one-line fix

`start-analytics/index.ts:66` describes its status flip as a *"deliberate divergence from
`process-video`, which flips AFTER Modal returns OK"*. `process-video` was since changed to
flip **before** the Modal call too (`process-video/index.ts:52-58`). The comment now
documents behaviour that no longer exists.
