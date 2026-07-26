# Metric pipeline audit — speed, distance, player identity — 2026-07-25

Companion to `2026-07-25-rally-detection-clipping-audit.md`. That one covered rally
segmentation and clipping; this one covers the **per-player metric pipeline**:
speed, distance, and the identity tracking everything else depends on.

Scope was deliberately narrowed to accuracy of the numbers shown to the user. UI
polish, infra tuning and security were not audited.

Verification status per finding. **[REPLAYED]** = the real code was executed offline
and its output is quoted. **[ARITHMETIC]** = constants read by grep, consequences
computed. **[SOURCE]** = read from source, mechanism reasoned, not executed.

Ranked by impact. §1 is the most severe and the cheapest to fix.

---

## 1. Degenerate net keypoints silently disable all court-side identity logic — [REPLAYED] — ✅ FIXED (defensive)

> **Severity corrected 2026-07-25.** An earlier revision of this document claimed
> this was reachable through ordinary UI use, citing
> `:disabled="manualKeypoints.length === 0"` on `CourtSetup.vue:407`. **That
> `:disabled` is on the Undo button, not Apply.** The confirm button is gated on
> `v-if="isComplete"` where `isComplete = manualKeypoints.length === TOTAL_KEYPOINTS`
> (12), and `saveAndProceed` re-checks it — so `CourtSetup` never emits `[0,0]` net
> endpoints. I also checked the other candidate writer (`App.vue`, below): its
> handlers are dead code. **No live path produces a degenerate net line today.**
> This is defensive hardening, not a live bug fix.

`CourtSetup.vue:334-335` emits the net endpoints with a `?? 0` fallback, so an
under-specified payload *would* be `[0, 0]` — it just can't be produced by the
current UI. The worker's guard (`modal_supabase_processor.py:2663-2668`) checked
only type and length, never non-degeneracy, so `(0,0)/(0,0)` was accepted as a valid
net line. Replayed against the real tracker:

```
net_line        = (0.0, 0.0, 0.0, 0.0)
court_midline_y = 0.0

  _get_court_side(y= 100.0, x=960) -> 'bottom'
  _get_court_side(y= 540.0, x=960) -> 'bottom'
  _get_court_side(y=1000.0, x=960) -> 'bottom'

court_sides after calibration (players clearly on OPPOSITE halves): {0: 'bottom', 1: 'bottom'}
swap corrections after 40 genuinely-swapped frames: 0
lone player at TOP of frame re-acquired as: []
```

`dx == 0` makes the net-interpolation branch unreachable, and the fallback compares
against `court_midline_y = 0`, so **every** point on the frame classifies as
"bottom". Consequences, all silent:

- `W_COURT_SIDE` — the heaviest weight in the cost matrix (0.8 × 1.5) — never
  differentiates, because every skeleton always matches every expected side. The
  tracker's strongest identity anchor is inert.
- Swap detection **never fires**: `violations` is always 0.
- Single-visible-player re-acquisition returns nothing, so those frames carry no
  player id at all.

The log line at `modal_supabase_processor.py:2678` reports *"manual net keypoints"*
in this state, actively hiding the problem.

**Fixed** — new `valid_net_line()` in `modal_supabase_processor.py`, applied at the
worker's keypoint guard. It rejects both-at-origin, identical endpoints,
insufficient horizontal separation (< 1% of frame width, so the y-at-x interpolation
stays well-conditioned), negative coordinates, and out-of-frame endpoints. On
rejection the worker falls back to the pixel midline and logs a **warning** naming
the offending coordinates, instead of reporting "manual net keypoints".

No `CourtSetup.vue` change was needed — it already requires all 12 keypoints.

**Related landmine, also fixed.** `App.vue:868` and `:899` wrote
`manual_court_keypoints` as a **4-corner-only** object. `manual_court_keypoints` is
`jsonb`, so an update replaces the whole value — that write would have *deleted*
`net_left`/`net_right`, silently dropping the tracker to the midline fallback on
every later run. Both handlers (`handleCourtKeypointsSet`, `handleKeypointsConfirmed`)
are currently **dead code**: `VideoPlayer`'s `defineEmits` declares only
`timeUpdate`/`frameUpdate`/`play`, and the template binds only those, so neither is
reachable. They now send the full 12-point set, so wiring them up later is safe.

---

## 2. Calibration can lock both players to the same court side — [REPLAYED] — ✅ FIXED

Independent of §1. `_run_calibration` completes at frame 15 and sets each player's
side with `court_sides[pid] = self._get_court_side(avg_y, avg_x)`, computed
independently per player, with **no check that the two differ**.

If the first 15 frames (0.5s at 30fps) have both players on the same half of the
midline — a warm-up, a handshake, an intro shot, a serve preparation with both
near the net — both lock to the same side. Replayed with the faithful call pattern
(the real loop only calls `_run_calibration` while `not calibration_complete`):

```
init court_midline_y = 540.0   (no net keypoints -> pixel centre)
completed at frame 15
court_sides : {0: 'bottom', 1: 'bottom'}
refined court_midline_y = 830.0
single-skeleton re-acquisition -> []   (pid 1 unreachable)
```

Note the ordering problem that causes it: `court_sides` is assigned at lines
1501-1507 using the **unrefined** midline, and only then is `court_midline_y`
refined to the midpoint of the two players' averages (lines 1510-1513). When both
players start on one half, that refinement places the tracker's notion of the net
*inside* one half of the court, and it governs every court-side decision for the
rest of the video.

**Fixed** in `_run_calibration`:

- The midline refinement now runs **before** sides are derived from it. Placing the
  midline midway between the two observed players makes opposite sides *automatic*
  whenever no net line is available — the same-side outcome becomes arithmetically
  impossible on that path.
- Sides are compared, and if they still match (only possible with a real net line,
  meaning the window is unrepresentative — a warm-up, a walk-on, one player fetching
  the shuttle) calibration **waits and re-evaluates** instead of locking an
  impossible state.
- New `CALIBRATION_MAX_FRAMES = 300` (10s at 30fps) bounds that wait. On expiry the
  net line is treated as empirically contradicted for this clip: it is **dropped**
  and players are split by relative position. Dropping it matters — keeping the net
  line while forcing a relative split would leave live classification disagreeing
  with the locked assignment on every frame.
- Calibration also no longer completes with observations for only one player.

*(One hypothesis I tested and refuted: calibration does **not** complete when a
2-player frame is never seen — the `len(skeletons) < 2` early return precedes the
completion check, so `calibration_complete` correctly stays `False`.)*

---

## 3. ~~Player identity flips at every end change~~ — NOT APPLICABLE (premise refuted)

**Closed 2026-07-25.** The premise was that a single uploaded video spans two or
more games. Confirmed with the project owner: **a video is always exactly one game,
and the players never switch sides.** The end-change flip cannot occur. No action.

This is a load-bearing invariant, so state it positively:

> **Fixed-sides invariant** — within any uploaded video, each player stays on one
> side of the net for its entire duration.

Two consequences that change the rest of this document:

1. Court side is a **fully reliable** identity anchor, not the approximation the
   tracker currently hedges against. See the Opportunity note after §4 — this
   invariant lets identity be made essentially bulletproof.
2. It **raises the stakes on §1 and §2**. Identity now rests *entirely* on court
   side, and those two findings are precisely the two ways that anchor gets
   destroyed. §1 in particular switches it off completely.

Original analysis retained for the record only.

<details>
<summary>Original finding — does not apply under the fixed-sides invariant</summary>

`court_sides[pid]` is fixed at calibration and never revised.
`_detect_and_correct_swap` treats "both players on the wrong side" as a swap to
correct *towards* those fixed sides — the code says so explicitly: *"Court_sides[pid]
stays fixed — those are the canonical identity anchors."*

Players change ends between games (and at 11 points in the third). After an end
change both are permanently on their "wrong" side, so the corrector fires:

```
simulating a post-end-change stretch (both players opposite sides)
swap correction fired at frame 6, new assignments [(1, 0), (0, 1)]
```

Player A, tracked continuously as pid 0, becomes pid 1. Every per-player metric —
distance, speed, zones, heatmap, summary, thumbnail — then contains **game 1 of one
person spliced to game 2 of the other**.

A second path produces the same flip independently: the `n_skeletons == 1` branch
(lines 1713-1750) re-acquires a lost player **by court side**, and tracking is
routinely lost while players walk around the net posts during an end change.

This is a design assumption, not a slip: identity == court side is right within a
game and wrong across a match.

**Fix, cheapest first**: detect the end change (a long rally gap plus both players
crossing) and swap `court_sides` instead of swapping the player assignment — or
surface the candidate end-change frame for the user to confirm and re-key the stats
around it. Both are scoped to this codebase. Appearance embedding or shirt-colour
matching would be more robust but is a new subsystem; don't start there.

</details>

---

## 4. `SWAP_CONSECUTIVE_THRESHOLD` does not require consecutive violations — [REPLAYED] — ✅ FIXED

`_detect_and_correct_swap` increments `swap_violation_count` on 2 violations and
resets it **only** on *zero*. A frame with exactly one violation neither increments
nor resets, so the counter survives interruptions:

```
feeding ALTERNATING frames (2 violations / 1 violation, never 0)
counter reached the threshold and swapped at frame: 11
```

A swap fired without the violation ever being sustained for the 6 consecutive
frames the constant name promises. Risk concentrates exactly where side
classification is noisiest — players near the net line.

Doc/code mismatches in the same class: the class docstring says swaps are confirmed
*"over 2 consecutive frames"* while `SWAP_CONSECUTIVE_THRESHOLD = 6`; and
`_match_to_prev_skeleton`'s docstring says *"half the max match distance"* while the
code uses the full `MAX_MATCH_DISTANCE`.

**Fixed**: the counter now resets on any frame with fewer than 2 violations
(previously only on exactly 0). A single violation — the normal signature of one
player being briefly mislocalised near the net line — is precisely what must not
accumulate. The suite confirms a sustained swap still fires at frame 6.

---

## Opportunity unlocked by the fixed-sides invariant — ✅ IMPLEMENTED

Because players never switch sides within a video (§3), court side is not a
heuristic — it is ground truth for the whole clip. The tracker currently treats it
as one weighted term among six (`W_COURT_SIDE = 0.8`, penalty ×1.5) that distance,
velocity and stickiness can outvote. Under the invariant it can instead be a **hard
constraint**: once calibrated, a skeleton on the top side can only ever be the top
player.

That would make the cost matrix a 1-to-1 choice within each side rather than a
2×N global assignment, and it would eliminate as *structurally impossible*:

- swap oscillation and the whole `_detect_and_correct_swap` mechanism (§4 disappears
  rather than being fixed);
- the majority-vote smoothing layer and its latent index bug (§8 likewise);
- mid-rally identity flips when players cross visually or occlude each other, which
  is where the weighted cost is weakest.

Prerequisites: §1 and §2 must be fixed first, because a hard constraint on a *wrong*
side assignment is worse than a soft one — it becomes unrecoverable.

**Implemented 2026-07-25**, after §1 and §2. How it works:

- `_claimable_skeletons(pid, skeletons)` returns only the skeletons on that
  player's side of the net. Because calibration guarantees the two players hold
  *different* sides, the two candidate lists are disjoint — so no ordering of costs
  can make the players trade places.
- The cheapest disjoint pair drawn from the two candidate lists wins. The composite
  cost (distance, velocity, area, track-ID, stickiness) now only breaks ties
  *within* a side.
- When no disjoint pair exists — one player isn't visible on its side this frame —
  the better-supported player is placed and the other is **left unassigned**. A
  missing frame costs a longer `dt` in the speed calculation; a wrong frame corrupts
  both players' statistics.
- `NET_BAND_PX` (1% of the long edge, ≈19px at 1080p) is a hysteresis band around
  the net line where the constraint relaxes, so a lunging front foot crossing the
  net line by a few pixels doesn't cost a frame. This is the only place the two
  candidate sets can overlap, and the disjoint-pair rule resolves it.

**Deleted** (145 lines): `_detect_and_correct_swap`, `_apply_majority_vote`, and the
state they needed — `SWAP_CONSECUTIVE_THRESHOLD`, `swap_violation_count`,
`total_swaps_corrected`, `VOTE_WINDOW_SIZE`, `assignment_vote_history`. The
`n_skeletons == 1` special case is gone too: the constrained path handles one
skeleton naturally, and its YOLO-track-id fallback was unreachable once the two
players are guaranteed different sides. This also closes **§8** — the latent
majority-vote index defect no longer has a mechanism to live in.

**New health signals — two, kept deliberately separate.** There is no swap counter
because swaps cannot occur.

- `frames_unsplittable` — 2+ skeletons existed but could not be split one-per-side,
  so a player was dropped. This is the constraint's *own* failure mode. Warned above
  5%.
- `frames_single_skeleton` — only one skeleton existed at all. Upstream pose
  coverage, not identity, and **common** on real footage (`useAdvancedAnalytics`
  notes pose "frequently drops the far player"). Informational only.

Merging them would have been a mistake: with one skeleton a disjoint pair is
impossible *by definition*, so a combined counter would have been large on healthy
video and the warning would fire routinely — making the one instrument meant to
validate this change unable to distinguish its failure mode from normal noise.

**`NET_BAND_PX` is the one knob this change adds, and `frames_unsplittable` is how to
size it.** It defaults to 1% of the long edge (≈19px at 1080p). Too narrow and net
play drops frames; too wide and the candidate sets overlap often enough that the
disjoint-pair tie-break starts doing real work instead of being a formality. It
cannot be sized from source — on the first real match, if `frames_unsplittable` is
near zero the band is right; if it spikes, widen it and re-measure.

**Downstream coverage — checked, no change needed.** The constraint drops frames the
old global assignment covered, so I traced the two consumers that depend on
two-player frames:

- `useAdvancedAnalytics.rallySpeedStats` — its 30% reliability gate is
  `speeds.length / totalFrames`, and `totalFrames` only increments when a player
  *appears* in `frame.players`. A dropped frame leaves both numerator and
  denominator, so the ratio is unaffected. (Its `distanceCovered` does integrate
  only non-zero samples, so it shrinks slightly with dropped frames — a per-rally
  display metric, not the headline distance.)
- `_pick_best_two_player_frame` — requires ≥25% of video height of Y-separation, so
  it only ever selects frames where the players are far apart, which are exactly the
  frames the constraint assigns most confidently and furthest from the net band. It
  also already falls back to "best anywhere" across the whole video.

---

## 5. Three speed/distance implementations with four threshold sets — [ARITHMETIC] — ✅ FIXED

| # | Where | Speed cap | Pixel-jump gate |
|---|---|---|---|
| 1 | Phase 2 inline loop (`modal_supabase_processor.py:3277-3350`) — writes `results.json` | `MAX_VALID_SPEED_MPS = 8.5` → **30.6 km/h** | `max(80, 0.07 × max(w,h))` — resolution-scaled |
| 2 | `speed_calc.py` — `recalculate_speeds`, returned to the browser | `MAX_REALISTIC_SPEED_KMH = 25.0` | `MAX_FRAME_JUMP_PIXELS = 80.0` — **not** scaled |
| 3 | `App.vue:640-700` — overwrites in-memory player summaries | caps at **25** | n/a |

```
at 1280x720 : inline 89px   vs speed_calc 80px
at 1920x1080: inline 134px  vs speed_calc 80px
at 3840x2160: inline 268px  vs speed_calc 80px
```

`speed_calc.py` rejects strictly more frames than the run that produced the stored
numbers — at 1080p its gate is 1.7× tighter — so it reports lower distance and can
report a lower max speed for identical input.

**What this does and does not affect.** `recalculate_speeds` does **not** write to
the DB (verified: the handler returns JSON, and
`supabase/functions/recalculate-speeds/index.ts` passes it straight back), so it is
not destructive on click. `App.vue` fires it automatically once keypoints are
confirmed and playback starts, then overwrites `player.avg_speed`, `max_speed`,
`total_distance`, and rewrites `skeleton_data[].players[].current_speed` (zeroing
anything > 25 km/h).

The PDF **does agree with the UI**: `ResultsDashboard.vue:98-104` posts its
overwritten `players` array as `config.players`, and `modal_pdf_export.py:808-811`
prefers `config["players"]` when present ("for consistency with dashboard"). So the
two user-facing surfaces match.

The real consequences are narrower but still real:

- **`results.json` — the durable record — disagrees with everything the user sees.**
  It stores the implementation-1 numbers. Any consumer reading it directly gets
  different speeds and distances: the KMP Android app when it ships, and
  `downloadPDFExport` in `api.ts:663`, which sends no `players` and therefore
  renders the stored numbers. (That function is currently unreferenced in `src/` —
  a latent path, not a live bug.)
- **Phase 2 spends GPU time computing per-frame speeds that are discarded** and
  recomputed client-side in the normal flow.
- The comment at `App.vue:648` claims the 25 km/h constant is *"UNIFIED with backend
  speed calculator (modal_supabase_processor.py)"*. That file uses 8.5 m/s = 30.6
  km/h. The comment is false and has likely shielded the divergence from review.

**Fixed**: `speed_calc.py` is now the single source of truth and the Phase 2 loop
imports from it — `MAX_REALISTIC_SPEED_MPS`, `SPEED_MEDIAN_WINDOW`,
`max_frame_jump_pixels()` and `median_speed_rejects()`. The loop's private
`MAX_VALID_SPEED_MPS = 8.5` and hand-rolled pixel gate are gone, and
`verify_speed_filters.py` scans the worker source to assert they are not
reintroduced.

The pixel gate now scales with resolution **and** frame rate (it previously
ignored frame rate in the loop and ignored both in `speed_calc`), and the median
spike logic is one shared helper rather than two hand-copied blocks.

Client-side re-capping is **removed** from both `App.vue` paths. It was masking
this very divergence, and in the `skeleton_data` path it was actively corrupting
downstream numbers: zeroing a speed above the local threshold dropped it out of
`useAdvancedAnalytics.rallySpeedStats`, which aggregates only non-zero
`current_speed` — so a player's fastest movements vanished from the per-rally
stats. The false *"UNIFIED with backend"* comment is gone with it.

**Behaviour change to expect**: the effective cap during Phase 2 was previously
27 km/h (the metres-per-frame gate bound before the 30.6 km/h speed cap at
30fps). It is now 25 km/h everywhere, so stored max speeds may come down slightly
and distances with them. Both surfaces and `results.json` now agree.

---

## 6. `MAX_DISTANCE_PER_FRAME_M = 0.25` is fps-dependent, hardcoded for 30fps — [ARITHMETIC] — ✅ FIXED

Both implementations gate on metres **per frame**, not per second:

```
at 25fps -> implied speed ceiling 22.5 km/h
at 30fps -> implied speed ceiling 27.0 km/h
at 50fps -> implied speed ceiling 45.0 km/h
at 60fps -> implied speed ceiling 54.0 km/h
```

At 30fps it sits just above the nominal caps and behaves as designed. At 25fps it
becomes the **binding** constraint and rejects legitimate 22.5–25 km/h movement,
under-counting distance. At 50/60fps it is inert, leaving the speed cap as the only
protection against tracking jumps. 60fps phone footage is common and nothing in the
upload path normalises or asserts frame rate.

**Fixed by deletion.** Expressing it as `MAX_SPEED / fps` would have made it
*exactly* the speed cap: since `speed_kmh == (d_metres / frames_elapsed) * fps * 3.6`,
a per-frame distance limit **is** a per-second speed limit divided by fps. The two
gates were always the same gate with mismatched constants, which is why one bound
at 25fps and the other at 60fps. The constant is removed and the speed cap does the
job at every frame rate.

`verify_speed_filters.py` asserts the property that matters: identical physical
motion measures the same at 25/30/50/60fps (spread < 0.5 km/h), and 23.5 km/h — which
the old gate zeroed at 25fps while passing at 30fps — now survives at both.

---

## 7. Distance is an unsmoothed integral of frame-to-frame displacement — [SOURCE]

Both implementations accumulate `distance += hypot(Δx, Δy)` per frame from the
ankle-midpoint position, with no smoothing and no minimum-movement floor; the only
bound is the upper gate in §6. Pose jitter therefore **adds** to total distance
rather than averaging out, since every displacement contributes its absolute
magnitude regardless of direction.

Unquantified, deliberately: the inflation depends on the jitter amplitude of real
skeleton data, and no saved run exists in the repo. **The measurement that would
settle it**: take a segment where a player is standing still and read off the
distance accumulated across it — all of it is noise. That number also tells you
what dead-band, if any, is justified. Do not tune this blind.

`skeleton_center_from_keypoints` already prefers the **ankle midpoint**, so
positions are on the court plane and the homography is being applied correctly —
this is a noise problem, not a projection problem.

---

## 8. Latent, currently unreachable: majority vote can reassign to a skeleton the cost matrix never chose — [REPLAYED, then proven unreachable]

`_apply_majority_vote` builds a canonical two-slot mapping from skeleton indices. If
an assignment references index ≥ 2, `slot_to_pid.get(1, 1)` supplies a default and
the reconstruction points at slots {0,1}:

```
cost-matrix assignments : [(0, 0), (1, 2)]   (player1 -> skeleton 2)
after majority vote     : [(1, 0), (0, 1)]
      -> now points at skeleton 1, which the cost matrix never selected;
         the real player at skeleton 2 is dropped from the frame.

raw assignments [(0,0),(1,2)] recorded in vote history as: [(0, 1)]
      -> the history records a mapping that never happened.
```

**This cannot currently happen.** `active_skeletons` is capped at 2 at all three of
its assignment sites (`modal_supabase_processor.py:3160-3181`): the `<= 2` branch
needs no truncation, the other two apply `[:2]`. So `n_skeletons > 2` never occurs,
which also makes the all-pairs branch at `match_skeletons:1779-1793` dead code.

Reported because the tracker *looks* like it handles N skeletons while the invariant
that saves it lives 1400 lines away in the caller. Anyone relaxing that truncation —
to support doubles, say — activates a silent wrong-player bug.

**Fix**: assert `len(active_skeletons) <= 2` at the top of `match_skeletons`, or
delete the `> 2` branch, or make the slot mapping index-safe. Don't leave it as a
landmine for doubles.

---

## 9. `fps == 0` disables rally detection silently — [SOURCE] — ✅ FIXED

Phase 1 reads `fps = cap.get(cv2.CAP_PROP_FPS)` (line 3931) with no validation.
OpenCV returns 0 for some containers and variable-frame-rate sources. The chain:

- `detect_rallies` returns `[]` (guard `fps <= 0`);
- `detect_shuttle_shots` returns `[]` (guard `not fps or fps <= 0`);
- no rallies → no clips → no `rally_clips` rows;
- `results.json` stores `fps: 0`, and the client does `result.fps || 30`, so the web
  app **still renders a client-side timeline at 30fps** while the backend produced
  nothing.

The user sees a completed analysis with a timeline and zero clips, with no error
anywhere. Phase 2 has an `else 30.0` guard at line 2596; the Phase 1 probe has none.

**Fixed** via a shared `normalize_fps()` helper (returns `(fps, was_substituted)`)
applied at **three** points, not one:

- the Phase 1 probe, with a warning log naming the probed value;
- **Phase 2's fps resolution** — `if not fps or fps <= 0: fps = probe_fps` trusted the
  probe, but the probe returns 0 on exactly the containers Phase 1 struggled with. A
  0 reaching Phase 2 is worse than a bad number: `dt = frames_elapsed / fps` in the
  speed loop is a **ZeroDivisionError**. (`effective_fps` at line 2681 looked like it
  covered this, but it only guards `lost_buffer_frames`.)
- the top of `_run_full_yolo_loop`, as a boundary guard, since that helper divides by
  the rate its caller hands it.

Handles 0, negative, `None`, non-numeric, NaN and infinity; passes 25/29.97/60
through untouched.

---

## Suggested order

Updated 2026-07-25. Everything actionable in this document is now either fixed or
explicitly deferred with a stated reason.

Done: ~~**§1**~~ (defensive hardening — no live path produced it), ~~**§2**~~ (was
live), ~~**§4**~~, ~~**§5**~~, ~~**§6**~~, ~~**§9**~~.
Closed: ~~**§3**~~ — not applicable under the fixed-sides invariant.

Done: ~~**§1**~~, ~~**§2**~~, ~~**§4**~~, ~~**§5**~~, ~~**§6**~~, ~~**§9**~~, and the
~~**Opportunity**~~ (court side promoted to a hard constraint, which deleted §4's
mechanism along with ~~**§8**~~).
Closed: ~~**§3**~~ — not applicable under the fixed-sides invariant.

Remaining:

1. **§7** — distance-as-noise-integral. The only finding left, and the only one with
   no safe fix available from source alone. Needs a real skeleton dump: capture one
   match, read the distance accumulated while a player stands still, and that number
   tells you what dead-band is justified. Do not tune blind.
2. **Validate on real footage.** Every fix here is verified against replays and
   synthetic fixtures, never video. On the first real run, check three things:
   - `frames_unsplittable` near zero → `NET_BAND_PX` is sized right for this camera
     angle. If it spikes, widen the band and re-measure.
   - the 25 km/h cap did not clip legitimate movement (compare a known-fast rally
     against the previous run's numbers, which used an effective 27 km/h ceiling).
   - the identity-tracker summary reports `net_line=yes`. `midline fallback` means the
     net keypoints were rejected or missing, and court side — now the sole identity
     anchor — is running off the pixel centre.

All three regression suites are green:

```
backend/scripts/verify_rally_bounds.py       exit=0
backend/scripts/verify_tracker_invariants.py exit=0
backend/scripts/verify_speed_filters.py      exit=0
```

## Verification artifacts

`backend/scripts/verify_tracker_invariants.py` holds the replays for §1, §2 and §4.
It exits non-zero today by design — those assertions are the acceptance criteria for
the fixes. `--document` prints observed behaviour without asserting.

Not committed: the §3 replay (finding closed — the behaviour it exercises cannot
occur under the fixed-sides invariant) and the §8 replay (unreachable behind the
upstream `[:2]` truncation). Both are reproducible from the snippets above.

If the Opportunity above is taken, §4's assertion should be **replaced** rather than
made to pass — a hard side constraint removes `_detect_and_correct_swap` entirely,
so the test that guards its counter logic would no longer have a subject. Add a test
asserting a top-side skeleton can never be assigned the bottom player instead.

Still true from the companion audit: no saved `results.json` or skeleton dump exists
in the repo, so §7 cannot be sized without capturing one.
