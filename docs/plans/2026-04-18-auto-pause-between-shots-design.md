# Auto-Pause Between Shots Design

**Date:** 2026-04-18
**Status:** Approved
**Goal:** When enabled, pause the video for a short configurable duration after each detected shot and overlay a summary card showing the movement segment and peak body mechanics leading up to that shot. The overlay updates on the next shot.

## Context

Shot detection already exists in `src/components/ShotSpeedList.vue` via `detectShots(frames)` — a cascade using shuttle trajectory, pose classification, and movement pattern analysis. It produces `ShotEvent`s and `ShotMovementSegment`s (per-segment max/avg speed km/h, distance covered, duration, speed profile).

Per-frame body-angle math (leg stretch, knee, hip, torso lean, elbow, shoulder) already lives in `src/components/VideoPlayer.vue` for the angle-overlay feature.

A rally-level auto-pause already exists in `src/App.vue`: 3-second countdown, wired via `watch(currentVideoTime, …)`, pauses the video and auto-skips to the next rally.

This feature adds a per-shot version with its own toggle, its own (configurable) duration, and a summary card. It deliberately reuses the existing shot pipeline and rally-pause plumbing patterns rather than inventing new ones.

## View Geometry / UI Placement

**Overlay:** single compact card (~320×140px), bottom-center of `.video-wrapper`, semi-transparent dark background with thin accent border, `z-index: 26` (above controls and pose overlay). Visible only while `shotPauseCountdown > 0`.

Rejected alternatives:
- **Per-player split** — doubles UI surface and invites complexity around court-side placement.
- **Full-width HUD bar** — obstructs a horizontal strip of court content.

**Controls:** toggle + duration picker live next to the existing "Pause Between Rallies" checkbox in `RallyTimeline.vue`, so both pause controls sit together.

**View modes:** works in both Video and Court modes — the overlay is a DOM layer over `.video-wrapper` and doesn't depend on the video pixels.

## Metric Set (final, not configurable in v1)

From the movement segment leading into the shot:
- Player who responded (moving player)
- Max speed (km/h)
- Avg speed (km/h)
- Distance covered (m)
- Duration (s)
- Peak leg stretch (°)
- Peak knee flex (°)

Rejected alternatives:
- **Movement stats only** — feels sparse once the video is paused.
- **Shot-at-impact data for the hitter** — different feature (strike mechanics); strike classification is less reliable than the movement segment we already compute.

## Pause Duration

Configurable per-user: `1s | 1.5s | 2s | 3s`. Default `1.5s`.

Rejected:
- **Fixed 1s** — too tight to read seven stats comfortably.
- **Pause-until-user-resume** — breaks watch-at-pace flow; future feature if demand arises.

## Interaction with Rally-Pause

Independent toggles, but with one guard: when both are on, shot-pause suppresses the pause that would fire on the *last* shot of each rally, letting rally-pause handle that moment. This avoids a double-pause at rally boundaries.

Rejected:
- **Both fire independently** — jarring double-pause at rally ends.
- **Collapse into one "Auto-pause" mode dropdown** — nicer long-term but requires reworking the existing toggle; deferred.

## Data Flow

```
analysisResult (loaded from Convex/Modal)
   └─ skeleton_data[]
        └─ useShotSegments composable (hoisted from ShotSpeedList.vue)
             ├─ detectShots()          → ShotEvent[]
             ├─ buildSegments()        → ShotMovementSegment[]
             └─ aggregateBodyAngles()  → extended ShotMovementSegment[] with peaks

App.vue
   ├─ pauseBetweenShots + shotPauseDurationSec state
   ├─ watch(currentVideoTime) → fire pause on shot crossing
   ├─ currentShotSegment ref → drives overlay content
   └─ passes viewMode-agnostic overlay prop to VideoPlayer

VideoPlayer
   └─ ShotSummaryOverlay  (new component; position: absolute in .video-wrapper)
```

No backend, Convex, or Modal changes.

## Components

### New — `src/composables/useShotSegments.ts`

Hoisted from `ShotSpeedList.vue`:
- `ShotEvent`, `ShotMovementSegment` types (exported).
- `detectShots(frames)` — unchanged logic.
- `buildSegments(shots, frames)` — pairs consecutive shots into movement segments for the responder (unchanged logic).
- New: `aggregateBodyAngles(segment, frames)` — walks `[startFrame, endFrame]` for the `movingPlayerId`, returns `{ peakLegStretch, peakKneeFlex, peakTorsoLean }`. Uses pure helpers extracted to `src/utils/bodyAngles.ts`.

Composable returns `segments: ComputedRef<ShotMovementSegment[]>` keyed on `skeletonData` reactivity.

### New — `src/utils/bodyAngles.ts`

Pure functions extracted from `VideoPlayer.vue`:
- `legStretchAngle(keypoints): number | null`
- `kneeFlexAngle(keypoints): number | null`
- `torsoLeanAngle(keypoints): number | null`

Both `VideoPlayer.vue` and `useShotSegments.ts` consume these.

### New — `src/components/ShotSummaryOverlay.vue`

Props:
- `segment: ShotMovementSegment` — the just-completed movement segment (with angle peaks).
- `countdownSec: number` — drives the shrinking timer badge.

Renders the card described under "View Geometry / UI Placement". No internal state; purely presentational.

### Modified — `src/App.vue`

Add:
- State: `pauseBetweenShots`, `shotPauseDurationSec`, `shotPauseCountdown`, `currentShotSegment`, `lastTriggeredShotTime`.
- Watch on `currentVideoTime` with the same shape as the existing rally-pause watcher:
  1. Reset tracking if user seeks backward past `lastTriggeredShotTime - 0.5`.
  2. Bail if feature off, or already in a pause.
  3. Find shot `s` where `time ∈ [s.timestamp, s.timestamp + 0.5)` and `s.timestamp > lastTriggeredShotTime`.
  4. Skip if `s` is the first shot of its rally (no preceding segment).
  5. Skip if `s` is the last shot of its rally AND `pauseBetweenRallies` is on.
  6. Otherwise: set `currentShotSegment`, pause video, run countdown, resume on completion.
- Pass `currentShotSegment` + `shotPauseCountdown` into `VideoPlayer`.

### Modified — `src/components/VideoPlayer.vue`

- Accept `shotSummarySegment?: ShotMovementSegment | null`, `shotSummaryCountdownSec?: number` props.
- Mount `<ShotSummaryOverlay>` conditionally inside `.video-wrapper` when the segment is present and countdown > 0.

### Modified — `src/components/RallyTimeline.vue`

- Accept `pauseBetweenShots` (v-model) and `shotPauseDurationSec` (v-model) props.
- Add a new row under the existing rally-pause checkbox: "Pause Between Shots" + duration select.

### Modified — `src/components/ShotSpeedList.vue`

- Replace its internal `detectShots` / segment-building code with calls to `useShotSegments`. Zero behavior change for the existing Shot Speed List view.

## User Interaction

- **User hits play during countdown** → cancel countdown, clear overlay, video resumes. Same handler as rally-pause.
- **User seeks backward past the triggered shot** → reset `lastTriggeredShotTime` so re-crossing can re-trigger. Same handler as rally-pause.
- **User seeks forward past multiple shots while paused** → next trigger is the first shot after the new position; no queued pauses.

## Performance

- Shot detection + segment + angle aggregation run once after `analysisResult` loads and are cached via `computed`. Per-segment angle aggregation walks ~60 frames (2s @ 30fps) — negligible.
- Countdown `setInterval` only runs during pauses.
- No per-playback-frame cost beyond what already exists.

## Edge Cases

| Case | Behavior |
|------|----------|
| First shot of a rally | No pause (no preceding segment) |
| Last shot of rally with `pauseBetweenRallies` on | No pause (rally-pause handles it) |
| Last shot of rally with `pauseBetweenRallies` off | Pause normally |
| Shots closer than `shotPauseDurationSec` | `shotPauseCountdown > 0` guard prevents stacking; later shot is skipped for pause purposes but still counted in the segment list |
| Body angle missing for a frame | Skip that frame in the max; if zero valid, display "—" for that field |
| Zero detected shots in analysis (rare) | Overlay never appears; toggle is functionally a no-op |
| Court view active | Same overlay layer, same z-index, same behavior |

## Testing

- Manual: enable toggle, play a rally, verify pause fires on each mid-rally shot with plausible stats (moving player's direction of travel ↔ which half of court they're in). Toggle different durations, verify countdown matches.
- Manual: enable both toggles, verify no double-pause at rally ends.
- Manual: verify overlay renders identically in Video vs Court mode.
- Manual: verify seek-backward re-triggers pauses; seek-forward skips queued pauses.
- Automated (type-check only — no test framework in repo): `npm run type-check` + `npm run build` pass with no new errors.

## Non-goals (explicit v1 scope)

- No per-player split overlay.
- No configurable metric selection.
- No shot-type / strike-mechanics display.
- No export of per-shot aggregates.
- No keyboard shortcut for toggle.
- No collapsed "Auto-pause mode" dropdown that unifies shot- and rally-pause toggles.

Each can be added in a follow-up once v1 is in use.
