# Match Title — Design

Date: 2026-07-25
Status: Implemented. §3 and §7 were corrected after testing on device — see the
addendum at the end before trusting the mobile claims in this document.

## 1. Problem

Rally clips cut by Modal land in Supabase with `rally_clips.title = NULL`. The
KMP mobile app renders each clip row as "title, or `Rally #{rallyIndex}` if
null" (see `docs/plans/2026-05-04-kmp-rally-clips-mobile-design.md` §5.2), with
the source video's filename and date as the subtitle. Because the backend never
writes `title`, every clip in every match reads the same generic label, so
matches are indistinguishable on the phone.

The user wants to type a match name at upload time and see that name on the
clips in the mobile app.

## 2. Goal

A single optional "Match name" field on the upload screen. Whatever is typed
there becomes the `title` of every rally clip cut from that video, so the phone
app's clip list identifies the match.

## 3. Approach

Name the *video* at upload; the clip cutter stamps that name onto every
`rally_clips` row it writes.

The mobile app already renders `rally_clips.title` when non-null, so no change
is required in the mobile repo. This repo owns the entire fix.

**Alternatives rejected:**

- *Render `videos.filename` in the mobile clip list.* The app lives in a
  separate repo with a different toolchain; the change could not ship from here,
  and filenames from a phone camera ("IMG_4821.mov") are not useful names.
- *Encode the name in the clip's storage path.* Storage keys are a poor place
  for user text — it needs escaping, and a later rename would orphan objects.

## 4. Decisions

| Question | Decision |
|---|---|
| Clip row label | Match name only — no rally number appended. All clips from one match share an identical title. |
| Blank match name | Allowed. `title` stays `NULL`, so the app falls back to `Rally #N` exactly as today. Nothing regresses. |
| Renaming after upload | Not supported. Set once, at upload. |
| Confirmation in web UI | Yes — echo the name on the court-setup and analysis-progress screens. |

The "match name only" choice was made with the identical-rows tradeoff shown
explicitly; it is deliberate, not an oversight.

## 5. Changes

### 5.1 Schema — `supabase/migrations/0008_video_title.sql`

```sql
alter table public.videos
  add column title text
  check (title is null or length(title) between 1 and 100);
```

**No `grant update (title)`.** `0002_rls_policies.sql` revokes UPDATE on
`public.videos` from `authenticated` and re-grants it per column; columns added
later are not covered automatically (this is why `0007` exists for
`pipeline_variant`). That mechanism governs UPDATE only — INSERT is
unrestricted, and `title` is insert-only under this design. Every existing
client write path uses an explicit narrow payload
(`src/services/api.ts:245`, the `videos` updates in `src/App.vue`), so none of
them will attempt to write `title`.

If rename-after-upload is ever added, `grant update (title) on public.videos to
authenticated;` becomes mandatory. A service-role smoke test cannot catch its
absence, because service_role has BYPASSRLS.

### 5.2 Upload form — `src/components/VideoUpload.vue`

Add `const matchTitle = ref('')`.

A text input inside the `.file-preview` block, placed above `.pipeline-select`:

- Label: "Match name"
- Placeholder: `e.g. Thu League vs Marco`
- `maxlength="100"`, `:disabled="isUploading"`
- Optional — no validation blocking upload

In `uploadAndCreate`, compute `const title = matchTitle.value.trim() || null`
and add `title` to the existing `.insert({...})` at line 99. Empty input must
persist as `null`, never `''`, so the mobile app's null-check fallback works.

`removeFile()` leaves `matchTitle` alone — the user may have typed the name
before swapping to the correct file.

### 5.3 Type — `src/types/analysis.ts`

Add `title: string | null` to `UploadResponse` (line 283) and include it in the
`emit('uploaded', ...)` payload in `startUpload`.

### 5.4 Propagation — `backend/modal_supabase_processor.py`

In `cut_and_upload_rally_clips` (line 132), fetch the title once near the top,
next to the existing `sb = supabase_client()`:

```python
try:
    row = sb.table("videos").select("title").eq("id", video_id).single().execute()
    match_title = (row.data or {}).get("title")
except Exception:
    match_title = None
```

Fetching inside the function rather than adding a parameter avoids touching
both call sites (lines ~4243 and ~4769) and their argument plumbing. A failed
fetch degrades to `None` and must never crash the cut — same posture as the
existing best-effort thumbnail block.

Then, per clip, **after** the existing `rally_clips` upsert (line 272):

```python
if match_title:
    sb.table("rally_clips").update({"title": match_title}) \
      .eq("video_id", video_id).eq("rally_index", rally_id) \
      .is_("title", "null").execute()
```

**Set-only-if-null is deliberate.** `0004_rally_annotations_and_clip_metadata.sql`
grants `update (title)` on `rally_clips` to `authenticated`, so a clip title can
already be edited from the phone. Including `title` in the upsert payload would
silently overwrite that edit on every re-cut, and this repo has both an in-place
re-run and an A/B duplicate flow that re-cut clips.

This `update` sits inside the same `try` that wraps the upload and upsert, so a
failure is logged to `processing_logs` as a warning and the pipeline continues.

### 5.5 A/B duplicate — `supabase/functions/duplicate-video/index.ts`

Add `title: video.title,` alongside the existing `filename: video.filename,` in
the `insert()` at line 48, so a `gb_fusion` sibling of a named match keeps the
name.

### 5.6 Confirmation in the web UI

Purely so the user can see the name took.

- `src/components/CourtSetup.vue`: add `title?: string | null` to props
  (line 9); render it in its own element above the existing
  `<p class="subtitle">{{ props.filename }}</p>` at line 376, guarded by `v-if`
  so nothing renders when the title is null. The filename line stays as-is.
- `src/components/AnalysisProgress.vue`: add `title?: string | null` to props
  (line 11); render it in its own element next to `<span class="filename">` at
  line 376, guarded by `v-if`. The filename stays as-is.
- `src/App.vue`: pass `:title="uploadedVideo.title"` at the three existing
  `:filename` call sites (lines 1402, 1413, 1439).
- `src/App.vue` line 1084: add `title` to the deep-link hydration
  `.select('id, filename, size, status, manual_court_keypoints')` and map it
  into `uploadedVideo.value`, so opening a run via `?videoId=` also shows the
  name. `hydrateFromExistingVideo()` only reads `status` and needs no change.

## 6. Verification

The project has no test framework; verification is by script, type-check, and
manual smoke (see the `no-test-framework` memory note).

1. `npm run type-check` passes.
2. Apply `0008` to the local/staging database; confirm the column and its check
   constraint exist.
3. Upload a video **with** a match name as a normal authenticated user — not
   service_role — and confirm the `videos` row carries `title`. Using an anon
   client here is what would surface a missing grant if the design ever changes
   to allow updates.
4. Upload a video **without** a match name; confirm `title IS NULL` (not `''`).
5. After a cut completes, confirm every `rally_clips` row for the named video
   has `title` set, and every row for the unnamed video has `title IS NULL`.
6. Manually set one clip's `title` to `'user edit'`, re-run the cut, and confirm
   that row still reads `'user edit'` while the others carry the match name.
7. Trigger the A/B duplicate on a named video; confirm the new `videos` row
   carries the same `title`.
8. Open the phone app and confirm the clip list shows the match name.

## 7. Out of scope

- Renaming a match after upload, and any UI for it.
- Per-clip rally numbers in the title.
- Changing the mobile app's subtitle line (video filename + date) — separate
  repo, and unnecessary once the title line is populated.
- Naming the downloaded `.mp4` file itself; the issue reported was the in-app
  clip list, not files in Photos.

## 8. Addendum (2026-07-25, after on-device testing)

**§3's central claim was wrong.** It said no mobile change was needed because
the app renders `rally_clips.title` per
`docs/plans/2026-05-04-kmp-rally-clips-mobile-design.md` §5.2. That design doc
no longer describes the shipped app. The app was rebuilt around a **two-level
structure**:

1. A **match list**, where clips are grouped by `video_id` into a `MatchSummary`
   whose row label was hardcoded to `"Match · <date>"`. `MatchSummary` had no
   title field, so `rally_clips.title` was never read here.
2. A **rally list** inside each match, plus a clip detail screen — these *are*
   the surfaces that render `clip.title ?? "Rally #N"`.

Everything in §5 (schema, upload field, backend propagation, duplicate) is
correct and unchanged. The title does reach the phone. It simply had nothing
rendering it at the match level.

Two consequences, both fixed in `badminton-rally-mobile`:

- `MatchSummary` gained a `title`, derived from the **most common** non-null
  title among the match's clips — not the cover clip's, so retitling one clip
  in the app cannot relabel the whole match.
- Because §4 chose "match name only", every clip carries the same title, which
  would have made every rally row and clip-detail header read the match name
  instead of "Rally #1", "Rally #2". The apps now suppress a clip title that
  equals the match title and fall back to the rally number. `rally_index` is
  1-based (`backend/rally_detection_shot_gap.py:141` writes `i + 1`), so rows
  read "Rally #1" upward.

The §4 "match name only" decision still stands — it was about the match row,
which is exactly where the name now appears. Rally numbering inside a match was
a surface that decision never covered.

**Note:** `badminton-rally-mobile` has its own `supabase/migrations/` that also
alters `public.videos` (e.g. `20260720000000_analyze_status_reset_grant.sql`).
Two repos issue migrations against the same table with different naming schemes
and no shared ordering. Out of scope here, but it is a real hazard — the §5.1
reasoning about grants can be invalidated from the other repo without warning.
