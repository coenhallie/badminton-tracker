# Rally clip annotations + mobile-friendly clip metadata

**Date:** 2026-05-02
**Status:** Design approved, ready for implementation plan
**Branch:** `feat/supabase-migration` (or follow-up branch)

## 1. Goal

Enable the user to view rally clips on a future mobile application and attach
time-pinned text notes ("YouTube-style timestamped comments") to each clip.

## 2. Scope

In scope:
- A new `rally_annotations` table for time-pinned text notes on clips.
- Small additions to `rally_clips` to support a richer mobile list view
  (`thumbnail_storage_path`, `title`, `annotation_count`).
- Thumbnail generation in the existing rally clip pipeline.
- RLS, grants, and a trigger to maintain `annotation_count`.

Explicitly out of scope (deferred):
- Sharing clips with other users (would need a `clip_shares` table).
- Frame-level drawings / shapes / overlays.
- Threaded replies on annotations.
- Backfill of thumbnails for already-processed clips.
- The mobile app itself — this design only covers the data layer it relies on.
- Realtime subscriptions.

## 3. Decisions captured during brainstorming

| Decision | Choice | Reason |
|---|---|---|
| Annotation type | Time-based text notes | Maps to the user's "YouTube-style comments" mental model. |
| Multi-user model | Single owner only | YAGNI — sharing can be added later via a `clip_shares` table without breaking existing rows. |
| Time anchor | Single moment (`timestamp_seconds`) | Simpler than ranges and matches typical coaching workflows. |
| Schema shape | Approach 2: annotations table + small `rally_clips` additions | Mobile UX needs poster thumbnails; the `thumbnails` bucket already exists. |

## 4. Schema changes

Single new migration `supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`.

```sql
-- Add user-editable + presentation fields to rally_clips
alter table public.rally_clips
  add column thumbnail_storage_path text,
  add column title                  text,
  add column annotation_count       int not null default 0;

-- New annotations table (single-owner, time-pinned text notes)
create table public.rally_annotations (
  id                 uuid primary key default gen_random_uuid(),
  clip_id            uuid not null references public.rally_clips(id) on delete cascade,
  owner_id           uuid not null references auth.users(id) on delete cascade,
  timestamp_seconds  real not null check (timestamp_seconds >= 0),
  body               text not null check (length(body) between 1 and 2000),
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);
create index rally_annotations_clip_idx
  on public.rally_annotations (clip_id, timestamp_seconds);
create index rally_annotations_owner_idx
  on public.rally_annotations (owner_id, created_at desc);
```

Notes:
- `clip_id` cascades on delete → deleting a clip deletes its annotations.
- `owner_id` is redundantly stored (could be derived via `clip_id` join) but kept
  for cheap RLS checks — same pattern as `rally_clips`.
- `length(body)` capped at 2000 chars; bump if longer notes are needed.

## 5. RLS and grants

```sql
alter table public.rally_annotations enable row level security;

create policy "annotations_owner_select" on public.rally_annotations
  for select using (owner_id = auth.uid());
create policy "annotations_owner_insert" on public.rally_annotations
  for insert with check (owner_id = auth.uid());
create policy "annotations_owner_update" on public.rally_annotations
  for update using (owner_id = auth.uid())
              with check (owner_id = auth.uid());
create policy "annotations_owner_delete" on public.rally_annotations
  for delete using (owner_id = auth.uid());

-- Allow authenticated users to update only the user-editable rally_clips columns.
grant update (title) on public.rally_clips to authenticated;
```

Mobile gets full CRUD on its own annotations via the standard Supabase client
(JS / Swift / Kotlin / Flutter) — no edge function required.
`thumbnail_storage_path` and `annotation_count` remain service-role-only;
the owner can update only `title` on `rally_clips`.

## 6. Annotation count trigger

```sql
create or replace function public.bump_annotation_count() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if (tg_op = 'INSERT') then
    update public.rally_clips set annotation_count = annotation_count + 1
      where id = new.clip_id;
  elsif (tg_op = 'DELETE') then
    update public.rally_clips set annotation_count = greatest(annotation_count - 1, 0)
      where id = old.clip_id;
  end if;
  return null;
end $$;

create trigger rally_annotations_count_trg
  after insert or delete on public.rally_annotations
  for each row execute function public.bump_annotation_count();
```

`security definer` is required because the trigger writes `annotation_count`,
which is not granted to the authenticated role.

## 7. Backend changes (thumbnail generation)

Extend `cut_and_upload_rally_clips` in `backend/modal_supabase_processor.py`.
After the clip mp4 upload succeeds, run a second `ffmpeg` call to extract one
frame near the start, upload to the `thumbnails` bucket, and include
`thumbnail_storage_path` in the `rally_clips` upsert.

```python
thumb_local = f"/cache/{video_id}_rally_{rally_id}.jpg"
subprocess.run([
    "ffmpeg", "-y",
    "-ss", "0.5",
    "-i", clip_local,
    "-vframes", "1",
    "-vf", "scale=480:-1",
    "-q:v", "3",
    thumb_local,
], check=True, capture_output=True, timeout=30)

thumb_path = f"{owner_id}/{video_id}/rally_{rally_id}.jpg"
with open(thumb_local, "rb") as f:
    sb.storage.from_("thumbnails").upload(
        path=thumb_path,
        file=f.read(),
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )

# In the rally_clips upsert payload, add:
#   "thumbnail_storage_path": thumb_path,
```

Failure handling: a thumbnail failure logs a warning and continues — clip
itself is still usable, just no poster image. Same pattern as the existing
per-rally error handling.

Cost: one extra ~50ms ffmpeg call + a small JPEG upload per clip. Negligible
vs. the rally re-encode.

## 8. Mobile access pattern

The mobile app uses the official Supabase client. With the user signed in via
Supabase Auth, the client sends a JWT and RLS handles authorization.

| Mobile action | Supabase call |
|---|---|
| List clips for a video | `from('rally_clips').select('*').eq('video_id', …).order('rally_index')` |
| Stream clip video | `storage.from('clips').createSignedUrl(clip.clip_storage_path, 3600)` |
| Show poster | `storage.from('thumbnails').createSignedUrl(clip.thumbnail_storage_path, 3600)` |
| List annotations | `from('rally_annotations').select('*').eq('clip_id', …).order('timestamp_seconds')` |
| Add annotation | `from('rally_annotations').insert({clip_id, owner_id, timestamp_seconds, body})` |
| Edit / delete | `update`/`delete` with `eq('id', …)` |

No new edge functions, no custom REST API.

## 9. Migration ordering and deploy

1. Apply the migration via `supabase db push` (same flow as existing migrations).
2. Deploy the updated processor with `modal deploy backend/modal_supabase_processor.py`.

The order matters only weakly: old processor code keeps writing existing
columns and the new column accepts `null`, so no rollout coordination is
required.

## 10. Testing checklist

1. Run the migration locally; confirm `\d rally_annotations` and the new
   columns on `\d rally_clips` look right.
2. Process one short test video end-to-end; verify thumbnails land in the
   `thumbnails` bucket and `thumbnail_storage_path` is populated on new rows.
3. From a signed-in JS client: insert an annotation; confirm the row appears
   and `annotation_count` on the parent clip went 0 → 1.
4. From a different signed-in user: select that clip's annotations → must
   return empty (RLS check).
5. Delete the annotation; confirm `annotation_count` returns to 0.
6. Delete the parent clip; confirm the annotation row is cascaded.
