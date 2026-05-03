# Rally Annotations + Clip Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the schema and backend changes that let a future mobile app
display rally clips with thumbnails and let the user attach time-pinned text
notes to each clip.

**Architecture:** One new migration adds a `rally_annotations` table plus
small additions to `rally_clips` (`thumbnail_storage_path`, `title`,
`annotation_count`). RLS mirrors the existing single-owner pattern; an
`annotation_count` trigger maintains the denorm. The Modal video processor
gets a small ffmpeg shell-out to write a poster JPEG per clip.

**Tech Stack:** PostgreSQL (Supabase), Supabase Storage, Python 3.9 + Modal,
ffmpeg (already on the Modal image), Supabase CLI (local dev).

**Design doc:** `docs/plans/2026-05-02-rally-annotations-design.md`

**Branch:** `feat/supabase-migration` (already checked out).

---

## Task 1: Create the migration file with schema additions

**Files:**
- Create: `supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`

**Step 1: Create the file with the schema portion only**

Write exactly this content:

```sql
-- 0004: rally_annotations table + clip metadata for mobile clip viewer.
-- See docs/plans/2026-05-02-rally-annotations-design.md for rationale.

-- New presentation/metadata columns on rally_clips.
-- thumbnail_storage_path: object key in the 'thumbnails' bucket; written by Modal.
-- title:                  user-editable display name; nullable.
-- annotation_count:       denorm maintained by trigger; default 0.
alter table public.rally_clips
  add column thumbnail_storage_path text,
  add column title                  text,
  add column annotation_count       int not null default 0;

-- New table: time-pinned text notes on rally clips, single-owner.
create table public.rally_annotations (
  id                 uuid primary key default gen_random_uuid(),
  clip_id            uuid not null references public.rally_clips(id) on delete cascade,
  owner_id           uuid not null references auth.users(id) on delete cascade,
  timestamp_seconds  real not null check (timestamp_seconds >= 0),
  body               text not null check (length(body) between 1 and 2000),
  created_at         timestamptz not null default now()
);
create index rally_annotations_clip_idx
  on public.rally_annotations (clip_id, timestamp_seconds);
create index rally_annotations_owner_idx
  on public.rally_annotations (owner_id, created_at desc);
```

**Step 2: Verify the file was written correctly**

Run: `wc -l supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
Expected: 24 lines (give or take whitespace).

Run: `head -5 supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
Expected: comment header matches.

**Step 3: Commit**

```bash
git add supabase/migrations/0004_rally_annotations_and_clip_metadata.sql
git commit -m "feat(db): add rally_annotations table and clip metadata columns"
```

---

## Task 2: Add RLS policies and grant to the migration

**Files:**
- Modify: `supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
  (append at end of file)

**Step 1: Append the RLS section**

Append exactly this to the end of the file:

```sql

-- RLS: single-owner, mirrors patterns in 0002_rls_policies.sql.
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

-- Allow authenticated users to update only the user-editable rally_clips column.
-- thumbnail_storage_path and annotation_count remain service-role-only.
-- Revoke first, then grant on specific columns — mirrors the pattern in 0002.
revoke update on public.rally_clips from authenticated;
grant update (title) on public.rally_clips to authenticated;
```

**Step 2: Verify with `tail`**

Run: `tail -20 supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
Expected: the file ends with `grant update (title) on public.rally_clips to authenticated;`
(the `tail -20` output will also include the preceding `revoke update on public.rally_clips
from authenticated;` line).

**Step 3: Commit**

```bash
git add supabase/migrations/0004_rally_annotations_and_clip_metadata.sql
git commit -m "feat(db): add RLS policies and grant for rally_annotations"
```

---

## Task 3: Add the annotation_count trigger to the migration

**Files:**
- Modify: `supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
  (append)

**Step 1: Append the trigger section**

Append exactly this:

```sql

-- Maintain rally_clips.annotation_count via trigger.
-- security definer is required because annotation_count is not granted to
-- the authenticated role.
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

**Step 2: Verify the migration file syntax is sound**

Run: `grep -c "create policy" supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
Expected: `4`.

Run: `grep -c "create trigger" supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`
Expected: `1`.

**Step 3: Commit**

```bash
git add supabase/migrations/0004_rally_annotations_and_clip_metadata.sql
git commit -m "feat(db): add annotation_count trigger for rally_clips"
```

---

## Task 4: Apply migration to a local Supabase instance

**Files:** none modified; verifying behavior of the migration file from
Tasks 1–3.

**Step 1: Start a local Supabase stack**

Run: `supabase start`
Expected: prints `API URL`, `DB URL`, `anon key`, etc. May take 60–90s on
the first run while Docker images pull.

If Docker isn't running, start Docker Desktop first.

**Step 2: Apply migrations to the local DB**

Run: `supabase db reset`
Expected: drops the local DB, re-applies all four migration files in order,
prints `Finished supabase db reset.`

This is the cleanest way to confirm the new migration applies on top of
0001/0002/0003 in order.

**Step 3: Inspect the new schema**

Run: `supabase db dump --local --schema public --data-only=false | grep -A 30 "CREATE TABLE.*rally_annotations"`
Expected: shows the table definition with all the columns from Task 1.

Alternative if `db dump` is awkward — connect with psql:

```bash
psql "$(supabase status --output env | grep '^DB_URL=' | cut -d= -f2- | tr -d '\"')" -c '\d public.rally_annotations'
psql "$(supabase status --output env | grep '^DB_URL=' | cut -d= -f2- | tr -d '\"')" -c '\d public.rally_clips'
```

Expected for `rally_annotations`: 7 columns including `timestamp_seconds`,
`body`, indexes `rally_annotations_clip_idx` and `rally_annotations_owner_idx`.

Expected for `rally_clips`: original columns plus
`thumbnail_storage_path`, `title`, `annotation_count`.

**Step 4: No commit needed**

This task verifies behavior; nothing changed in tracked files.

---

## Task 5: Write a SQL smoke test for RLS and trigger

**Files:**
- Create: `supabase/tests/0004_rally_annotations.sql`

**Step 1: Create the test directory and file**

```bash
mkdir -p supabase/tests
```

Write `supabase/tests/0004_rally_annotations.sql` with exactly this content:

```sql
-- Smoke test for migration 0004. Run against a local Supabase DB:
--   psql "$DB_URL" -f supabase/tests/0004_rally_annotations.sql
-- Exits non-zero on the first failed assertion.
-- Cleans up after itself.

\set ON_ERROR_STOP on
begin;

-- Two test users via auth.users.
insert into auth.users (id, instance_id, aud, role, email, encrypted_password,
                        email_confirmed_at, created_at, updated_at)
values
  ('11111111-1111-1111-1111-111111111111'::uuid,
   '00000000-0000-0000-0000-000000000000'::uuid,
   'authenticated', 'authenticated', 'a@test.local', '', now(), now(), now()),
  ('22222222-2222-2222-2222-222222222222'::uuid,
   '00000000-0000-0000-0000-000000000000'::uuid,
   'authenticated', 'authenticated', 'b@test.local', '', now(), now(), now());

-- Owner A's video and clip (service-role context, RLS bypassed).
insert into public.videos (id, owner_id, filename, size, storage_path, status)
values ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'::uuid,
        '11111111-1111-1111-1111-111111111111'::uuid,
        'a.mp4', 1, 'a/a.mp4', 'completed');

insert into public.rally_clips
  (id, video_id, owner_id, rally_index, start_timestamp, end_timestamp,
   duration_seconds, clip_storage_path)
values ('cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid,
        'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'::uuid,
        '11111111-1111-1111-1111-111111111111'::uuid,
        0, 0.0, 5.0, 5.0, 'a/a/rally_0.mp4');

-- New columns default correctly.
do $$
declare c int;
begin
  select annotation_count into c from public.rally_clips
    where id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
  if c <> 0 then raise exception 'expected annotation_count 0, got %', c; end if;
end $$;

-- Insert two annotations as user A. Set the role + JWT claims so RLS sees A.
set local role authenticated;
set local "request.jwt.claims" to '{"sub":"11111111-1111-1111-1111-111111111111"}';

insert into public.rally_annotations (clip_id, owner_id, timestamp_seconds, body)
values ('cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid,
        '11111111-1111-1111-1111-111111111111'::uuid, 1.0, 'first');
insert into public.rally_annotations (clip_id, owner_id, timestamp_seconds, body)
values ('cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid,
        '11111111-1111-1111-1111-111111111111'::uuid, 2.5, 'second');

-- Trigger should have bumped annotation_count to 2.
do $$
declare c int;
begin
  reset role;
  select annotation_count into c from public.rally_clips
    where id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
  if c <> 2 then raise exception 'expected annotation_count 2, got %', c; end if;
end $$;

-- A can read its own annotations (should be 2).
set local role authenticated;
set local "request.jwt.claims" to '{"sub":"11111111-1111-1111-1111-111111111111"}';
do $$
declare n int;
begin
  select count(*) into n from public.rally_annotations
    where clip_id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
  if n <> 2 then raise exception 'expected A to see 2 rows, got %', n; end if;
end $$;

-- B cannot see A's annotations (RLS isolation).
set local "request.jwt.claims" to '{"sub":"22222222-2222-2222-2222-222222222222"}';
do $$
declare n int;
begin
  select count(*) into n from public.rally_annotations
    where clip_id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
  if n <> 0 then raise exception 'expected B to see 0 rows, got %', n; end if;
end $$;

-- Cascade: deleting the clip removes its annotations and (by extension) we
-- never touch annotation_count again (clip is gone).
reset role;
delete from public.rally_clips where id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
do $$
declare n int;
begin
  select count(*) into n from public.rally_annotations;
  if n <> 0 then raise exception 'expected cascade to remove annotations, got %', n; end if;
end $$;

rollback;

\echo 'OK: migration 0004 smoke test passed'
```

**Step 2: Run the smoke test against the local DB**

Get the DB URL once:

```bash
DB_URL="$(supabase status --output env | grep '^DB_URL=' | cut -d= -f2- | tr -d '\"')"
echo "$DB_URL"
```

Then run:

```bash
psql "$DB_URL" -v ON_ERROR_STOP=1 -f supabase/tests/0004_rally_annotations.sql
```

Expected last line: `OK: migration 0004 smoke test passed`.
Any `raise exception` along the way will print the message and exit non-zero.

**Step 3: Commit**

```bash
git add supabase/tests/0004_rally_annotations.sql
git commit -m "test(db): smoke test for migration 0004 RLS and trigger"
```

---

## Task 6: Update the Modal processor to generate thumbnails

**Files:**
- Modify: `backend/modal_supabase_processor.py` — extend `cut_and_upload_rally_clips`
  (function starts around line 43; the upsert block we're inserting near is
  around lines 107–123).

**Step 1: Read the current function body**

Run: `sed -n '43,145p' backend/modal_supabase_processor.py`
to refresh on the current control flow before editing.

**Step 2: Insert thumbnail generation between clip upload and rally_clips upsert**

Locate this existing block (around line 107):

```python
        storage_path = f"{owner_id}/{video_id}/rally_{rally_id}.mp4"
        try:
            with open(clip_local, "rb") as f:
                sb.storage.from_("clips").upload(
                    path=storage_path,
                    file=f.read(),
                    file_options={"content-type": "video/mp4", "upsert": "true"},
                )
            sb.table("rally_clips").upsert({
                "video_id":          video_id,
                "owner_id":          owner_id,
                "rally_index":       rally_id,
                "start_timestamp":   rally["start_timestamp"],
                "end_timestamp":     rally["end_timestamp"],
                "duration_seconds":  rally["duration_seconds"],
                "clip_storage_path": storage_path,
            }, on_conflict="video_id,rally_index").execute()
```

Replace it with:

```python
        storage_path = f"{owner_id}/{video_id}/rally_{rally_id}.mp4"
        thumb_storage_path = None
        try:
            with open(clip_local, "rb") as f:
                sb.storage.from_("clips").upload(
                    path=storage_path,
                    file=f.read(),
                    file_options={"content-type": "video/mp4", "upsert": "true"},
                )

            # Best-effort thumbnail. A failure here must not block the clip insert.
            thumb_local = f"/cache/{video_id}_rally_{rally_id}.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", "0.5",
                        "-i", clip_local,
                        "-vframes", "1",
                        "-vf", "scale=480:-1",
                        "-q:v", "3",
                        thumb_local,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                candidate = f"{owner_id}/{video_id}/rally_{rally_id}.jpg"
                with open(thumb_local, "rb") as f:
                    sb.storage.from_("thumbnails").upload(
                        path=candidate,
                        file=f.read(),
                        file_options={"content-type": "image/jpeg", "upsert": "true"},
                    )
                thumb_storage_path = candidate
            except Exception as e:
                stderr = (
                    e.stderr.decode(errors="replace")
                    if hasattr(e, "stderr") and e.stderr
                    else str(e)
                )
                try:
                    sb.table("processing_logs").insert({
                        "video_id": video_id,
                        "owner_id": owner_id,
                        "message": f"thumbnail generation failed for rally {rally_id}: {stderr[:200]}",
                        "level": "warning",
                        "category": "processing",
                    }).execute()
                except Exception:
                    pass
            finally:
                try:
                    if os.path.exists(thumb_local):
                        os.remove(thumb_local)
                except OSError:
                    pass

            sb.table("rally_clips").upsert({
                "video_id":               video_id,
                "owner_id":               owner_id,
                "rally_index":            rally_id,
                "start_timestamp":        rally["start_timestamp"],
                "end_timestamp":          rally["end_timestamp"],
                "duration_seconds":       rally["duration_seconds"],
                "clip_storage_path":      storage_path,
                "thumbnail_storage_path": thumb_storage_path,
            }, on_conflict="video_id,rally_index").execute()
```

The diff:
- A nested `try/except/finally` for the thumbnail step that logs a warning
  and continues on failure.
- `thumb_storage_path` is `None` if the thumbnail step failed; the column
  is nullable, so the upsert still succeeds.
- Local thumbnail file is always cleaned up.

**Step 3: Lint check (syntactic only)**

Run: `python -m py_compile backend/modal_supabase_processor.py`
Expected: no output (success).

**Step 4: Commit**

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(processor): generate per-rally thumbnails into thumbnails bucket"
```

---

## Task 7: Apply migration to the remote Supabase project

**Files:** none modified.

**Step 1: Confirm we're linked to the right Supabase project**

Run: `supabase projects list`
Expected: lists projects; the linked one is marked.

If not linked: `supabase link --project-ref <ref>` (the ref is in
`supabase/.temp/project-ref`).

**Step 2: Push migration to remote**

Run: `supabase db push`
Expected: prints something like
`Applying migration 20260502_rally_annotations_and_clip_metadata.sql...`
(or similar, depending on naming) and ends successfully.

If it complains about already-applied migrations, pass `--include-all` and
re-read the output before running anything destructive.

**Step 3: Verify on the remote DB via the dashboard**

Open the Supabase dashboard → Table Editor → `public.rally_annotations`.
Expected: the table exists with the columns we defined.

Also: Table Editor → `public.rally_clips` → confirm the three new columns
appear at the right of the existing list.

**Step 4: No commit needed**

---

## Task 8: Deploy the updated Modal processor and end-to-end smoke test

**Files:** none modified.

**Step 1: Deploy**

Run: `modal deploy backend/modal_supabase_processor.py`
Expected: prints the deployment URL; exits successfully.

**Step 2: Process a short test video end-to-end via the existing frontend flow**

Use the running Vue dev server (`npm run dev` if not already up) and upload
a short clip (≤30s) with at least 2 rallies. Wait for processing to finish.

**Step 3: Verify thumbnails landed**

In the Supabase dashboard → Storage → `thumbnails` bucket, navigate to
`<your-uid>/<video-id>/`. Expected: one `rally_<n>.jpg` per rally.

In Table Editor → `rally_clips`, filter by your test video. Expected:
`thumbnail_storage_path` is populated for each row.

**Step 4: Verify annotation insert via the dashboard SQL editor**

Run, in the Supabase dashboard SQL Editor (signed in as your admin user):

```sql
-- pick any clip from your test video
select id, annotation_count from public.rally_clips
  where video_id = '<video-id>' order by rally_index limit 1;

insert into public.rally_annotations (clip_id, owner_id, timestamp_seconds, body)
  values ('<clip-id>', auth.uid(), 1.5, 'smoke test note');

select annotation_count from public.rally_clips where id = '<clip-id>';
```

Expected: starts at 0, ends at 1.

Then clean up:

```sql
delete from public.rally_annotations where body = 'smoke test note';
select annotation_count from public.rally_clips where id = '<clip-id>';
```

Expected: ends at 0.

**Step 5: No commit needed**

If everything passes, this branch is ready to merge.

---

## Out of scope (do NOT do as part of this plan)

- Mobile app code itself.
- Backfilling thumbnails for already-processed clips.
- A web UI for adding annotations from the existing Vue frontend
  (that's a follow-up plan).
- Any sharing / multi-user / commenting-as-coach feature.
