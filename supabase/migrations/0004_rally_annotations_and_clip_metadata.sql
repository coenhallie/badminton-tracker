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
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);
create index rally_annotations_clip_idx
  on public.rally_annotations (clip_id, timestamp_seconds);
create index rally_annotations_owner_idx
  on public.rally_annotations (owner_id, created_at desc);

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
grant update (title) on public.rally_clips to authenticated;
