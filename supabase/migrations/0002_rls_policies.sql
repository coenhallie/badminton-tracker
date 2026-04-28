-- RLS: every table is owner-scoped via auth.uid()
-- See docs/plans/2026-04-28-supabase-migration-design.md §4.3.

alter table public.videos          enable row level security;
alter table public.processing_logs enable row level security;
alter table public.rally_clips     enable row level security;

-- videos
create policy "videos_owner_select" on public.videos
  for select using (owner_id = auth.uid());
create policy "videos_owner_insert" on public.videos
  for insert with check (owner_id = auth.uid());
create policy "videos_owner_update" on public.videos
  for update using (owner_id = auth.uid()) with check (owner_id = auth.uid());
create policy "videos_owner_delete" on public.videos
  for delete using (owner_id = auth.uid());

-- processing_logs
create policy "logs_owner_select" on public.processing_logs
  for select using (owner_id = auth.uid());
-- INSERT only via service role (Modal); no client-side insert policy.
-- DELETE: cascade from videos handles this.

-- rally_clips
create policy "clips_owner_select" on public.rally_clips
  for select using (owner_id = auth.uid());
-- INSERT only via service role (Modal).
-- DELETE: cascade from videos handles this.

-- Column-level UPDATE restrictions on public.videos.
-- The owner_id RLS policy above scopes WHICH rows can be updated; this
-- scopes WHICH COLUMNS the authenticated role can update. Modal uses the
-- service_role key, which has BYPASSRLS and is unaffected by these grants.
revoke update on public.videos from authenticated;
grant update (filename, manual_court_keypoints, player_labels)
  on public.videos to authenticated;
