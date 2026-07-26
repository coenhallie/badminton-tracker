-- 0009: put the realtime dependency in the repo.
--
-- The progress UI is realtime-only. `AnalysisProgress.vue` reads the video row
-- through `useReactiveRow('videos', …)` and the log stream through
-- `useReactiveList('processing_logs', …)`; both do one SELECT and then rely
-- entirely on `postgres_changes`. Postgres only emits those events for tables
-- in the `supabase_realtime` publication.
--
-- Nothing in migrations 0001-0008 adds them, and SETUP.md never mentioned it:
-- the live project works only because the membership was toggled in the
-- dashboard at some point. That made the repo unable to reproduce a working
-- environment — rebuild from these migrations and the progress screen sits at
-- "Starting analysis..." with an empty log pane, while the row's status and
-- progress advance in the DB. No error anywhere.
--
-- Written idempotently because it has to run against BOTH a fresh project
-- (membership absent) and the existing one (membership already present, where a
-- bare `alter publication ... add table` raises duplicate_object and would fail
-- the migration).
--
-- Only the two tables the web app actually subscribes to are added.
-- `rally_clips` is deliberately NOT included: RallyReview polls
-- `videos.status` every 8s and refetches, and the mobile app fetches on
-- appearance — neither holds a postgres_changes subscription.

do $$
begin
  -- Supabase provisions `supabase_realtime` for every hosted project. Create it
  -- empty if absent (self-hosted / bare Postgres) so the adds below succeed.
  if not exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    create publication supabase_realtime;
  end if;

  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'videos'
  ) then
    alter publication supabase_realtime add table public.videos;
  end if;

  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'processing_logs'
  ) then
    alter publication supabase_realtime add table public.processing_logs;
  end if;
end $$;

-- Replica identity is left at the default (primary key), deliberately.
--
-- Realtime evaluates the RLS policies from 0002 against the changed record
-- before delivering it. INSERT and UPDATE carry every column, so `owner_id =
-- auth.uid()` evaluates fine — those are the events the progress UI needs.
-- DELETE carries only the primary key under the default replica identity, so
-- the policy cannot be evaluated and DELETE events are not delivered.
--
-- That is an accepted trade: `REPLICA IDENTITY FULL` on these two tables would
-- write every column of every row change to the WAL — expensive on
-- `processing_logs`, which the worker appends to throughout a run — to buy only
-- the DELETE handlers in useReactiveRow/useReactiveList. Those handlers matter
-- solely if a video is deleted elsewhere (e.g. from the phone) while a web tab
-- has it open; the tab shows a stale row until reload. Not worth the WAL.
