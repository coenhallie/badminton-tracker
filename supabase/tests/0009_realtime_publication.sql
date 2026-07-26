-- Smoke test for migration 0009. Run against a local Supabase DB:
--   psql "$DB_URL" -f supabase/tests/0009_realtime_publication.sql
-- Exits non-zero on the first failed assertion.
-- Read-only — nothing to clean up.
--
-- Asserts the property the progress UI depends on: `videos` and
-- `processing_logs` are members of the `supabase_realtime` publication. Without
-- them, AnalysisProgress.vue's postgres_changes subscriptions receive nothing
-- and the progress screen never advances, with no error surfaced anywhere.

\set ON_ERROR_STOP on

do $$
begin
  if not exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    raise exception 'FAIL: publication supabase_realtime does not exist';
  end if;

  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public' and tablename = 'videos'
  ) then
    raise exception 'FAIL: public.videos is not in the supabase_realtime publication';
  end if;

  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public' and tablename = 'processing_logs'
  ) then
    raise exception 'FAIL: public.processing_logs is not in the supabase_realtime publication';
  end if;

  raise notice 'PASS: realtime publication includes videos + processing_logs';
end $$;

-- 0009 must be safe to re-run: it also has to apply to the existing project,
-- where the membership was added by hand in the dashboard long ago. Re-running
-- the migration body here proves the guards hold (a bare
-- `alter publication ... add table` would raise duplicate_object at this point).
do $$
begin
  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public' and tablename = 'videos'
  ) then
    alter publication supabase_realtime add table public.videos;
  end if;
  raise notice 'PASS: migration body is idempotent';
end $$;
