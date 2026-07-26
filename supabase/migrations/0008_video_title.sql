-- 0008: user-supplied match name, set once at upload.
-- See docs/plans/2026-07-25-match-title-design.md for rationale.
--
-- Distinct from `filename`, which is the uploaded file's real name and is read
-- by the web app's court-setup / progress screens. `title` is what the KMP
-- mobile app shows on each rally clip: Modal copies it into rally_clips.title,
-- which the app renders in place of its "Rally #{index}" fallback.
--
-- No `grant update (title) ... to authenticated` here, deliberately. 0002
-- revokes UPDATE on public.videos and re-grants it column by column (0007 does
-- the same for pipeline_variant), but that mechanism governs UPDATE only —
-- INSERT is unrestricted, and title is insert-only by design: there is no
-- rename-after-upload flow. If one is ever added, that grant becomes mandatory,
-- and a service-role smoke test will NOT catch its absence because service_role
-- has BYPASSRLS.
--
-- The check rejects the empty string as well as over-long names: VideoUpload
-- normalises a blank field to null (never ''), because the mobile app's
-- fallback to "Rally #{index}" keys on null. The 100-char bound is the DB
-- guard; the client caps typing at 60 (MAX_TITLE_LENGTH) for layout reasons.
--
-- Written to converge from either starting point, because this file originally
-- shipped as comments only — the statement below was missing — and the column
-- was added by hand in the SQL Editor to unblock the feature. So on the live
-- project the column already exists (untyped by any constraint), while a fresh
-- rebuild has neither. A bare `add column` would abort `db push` on the former;
-- `add column if not exists` alone would silently skip the CHECK there, leaving
-- the two environments permanently divergent. Hence: column and constraint
-- added independently, each guarded.

alter table public.videos
  add column if not exists title text;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.videos'::regclass
      and conname = 'videos_title_length_check'
  ) then
    -- Validates existing rows. If this errors, some row already violates it
    -- (an empty-string or >100-char title written directly to the DB) — fix the
    -- row rather than weakening the constraint.
    alter table public.videos
      add constraint videos_title_length_check
      check (title is null or length(title) between 1 and 100);
  end if;
end $$;
