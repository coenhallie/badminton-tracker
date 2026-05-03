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
-- reset role at top level: SET/RESET inside a DO block doesn't propagate.
reset role;
do $$
declare c int;
begin
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
  select count(*) into n from public.rally_annotations
    where clip_id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'::uuid;
  if n <> 0 then raise exception 'expected cascade to remove annotations, got %', n; end if;
end $$;

rollback;

\echo 'OK: migration 0004 smoke test passed'
