-- Initial schema: videos, processing_logs, rally_clips
-- See docs/plans/2026-04-28-supabase-migration-design.md §4 for full rationale.

create table public.videos (
  id              uuid primary key default gen_random_uuid(),
  owner_id        uuid not null references auth.users(id) on delete cascade,
  filename        text not null,
  size            bigint not null,
  storage_path    text not null,

  status          text not null check (status in ('uploaded','processing','completed','failed')),
  progress        real,
  current_frame   int,
  total_frames    int,
  error           text,

  results_meta            jsonb,
  results_storage_path    text,
  processed_video_path    text,
  skeleton_data_path      text,

  manual_court_keypoints  jsonb,
  player_labels           jsonb,

  created_at              timestamptz not null default now(),
  processing_started_at   timestamptz,
  completed_at            timestamptz
);
create index videos_owner_created_idx on public.videos (owner_id, created_at desc);
create index videos_status_idx        on public.videos (status);

create table public.processing_logs (
  id          bigserial primary key,
  video_id    uuid not null references public.videos(id) on delete cascade,
  owner_id    uuid not null references auth.users(id) on delete cascade,
  message     text not null,
  level       text not null check (level    in ('info','success','warning','error','debug')),
  category    text not null check (category in ('processing','detection','model','court','modal')),
  timestamp   timestamptz not null default now()
);
create index processing_logs_video_ts_idx on public.processing_logs (video_id, timestamp);
create index processing_logs_owner_idx on public.processing_logs (owner_id);

create table public.rally_clips (
  id                 uuid primary key default gen_random_uuid(),
  video_id           uuid not null references public.videos(id) on delete cascade,
  owner_id           uuid not null references auth.users(id) on delete cascade,
  rally_index        int  not null,
  start_timestamp    real not null,
  end_timestamp      real not null,
  duration_seconds   real not null,
  clip_storage_path  text not null,
  created_at         timestamptz not null default now(),
  unique (video_id, rally_index)
);
create index rally_clips_owner_created_idx on public.rally_clips (owner_id, created_at desc);
