-- 0006: pipeline variant toggle + duplicate lineage for GB pipeline A/B.
-- pipeline_variant: which Phase 1 shuttle pipeline processes this video.
-- source_video_id: set on rows created by the duplicate-video edge function,
-- pointing at the video they were cloned from (sibling lookup is
-- bidirectional: follow source_video_id, or reverse-query on it).
alter table videos
  add column pipeline_variant text not null default 'legacy'
    check (pipeline_variant in ('legacy', 'gb_fusion')),
  add column source_video_id uuid references videos(id) on delete set null;
