-- 0007: allow clients to set the pipeline variant during court setup.
-- 0002 uses column-level UPDATE grants; columns added later (0006) are not
-- covered automatically.
grant update (pipeline_variant) on public.videos to authenticated;
