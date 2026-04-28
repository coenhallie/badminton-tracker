-- Four private buckets, each enforcing path-prefix RLS so the first
-- folder segment must match the auth.uid() of the requester.
-- Modal uses the service role and bypasses these policies.

insert into storage.buckets (id, name, public) values
  ('videos',     'videos',     false),
  ('results',    'results',    false),
  ('clips',      'clips',      false),
  ('thumbnails', 'thumbnails', false)
on conflict (id) do nothing;

-- Helper: extract first path segment as text
-- (storage.foldername returns text[]; we want the first element as text)

-- Apply identical SELECT policy to all four buckets
do $$
declare
  bucket_id text;
begin
  for bucket_id in select unnest(array['videos','results','clips','thumbnails'])
  loop
    execute format($f$
      create policy "%s_owner_read" on storage.objects
        for select using (
          bucket_id = %L
          and (storage.foldername(name))[1] = auth.uid()::text
        );
    $f$, bucket_id, bucket_id);

    execute format($f$
      create policy "%s_owner_insert" on storage.objects
        for insert with check (
          bucket_id = %L
          and (storage.foldername(name))[1] = auth.uid()::text
        );
    $f$, bucket_id, bucket_id);

    execute format($f$
      create policy "%s_owner_delete" on storage.objects
        for delete using (
          bucket_id = %L
          and (storage.foldername(name))[1] = auth.uid()::text
        );
    $f$, bucket_id, bucket_id);
  end loop;
end $$;
