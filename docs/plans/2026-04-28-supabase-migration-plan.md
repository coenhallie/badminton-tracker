# Supabase Migration + Rally Clip Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Migrate the badminton-tracker app from Convex to Supabase. Add closed-registration multi-user authentication. Generate per-rally video clips during the Modal pipeline so a separate KMP mobile app (planned in Milestone 2, separate plan) can fetch them. Remove Convex completely with zero leftovers.

**Architecture:** Vue 3 SPA + Supabase (Postgres + Auth + Storage + Edge Functions + Realtime) + Modal (GPU). Modal speaks to Supabase directly via the Python client with the service-role key. The Vue app and (future) KMP mobile app authenticate the same user via Supabase Auth. Edge Functions are the only path between authenticated clients and Modal; HMAC verifies every Modal call. Rally clips are produced by ffmpeg `-c copy` at the end of the Modal pipeline and uploaded to a private `clips` bucket. RLS scopes everything per `owner_id = auth.uid()`.

**Tech Stack:** Vue 3, Vite, `@supabase/supabase-js`, Supabase (Postgres 15, Auth, Storage, Edge Functions in Deno, Realtime), Modal (Python 3.11), `supabase-py`, ffmpeg (already in Modal container). YOLO26 / TrackNet stay unchanged.

**Reference:** Full design in [`2026-04-28-supabase-migration-design.md`](./2026-04-28-supabase-migration-design.md). Read it before starting.

**Out of scope for this plan:** the KMP mobile app (Milestone 2 — separate plan after Milestone 1 ships).

---

## Pre-flight assumptions

- You're working on branch `feat/supabase-migration` off `main`.
- The user creates the Supabase project and admin user manually via the dashboard — you cannot do this for them. They will provide `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` after Phase 1 Task 3.
- You are NOT to run a "smoke test" on production until the user provides credentials. All earlier work happens on the branch with placeholder env vars; integration tests against a real Supabase project happen at Phase 8.
- Modal CLI is logged in (`modal token current` returns a token).
- The current branch already contains the design doc (`docs/plans/2026-04-28-supabase-migration-design.md`).

---

## Phase 0 — Tooling preflight

### Task 1: Verify CLI tooling

**Files:** None — verification only.

**Step 1:** Verify supabase CLI is installed.

Run: `supabase --version`
Expected: any version `>= 1.150` is fine. If missing, install: `brew install supabase/tap/supabase`.

**Step 2:** Verify Modal CLI authentication.

Run: `modal token current`
Expected: prints a token ID. If not, run `modal token new`.

**Step 3:** Verify Deno is installed (for local Edge Function testing).

Run: `deno --version`
Expected: any version `>= 1.40`. If missing: `brew install deno`.

**Step 4:** Verify ffmpeg is installed locally (only used during plan validation; production ffmpeg is in the Modal container).

Run: `ffmpeg -version`
Expected: any recent version.

**Step 5:** Confirm working tree is clean and we're on the right branch.

Run: `git status && git branch --show-current`
Expected: clean tree, branch `feat/supabase-migration`.

---

## Phase 1 — Supabase project setup (USER must do dashboard tasks; you write the SQL)

### Task 2: Author the schema migration SQL

**Files:**
- Create: `supabase/migrations/0001_initial_schema.sql`

**Step 1:** Create the directory.

Run: `mkdir -p supabase/migrations`

**Step 2:** Write the schema migration.

Create `supabase/migrations/0001_initial_schema.sql` with:

```sql
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
create index rally_clips_video_idx         on public.rally_clips (video_id, rally_index);
```

**Step 3:** Commit.

```bash
git add supabase/migrations/0001_initial_schema.sql
git commit -m "feat(supabase): add initial schema migration"
```

---

### Task 3: Author the RLS policies migration

**Files:**
- Create: `supabase/migrations/0002_rls_policies.sql`

**Step 1:** Write the migration.

Create `supabase/migrations/0002_rls_policies.sql` with:

```sql
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
```

**Step 2:** Commit.

```bash
git add supabase/migrations/0002_rls_policies.sql
git commit -m "feat(supabase): add RLS policies for owner-scoped access"
```

---

### Task 4: Author the storage buckets migration

**Files:**
- Create: `supabase/migrations/0003_storage_buckets.sql`

**Step 1:** Write the migration.

Create `supabase/migrations/0003_storage_buckets.sql`:

```sql
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
```

**Step 2:** Commit.

```bash
git add supabase/migrations/0003_storage_buckets.sql
git commit -m "feat(supabase): add storage buckets and per-folder RLS"
```

---

### Task 5: Document Supabase project setup steps for the user

**Files:**
- Create: `supabase/SETUP.md`

**Step 1:** Write the setup instructions.

Create `supabase/SETUP.md`:

```markdown
# Supabase project setup (one-time, manual)

These steps create the Supabase project, run the migrations, and create
the admin user. They cannot be automated — they require dashboard access.

## 1. Create the project

1. Go to https://supabase.com/dashboard, sign in, click "New project".
2. Name: `badminton-tracker` (or whatever).
3. Region: closest to you and to the Modal region you use.
4. Pick a strong DB password and store it in your password manager.
5. Wait for the project to provision (~1 minute).

## 2. Capture credentials

From Project Settings → API:
- `SUPABASE_URL` — Project URL.
- `SUPABASE_ANON_KEY` — anon/public key.
- `SUPABASE_SERVICE_ROLE_KEY` — service_role key. **Treat as a secret.
  Never commit it; never put it in client bundles.**

Store them somewhere safe (1Password, etc.) — you'll paste them into
.env files and Modal Secrets later.

## 3. Run migrations

Two options. Pick one.

### Option A — `supabase` CLI (preferred)

From the repo root:

    supabase link --project-ref <your-project-ref>
    supabase db push

This applies every file in `supabase/migrations/` in order.

### Option B — SQL Editor (manual)

For each file in `supabase/migrations/` in order:
1. Open the SQL Editor in the dashboard.
2. Paste the file contents.
3. Run.

## 4. Configure Auth

In Authentication → Providers:

1. **Email**:
   - Enable.
   - Confirm email: ON.
   - **Disable signups**: ON (so only you can create users from the dashboard).
2. **Google** (optional but recommended):
   - Get a Google Cloud OAuth client ID/secret. (Google Cloud Console →
     APIs & Services → Credentials → Create OAuth 2.0 client ID; web type;
     authorized redirect URI: `https://<project-ref>.supabase.co/auth/v1/callback`).
   - Paste client ID and secret into Supabase Auth → Providers → Google.
   - Enable.

## 5. Create your admin user

In Authentication → Users:

1. Click "Add user" → "Send invite".
2. Enter your email.
3. Open the invite email and click the link to set your password.
4. You're now logged in via the Supabase studio with that user.

## 6. Hand off credentials to the implementation

Paste these into the project's `.env.local` (frontend) and into Modal
Secrets (backend) per the plan tasks that come next.
```

**Step 2:** Commit.

```bash
git add supabase/SETUP.md
git commit -m "docs(supabase): setup instructions for the dashboard portion"
```

**Step 3:** STOP and ask the user to perform the SETUP.md steps. The plan cannot proceed past the smoke test (Phase 8) without real credentials.

---

## Phase 2 — Edge Functions

### Task 6: Scaffold Edge Functions directory

**Files:**
- Create: `supabase/functions/_shared/cors.ts`
- Create: `supabase/functions/_shared/hmac.ts`

**Step 1:** Create the shared CORS helper.

Create `supabase/functions/_shared/cors.ts`:

```ts
export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
```

**Step 2:** Create the shared HMAC signer.

Create `supabase/functions/_shared/hmac.ts`:

```ts
// HMAC-SHA256 of body using the shared secret.
// Used to sign requests to Modal so Modal can verify the call came from us.

export async function signBody(body: string, secret: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(body));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
```

**Step 3:** Commit.

```bash
git add supabase/functions/_shared/
git commit -m "feat(edge): shared CORS + HMAC helpers"
```

---

### Task 7: Implement `/process-video` Edge Function

**Files:**
- Create: `supabase/functions/process-video/index.ts`

**Step 1:** Write the function.

Create `supabase/functions/process-video/index.ts`:

```ts
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_PROCESS_URL = Deno.env.get("MODAL_PROCESS_URL")!;
const MODAL_SHARED_SECRET = Deno.env.get("MODAL_SHARED_SECRET")!;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  const jwt = req.headers.get("Authorization")?.replace("Bearer ", "");
  if (!jwt) return resp(401, { error: "Missing Authorization" });

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: { user }, error: authErr } = await adminClient.auth.getUser(jwt);
  if (authErr || !user) return resp(401, { error: "Invalid JWT" });

  const { video_id } = await req.json();
  if (!video_id) return resp(400, { error: "video_id required" });

  // Verify ownership using a user-scoped client (RLS enforces it).
  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("*").eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  // Generate a signed URL for Modal to download the source video.
  const { data: signed, error: sErr } = await adminClient
    .storage.from("videos").createSignedUrl(video.storage_path, 3600);
  if (sErr || !signed) return resp(500, { error: "Could not sign video URL" });

  const body = JSON.stringify({
    video_id,
    owner_id: user.id,
    video_url: signed.signedUrl,
  });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  const modalRes = await fetch(MODAL_PROCESS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Signature": signature },
    body,
  });
  if (!modalRes.ok) {
    const text = await modalRes.text();
    return resp(502, { error: "Modal rejected", detail: text });
  }

  // Mark as processing immediately so the UI sees it.
  await adminClient.from("videos")
    .update({ status: "processing", processing_started_at: new Date().toISOString() })
    .eq("id", video_id);

  return resp(200, { ok: true });
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
```

**Step 2:** Commit.

```bash
git add supabase/functions/process-video/
git commit -m "feat(edge): process-video function with HMAC + ownership check"
```

---

### Task 8: Implement `/recalculate-speeds` Edge Function

**Files:**
- Create: `supabase/functions/recalculate-speeds/index.ts`

**Step 1:** Write the function. Same skeleton as `process-video`, but the Modal endpoint and the work it does are different.

Create `supabase/functions/recalculate-speeds/index.ts`:

```ts
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_SPEED_URL = Deno.env.get("MODAL_SPEED_URL")!;
const MODAL_SHARED_SECRET = Deno.env.get("MODAL_SHARED_SECRET")!;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  const jwt = req.headers.get("Authorization")?.replace("Bearer ", "");
  if (!jwt) return resp(401, { error: "Missing Authorization" });

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: { user }, error: authErr } = await adminClient.auth.getUser(jwt);
  if (authErr || !user) return resp(401, { error: "Invalid JWT" });

  const { video_id } = await req.json();
  if (!video_id) return resp(400, { error: "video_id required" });

  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("results_storage_path, manual_court_keypoints")
    .eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });
  if (!video.results_storage_path) return resp(400, { error: "No results yet" });

  const body = JSON.stringify({
    video_id,
    owner_id: user.id,
    results_storage_path: video.results_storage_path,
    manual_court_keypoints: video.manual_court_keypoints,
  });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  const modalRes = await fetch(MODAL_SPEED_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Signature": signature },
    body,
  });
  if (!modalRes.ok) {
    return resp(502, { error: "Modal speed call failed", detail: await modalRes.text() });
  }
  const speedJson = await modalRes.json();
  return resp(200, speedJson);
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
```

**Step 2:** Commit.

```bash
git add supabase/functions/recalculate-speeds/
git commit -m "feat(edge): recalculate-speeds proxies to Modal speed endpoint"
```

---

### Task 9: Implement `/export-pdf` Edge Function

**Files:**
- Create: `supabase/functions/export-pdf/index.ts`

**Step 1:** Write the function. Same pattern.

Create `supabase/functions/export-pdf/index.ts`:

```ts
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";
import { signBody } from "../_shared/hmac.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const MODAL_PDF_URL = Deno.env.get("MODAL_PDF_URL")!;
const MODAL_SHARED_SECRET = Deno.env.get("MODAL_SHARED_SECRET")!;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  const jwt = req.headers.get("Authorization")?.replace("Bearer ", "");
  if (!jwt) return resp(401, { error: "Missing Authorization" });

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: { user }, error: authErr } = await adminClient.auth.getUser(jwt);
  if (authErr || !user) return resp(401, { error: "Invalid JWT" });

  const reqBody = await req.json();
  const { video_id, config } = reqBody;
  if (!video_id) return resp(400, { error: "video_id required" });

  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("results_storage_path, storage_path")
    .eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  const body = JSON.stringify({
    video_id,
    owner_id: user.id,
    results_storage_path: video.results_storage_path,
    video_storage_path: video.storage_path,
    config,
  });
  const signature = await signBody(body, MODAL_SHARED_SECRET);

  const modalRes = await fetch(MODAL_PDF_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Signature": signature },
    body,
  });
  if (!modalRes.ok) {
    return resp(502, { error: "Modal PDF call failed", detail: await modalRes.text() });
  }
  return resp(200, await modalRes.json());
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
```

**Step 2:** Commit.

```bash
git add supabase/functions/export-pdf/
git commit -m "feat(edge): export-pdf proxies to Modal PDF endpoint"
```

---

### Task 10: Set Edge Function secrets and deploy

**Files:** None — uses CLI.

**Step 1:** Confirm secrets to set. The user must have completed `supabase/SETUP.md` and provided real credentials before this task.

Required secrets:
- `MODAL_PROCESS_URL` — set in Phase 3 Task 17 once Modal deploys.
- `MODAL_SPEED_URL` — set in Phase 3 Task 22.
- `MODAL_PDF_URL` — set in Phase 4 Task 24.
- `MODAL_SHARED_SECRET` — generate now: `openssl rand -hex 32`.

**Step 2:** Generate and persist the HMAC secret to a safe place (1Password etc.). Set it now in Supabase Edge Function env:

```bash
supabase secrets set MODAL_SHARED_SECRET=<value-from-openssl>
```

**Step 3:** Deferred — the three `MODAL_*_URL` secrets are set after Modal deploys (Tasks 17, 22, 24). For now, leave them unset.

**Step 4:** Local-test all three Edge Functions to make sure imports compile.

Run: `supabase functions serve` (in a separate terminal)
Then: `curl -X OPTIONS http://localhost:54321/functions/v1/process-video -H 'Access-Control-Request-Method: POST'`
Expected: 204 with CORS headers. (Real invocation comes later.)

Stop the serve process. Commit any local config changes if generated.

---

## Phase 3 — Modal pipeline rewrite

### Task 11: Copy `modal_convex_processor.py` → `modal_supabase_processor.py`

**Files:**
- Create: `backend/modal_supabase_processor.py` (initially as a copy)

**Step 1:** Copy the file.

```bash
cp backend/modal_convex_processor.py backend/modal_supabase_processor.py
```

**Step 2:** Open `backend/modal_supabase_processor.py`. Find the Modal app declaration (top of file, looks like `app = modal.App(name="...")`). Rename the app:

- Old: `name="badminton-convex-processor"` (or similar)
- New: `name="badminton-supabase-processor"`

**Step 3:** Commit the rename-only state (so the diff is clear in subsequent commits).

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): copy convex processor as supabase processor (no logic change yet)"
```

---

### Task 12: Replace Convex secret with Supabase secret in Modal app

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Find every `modal.Secret.from_name("convex-secrets")` reference (Section 7 of the audit pointed at lines 1018, 1053). Replace with `modal.Secret.from_name("supabase-secrets")`.

Also add a second secret reference for the HMAC: `modal.Secret.from_name("modal-shared-secret")` to functions that handle HTTP entry.

**Step 2:** Add module-level Supabase client setup near the top of the file:

```python
import os
from supabase import create_client, Client

_supabase: Client | None = None

def supabase_client() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _supabase = create_client(url, key)
    return _supabase
```

**Step 3:** Add `supabase` to the Modal image's pip install list. Find where the image is defined (look for `modal.Image.debian_slim()` or similar) and add `supabase>=2.5.0` to the `pip_install([...])` list.

**Step 4:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): switch to supabase secret + add supabase client"
```

---

### Task 13: Add HMAC verifier to Modal entry handler

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Add helper near top of file:

```python
import hmac
import hashlib

def verify_hmac(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

**Step 2:** Find the existing `@web_endpoint` (or `@asgi_app` / `@web_function`) entry — the one Convex's action POSTs to. At the top of its handler, add:

```python
import json

raw_body = await request.body()  # or however the framework hands it to you; adapt to actual decorator
sig = request.headers.get("X-Signature")
secret = os.environ["MODAL_SHARED_SECRET"]
if not verify_hmac(raw_body, sig, secret):
    return JSONResponse({"error": "unauthorized"}, status_code=401)

payload = json.loads(raw_body)
video_id = payload["video_id"]
owner_id = payload["owner_id"]
video_url = payload["video_url"]
# (callback_url no longer in payload — Modal speaks Supabase directly)
```

Remove any code that reads `callback_url` from the payload — it's gone.

**Step 3:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): verify HMAC on entry, remove callback_url"
```

---

### Task 14: Replace status/log/results callbacks with direct Supabase writes

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Find every `requests.post(f"{callback_url}/updateStatus", ...)` call. Replace with:

```python
supabase_client().table("videos").update({
    "progress": pct,
    "current_frame": frame,
    "total_frames": total,
}).eq("id", video_id).execute()
```

(Match the field names: `current_frame` not `currentFrame`, `total_frames` not `totalFrames`. Postgres column convention is snake_case.)

**Step 2:** Find every `requests.post(f"{callback_url}/addLog", ...)` call. Replace with:

```python
supabase_client().table("processing_logs").insert({
    "video_id": video_id,
    "owner_id": owner_id,
    "message": message,
    "level": level,
    "category": category,
}).execute()
```

Make sure `owner_id` is in scope wherever a log is written. If not, plumb it through from the entry payload (it's in `payload["owner_id"]`).

**Step 3:** Find every `requests.post(f"{callback_url}/updateResults", ...)` call. Replace with:

```python
supabase_client().table("videos").update({
    "status": "processing",  # final status flip happens AFTER clip generation
    "results_meta": meta_dict,
    "results_storage_path": results_storage_path,
    "processed_video_path": processed_video_path,  # if applicable
    "skeleton_data_path": skeleton_data_path,      # if applicable
}).eq("id", video_id).execute()
```

**Step 4:** Find the `/generateUploadUrl` + signed-URL PUT pattern for the results JSON. Replace with one call:

```python
results_storage_path = f"{owner_id}/{video_id}/results.json"
supabase_client().storage.from_("results").upload(
    path=results_storage_path,
    file=json.dumps(results_dict).encode(),
    file_options={"content-type": "application/json", "upsert": "true"},
)
```

Then write `results_storage_path` into the videos row update from Step 3.

**Step 5:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): replace HTTP callbacks with direct Supabase writes"
```

---

### Task 15: Replace player thumbnails upload + fetch source video

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Find the player thumbnail upload block (originally posts to `/setPlayerThumbnails`). Replace with:

```python
def upload_player_thumbnail(supabase, video_id, owner_id, player_idx, jpeg_bytes):
    path = f"{owner_id}/{video_id}/player_{player_idx}.jpg"
    supabase.storage.from_("thumbnails").upload(
        path=path,
        file=jpeg_bytes,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )
    return path

# Caller:
p0_path = upload_player_thumbnail(supabase_client(), video_id, owner_id, 0, p0_jpeg)
p1_path = upload_player_thumbnail(supabase_client(), video_id, owner_id, 1, p1_jpeg)

supabase_client().table("videos").update({
    "player_labels": {
        "player_0_thumbnail_path": p0_path,
        "player_1_thumbnail_path": p1_path,
    }
}).eq("id", video_id).execute()
```

Note the schema change: storage paths are stored as `text` not as opaque IDs. `player_labels` is a JSONB column whose shape is application-defined.

**Step 2:** Find the source video download block. The video URL now arrives in the payload (`video_url`) as a pre-signed Supabase Storage URL good for 1 hour. Just download it:

```python
import urllib.request
local_path = f"/cache/{video_id}.mp4"
urllib.request.urlretrieve(video_url, local_path)
```

(If the existing code uses `requests.get(video_url, stream=True)` or similar streaming download, keep that pattern — just confirm the URL is `payload["video_url"]` from the entry handler.)

**Step 3:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): swap thumbnails + source-video to Supabase Storage"
```

---

### Task 16: Add rally clip generation function

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Add the function. Place it near the top-level helpers or near `rally_detection` import:

```python
import subprocess

def cut_and_upload_rally_clips(video_path: str, rallies: list, video_id: str, owner_id: str):
    """
    For each detected rally, cut the source video using ffmpeg stream copy
    (no re-encode), upload to the 'clips' bucket, and insert a rally_clips row.
    Idempotent via UNIQUE (video_id, rally_index).
    """
    sb = supabase_client()
    for rally in rallies:
        clip_local = f"/cache/{video_id}_rally_{rally['id']}.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(rally["start_timestamp"]),
                "-to", str(rally["end_timestamp"]),
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                clip_local,
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            sb.table("processing_logs").insert({
                "video_id": video_id,
                "owner_id": owner_id,
                "message": f"clip generation failed for rally {rally['id']}: "
                           f"{e.stderr.decode(errors='replace')[:200]}",
                "level": "warning",
                "category": "processing",
            }).execute()
            continue

        storage_path = f"{owner_id}/{video_id}/rally_{rally['id']}.mp4"
        with open(clip_local, "rb") as f:
            sb.storage.from_("clips").upload(
                path=storage_path,
                file=f.read(),
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )

        sb.table("rally_clips").upsert({
            "video_id": video_id,
            "owner_id": owner_id,
            "rally_index": rally["id"],
            "start_timestamp": rally["start_timestamp"],
            "end_timestamp": rally["end_timestamp"],
            "duration_seconds": rally["duration_seconds"],
            "clip_storage_path": storage_path,
        }, on_conflict="video_id,rally_index").execute()

        os.remove(clip_local)
```

**Step 2:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): add rally clip cutter (ffmpeg -c copy + upload)"
```

---

### Task 17: Wire clip generation into the pipeline

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Find the place where the pipeline:
1. Has just finished rally detection (look for `rally_detection.detect(...)` or similar).
2. Has just uploaded the results JSON.
3. Is about to flip status to `'completed'`.

In that order, insert a call:

```python
cut_and_upload_rally_clips(
    video_path=local_path,         # path to /cache/{video_id}.mp4
    rallies=results_dict["rallies"],
    video_id=video_id,
    owner_id=owner_id,
)
```

**Step 2:** Confirm the status flip to `'completed'` happens AFTER the clip step (so the UI sees `completed` only when clips are ready):

```python
supabase_client().table("videos").update({
    "status": "completed",
    "completed_at": "now()",
}).eq("id", video_id).execute()
```

(Use `datetime.now(timezone.utc).isoformat()` if `"now()"` isn't accepted by the client.)

**Step 3:** Confirm cleanup of `/cache/{video_id}*` runs at the end. If it was using `glob.glob` patterns, leave them — they'll catch the rally clips that were `os.remove`-ed already plus the source video.

**Step 4:** Commit.

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(modal): integrate clip cutting between rally detection and status flip"
```

---

### Task 18: Add `recalculate_speeds` Modal endpoint

**Files:**
- Modify: `backend/modal_supabase_processor.py`

**Step 1:** Identify where speed math currently runs. The Convex `/api/speed` endpoint computed in TS — that won't survive. The Python homography code is partially in `rally_detection.py` and other modules; locate the kinematics + homography code via `grep -rn 'homography\|speed' backend/`.

**Step 2:** Add a new web endpoint to the Modal app:

```python
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("supabase-secrets"),
             modal.Secret.from_name("modal-shared-secret")],
)
@modal.web_endpoint(method="POST")
async def recalculate_speeds(request):
    raw_body = await request.body()
    sig = request.headers.get("X-Signature")
    secret = os.environ["MODAL_SHARED_SECRET"]
    if not verify_hmac(raw_body, sig, secret):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    payload = json.loads(raw_body)
    video_id = payload["video_id"]
    owner_id = payload["owner_id"]
    results_storage_path = payload["results_storage_path"]
    manual_keypoints = payload.get("manual_court_keypoints")

    # Download results.json from Supabase Storage
    sb = supabase_client()
    blob = sb.storage.from_("results").download(results_storage_path)
    results = json.loads(blob)

    # Recompute speeds with manual keypoints if provided.
    speed_data = compute_speeds_from_skeleton(  # exists in repo or port from convex/http.ts
        skeleton_data=results.get("skeleton_data", []),
        fps=results.get("fps", 30),
        video_width=results.get("video_width"),
        video_height=results.get("video_height"),
        manual_court_keypoints=manual_keypoints,
    )

    return JSONResponse({
        "video_id": video_id,
        "speed_data": speed_data,
        "manual_keypoints_used": manual_keypoints is not None,
        "detection_source": "modal",
        "status": "success",
    })
```

**Step 3:** If `compute_speeds_from_skeleton` does not yet exist as a callable Python function (it might live only as TS in `convex/http.ts`), port it now:

- Open `convex/http.ts`. Find the `/api/speed` httpAction handler.
- Re-implement the algorithm in Python in a new file: `backend/speed_calc.py`.
- The math: pixel→meter homography from court corners + Kalman smoothing + outlier rejection. Reuse OpenCV (`cv2.findHomography`, `cv2.perspectiveTransform`) which is already in the Modal image.
- Match the response shape (see `SpeedDataResponse` in `src/services/api.ts:638-663`) so the frontend doesn't have to change beyond the URL.

**Step 4:** Commit.

```bash
git add backend/modal_supabase_processor.py backend/speed_calc.py
git commit -m "feat(modal): add recalculate_speeds endpoint with HMAC + Python homography"
```

---

### Task 19: Configure Modal Secrets and deploy

**Files:** None — uses CLI.

**Step 1:** Set Modal Secrets:

```bash
modal secret create supabase-secrets \
  SUPABASE_URL=<value> \
  SUPABASE_SERVICE_ROLE_KEY=<value>

modal secret create modal-shared-secret \
  MODAL_SHARED_SECRET=<the value generated in Phase 2 Task 10 Step 2>
```

(Get values from the SETUP.md output and the openssl-generated secret.)

**Step 2:** Deploy.

Run: `modal deploy backend/modal_supabase_processor.py`
Expected: prints two web endpoint URLs:
- `MODAL_PROCESS_URL` = the URL of the original entry function (the one that processes a full video).
- `MODAL_SPEED_URL` = the URL of `recalculate_speeds`.

**Step 3:** Set those URLs as Edge Function secrets:

```bash
supabase secrets set MODAL_PROCESS_URL=<url-1>
supabase secrets set MODAL_SPEED_URL=<url-2>
```

**Step 4:** Sanity check the deploy.

Run: `curl -X POST <MODAL_PROCESS_URL> -H 'Content-Type: application/json' -d '{"video_id":"test"}'`
Expected: 401 with `{"error": "unauthorized"}` (because no HMAC signature). Confirms the auth gate works.

**Step 5:** Commit (no code changes; this is just a checkpoint task).

---

## Phase 4 — Modal PDF export adapter

### Task 20: Update `modal_pdf_export.py` for HMAC + Supabase Storage

**Files:**
- Modify: `backend/modal_pdf_export.py`

**Step 1:** Read the current file. Identify the entry handler (likely `@modal.web_endpoint`).

**Step 2:** Add HMAC verification at the top, identical to Task 13 — `verify_hmac(raw_body, sig, secret)` against `MODAL_SHARED_SECRET`.

**Step 3:** Replace whatever fetches input data (today probably from a Convex URL) with Supabase Storage downloads:

```python
sb = supabase_client()  # use the same helper from modal_supabase_processor.py — extract to a shared module if needed
results = json.loads(sb.storage.from_("results").download(results_storage_path))
# If video frame is needed for heatmap rendering:
video_bytes = sb.storage.from_("videos").download(video_storage_path)
# write to /tmp and read with cv2.VideoCapture or similar
```

**Step 4:** Return the PDF either:
- Inline as base64 (current behavior of `downloadPDFExport`). Easiest. Frontend already handles this.
- OR upload to a `pdfs/` bucket and return a signed URL. Optional improvement; defer.

Pick base64 inline for v1.

**Step 5:** If `supabase_client()` is duplicated between `modal_supabase_processor.py` and `modal_pdf_export.py`, factor into `backend/supabase_helpers.py`. Add the file. Update both Modal apps to import from it.

**Step 6:** Deploy and capture the URL:

```bash
modal deploy backend/modal_pdf_export.py
```

Set `MODAL_PDF_URL`:

```bash
supabase secrets set MODAL_PDF_URL=<url>
```

**Step 7:** Commit.

```bash
git add backend/modal_pdf_export.py backend/supabase_helpers.py
git commit -m "feat(modal): adapt pdf export for HMAC + Supabase storage inputs"
```

---

## Phase 5 — Frontend infrastructure

### Task 21: Replace dependencies

**Files:**
- Modify: `package.json`

**Step 1:** Remove `convex`, `convex-vue`, `@convex-vue/core`. Add `@supabase/supabase-js`.

```bash
npm uninstall convex convex-vue @convex-vue/core
npm install @supabase/supabase-js@^2.45.0
```

**Step 2:** Verify lockfile is clean (no transitive convex packages).

Run: `grep -i 'convex' package-lock.json | head`
Expected: zero matches.

**Step 3:** Commit.

```bash
git add package.json package-lock.json
git commit -m "chore(deps): drop convex packages, add @supabase/supabase-js"
```

---

### Task 22: Create `src/lib/supabase.ts`

**Files:**
- Create: `src/lib/supabase.ts`

**Step 1:** Write it.

```ts
import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string;
const anon = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!url || !anon) {
  // Fail loud during dev; production builds should always have these set.
  console.error("Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY");
}

export const supabase = createClient(url, anon, {
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
});
```

**Step 2:** Commit.

```bash
git add src/lib/supabase.ts
git commit -m "feat(frontend): add supabase-js client setup"
```

---

### Task 23: Create `useSession` composable

**Files:**
- Create: `src/composables/useSession.ts`

**Step 1:** Write it.

```ts
import { ref, computed, onUnmounted } from "vue";
import type { Session, User } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";

const session = ref<Session | null>(null);
const ready = ref(false);

let initialized = false;
function init() {
  if (initialized) return;
  initialized = true;
  supabase.auth.getSession().then(({ data }) => {
    session.value = data.session;
    ready.value = true;
  });
  supabase.auth.onAuthStateChange((_evt, s) => {
    session.value = s;
  });
}

export function useSession() {
  init();
  const user = computed<User | null>(() => session.value?.user ?? null);
  const isAuthenticated = computed(() => !!session.value);
  return { session, user, isAuthenticated, ready };
}

export async function signOut() {
  await supabase.auth.signOut();
}
```

**Step 2:** Commit.

```bash
git add src/composables/useSession.ts
git commit -m "feat(frontend): add useSession composable"
```

---

### Task 24: Create `useReactiveRow` composable

**Files:**
- Create: `src/composables/useReactiveRow.ts`

**Step 1:** Write it.

```ts
import { ref, watchEffect, type Ref } from "vue";
import { supabase } from "@/lib/supabase";

export function useReactiveRow<T>(table: string, id: Ref<string | null | undefined>) {
  const row = ref<T | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  watchEffect(async (onCleanup) => {
    if (!id.value) {
      row.value = null;
      return;
    }
    loading.value = true;
    error.value = null;

    const { data, error: e } = await supabase.from(table).select("*").eq("id", id.value).maybeSingle();
    if (e) {
      error.value = e.message;
      loading.value = false;
      return;
    }
    row.value = data as T | null;
    loading.value = false;

    const channel = supabase
      .channel(`${table}-row-${id.value}`)
      .on("postgres_changes",
          { event: "UPDATE", schema: "public", table, filter: `id=eq.${id.value}` },
          (payload) => { row.value = payload.new as T; })
      .on("postgres_changes",
          { event: "DELETE", schema: "public", table, filter: `id=eq.${id.value}` },
          () => { row.value = null; })
      .subscribe();

    onCleanup(() => { supabase.removeChannel(channel); });
  });

  return { row, loading, error };
}
```

**Step 2:** Commit.

```bash
git add src/composables/useReactiveRow.ts
git commit -m "feat(frontend): add useReactiveRow composable"
```

---

### Task 25: Create `useReactiveList` composable

**Files:**
- Create: `src/composables/useReactiveList.ts`

**Step 1:** Write it.

```ts
import { ref, watchEffect, type Ref } from "vue";
import { supabase } from "@/lib/supabase";

interface ListFilter { column: string; value: string; }
interface Options { orderBy?: string; ascending?: boolean; }

export function useReactiveList<T extends { id: string | number }>(
  table: string,
  filter: Ref<ListFilter | null>,
  options: Options = {}
) {
  const items = ref<T[]>([]) as Ref<T[]>;
  const loading = ref(false);
  const error = ref<string | null>(null);

  watchEffect(async (onCleanup) => {
    if (!filter.value) {
      items.value = [];
      return;
    }
    loading.value = true;
    error.value = null;
    const { column, value } = filter.value;

    let q = supabase.from(table).select("*").eq(column, value);
    if (options.orderBy) {
      q = q.order(options.orderBy, { ascending: options.ascending ?? true });
    }
    const { data, error: e } = await q;
    if (e) {
      error.value = e.message;
      loading.value = false;
      return;
    }
    items.value = (data ?? []) as T[];
    loading.value = false;

    const channel = supabase
      .channel(`${table}-list-${column}-${value}`)
      .on("postgres_changes",
          { event: "INSERT", schema: "public", table, filter: `${column}=eq.${value}` },
          (p) => { items.value = [...items.value, p.new as T]; })
      .on("postgres_changes",
          { event: "DELETE", schema: "public", table, filter: `${column}=eq.${value}` },
          (p) => { items.value = items.value.filter((it) => it.id !== (p.old as T).id); })
      .subscribe();

    onCleanup(() => { supabase.removeChannel(channel); });
  });

  return { items, loading, error };
}
```

**Step 2:** Commit.

```bash
git add src/composables/useReactiveList.ts
git commit -m "feat(frontend): add useReactiveList composable"
```

---

### Task 26: Create `LoginView.vue`

**Files:**
- Create: `src/views/LoginView.vue`

**Step 1:** Write it.

```vue
<script setup lang="ts">
import { ref } from "vue";
import { supabase } from "@/lib/supabase";

const email = ref("");
const password = ref("");
const error = ref<string | null>(null);
const submitting = ref(false);

async function signInEmail() {
  submitting.value = true;
  error.value = null;
  const { error: e } = await supabase.auth.signInWithPassword({
    email: email.value,
    password: password.value,
  });
  submitting.value = false;
  if (e) error.value = e.message;
}

async function signInGoogle() {
  const { error: e } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });
  if (e) error.value = e.message;
}
</script>

<template>
  <div class="login">
    <h1>Sign in</h1>
    <form @submit.prevent="signInEmail">
      <input v-model="email" type="email" placeholder="email" autocomplete="email" required />
      <input v-model="password" type="password" placeholder="password" autocomplete="current-password" required />
      <button type="submit" :disabled="submitting">{{ submitting ? "Signing in…" : "Sign in" }}</button>
    </form>
    <button @click="signInGoogle">Sign in with Google</button>
    <p v-if="error" class="error">{{ error }}</p>
    <p class="hint">Registration is closed. Contact the admin if you need an account.</p>
  </div>
</template>

<style scoped>
.login { max-width: 320px; margin: 8rem auto; display: flex; flex-direction: column; gap: 1rem; }
.error { color: tomato; }
.hint { font-size: 0.85rem; opacity: 0.7; }
</style>
```

**Step 2:** Commit.

```bash
git add src/views/LoginView.vue
git commit -m "feat(frontend): add LoginView (no signup link, closed registration)"
```

---

### Task 27: Add route guard to `App.vue`

**Files:**
- Modify: `src/App.vue`

**Step 1:** At the top of `<script setup>`, import:

```ts
import { useSession } from "@/composables/useSession";
import LoginView from "@/views/LoginView.vue";

const { isAuthenticated, ready } = useSession();
```

**Step 2:** In the template, wrap the existing root content with a v-if/else block:

```vue
<template>
  <template v-if="!ready">
    <!-- Brief loading flash; usually invisible -->
  </template>
  <template v-else-if="!isAuthenticated">
    <LoginView />
  </template>
  <template v-else>
    <!-- existing app content goes here -->
  </template>
</template>
```

**Step 3:** Commit.

```bash
git add src/App.vue
git commit -m "feat(frontend): gate app on isAuthenticated; show LoginView when signed out"
```

---

## Phase 6 — Component migrations

### Task 28: Migrate `VideoUpload.vue`

**Files:**
- Modify: `src/components/VideoUpload.vue`

**Step 1:** Remove the convex imports (lines 3-4):

```diff
- import { useConvexClient } from 'convex-vue'
- import { api } from '../../convex/_generated/api'
```

**Step 2:** Add Supabase imports at top of `<script setup>`:

```ts
import { supabase } from "@/lib/supabase";
import { useSession } from "@/composables/useSession";
import { v4 as uuid } from "uuid"; // ensure 'uuid' is in deps; install if not: npm i uuid && npm i -D @types/uuid
```

**Step 3:** Replace the upload flow (around line 182-188). The current flow is:
1. `generateUploadUrl` mutation → signed URL
2. POST file bytes to that URL via XHR
3. `createVideo` mutation with the storage ID

New flow:

```ts
async function uploadAndCreate(file: File) {
  const { user } = useSession();
  if (!user.value) throw new Error("Not signed in");

  const videoId = uuid();
  const path = `${user.value.id}/${videoId}.mp4`;

  // 1. Upload bytes
  const { error: upErr } = await supabase.storage
    .from("videos")
    .upload(path, file, { contentType: file.type, upsert: false });
  if (upErr) throw upErr;

  // 2. Insert row (RLS lets us insert because owner_id = auth.uid())
  const { data: row, error: insErr } = await supabase
    .from("videos")
    .insert({
      id: videoId,
      owner_id: user.value.id,
      filename: file.name,
      size: file.size,
      storage_path: path,
      status: "uploaded",
    })
    .select()
    .single();
  if (insErr) throw insErr;

  return row.id;
}
```

**Step 4:** Plumb progress: the existing XHR-based progress UI used XHR's `onprogress`. supabase-js's `upload` does NOT report progress; for v1, replace progress with a simple "Uploading…" indicator. (If you really need progress, use `tus-js-client` against Supabase's TUS endpoint — defer to v2.)

**Step 5:** Run type-check to make sure there are no leftover `api.*` references.

Run: `npm run type-check`
Expected: zero TypeScript errors related to this file.

**Step 6:** Commit.

```bash
git add src/components/VideoUpload.vue
git commit -m "feat(frontend): migrate VideoUpload to Supabase Storage + insert"
```

---

### Task 29: Migrate `AnalysisProgress.vue`

**Files:**
- Modify: `src/components/AnalysisProgress.vue`

**Step 1:** Remove convex imports (lines 3-6):

```diff
- import { useConvexQuery, useConvexClient } from 'convex-vue'
- import { api } from '../../convex/_generated/api'
- import type { Id } from '../../convex/_generated/dataModel'
```

**Step 2:** Replace `Id<"videos">` types with plain `string`. The file passes around `convexVideoId: Id<"videos">`; rename to `videoId: string` throughout.

**Step 3:** Swap the two `useConvexQuery` calls (lines 24-34 area):

```ts
// BEFORE
const { data: videoData } = useConvexQuery(
  api.videos.getVideo,
  computed(() => ({ videoId: convexVideoId.value }))
);
const { data: logsData } = useConvexQuery(
  api.videos.getProcessingLogs,
  computed(() => ({ videoId: convexVideoId.value }))
);

// AFTER
import { useReactiveRow } from "@/composables/useReactiveRow";
import { useReactiveList } from "@/composables/useReactiveList";
import type { Video, ProcessingLog } from "@/types/db";

const { row: videoData } = useReactiveRow<Video>("videos", videoId);
const { items: logsData } = useReactiveList<ProcessingLog>(
  "processing_logs",
  computed(() => ({ column: "video_id", value: videoId.value })),
  { orderBy: "timestamp", ascending: true }
);
```

(Add `Video` and `ProcessingLog` types in a new file `src/types/db.ts` matching the schema.)

**Step 4:** Replace the `processVideo` action call (around line 243):

```ts
// BEFORE
await convex.action(api.videos.processVideo, { videoId: convexVideoId.value });

// AFTER
const { error } = await supabase.functions.invoke("process-video", {
  body: { video_id: videoId.value },
});
if (error) throw error;
```

**Step 5:** Replace the `resultsUrl` consumption (around line 102). The current code does `fetch(resultsUrl)` to get the results JSON. `videoData.results_storage_path` now holds the path; sign it on demand:

```ts
async function fetchResults() {
  if (!videoData.value?.results_storage_path) return null;
  const { data, error } = await supabase
    .storage.from("results")
    .createSignedUrl(videoData.value.results_storage_path, 3600);
  if (error) throw error;
  const res = await fetch(data.signedUrl);
  return res.json();
}
```

**Step 6:** Field rename sweep. The Convex shape used camelCase (`currentFrame`, `totalFrames`, `processedVideoStorageId`). Postgres uses snake_case (`current_frame`, `total_frames`, `processed_video_path`). Update every reference in this file's template and script.

**Step 7:** Type-check + manual lint pass.

Run: `npm run type-check`

**Step 8:** Commit.

```bash
git add src/components/AnalysisProgress.vue src/types/db.ts
git commit -m "feat(frontend): migrate AnalysisProgress to Supabase reactive queries"
```

---

### Task 30: Migrate `CourtSetup.vue`

**Files:**
- Modify: `src/components/CourtSetup.vue`

**Step 1:** Remove convex imports.

**Step 2:** Replace the `setManualCourtKeypoints` mutation (around line 355):

```ts
// BEFORE
await convex.mutation(api.videos.setManualCourtKeypoints, { videoId, keypoints });

// AFTER
const { error } = await supabase
  .from("videos")
  .update({ manual_court_keypoints: keypoints })
  .eq("id", videoId);
if (error) throw error;
```

**Step 3:** Replace `videoUrl` resolution (around line 103):

```ts
// BEFORE
const videoUrl = await fetchVideoUrl(videoId); // services/api.ts

// AFTER (inline)
const { data: row } = await supabase.from("videos").select("storage_path").eq("id", videoId).single();
const { data: signed } = await supabase.storage.from("videos")
  .createSignedUrl(row!.storage_path, 3600);
const videoUrl = signed!.signedUrl;
```

(`fetchVideoUrl` will be deleted from `services/api.ts` in Phase 7. Inline the lookup here for now.)

**Step 4:** Type-check.

**Step 5:** Commit.

```bash
git add src/components/CourtSetup.vue
git commit -m "feat(frontend): migrate CourtSetup to Supabase update + signed URL"
```

---

### Task 31: Migrate `App.vue`, `VideoPlayer.vue`, `ResultsDashboard.vue`

**Files:**
- Modify: `src/App.vue` (line 557 area — `videoUrl` fetch)
- Modify: `src/components/VideoPlayer.vue` (line 258 area — receives `videoUrl` prop)
- Modify: `src/components/ResultsDashboard.vue` (consumes `services/api.ts`)

**Step 1:** In `App.vue`, replace the `fetchVideoUrl(videoId)` call with the same inline Supabase Storage signed-URL pattern from Task 30.

**Step 2:** `VideoPlayer.vue` only receives `videoUrl` as a prop — no Convex changes needed beyond verifying the prop still arrives. After App.vue is migrated, this should "just work."

**Step 3:** `ResultsDashboard.vue` likely uses `getHeatmap`, `getSpeedData`, `getRecalculatedZoneAnalytics`, etc. from `services/api.ts`. Those will be rewritten in Phase 7; for now, commit the file unchanged BUT remove any direct `api.*` Convex imports if present.

**Step 4:** Type-check.

**Step 5:** Commit.

```bash
git add src/App.vue src/components/VideoPlayer.vue src/components/ResultsDashboard.vue
git commit -m "feat(frontend): migrate App + VideoPlayer + ResultsDashboard videoUrl resolution"
```

---

## Phase 7 — `src/services/api.ts` rewrite

### Task 32: Strip dead code (Flask fallbacks + USE_CONVEX branching)

**Files:**
- Modify: `src/services/api.ts`

**Step 1:** Delete the top-level constants:

```diff
- const CONVEX_URL = import.meta.env.VITE_CONVEX_URL as string | undefined
- const USE_CONVEX = !!CONVEX_URL
- const CONVEX_SITE_URL = CONVEX_URL?.replace('.convex.cloud', '.convex.site') || ''
- const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
- const WS_BASE_URL = API_BASE_URL.replace('http', 'ws')
```

**Step 2:** Delete every function that has a `USE_CONVEX` ? branch. For each, keep ONLY the logic for the new path (Supabase-direct, Edge Function, or removed). This includes:

Functions to **delete entirely** (no replacement needed):
- `uploadVideo` (Flask-only)
- `analyzeVideo` (Flask-only)
- `getResults` (Flask-only — frontend reads results JSON via signed URL now)
- `getOriginalVideoUrl` (Flask-only)
- `AnalysisProgressSocket` (Flask WebSocket — replaced by Supabase Realtime)
- `getApiBaseUrl`, `isUsingConvex` (no longer meaningful)
- `getPDFExportUrl`, `exportPDFWithConfig`, `getPDFExportPreview` (Flask-only paths)

Functions to **rewrite** (covered in next tasks):
- `fetchVideoUrl`
- `checkApiHealth`, `getApiHealthDetails`
- `getManualKeypointsStatus`, `setManualCourtKeypoints`
- `getHeatmap`, `preloadHeatmap`
- `getSpeedData`, `getSpeedTimeline`, `triggerSpeedRecalculation`
- `recalculateSpeedsFromSkeleton`
- `getRecalculatedZoneAnalytics`
- `downloadPDFExport`, `exportPDFWithFrontendData`

**Step 3:** Keep the utility classes (`debounce`, `throttle`, `SimpleCache`, `apiCache`).

**Step 4:** Type-check (will fail due to missing functions; that's expected at this point).

**Step 5:** Commit work-in-progress (broken state, but isolated).

```bash
git add src/services/api.ts
git commit -m "refactor(api): delete Flask fallbacks + USE_CONVEX branching (WIP)"
```

---

### Task 33: Rewrite `fetchVideoUrl`, `checkApiHealth`, manual keypoints

**Files:**
- Modify: `src/services/api.ts`

**Step 1:** Add the new `fetchVideoUrl`:

```ts
import { supabase } from "@/lib/supabase";

export async function fetchVideoUrl(videoId: string): Promise<string> {
  const { data: row, error } = await supabase
    .from("videos").select("storage_path").eq("id", videoId).single();
  if (error || !row) throw error ?? new Error("Video not found");
  const { data: signed, error: e2 } = await supabase
    .storage.from("videos").createSignedUrl(row.storage_path, 3600);
  if (e2 || !signed) throw e2 ?? new Error("Could not sign URL");
  return signed.signedUrl;
}
```

**Step 2:** Trivial health checks:

```ts
export async function checkApiHealth(): Promise<boolean> {
  try {
    const { error } = await supabase.from("videos").select("id").limit(1);
    return !error;
  } catch { return false; }
}

export async function getApiHealthDetails() {
  return null; // Drop the elaborate health response; UI only needs the boolean.
}

export function getApiBaseUrl(): string {
  return import.meta.env.VITE_SUPABASE_URL as string;
}
```

(Update any consumers of `getApiHealthDetails` to handle `null` gracefully; or just delete consumers.)

**Step 3:** Manual keypoints — direct Supabase:

```ts
export async function getManualKeypointsStatus(videoId: string): Promise<ManualKeypointsStatus> {
  const { data, error } = await supabase
    .from("videos")
    .select("manual_court_keypoints")
    .eq("id", videoId)
    .single();
  if (error || !data) return { has_manual_keypoints: false, keypoints: null };
  return {
    has_manual_keypoints: !!data.manual_court_keypoints,
    keypoints: data.manual_court_keypoints,
  };
}

export async function setManualCourtKeypoints(
  keypoints: ManualKeypointsRequest,
  videoId: string,
): Promise<ManualKeypointsResponse> {
  const { error } = await supabase
    .from("videos")
    .update({ manual_court_keypoints: keypoints })
    .eq("id", videoId);
  if (error) throw error;
  return { status: "success", message: "ok", keypoints };
}
```

**Step 4:** Delete the `currentVideoId` module-level state and the `setCurrentVideoId` setter — callers always know the videoId now.

**Step 5:** Commit.

```bash
git add src/services/api.ts
git commit -m "feat(api): rewrite fetchVideoUrl + health + manual keypoints to Supabase"
```

---

### Task 34: Port heatmap and zone analytics to client-side composables

**Files:**
- Create: `src/composables/useHeatmap.ts`
- Create: `src/composables/useZoneAnalytics.ts`
- Modify: `src/services/api.ts`

**Step 1:** Read `convex/http.ts`. Find the `/api/heatmap` httpAction handler. Copy its math (TS) into a new composable.

Create `src/composables/useHeatmap.ts`:

```ts
import { ref } from "vue";
import { supabase } from "@/lib/supabase";
import type { HeatmapData } from "@/services/api";

async function downloadResults(videoId: string): Promise<any> {
  const { data: row, error } = await supabase
    .from("videos").select("results_storage_path").eq("id", videoId).single();
  if (error || !row?.results_storage_path) throw error ?? new Error("No results");
  const { data: signed } = await supabase.storage.from("results")
    .createSignedUrl(row.results_storage_path, 3600);
  const res = await fetch(signed!.signedUrl);
  return res.json();
}

export async function computeHeatmap(videoId: string, playerId?: number): Promise<HeatmapData> {
  const results = await downloadResults(videoId);
  // PORT THE TS MATH FROM convex/http.ts /api/heatmap HERE:
  // - Iterate skeleton_data frames
  // - For each player position, accumulate into a 2D grid
  // - Apply Gaussian blur (or whatever the original did)
  // - Return { width, height, combined_heatmap, player_heatmaps, ... }
  // Keep the response shape identical to today's HeatmapData interface.
  throw new Error("TODO: implement port of convex/http.ts heatmap math");
}
```

**Step 2:** Actually do the port. Open `convex/http.ts`, locate the heatmap handler, copy the algorithm into the function above. Delete the `throw new Error("TODO ...")`. Keep the response shape identical to the existing `HeatmapData` interface so consumers don't have to change.

**Step 3:** In `src/services/api.ts`, rewrite `getHeatmap`:

```ts
export async function getHeatmap(videoId: string, playerId?: number): Promise<HeatmapResponse> {
  const cacheKey = `heatmap:${videoId}:${playerId ?? 'all'}`;
  const cached = heatmapCache.get(cacheKey) as HeatmapData | undefined;
  if (cached) return { video_id: videoId, player_id: playerId ?? null, heatmap: cached, status: 'success' };

  const { computeHeatmap } = await import("@/composables/useHeatmap");
  const heatmap = await computeHeatmap(videoId, playerId);
  heatmapCache.set(cacheKey, heatmap);
  return { video_id: videoId, player_id: playerId ?? null, heatmap, status: 'success' };
}

export async function preloadHeatmap(videoId: string): Promise<void> {
  try { await getHeatmap(videoId); } catch (e) { console.warn('[Heatmap] preload failed', e); }
}
```

**Step 4:** Repeat for zone analytics. Create `src/composables/useZoneAnalytics.ts` with `computeZoneAnalytics(videoId)` ported from `convex/http.ts /api/zone-analytics`. Rewrite `getRecalculatedZoneAnalytics` in `services/api.ts` to call it.

**Step 5:** Type-check.

**Step 6:** Commit.

```bash
git add src/composables/useHeatmap.ts src/composables/useZoneAnalytics.ts src/services/api.ts
git commit -m "feat(api): port heatmap + zone analytics to client-side composables"
```

---

### Task 35: Rewrite speed API to call Edge Function

**Files:**
- Modify: `src/services/api.ts`

**Step 1:** Replace `getSpeedData`:

```ts
export async function getSpeedData(
  videoId: string,
  windowSeconds: number = 60.0,
  forceRefresh: boolean = false,
): Promise<SpeedDataResponse> {
  const cacheKey = `speed:${videoId}:${windowSeconds}`;
  if (!forceRefresh) {
    const cached = speedCache.get(cacheKey);
    if (cached) return { /* same shape as before */ } as SpeedDataResponse;
  }

  const { data, error } = await supabase.functions.invoke("recalculate-speeds", {
    body: { video_id: videoId, window_seconds: windowSeconds },
  });
  if (error) throw error;

  const resp = data as SpeedDataResponse;
  if (resp.speed_data) speedCache.set(cacheKey, resp.speed_data);
  return resp;
}
```

**Step 2:** Replace `getSpeedTimeline` similarly — same Edge Function, same response, just different shape extraction. Or merge them into one call if the response covers both.

**Step 3:** Delete `recalculateSpeedsFromSkeleton` — that was the old `${API_BASE_URL}/api/speed/recalculate` Flask path. The Edge Function call covers it.

**Step 4:** `triggerSpeedRecalculation` stays — it just clears cache and calls `getSpeedData(videoId, 60, true)`.

**Step 5:** Type-check.

**Step 6:** Commit.

```bash
git add src/services/api.ts
git commit -m "feat(api): route speed calculations through recalculate-speeds Edge Function"
```

---

### Task 36: Rewrite PDF export to call Edge Function

**Files:**
- Modify: `src/services/api.ts`

**Step 1:** Replace `downloadPDFExport`:

```ts
export async function downloadPDFExport(
  videoId: string,
  options?: { frame_number?: number; include_heatmap?: boolean; heatmap_colormap?: string; heatmap_alpha?: number; },
): Promise<void> {
  const { data, error } = await supabase.functions.invoke("export-pdf", {
    body: {
      video_id: videoId,
      config: {
        frame_number: options?.frame_number,
        include_heatmap: options?.include_heatmap ?? true,
        heatmap_colormap: options?.heatmap_colormap ?? "turbo",
        heatmap_alpha: options?.heatmap_alpha ?? 0.6,
      },
    },
  });
  if (error) throw error;
  const result = data as { success: boolean; pdfBase64: string; filename?: string };
  if (!result.success || !result.pdfBase64) throw new Error("PDF generation failed");

  const binary = atob(result.pdfBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = result.filename || `badminton_analysis_${videoId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
```

**Step 2:** `exportPDFWithFrontendData` — same pattern: call `export-pdf` Edge Function with the frontend data in `body.config`. Reuse the download-blob code.

**Step 3:** Type-check. Should now have zero TS errors.

```bash
npm run type-check
```

**Step 4:** Commit.

```bash
git add src/services/api.ts
git commit -m "feat(api): route PDF export through export-pdf Edge Function"
```

---

### Task 37: Update `src/main.ts` and env files

**Files:**
- Modify: `src/main.ts`
- Modify: `.env.example`
- Create or modify: `.env.local` (gitignored)

**Step 1:** Strip convex from `src/main.ts`:

```ts
// New main.ts
import { createApp } from "vue";
import "./app.css";
import App from "./App.vue";
createApp(App).mount("#app");
```

**Step 2:** Update `.env.example`:

```
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

(Delete any `VITE_CONVEX_URL` and `VITE_API_URL` lines.)

**Step 3:** Tell the user (in plan output, not in code) to populate `.env.local` with their real values from `supabase/SETUP.md` step 2.

**Step 4:** Commit.

```bash
git add src/main.ts .env.example
git commit -m "chore(env): drop convex/flask env vars; document supabase vars"
```

---

## Phase 8 — Smoke test (cutover gate)

### Task 38: Run dev server, sign in, see empty state

**Files:** None — manual verification.

**Step 1:** Start dev server.

Run: `npm run dev`
Open: http://localhost:5173

**Step 2:** You should see the LoginView.

**Step 3:** Sign in with the admin user you created in `supabase/SETUP.md` step 5.

**Step 4:** App loads. No videos visible (correct — fresh DB). No console errors.

**Step 5:** Click sign out (or clear localStorage); LoginView reappears.

If any of those fail, stop and debug before continuing.

---

### Task 39: Upload a test video end-to-end

**Files:** None — manual verification.

**Step 1:** Sign in. Pick a short test video (~30s, MP4, ideally one you've used during dev).

**Step 2:** Upload via the UI. Confirm:
- File appears in Supabase Storage `videos/<your-uid>/<video_id>.mp4` (check via dashboard).
- A row appears in `videos` table with `status = 'uploaded'`.

**Step 3:** Trigger processing. Confirm:
- A POST goes to `/functions/v1/process-video` (check browser network tab).
- Status flips to `processing`.

**Step 4:** Watch progress. Confirm in real time (no refresh):
- Progress bar increments.
- Logs stream in.

**Step 5:** Wait for completion (~few minutes for a short video).

**Step 6:** Confirm `status = 'completed'`. Confirm:
- `results_storage_path` filled in `videos`.
- `results.json` exists at that path in the `results` bucket.
- `rally_clips` has rows for each detected rally.
- Each rally clip's `clip_storage_path` resolves to a playable MP4 in the `clips` bucket.
- Player thumbnails in `thumbnails` bucket.

**Step 7:** Open a rally clip's signed URL in the browser → it plays.

**Step 8:** Test heatmap, zone analytics, speed in the existing dashboard UI. All should work.

**Step 9:** Test PDF export. PDF downloads.

**Smoke test passes if all 9 steps green.** If any fails, stop and fix before Phase 9.

---

## Phase 9 — Convex removal (no leftovers)

### Task 40: Delete the Convex deployment

**Files:** None — dashboard action.

**Step 1:** Open the Convex dashboard.

**Step 2:** Settings → Delete project. Confirm.

**Step 3:** Done. Cannot be undone.

---

### Task 41: Delete `convex/` directory and dependencies

**Files:**
- Delete: `convex/` (entire directory)

**Step 1:** Remove the directory.

```bash
rm -rf convex/
```

**Step 2:** Verify package.json no longer has convex deps (Task 21 already removed them, but double-check):

```bash
grep -E '"(convex|convex-vue|@convex-vue)"' package.json
```

Expected: zero matches.

**Step 3:** If any matches: `npm uninstall convex convex-vue @convex-vue/core` and rerun.

**Step 4:** Reinstall to prune:

```bash
rm -rf node_modules package-lock.json
npm install
```

**Step 5:** Commit.

```bash
git add convex package.json package-lock.json
git commit -m "chore: delete convex/ directory and prune deps"
```

(Note: `git add convex` will add the deletion of the directory.)

---

### Task 42: Delete legacy backend processor + Modal app

**Files:**
- Delete: `backend/modal_convex_processor.py`

**Step 1:** Delete the file.

```bash
rm backend/modal_convex_processor.py
```

**Step 2:** Stop and delete the old Modal deployment from the Modal dashboard (the convex-processor app).

**Step 3:** Delete legacy Modal Secrets:

```bash
modal secret delete convex-secrets
```

(If `convex-secrets` doesn't exist, that's fine — already gone.)

**Step 4:** Commit.

```bash
git add backend/modal_convex_processor.py
git commit -m "chore: delete modal_convex_processor.py (replaced by supabase processor)"
```

---

### Task 43: Final grep sweep — must return zero

**Files:** None — verification.

**Step 1:** Search the repo.

```bash
grep -ri --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=venv --exclude-dir=docs/plans 'convex' .
```

**Step 2:** Expected: zero matches.

**Step 3:** If matches exist, list them, evaluate each:
- Code reference → delete or rewrite.
- Doc reference (other than design plan files) → delete or rewrite.
- Comment → delete.

**Step 4:** Re-run the grep until it returns zero.

**Step 5:** Allow in `docs/plans/`: the historical design + plan files mention Convex by design (they document the migration). They stay.

```bash
grep -ri --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=venv 'convex' . | grep -v 'docs/plans'
```

This must return zero.

**Step 6:** Commit any final cleanup.

```bash
git add -A
git commit -m "chore: scrub final Convex references (zero outside docs/plans)"
```

---

### Task 44: Open PR

**Files:** None — git operation.

**Step 1:** Push the branch.

```bash
git push -u origin feat/supabase-migration
```

**Step 2:** Open a PR to `main` with the design and plan as the description (or link to them).

**Step 3:** Done. Milestone 1 complete.

---

## Verification checklist (final)

- [ ] `supabase/migrations/` has 0001, 0002, 0003 — all applied to project
- [ ] `supabase/functions/` has process-video, recalculate-speeds, export-pdf — all deployed
- [ ] Modal apps `badminton-supabase-processor` and `badminton-pdf-export` deployed; URLs set as Edge Function secrets; old Modal apps deleted
- [ ] `MODAL_SHARED_SECRET` set in both Supabase Edge Function env and Modal Secrets (same value)
- [ ] `convex/` directory does not exist
- [ ] `package.json` contains `@supabase/supabase-js`, no `convex*` packages
- [ ] `.env.example` has `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`, no `VITE_CONVEX_*` or `VITE_API_URL`
- [ ] `grep -ri convex .` (excluding node_modules, .git, venv, docs/plans) returns zero matches
- [ ] Sign-in works on web; signing out + back in works
- [ ] Upload → process → playback → analytics → PDF export all work end-to-end
- [ ] Rally clips visible in `rally_clips` table and playable from `clips` bucket signed URLs
- [ ] Convex Vercel/Convex deployment deleted; no production traffic still hits it

---

## Out of scope for this plan

- KMP mobile app (Milestone 2 — separate plan after this is shipped)
- Per-rally thumbnails (deferred to mobile v2)
- Public sharing of clips (closed registration + private-by-default)
- Apple Sign-In (deferred until iOS submission)
- Supabase TUS-resumable uploads (deferred; supabase-js `upload` is fine for v1)

---

## Notes for the executor

- The Supabase project setup (`supabase/SETUP.md`) cannot be automated. **You must pause and wait for the user** between Task 5 and Task 6 until the user provides credentials.
- The Modal deploy steps (Task 19, Task 20 step 6) require the user's Modal account to be logged in via `modal token current`. If you're a subagent without those creds, **stop and surface this to the user**.
- All commits should be small and topical. Don't batch unrelated changes.
- The smoke test in Phase 8 is the only gate before Convex deletion. Take it seriously — investigate every red.
