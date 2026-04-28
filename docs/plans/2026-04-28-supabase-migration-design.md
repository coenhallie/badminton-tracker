# Supabase Migration + KMP Mobile — Design

**Status:** Approved (brainstormed 2026-04-28)
**Owner:** Coen
**Goal:** Migrate the badminton-tracker backend from Convex to Supabase, add closed-registration multi-user authentication, generate per-rally video clips during the existing Modal pipeline, and stand up a Kotlin Multiplatform mobile app (Android + iOS) that lets a signed-in user browse and play their own rally clips using the same credentials as the web app.

---

## 1. Background and motivation

The current stack is Vue 3 + Convex + Modal. Convex is used as a managed database, blob storage, HTTP ingress for Modal callbacks, and reactive backplane for the Vue frontend. There is no authentication; all data is effectively public.

The end goal that drove this design is a private rally clip library accessible from a separate Kotlin Multiplatform mobile app (Android + iOS) under the same login the web app uses.

The change of stack was evaluated explicitly. With the app staying as-is, Convex + Clerk would have been the right call. Once the future shape includes (a) KMP mobile, (b) per-user private clip storage, (c) Modal as the only writer of large results, three Convex value props (Vue + Convex Auth glue, manual JWT bridge for KMP, function-layer storage access checks) become ongoing complexity that Supabase eliminates structurally:

- `supabase-kt` is the official multiplatform SDK and covers Auth + DB + Storage + Realtime from `commonMain`. Convex has no KMP SDK.
- Supabase Storage RLS enforces "user can only read own files" at the storage layer.
- PostgREST + RLS turns the entire mobile read surface into zero custom backend code.
- Modal can write directly to Postgres + Storage with the service-role key, collapsing the ~430-line `convex/http.ts` callback layer to inline calls.

The migration is justified by the *future* shape of the app, not the current shape. Without KMP and per-user data, the migration would not be worth it.

## 2. Decisions log

- **Database / Backend:** Supabase (Postgres + Auth + Storage + Edge Functions + Realtime).
- **Auth model:** multi-user, private-by-default. Each video and each rally clip belongs to exactly one owner. No sharing in v1.
- **Registration:** closed. Public signup is disabled in Supabase. Users are created manually from the Supabase dashboard, which sends them an invite email to set a password.
- **Sign-in methods:** email + password (primary), Google OAuth. Apple Sign-In deferred until iOS App Store submission.
- **Existing data:** wiped. No migration of current Convex video records. Fresh Supabase project, fresh tables.
- **ML pipeline accuracy:** unaffected. Migration touches transport and persistence only; YOLO26 / TrackNet / rally detection / homography / video bytes are unchanged.
- **Modal ingress:** one Edge Function (`/process-video`) that verifies user ownership and signs requests to Modal. Modal verifies an HMAC of the body using a shared secret. Modal writes back to Supabase directly via the Python client with the service-role key. No HTTP callback router.
- **Real-time UX:** Supabase Realtime row-level subscriptions reconcile on the client. Three thin Vue composables replace `convex-vue`.
- **Rally clip generation:** runs at the end of the existing Modal pipeline, after rally detection, before `status = 'completed'` flips. ffmpeg `-c copy` (stream copy) per rally.
- **Per-rally thumbnails:** deferred to a v2 of the mobile app.
- **Mobile UI:** Compose Multiplatform (shared UI on Android + iOS) for v1.
- **Mobile v1 surface:** login + clip list + clip player + sign-out. No upload, no analytics, no editing.
- **Cutover:** single sitting. Smoke test gate, then Convex deleted in the same session — no buffer, no fallback code, no leftovers.

## 3. Architecture overview

```
┌─────────────────────────┐         ┌─────────────────────────┐
│  Vue web app (Vite)     │         │  KMP mobile app         │
│  - supabase-js          │         │  Android + iOS          │
│  - login / upload /     │         │  - supabase-kt          │
│    process / view       │         │  - login / list /       │
│  - existing analytics   │         │    play clips           │
└──────────┬──────────────┘         └──────────┬──────────────┘
           │ JWT (Supabase Auth)               │ JWT (Supabase Auth)
           ▼                                   ▼
   ┌───────────────────────────────────────────────────┐
   │                    Supabase                       │
   │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
   │  │ Postgres     │  │ Storage      │  │ Auth    │  │
   │  │ - users      │  │ - videos/    │  │ - email │  │
   │  │ - videos     │  │ - clips/     │  │ - OAuth │  │
   │  │ - logs       │  │ - thumbs/    │  │         │  │
   │  │ - rally_clips│  │ (RLS-scoped) │  │         │  │
   │  │ + RLS        │  │              │  │         │  │
   │  └──────┬───────┘  └──────┬───────┘  └─────────┘  │
   │         │                 │                       │
   │  ┌──────┴─────────────────┴─────┐                 │
   │  │ Edge Functions (Deno)        │                 │
   │  │ - /process-video (start job) │                 │
   │  └──────────────┬───────────────┘                 │
   └─────────────────┼─────────────────────────────────┘
                     │ POST start (HMAC-signed body)
                     ▼
              ┌─────────────────────┐
              │  Modal (GPU)        │
              │  - YOLO + TrackNet  │
              │  - rally detection  │
              │  - ffmpeg clip cut  │
              │                     │
              │  writes back via:   │
              │  - supabase Python  │
              │    client (SERVICE  │
              │    ROLE key) →      │
              │    DB + Storage     │
              │    direct           │
              └─────────────────────┘
```

Key shifts vs. today:
1. Modal stops calling 11 HTTP endpoints; it uses the Supabase Python client directly. `convex/http.ts` (~430 lines) disappears.
2. Mobile reads via PostgREST with `supabase-kt`; no custom mobile API.
3. One Edge Function for the trigger is the only custom backend code.
4. Real-time uses Supabase Realtime row events; client reconciles.
5. Storage path embeds owner UID (`{uid}/{video_id}/...`); storage RLS enforces ownership without DB lookup.

What disappears: `convex/`, `convex-vue`, `@convex-vue/core`, `convex/http.ts`, `_generated/`, `manualCourtKeypoints` mutation chain, the 4096-read-cap log-deletion workaround, manual signed-URL generation in queries.

What gets added: `supabase/migrations/`, `supabase/functions/`, `src/lib/supabase.ts`, three composables, login view, route guard, `mobile/` Gradle project root.

## 4. Data model

### 4.1 Tables

```sql
-- Supabase Auth provides auth.users automatically.
-- We don't create a separate profiles table — auth.uid() is enough for ownership.

create table public.videos (
  id              uuid primary key default gen_random_uuid(),
  owner_id        uuid not null references auth.users(id) on delete cascade,
  filename        text not null,
  size            bigint not null,
  storage_path    text not null,                    -- "<uid>/<video_id>.mp4" in 'videos' bucket

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

Three deliberate choices:

- **`owner_id` denormalized** onto `processing_logs` and `rally_clips` so RLS policies are `using (owner_id = auth.uid())` — single index lookup, no join. Set once at insert, never changes.
- **`on delete cascade` everywhere.** Real referential integrity that Convex couldn't enforce.
- **Storage paths as plain text** (not opaque IDs). Owner UID is the first segment; storage RLS enforces ownership without a DB round-trip.

### 4.2 Storage buckets

All four buckets are private (no anon read), all enforce RLS on `(storage.foldername(name))[1] = auth.uid()::text`:

| Bucket       | Path layout                              | Contents                          |
|---           |---                                       |---                                |
| `videos`     | `{uid}/{video_id}.mp4`                   | original uploads                  |
| `results`    | `{uid}/{video_id}/results.json`          | full results JSON written by Modal|
| `clips`      | `{uid}/{video_id}/rally_{n}.mp4`         | per-rally clips                   |
| `thumbnails` | `{uid}/{video_id}/player_{0,1}.jpg`      | player thumbnails                 |

Modal uses the service-role key, which bypasses RLS — fine because Modal is trusted backend infrastructure.

### 4.3 RLS policy shape

```sql
alter table public.videos          enable row level security;
alter table public.processing_logs enable row level security;
alter table public.rally_clips     enable row level security;

-- Same shape on all three: owner_id = auth.uid() for all CRUD.
create policy "owner_select" on public.videos for select using (owner_id = auth.uid());
create policy "owner_insert" on public.videos for insert with check (owner_id = auth.uid());
create policy "owner_update" on public.videos for update using (owner_id = auth.uid());
create policy "owner_delete" on public.videos for delete using (owner_id = auth.uid());
-- (mirror for processing_logs and rally_clips; mirror as bucket policies for storage)
```

## 5. Auth flow

Supabase Auth issues a JWT; Postgres RLS, Storage RLS, and the Edge Function all derive `auth.uid()` from that JWT. Modal is the only thing that doesn't carry a user JWT — it uses the service-role key.

### 5.1 Sign-in methods

Enabled in the Supabase dashboard:

- **Email + password** — primary, identical experience on Vue, Android, iOS via supabase-kt.
- **Google OAuth** — one-tap on web and Android. Dashboard config + Google Cloud OAuth client ID/secret.
- **Apple Sign-In** — deferred until iOS App Store submission. Required by App Store Review Guideline 4.8 if any third-party login is offered on iOS.

Magic link and passkeys explicitly skipped for v1.

### 5.2 Registration

Public signup is **disabled** in Auth → Providers → Email. Users are created manually from the Supabase dashboard via Auth → Users → "Add user" → "Send invite." Supabase emails them an invite link to set a password.

No invite-code table, no waitlist, no admin UI.

### 5.3 Session handling

**Vue web (`supabase-js`):** sessions persist in `localStorage`; refresh handled by the SDK. A `useSession()` composable exposes `session`, `user`, `isAuthenticated`. Route guard wrapper renders `<LoginView />` when no session.

**KMP mobile (`supabase-kt`):** sessions persist via `Settings` (multiplatform key-value, backed by `SharedPreferences` on Android and `NSUserDefaults` on iOS — auto-wired by supabase-kt). Refresh handled by the SDK. Auth code lives in `commonMain`.

### 5.4 JWT propagation table

| Call site                                            | How auth attaches                                      | Who validates                                    |
|---                                                   |---                                                     |---                                               |
| Vue → Postgres                                       | `supabase-js` attaches `Authorization: Bearer <jwt>`   | PostgREST verifies; RLS scopes to `auth.uid()`   |
| Vue → Storage                                        | Same JWT                                               | Storage RLS checks first path segment            |
| Vue → Edge Function `/process-video`                 | Same JWT                                               | Edge Fn calls `supabase.auth.getUser(jwt)`       |
| KMP mobile → Postgres / Storage                      | `supabase-kt` attaches automatically                   | Same as web                                      |
| Edge Function → Modal                                | `Authorization: Bearer <SHARED_SECRET>` + body HMAC    | Modal verifies HMAC against `MODAL_SHARED_SECRET`|
| Modal → Postgres / Storage                           | Service-role key in environment                        | Bypasses RLS; trusted infrastructure             |

### 5.5 Edge Function: `/process-video`

The only piece of custom auth-aware backend code, ~40 lines of Deno.

```ts
// supabase/functions/process-video/index.ts (sketch)
serve(async (req) => {
  const jwt = req.headers.get("Authorization")?.replace("Bearer ", "");
  const { data: { user }, error } = await supabase.auth.getUser(jwt);
  if (error || !user) return new Response("Unauthorized", { status: 401 });

  const { video_id } = await req.json();

  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video } = await userClient.from("videos")
    .select("*").eq("id", video_id).single();
  if (!video) return new Response("Not found", { status: 404 });

  const body = JSON.stringify({
    video_id, owner_id: user.id,
    video_url: signedUrlFor(video.storage_path),
  });
  await fetch(MODAL_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Signature": hmac(body, SHARED_SECRET),
    },
    body,
  });

  return new Response(JSON.stringify({ ok: true }));
});
```

Why an Edge Function instead of letting clients call Modal directly: keep the shared secret out of any client bundle, single trusted place to verify ownership before kicking off paid GPU work, single place for future rate limiting.

## 6. Modal pipeline rewrite

### 6.1 Conceptual change

`convex/http.ts` (~430 lines of route handlers) and the corresponding `requests.post(...)` callback code in Modal both disappear. Modal speaks Supabase directly using the Python `supabase` client with the service-role key.

Example callback before/after:

```python
# BEFORE — current Convex
requests.post(
    f"{callback_url}/updateStatus",
    json={"videoId": video_id, "progress": pct,
          "currentFrame": frame, "totalFrames": total},
)
```

```python
# AFTER — Supabase
supabase.table("videos").update({
    "progress": pct,
    "current_frame": frame,
    "total_frames": total,
}).eq("id", video_id).execute()
```

### 6.2 Full callback surface mapping

| Today's call                           | After                                                                                |
|---                                     |---                                                                                   |
| `POST /updateStatus`                   | `supabase.table("videos").update({...}).eq("id", video_id)`                          |
| `POST /addLog`                         | `supabase.table("processing_logs").insert({video_id, owner_id, ...})`                |
| `POST /updateResults`                  | `supabase.table("videos").update({status, results_meta, ...}).eq(...)`               |
| Two-step upload (`/generateUploadUrl` → PUT → `/setPlayerThumbnails`) | `supabase.storage.from("thumbnails").upload(...)` then update `videos.player_labels` |
| Signed-URL fetch of source video       | `supabase.storage.from("videos").create_signed_url(path, ttl=3600)`                  |

### 6.3 Modal entry handler

Modal's HTTP entry verifies one HMAC and reads the request body:

```python
# backend/modal_supabase_processor.py — sketch
@app.function(secrets=[Secret.from_name("supabase"), Secret.from_name("modal-shared-secret")])
@web_endpoint(method="POST")
def process(request: Request):
    body = request.body()
    signature = request.headers.get("X-Signature")
    if not hmac.compare_digest(signature, hmac_sha256(body, MODAL_SHARED_SECRET)):
        return {"error": "unauthorized"}, 401

    payload = json.loads(body)
    return run_pipeline.spawn(payload["video_id"], payload["owner_id"], payload["video_url"])
```

### 6.4 Modal Secrets

Set in the Modal dashboard or via `modal secret create`:

- `SUPABASE_URL` — same project URL.
- `SUPABASE_SERVICE_ROLE_KEY` — full DB+Storage access, bypasses RLS. Never goes client-side.
- `MODAL_SHARED_SECRET` — HMAC verifier for entry handler.

Existing Convex-related Modal Secrets (`CONVEX_SITE_URL`, etc.) are deleted at cutover.

### 6.5 What stays unchanged inside Modal

- All ML inference (`backend/modal_inference.py`, YOLO + TrackNet weights, court detection).
- `backend/rally_detection.py` — same rally objects.
- `/cache/{video_id}.mp4` working-directory pattern.
- Frame-loop progress reporting (only the destination of the report changes).

### 6.6 What's new inside Modal

- ffmpeg clip-cutting step at the end of the pipeline (Section 7).
- HMAC verifier on the entry handler.
- Per-batch log buffering (optional optimization — `processing_logs` inserts batched every N frames).

## 7. Rally clip generation

### 7.1 Pipeline placement

```
Modal /process-video entry (HMAC verified)
  ↓
download original video → /cache/{video_id}.mp4
  ↓
YOLO + TrackNet inference loop (existing)
  ↓
rally_detection.detect(...) → rallies = [...]
  ↓
upload results.json to Storage         (existing)
update videos.results_meta             (existing)
  ↓
NEW: cut_and_upload_rally_clips(...)
  ↓
update videos.status = 'completed'
cleanup /cache/{video_id}*
```

The clip step runs after `results_meta` is set but before `status = 'completed'` flips. Frontend sees status flip to completed only when clips are also ready — no half-state.

### 7.2 ffmpeg invocation

```bash
ffmpeg -y \
  -ss <start_seconds>      \
  -to <end_seconds>        \
  -i /cache/{video_id}.mp4 \
  -c copy                  \
  -avoid_negative_ts make_zero \
  -movflags +faststart     \
  /cache/{video_id}_rally_{n}.mp4
```

- `-c copy` (stream copy): ~0.5–2s per clip vs. ~10–60s if re-encoding. No quality loss. Trade-off: clips may start up to ~2s before the requested timestamp because cuts happen at the nearest preceding keyframe — desirable as a lead-in for rally review.
- `-movflags +faststart`: moves moov atom to front so mobile players can begin playback before the full file is downloaded. Critical for KMP playback experience.

For a 10-minute video with ~25 rallies: adds ~30–60 seconds to total processing time. CPU + disk + upload bandwidth only; no GPU cost.

### 7.3 Storage + DB write per clip

```python
# backend/modal_supabase_processor.py — new function
def cut_and_upload_rally_clips(video_path, rallies, video_id, owner_id, supabase):
    for rally in rallies:
        clip_path = f"/cache/{video_id}_rally_{rally['id']}.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(rally["start_timestamp"]),
                "-to", str(rally["end_timestamp"]),
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                clip_path,
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            log_to_supabase(video_id, owner_id,
                f"clip generation failed for rally {rally['id']}: {e.stderr.decode()[:200]}",
                level="warning", category="processing")
            continue

        storage_path = f"{owner_id}/{video_id}/rally_{rally['id']}.mp4"
        with open(clip_path, "rb") as f:
            supabase.storage.from_("clips").upload(
                storage_path, f.read(),
                {"content-type": "video/mp4", "upsert": "true"},
            )

        supabase.table("rally_clips").upsert({
            "video_id":          video_id,
            "owner_id":          owner_id,
            "rally_index":       rally["id"],
            "start_timestamp":   rally["start_timestamp"],
            "end_timestamp":     rally["end_timestamp"],
            "duration_seconds":  rally["duration_seconds"],
            "clip_storage_path": storage_path,
        }, on_conflict="video_id,rally_index").execute()

        os.remove(clip_path)
```

### 7.4 Idempotency and partial-failure behavior

- `unique (video_id, rally_index)` + `upsert` + `upsert: "true"` on storage means re-running the pipeline overwrites cleanly.
- A single bad rally (e.g., end_timestamp past actual video duration) logs a warning and continues. The rally is still in `results_data["rallies"]` (so the existing in-page timeline works) but missing from `rally_clips` (so it won't appear in mobile). Soft failure surfaced to the user via the log stream.

### 7.5 Mobile read pattern

```kotlin
// KMP commonMain — list my rally clips, newest first
val clips = supabase.from("rally_clips")
    .select { order("created_at", Order.DESCENDING) }
    .decodeList<RallyClip>()

// Play one clip — signed URL good for 1 hour
val signedUrl = supabase.storage.from("clips")
    .createSignedUrl(clip.clipStoragePath, expiresIn = 3600.seconds)
```

RLS scopes the list automatically. Two lines, no custom backend.

### 7.6 Web app behavior unchanged

`RallyTimeline.vue` continues to read rallies from results JSON and seek within the single original video. Clips are an addition for the mobile app, not a replacement for the existing in-page UX.

## 8. Frontend rewrite

### 8.1 Library swap

| Out                  | In                                                     |
|---                   |---                                                     |
| `convex`             | `@supabase/supabase-js`                                |
| `convex-vue`         | (none — three thin Vue composables)                    |
| `@convex-vue/core`   | (delete)                                               |

`supabase-js` is framework-agnostic. No first-party Vue wrapper needed.

### 8.2 New files

**`src/lib/supabase.ts`** (~15 lines):

```ts
import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY,
  { auth: { persistSession: true, autoRefreshToken: true } }
);
```

**`src/composables/useSession.ts`** (~30 lines): exposes `session`, `user`, `isAuthenticated`, subscribes to `onAuthStateChange`.

**`src/composables/useReactiveRow.ts`** (~40 lines): replaces single-row `useConvexQuery`. Initial fetch + Realtime UPDATE subscription on `id=eq.<id>` filter.

**`src/composables/useReactiveList.ts`** (~50 lines): replaces list `useConvexQuery`. Initial fetch with order + Realtime INSERT subscription on configurable column-equality filter.

**`src/views/LoginView.vue`** (~60 lines): email + password form, Google button. No signup link.

### 8.3 Component diff (representative)

```ts
// BEFORE
const { data: videoData } = useConvexQuery(
  api.videos.getVideo, computed(() => ({ videoId: convexVideoId.value }))
);
const { data: logsData } = useConvexQuery(
  api.videos.getProcessingLogs, computed(() => ({ videoId: convexVideoId.value }))
);

// AFTER
const { row: video } = useReactiveRow<Video>("videos", videoId);
const { items: logs } = useReactiveList<ProcessingLog>(
  "processing_logs",
  computed(() => ({ column: "video_id", value: videoId.value }))
);
```

Identical shape. Every `useConvexQuery` callsite translates this directly.

### 8.4 Route guard

App-level `v-if` on `isAuthenticated` shows `<LoginView />` when no session. ~10 lines.

### 8.5 Explicitly not built

- No SSR, no Nuxt, no `@supabase/auth-helpers`. The app is an SPA.
- No client-side query cache layer. supabase-js doesn't ship one and the existing components don't need it.
- No Pinia/Vuex for auth. The `useSession` composable is module-scope; Vue 3 composables share state across consumers when defined at module scope.

## 9. KMP mobile app

Mobile is built **after** Milestone 1 (web migration). This section locks in the architecture so Milestone 1 doesn't paint mobile into a corner.

### 9.1 Project layout

A separate Gradle project at `mobile/`, sibling to the Vue app:

```
mobile/
├── build.gradle.kts
├── gradle/libs.versions.toml
├── shared/                 ← KMP module
│   └── src/
│       ├── commonMain/     ← 80%+ of code
│       │   ├── data/       ← supabase-kt client, repositories
│       │   ├── domain/     ← models, use cases
│       │   └── ui/         ← Compose Multiplatform screens
│       ├── androidMain/    ← VideoPlayer (ExoPlayer/media3)
│       └── iosMain/        ← VideoPlayer (AVPlayer)
├── androidApp/             ← Android entry point + manifest
└── iosApp/                 ← Xcode project + SwiftUI host (one screen)
```

### 9.2 UI: Compose Multiplatform

Compose Multiplatform (stable on iOS as of Compose 1.6) for shared UI. Trades "perfectly native iOS feel" for one shared UI codebase. Right trade for v1; `data/` and `domain/` layers stay reusable if iOS UI is later rebuilt natively.

### 9.3 Dependencies

```toml
[versions]
kotlin = "2.1.0"
compose-multiplatform = "1.7.0"
supabase = "3.0.0"
ktor = "3.0.0"
media3 = "1.4.1"

[libraries]
supabase-postgrest = { module = "io.github.jan-tennert.supabase:postgrest-kt", version.ref = "supabase" }
supabase-storage   = { module = "io.github.jan-tennert.supabase:storage-kt",   version.ref = "supabase" }
supabase-auth      = { module = "io.github.jan-tennert.supabase:auth-kt",      version.ref = "supabase" }
ktor-client-okhttp = { module = "io.ktor:ktor-client-okhttp",                  version.ref = "ktor" }
ktor-client-darwin = { module = "io.ktor:ktor-client-darwin",                  version.ref = "ktor" }
```

### 9.4 expect/actual surface

The only `expect`/`actual` written by hand is `VideoPlayer` — there's no cross-platform video player primitive:

- `androidMain`: ExoPlayer (androidx.media3) wrapped in a Compose `AndroidView`.
- `iosMain`: `AVPlayer` wrapped in a Compose `UIKitView`.

~80 lines each side.

### 9.5 v1 screens

| Screen             | LOC est. | Lives in                              |
|---                 |---       |---                                    |
| Login              | ~100     | commonMain                            |
| Clip list          | ~120     | commonMain                            |
| Clip player        | ~200     | commonMain (controls) + platform (player surface) |
| Settings/profile   | ~30      | commonMain                            |

### 9.6 iOS quirks (deferred to submission time)

- Apple Sign-In required by Guideline 4.8 if any third-party login is on iOS. Easiest: enable Apple on iOS only via build flag.
- Privacy manifest (`PrivacyInfo.xcprivacy`) declaring Supabase Auth's email collection.
- Apple Developer Program enrollment ($99/year) for TestFlight + submission.

### 9.7 Distribution v1

- Android: Google Play Internal Testing track or sideload to own device.
- iOS: TestFlight against Apple Developer enrollment.

Public store listings deferred indefinitely; internal/TestFlight is enough for personal use.

## 10. Cutover plan

### Phase 1 — Stand up Supabase (no traffic yet)

Done one-time in the Supabase dashboard:

1. Create the project. Copy `SUPABASE_URL`, `ANON_KEY`, `SERVICE_ROLE_KEY` to a secure store.
2. Disable public signups in Auth → Providers → Email. Enable email confirmation. Configure Google OAuth (Google Cloud OAuth client).
3. Run the Section 4 SQL migration (tables, indexes, RLS policies). SQL editor or `supabase db push`.
4. Create the four storage buckets (private) with the path-prefix RLS policies.
5. Create your admin user via Auth → Users → "Add user" → "Send invite." Set password from invite email.

### Phase 2 — Build new code in parallel (Convex still running)

Done on a feature branch. Nothing deployed yet:

1. Add `@supabase/supabase-js` to `package.json`. Build `src/lib/supabase.ts`, three composables, `LoginView.vue`, route guard.
2. Rewrite consumers of `useConvexQuery` → `useReactiveRow` / `useReactiveList`.
3. `backend/modal_supabase_processor.py` — rewrite of the processor against the Supabase Python client. ffmpeg clip-cutting step included.
4. `supabase/functions/process-video/` Edge Function.
5. Local dev: hit Supabase from the dev branch, exercise upload → process → playback end-to-end with one test video under your admin account.

### Phase 3 — Cutover + complete Convex removal (single sitting)

In order:

1. Deploy new Modal app. Add Modal Secrets `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `MODAL_SHARED_SECRET`.
2. Deploy Edge Function: `supabase functions deploy process-video`.
3. Deploy new Vue frontend with `VITE_SUPABASE_URL` + `VITE_SUPABASE_ANON_KEY`. Old `VITE_CONVEX_URL` removed.
4. **Smoke test gate:** upload one test video, watch progress + logs, confirm `results.json` lands in Storage, confirm `rally_clips` rows appear, confirm clips play in the browser. **Stop and fix before continuing if anything fails.**
5. Once green, immediately:
   - Delete the Convex deployment.
   - Delete the `convex/` directory entirely (including `_generated/`).
   - Remove `convex`, `convex-vue`, `@convex-vue/core` from `package.json`; reinstall to prune the lockfile.
   - Delete `backend/modal_convex_processor.py`.
   - Delete the old Modal app deployment (the Convex-targeted one).
   - Delete Modal Secrets `CONVEX_SITE_URL` and any other `CONVEX_*` entries.
   - Delete `VITE_CONVEX_URL` from `.env*` files.
   - `grep -ri convex .` from the repo root must return zero matches.
6. Commit: `feat: migrate from Convex to Supabase`. Single squash commit.

### Decommission checklist

| Location                              | Verify gone                                                          |
|---                                    |---                                                                   |
| `convex/`                             | Directory deleted entirely                                           |
| `convex/_generated/`                  | Gone                                                                 |
| `package.json`                        | No `convex`, `convex-vue`, `@convex-vue/core`                        |
| Lockfile                              | No transitive `convex` packages                                      |
| `src/main.ts`                         | No `convexVue` import or `app.use(convexVue)`                        |
| `src/**/*.ts`, `src/**/*.vue`         | No `useConvexQuery`, `api.*`, `convex/react`, `convex-vue` imports   |
| `.env*`                               | No `VITE_CONVEX_*`                                                   |
| `backend/`                            | No `modal_convex_processor.py`; no `requests.post(...callback...)`   |
| Modal dashboard                       | Old Convex-targeted app deleted; `CONVEX_*` Secrets deleted          |
| Convex dashboard                      | Project deleted                                                      |
| Repo grep                             | `grep -ri convex .` returns zero matches                             |

### Rollback

After Phase 3, rollback = `git revert` + redeploy frontend, plus rebuild Convex from the reverted code. Costly. The smoke test gate in step 4 is the only gate.

The class of bugs that requires multi-session real-world usage to surface (specific video formats, edge-case rally counts, etc.) will be discovered in production and fixed forward. Worst case is a temporarily broken upload; data is durable in Postgres + Storage.

## 11. Build sequencing

Two milestones, each a complete testable end-state.

### Milestone 1 — Foundation + Rally Clips (web complete)

Sections 4 through 8 + 10. Built on a feature branch, cut over in one sitting.

Order:

1. Supabase project setup (Section 10 Phase 1).
2. Modal-side rewrite (Section 6) — `backend/modal_supabase_processor.py`.
3. Rally clip generation (Section 7) — ffmpeg cutting added to the same processor file. Test against a real video on Modal.
4. Edge Function (Section 5.5) — `process-video` deployed.
5. Frontend rewrite (Section 8) — supabase-js client, three composables, login view, route guard, every `useConvexQuery` callsite swapped.
6. Smoke test + Convex deletion (Section 10 Phase 3).

End state: signed-in users can upload videos on the web, watch live progress, view rallies in the existing timeline UI, and rally clips exist as separate MP4s in Storage with `rally_clips` rows. No mobile app yet, but the data the mobile app needs is fully populated.

Estimate (single developer): 1–2 weeks of focused work.

### Milestone 2 — KMP Mobile App

Section 9. Separate Gradle project at `mobile/`, no impact on the web app.

Order:

1. KMP project scaffold + Compose Multiplatform setup. "Hello world" running on Android emulator and iOS simulator.
2. supabase-kt integrated, login screen working against the same Supabase project. Verify same credentials log in on web AND mobile.
3. Clip list screen — `supabase.from("rally_clips").select(...)` with RLS. Confirm own-clips-only.
4. Clip player — `expect/actual` `VideoPlayer`, signed Storage URLs, play.
5. Sign-out + settings.
6. Apple Sign-In (iOS-only), Apple Developer enrollment, TestFlight build — when ready to share.

End state: Android + iOS apps installable to your devices. Browse and play your own rally clips on the go.

Estimate (single developer): 2–3 weeks if Compose Multiplatform / KMP are new; 1–2 weeks if familiar.

## 12. Out of scope / deferred

| Item                                       | Reason                                                                 |
|---                                         |---                                                                     |
| Per-rally thumbnails on mobile             | v2 of mobile after seeing how the bare list view feels                 |
| In-mobile analytics views (charts, heatmaps) | Out of scope for v1 mobile; web app keeps these                       |
| Video uploading from mobile                | Out of scope for v1                                                    |
| Sharing clips between users                | Closed registration + private-by-default explicitly rules this out for v1 |
| Migration of existing Convex video data    | Wiped per decision                                                     |
| Magic link, passkeys                       | Email + password covers it; revisit if needed                          |
| Apple Sign-In on iOS                       | Add when packaging for iOS App Store, not before                       |
| Adaptive bitrate / HLS for clips           | Source MP4 + faststart is fine for mobile playback                     |
| SSR / Nuxt / `@supabase/auth-helpers`      | App is SPA                                                             |
| Pinia/Vuex for auth                        | Module-scope composable shares state                                   |

## 14. Analytics endpoints migration (extension to Sections 5–8)

Discovered during plan-writing: `convex/http.ts` is ~1577 lines and `src/services/api.ts` is ~1434 lines. Beyond the Modal callbacks already covered, there are **server-side compute endpoints** consumed by the frontend that the earlier sections did not address. They must migrate too — no Convex leftovers means no leftovers from these consumers either.

### 14.1 Endpoints in scope

| Endpoint family                           | Consumer in `src/services/api.ts`                                                            | Today's compute location              |
|---                                        |---                                                                                           |---                                    |
| Heatmap rendering                         | `getHeatmap`, `preloadHeatmap`                                                               | Convex httpAction (TS)                |
| Zone analytics                            | `getRecalculatedZoneAnalytics`                                                               | Convex httpAction (TS)                |
| Speed recalculation w/ homography         | `getSpeedData`, `getSpeedTimeline`, `triggerSpeedRecalculation`, `recalculateSpeedsFromSkeleton` | Convex httpAction (TS)            |
| PDF export                                | `downloadPDFExport`, `exportPDFWithFrontendData`                                             | Convex httpAction → Modal (`modal_pdf_export.py`) |
| Manual court keypoints status/set         | `getManualKeypointsStatus`, `setManualCourtKeypoints`                                        | Convex httpAction (trivial DB writes) |
| Health check, video-url fetch             | `checkApiHealth`, `getApiHealthDetails`, `fetchVideoUrl`                                     | Convex httpAction (trivial)           |

### 14.2 Migration target — Option D (mixed)

| Endpoint                          | Migration target                                                                                             |
|---                                |---                                                                                                           |
| Heatmap                           | **Client-side** Vue composable. Port the existing TS implementation from `convex/http.ts` into the browser. Reads results JSON via signed URL, computes the 2D intensity grid in memory. |
| Zone analytics                    | **Client-side** Vue composable. Same approach — port TS math to browser.                                     |
| Speed (recalc with homography)    | **New Modal endpoint** behind Edge Function. The math is heavier and benefits from the GPU container's memory headroom. Edge Function HMAC-signs the request, Modal reads results from Supabase Storage, computes, returns JSON. |
| PDF export                        | **Edge Function → existing `modal_pdf_export.py`**. Update `modal_pdf_export.py` to verify HMAC, read inputs from Supabase Storage, write the PDF blob back via Supabase Storage (or return base64 inline if small). |
| Manual court keypoints            | **Direct `supabase.from("videos").update({manual_court_keypoints: ...})`** from the frontend. No server endpoint needed — it's a JSONB write subject to RLS. JSON encoding preserves float precision losslessly; accuracy unchanged. |
| Health check                      | **Delete.** No analog needed — Supabase exposes its own status; if needed, the frontend can ping `supabase.from("videos").select("id").limit(1)`. |
| Video-url fetch                   | **Delete.** Replaced by `supabase.storage.from("videos").createSignedUrl(path)` called inline where needed.   |

### 14.3 Local Flask backend removal

`src/services/api.ts` contains dual-mode code: `if (USE_CONVEX) { ... } else { fetch(API_BASE_URL/...) }`. The local Flask backend referenced by `API_BASE_URL` no longer exists in the repo (`backend/` contains only Modal-related Python files: `modal_convex_processor.py`, `modal_inference.py`, `modal_pdf_export.py`, `rally_detection.py`, `tracknet/`, `upload_tracknet.py`). The fallback paths are dead code.

**Action:** delete every `API_BASE_URL` branch and the `VITE_API_URL` env var from `.env*`. After cutover, all backend traffic goes through Supabase or through Edge Function → Modal.

### 14.4 New Modal endpoints introduced

Two new Modal HTTP entry points, both gated by HMAC verification of the body using `MODAL_SHARED_SECRET`:

- `recalculate_speeds` — input: `{video_id, owner_id, results_storage_path}`. Reads `results.json` from Supabase Storage with the service-role key, recomputes speeds with homography, returns the speed payload as JSON. Lives in a new module `backend/modal_speed.py` or as a new function in `modal_supabase_processor.py`.
- `export_pdf` — input: `{video_id, owner_id, config}`. Already exists as `modal_pdf_export.py`; adapt to read from Supabase Storage and verify HMAC.

The Edge Function `process-video` is extended to be a router for these calls — `/process-video`, `/recalculate-speeds`, `/export-pdf` — or split into three Edge Functions. Implementation choice deferred to plan; routing is straightforward either way.

### 14.5 Schema impact

None. `results_storage_path` already covers the input data location; `manual_court_keypoints` is already on `videos` as JSONB. No table additions needed.

### 14.6 Frontend impact

`src/services/api.ts` is rewritten end-to-end:
- All `USE_CONVEX` branching removed.
- All `API_BASE_URL` branching removed.
- Heatmap and zone analytics functions become local computations (or move into composables under `src/composables/`).
- Speed and PDF functions call Edge Function endpoints with the user's JWT.
- Manual keypoints functions become 2-line `supabase.from("videos")` calls.

Estimated reduction: ~1434 lines → ~300 lines.

---

## 13. Risks and trade-offs

| Risk                                                                                              | Acceptance / mitigation                                                                                        |
|---                                                                                                |---                                                                                                             |
| No buffer between cutover and Convex deletion                                                     | Explicit user choice. Forward-fix any post-cutover bugs.                                                       |
| Single smoke test as the gate                                                                     | Personal project; manual smoke is the right level. Multi-session bugs fixed forward.                           |
| Stream-copy clip cuts may start ~2s before requested timestamp                                    | Acceptable for rally review; arguably desirable as a lead-in. Re-encoding ruled out as too costly.             |
| Supabase Realtime is row-level; clients reconcile vs. Convex's query-level reactivity             | Small downgrade for this app's current shape; reconciliation is ~20 lines per consumer.                        |
| Compose Multiplatform on iOS is younger than UIKit/SwiftUI                                        | Stable since Compose 1.6 (2024). For a personal video browser, the shared-UI win outweighs the maturity gap.   |
| Modal service-role key is a high-trust credential                                                 | Stored only as a Modal Secret, never in client bundles. Rotated if leaked.                                     |
| `convex-vue` removal touches every component using `useConvexQuery`                               | Mechanical swap to `useReactiveRow` / `useReactiveList`; identical call shape.                                 |
