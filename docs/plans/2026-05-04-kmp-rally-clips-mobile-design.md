# Kotlin Multiplatform Rally Clips Mobile App — Design

Date: 2026-05-04
Status: Approved (brainstorming complete, ready for implementation plan)
Repo: separate from `badminton-tracker` (different toolchain)

## 1. Goal

A Kotlin Multiplatform mobile app for Android + iOS that lets the existing
badminton-tracker user browse rally clips on the go, watch them, edit clip
titles, and add or delete time-pinned annotations. Read-mostly companion to
the web app — no upload, no processing.

## 2. Scope

**In scope (v1):**
- Email/password + Google sign-in (matches web).
- Clip list across all videos, newest first.
- Clip detail with native player (ExoPlayer / AVPlayer), playback speed,
  scrub bar with annotation markers.
- Inline title editing.
- Add and delete annotations. No edit (annotations are short).
- Pull-to-refresh + refresh-on-foreground.

**Out of scope (deferred):**
- Video upload from mobile.
- Realtime subscriptions (clip list changes only when Modal completes a
  job — pull-to-refresh is sufficient).
- Sign in with Apple — defer until App Store submission requires it.
- Offline playback / "save for offline".
- Crashlytics / Sentry.
- Sharing or social features.

## 3. Architecture

**Approach A — thin shared layer, native UIs.** The Kotlin `shared` module
exposes data models, the Supabase client factory, session persistence, and
repository interfaces. Each native app owns auth UI, navigation, screens,
and the player. Lift code into `shared` only when a real second consumer
appears.

Rationale: the user-facing surface is small (browse → play → annotate) and
the business logic is essentially "fetch and render". Sharing more upfront
would mainly buy Kotlin-flavored ViewModels in Swift, which adds interop
friction without enough reward at this size.

### 3.1 Project layout

```
badminton-rally-mobile/                   (new repo)
├── shared/                               KMP library
│   ├── src/commonMain/kotlin/            models, repos, SupabaseClient factory
│   ├── src/androidMain/kotlin/           Android session storage glue
│   └── src/iosMain/kotlin/               iOS session storage glue
├── androidApp/                           Compose + Media3 ExoPlayer
├── iosApp/                               Xcode project, SwiftUI + AVKit
├── gradle/libs.versions.toml
└── settings.gradle.kts
```

iOS consumes `shared` as a SwiftPM-published XCFramework — no Cocoapods, no
Ruby toolchain.

### 3.2 Toolchain

| Component | Version / Choice |
|---|---|
| Kotlin | 2.x with `kotlin("multiplatform")` |
| KMP targets | `androidTarget()`, `iosX64()`, `iosArm64()`, `iosSimulatorArm64()` |
| supabase-kt | BOM **3.5.0** — `auth-kt`, `postgrest-kt`, `storage-kt` |
| Ktor | **3.4.0** — `ktor-client-cio` (Android), `ktor-client-darwin` (iOS) |
| Session persistence | `multiplatform-settings` (EncryptedSharedPreferences on Android, NSUserDefaults on iOS) |
| Serialization | `kotlinx.serialization` |
| Android UI | Jetpack Compose, Material 3, Media3 ExoPlayer |
| iOS UI | SwiftUI, AVKit, `AVPlayerLayer` via `UIViewRepresentable` |
| iOS framework distribution | SwiftPM XCFramework |

## 4. Shared data layer

### 4.1 SupabaseClient factory (commonMain)

```kotlin
expect fun createSettings(): Settings

fun buildSupabase(): SupabaseClient = createSupabaseClient(
    supabaseUrl = BuildConfig.SUPABASE_URL,
    supabaseKey = BuildConfig.SUPABASE_ANON_KEY,
) {
    install(Auth) {
        scheme = "badmintontracker"
        host   = "login"
        sessionManager = SettingsSessionManager(createSettings())
    }
    install(Postgrest)
    install(Storage)
}
```

`SettingsSessionManager` keeps the user signed in across launches and
refreshes the JWT automatically.

### 4.2 Models (commonMain, kotlinx.serialization)

Mirror the Postgres schema 1:1 with `@SerialName` for snake_case columns.

```kotlin
@Serializable data class RallyClip(
    val id: String,
    @SerialName("video_id")               val videoId: String,
    @SerialName("rally_index")            val rallyIndex: Int,
    @SerialName("start_timestamp")        val startTimestamp: Float,
    @SerialName("end_timestamp")          val endTimestamp: Float,
    @SerialName("duration_seconds")       val durationSeconds: Float,
    @SerialName("clip_storage_path")      val clipStoragePath: String,
    @SerialName("thumbnail_storage_path") val thumbnailStoragePath: String?,
    val title: String?,
    @SerialName("annotation_count")       val annotationCount: Int,
    @SerialName("created_at")             val createdAt: Instant,
)

@Serializable data class RallyAnnotation(
    val id: String,
    @SerialName("clip_id")            val clipId: String,
    @SerialName("timestamp_seconds")  val timestampSeconds: Float,
    val body: String,
    @SerialName("created_at")         val createdAt: Instant,
)

@Serializable data class VideoSummary(
    val id: String,
    val filename: String,
    @SerialName("created_at") val createdAt: Instant,
)
```

### 4.3 Repositories

Interfaces in commonMain; single supabase-kt-backed impl per repo.

| Repo | Method | Notes |
|---|---|---|
| `AuthRepository` | `sessionFlow: StateFlow<SessionStatus>` | drives sign-in / signed-in routing |
|   | `signInEmail(email, pw): Result<Unit>` | |
|   | `signInWithGoogle(): Result<Unit>` | deeplink `badmintontracker://login` |
|   | `signOut()` | |
| `ClipsRepository` | `observeClips(): Flow<List<RallyClip>>` | initial fetch + manual `refresh()` |
|   | `updateTitle(clipId, title): Result<Unit>` | only column user can update per RLS |
| `AnnotationsRepository` | `observeAnnotations(clipId): Flow<List<RallyAnnotation>>` | |
|   | `addAnnotation(clipId, ts, body): Result<Unit>` | |
|   | `deleteAnnotation(id): Result<Unit>` | |
| `MediaRepository` | `signedClipUrl(clip): String` | `storage.from("clips").createSignedUrl(path, 1.hours)` |
|   | `signedThumbnailUrl(clip): String?` | |

**RLS does the auth work.** No `where owner_id = ...` clauses anywhere in
the mobile code — Postgres policies (migration 0002) enforce ownership.

## 5. UI

Three screens per platform; identical UX in two codebases (Compose +
SwiftUI). Navigation: Jetpack Navigation Compose on Android, `NavigationStack`
on iOS.

### 5.1 Sign in
Email + password fields, "Sign in with Google" button. Success → list.

### 5.2 Clip list
Vertical list of cards, newest first, **flat across all videos** (a video
filter chip can be added later). Each card shows:
- Thumbnail (signed URL from `thumbnails` bucket).
- Title or `Rally #{rallyIndex}` if null.
- Source video filename + date as subtitle.
- Duration badge.
- Annotation-count badge.

Pull-to-refresh + refresh on app foreground.

### 5.3 Clip detail
Native player on top, scrub bar, play/pause, playback speed
(0.25× / 0.5× / 1×). Below: editable title field, then annotations list.

**Annotations interaction.** Annotations show as dots on the scrub bar at
their `timestamp_seconds` plus a list below. Tap dot or row → seek.
Long-press a row → delete (undo snackbar). Floating "+" button adds an
annotation pinned to the current playhead.

### 5.4 Player

| | Android | iOS |
|---|---|---|
| Engine | Media3 ExoPlayer | AVPlayer + AVPlayerLayer |
| URL source | `MediaItem.fromUri(signedUrl)` | `AVPlayerItem(url:)` |
| Playback speed | `player.setPlaybackSpeed(rate)` | `player.rate = rate` |
| UI | `androidx.media3.ui.PlayerView` in `AndroidView` | `AVPlayerLayer` via `UIViewRepresentable` |

No shared design system. Material 3 on Android, SwiftUI defaults on iOS.

## 6. Error handling

| Class | Where | Handling |
|---|---|---|
| Auth / session expired | Any 401 from authenticated call | supabase-kt auto-refreshes; if refresh fails, observe `SessionStatus.NotAuthenticated` → route to sign-in |
| Signed URL expired mid-playback | Player error after ~1h | Catch in player error listener, fetch fresh URL, resume from last position |
| Network / Postgrest | Repo calls throw `RestException` / IO | Repos return `Result<T>`; UI shows per-screen snackbar with retry |

No global error overlay. Per-screen recovery is sufficient at this size.
No telemetry SDK in v1 — `Log.e` / `os_log` only.

## 7. Testing

- **Shared (commonTest, JVM)** — unit tests for repos using ktor
  `MockEngine` and a fake `SessionManager`. Serialization round-trips,
  auth state transitions, signed-URL caching. Most tests live here.
- **Android** — Compose UI test for sign-in → list → detail with mocked
  repos. **No player tests** (Media3 instrumentation tests are flaky).
- **iOS** — equivalent XCUITest for the same three flows.

**CI (GitHub Actions, single workflow):**
1. `./gradlew :shared:jvmTest`
2. `./gradlew :androidApp:assembleDebug`
3. `xcodebuild -scheme iosApp build` (macOS runner)

Player playback verified manually before each release.

## 8. Secrets

- `SUPABASE_URL`, `SUPABASE_ANON_KEY` baked into `BuildConfig` from
  `local.properties` (Android) and `Config.xcconfig` (iOS). Neither file
  is committed.
- Anon key is safe to embed — **RLS is the security boundary**, not key
  secrecy.
- No service-role key on the client. Ever.

## 9. Open items / future

- Realtime subscription on `rally_annotations` so two devices stay in sync.
- Realtime on `rally_clips` for live "new clip arrived" UX while Modal
  processes a video.
- Sign in with Apple before App Store submission.
- Offline playback ("save for offline").
- Crashlytics / Sentry.
- Video filter / grouping in the clip list.
- Annotation editing (currently add + delete only).

## 10. Cross-references

- Schema: `supabase/migrations/0001_initial_schema.sql`,
  `supabase/migrations/0004_rally_annotations_and_clip_metadata.sql`.
- RLS policies: `supabase/migrations/0002_rls_policies.sql`,
  `supabase/migrations/0003_storage_buckets.sql`.
- Web auth (reference for parity): `src/views/LoginView.vue`,
  `src/composables/useSession.ts`.
