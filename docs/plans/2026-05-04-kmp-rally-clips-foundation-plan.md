# KMP Rally Clips — Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap a new Kotlin Multiplatform repo and build the fully-tested `shared` module that exposes Supabase auth, clip data, annotation CRUD, and signed-URL access. After this plan, the project is ready for Android and iOS UI work in follow-up plans.

**Architecture:** Approach A from the design — thin shared Kotlin layer (models + repositories + Supabase client factory + session persistence), no shared UI. Each native app will later own auth UI, navigation, screens, and the player. Repos return `Flow` for observed data and `Result<T>` for one-shot mutations. RLS is the security boundary; mobile code never adds `where owner_id = ...` clauses.

**Tech Stack:** Kotlin 2.1, supabase-kt BOM 3.5.0 (`auth-kt`, `postgrest-kt`, `storage-kt`), Ktor 3.4.0 (`ktor-client-cio` Android, `ktor-client-darwin` iOS), `multiplatform-settings`, `kotlinx.serialization`, `kotlinx.datetime`. JVM unit tests using ktor `MockEngine`.

**Repo:** New repo `badminton-rally-mobile` (separate from `badminton-tracker`). All paths in this plan are relative to that new repo's root.

**Reference design:** `docs/plans/2026-05-04-kmp-rally-clips-mobile-design.md` (in `badminton-tracker`).

---

## Phase 0 — Repo & toolchain

### Task 1: Create the new repo

**Step 1: Create empty repo locally**

```bash
mkdir -p ~/Desktop/projects/badminton-rally-mobile
cd ~/Desktop/projects/badminton-rally-mobile
git init
```

**Step 2: Add `.gitignore`**

Create `.gitignore`:

```
# Gradle
.gradle/
build/
*.gradle.kts.bak

# IntelliJ / Android Studio
.idea/
*.iml
local.properties

# Xcode / iOS
xcuserdata/
*.xcworkspace/xcuserdata/
*.xcuserstate
DerivedData/
.swiftpm/

# Kotlin / KMP
.kotlin/
kotlin-js-store/

# Secrets
**/Config.xcconfig
```

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: init repo with .gitignore"
```

---

### Task 2: Version catalog

**Files:**
- Create: `gradle/libs.versions.toml`

**Step 1: Write the catalog**

```toml
[versions]
kotlin            = "2.1.0"
agp               = "8.7.3"
supabase          = "3.5.0"
ktor              = "3.4.0"
coroutines        = "1.9.0"
serialization     = "1.7.3"
datetime          = "0.6.1"
settings          = "1.2.0"
kotest-assertions = "5.9.1"
turbine           = "1.2.0"

[libraries]
supabase-bom        = { module = "io.github.jan-tennert.supabase:bom", version.ref = "supabase" }
supabase-auth       = { module = "io.github.jan-tennert.supabase:auth-kt" }
supabase-postgrest  = { module = "io.github.jan-tennert.supabase:postgrest-kt" }
supabase-storage    = { module = "io.github.jan-tennert.supabase:storage-kt" }

ktor-client-core    = { module = "io.ktor:ktor-client-core",      version.ref = "ktor" }
ktor-client-cio     = { module = "io.ktor:ktor-client-cio",       version.ref = "ktor" }
ktor-client-darwin  = { module = "io.ktor:ktor-client-darwin",    version.ref = "ktor" }
ktor-client-mock    = { module = "io.ktor:ktor-client-mock",      version.ref = "ktor" }

kotlinx-coroutines  = { module = "org.jetbrains.kotlinx:kotlinx-coroutines-core",   version.ref = "coroutines" }
kotlinx-serialization-json = { module = "org.jetbrains.kotlinx:kotlinx-serialization-json", version.ref = "serialization" }
kotlinx-datetime    = { module = "org.jetbrains.kotlinx:kotlinx-datetime",          version.ref = "datetime" }

settings            = { module = "com.russhwolf:multiplatform-settings",           version.ref = "settings" }
settings-no-arg     = { module = "com.russhwolf:multiplatform-settings-no-arg",    version.ref = "settings" }

kotest-assertions   = { module = "io.kotest:kotest-assertions-core",               version.ref = "kotest-assertions" }
turbine             = { module = "app.cash.turbine:turbine",                       version.ref = "turbine" }

[plugins]
kotlin-multiplatform = { id = "org.jetbrains.kotlin.multiplatform",  version.ref = "kotlin" }
kotlin-serialization = { id = "org.jetbrains.kotlin.plugin.serialization", version.ref = "kotlin" }
android-application  = { id = "com.android.application",             version.ref = "agp" }
android-library      = { id = "com.android.library",                 version.ref = "agp" }
```

**Step 2: Commit**

```bash
git add gradle/libs.versions.toml
git commit -m "chore: version catalog"
```

---

### Task 3: Root Gradle setup

**Files:**
- Create: `settings.gradle.kts`
- Create: `build.gradle.kts`
- Create: `gradle.properties`

**Step 1: Write `settings.gradle.kts`**

```kotlin
rootProject.name = "badminton-rally-mobile"

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

include(":shared")
include(":androidApp")
```

**Step 2: Write root `build.gradle.kts`**

```kotlin
plugins {
    alias(libs.plugins.kotlin.multiplatform)  apply false
    alias(libs.plugins.kotlin.serialization)  apply false
    alias(libs.plugins.android.application)   apply false
    alias(libs.plugins.android.library)       apply false
}
```

**Step 3: Write `gradle.properties`**

```
org.gradle.jvmargs=-Xmx4g -XX:+UseG1GC
org.gradle.parallel=true
org.gradle.caching=true
kotlin.code.style=official
android.useAndroidX=true
```

**Step 4: Verify Gradle resolves**

Run: `./gradlew help`
Expected: `BUILD SUCCESSFUL` (Gradle wrapper will be generated on first run; if not, run `gradle wrapper --gradle-version 8.10.2` first).

**Step 5: Commit**

```bash
git add settings.gradle.kts build.gradle.kts gradle.properties gradle/wrapper gradlew gradlew.bat
git commit -m "chore: gradle root setup"
```

---

## Phase 1 — `shared` module skeleton

### Task 4: Configure the shared module

**Files:**
- Create: `shared/build.gradle.kts`

**Step 1: Write `shared/build.gradle.kts`**

```kotlin
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.android.library)
}

kotlin {
    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    }
    jvm()  // for unit tests on host
    iosX64()
    iosArm64()
    iosSimulatorArm64()

    listOf(iosX64(), iosArm64(), iosSimulatorArm64()).forEach {
        it.binaries.framework {
            baseName = "Shared"
            isStatic = true
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(libs.supabase.bom))
            implementation(libs.supabase.auth)
            implementation(libs.supabase.postgrest)
            implementation(libs.supabase.storage)
            implementation(libs.ktor.client.core)
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.serialization.json)
            implementation(libs.kotlinx.datetime)
            implementation(libs.settings)
            implementation(libs.settings.no.arg)
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation(libs.kotlinx.coroutines)
            implementation(libs.ktor.client.mock)
            implementation(libs.kotest.assertions)
            implementation(libs.turbine)
        }
        androidMain.dependencies {
            implementation(libs.ktor.client.cio)
        }
        jvmMain.dependencies {
            implementation(libs.ktor.client.cio)
        }
        iosMain.dependencies {
            implementation(libs.ktor.client.darwin)
        }
    }
}

android {
    namespace = "com.badmintontracker.shared"
    compileSdk = 35
    defaultConfig { minSdk = 26 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
```

**Step 2: Verify the shared module configures**

Run: `./gradlew :shared:tasks --group="kotlin"`
Expected: tasks like `compileKotlinJvm`, `compileKotlinIosArm64`, `compileKotlinAndroid` listed.

**Step 3: Commit**

```bash
git add shared/build.gradle.kts
git commit -m "feat(shared): configure KMP module with android, jvm, ios targets"
```

---

### Task 5: SupabaseClient factory (no behavior, just compiles)

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/SupabaseFactory.kt`
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/SupabaseConfig.kt`

**Step 1: Write `SupabaseConfig.kt`**

```kotlin
package com.badmintontracker.shared

data class SupabaseConfig(
    val url: String,
    val anonKey: String,
    val deeplinkScheme: String = "badmintontracker",
    val deeplinkHost:   String = "login",
)
```

**Step 2: Write `SupabaseFactory.kt`**

```kotlin
package com.badmintontracker.shared

import com.russhwolf.settings.Settings
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.auth.Auth
import io.github.jan.supabase.auth.SettingsSessionManager
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.storage.Storage
import io.ktor.client.engine.HttpClientEngine

fun buildSupabaseClient(
    config: SupabaseConfig,
    settings: Settings,
    httpEngine: HttpClientEngine? = null,
): SupabaseClient = createSupabaseClient(
    supabaseUrl = config.url,
    supabaseKey = config.anonKey,
) {
    httpEngine?.let { this.httpEngine = it }
    install(Auth) {
        scheme = config.deeplinkScheme
        host   = config.deeplinkHost
        sessionManager = SettingsSessionManager(settings)
    }
    install(Postgrest)
    install(Storage)
}
```

> The `httpEngine` parameter is the test seam — production callers omit it; tests pass a `MockEngine`.

**Step 3: Verify it compiles**

Run: `./gradlew :shared:compileKotlinJvm`
Expected: `BUILD SUCCESSFUL`.

**Step 4: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/
git commit -m "feat(shared): SupabaseClient factory with injectable HTTP engine"
```

---

## Phase 2 — Models

### Task 6: `RallyClip` model + serialization round-trip test

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/model/RallyClip.kt`
- Test: `shared/src/commonTest/kotlin/com/badmintontracker/shared/model/RallyClipSerializationTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.model

import io.kotest.matchers.shouldBe
import kotlinx.datetime.Instant
import kotlinx.serialization.json.Json
import kotlin.test.Test

class RallyClipSerializationTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun decodes_postgrest_payload() {
        val payload = """
            {
              "id": "11111111-1111-1111-1111-111111111111",
              "video_id": "22222222-2222-2222-2222-222222222222",
              "rally_index": 7,
              "start_timestamp": 12.5,
              "end_timestamp": 18.25,
              "duration_seconds": 5.75,
              "clip_storage_path": "uid/video/clip-7.mp4",
              "thumbnail_storage_path": "uid/video/clip-7.jpg",
              "title": "good smash",
              "annotation_count": 2,
              "created_at": "2026-05-04T12:00:00Z"
            }
        """.trimIndent()

        val clip = json.decodeFromString(RallyClip.serializer(), payload)

        clip.id shouldBe "11111111-1111-1111-1111-111111111111"
        clip.rallyIndex shouldBe 7
        clip.startTimestamp shouldBe 12.5f
        clip.thumbnailStoragePath shouldBe "uid/video/clip-7.jpg"
        clip.title shouldBe "good smash"
        clip.annotationCount shouldBe 2
        clip.createdAt shouldBe Instant.parse("2026-05-04T12:00:00Z")
    }

    @Test
    fun handles_null_title_and_thumbnail() {
        val payload = """
            {
              "id": "11111111-1111-1111-1111-111111111111",
              "video_id": "22222222-2222-2222-2222-222222222222",
              "rally_index": 0,
              "start_timestamp": 0.0,
              "end_timestamp": 1.0,
              "duration_seconds": 1.0,
              "clip_storage_path": "uid/video/clip-0.mp4",
              "thumbnail_storage_path": null,
              "title": null,
              "annotation_count": 0,
              "created_at": "2026-05-04T12:00:00Z"
            }
        """.trimIndent()

        val clip = json.decodeFromString(RallyClip.serializer(), payload)

        clip.title shouldBe null
        clip.thumbnailStoragePath shouldBe null
    }
}
```

**Step 2: Run test to verify it fails**

Run: `./gradlew :shared:jvmTest --tests "*RallyClipSerializationTest*"`
Expected: FAIL with "unresolved reference: RallyClip".

**Step 3: Write minimal implementation**

```kotlin
package com.badmintontracker.shared.model

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RallyClip(
    val id: String,
    @SerialName("video_id")               val videoId: String,
    @SerialName("rally_index")            val rallyIndex: Int,
    @SerialName("start_timestamp")        val startTimestamp: Float,
    @SerialName("end_timestamp")          val endTimestamp: Float,
    @SerialName("duration_seconds")       val durationSeconds: Float,
    @SerialName("clip_storage_path")      val clipStoragePath: String,
    @SerialName("thumbnail_storage_path") val thumbnailStoragePath: String? = null,
    val title: String? = null,
    @SerialName("annotation_count")       val annotationCount: Int,
    @SerialName("created_at")             val createdAt: Instant,
)
```

**Step 4: Run test to verify it passes**

Run: `./gradlew :shared:jvmTest --tests "*RallyClipSerializationTest*"`
Expected: PASS.

**Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/model/RallyClip.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/model/RallyClipSerializationTest.kt
git commit -m "feat(shared): RallyClip model with serialization tests"
```

---

### Task 7: `RallyAnnotation` model

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/model/RallyAnnotation.kt`
- Test: `shared/src/commonTest/kotlin/com/badmintontracker/shared/model/RallyAnnotationSerializationTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.model

import io.kotest.matchers.shouldBe
import kotlinx.datetime.Instant
import kotlinx.serialization.json.Json
import kotlin.test.Test

class RallyAnnotationSerializationTest {

    @Test
    fun decodes_postgrest_payload() {
        val payload = """
            {
              "id":               "33333333-3333-3333-3333-333333333333",
              "clip_id":          "11111111-1111-1111-1111-111111111111",
              "timestamp_seconds": 4.2,
              "body":             "great footwork",
              "created_at":       "2026-05-04T12:00:00Z"
            }
        """.trimIndent()

        val a = Json.decodeFromString(RallyAnnotation.serializer(), payload)

        a.clipId shouldBe "11111111-1111-1111-1111-111111111111"
        a.timestampSeconds shouldBe 4.2f
        a.body shouldBe "great footwork"
        a.createdAt shouldBe Instant.parse("2026-05-04T12:00:00Z")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `./gradlew :shared:jvmTest --tests "*RallyAnnotationSerializationTest*"`
Expected: FAIL.

**Step 3: Write minimal implementation**

```kotlin
package com.badmintontracker.shared.model

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RallyAnnotation(
    val id: String,
    @SerialName("clip_id")             val clipId: String,
    @SerialName("timestamp_seconds")   val timestampSeconds: Float,
    val body: String,
    @SerialName("created_at")          val createdAt: Instant,
)
```

**Step 4: Run test to verify it passes**

Run: `./gradlew :shared:jvmTest --tests "*RallyAnnotationSerializationTest*"`
Expected: PASS.

**Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/model/RallyAnnotation.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/model/RallyAnnotationSerializationTest.kt
git commit -m "feat(shared): RallyAnnotation model"
```

---

### Task 8: `VideoSummary` model

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/model/VideoSummary.kt`
- Test: `shared/src/commonTest/kotlin/com/badmintontracker/shared/model/VideoSummarySerializationTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.model

import io.kotest.matchers.shouldBe
import kotlinx.datetime.Instant
import kotlinx.serialization.json.Json
import kotlin.test.Test

class VideoSummarySerializationTest {
    @Test
    fun decodes_payload() {
        val payload = """
            {
              "id":         "22222222-2222-2222-2222-222222222222",
              "filename":   "match-2026-05-01.mp4",
              "created_at": "2026-05-01T10:00:00Z"
            }
        """.trimIndent()
        val v = Json.decodeFromString(VideoSummary.serializer(), payload)
        v.filename shouldBe "match-2026-05-01.mp4"
        v.createdAt shouldBe Instant.parse("2026-05-01T10:00:00Z")
    }
}
```

**Step 2: Run, fail, implement, pass, commit** (same pattern as Tasks 6–7)

```kotlin
package com.badmintontracker.shared.model

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class VideoSummary(
    val id: String,
    val filename: String,
    @SerialName("created_at") val createdAt: Instant,
)
```

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/model/VideoSummary.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/model/VideoSummarySerializationTest.kt
git commit -m "feat(shared): VideoSummary model"
```

---

## Phase 3 — Test infrastructure

### Task 9: MockEngine test helper

**Files:**
- Create: `shared/src/commonTest/kotlin/com/badmintontracker/shared/testing/SupabaseTestClient.kt`

**Step 1: Write the helper**

```kotlin
package com.badmintontracker.shared.testing

import com.badmintontracker.shared.SupabaseConfig
import com.badmintontracker.shared.buildSupabaseClient
import com.russhwolf.settings.MapSettings
import com.russhwolf.settings.Settings
import io.github.jan.supabase.SupabaseClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandler
import io.ktor.http.*
import io.ktor.utils.io.*

object TestSupabase {
    val config = SupabaseConfig(
        url = "https://test.supabase.co",
        anonKey = "test-anon-key",
    )

    fun client(
        settings: Settings = MapSettings(),
        handler: MockRequestHandler,
    ): SupabaseClient = buildSupabaseClient(
        config = config,
        settings = settings,
        httpEngine = MockEngine { request -> handler(request) },
    )

    fun jsonResponse(body: String, status: HttpStatusCode = HttpStatusCode.OK) =
        respond(
            content = ByteReadChannel(body),
            status = status,
            headers = headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
        )
}
```

> `respond` is `io.ktor.client.engine.mock.respond`. Adjust the import in your IDE.

**Step 2: Verify it compiles**

Run: `./gradlew :shared:compileTestKotlinJvm`
Expected: `BUILD SUCCESSFUL`.

**Step 3: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/testing/SupabaseTestClient.kt
git commit -m "test(shared): MockEngine test helper for supabase-kt"
```

---

## Phase 4 — `MediaRepository` (smallest, no auth state)

### Task 10: Signed URL repository

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/MediaRepository.kt`
- Test:   `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/MediaRepositoryTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.model.RallyClip
import com.badmintontracker.shared.testing.TestSupabase
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.ktor.client.engine.mock.respond
import io.ktor.http.*
import io.ktor.utils.io.*
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Instant
import kotlin.test.Test

class MediaRepositoryTest {
    private fun fakeClip(
        clipPath: String = "uid/video/clip-7.mp4",
        thumbPath: String? = "uid/video/clip-7.jpg",
    ) = RallyClip(
        id = "c1", videoId = "v1", rallyIndex = 7,
        startTimestamp = 0f, endTimestamp = 1f, durationSeconds = 1f,
        clipStoragePath = clipPath, thumbnailStoragePath = thumbPath,
        title = null, annotationCount = 0,
        createdAt = Instant.parse("2026-05-04T12:00:00Z"),
    )

    @Test
    fun signedClipUrl_calls_storage_sign_endpoint() = runTest {
        var called: String? = null
        val client = TestSupabase.client { request ->
            called = "${request.method.value} ${request.url.encodedPath}"
            respond(
                content = ByteReadChannel("""{"signedURL":"/storage/v1/object/sign/clips/uid/video/clip-7.mp4?token=abc"}"""),
                status  = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
            )
        }
        val repo = MediaRepositoryImpl(client)

        val url = repo.signedClipUrl(fakeClip())

        called!!.shouldContain("/storage/v1/object/sign/clips/uid/video/clip-7.mp4")
        url.shouldContain("token=abc")
    }

    @Test
    fun signedThumbnailUrl_returns_null_when_no_thumbnail() = runTest {
        val client = TestSupabase.client { _ ->
            respond("", HttpStatusCode.OK)  // unused
        }
        val repo = MediaRepositoryImpl(client)
        repo.signedThumbnailUrl(fakeClip(thumbPath = null)) shouldBe null
    }
}
```

**Step 2: Run test to verify it fails**

Run: `./gradlew :shared:jvmTest --tests "*MediaRepositoryTest*"`
Expected: FAIL — `MediaRepositoryImpl` unresolved.

**Step 3: Write minimal implementation**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.model.RallyClip
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.storage.storage
import kotlin.time.Duration.Companion.hours

interface MediaRepository {
    suspend fun signedClipUrl(clip: RallyClip): String
    suspend fun signedThumbnailUrl(clip: RallyClip): String?
}

class MediaRepositoryImpl(private val client: SupabaseClient) : MediaRepository {

    override suspend fun signedClipUrl(clip: RallyClip): String =
        client.storage.from("clips").createSignedUrl(clip.clipStoragePath, 1.hours)

    override suspend fun signedThumbnailUrl(clip: RallyClip): String? {
        val path = clip.thumbnailStoragePath ?: return null
        return client.storage.from("thumbnails").createSignedUrl(path, 1.hours)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `./gradlew :shared:jvmTest --tests "*MediaRepositoryTest*"`
Expected: PASS.

**Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/MediaRepository.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/MediaRepositoryTest.kt
git commit -m "feat(shared): MediaRepository for signed URLs"
```

---

## Phase 5 — `ClipsRepository`

### Task 11: Read clips

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/ClipsRepository.kt`
- Test:   `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.testing.TestSupabase
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.ktor.client.engine.mock.respond
import io.ktor.http.*
import io.ktor.utils.io.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ClipsRepositoryTest {

    private val twoClips = """
      [
        {"id":"c1","video_id":"v1","rally_index":0,"start_timestamp":0,"end_timestamp":1,
         "duration_seconds":1,"clip_storage_path":"uid/v1/c0.mp4","thumbnail_storage_path":null,
         "title":null,"annotation_count":0,"created_at":"2026-05-04T12:00:00Z"},
        {"id":"c2","video_id":"v1","rally_index":1,"start_timestamp":2,"end_timestamp":3,
         "duration_seconds":1,"clip_storage_path":"uid/v1/c1.mp4","thumbnail_storage_path":null,
         "title":"nice","annotation_count":1,"created_at":"2026-05-04T12:01:00Z"}
      ]
    """.trimIndent()

    @Test
    fun listClips_orders_by_created_at_desc() = runTest {
        var capturedQuery: String? = null
        val client = TestSupabase.client { request ->
            capturedQuery = request.url.toString()
            respond(
                ByteReadChannel(twoClips),
                HttpStatusCode.OK,
                headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
            )
        }
        val repo = ClipsRepositoryImpl(client)

        val clips = repo.listClips()

        clips shouldHaveSize 2
        clips[0].id shouldBe "c1"
        capturedQuery!!.shouldContain("rally_clips")
        capturedQuery!!.shouldContain("order=created_at.desc")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `./gradlew :shared:jvmTest --tests "*ClipsRepositoryTest*"`
Expected: FAIL.

**Step 3: Write minimal implementation**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.model.RallyClip
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.postgrest.postgrest
import io.github.jan.supabase.postgrest.query.Order
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

interface ClipsRepository {
    suspend fun listClips(): List<RallyClip>
    fun observeClips(): Flow<List<RallyClip>>
    suspend fun refresh()
    suspend fun updateTitle(clipId: String, title: String?): Result<Unit>
}

class ClipsRepositoryImpl(private val client: SupabaseClient) : ClipsRepository {

    private val _clips = MutableStateFlow<List<RallyClip>>(emptyList())

    override suspend fun listClips(): List<RallyClip> =
        client.postgrest.from("rally_clips")
            .select { order("created_at", Order.DESCENDING) }
            .decodeList<RallyClip>()

    override fun observeClips(): Flow<List<RallyClip>> = _clips.asStateFlow()

    override suspend fun refresh() {
        _clips.value = listClips()
    }

    override suspend fun updateTitle(clipId: String, title: String?): Result<Unit> = runCatching {
        client.postgrest.from("rally_clips")
            .update(mapOf("title" to title)) {
                filter { eq("id", clipId) }
            }
        Unit
    }
}
```

**Step 4: Run test to verify it passes**

Run: `./gradlew :shared:jvmTest --tests "*ClipsRepositoryTest*"`
Expected: PASS.

**Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/ClipsRepository.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt
git commit -m "feat(shared): ClipsRepository.listClips with desc order"
```

---

### Task 12: `observeClips()` emits after refresh

**Files:**
- Modify: `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt` (add test)

**Step 1: Write the failing test**

Append to `ClipsRepositoryTest`:

```kotlin
@Test
fun observeClips_emits_initial_empty_then_refreshed_list() = runTest {
    val client = TestSupabase.client { _ ->
        respond(
            ByteReadChannel(twoClips),
            HttpStatusCode.OK,
            headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
        )
    }
    val repo = ClipsRepositoryImpl(client)

    app.cash.turbine.turbineScope {
        val flow = repo.observeClips().testIn(backgroundScope)
        flow.awaitItem() shouldBe emptyList()
        repo.refresh()
        flow.awaitItem().map { it.id } shouldBe listOf("c1", "c2")
        flow.cancelAndIgnoreRemainingEvents()
    }
}
```

**Step 2: Run, verify it passes** (implementation already done in Task 11).

Run: `./gradlew :shared:jvmTest --tests "*ClipsRepositoryTest*"`
Expected: PASS.

**Step 3: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt
git commit -m "test(shared): observeClips emits after refresh"
```

---

### Task 13: `updateTitle` PATCHes the row

**Files:**
- Modify: `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt`

**Step 1: Write the failing test**

```kotlin
@Test
fun updateTitle_sends_patch_with_title_field() = runTest {
    var captured: Pair<String, String>? = null
    val client = TestSupabase.client { request ->
        captured = request.method.value to request.body.toByteArray().decodeToString()
        respond("[]", HttpStatusCode.OK,
            headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()))
    }
    val repo = ClipsRepositoryImpl(client)

    val result = repo.updateTitle("c1", "renamed")

    result.isSuccess shouldBe true
    captured!!.first shouldBe "PATCH"
    captured!!.second.shouldContain(""""title":"renamed"""")
}
```

> `request.body.toByteArray()` requires `import io.ktor.client.request.HttpRequestData` and reading from the OutgoingContent. If awkward, use a recording handler that copies bytes — adjust during execution.

**Step 2–4: Run, fail, fix, pass.** Already implemented in Task 11 — this test verifies wire format.

**Step 5: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/ClipsRepositoryTest.kt
git commit -m "test(shared): updateTitle wire-format assertion"
```

---

## Phase 6 — `AnnotationsRepository`

### Task 14: List annotations for a clip

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/AnnotationsRepository.kt`
- Test:   `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AnnotationsRepositoryTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.testing.TestSupabase
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.ktor.client.engine.mock.respond
import io.ktor.http.*
import io.ktor.utils.io.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AnnotationsRepositoryTest {

    private val twoAnnotations = """
      [
        {"id":"a1","clip_id":"c1","timestamp_seconds":1.5,"body":"first",
         "created_at":"2026-05-04T12:00:00Z"},
        {"id":"a2","clip_id":"c1","timestamp_seconds":3.0,"body":"second",
         "created_at":"2026-05-04T12:00:01Z"}
      ]
    """.trimIndent()

    @Test
    fun list_filters_by_clipId_and_orders_by_timestamp() = runTest {
        var capturedUrl: String? = null
        val client = TestSupabase.client { request ->
            capturedUrl = request.url.toString()
            respond(ByteReadChannel(twoAnnotations), HttpStatusCode.OK,
                headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()))
        }
        val repo = AnnotationsRepositoryImpl(client)

        val items = repo.list("c1")

        items shouldHaveSize 2
        items[0].body shouldBe "first"
        capturedUrl!!.shouldContain("rally_annotations")
        capturedUrl!!.shouldContain("clip_id=eq.c1")
        capturedUrl!!.shouldContain("order=timestamp_seconds.asc")
    }
}
```

**Step 2: Run, fail, implement, pass**

```kotlin
package com.badmintontracker.shared.repo

import com.badmintontracker.shared.model.RallyAnnotation
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.postgrest.postgrest
import io.github.jan.supabase.postgrest.query.Order

interface AnnotationsRepository {
    suspend fun list(clipId: String): List<RallyAnnotation>
    suspend fun add(clipId: String, timestampSeconds: Float, body: String): Result<RallyAnnotation>
    suspend fun delete(id: String): Result<Unit>
}

class AnnotationsRepositoryImpl(private val client: SupabaseClient) : AnnotationsRepository {
    override suspend fun list(clipId: String): List<RallyAnnotation> =
        client.postgrest.from("rally_annotations")
            .select {
                filter { eq("clip_id", clipId) }
                order("timestamp_seconds", Order.ASCENDING)
            }
            .decodeList()

    override suspend fun add(
        clipId: String, timestampSeconds: Float, body: String,
    ): Result<RallyAnnotation> = runCatching {
        // owner_id is set server-side via auth.uid() — RLS enforces it.
        client.postgrest.from("rally_annotations")
            .insert(mapOf(
                "clip_id"           to clipId,
                "timestamp_seconds" to timestampSeconds.toString(),
                "body"              to body,
            )) { select() }
            .decodeSingle<RallyAnnotation>()
    }

    override suspend fun delete(id: String): Result<Unit> = runCatching {
        client.postgrest.from("rally_annotations").delete { filter { eq("id", id) } }
        Unit
    }
}
```

> The insert payload uses `Map<String,String>` for portability; adjust to `JsonObject` if supabase-kt's typed insert helpers fit better. The RLS policy requires `owner_id = auth.uid()`. Supabase's PostgREST does NOT auto-fill `owner_id`; we need to either fill it client-side from the current session, OR add a `default auth.uid()` to the column. **Decision: add a default in a migration before this task ships, OR fill it client-side via `client.auth.currentUserOrNull()?.id`. Pick one during execution and update the test accordingly.**

**Step 3: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/AnnotationsRepository.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AnnotationsRepositoryTest.kt
git commit -m "feat(shared): AnnotationsRepository.list"
```

---

### Task 15: Add annotation

**Step 1: Write the failing test** (append to `AnnotationsRepositoryTest`)

```kotlin
@Test
fun add_posts_to_rally_annotations() = runTest {
    var captured: Pair<String, String>? = null
    val client = TestSupabase.client { request ->
        captured = request.method.value to request.body.toByteArray().decodeToString()
        respond(
            content = ByteReadChannel("""[{"id":"a1","clip_id":"c1","timestamp_seconds":1.5,"body":"hi","created_at":"2026-05-04T12:00:00Z"}]"""),
            status  = HttpStatusCode.Created,
            headers = headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
        )
    }
    val repo = AnnotationsRepositoryImpl(client)

    val result = repo.add("c1", 1.5f, "hi")

    result.isSuccess shouldBe true
    result.getOrThrow().body shouldBe "hi"
    captured!!.first shouldBe "POST"
    captured!!.second.shouldContain(""""body":"hi"""")
}
```

**Step 2: Run, pass** — already implemented in Task 14.

**Step 3: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AnnotationsRepositoryTest.kt
git commit -m "test(shared): add annotation wire format"
```

---

### Task 16: Delete annotation

**Step 1: Write the failing test**

```kotlin
@Test
fun delete_sends_delete_for_id() = runTest {
    var capturedUrl: String? = null
    var capturedMethod: String? = null
    val client = TestSupabase.client { request ->
        capturedMethod = request.method.value
        capturedUrl = request.url.toString()
        respond("[]", HttpStatusCode.OK,
            headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()))
    }
    val repo = AnnotationsRepositoryImpl(client)

    val result = repo.delete("a1")

    result.isSuccess shouldBe true
    capturedMethod shouldBe "DELETE"
    capturedUrl!!.shouldContain("id=eq.a1")
}
```

**Step 2: Run, pass** (already implemented).

**Step 3: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AnnotationsRepositoryTest.kt
git commit -m "test(shared): delete annotation wire format"
```

---

## Phase 7 — `AuthRepository`

### Task 17: Auth interface + signed-out session flow

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/AuthRepository.kt`
- Test:   `shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AuthRepositoryTest.kt`

**Step 1: Write the failing test**

```kotlin
package com.badmintontracker.shared.repo

import app.cash.turbine.test
import com.badmintontracker.shared.testing.TestSupabase
import io.github.jan.supabase.auth.status.SessionStatus
import io.kotest.matchers.types.shouldBeInstanceOf
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpStatusCode
import io.ktor.utils.io.ByteReadChannel
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AuthRepositoryTest {
    @Test
    fun fresh_client_starts_in_NotAuthenticated() = runTest {
        val client = TestSupabase.client { _ ->
            respond("", HttpStatusCode.OK)
        }
        val repo = AuthRepositoryImpl(client)
        repo.sessionFlow.test {
            val first = awaitItem()
            first.shouldBeInstanceOf<SessionStatus.NotAuthenticated>()
            cancelAndIgnoreRemainingEvents()
        }
    }
}
```

**Step 2: Run, fail, implement minimally**

```kotlin
package com.badmintontracker.shared.repo

import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.auth.auth
import io.github.jan.supabase.auth.providers.Google
import io.github.jan.supabase.auth.providers.builtin.Email
import io.github.jan.supabase.auth.status.SessionStatus
import kotlinx.coroutines.flow.Flow

interface AuthRepository {
    val sessionFlow: Flow<SessionStatus>
    suspend fun signInEmail(email: String, password: String): Result<Unit>
    suspend fun signInWithGoogle(): Result<Unit>
    suspend fun signOut(): Result<Unit>
}

class AuthRepositoryImpl(private val client: SupabaseClient) : AuthRepository {

    override val sessionFlow: Flow<SessionStatus> = client.auth.sessionStatus

    override suspend fun signInEmail(email: String, password: String): Result<Unit> =
        runCatching {
            client.auth.signInWith(Email) {
                this.email = email
                this.password = password
            }
        }

    override suspend fun signInWithGoogle(): Result<Unit> = runCatching {
        client.auth.signInWith(Google)
    }

    override suspend fun signOut(): Result<Unit> = runCatching {
        client.auth.signOut()
    }
}
```

**Step 3: Run test to verify it passes**

Run: `./gradlew :shared:jvmTest --tests "*AuthRepositoryTest*"`
Expected: PASS.

**Step 4: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/repo/AuthRepository.kt \
        shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AuthRepositoryTest.kt
git commit -m "feat(shared): AuthRepository interface and impl"
```

---

### Task 18: signInEmail success path

**Step 1: Write the failing test**

```kotlin
@Test
fun signInEmail_calls_token_grant_endpoint() = runTest {
    var capturedPath: String? = null
    var capturedBody:  String? = null
    val client = TestSupabase.client { request ->
        capturedPath = request.url.encodedPath
        capturedBody = request.body.toByteArray().decodeToString()
        respond(
            ByteReadChannel("""
              {
                "access_token":"jwt-abc","token_type":"bearer","expires_in":3600,
                "refresh_token":"refresh-1",
                "user":{"id":"u1","aud":"authenticated","role":"authenticated",
                        "email":"a@b.co","created_at":"2026-05-04T12:00:00Z",
                        "updated_at":"2026-05-04T12:00:00Z"}
              }
            """.trimIndent()),
            HttpStatusCode.OK,
            headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
        )
    }
    val repo = AuthRepositoryImpl(client)

    val result = repo.signInEmail("a@b.co", "secret")

    result.isSuccess shouldBe true
    capturedPath!!.shouldContain("/auth/v1/token")
    capturedBody!!.shouldContain(""""email":"a@b.co"""")
}
```

> Exact path may be `/auth/v1/token?grant_type=password` — verify against current supabase-kt during execution and adjust.

**Step 2: Run, pass** — already implemented in Task 17.

**Step 3: Commit**

```bash
git add shared/src/commonTest/kotlin/com/badmintontracker/shared/repo/AuthRepositoryTest.kt
git commit -m "test(shared): signInEmail wire format"
```

---

## Phase 8 — Wiring & smoke

### Task 19: Public DI surface

**Files:**
- Create: `shared/src/commonMain/kotlin/com/badmintontracker/shared/RallyApp.kt`

**Step 1: Write the holder**

```kotlin
package com.badmintontracker.shared

import com.badmintontracker.shared.repo.*
import com.russhwolf.settings.Settings
import io.github.jan.supabase.SupabaseClient
import io.ktor.client.engine.HttpClientEngine

class RallyApp(
    config: SupabaseConfig,
    settings: Settings,
    httpEngine: HttpClientEngine? = null,
) {
    val client: SupabaseClient = buildSupabaseClient(config, settings, httpEngine)
    val auth:        AuthRepository        = AuthRepositoryImpl(client)
    val clips:       ClipsRepository       = ClipsRepositoryImpl(client)
    val annotations: AnnotationsRepository = AnnotationsRepositoryImpl(client)
    val media:       MediaRepository       = MediaRepositoryImpl(client)
}
```

**Step 2: Verify it compiles**

Run: `./gradlew :shared:compileKotlinJvm`
Expected: `BUILD SUCCESSFUL`.

**Step 3: Commit**

```bash
git add shared/src/commonMain/kotlin/com/badmintontracker/shared/RallyApp.kt
git commit -m "feat(shared): RallyApp DI holder"
```

---

### Task 20: Run full test suite

**Step 1: Run all `shared` tests**

Run: `./gradlew :shared:jvmTest`
Expected: all tests PASS, no skips, no warnings about deprecated APIs in our code.

**Step 2: Run iOS framework assembly to confirm nothing broke for Apple targets**

Run: `./gradlew :shared:linkDebugFrameworkIosSimulatorArm64`
Expected: `BUILD SUCCESSFUL`. (Run on macOS only.)

**Step 3: Commit a marker if anything was tweaked**

If no changes, skip. Otherwise:

```bash
git add -A
git commit -m "chore(shared): pass full test suite + iOS framework link"
```

---

## Phase 9 — CI

### Task 21: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the workflow**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  shared-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: 17
      - uses: gradle/actions/setup-gradle@v4
      - run: ./gradlew :shared:jvmTest

  ios-framework:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: 17
      - uses: gradle/actions/setup-gradle@v4
      - run: ./gradlew :shared:linkDebugFrameworkIosSimulatorArm64
```

**Step 2: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: jvm tests + ios framework link"
git push -u origin main   # only after creating the GitHub remote
```

**Step 3: Verify CI passes** by visiting the Actions tab.

---

## Out of scope for this plan (follow-ups)

The next plans should cover, in order:

1. **Android UI plan** — Compose scaffold, sign-in screen, clip list, clip detail with Media3 ExoPlayer, signed-URL refresh on player error, annotation timeline UI.
2. **iOS UI plan** — SwiftUI scaffold, sign-in, clip list, clip detail with `AVPlayer` + `UIViewRepresentable`, equivalent annotation UI.
3. **Polish plan** — README, screenshots, signing for release builds, optional Apple sign-in, optional Realtime on `rally_annotations`.

Each follow-up should reference the design doc (`2026-05-04-kmp-rally-clips-mobile-design.md`) and assume this plan has shipped.

## Open decisions to lock during execution

- **`owner_id` on `rally_annotations` insert.** Either (a) add `default auth.uid()` to the column in a new Supabase migration before shipping the mobile app, or (b) fill `owner_id` client-side from `client.auth.currentUserOrNull()?.id`. (a) is cleaner; pick during Task 14 and update both the test and the implementation.
- **Exact PostgREST URL/body shapes asserted by tests.** Verify against supabase-kt 3.5.0 wire format during execution; adjust assertions if needed. Behaviour is what matters; the URL strings are guard-rails.

## Verification checklist (pre-handoff)

- `./gradlew :shared:jvmTest` — all green
- `./gradlew :shared:linkDebugFrameworkIosSimulatorArm64` — green on macOS
- CI workflow green on the new GitHub repo
- No `TODO:` / `FIXME:` left in committed code
- All commits follow conventional-commit style
