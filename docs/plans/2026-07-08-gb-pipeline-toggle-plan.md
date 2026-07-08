# Good-Badminton Pipeline Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-video toggle between the existing Phase 1 pipeline (`legacy`) and a Good-Badminton-enhanced one (`gb_fusion`), plus a duplicate-and-rerun flow so the same video can be compared side by side.

**Architecture:** A `pipeline_variant` column on `videos` is the single source of truth; the Modal Phase 1 worker fetches it and, for `gb_fusion`, runs the Good-Badminton `yolo11s-ball.pt` detector and fuses its per-frame shuttle positions into the TrackNet dict (TrackNet wins, GB fills gaps, every position source-tagged). A new `duplicate-video` edge function clones a video row + storage object with the opposite variant. The frontend adds a selector in CourtSetup, badges/actions in ResultsDashboard, and minimal `?videoId=` URL hydration so two runs can be opened in two tabs.

**Tech Stack:** Vue 3 + TypeScript (vue-tsc), Supabase (Postgres migrations, Deno edge functions, storage), Modal (Python 3.11, Ultralytics YOLO, A10G).

**Spec:** `docs/plans/2026-07-08-gb-pipeline-toggle-design.md`

## Global Constraints

- Variants are exactly `'legacy'` and `'gb_fusion'`; default `'legacy'` (CHECK-constrained).
- `supabase/functions/process-video/index.ts` must NOT be modified.
- A `gb_fusion` run with a missing weight must fail (`failed_phase1`), never silently run legacy.
- GB detector params (from their `ShuttlecockTracker` defaults): conf `0.18`, box area ≤ `0.004` of frame area, aspect ratio ≤ `4.0`.
- Shuttle `source` tags: `"tracknet"` / `"gb_yolo"` / `"yolo"` (existing YOLO fallback keeps `"yolo"`).
- This repo has **no test framework** (per project convention): each task verifies via `npm run type-check`, `python -m py_compile`, deploy success, and the final end-to-end smoke task. Do not add a test framework.
- All file paths are relative to the repo root `/Users/coenhallie/Desktop/projects/badminton-tracker`.
- The GB weight is Apache-2.0 from https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt (19,173,075 bytes; single class `badminton`). Do NOT commit it to the repo.
- Python for Modal commands: use the project venv, e.g. `backend/venv/bin/modal`, `backend/venv/bin/python`.

---

### Task 1: Migration + shared TypeScript types

**Files:**
- Create: `supabase/migrations/0006_pipeline_variant.sql`
- Modify: `src/types/analysis.ts` (add `PipelineVariant` near `UploadResponse`, ~line 277)

**Interfaces:**
- Consumes: nothing.
- Produces: `videos.pipeline_variant` (text, default `'legacy'`), `videos.source_video_id` (uuid, nullable); TS type `PipelineVariant = 'legacy' | 'gb_fusion'` exported from `@/types/analysis` — used by Tasks 5–7.

- [ ] **Step 1: Write the migration**

```sql
-- 0006: pipeline variant toggle + duplicate lineage for GB pipeline A/B.
-- pipeline_variant: which Phase 1 shuttle pipeline processes this video.
-- source_video_id: set on rows created by the duplicate-video edge function,
-- pointing at the video they were cloned from (sibling lookup is
-- bidirectional: follow source_video_id, or reverse-query on it).
alter table videos
  add column pipeline_variant text not null default 'legacy'
    check (pipeline_variant in ('legacy', 'gb_fusion')),
  add column source_video_id uuid references videos(id) on delete set null;
```

- [ ] **Step 2: Apply the migration**

Run: `supabase db push`
Expected: `Applying migration 0006_pipeline_variant.sql... Finished supabase db push.`
(If the CLI is not linked, apply the SQL via the Supabase dashboard SQL editor instead, then still commit the file.)

- [ ] **Step 3: Verify columns exist**

Run: `supabase db push` again (or re-run the dashboard query `select pipeline_variant, source_video_id from videos limit 1;`)
Expected: "Remote database is up to date." / one row with `pipeline_variant = 'legacy'`, `source_video_id = null`.

- [ ] **Step 4: Add the TS type**

In `src/types/analysis.ts`, directly above `export interface UploadResponse` (~line 277), add:

```ts
/**
 * Which Phase 1 shuttle pipeline processes a video.
 * Mirrors videos.pipeline_variant (migration 0006).
 */
export type PipelineVariant = 'legacy' | 'gb_fusion'
```

- [ ] **Step 5: Type-check**

Run: `npm run type-check`
Expected: exits 0.

- [ ] **Step 6: Commit**

```bash
git add supabase/migrations/0006_pipeline_variant.sql src/types/analysis.ts
git commit -m "feat(db): pipeline_variant + source_video_id columns for GB pipeline A/B"
```

---

### Task 2: GB weight upload script + upload to Modal volume

**Files:**
- Create: `backend/upload_gb_ball.py`

**Interfaces:**
- Consumes: nothing.
- Produces: weight file on the Modal volume `badminton-tracker-models` at `/gb_ball/yolo11s-ball.pt` — Task 3's `_run_gb_ball_pass` loads it from `{MODELS_PATH}/gb_ball/yolo11s-ball.pt` (MODELS_PATH = `/models`).

- [ ] **Step 1: Write the upload script**

```python
"""
Upload the Good-Badminton yolo11s-ball.pt shuttlecock detector to the Modal
models volume, for the gb_fusion Phase 1 pipeline.

Weight source (Apache-2.0):
https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt

Usage:
    backend/venv/bin/python backend/upload_gb_ball.py --path /path/to/yolo11s-ball.pt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Upload GB ball weight to Modal")
    parser.add_argument("--path", required=True, help="Path to yolo11s-ball.pt")
    args = parser.parse_args()

    try:
        import modal
    except ImportError:
        print("Error: modal package not installed. Run: pip install modal")
        sys.exit(1)

    vol = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(args.path, "/gb_ball/yolo11s-ball.pt")
    print("Uploaded to badminton-tracker-models:/gb_ball/yolo11s-ball.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `backend/venv/bin/python -m py_compile backend/upload_gb_ball.py`
Expected: exits 0, no output.

- [ ] **Step 3: Obtain the weight and upload**

```bash
curl -sL -o /tmp/yolo11s-ball.pt \
  "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt"
ls -l /tmp/yolo11s-ball.pt   # expect 19173075 bytes
backend/venv/bin/python backend/upload_gb_ball.py --path /tmp/yolo11s-ball.pt
```

Expected: `Uploaded to badminton-tracker-models:/gb_ball/yolo11s-ball.pt`

- [ ] **Step 4: Verify it is on the volume**

Run: `backend/venv/bin/modal volume ls badminton-tracker-models gb_ball`
Expected: listing shows `yolo11s-ball.pt` (~19 MB).

- [ ] **Step 5: Commit**

```bash
git add backend/upload_gb_ball.py
git commit -m "feat(backend): upload script for Good-Badminton ball detector weight"
```

---

### Task 3: Modal Phase 1 — variant fetch, GB ball pass, fusion, provenance

**Files:**
- Modify: `backend/modal_supabase_processor.py`
  - New constants + two new functions after `_run_tracknet_pass` (which ends ~line 1849)
  - Detection loop source tags (~lines 2054, 2056)
  - Worker `_process_video_worker`: variant fetch (~line 3620, after the first `send_status_update`) and GB pass call (after the TrackNet section, ~line 3667)
  - Phase 1 results dict (~line 3779)

**Interfaces:**
- Consumes: `videos.pipeline_variant` (Task 1); weight at `/models/gb_ball/yolo11s-ball.pt` (Task 2); existing `_run_tracknet_pass`, `_run_detection_only_loop`, `_build_shuttle_positions_dict`, `MODELS_PATH`, `send_log`.
- Produces: `_run_gb_ball_pass(video_path, send_log) -> Dict[int, Dict[str, Any]]` (raises `RuntimeError` if weight missing); `_merge_shuttle_sources(tracknet_positions, gb_positions) -> Dict[int, Dict[str, Any]]` (entries gain `source`); results JSON gains top-level `"pipeline_variant"`; per-frame `shuttle_position.source` may be `"gb_yolo"`.

- [ ] **Step 1: Add constants and the two new functions**

Insert immediately after the end of `_run_tracknet_pass` (after its `return tracknet_positions, tracknet_available`, ~line 1849):

```python
# Good-Badminton shuttlecock detector (Apache-2.0, yo-WASSUP/Good-Badminton
# v0.1.0). Params mirror their ShuttlecockTracker defaults.
GB_BALL_MODEL_PATH = f"{MODELS_PATH}/gb_ball/yolo11s-ball.pt"
GB_BALL_CONF = 0.18
GB_BALL_MAX_AREA_RATIO = 0.004
GB_BALL_MAX_ASPECT = 4.0


def _gb_ball_detect_sync(video_path: str, model) -> Dict[int, Dict[str, Any]]:
    """Blocking full-video pass with the GB ball detector, batched 32 frames."""
    import cv2

    positions: Dict[int, Dict[str, Any]] = {}
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
    frame_area = float(width * height)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    batch: List[Any] = []
    batch_idx: List[int] = []
    frame_idx = 0

    def flush():
        if not batch:
            return
        preds = model(batch, conf=GB_BALL_CONF, verbose=False)
        for fi, pred in zip(batch_idx, preds):
            best = None
            if pred.boxes is not None and len(pred.boxes) > 0:
                xywh = pred.boxes.xywh.cpu().numpy()
                confs = pred.boxes.conf.cpu().numpy()
                for (cx, cy, w, h), conf in zip(xywh, confs):
                    if w * h / frame_area > GB_BALL_MAX_AREA_RATIO:
                        continue
                    aspect = max(w, h) / max(min(w, h), 1e-6)
                    if aspect > GB_BALL_MAX_ASPECT:
                        continue
                    if best is None or conf > best[2]:
                        best = (float(cx), float(cy), float(conf))
            if best is not None:
                positions[fi] = {"x": best[0], "y": best[1], "visible": True}
            else:
                positions[fi] = {"x": 0, "y": 0, "visible": False}
        batch.clear()
        batch_idx.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(frame)
        batch_idx.append(frame_idx)
        if len(batch) >= 32:
            flush()
            if total and frame_idx % 2048 < 32:
                print(f"[MODAL] [phase1] GB ball pass: {frame_idx}/{total}")
        frame_idx += 1
    flush()
    cap.release()
    return positions


async def _run_gb_ball_pass(
    video_path: "Path",
    send_log,
) -> Dict[int, Dict[str, Any]]:
    """
    Run the Good-Badminton yolo11s-ball shuttlecock detector over the full
    video. Returns `{frame_num: {x, y, visible}}`.

    Raises RuntimeError if the weight is missing: a gb_fusion run must never
    silently degrade to legacy, or the A/B comparison is poisoned.
    """
    from ultralytics import YOLO

    if not os.path.exists(GB_BALL_MODEL_PATH):
        raise RuntimeError(
            "pipeline_variant=gb_fusion but the GB ball weight is missing at "
            f"{GB_BALL_MODEL_PATH}. Upload it with backend/upload_gb_ball.py."
        )

    await send_log("Loading Good-Badminton ball detector (yolo11s-ball)...", "info", "model")
    model = YOLO(GB_BALL_MODEL_PATH)
    await send_log("GB ball detector loaded, running full-video pass...", "info", "model")
    positions = await asyncio.to_thread(_gb_ball_detect_sync, str(video_path), model)

    visible = sum(1 for p in positions.values() if p.get("visible"))
    pct = 100 * visible / max(len(positions), 1)
    await send_log(
        f"GB ball detector: shuttle detected in {visible}/{len(positions)} frames ({pct:.1f}%)",
        "success", "model",
    )
    return positions


def _merge_shuttle_sources(
    tracknet_positions: Dict[int, Dict[str, Any]],
    gb_positions: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Fuse per-frame shuttle positions: TrackNet wins where visible, the GB
    detector fills the gaps. Visible entries gain a `source` tag
    ('tracknet' | 'gb_yolo') that the detection loop propagates into each
    frame's shuttle_position.
    """
    merged: Dict[int, Dict[str, Any]] = {}
    for fn, pos in tracknet_positions.items():
        if pos.get("visible"):
            merged[fn] = {**pos, "source": "tracknet"}
    for fn, pos in gb_positions.items():
        if fn in merged:
            continue
        if pos.get("visible"):
            merged[fn] = {**pos, "source": "gb_yolo"}
        else:
            merged[fn] = {"x": 0, "y": 0, "visible": False}
    for fn, pos in tracknet_positions.items():
        if fn not in merged:
            merged[fn] = pos
    return merged
```

- [ ] **Step 2: Propagate the source tag in the detection loop**

In `_run_detection_only_loop`, the TrackNet branch builds `shuttle_position` in two places (~lines 2054 and 2056). Change both from:

```python
                                shuttle_position = {"x": tx, "y": ty, "source": "tracknet"}
```

to:

```python
                                shuttle_position = {"x": tx, "y": ty, "source": tn_pos.get("source", "tracknet")}
```

(Same edit at both occurrences; keep each line's original indentation.)

- [ ] **Step 3: Fetch the variant in the worker**

In `_process_video_worker`, directly after the first `await send_status_update("processing_phase1", 0, 0, 0)` and its `send_log("Phase 1: rally segmentation + clip cutting", ...)` (~line 3621), add:

```python
        # Pipeline variant (migration 0006). Fetched from the row rather than
        # the request payload so re-runs and duplicates are self-describing.
        # A fetch failure fails the run — never silently run legacy.
        variant_row = await asyncio.to_thread(
            lambda: supabase_client()
            .table("videos")
            .select("pipeline_variant")
            .eq("id", video_id)
            .single()
            .execute()
        )
        pipeline_variant = (variant_row.data or {}).get("pipeline_variant") or "legacy"
        await send_log(f"Pipeline variant: {pipeline_variant}", "info", "processing")
```

- [ ] **Step 4: Run the GB pass and fuse (gb_fusion only)**

In `_process_video_worker`, after the TrackNet section's timing log (`f"TrackNet phase complete in {tracknet_time:.1f}s ..."`) and its `phase_start = time.time()` (~line 3667), add:

```python
        # ------------------------------------------------------------------
        # 3b. Good-Badminton ball detector pass (gb_fusion variant only)
        # ------------------------------------------------------------------
        if pipeline_variant == "gb_fusion":
            gb_positions = await _run_gb_ball_pass(video_path, send_log)
            tracknet_positions = _merge_shuttle_sources(tracknet_positions, gb_positions)
            tracknet_available = tracknet_available or any(
                p.get("visible") for p in gb_positions.values()
            )
            from collections import Counter
            src_counts = Counter(
                p["source"]
                for p in tracknet_positions.values()
                if p.get("visible") and p.get("source")
            )
            await send_log(
                f"Fused shuttle coverage by source: {dict(src_counts)}",
                "info", "model",
            )
            gb_time = time.time() - phase_start
            await send_log(f"GB ball phase complete in {gb_time:.1f}s", "info", "processing")
            phase_start = time.time()
```

- [ ] **Step 5: Record provenance in the Phase 1 results JSON**

In the `phase1_results` dict (~line 3779), add one key after `"phase": "phase1",`:

```python
            "pipeline_variant": pipeline_variant,
```

- [ ] **Step 6: Syntax-check and deploy**

```bash
backend/venv/bin/python -m py_compile backend/modal_supabase_processor.py
backend/venv/bin/modal deploy backend/modal_supabase_processor.py
```

Expected: py_compile silent; deploy ends with `✓ App deployed` (app `badminton-supabase-processor`).

- [ ] **Step 7: Commit**

```bash
git add backend/modal_supabase_processor.py
git commit -m "feat(backend): gb_fusion Phase 1 variant — GB ball pass fused with TrackNet"
```

---

### Task 4: `duplicate-video` edge function

**Files:**
- Create: `supabase/functions/duplicate-video/index.ts`

**Interfaces:**
- Consumes: `videos.pipeline_variant` / `source_video_id` (Task 1); shared `corsHeaders`.
- Produces: `POST duplicate-video` with body `{ video_id: string }` → `200 { new_video_id: string, pipeline_variant: 'legacy' | 'gb_fusion' }`. The new row is `status='uploaded'` with copied keypoints, so the existing unchanged `process-video` accepts it. Used by Task 7's `rerunWithOtherPipeline()`.

- [ ] **Step 1: Write the function**

```ts
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  const auth = req.headers.get("Authorization") ?? "";
  const m = auth.match(/^\s*Bearer\s+(.+?)\s*$/i);
  const jwt = m?.[1];
  if (!jwt) return resp(401, { error: "Missing Authorization" });

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
  const { data: { user }, error: authErr } = await adminClient.auth.getUser(jwt);
  if (authErr || !user) return resp(401, { error: "Invalid JWT" });

  const { video_id } = await req.json();
  if (!video_id) return resp(400, { error: "video_id required" });

  // RLS-scoped read: only the owner sees the row.
  const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${jwt}` } },
  });
  const { data: video, error: vErr } = await userClient
    .from("videos").select("*").eq("id", video_id).single();
  if (vErr || !video) return resp(404, { error: "Video not found" });

  if (["processing_phase1", "processing_phase2", "processing"].includes(video.status)) {
    return resp(409, { error: `Source video is mid-processing (${video.status})` });
  }
  if (!video.manual_court_keypoints) {
    return resp(400, { error: "Source video has no court keypoints to copy" });
  }

  const newId = crypto.randomUUID();
  const newPath = `${user.id}/${newId}.mp4`;
  const newVariant = video.pipeline_variant === "gb_fusion" ? "legacy" : "gb_fusion";

  // Server-side copy: identical bytes, no re-upload.
  const { error: copyErr } = await adminClient.storage
    .from("videos").copy(video.storage_path, newPath);
  if (copyErr) return resp(500, { error: `Storage copy failed: ${copyErr.message}` });

  const { error: insErr } = await adminClient.from("videos").insert({
    id: newId,
    owner_id: user.id,
    filename: video.filename,
    size: video.size,
    storage_path: newPath,
    status: "uploaded",
    manual_court_keypoints: video.manual_court_keypoints,
    player_labels: video.player_labels,
    pipeline_variant: newVariant,
    source_video_id: video.id,
  });
  if (insErr) {
    // Best-effort cleanup of the copied object so a retry doesn't collide.
    try {
      await adminClient.storage.from("videos").remove([newPath]);
    } catch (_) { /* ignore */ }
    return resp(500, { error: `Row insert failed: ${insErr.message}` });
  }

  return resp(200, { new_video_id: newId, pipeline_variant: newVariant });
});

function resp(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}
```

- [ ] **Step 2: Deploy**

Run: `supabase functions deploy duplicate-video`
Expected: `Deployed Function duplicate-video`.

- [ ] **Step 3: Smoke the auth guard**

```bash
SUPABASE_URL=$(grep '^VITE_SUPABASE_URL=' .env.local | cut -d= -f2)
curl -s -w "\n%{http_code}\n" -X POST "$SUPABASE_URL/functions/v1/duplicate-video" \
  -H "Content-Type: application/json" -d '{}'
```

Expected: `401` with `{"error":"Missing Authorization"}` (proves deployment + auth gate; the happy path is exercised in Task 8).

- [ ] **Step 4: Commit**

```bash
git add supabase/functions/duplicate-video/index.ts
git commit -m "feat(edge): duplicate-video function for pipeline A/B re-runs"
```

---

### Task 5: CourtSetup pipeline selector

**Files:**
- Modify: `src/components/CourtSetup.vue` (script setup imports/state; the `.update()` call ~line 350; template above the `.keypoint-buttons` div ~line 404; style block)

**Interfaces:**
- Consumes: `PipelineVariant` from `@/types/analysis` (Task 1); column `videos.pipeline_variant`.
- Produces: the video row's `pipeline_variant` is set when the user clicks "Start Analysis". No prop/emit changes — App.vue is untouched by this task.

- [ ] **Step 1: Add state**

In the `<script setup>` block, extend the existing type import from `@/types/analysis` to include `PipelineVariant`, and add near the other refs (e.g. next to `isSaving`):

```ts
const pipelineVariant = ref<PipelineVariant>('legacy')
```

- [ ] **Step 2: Persist it with the keypoints**

Change the existing update call (~line 350):

```ts
    const { error: updateError } = await supabase
      .from('videos')
      .update({ manual_court_keypoints: keypoints, pipeline_variant: pipelineVariant.value })
      .eq('id', props.videoId)
```

- [ ] **Step 3: Add the selector to the template**

Insert immediately BEFORE the `<div class="keypoint-buttons">` (~line 404):

```html
            <div class="pipeline-select">
              <span class="pipeline-select-label">Pipeline</span>
              <label class="pipeline-option">
                <input type="radio" value="legacy" v-model="pipelineVariant" />
                <span><strong>Current</strong> — TrackNet + YOLO shuttle detection</span>
              </label>
              <label class="pipeline-option">
                <input type="radio" value="gb_fusion" v-model="pipelineVariant" />
                <span><strong>Good-Badminton fusion</strong> — adds the GB ball detector (A/B experiment)</span>
              </label>
            </div>
```

- [ ] **Step 4: Add styles**

Append to the component's `<style scoped>` block:

```css
.pipeline-select {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin: 0.75rem 0;
  font-size: 0.85rem;
}
.pipeline-select-label {
  font-weight: 600;
  opacity: 0.75;
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
}
.pipeline-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}
```

- [ ] **Step 5: Type-check**

Run: `npm run type-check`
Expected: exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/components/CourtSetup.vue
git commit -m "feat(frontend): pipeline variant selector in court setup"
```

---

### Task 6: `?videoId=` URL hydration (enables two-tab compare)

**Files:**
- Modify: `src/App.vue` (`onMounted` ~line 985–1000; `hydrateFromExistingVideo` ~line 947; new `loadCompletedResults` helper near `loadVideoUrl` ~line 30)

**Interfaces:**
- Consumes: existing `hydrateFromExistingVideo()`, `loadVideoUrl(videoId)`, `uploadedVideo`, `analysisResult`, `currentState`.
- Produces: opening `/?videoId=<id>` restores that video's state; for `completed` videos the results dashboard renders. Task 7's `openSibling()` relies on this URL contract.

- [ ] **Step 1: Add `loadCompletedResults`**

Add below `loadVideoUrl` (~line 40):

```ts
/**
 * Load a completed video's results JSON from storage so the dashboard can
 * render outside the live AnalysisProgress flow (URL hydration, siblings).
 * Mirrors AnalysisProgress.fetchResultsJson + handlePhase2Complete.
 */
async function loadCompletedResults(videoId: string): Promise<boolean> {
  const { data: row } = await supabase
    .from('videos')
    .select('results_storage_path')
    .eq('id', videoId)
    .single()
  if (!row?.results_storage_path) return false
  const { data: signed } = await supabase.storage
    .from('results')
    .createSignedUrl(row.results_storage_path, 3600)
  if (!signed) return false
  const res = await fetch(signed.signedUrl)
  if (!res.ok) return false
  const results = await res.json()
  analysisResult.value = { ...results, video_id: videoId } as AnalysisResult
  await loadVideoUrl(videoId)
  return true
}
```

- [ ] **Step 2: Route `completed` through it in `hydrateFromExistingVideo`**

Replace the `case 'completed':` branch (currently `currentState.value = 'results'; break`):

```ts
    case 'completed': {
      const loaded = await loadCompletedResults(uploadedVideo.value.video_id)
      currentState.value = loaded ? 'results' : 'error'
      if (!loaded) errorMessage.value = 'Could not load stored results for this video'
      break
    }
```

- [ ] **Step 3: Hydrate from the URL on mount**

In `onMounted`, immediately BEFORE the existing `await hydrateFromExistingVideo()` call (~line 1000):

```ts
  // Deep link: /?videoId=<id> restores that video (used by the pipeline
  // comparison sibling links to open two runs in two tabs).
  const urlVideoId = new URLSearchParams(window.location.search).get('videoId')
  if (urlVideoId && !uploadedVideo.value) {
    const { data: urlVideo } = await supabase
      .from('videos')
      .select('id, filename, size, status')
      .eq('id', urlVideoId)
      .single()
    if (urlVideo) {
      uploadedVideo.value = {
        video_id: urlVideo.id,
        filename: urlVideo.filename,
        size: urlVideo.size,
        status: urlVideo.status,
      }
    }
  }
```

- [ ] **Step 4: Type-check**

Run: `npm run type-check`
Expected: exits 0.

- [ ] **Step 5: Manual check**

Run `npm run dev`, log in, open `http://localhost:5173/?videoId=<id-of-a-completed-video>` (grab an id from the Supabase dashboard).
Expected: results dashboard renders for that video without going through upload.

- [ ] **Step 6: Commit**

```bash
git add src/App.vue
git commit -m "feat(frontend): ?videoId= deep-link hydration incl. completed results"
```

---

### Task 7: Badges, re-run action, sibling links

**Files:**
- Modify: `src/App.vue` (new refs + `loadPipelineInfo` / `rerunWithOtherPipeline` / `openSibling`; call sites in `handlePhase1Complete` ~line 730, `handlePhase2Complete` ~line 722, and Task 6's `loadCompletedResults` success path; `<ResultsDashboard>` usage ~line 1713; `<RallyReview>` usage)
- Modify: `src/components/ResultsDashboard.vue` (props ~line 18, emits ~line 31, template header, styles)
- Modify: `src/components/RallyReview.vue` (props ~line 5, small badge in template header)

**Interfaces:**
- Consumes: `PipelineVariant` (Task 1), `duplicate-video` edge function (Task 4), URL contract `/?videoId=` (Task 6).
- Produces: `ResultsDashboard` new optional props `pipelineVariant?: PipelineVariant`, `hasSibling?: boolean` and new emits `rerunOtherPipeline: []`, `openSibling: []`; `RallyReview` new optional prop `pipelineVariant?: PipelineVariant`.

- [ ] **Step 1: App.vue state + functions**

Add `PipelineVariant` to the existing `@/types/analysis` type imports. Near `uploadedVideo` (~line 60):

```ts
const pipelineVariant = ref<PipelineVariant>('legacy')
const siblingVideoId = ref<string | null>(null)
```

Add below `hydrateFromExistingVideo`:

```ts
/** Load the current video's pipeline variant + sibling (if any). */
async function loadPipelineInfo(videoId: string) {
  pipelineVariant.value = 'legacy'
  siblingVideoId.value = null
  const { data } = await supabase
    .from('videos')
    .select('pipeline_variant, source_video_id')
    .eq('id', videoId)
    .single()
  if (!data) return
  pipelineVariant.value = (data.pipeline_variant ?? 'legacy') as PipelineVariant
  if (data.source_video_id) {
    siblingVideoId.value = data.source_video_id
    return
  }
  const { data: dup } = await supabase
    .from('videos')
    .select('id')
    .eq('source_video_id', videoId)
    .order('created_at', { ascending: false })
    .limit(1)
  siblingVideoId.value = dup?.[0]?.id ?? null
}

/** Duplicate the current video with the opposite pipeline and process it. */
async function rerunWithOtherPipeline() {
  if (!uploadedVideo.value?.video_id) return
  const { data, error } = await supabase.functions.invoke('duplicate-video', {
    body: { video_id: uploadedVideo.value.video_id },
  })
  if (error || !data?.new_video_id) {
    errorMessage.value = 'Could not duplicate the video for a pipeline re-run'
    return
  }
  uploadedVideo.value = { ...uploadedVideo.value, video_id: data.new_video_id, status: 'uploaded' }
  analysisResult.value = null
  pipelineVariant.value = data.pipeline_variant as PipelineVariant
  siblingVideoId.value = null
  // AnalysisProgress invokes process-video on mount; keypoints were copied,
  // so court setup is skipped entirely.
  currentState.value = 'analyzing-phase1'
}

/** Open the sibling run in a new tab for side-by-side comparison. */
function openSibling() {
  if (!siblingVideoId.value) return
  window.open(`${window.location.pathname}?videoId=${siblingVideoId.value}`, '_blank')
}
```

- [ ] **Step 2: Call `loadPipelineInfo` at the three entry points**

In `handlePhase2Complete(result)` (~line 722), after `currentState.value = 'results'`:

```ts
  await loadPipelineInfo(result.video_id)
```

In `handlePhase1Complete()` (~line 730), after it sets its state, add (guarded):

```ts
  if (uploadedVideo.value?.video_id) void loadPipelineInfo(uploadedVideo.value.video_id)
```

In `loadCompletedResults` (Task 6), before `return true`:

```ts
  await loadPipelineInfo(videoId)
```

- [ ] **Step 3: ResultsDashboard props/emits**

Extend the `defineProps` object type (~line 18) with:

```ts
  // Pipeline A/B comparison (design doc 2026-07-08)
  pipelineVariant?: PipelineVariant
  hasSibling?: boolean
```

Add `PipelineVariant` to the `@/types/analysis` type import. Extend `defineEmits` (~line 31):

```ts
  rerunOtherPipeline: []
  openSibling: []
```

- [ ] **Step 4: ResultsDashboard template + styles**

Insert as the FIRST child of the component's root template element:

```html
  <div class="pipeline-bar">
    <span
      class="pipeline-badge"
      :class="pipelineVariant === 'gb_fusion' ? 'pipeline-badge--gb' : 'pipeline-badge--legacy'"
    >
      {{ pipelineVariant === 'gb_fusion' ? 'Good-Badminton fusion' : 'Legacy pipeline' }}
    </span>
    <button v-if="hasSibling" class="pipeline-bar-action" @click="emit('openSibling')">
      Compare: open {{ pipelineVariant === 'gb_fusion' ? 'legacy' : 'Good-Badminton' }} run
    </button>
    <button v-else class="pipeline-bar-action" @click="emit('rerunOtherPipeline')">
      Re-run with {{ pipelineVariant === 'gb_fusion' ? 'legacy' : 'Good-Badminton' }} pipeline
    </button>
  </div>
```

Append styles:

```css
.pipeline-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.pipeline-badge {
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
}
.pipeline-badge--legacy {
  background: rgba(100, 116, 139, 0.15);
  color: #64748b;
}
.pipeline-badge--gb {
  background: rgba(16, 185, 129, 0.15);
  color: #10b981;
}
.pipeline-bar-action {
  font-size: 0.8rem;
  padding: 0.3rem 0.75rem;
  border-radius: 6px;
  border: 1px solid currentColor;
  background: transparent;
  cursor: pointer;
}
```

- [ ] **Step 5: Wire it in App.vue**

At the `<ResultsDashboard>` usage (~line 1713), add:

```html
                :pipeline-variant="pipelineVariant"
                :has-sibling="!!siblingVideoId"
                @rerun-other-pipeline="rerunWithOtherPipeline"
                @open-sibling="openSibling"
```

- [ ] **Step 6: RallyReview badge**

In `src/components/RallyReview.vue`, extend props (~line 5):

```ts
const props = defineProps<{
  videoId: string
  pipelineVariant?: PipelineVariant
}>()
```

with `import type { PipelineVariant } from '@/types/analysis'`. Add near the top of its template (first child of the root element):

```html
  <span v-if="pipelineVariant === 'gb_fusion'" class="pipeline-badge-mini">Good-Badminton fusion</span>
```

```css
.pipeline-badge-mini {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  background: rgba(16, 185, 129, 0.15);
  color: #10b981;
  margin-bottom: 0.5rem;
}
```

At the `<RallyReview>` usage in App.vue, add `:pipeline-variant="pipelineVariant"`.

- [ ] **Step 7: Type-check**

Run: `npm run type-check`
Expected: exits 0.

- [ ] **Step 8: Commit**

```bash
git add src/App.vue src/components/ResultsDashboard.vue src/components/RallyReview.vue
git commit -m "feat(frontend): pipeline badges, re-run action, sibling compare links"
```

---

### Task 8: End-to-end smoke + quantitative comparison

**Files:**
- No code changes. Uses `backend/scripts/compare_rallies.py` and the deployed stack.

**Interfaces:**
- Consumes: everything above, deployed (Tasks 1–4) and running locally (`npm run dev`).

- [ ] **Step 1: Legacy baseline run**

In the app: upload a short test video (~2 min), complete court setup with **Pipeline = Current**, run Phase 1 → rally review → Phase 2 → results.
Expected: dashboard shows the **Legacy pipeline** badge and a **"Re-run with Good-Badminton pipeline"** button.

- [ ] **Step 2: Re-run with the other pipeline**

Click the re-run button.
Expected: app jumps straight to Phase 1 progress (no court setup); processing logs include `Pipeline variant: gb_fusion`, `GB ball detector: shuttle detected in N/M frames`, and `Fused shuttle coverage by source: {...}` with nonzero `gb_yolo`.

- [ ] **Step 3: Verify sibling links both ways**

When the re-run completes: its dashboard shows the **Good-Badminton fusion** badge and **"Compare: open legacy run"**. Click it — the original opens in a new tab via `?videoId=` showing the legacy badge and its own compare link back.

- [ ] **Step 4: Fail-loud check (optional but recommended)**

Temporarily rename the weight on the volume, duplicate + re-run, expect `failed_phase1` with an error mentioning `upload_gb_ball.py`; restore:

```bash
backend/venv/bin/modal volume rm badminton-tracker-models gb_ball/yolo11s-ball.pt
# ...trigger a gb_fusion run, observe failed_phase1...
backend/venv/bin/python backend/upload_gb_ball.py --path /tmp/yolo11s-ball.pt
```

- [ ] **Step 5: Quantitative rally diff**

Run: `backend/venv/bin/modal run backend/scripts/compare_rallies.py` with the two sibling video ids (see that script's `--help` for exact flags).
Expected: a rally-count/boundary comparison between the legacy and gb_fusion runs — this is the side-by-side number the feature exists to produce.

- [ ] **Step 6: Verify no stray diffs and finish**

```bash
git status
npm run type-check
```

Expected: clean tree (all work committed in Tasks 1–7), type-check exits 0.

---

## Self-Review Notes

- **Spec coverage:** migration (§1→Task 1), weight + fail-loud + fusion + provenance + per-source logs (§2→Tasks 2–3), duplicate-video with copy/rollback/409 (§3→Task 4), CourtSetup selector (§4→Task 5), badges/re-run/sibling links (§4→Task 7), verification (§6→Task 8). The spec's "open both side by side" requires deep-linking the app lacked; Task 6 adds the minimal `?videoId=` hydration (the codebase's `hydrateFromExistingVideo` docstring explicitly anticipates this).
- **Type consistency:** `PipelineVariant` defined once (Task 1), consumed in Tasks 5–7; edge function returns `new_video_id` + `pipeline_variant`, consumed verbatim in Task 7; `source` tag `"gb_yolo"` produced in Task 3 matches the log assertions in Task 8.
- **Deliberate deviation:** no TDD test cycles — this repo has no test framework (project convention: scripts + type-check + manual smoke). Each task carries its own verification commands instead.
