# Rally Detection Comparison Timeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-wire backend shot-gap rally detection and add a combined comparison timeline bar below the existing client-side rally timeline.

**Architecture:** Three layers of change: (1) restore backend rally detection in the Modal processing pipeline so `rallies` data is included in results JSON, (2) expose backend rallies from the composable alongside client-side rallies, (3) add a second timeline bar in `RallyTimeline.vue` showing combined client+backend rallies color-coded by agreement (both, client-only, backend-only).

**Tech Stack:** Python (Modal serverless), Vue 3, TypeScript, Tailwind CSS variables

---

### Task 1: Restore rally_detection.py in the Modal container image

**Files:**
- Modify: `backend/modal_convex_processor.py:1011`

**Step 1: Add rally_detection.py back to the Modal image**

At line 1011, after `.add_local_dir(str(_backend_dir / "tracknet"), ...)`, add the local file back:

```python
    .add_local_dir(str(_backend_dir / "tracknet"), remote_path="/root/tracknet")
    .add_local_file(str(_backend_dir / "rally_detection.py"), remote_path="/root/rally_detection.py")
```

Also update the comment at line 988 from:
```python
# Add local backend modules (tracknet) to the container
```
to:
```python
# Add local backend modules (tracknet, rally_detection) to the container
```

**Step 2: Commit**

```
feat: restore rally_detection.py in Modal container image
```

---

### Task 2: Re-wire rally detection into the processing pipeline

**Files:**
- Modify: `backend/modal_convex_processor.py:2258` (after `cap.release()`, before `# Build results`)

**Step 1: Insert the rally detection block**

Between `cap.release()` (line 2257) and `# Build results` (line 2259), insert the rally detection phase. This is a restore of the code from commit `64bfc14` with one small update — the new `detect_rallies()` signature accepts `player_positions` as a keyword arg:

```python
        # =================================================================
        # RALLY DETECTION (from shuttle positions)
        # =================================================================
        phase_start = time.time()
        await send_log("Starting rally detection...", "info", "processing")
        detected_rallies = []
        rally_stats = {}

        # Build shuttle positions dict for rally detection
        # Use TrackNet data if available (much denser), otherwise fall back to
        # the per-frame shuttle_position from skeleton_frames.
        # IMPORTANT: Apply court ROI + static position filtering so false
        # positives (e.g. a white object lying on the ground outside the court)
        # don't corrupt the rally gradient analysis.
        rally_shuttle_positions = {}
        if tracknet_available and tracknet_positions:
            _rally_static_clusters: list[dict] = []
            _rally_prev_pos: dict | None = None
            _fps_scale = 30.0 / fps
            _RALLY_STATIC_DIST = max(4, int(0.013 * max(width, height) * _fps_scale))
            _RALLY_MIN_MOVE = max(2, int(0.007 * max(width, height) * _fps_scale))

            _rally_court_polygon = None
            if court_polygon is not None:
                _rc = court_polygon.astype(np.float32).mean(axis=0)
                _rally_court_polygon = (_rc + (court_polygon.astype(np.float32) - _rc) * 1.15).astype(np.int32)

            for fn in sorted(tracknet_positions.keys()):
                pos = tracknet_positions[fn]
                if not pos.get("visible"):
                    rally_shuttle_positions[fn] = {"x": 0, "y": 0, "visible": False}
                    continue

                px, py = pos["x"], pos["y"]

                if _rally_court_polygon is not None:
                    if cv2.pointPolygonTest(_rally_court_polygon, (float(px), float(py)), measureDist=False) < 0:
                        rally_shuttle_positions[fn] = {"x": 0, "y": 0, "visible": False}
                        continue

                is_static = False
                for cl in _rally_static_clusters:
                    if math.sqrt((px - cl["x"])**2 + (py - cl["y"])**2) < _RALLY_STATIC_DIST:
                        cl["count"] += 1
                        cl["x"] = (cl["x"] * (cl["count"] - 1) + px) / cl["count"]
                        cl["y"] = (cl["y"] * (cl["count"] - 1) + py) / cl["count"]
                        is_static = True
                        break

                if is_static:
                    rally_shuttle_positions[fn] = {"x": 0, "y": 0, "visible": False}
                    continue

                if _rally_prev_pos is not None:
                    movement = math.sqrt((px - _rally_prev_pos["x"])**2 + (py - _rally_prev_pos["y"])**2)
                    if movement < _RALLY_MIN_MOVE:
                        found = False
                        for cl in _rally_static_clusters:
                            if math.sqrt((px - cl["x"])**2 + (py - cl["y"])**2) < _RALLY_STATIC_DIST * 2:
                                cl["count"] += 1
                                found = True
                                break
                        if not found:
                            _rally_static_clusters.append({"x": px, "y": py, "count": 1})
                        rally_shuttle_positions[fn] = {"x": 0, "y": 0, "visible": False}
                        continue

                _rally_static_clusters = [c for c in _rally_static_clusters if c["count"] >= 3]

                rally_shuttle_positions[fn] = {"x": px, "y": py, "visible": True}
                _rally_prev_pos = {"x": px, "y": py}

            filtered_count = sum(1 for v in rally_shuttle_positions.values() if v["visible"])
            total_tn = sum(1 for v in tracknet_positions.values() if v.get("visible"))
            await send_log(
                f"Running rally detection using TrackNet data ({filtered_count}/{total_tn} positions after court+static filtering)...",
                "info", "processing"
            )
        else:
            for sf in skeleton_frames:
                fn = sf["frame"]
                sp = sf.get("shuttle_position")
                if sp and sp.get("x") is not None:
                    rally_shuttle_positions[fn] = {"x": sp["x"], "y": sp["y"], "visible": True}
                else:
                    rally_shuttle_positions[fn] = {"x": 0, "y": 0, "visible": False}
            await send_log("Running rally detection using YOLO shuttle data...", "info", "processing")

        if rally_shuttle_positions:
            try:
                sys.path.insert(0, "/root")
                from rally_detection import detect_rallies, compute_rally_stats

                detected_rallies = detect_rallies(
                    rally_shuttle_positions,
                    fps=fps,
                    total_frames=total_frames,
                )
                rally_stats = compute_rally_stats(
                    detected_rallies, rally_shuttle_positions, fps
                )

                if detected_rallies:
                    await send_log(
                        f"Detected {len(detected_rallies)} rallies "
                        f"(avg {rally_stats.get('avg_rally_duration_s', 0):.1f}s, "
                        f"{rally_stats.get('rally_percentage', 0):.0f}% active play)",
                        "success", "processing"
                    )
                else:
                    await send_log("No rallies detected from shuttle data", "warning", "processing")
            except Exception as e:
                await send_log(f"Rally detection error: {e}", "warning", "processing")
                print(f"[MODAL] Rally detection error: {e}")

        rally_time = time.time() - phase_start
        mem_mb = get_memory_mb()
        await send_log(f"Rally detection complete in {rally_time:.1f}s (RAM: {mem_mb:.0f} MB)", "info", "processing")
        phase_start = time.time()
```

**Step 2: Add rallies to results_data**

In the `results_data` dict (line ~2260), add after `"player_zone_analytics": None,`:

```python
            "rallies": detected_rallies,
            "rally_stats": rally_stats,
```

**Step 3: Fix the metadata**

Change the hardcoded metadata (line ~2339-2340) from:
```python
                        "has_rally_detection": False,
                        "rally_count": 0,
```
to:
```python
                        "has_rally_detection": len(detected_rallies) > 0,
                        "rally_count": len(detected_rallies),
```

**Step 4: Commit**

```
feat: re-wire backend rally detection into processing pipeline
```

---

### Task 3: Expose backend rallies from useAdvancedAnalytics

**Files:**
- Modify: `src/composables/useAdvancedAnalytics.ts:344-347` (rallySource computed) and return block (~line 944)

**Step 1: Add backendRallies computed**

After the `rallySource` computed (line 344-347), add:

```typescript
  const backendRallies = computed(() => {
    const result = analysisResult.value
    if (!result?.rallies || result.rallies.length === 0) return []
    return result.rallies.map(r => ({
      startTimestamp: r.start_timestamp,
      endTimestamp: r.end_timestamp,
      durationSeconds: r.duration_seconds,
    }))
  })
```

**Step 2: Update rallySource to detect backend**

Replace the `rallySource` computed (lines 344-347):
```typescript
  const rallySource = computed<'client' | null>(() => {
    if (rallies.value.length > 0) return 'client'
    return null
  })
```
with:
```typescript
  const rallySource = computed<'client' | 'backend' | 'both' | null>(() => {
    const hasClient = rallies.value.length > 0
    const hasBackend = backendRallies.value.length > 0
    if (hasClient && hasBackend) return 'both'
    if (hasClient) return 'client'
    if (hasBackend) return 'backend'
    return null
  })
```

**Step 3: Add to return block**

In the return object (~line 944), add `backendRallies`:

```typescript
  return {
    rallies,
    backendRallies,
    rallySource,
    ...
  }
```

**Step 4: Commit**

```
feat: expose backend rallies from useAdvancedAnalytics composable
```

---

### Task 4: Add combined comparison timeline bar to RallyTimeline.vue

**Files:**
- Modify: `src/components/RallyTimeline.vue`

**Step 1: Add backendRallies prop**

Add to the props interface (line ~6-15):

```typescript
const props = defineProps<{
  rallies: Rally[]
  backendRallies: { startTimestamp: number; endTimestamp: number; durationSeconds: number }[]
  duration: number
  rallySource: 'client' | 'backend' | 'both' | null
  currentTime: number
  pauseBetweenRallies: boolean
  rallyPauseCountdown: number
  pausedAfterRallyId: number | null
  rallySpeedStats?: RallySpeedStats[]
}>()
```

**Step 2: Add the combined segments computed**

After `playbackPct` computed (~line 39), add:

```typescript
/** Build combined timeline segments color-coded by source agreement */
const combinedTimeline = computed(() => {
  if (!props.duration || props.duration <= 0) return []
  if (props.rallies.length === 0 && props.backendRallies.length === 0) return []

  // Collect all time ranges with their source
  type Seg = { start: number; end: number; source: 'client' | 'backend' }
  const segments: Seg[] = []

  for (const r of props.rallies) {
    segments.push({ start: r.startTimestamp, end: r.endTimestamp, source: 'client' })
  }
  for (const r of props.backendRallies) {
    segments.push({ start: r.startTimestamp, end: r.endTimestamp, source: 'backend' })
  }

  // Build a list of non-overlapping segments tagged with agreement
  // Use a sweep-line approach: collect all start/end events, walk through
  type Event = { time: number; type: 'start' | 'end'; source: 'client' | 'backend' }
  const events: Event[] = []
  for (const s of segments) {
    events.push({ time: s.start, type: 'start', source: s.source })
    events.push({ time: s.end, type: 'end', source: s.source })
  }
  events.sort((a, b) => a.time - b.time || (a.type === 'start' ? -1 : 1))

  const result: { leftPct: number; widthPct: number; kind: 'both' | 'client' | 'backend' }[] = []
  let clientActive = 0
  let backendActive = 0
  let prevTime = 0

  for (const ev of events) {
    if ((clientActive > 0 || backendActive > 0) && ev.time > prevTime) {
      const kind = (clientActive > 0 && backendActive > 0) ? 'both'
        : clientActive > 0 ? 'client' : 'backend'
      const leftPct = (prevTime / props.duration) * 100
      const widthPct = ((ev.time - prevTime) / props.duration) * 100
      if (widthPct > 0.01) {
        result.push({ leftPct, widthPct, kind })
      }
    }
    prevTime = ev.time
    if (ev.type === 'start') {
      if (ev.source === 'client') clientActive++
      else backendActive++
    } else {
      if (ev.source === 'client') clientActive--
      else backendActive--
    }
  }

  return result
})
```

**Step 3: Add the combined bar in the template**

After the existing `rally-tl-labels` div (line ~236), before the closing `</div>` of `rally-tl`, add:

```html
    <!-- Combined comparison bar -->
    <div v-if="backendRallies.length > 0" class="rally-tl-combined">
      <div class="rally-tl-combined-header">
        <span class="rally-tl-label">Combined</span>
        <span class="rally-tl-count">client + backend</span>
        <div class="rally-tl-legend">
          <span class="rally-tl-legend-item both">Both</span>
          <span class="rally-tl-legend-item client-only">Client</span>
          <span class="rally-tl-legend-item backend-only">Backend</span>
        </div>
      </div>
      <div class="rally-tl-bar combined-bar">
        <div
          v-for="(seg, i) in combinedTimeline"
          :key="i"
          class="rally-tl-combined-segment"
          :class="seg.kind"
          :style="{
            left: seg.leftPct + '%',
            width: Math.max(seg.widthPct, 0.4) + '%',
          }"
        />
        <div class="rally-tl-playhead" :style="{ left: playbackPct + '%' }" />
      </div>
      <div class="rally-tl-labels">
        <span>0:00</span>
        <span>{{ formatTime(duration / 2) }}</span>
        <span>{{ formatTime(duration) }}</span>
      </div>
    </div>
```

**Step 4: Add the CSS**

Append these styles inside the `<style scoped>` block:

```css
/* Combined comparison timeline */
.rally-tl-combined {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--color-border);
}

.rally-tl-combined-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.rally-tl-legend {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.rally-tl-legend-item {
  font-size: 0.6rem;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 500;
}

.rally-tl-legend-item.both {
  color: #10b981;
  background: color-mix(in srgb, #10b981 12%, transparent);
  border: 1px solid color-mix(in srgb, #10b981 25%, transparent);
}

.rally-tl-legend-item.client-only {
  color: #f59e0b;
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  border: 1px solid color-mix(in srgb, #f59e0b 25%, transparent);
}

.rally-tl-legend-item.backend-only {
  color: #3b82f6;
  background: color-mix(in srgb, #3b82f6 12%, transparent);
  border: 1px solid color-mix(in srgb, #3b82f6 25%, transparent);
}

.combined-bar {
  height: 20px;
}

.rally-tl-combined-segment {
  position: absolute;
  top: 3px;
  bottom: 3px;
  min-width: 3px;
  opacity: 0.75;
  transition: opacity 0.15s;
}

.rally-tl-combined-segment:hover {
  opacity: 1;
}

.rally-tl-combined-segment.both {
  background: #10b981;
}

.rally-tl-combined-segment.client {
  background: #f59e0b;
}

.rally-tl-combined-segment.backend {
  background: #3b82f6;
}
```

**Step 5: Commit**

```
feat: add combined comparison timeline bar to RallyTimeline
```

---

### Task 5: Wire backendRallies through App.vue

**Files:**
- Modify: `src/App.vue:130-131` (destructure) and `src/App.vue:1573-1588` (RallyTimeline usage)

**Step 1: Destructure backendRallies from the composable**

At line ~130, update:
```typescript
const { rallies: detectedRallies, rallySource, rallySpeedStats } = useAdvancedAnalytics(
```
to:
```typescript
const { rallies: detectedRallies, backendRallies, rallySource, rallySpeedStats } = useAdvancedAnalytics(
```

**Step 2: Pass backendRallies to RallyTimeline**

In the `<RallyTimeline>` component (~line 1573), add the prop:
```html
              <RallyTimeline
                v-if="detectedRallies.length > 0 && analysisResult"
                :rallies="detectedRallies"
                :backend-rallies="backendRallies"
                :duration="analysisResult.duration"
```

**Step 3: Update the v-if to also show when backend rallies exist**

Change the v-if condition:
```html
v-if="(detectedRallies.length > 0 || backendRallies.length > 0) && analysisResult"
```

**Step 4: Commit**

```
feat: wire backendRallies from composable to RallyTimeline
```

---

### Task 6: Update types — un-mark BackendRally as legacy

**Files:**
- Modify: `src/types/analysis.ts:258` and `src/types/analysis.ts:290-291`

**Step 1: Update the comment on BackendRally**

Change:
```typescript
// Legacy types kept for backwards compatibility with older processed videos
export interface BackendRally {
```
to:
```typescript
// Backend rally detection result (shot-gap algorithm on server-side TrackNet data)
export interface BackendRally {
```

**Step 2: Update the AnalysisResult comments**

Change:
```typescript
  rallies?: BackendRally[] | null        // Legacy — rally detection now runs client-side
  rally_stats?: RallyStats | null        // Legacy
```
to:
```typescript
  rallies?: BackendRally[] | null
  rally_stats?: RallyStats | null
```

**Step 3: Update rallySource type**

Find the `Rally` interface section and make sure the `rallySource` type used by `RallyTimeline.vue` props accepts the new union. This is already handled by the prop type change in Task 4.

**Step 4: Commit**

```
chore: update rally type comments to reflect restored backend detection
```
