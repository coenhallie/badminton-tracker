"""
Offline regression suite for backend/camera_motion.py, the cloud's anchored
Lucas-Kanade camera tracker.

Usage:
    python backend/scripts/verify_camera_motion.py

No GPU, no network. Run after any change to backend/camera_motion.py.

This is the only test camera_motion.py has: it is a standalone module (Task
14 wires it into the Phase 2 loop), so its own logic is proven here rather
than through the processor's integration tests. Mirrors, case for case, the
Kotlin reference implementation's own tests in badminton-rally-mobile
(CameraMotionTest.kt, CameraMotionTrackerTest.kt), per the mobile design doc
docs/plans/2026-09-27-camera-motion-design.md section 4, and the plan's
Global Constraints
(.superpowers/sdd/2026-09-27-camera-motion-plan/2026-09-27-camera-motion-plan.md).

Sections:
  1. A synthetic pan, recovered from four source corners to within 2 px.
  2. Sampling matches Kotlin's variable_frame_rate_samples_by_time_not_by_count.
  3. CameraMotion.at()'s seven rules, same cases as Kotlin's CameraMotionTest.
  4. marking_time_from's defaulting.
  5. The corpus comparison against opencv_motion.json, when it exists.
  6. Extra tracker coverage: a late marking time with a chain of buffered
     frames, a video that ends before T, a textureless marking frame, and a
     mid-track scene change.

cv2 and numpy are imported here at module level: this script IS the OpenCV
test, unlike camera_motion.py itself, which the processor may be imported
without OpenCV present (see that module's docstring).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from camera_motion import (  # noqa: E402
    CameraMotion,
    CameraMotionTracker,
    marking_time_from,
    tracker_size,
)

FAILURES: list[str] = []
EPS_S = 1e-9


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def make_texture(width: int, height: int, seed: int):
    """A textured BGR image: seeded noise, blurred, contrast-stretched so the
    texture survives the tracker's 4x INTER_AREA downsample.
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=6.0)
    stretched = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def apply_h(h, x: float, y: float) -> tuple[float, float]:
    v = h @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def worst_corner_distance(expected, got, width: int, height: int) -> float:
    corners = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))
    worst = 0.0
    for x, y in corners:
        ex, ey = apply_h(expected, x, y)
        gx, gy = apply_h(got, x, y)
        worst = max(worst, math.hypot(ex - gx, ey - gy))
    return worst


# ---------------------------------------------------------------------------
print("1. a synthetic pan is recovered to within 2 px at the source corners")
# ---------------------------------------------------------------------------
# A known slow pan: 96 source px/s in x, -36 source px/s in y (the same
# tracker-space slope as the Kotlin CameraMotionTrackerTest pan, scaled by
# the 4x from a 1920-wide source to the 480-wide tracker), over 3 s at 30
# fps. M(t) maps the marking frame's pixels to frame t's pixels (design doc
# section 3), so with frame(t) = warpPerspective(base, pan(t - T)), pan(t -
# T) IS the ground truth for M(t) by construction.
WIDTH, HEIGHT = 1920, 1080
MARKING_TIME = 0.1
FPS = 30.0
N_FRAMES = int(round(3.0 * FPS))
base = make_texture(WIDTH, HEIGHT, seed=12345)


def pan(dt: float):
    return np.array([[1.0, 0.0, 96.0 * dt], [0.0, 1.0, -36.0 * dt], [0.0, 0.0, 1.0]])


tracker = CameraMotionTracker(MARKING_TIME, WIDTH, HEIGHT)
for i in range(N_FRAMES):
    t = i / FPS
    frame = cv2.warpPerspective(base, pan(t - MARKING_TIME), (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
    if tracker.wants(t):
        tracker.offer(t, frame)

motion = tracker.finish()
check("the marking frame had texture to track", motion is not None)
check("no sample was dropped", tracker.dropped == 0, f"dropped={tracker.dropped}")
check("thirty samples at 10 Hz over 3 s", tracker.accepted == 30, f"accepted={tracker.accepted}")

if motion is not None:
    last_sample_time = motion.samples[-1][0]
    known = []
    interpolated = []
    for i in range(N_FRAMES):
        t = i / FPS
        m = motion.at(t)
        if m is None:
            continue
        known.append(t)
        # Beyond the last 10 Hz sample, at() HOLDS that sample rather than
        # extrapolating (design doc section 4.4, rule 3): correct behaviour,
        # but not pixel-accurate against a continuously moving synthetic
        # pan. The 2 px bound below applies only inside the sampled range,
        # where at() blends between two real tracked samples.
        if t <= last_sample_time + EPS_S:
            interpolated.append(worst_corner_distance(pan(t - MARKING_TIME), m, WIDTH, HEIGHT))
    check("every one of the 90 frames has a known camera", len(known) == N_FRAMES, f"{len(known)}/{N_FRAMES}")
    check(
        "within 2 px of the true pan inside the sampled range",
        bool(interpolated) and max(interpolated) <= 2.0,
        f"max {max(interpolated):.3f} px over {len(interpolated)} frames" if interpolated else "no samples",
    )

# ---------------------------------------------------------------------------
print("2. sampling matches Kotlin's variable_frame_rate_samples_by_time_not_by_count")
# ---------------------------------------------------------------------------
still = make_texture(WIDTH, HEIGHT, seed=99)
irregular_tracker = CameraMotionTracker(0.1, WIDTH, HEIGHT)
TIMES = [0.0, 0.0, 0.03, 0.07, 0.1, 0.13, 0.19, 0.21, 0.25, 0.33, 0.41, 0.52]
for t in TIMES:
    if irregular_tracker.wants(t):
        irregular_tracker.offer(t, still)
irregular_motion = irregular_tracker.finish()
check("a still scene still tracks", irregular_motion is not None)
if irregular_motion is not None:
    got_times = [s[0] for s in irregular_motion.samples]
    expected_times = [0.0, 0.1, 0.21, 0.33, 0.41, 0.52]
    check("same sample timestamps as the Kotlin test", got_times == expected_times, f"got {got_times}")

# ---------------------------------------------------------------------------
print("3. CameraMotion.at()'s rules: the same seven cases as Kotlin's CameraMotionTest")
# ---------------------------------------------------------------------------
def shift(x: float):
    return np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


rules_motion = CameraMotion.of(
    0.1,
    [(0.1, shift(0.0)), (0.2, shift(10.0)), (1.2, shift(50.0))],
)

check("an exact sample is returned as is", rules_motion.at(0.2)[0][2] == 10.0)
check("between close samples is a blend", abs(rules_motion.at(0.15)[0][2] - 5.0) < 1e-9)
check("a blend is normalised", rules_motion.at(0.15)[2][2] == 1.0)
check("inside a long gap near a sample holds it", rules_motion.at(0.3)[0][2] == 10.0)
check("deep inside a long gap is unknown", rules_motion.at(0.7) is None)
check("just before the first holds it", rules_motion.at(0.0)[0][2] == 0.0)
check("long after the last is unknown", rules_motion.at(1.5) is None)

# Not one of the seven, but the same design rule (section 4.4) and cheap to check:
dup_motion = CameraMotion.of(0.0, [(0.0, shift(1.0)), (0.0, shift(2.0))])
check("duplicate timestamps keep the first", len(dup_motion.samples) == 1 and dup_motion.at(0.0)[0][2] == 1.0)

# ---------------------------------------------------------------------------
print("4. marking_time_from defaults to 0.1 s except for a finite number")
# ---------------------------------------------------------------------------
check("missing key gives 0.1", marking_time_from({}) == 0.1)
check("None keypoints gives 0.1", marking_time_from(None) == 0.1)
check("a None value gives 0.1", marking_time_from({"marked_at_s": None}) == 0.1)
check("a string gives 0.1", marking_time_from({"marked_at_s": "0.5"}) == 0.1)
check("NaN gives 0.1", marking_time_from({"marked_at_s": float("nan")}) == 0.1)
check("infinity gives 0.1", marking_time_from({"marked_at_s": float("inf")}) == 0.1)
check("a bool gives 0.1, not 1.0 or 0.0", marking_time_from({"marked_at_s": True}) == 0.1)
check("1.0 gives 1.0", marking_time_from({"marked_at_s": 1.0}) == 1.0)

# ---------------------------------------------------------------------------
print("5. the painted corpus comparison, when opencv_motion.json exists")
# ---------------------------------------------------------------------------
CORPUS_DIR = Path.home() / "shuttl-corpus" / "IMG_5892"
OPENCV_MOTION_PATH = CORPUS_DIR / "opencv_motion.json"
SOURCE_MOV = CORPUS_DIR / "source.mov"

if not OPENCV_MOTION_PATH.exists():
    print(f"  SKIPPED: {OPENCV_MOTION_PATH} does not exist yet (another task writes it).")
else:
    # Assumed schema, since the writer (a separate task) had not run at the
    # time this script was written: a JSON object with a "samples" list of
    # {"timestamp": float, "M": [[3 floats], [3 floats], [3 floats]]}, in
    # source pixels. A file that exists but does not match this shape is a
    # FAILURE (a schema mismatch to fix), not something to skip past.
    raw = json.loads(OPENCV_MOTION_PATH.read_text())
    reference_samples = None
    if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
        reference_samples = raw["samples"]
    elif isinstance(raw, list):
        reference_samples = raw
    check(
        "opencv_motion.json parses as a list of {timestamp, M} samples",
        reference_samples is not None,
        f"top-level type {type(raw).__name__}",
    )
    if reference_samples is not None and not SOURCE_MOV.exists():
        check("source.mov is present next to opencv_motion.json", False, str(SOURCE_MOV))
    elif reference_samples is not None:
        reference = [(float(s["timestamp"]), np.array(s["M"], dtype=np.float64)) for s in reference_samples]
        reference.sort(key=lambda s: s[0])
        ref_times = [t for t, _ in reference]

        cap = cv2.VideoCapture(str(SOURCE_MOV))
        check("source.mov opens", cap.isOpened())
        corpus_tracker = CameraMotionTracker(0.1, WIDTH, HEIGHT)
        own_samples: list[tuple[float, "np.ndarray"]] = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # Per this task's brief: read AFTER read(), to match the phone's
            # clock for this comparison (the production loop's own frame
            # numbering reads it before read(), one frame earlier; see
            # design doc section 2.4 and section 3).
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if corpus_tracker.wants(t):
                corpus_tracker.offer(t, frame)
        cap.release()
        corpus_motion = corpus_tracker.finish()
        check("the corpus video tracked", corpus_motion is not None)

        if corpus_motion is not None:
            HALF_FRAME_S = 0.5 / 30.0
            worst = []
            for t, m in corpus_motion.samples:
                idx = min(range(len(ref_times)), key=lambda i: abs(ref_times[i] - t))
                if abs(ref_times[idx] - t) <= HALF_FRAME_S:
                    worst.append(worst_corner_distance(reference[idx][1], m, WIDTH, HEIGHT))
            check(
                "at least 95% of samples paired with a reference timestamp",
                len(worst) >= 0.95 * len(corpus_motion.samples),
                f"{len(worst)}/{len(corpus_motion.samples)}",
            )
            if worst:
                median = float(np.median(worst))
                check("median worst-corner distance to opencv_motion.json <= 1 px", median <= 1.0, f"median {median:.3f} px")

# ---------------------------------------------------------------------------
print("6. extra tracker coverage")
# ---------------------------------------------------------------------------

print("  a late marking time tracks a chain of buffered frames backwards")
LATE_T = 1.0
late_base = make_texture(WIDTH, HEIGHT, seed=777)
late_tracker = CameraMotionTracker(LATE_T, WIDTH, HEIGHT)
LATE_N_FRAMES = int(round(2.0 * FPS))
for i in range(LATE_N_FRAMES):
    t = i / FPS
    frame = cv2.warpPerspective(late_base, pan(t - LATE_T), (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
    if late_tracker.wants(t):
        late_tracker.offer(t, frame)
late_motion = late_tracker.finish()
check("a late marking time still tracks", late_motion is not None)
if late_motion is not None:
    check(
        "samples exist before the marking time",
        late_motion.samples[0][0] < LATE_T,
        f"first sample at {late_motion.samples[0][0]}",
    )
    for probe_t in (0.0, 0.3, 0.6):
        m = late_motion.at(probe_t)
        check(
            f"buffered frame at t={probe_t} recovered within 2 px",
            m is not None and worst_corner_distance(pan(probe_t - LATE_T), m, WIDTH, HEIGHT) <= 2.0,
        )

print("  a video that ends before the marking time has no camera")
early_end_tracker = CameraMotionTracker(5.0, WIDTH, HEIGHT)
for t in (0.0, 0.1, 0.2):
    if early_end_tracker.wants(t):
        early_end_tracker.offer(t, base)
check("finish() is None", early_end_tracker.finish() is None)

print("  a textureless marking frame gives up")
flat = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
flat_tracker = CameraMotionTracker(0.1, WIDTH, HEIGHT)
for t in (0.0, 0.1, 0.2):
    if flat_tracker.wants(t):
        flat_tracker.offer(t, flat)
check("finish() is None after giving up", flat_tracker.finish() is None)
check("wants() is False after giving up", flat_tracker.wants(0.3) is False)

print("  a frame from another scene is dropped, not accepted")
other = make_texture(WIDTH, HEIGHT, seed=54321)
scene_tracker = CameraMotionTracker(0.1, WIDTH, HEIGHT)
for t in (0.1, 0.2, 0.3):
    frame = other if abs(t - 0.2) < 1e-9 else base
    if scene_tracker.wants(t):
        scene_tracker.offer(t, frame)
scene_motion = scene_tracker.finish()
check("the marking frame and the good frame still tracked", scene_motion is not None)
if scene_motion is not None:
    check(
        "the other-scene sample is missing, not present",
        [s[0] for s in scene_motion.samples] == [0.1, 0.3],
        f"got {[s[0] for s in scene_motion.samples]}",
    )
check("exactly one sample dropped", scene_tracker.dropped == 1, f"dropped={scene_tracker.dropped}")

print("  a portrait source uses a portrait tracker")
check("1080x1920 trackers at 270x480", tracker_size(1080, 1920) == (270, 480))


if FAILURES:
    print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("\nall checks passed")
