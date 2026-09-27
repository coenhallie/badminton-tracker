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
  5. The painted-line gate on the real cloud decode path (VideoCapture BGR ->
     grey -> INTER_AREA), plus parity against opencv_motion.json at the same
     paint points, when that file exists. Fix round 1 (plan-mandated):
     replaces an earlier version of this section that compared against
     opencv_motion.json at the four IMAGE CORNERS, which fails by
     construction (corner distances extrapolate far past where either
     tracker has anything to track, and blew past the gate on a genuine,
     if imprecise, port).
  6. Extra tracker coverage: a late marking time with a chain of buffered
     frames, a video that ends before T, a textureless marking frame, a
     mid-track scene change, and a marking time that falls off a 0.1 s
     sampling boundary.
  7. Phase 2's court area filter on the shadows the camera produces: box
     bottom, or ankles while the camera is followed, and never fewer people
     than the box bottom alone.
  8. Phase 2 counts one detection reported under two track ids once.
  9. CameraMotionTracker.following, which gates section 7's ankle clause.

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
print("5. the painted-line gate on the real cloud decode path")
# ---------------------------------------------------------------------------
# Controller ruling, Task 12 fix round 1: the corner-based version of this
# gate failed by construction (corner distances extrapolate past where
# either tracker has texture to hold on to, and the tracker's own
# reproducibility floor is well under a pixel at the paint points but not at
# the corners). The primary gate is now the painted lines themselves, on this
# script's own VideoCapture BGR decode; parity against opencv_motion.json (a
# separate implementation's measurement, when that file exists) is checked
# at the same paint points, not at the corners.
CORPUS_DIR = Path.home() / "shuttl-corpus" / "IMG_5892"
SOURCE_MOV = CORPUS_DIR / "source.mov"
H_IDEAL_PATH = CORPUS_DIR / "H_ideal.json"
OPENCV_MOTION_PATH = CORPUS_DIR / "opencv_motion.json"

# Design doc's painted-line points (near short service line, near baseline,
# near long service line), in court metres. Matches the plan's Task 5 probe
# and /tmp/hm/cost/acc.py's LINES exactly.
COURT_LENGTH_M = 13.4
SERVICE_LINE_M = 1.98
DOUBLES_LONG_SERVICE_M = 0.76
PAINT_COURT_POINTS = (
    [(x, COURT_LENGTH_M / 2 + SERVICE_LINE_M) for x in np.linspace(0.8, 5.3, 8)]
    + [(x, COURT_LENGTH_M) for x in np.linspace(0.8, 5.3, 8)]
    + [(x, COURT_LENGTH_M - DOUBLES_LONG_SERVICE_M) for x in np.linspace(0.8, 5.3, 8)]
)


def paint_line_vertical_error(grey_float, x_px: float, y_px: float):
    """The vertical distance from (x_px, y_px) to the painted line's centre
    nearby in `grey_float` (a full-resolution grey frame, float dtype), or
    None if there is no line to find there.

    A faithful port of /tmp/hm/cost/acc.py's `line_err`, which the
    controller ruling names as the method for this gate: a grey-weighted
    centroid of the brightest pixels in a +-12 px vertical window, only
    counted when that window's peak is clearly brighter than its wider
    surroundings (the local column's own median, in a +-40 px window, plus
    40). A point with no such peak contributes NOTHING to the aggregate
    (not a fixed penalty); a point outside the image, or too close to the
    top or bottom edge for a full window, is penalised at 99 px so it
    cannot silently vanish from the gate; a window with a peak but no clear
    line shape (degenerate weights) is penalised at 12 px, the edge of the
    search window.
    """
    height, width = grey_float.shape
    if not (0 <= x_px < width and 12 <= y_px < height - 13):
        return 99.0
    x = int(round(x_px))
    offsets = np.arange(-12, 13)
    rows = np.clip((y_px + offsets).astype(int), 0, height - 1)
    column = grey_float[rows, x]
    wide_lo = max(0, int(y_px) - 40)
    wide_hi = int(y_px) + 40
    wide_median = np.median(grey_float[wide_lo:wide_hi, x])
    if column.max() < wide_median + 40:
        return None
    weights = np.clip(column - np.median(column), 0.0, None)
    if weights.max() <= 0:
        return 12.0
    weights = np.where(weights < 0.5 * weights.max(), 0.0, weights)
    return abs(float((weights * offsets).sum() / weights.sum()))


def project_court_point(m, h_ideal_inv, cx: float, cy: float):
    """A court point (metres) into the frame `m` maps to: court -> marking
    frame pixels via `h_ideal_inv` (H_ideal is pixel -> court, per the
    ruling), then marking frame -> this frame's pixels via `m`.
    """
    v = m @ (h_ideal_inv @ np.array([cx, cy, 1.0]))
    return v[0] / v[2], v[1] / v[2]


if not (SOURCE_MOV.exists() and H_IDEAL_PATH.exists()):
    print(f"  SKIPPED: {SOURCE_MOV} and/or {H_IDEAL_PATH} do not exist on this machine.")
else:
    h_ideal = np.array(json.loads(H_IDEAL_PATH.read_text())["H"], dtype=np.float64)
    h_ideal_inv = np.linalg.inv(h_ideal)

    cap = cv2.VideoCapture(str(SOURCE_MOV))
    check("source.mov opens", cap.isOpened())
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paint_tracker = CameraMotionTracker(0.1, source_width, source_height)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        # Per the controller ruling: read AFTER read(), to match the phone's
        # clock for this comparison (the production loop's own per-frame
        # numbering reads it before read(), one frame earlier; see design
        # doc section 2.4 and section 3).
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if paint_tracker.wants(t):
            paint_tracker.offer(t, frame)
    cap.release()
    paint_motion = paint_tracker.finish()
    check("the corpus video tracked", paint_motion is not None)

    if paint_motion is not None:
        # A second, separate decode pass to fetch each accepted sample's own
        # full-resolution frame. Necessary (not just simpler) because a
        # single pass cannot: a marking time that is not the video's first
        # frame accepts several samples in the SAME offer() call (the
        # marking frame itself, plus every frame buffered before it,
        # tracked backwards), so "the frame just read" is not the frame
        # every newly accepted sample belongs to. Re-reading by exact
        # timestamp match sidesteps that: it is the same deterministic
        # decode, so timestamps agree bit for bit between passes.
        remaining = {t: m for t, m in paint_motion.samples}
        errors: list[float] = []
        cap = cv2.VideoCapture(str(SOURCE_MOV))
        while cap.isOpened() and remaining:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            m = remaining.pop(t, None)
            if m is None:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            for cx, cy in PAINT_COURT_POINTS:
                px, py = project_court_point(m, h_ideal_inv, cx, cy)
                error = paint_line_vertical_error(grey, px, py)
                if error is not None:
                    errors.append(error)
        cap.release()
        check("every accepted sample's own frame was found again", not remaining, f"{len(remaining)} missing")
        check("the painted lines were found somewhere", len(errors) > 0)
        if errors:
            median = float(np.median(errors))
            p95 = float(np.percentile(errors, 95))
            check(
                "painted-line vertical error median <= 1.0 px",
                median <= 1.0,
                f"median {median:.3f} px over {len(errors)} (sample, point) pairs",
            )
            check("painted-line vertical error p95 <= 2.0 px", p95 <= 2.0, f"p95 {p95:.3f} px")

        if not OPENCV_MOTION_PATH.exists():
            print(f"  parity SKIPPED: {OPENCV_MOTION_PATH} does not exist yet (another task writes it).")
        else:
            # Task 5 writes exactly this schema (controller ruling): a JSON
            # object with a "samples" list of {"timestamp": float, "M": [[3
            # floats], [3 floats], [3 floats]]}, in source pixels, h22 = 1.
            raw = json.loads(OPENCV_MOTION_PATH.read_text())
            reference_samples = raw.get("samples") if isinstance(raw, dict) else None
            check(
                "opencv_motion.json parses as {'samples': [{timestamp, M}, ...]}",
                isinstance(reference_samples, list),
                f"top-level type {type(raw).__name__}",
            )
            if isinstance(reference_samples, list):
                reference = sorted(
                    (float(s["timestamp"]), np.array(s["M"], dtype=np.float64)) for s in reference_samples
                )
                ref_times = [t for t, _ in reference]
                HALF_FRAME_S = 0.5 / 30.0
                worst_per_sample = []
                for t, m in paint_motion.samples:
                    idx = min(range(len(ref_times)), key=lambda i: abs(ref_times[i] - t))
                    if abs(ref_times[idx] - t) > HALF_FRAME_S:
                        continue
                    ref_m = reference[idx][1]
                    worst = 0.0
                    for cx, cy in PAINT_COURT_POINTS:
                        ox, oy = project_court_point(m, h_ideal_inv, cx, cy)
                        rx, ry = project_court_point(ref_m, h_ideal_inv, cx, cy)
                        worst = max(worst, math.hypot(ox - rx, oy - ry))
                    worst_per_sample.append(worst)
                check(
                    "at least 95% of samples paired with a reference timestamp",
                    len(worst_per_sample) >= 0.95 * len(paint_motion.samples),
                    f"{len(worst_per_sample)}/{len(paint_motion.samples)}",
                )
                if worst_per_sample:
                    parity_median = float(np.median(worst_per_sample))
                    parity_p95 = float(np.percentile(worst_per_sample, 95))
                    check(
                        "parity with opencv_motion.json at the paint points: median <= 1.0 px",
                        parity_median <= 1.0,
                        f"median {parity_median:.3f} px",
                    )
                    check(
                        "parity with opencv_motion.json at the paint points: p95 <= 3.0 px",
                        parity_p95 <= 3.0,
                        f"p95 {parity_p95:.3f} px",
                    )

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

print("  a marking time off a 0.1 s boundary is still sampled as the marking frame")
# Mirrors Kotlin's the_marking_frame_is_the_identity_and_is_sampled_even_off_a_boundary.
OFF_BOUNDARY_T = 0.1234
off_boundary_base = make_texture(WIDTH, HEIGHT, seed=246)
off_boundary_tracker = CameraMotionTracker(OFF_BOUNDARY_T, WIDTH, HEIGHT)
OFF_BOUNDARY_N_FRAMES = 12
marking_index = None
for i in range(OFF_BOUNDARY_N_FRAMES):
    t = i / FPS
    if off_boundary_tracker.wants(t):
        off_boundary_tracker.offer(t, off_boundary_base)
        if marking_index is None and t >= OFF_BOUNDARY_T:
            marking_index = i
off_boundary_motion = off_boundary_tracker.finish()
check("a marking time off the 0.1 s grid still tracks", off_boundary_motion is not None)
if off_boundary_motion is not None:
    expected_marking_t = marking_index / FPS
    check(
        "the reference timestamp is the first frame at or after T, not T itself",
        off_boundary_motion.reference_timestamp == expected_marking_t,
        f"got {off_boundary_motion.reference_timestamp}, expected {expected_marking_t}",
    )
    identity_m = off_boundary_motion.at(expected_marking_t)
    check(
        "the marking frame's own M is the identity",
        identity_m is not None and np.allclose(identity_m, np.eye(3)),
    )


# ---------------------------------------------------------------------------
print("7. Phase 2's court area filter: box bottom, or ankles while the camera is followed")
# ---------------------------------------------------------------------------
# The filter runs on shadows (marking-frame pixels) in
# modal_supabase_processor._run_full_yolo_loop. Its box-bottom test dropped a
# player standing on the near baseline on IMG_5892 (her box bottom lands
# behind the line while her ankles are on it); the ankle point is the
# phone's own ground point, at the phone's confidence. The ankles count only
# while the camera is followed: on 743d7fb1's cut shots, where the tracker
# had lost the camera, they admitted an official and a photographer. The
# union may only add: whoever the box bottom admits must stay admitted.
from modal_supabase_processor import court_area_admits, stabilised_skeleton  # noqa: E402

COURT = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)


def person(ankle_y: float, box_bottom: float, ankle_conf: tuple = (0.9, 0.9), x: float = 300.0) -> dict:
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[:15] = (x, ankle_y - 100.0)
    kpts[15] = (x - 10.0, ankle_y)
    kpts[16] = (x + 10.0, ankle_y)
    conf = np.full(17, 0.9, dtype=np.float32)
    conf[15], conf[16] = ankle_conf
    return {
        "track_id": 1,
        "center": (x, ankle_y),
        "kpts": kpts,
        "conf": conf,
        "bbox": {"x1": x - 40.0, "y1": ankle_y - 200.0, "x2": x + 40.0, "y2": box_bottom},
        "area": 80.0 * (box_bottom - ankle_y + 200.0),
    }


check("box bottom inside is admitted", court_area_admits(person(380.0, 395.0), COURT, True))
check(
    "box bottom inside is admitted whatever the ankles say",
    court_area_admits(person(380.0, 395.0, ankle_conf=(0.1, 0.1)), COURT, True),
)
check(
    "box bottom behind the baseline, ankles on the court: admitted",
    court_area_admits(person(390.0, 440.0), COURT, True),
)
check(
    "the same, with one ankle under the phone's 0.5 confidence: not admitted",
    not court_area_admits(person(390.0, 440.0, ankle_conf=(0.9, 0.4)), COURT, True),
)
check(
    "an ankle confidence of exactly 0.5 counts, as on the phone",
    court_area_admits(person(390.0, 440.0, ankle_conf=(0.5, 0.5)), COURT, True),
)
check("box bottom and ankles both off the court: not admitted", not court_area_admits(person(430.0, 460.0), COURT, True))
check(
    "no confidences at all falls back to the box bottom alone",
    not court_area_admits(dict(person(390.0, 440.0), conf=None), COURT, True),
)
# A pan of 60 px down: in frame pixels the ankles are off the marked court;
# the shadow, mapped back by M^-1, stands on it.
panned_m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 60.0], [0.0, 0.0, 1.0]])
panned = person(450.0, 500.0)
check("in frame pixels the panned player is off the court", not court_area_admits(panned, COURT, True))
check(
    "her shadow in the marking frame is on it",
    court_area_admits(stabilised_skeleton(panned, np.linalg.inv(panned_m)), COURT, True),
)
rng = np.random.default_rng(7)
lost = 0
for _ in range(2000):
    ankle_y = float(rng.uniform(50.0, 480.0))
    candidate = person(
        ankle_y,
        ankle_y + float(rng.uniform(0.0, 80.0)),
        ankle_conf=(float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0))),
        x=float(rng.uniform(50.0, 550.0)),
    )
    box_point = (candidate["center"][0], candidate["bbox"]["y2"])
    if cv2.pointPolygonTest(COURT, box_point, measureDist=False) >= 0 and not court_area_admits(candidate, COURT, True):
        lost += 1
check("nobody the box bottom admits is dropped (2000 random people)", lost == 0, f"{lost} dropped")
check(
    "ankles on the court while the camera is not followed: not admitted",
    not court_area_admits(person(390.0, 440.0), COURT, False),
)
check(
    "box bottom inside while the camera is not followed: still admitted",
    court_area_admits(person(380.0, 395.0), COURT, False),
)
rng = np.random.default_rng(11)
differ = 0
for _ in range(2000):
    ankle_y = float(rng.uniform(50.0, 480.0))
    candidate = person(ankle_y, ankle_y + float(rng.uniform(0.0, 80.0)), x=float(rng.uniform(50.0, 550.0)))
    box_in = cv2.pointPolygonTest(COURT, (candidate["center"][0], candidate["bbox"]["y2"]), measureDist=False) >= 0
    if court_area_admits(candidate, COURT, False) != box_in:
        differ += 1
check("with the camera not followed the filter is the box bottom exactly (2000 random people)", differ == 0, f"{differ} differ")


# ---------------------------------------------------------------------------
print("8. one detection reported under two track ids is one person")
# ---------------------------------------------------------------------------
# Measured on IMG_5892 and 2eabfc01: in track mode ultralytics can report
# bit-identical keypoints under two track ids, one with a stale box elsewhere.
# Admitting her by her ankles made both copies candidates, and they could
# push the far player out of the top two.
from modal_supabase_processor import distinct_people  # noqa: E402

player = person(390.0, 400.0)["kpts"]
other = person(200.0, 210.0, x=250.0)["kpts"]
own_box = [290.0, 190.0, 310.0, 392.0]
stale_box = [20.0, 50.0, 90.0, 180.0]
check(
    "a copy under a stale box collapses to the copy under her own box",
    distinct_people([player, other, player.copy()], np.array([stale_box, [240.0, 0.0, 260.0, 202.0], own_box])) == [1, 2],
)
check(
    "the same when her own box comes first",
    distinct_people([player, player.copy()], np.array([own_box, stale_box])) == [0],
)
nudged = player.copy()
nudged[3] += 0.5
check(
    "two people whose keypoints differ at all both stay",
    distinct_people([player, nudged], np.array([own_box, own_box])) == [0, 1],
)
check("with no boxes the first copy is kept", distinct_people([player, player.copy(), other], None) == [0, 2])


# ---------------------------------------------------------------------------
print("9. following: whether the most recent offered sample was accepted")
# ---------------------------------------------------------------------------
follow_base = make_texture(WIDTH, HEIGHT, seed=91)
follow_other = make_texture(WIDTH, HEIGHT, seed=92)
follow = CameraMotionTracker(0.1, WIDTH, HEIGHT)
check("not following before any frame", not follow.following)
follow.offer(0.0, follow_base)
check("not following on a frame buffered before the marking time", not follow.following)
follow.offer(0.1, follow_base)
check("following once the marking frame is taken", follow.following)
follow.offer(0.2, follow_base)
check("following after an accepted sample", follow.following and follow.accepted == 3)
follow.offer(0.3, follow_other)
check("not following after a dropped sample (a cut to another scene)", not follow.following and follow.dropped == 1)
check("latest() still holds the last accepted camera meanwhile", follow.latest() is not None)
follow.offer(0.4, follow_base)
check("following again once a sample is accepted", follow.following)
blank = CameraMotionTracker(0.1, WIDTH, HEIGHT)
blank.offer(0.1, np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8))
check("not following after the tracker gives up", not blank.following)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("\nall checks passed")
