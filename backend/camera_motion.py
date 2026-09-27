"""The cloud's camera tracker: anchored Lucas-Kanade through OpenCV.

Mirrors the pure-Kotlin tracker in
analysis/src/commonMain/kotlin/com/badmintontracker/analysis/camera/
(CameraMotionTracker.kt, CameraMotion.kt) in badminton-rally-mobile, through
OpenCV, per the mobile design doc
docs/plans/2026-09-27-camera-motion-design.md section 4 (the algorithm) and
section 6 (the cloud). The two implementations are not bit-identical; both
are held to the painted lines on the same video (design doc section 8).

This module is standalone (no import of modal_supabase_processor) so it can
be tested offline. Neither cv2 nor numpy is imported at module level: both
are imported inside the functions that need them, mirroring
modal_supabase_processor.py's own lazy imports of both, so any offline script
that imports this module (directly, or through the processor) still loads
without OpenCV or numpy installed.

The Phase 2 loop (`_run_full_yolo_loop` in modal_supabase_processor.py)
drives it. It is exercised offline by backend/scripts/verify_camera_motion.py,
and end to end on a real video by backend/scripts/run_phase2_locally.py.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

# Design doc section 4.1: the tracker's long side, in pixels.
TRACKER_LONG_SIDE = 480

# Design doc section 4.2 and the plan's Global Constraints: sampling.
SAMPLE_INTERVAL_S = 0.1

# Design doc section 4.3 step 1: reference corners.
GOOD_FEATURES_MAX_CORNERS = 150
GOOD_FEATURES_QUALITY = 0.01
GOOD_FEATURES_MIN_DISTANCE = 8
GOOD_FEATURES_BLOCK_SIZE = 3
CORNER_BORDER_PX = 10
MIN_CORNERS = 40

# Design doc section 4.3 step 3: anchored pyramidal Lucas-Kanade.
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 2  # calcOpticalFlowPyrLK's maxLevel: 3 pyramid levels (0, 1, 2).
LK_MAX_ITER = 20
LK_EPS = 0.03
LK_MIN_EIGEN = 1e-4

# Design doc section 4.3 step 4: RANSAC homography.
RANSAC_THRESHOLD_PX = 1.0
RANSAC_MAX_ITERS = 300

# Design doc section 4.3 step 5: acceptance.
MIN_INLIERS = 40
MAX_CORNER_SHIFT = 0.25

# Design doc section 4.4: reading M at a frame.
EXACT_S = 1e-6
MAX_BLEND_GAP_S = 0.5
MAX_HOLD_S = 0.15


def tracker_size(width: int, height: int) -> tuple[int, int]:
    """The tracker's working size for a source of (width, height).

    Mirrors Luma.trackerSize in
    analysis/src/commonMain/kotlin/com/badmintontracker/analysis/camera/Luma.kt,
    per the mobile design doc section 4.1: the long side is scaled down to
    TRACKER_LONG_SIDE px; a source already at or under that size is used as
    is. Holds for portrait sources too: the long side stays the long side.
    """
    if width < 1 or height < 1:
        raise ValueError(f"invalid source size {width}x{height}")
    long_side = max(width, height)
    if long_side <= TRACKER_LONG_SIDE:
        return width, height
    if width >= height:
        return TRACKER_LONG_SIDE, max(1, (height * TRACKER_LONG_SIDE) // width)
    return max(1, (width * TRACKER_LONG_SIDE) // height), TRACKER_LONG_SIDE


def _tracker_to_source(width: int, height: int, tracker_width: int, tracker_height: int) -> "np.ndarray":
    """The pixel-centre mapping S from tracker pixels to source pixels.

    Design doc section 4.1: `x_src = (x_t + 0.5) * sx - 0.5`, `sx = W / w`.
    """
    import numpy as np

    sx = width / tracker_width
    sy = height / tracker_height
    return np.array(
        [
            [sx, 0.0, 0.5 * sx - 0.5],
            [0.0, sy, 0.5 * sy - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reference_corners(reference_grey) -> "np.ndarray":
    """Shi-Tomasi corners on the marking frame, borders removed after.

    Mirrors shiTomasi() in .../camera/Corners.kt through
    cv2.goodFeaturesToTrack, design doc section 4.3 step 1: up to 150
    corners, quality 0.01 of the strongest response, at least 8 px apart, a
    3x3 block over Sobel gradients, then at least 10 px from the border
    (goodFeaturesToTrack has no border parameter of its own, so this is
    applied after, as the design doc specifies).
    """
    import cv2
    import numpy as np

    corners = cv2.goodFeaturesToTrack(
        reference_grey,
        GOOD_FEATURES_MAX_CORNERS,
        GOOD_FEATURES_QUALITY,
        GOOD_FEATURES_MIN_DISTANCE,
        blockSize=GOOD_FEATURES_BLOCK_SIZE,
    )
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    height, width = reference_grey.shape[:2]
    points = corners.reshape(-1, 2)
    inside = (
        (points[:, 0] >= CORNER_BORDER_PX)
        & (points[:, 0] < width - CORNER_BORDER_PX)
        & (points[:, 1] >= CORNER_BORDER_PX)
        & (points[:, 1] < height - CORNER_BORDER_PX)
    )
    return points[inside].reshape(-1, 1, 2).astype(np.float32)


def _plausible(h, tracker_width: int, tracker_height: int) -> bool:
    """Design doc section 4.3 step 5: every marking-frame image corner moves
    at most a quarter of the tracker width, and the fit neither flips nor
    collapses. Mirrors CameraMotionTracker.kt's private `plausible`, whose
    corner check is `Matrix3x3.apply`: null (here, an unusably small
    homogeneous denominator) fails the check rather than raising.
    """
    limit = MAX_CORNER_SHIFT * tracker_width
    corners = (
        (0.0, 0.0),
        (float(tracker_width), 0.0),
        (float(tracker_width), float(tracker_height)),
        (0.0, float(tracker_height)),
    )
    for x, y in corners:
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if abs(w) < 1e-10:
            return False
        px = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
        py = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
        if math.hypot(px - x, py - y) > limit:
            return False
    det = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]
    return 0.5 <= det <= 2.0


def _normalise(h):
    """Divide through so h[2][2] is exactly 1.0; None when it is near zero.

    Mirrors the private `normalise` inside CameraMotionTracker.kt, design doc
    section 4.4 ("every stored M has h22 exactly 1.0").
    """
    s = h[2, 2]
    if abs(s) < 1e-12:
        return None
    out = h / s
    out[2, 2] = 1.0
    return out


def _blend(a, b, weight: float):
    """Element-wise linear blend of two homographies, renormalised so h22 is
    exactly 1.0. Design doc section 4.4, rule 2.
    """
    m = a * (1.0 - weight) + b * weight
    m = m / m[2, 2]
    m[2, 2] = 1.0
    return m


class CameraMotion:
    """Where the camera was, over one decode pass. Design doc section 4.4.

    `samples` is a list of `(timestamp, M)` pairs, sorted by timestamp and
    distinct (construct through `of`, which sorts and drops duplicates,
    keeping the first by original order). `M` is a 3x3 numpy array in source
    pixels, mapping the marking frame's pixels to this timestamp's frame.
    """

    def __init__(self, reference_timestamp: float, samples) -> None:
        for i in range(1, len(samples)):
            if not samples[i][0] > samples[i - 1][0]:
                raise ValueError("samples must be sorted and distinct")
        self.reference_timestamp = reference_timestamp
        self.samples = list(samples)

    @classmethod
    def of(cls, reference_timestamp: float, samples) -> "CameraMotion":
        """Sorted by timestamp; where two share one (within EXACT_S), the
        first by original order is kept.
        """
        ordered = sorted(enumerate(samples), key=lambda pair: (pair[1][0], pair[0]))
        kept = []
        for _, sample in ordered:
            if not kept or sample[0] - kept[-1][0] > EXACT_S:
                kept.append(sample)
        return cls(reference_timestamp, kept)

    def at(self, timestamp: float):
        """The camera at `timestamp`, or None when it is not known there.

        In order: a sample within EXACT_S of `timestamp`; a blend of two
        neighbours no more than MAX_BLEND_GAP_S apart; the nearest sample
        within MAX_HOLD_S; otherwise None. Design doc section 4.4.
        """
        import bisect

        if not self.samples:
            return None
        times = [s[0] for s in self.samples]
        lo = bisect.bisect_left(times, timestamp)
        if lo >= len(times):
            lo = len(times) - 1
        after = self.samples[lo] if self.samples[lo][0] >= timestamp else None
        if after is None:
            before = self.samples[lo]
        else:
            before = self.samples[lo - 1] if lo - 1 >= 0 else None

        if after is not None and abs(after[0] - timestamp) <= EXACT_S:
            return after[1].copy()
        if before is not None and abs(before[0] - timestamp) <= EXACT_S:
            return before[1].copy()
        if before is not None and after is not None and after[0] - before[0] <= MAX_BLEND_GAP_S:
            weight = (timestamp - before[0]) / (after[0] - before[0])
            return _blend(before[1], after[1], weight)

        candidates = [c for c in (before, after) if c is not None]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda c: abs(c[0] - timestamp))
        if abs(nearest[0] - timestamp) <= MAX_HOLD_S:
            return nearest[1].copy()
        return None


class CameraMotionTracker:
    """Follows the camera through one decode pass, in the cloud.

    Mirrors CameraMotionTracker.kt in
    analysis/src/commonMain/kotlin/com/badmintontracker/analysis/camera/
    through OpenCV, per the mobile design doc section 4 (the algorithm) and
    section 6 (the cloud's causal use of `latest()`, and "stabilise early,
    emit late").

    Construct with the marking time T and the source frame size; call
    `wants` for every decoded frame and `offer` with that frame's BGR pixels
    when it says yes; call `finish` after the last frame. Not thread safe:
    the loop that decodes frames owns it.
    """

    def __init__(self, marking_time_s: float, source_width: int, source_height: int) -> None:
        import numpy as np

        self._marking_time_s = float(marking_time_s)
        self._source_width = int(source_width)
        self._source_height = int(source_height)
        self.tracker_width, self.tracker_height = tracker_size(self._source_width, self._source_height)
        self._to_source = _tracker_to_source(
            self._source_width, self._source_height, self.tracker_width, self.tracker_height,
        )
        self._from_source = np.linalg.inv(self._to_source)

        self.accepted = 0
        self.dropped = 0

        self._last_bucket: Optional[int] = None
        self._reference_grey = None
        self._reference_points = None
        self._reference_timestamp = 0.0
        self._gave_up = False
        self._pending: list = []
        self._samples: list = []
        self._guess = None  # tracker-space homography: LK's initial-flow anchor.
        self._latest = None  # source-space M, for the causal in-loop use (section 6).
        self._following = False

    def _bucket(self, timestamp: float) -> int:
        return math.floor(timestamp / SAMPLE_INTERVAL_S + 1e-9)

    def wants(self, timestamp: float) -> bool:
        """Whether the frame at `timestamp` should be offered: the first
        frame of each 0.1 s bucket, and the marking frame (the first at or
        after T) whether or not it starts a bucket. Design doc section 4.2.
        """
        if self._gave_up:
            return False
        if self._reference_grey is None and timestamp >= self._marking_time_s - EXACT_S:
            return True
        return self._last_bucket is None or self._bucket(timestamp) > self._last_bucket

    def offer(self, timestamp: float, frame_bgr) -> None:
        """One sampled frame, as full source-pixel BGR.

        Converts to grey and area-resizes to tracker size itself (design doc
        section 4.1). The phone's tracker takes an already-downsampled luma
        image because the phone does that step on the decode thread itself;
        the cloud loop hands this tracker the raw frame it already holds
        (design doc section 6), so this method does both steps.
        """
        import cv2

        if frame_bgr.shape[0] != self._source_height or frame_bgr.shape[1] != self._source_width:
            raise ValueError(
                f"expected a {self._source_width}x{self._source_height} frame, "
                f"got {frame_bgr.shape[1]}x{frame_bgr.shape[0]}"
            )
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (self.tracker_width, self.tracker_height), interpolation=cv2.INTER_AREA)
        self._offer_grey(timestamp, grey)

    def _offer_grey(self, timestamp: float, grey) -> None:
        import numpy as np

        bucket = self._bucket(timestamp)
        if self._last_bucket is None or bucket > self._last_bucket:
            self._last_bucket = bucket
        if self._gave_up:
            return
        if self._reference_grey is None:
            if timestamp < self._marking_time_s - EXACT_S:
                self._pending.append((timestamp, grey))
                return
            corners = _reference_corners(grey)
            if corners.shape[0] < MIN_CORNERS:
                self._gave_up = True
                self._following = False
                self._pending.clear()
                return
            self._reference_grey = grey
            self._reference_points = corners
            self._reference_timestamp = timestamp
            identity = np.eye(3, dtype=np.float64)
            self._samples.append((timestamp, identity))
            self.accepted += 1
            self._latest = identity
            # Earlier buffered frames, nearest the marking frame first, each
            # guessing from the one after it in this backward chase; it uses
            # its own guess chain and never touches `_latest`, which must
            # stay the marking frame's own identity until the forward pass
            # resumes (design doc section 6: "stabilise early, emit late").
            backward_guess = np.eye(3, dtype=np.float64)
            for pending_t, pending_grey in sorted(self._pending, key=lambda p: p[0], reverse=True):
                backward_guess = self._track(pending_t, pending_grey, backward_guess, update_latest=False)
            self._pending.clear()
            self._guess = np.eye(3, dtype=np.float64)
            self._following = True
            return
        accepted_before = self.accepted
        self._guess = self._track(timestamp, grey, self._guess, update_latest=True)
        self._following = self.accepted > accepted_before

    def _track(self, timestamp: float, grey, guess, update_latest: bool):
        """One anchored LK sample, started from `guess` (tracker space).

        Returns the guess the next call in this chain should start from:
        `guess` again when the sample is dropped, since "the next one starts
        from the last accepted M" (design doc section 4.3 step 5); the newly
        accepted tracker-space homography on success.
        """
        import cv2

        guess_points = cv2.perspectiveTransform(self._reference_points, guess)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._reference_grey,
            grey,
            self._reference_points,
            guess_points.copy(),
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, LK_MAX_ITER, LK_EPS),
            minEigThreshold=LK_MIN_EIGEN,
        )
        ok = status.reshape(-1) == 1
        # Design doc section 4.3 step 3: "A corner fails if ... it ends
        # outside the image." calcOpticalFlowPyrLK's own status flag does not
        # cover this, so it is checked here explicitly, at tracker scale,
        # before RANSAC ever sees the point.
        tracked = next_points.reshape(-1, 2)
        inside = (
            (tracked[:, 0] >= 0)
            & (tracked[:, 0] <= self.tracker_width - 1)
            & (tracked[:, 1] >= 0)
            & (tracked[:, 1] <= self.tracker_height - 1)
        )
        ok = ok & inside
        if int(ok.sum()) < MIN_INLIERS:
            self.dropped += 1
            return guess
        src = self._reference_points.reshape(-1, 2)[ok]
        dst = next_points.reshape(-1, 2)[ok]
        homography, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, RANSAC_THRESHOLD_PX, maxIters=RANSAC_MAX_ITERS,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if homography is None or inliers < MIN_INLIERS or not _plausible(
            homography, self.tracker_width, self.tracker_height,
        ):
            self.dropped += 1
            return guess
        in_source = _normalise(self._to_source @ homography @ self._from_source)
        if in_source is None:
            self.dropped += 1
            return guess
        self._samples.append((timestamp, in_source))
        self.accepted += 1
        if update_latest:
            self._latest = in_source
        return homography

    def latest(self):
        """The last accepted source-pixel M, for the cloud's causal in-loop
        use (design doc section 6): identity at the marking frame itself,
        unchanged by a dropped sample, and None before the marking frame is
        reached or once the tracker has given up. Only the forward pass
        updates this; the backward pass over frames buffered before a late
        marking time never does (see `_track`'s `update_latest`).
        """
        return None if self._latest is None else self._latest.copy()

    @property
    def following(self) -> bool:
        """Whether the most recent sample offered was accepted: the camera is
        being followed right now, rather than `latest()` holding the last
        known M through a cut to another camera or a fast pan. False before
        the marking frame, after a give-up, and after any dropped sample
        until the next accepted one. Frames that are not offered (between
        0.1 s samples) leave it as it was. Only the forward pass sets it; the
        backward pass over frames buffered before a late marking time never
        does.
        """
        return self._following

    def finish(self):
        """The camera over the whole pass, or None when the marking frame was
        never reached, or had too little texture to track (design doc
        section 4.3 step 1: fewer than 40 corners).
        """
        if self._reference_grey is None:
            return None
        return CameraMotion.of(self._reference_timestamp, list(self._samples))


def marking_time_from(keypoints) -> float:
    """T, the marking time, from stored court keypoints.

    `keypoints["marked_at_s"]` when present and a finite number, else the
    legacy default of 0.1 s. Design doc
    docs/plans/2026-09-27-camera-motion-design.md section 3.
    """
    if not isinstance(keypoints, dict):
        return 0.1
    value = keypoints.get("marked_at_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.1
    if not math.isfinite(value):
        return 0.1
    return float(value)
