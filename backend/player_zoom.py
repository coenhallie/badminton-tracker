"""
Phase 2's zoom pass: find a player the full-frame pose pass is too coarse for.

The full-frame pass runs the pose model at imgsz 960, where a far player about
115 px tall in a 1920 px frame is about 57 px: on the club-hall video aSASa
(d71b0798) the model found him in none of 603 frames, at any model size. The
same model on a crop around him, enlarged about 2x, finds him in most of them
(badminton-rally-mobile docs/plans/2026-09-29-two-player-tracking-design.md
sections 1 and 4.1). This module decides where to look; the loop runs the
model and merges what it finds.

Per identity slot (the identity tracker's player 0 and player 1):

  * Follow: a slot filled within the last LOST_FRAMES frames whose player is
    small at the model's input gets a window around their last box, run at
    ZOOM_IMGSZ. Only the one detection that continues them is kept.
  * Find: a slot empty for LOST_FRAMES frames (or never filled) gets its half
    of the court searched, the half plus SEARCH_MARGIN_M with room above for a
    jumping player, run at SEARCH_SCALE of its size. Only someone whose ankles
    stand inside that half's court lines is adopted: the umpire and bystanders
    stand in the margin, and adopting one would be followed from then on.

What the zoom keeps is merged into the frame's detections (merge), and the
loop's own filters and identity tracker still decide who is who. The zoom
never emits anything itself.

Coordinates. Boxes, windows and detections are in the frame's own pixels, the
ones the loop emits. Every test against the marks (which side of the net,
inside the lines) first maps ankles into the marking frame through
`frame_to_marking`, as every other decision in the loop does; the Find region
goes the other way, through `camera_now` (marking frame to this frame). A
frame the marks do not describe gets no Find search.

numpy and cv2 only: no model is imported here, so the geometry and the rules
are tested without a GPU (scripts/verify_player_zoom.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4

# The pose model's input size in the full-frame pass (the loop's imgsz=960).
POSE_INPUT_PX = 960.0
# A player whose box, scaled to the model's input, is shorter than this is
# followed with a zoom window. aSASa's far player is about 57 px there; its
# near player about 125, and the IMG_5892 near player 96 to 161.
ZOOM_TRIGGER_MODEL_PX = 80.0
ZOOM_IMGSZ = 640

FOLLOW_WIDTH_H = 2.6
FOLLOW_HEIGHT_H = 1.8
FOLLOW_MIN_W = 200.0
FOLLOW_MIN_H = 160.0
# Each missing frame widens the window, and the reach of a continuation, by
# half again: the player kept moving while they were not seen.
FOLLOW_GROWTH_PER_EMPTY = 0.5
CONTINUE_RADIUS_H = 0.6

LOST_FRAMES = 3
SEARCH_MARGIN_M = 2.0
SEARCH_HEADROOM_M = 2.5
SEARCH_PAD_PX = 20.0
SEARCH_SCALE = 1.5
SEARCH_MAX_IMGSZ = 1920
SEARCH_BACKOFF_S = 1.0
SEARCH_BACKOFF_EVERY = 5

MERGE_IOU = 0.3
ZOOM_TRACK_ID_BASE = 1_000_000
ZOOM_SESSION_STRIDE = 100_000
# The phone's ground-point confidence (NearPlayer.kt MIN_KEYPOINT_CONFIDENCE).
ANKLE_MIN_CONF = 0.5
HEIGHT_EMA = 0.2

LEFT_ANKLE = 15
RIGHT_ANKLE = 16

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class Detection:
    """One person, in frame pixels: box (x1, y1, x2, y2), 17 keypoints, 17 confidences."""
    box: Box
    kpts: np.ndarray
    conf: np.ndarray

    @property
    def mean_conf(self) -> float:
        return float(np.mean(self.conf))


@dataclass(frozen=True)
class ZoomRequest:
    pid: int
    side: str  # the identity tracker's "top" or "bottom"
    kind: str  # "follow" or "search"
    rect: Rect
    imgsz: int


def _apply(m: np.ndarray, x: float, y: float) -> np.ndarray:
    v = m @ np.array([x, y, 1.0])
    return v[:2] / v[2]


class CourtFrame:
    """The marked court: marking-frame pixels to court metres and back, and the net line.

    Four corners, as the loop's own homography (compute_homography_matrix):
    top_left is court (0, 0), the top (far) baseline is y = 0.
    """

    def __init__(self, to_court: np.ndarray, net_left: Tuple[float, float], net_right: Tuple[float, float]):
        self._to_court = to_court
        self._to_marking = np.linalg.inv(to_court)
        self.net_left = net_left
        self.net_right = net_right

    @classmethod
    def from_marks(cls, marks: Optional[Dict]) -> Optional["CourtFrame"]:
        import cv2

        if not marks:
            return None
        try:
            src = np.array([marks[k][:2] for k in ("top_left", "top_right", "bottom_right", "bottom_left")],
                           dtype=np.float32)
            nl = (float(marks["net_left"][0]), float(marks["net_left"][1]))
            nr = (float(marks["net_right"][0]), float(marks["net_right"][1]))
        except (KeyError, TypeError, IndexError, ValueError):
            return None
        dst = np.array([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
                       dtype=np.float32)
        h = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        if abs(nr[0] - nl[0]) < 1e-6 or abs(np.linalg.det(h)) < 1e-12:
            return None
        return cls(h, nl, nr)

    def to_court(self, p: Sequence[float]) -> np.ndarray:
        return _apply(self._to_court, float(p[0]), float(p[1]))

    def to_marking(self, c: Sequence[float]) -> np.ndarray:
        return _apply(self._to_marking, float(c[0]), float(c[1]))

    def side_of(self, p: Sequence[float]) -> str:
        """"top" above the marked net line at this x, "bottom" below it: the identity tracker's sides."""
        (x0, y0), (x1, y1) = self.net_left, self.net_right
        net_y = y0 + (float(p[0]) - x0) / (x1 - x0) * (y1 - y0)
        return "top" if float(p[1]) < net_y else "bottom"

    @staticmethod
    def inside_lines(c: Sequence[float], side: str) -> bool:
        lo, hi = (0.0, COURT_LENGTH_M / 2) if side == "top" else (COURT_LENGTH_M / 2, COURT_LENGTH_M)
        return 0.0 <= float(c[0]) <= COURT_WIDTH_M and lo <= float(c[1]) <= hi


def ground_point(det: Detection) -> Optional[np.ndarray]:
    """The ankle midpoint in frame pixels, or None unless both ankles are confident."""
    if det.conf[LEFT_ANKLE] < ANKLE_MIN_CONF or det.conf[RIGHT_ANKLE] < ANKLE_MIN_CONF:
        return None
    return (np.asarray(det.kpts[LEFT_ANKLE], float) + np.asarray(det.kpts[RIGHT_ANKLE], float)) / 2.0


def to_marking(frame_to_marking: Optional[np.ndarray], p: np.ndarray) -> np.ndarray:
    if frame_to_marking is None:
        return np.asarray(p, float)
    return _apply(frame_to_marking, float(p[0]), float(p[1]))


def _clip(x0: float, y0: float, x1: float, y1: float, fw: int, fh: int) -> Rect:
    return Rect(int(max(0, math.floor(x0))), int(max(0, math.floor(y0))),
                int(min(fw, math.ceil(x1))), int(min(fh, math.ceil(y1))))


def follow_window(last_box: Box, frame_w: int, frame_h: int, empty_frames: int = 0) -> Rect:
    """A window around the last box: 2.6 h by 1.8 h, at least 200 by 160, wider while they are missing."""
    x1, y1, x2, y2 = last_box
    h = max(1.0, y2 - y1)
    grow = 1.0 + FOLLOW_GROWTH_PER_EMPTY * empty_frames
    w = max(FOLLOW_WIDTH_H * h, FOLLOW_MIN_W) * grow
    wh = max(FOLLOW_HEIGHT_H * h, FOLLOW_MIN_H) * grow
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return _clip(cx - w / 2, cy - wh / 2, cx + w / 2, cy + wh / 2, frame_w, frame_h)


def search_region(court: CourtFrame, side: str, camera_now: Optional[np.ndarray],
                  frame_w: int, frame_h: int) -> Optional[Rect]:
    """The slot's half plus SEARCH_MARGIN_M, with SEARCH_HEADROOM_M above its far edge, in frame pixels."""
    m = SEARCH_MARGIN_M
    if side == "top":
        y_far, y_near = -m, COURT_LENGTH_M / 2
    else:
        y_far, y_near = COURT_LENGTH_M / 2, COURT_LENGTH_M + m
    corners = [(-m, y_far), (COURT_WIDTH_M + m, y_far), (COURT_WIDTH_M + m, y_near), (-m, y_near)]
    cam = np.eye(3) if camera_now is None else np.asarray(camera_now, float)

    def to_frame(c):
        p = court.to_marking(c)
        return _apply(cam, float(p[0]), float(p[1]))

    px = np.array([to_frame(c) for c in corners])
    a = to_frame((COURT_WIDTH_M / 2 - 0.5, y_far))
    b = to_frame((COURT_WIDTH_M / 2 + 0.5, y_far))
    headroom = SEARCH_HEADROOM_M * float(np.hypot(*(b - a)))
    rect = _clip(px[:, 0].min() - SEARCH_PAD_PX, px[:, 1].min() - headroom - SEARCH_PAD_PX,
                 px[:, 0].max() + SEARCH_PAD_PX, px[:, 1].max() + SEARCH_PAD_PX, frame_w, frame_h)
    if rect.width < 32 or rect.height < 32:
        return None
    return rect


def search_imgsz(rect: Rect) -> int:
    """SEARCH_SCALE of the region's long side, in multiples of 32, at most SEARCH_MAX_IMGSZ."""
    size = int(math.ceil(SEARCH_SCALE * max(rect.width, rect.height) / 32.0) * 32)
    return min(SEARCH_MAX_IMGSZ, size)


def _centre(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def pick_follow(dets: Sequence[Detection], last_box: Box, side: str, court: CourtFrame,
                frame_to_marking: Optional[np.ndarray], empty_frames: int = 0) -> Optional[Detection]:
    """The detection that continues the slot's player, or None.

    Near the last box (CONTINUE_RADIUS_H of its height, wider while they were
    missing), with confident ankles on the slot's side of the net in the
    marking frame; the nearest if several.
    """
    h = max(1.0, last_box[3] - last_box[1])
    radius = CONTINUE_RADIUS_H * h * (1.0 + FOLLOW_GROWTH_PER_EMPTY * empty_frames)
    last = _centre(last_box)
    best, best_d = None, float("inf")
    for det in dets:
        gp = ground_point(det)
        if gp is None or court.side_of(to_marking(frame_to_marking, gp)) != side:
            continue
        d = float(np.hypot(*(_centre(det.box) - last)))
        if d <= radius and d < best_d:
            best, best_d = det, d
    return best


def pick_search(dets: Sequence[Detection], side: str, court: CourtFrame,
                frame_to_marking: Optional[np.ndarray]) -> Optional[Detection]:
    """The most confident person standing inside the half's court lines, or None."""
    best = None
    for det in dets:
        gp = ground_point(det)
        if gp is None:
            continue
        p = to_marking(frame_to_marking, gp)
        if court.side_of(p) != side or not court.inside_lines(court.to_court(p), side):
            continue
        if best is None or det.mean_conf > best.mean_conf:
            best = det
    return best


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge(kpts_xy, kpts_conf, boxes, track_ids, picks: Sequence[Tuple[Detection, int]]):
    """The frame's detections with the zoom's picks folded in.

    A pick overlapping a full-frame detection (IoU over MERGE_IOU) replaces its
    keypoints, confidences and box, keeping the tracker's id: the zoomed
    keypoints are the better measurement of the same person. Otherwise it is
    appended with its zoom track id. Missing confidences read 0.5 and missing
    ids -1, the loop's own defaults, so the arrays always line up.
    """
    xy = np.zeros((0, 17, 2)) if kpts_xy is None else np.asarray(kpts_xy, float).reshape(-1, 17, 2).copy()
    n = xy.shape[0]
    conf = np.full((n, 17), 0.5) if kpts_conf is None else np.asarray(kpts_conf, float).reshape(n, 17).copy()
    bx = np.zeros((0, 4)) if boxes is None or n == 0 else np.asarray(boxes, float).reshape(n, 4).copy()
    if bx.shape[0] != n:
        bx = np.zeros((n, 4))
    ids = [-1] * n if track_ids is None else list(track_ids)[:n] + [-1] * max(0, n - len(track_ids))
    for det, zoom_id in picks:
        overlaps = [(_iou(bx[i], det.box), i) for i in range(bx.shape[0])]
        best = max(overlaps, default=(0.0, -1))
        if best[0] > MERGE_IOU:
            i = best[1]
            xy[i] = det.kpts
            conf[i] = det.conf
            bx[i] = det.box
        else:
            xy = np.concatenate([xy, np.asarray(det.kpts, float)[None]], axis=0)
            conf = np.concatenate([conf, np.asarray(det.conf, float)[None]], axis=0)
            bx = np.concatenate([bx, np.asarray(det.box, float)[None]], axis=0)
            ids.append(zoom_id)
    return xy, conf, bx, ids


@dataclass
class _Slot:
    last_box: Optional[Box] = None
    mean_height: Optional[float] = None
    empty_frames: int = LOST_FRAMES
    session: int = 0


@dataclass
class PlayerZoom:
    """The zoom pass's memory: per identity slot, where the player was and how long they have been missing."""
    court: CourtFrame
    frame_w: int
    frame_h: int
    fps: float
    _slots: Dict[int, _Slot] = field(default_factory=lambda: {0: _Slot(), 1: _Slot()})

    def empty_frames(self, pid: int) -> int:
        return self._slots[pid].empty_frames

    def track_id(self, pid: int) -> int:
        """The id a zoom-only detection of this slot is appended under; new for every re-find."""
        return ZOOM_TRACK_ID_BASE + pid * ZOOM_SESSION_STRIDE + self._slots[pid].session

    def _small(self, slot: _Slot) -> bool:
        if slot.mean_height is None:
            return False
        return slot.mean_height * POSE_INPUT_PX / max(self.frame_w, self.frame_h) < ZOOM_TRIGGER_MODEL_PX

    def plan(self, sides: Dict[int, str], camera_now: Optional[np.ndarray],
             marks_describe_frame: bool) -> List[ZoomRequest]:
        """What to look at on this frame, before the loop's filters run."""
        requests = []
        backoff = int(round(SEARCH_BACKOFF_S * self.fps))
        for pid in (0, 1):
            slot, side = self._slots[pid], sides.get(pid)
            if side not in ("top", "bottom"):
                continue
            if slot.last_box is not None and slot.empty_frames < LOST_FRAMES:
                if self._small(slot):
                    rect = follow_window(slot.last_box, self.frame_w, self.frame_h, slot.empty_frames)
                    requests.append(ZoomRequest(pid, side, "follow", rect, ZOOM_IMGSZ))
                continue
            if not marks_describe_frame:
                continue
            if slot.empty_frames > backoff and slot.empty_frames % SEARCH_BACKOFF_EVERY != 0:
                continue
            rect = search_region(self.court, side, camera_now, self.frame_w, self.frame_h)
            if rect is not None:
                requests.append(ZoomRequest(pid, side, "search", rect, search_imgsz(rect)))
        return requests

    def pick(self, request: ZoomRequest, dets: Sequence[Detection],
             frame_to_marking: Optional[np.ndarray]) -> Optional[Detection]:
        """The one detection a request keeps, from its detections mapped back to frame pixels."""
        slot = self._slots[request.pid]
        if request.kind == "follow":
            return pick_follow(dets, slot.last_box, request.side, self.court, frame_to_marking, slot.empty_frames)
        found = pick_search(dets, request.side, self.court, frame_to_marking)
        if found is not None:
            slot.session += 1
        return found

    def observe(self, emitted: Dict[int, Optional[Box]]) -> None:
        """After the identity tracker: each slot's emitted box in frame pixels, or None."""
        for pid in (0, 1):
            slot, box = self._slots[pid], emitted.get(pid)
            if box is None:
                slot.empty_frames += 1
                continue
            h = float(box[3] - box[1])
            slot.mean_height = h if slot.mean_height is None else HEIGHT_EMA * h + (1 - HEIGHT_EMA) * slot.mean_height
            slot.last_box = tuple(float(v) for v in box)
            slot.empty_frames = 0
