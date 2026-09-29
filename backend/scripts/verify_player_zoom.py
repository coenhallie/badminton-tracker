"""
Offline regression suite for backend/player_zoom.py, Phase 2's zoom pass.

Usage:
    python backend/scripts/verify_player_zoom.py

No GPU, no network, no model. Run after any change to backend/player_zoom.py.

The design is badminton-rally-mobile
docs/plans/2026-09-29-two-player-tracking-design.md section 4.1: follow a
player too small for the full-frame pass with a window around them, re-find a
lost one by searching their half of the court, and merge what the zoom finds
into the frame's detections for the loop's own filters to judge.

Sections:
  1. follow_window: size, minimum, clipping, growth while empty.
  2. search_region on aSASa's marks, through an identity and a moving camera;
     search_imgsz.
  3. pick_follow: nearest continuing person, on the slot's side, read in the
     marking frame through a non-identity camera.
  4. pick_search: inside the court lines (within ACQUIRE_MARGIN_M) only, most
     confident.
  5. merge: replacement keeps the tracker's id, an append gets the zoom id,
     empty and None inputs stay well formed.
  6. PlayerZoom: what it asks for, when, and how it backs off.
  7. ZoomSwitch: an error in the zoom never fails a run; SHUTTL_ZOOM_PASS
     turns it off without a code change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from player_zoom import (  # noqa: E402
    LOST_FRAMES,
    SEARCH_BACKOFF_EVERY,
    SEARCH_BACKOFF_S,
    ZOOM_IMGSZ,
    ZOOM_TRACK_ID_BASE,
    CourtFrame,
    ZoomSwitch,
    zoom_enabled_from_env,
    Detection,
    PlayerZoom,
    follow_window,
    merge,
    pick_follow,
    pick_search,
    search_imgsz,
    search_region,
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


# aSASa's marks (video d71b0798), 1920x1080, the camera behind the near baseline.
ASASA = {
    "top_left": [615.9945, 580.96796], "top_right": [1009.8512, 579.8045],
    "bottom_right": [1873.3046, 897.26373], "bottom_left": [303.15527, 1041.8121],
    "net_left": [563.37305, 658.48474], "net_right": [1191.8235, 646.70886],
    "service_line_near_left": [585.0017, 626.62354], "service_line_near_right": [1119.3301, 620.0557],
    "service_line_far_left": [531.192, 705.8907], "service_line_far_right": [1294.2153, 684.35443],
    "center_near": [867.6406, 623.1494], "center_far": [944.6793, 694.22003],
    "marked_at_s": 0.1,
}
FW, FH = 1920, 1080
COURT = CourtFrame.from_marks(ASASA)
IDENTITY = np.eye(3)


def shift(dx: float, dy: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def person(ankle_x: float, ankle_y: float, height: float = 115.0, conf: float = 0.9,
           ankle_conf: float = 0.9) -> Detection:
    """A standing person whose ankles are at (ankle_x -/+ 8, ankle_y)."""
    k = np.zeros((17, 2))
    k[:, 0] = ankle_x
    k[:, 1] = np.linspace(ankle_y - height, ankle_y, 17)
    k[15] = (ankle_x - 8, ankle_y)
    k[16] = (ankle_x + 8, ankle_y)
    c = np.full(17, conf)
    c[15] = c[16] = ankle_conf
    box = (ankle_x - 25.0, ankle_y - height - 5.0, ankle_x + 25.0, ankle_y + 5.0)
    return Detection(box=box, kpts=k, conf=c)


print("1. follow_window")
r = follow_window((850.0, 490.0, 910.0, 605.0), FW, FH)
check("centred on the box", abs((r.x0 + r.x1) / 2 - 880) <= 1 and abs((r.y0 + r.y1) / 2 - 547.5) <= 1, str(r))
check("2.6 h wide and 1.8 h tall", abs(r.width - 2.6 * 115) <= 2 and abs(r.height - 1.8 * 115) <= 2, str(r))
r = follow_window((850.0, 560.0, 880.0, 600.0), FW, FH)
check("at least 200 x 160", r.width >= 199 and r.height >= 159, str(r))
r = follow_window((1880.0, 1000.0, 1920.0, 1080.0), FW, FH)
check("clipped at the frame's right and bottom edges", r.x1 == FW and r.y1 == FH and r.x0 < FW - 50, str(r))
r = follow_window((0.0, 0.0, 40.0, 100.0), FW, FH)
check("clipped at the left and top edges", r.x0 == 0 and r.y0 == 0, str(r))
a = follow_window((850.0, 490.0, 910.0, 605.0), FW, FH, empty_frames=0)
b = follow_window((850.0, 490.0, 910.0, 605.0), FW, FH, empty_frames=2)
check("grows while the player is missing (1 + 0.5 per empty frame)", abs(b.width - 2.0 * a.width) <= 2, f"{a} -> {b}")

print("2. search_region")
check("aSASa's marks give a court", COURT is not None)
r = search_region(COURT, "top", IDENTITY, FW, FH)
check("far half plus 2 m, with headroom (the /tmp/asasa crop 295,386..1384,683 within 25 px)",
      r is not None and abs(r.x0 - 295) <= 25 and abs(r.x1 - 1384) <= 25
      and abs(r.y0 - 386) <= 25 and abs(r.y1 - 683) <= 25, str(r))
check("search_imgsz is 1.5x the long side, a multiple of 32",
      r is not None and search_imgsz(r) % 32 == 0 and abs(search_imgsz(r) - 1.5 * r.width) <= 32, str(search_imgsz(r)))
moved = search_region(COURT, "top", shift(100.0), FW, FH)
check("a camera that moved 100 px right moves the region 100 px right",
      moved is not None and abs(moved.x0 - (r.x0 + 100)) <= 2 and abs(moved.x1 - (r.x1 + 100)) <= 2, str(moved))
near = search_region(COURT, "bottom", IDENTITY, FW, FH)
check("the near half reaches the frame's bottom and is clipped to it", near is not None and near.y1 == FH, str(near))
check("search_imgsz never above 1920", near is not None and search_imgsz(near) <= 1920, str(search_imgsz(near)))
check("a half the camera shows none of gives no region", search_region(COURT, "top", shift(5000.0), FW, FH) is None)

print("3. pick_follow")
last = (855.0, 485.0, 905.0, 605.0)  # the far player, ankles near (880, 600)
dets = [person(885.0, 598.0), person(1050.0, 600.0), person(930.0, 760.0, height=250.0)]
got = pick_follow(dets, last, "top", COURT, IDENTITY)
check("the continuing far player is picked", got is dets[0])
got = pick_follow([person(1050.0, 600.0)], last, "top", COURT, IDENTITY)
check("nobody within 0.6 h of the last box: no pick", got is None)
got = pick_follow([person(885.0, 598.0, ankle_conf=0.3)], last, "top", COURT, IDENTITY)
check("unconfident ankles: no pick", got is None)
got = pick_follow([person(885.0, 598.0)], last, "top", COURT, shift(0.0, 100.0))
check("through a camera that maps the ankles below the net, the far slot rejects them", got is None)
got = pick_follow([person(885.0, 598.0)], last, "bottom", COURT, IDENTITY)
check("a far-side person is never the near slot's", got is None)

print("4. pick_search")
umpire = person(1285.0, 645.0, height=170.0, conf=0.95)
player = person(880.0, 600.0, conf=0.6)
bystander = person(560.0, 570.0, height=110.0, conf=0.9)
got = pick_search([umpire, bystander, player], "top", COURT, IDENTITY)
check("the umpire and the bystander, inside the margin but outside the lines, lose to the player", got is player)
got = pick_search([umpire, bystander], "top", COURT, IDENTITY)
check("nobody inside the lines: no pick", got is None)
behind = COURT.to_marking((3.0, -0.4))
got = pick_search([person(float(behind[0]), float(behind[1]))], "top", COURT, IDENTITY)
check("a player 0.4 m behind the far baseline is adopted (ACQUIRE_MARGIN_M)", got is not None)
wide = COURT.to_marking((-1.0, 2.0))
got = pick_search([person(float(wide[0]), float(wide[1]))], "top", COURT, IDENTITY)
check("someone a metre outside the sideline is not", got is None)
a2 = person(700.0, 600.0, conf=0.5)
got = pick_search([a2, player], "top", COURT, IDENTITY)
check("among people inside the lines, the most confident", got is player)

print("5. merge")
xy = np.stack([person(930.0, 760.0).kpts, person(880.0, 600.0).kpts])
cf = np.stack([np.full(17, 0.8), np.full(17, 0.4)])
bx = np.array([person(930.0, 760.0).box, person(880.0, 600.0).box])
zoomed = person(881.0, 601.0, conf=0.97)
out_xy, out_cf, out_bx, out_ids = merge(xy, cf, bx, [7, 9], [(zoomed, ZOOM_TRACK_ID_BASE + 1)])
check("an overlapping zoom detection replaces the row", len(out_ids) == 2 and np.allclose(out_xy[1], zoomed.kpts))
check("and keeps the tracker's id", out_ids == [7, 9], str(out_ids))
check("and its confidences", np.allclose(out_cf[1], zoomed.conf))
fresh = person(1500.0, 600.0)
out_xy, out_cf, out_bx, out_ids = merge(xy, cf, bx, [7, 9], [(fresh, ZOOM_TRACK_ID_BASE + 1)])
check("a new person is appended with the zoom id", len(out_ids) == 3 and out_ids[2] == ZOOM_TRACK_ID_BASE + 1)
check("appended arrays line up", out_xy.shape == (3, 17, 2) and out_cf.shape == (3, 17) and out_bx.shape == (3, 4))
out_xy, out_cf, out_bx, out_ids = merge(np.zeros((0, 17, 2)), None, None, None, [(fresh, ZOOM_TRACK_ID_BASE + 2)])
check("a frame where the full pass found nobody (boxes None) gets the pick alone",
      out_xy.shape == (1, 17, 2) and out_cf.shape == (1, 17) and out_bx.shape == (1, 4) and out_ids == [ZOOM_TRACK_ID_BASE + 2])
out_xy, out_cf, out_bx, out_ids = merge(xy, None, bx, None, [])
check("no picks and no confidences or ids: rows unchanged, confidences 0.5, ids -1",
      out_xy.shape == (2, 17, 2) and np.allclose(out_cf, 0.5) and out_ids == [-1, -1])

other = person(930.0, 700.0, height=200.0)  # boxes overlap (IoU 0.42), joints do not
far_pick = person(935.0, 600.0, height=115.0)
out_xy, out_cf, out_bx, out_ids = merge(other.kpts[None], other.conf[None], np.array([other.box]), [4],
                                        [(far_pick, ZOOM_TRACK_ID_BASE + 1)])
check("an overlapping box of a DIFFERENT person is not replaced: the pick is appended",
      out_ids == [4, ZOOM_TRACK_ID_BASE + 1] and np.allclose(out_xy[0], other.kpts), str(out_ids))

print("6. PlayerZoom")
zoom = PlayerZoom(COURT, FW, FH, fps=30.0)
sides = {0: "top", 1: "bottom"}
reqs = zoom.plan(sides, IDENTITY, marks_describe_frame=True)
check("never filled: both slots search at once", sorted((q.pid, q.kind) for q in reqs) == [(0, "search"), (1, "search")],
      str([(q.pid, q.kind) for q in reqs]))
zoom.observe({0: (855.0, 485.0, 905.0, 605.0), 1: (900.0, 510.0, 990.0, 765.0)})
reqs = zoom.plan(sides, IDENTITY, marks_describe_frame=True)
check("a small filled slot follows; a large one gets nothing",
      [(q.pid, q.kind) for q in reqs] == [(0, "follow")], str([(q.pid, q.kind) for q in reqs]))
check("follow runs at ZOOM_IMGSZ", reqs and reqs[0].imgsz == ZOOM_IMGSZ)
mid = PlayerZoom(COURT, FW, FH, fps=30.0)
mid.observe({0: (855.0, 400.0, 925.0, 600.0), 1: (900.0, 400.0, 1010.0, 780.0)})
check("a mid-size player (100 px at the model's input, IMG_5892's far player) is followed too",
      [(q.pid, q.kind) for q in mid.plan(sides, IDENTITY, marks_describe_frame=True)] == [(0, "follow")])
for _ in range(LOST_FRAMES - 1):
    zoom.observe({0: None, 1: (900.0, 510.0, 990.0, 765.0)})
reqs = zoom.plan(sides, IDENTITY, marks_describe_frame=True)
check("still following while missing for fewer than LOST_FRAMES", [q.kind for q in reqs] == ["follow"])
zoom.observe({0: None, 1: (900.0, 510.0, 990.0, 765.0)})
reqs = zoom.plan(sides, IDENTITY, marks_describe_frame=True)
check("missing for LOST_FRAMES: search the slot's half", [(q.pid, q.kind) for q in reqs] == [(0, "search")])
check("no search on a frame the marks do not describe",
      zoom.plan(sides, IDENTITY, marks_describe_frame=False) == [])
backoff = int(SEARCH_BACKOFF_S * 30.0)
while zoom.empty_frames(0) <= backoff:
    zoom.observe({0: None, 1: (900.0, 510.0, 990.0, 765.0)})
searched = 0
for _ in range(SEARCH_BACKOFF_EVERY * 4):
    searched += any(q.kind == "search" for q in zoom.plan(sides, IDENTITY, marks_describe_frame=True))
    zoom.observe({0: None, 1: (900.0, 510.0, 990.0, 765.0)})
check("after SEARCH_BACKOFF_S missing, one search in every SEARCH_BACKOFF_EVERY frames", searched == 4, str(searched))
before = zoom.track_id(0)
# A search that finds the player starts a new zoom session, so a new track id.
while not any(q.kind == "search" for q in zoom.plan(sides, IDENTITY, marks_describe_frame=True)):
    zoom.observe({0: None, 1: (900.0, 510.0, 990.0, 765.0)})
search = next(q for q in zoom.plan(sides, IDENTITY, marks_describe_frame=True) if q.kind == "search")
found = zoom.pick(search, [player], IDENTITY)
check("a search pick is the player", found is player)
check("and starts a new zoom session (a new track id)", zoom.track_id(0) != before, f"{before} -> {zoom.track_id(0)}")
check("zoom track ids sit above the tracker's", zoom.track_id(0) >= ZOOM_TRACK_ID_BASE)

print("7. ZoomSwitch")
calls = []


def boom():
    calls.append(1)
    raise RuntimeError("CUDA out of memory")


switch = ZoomSwitch(True)
check("a step that raises returns the fallback", switch.run(boom, "full-frame") == "full-frame")
check("and switches the zoom off, keeping why", not switch.on and "CUDA" in (switch.failure or ""))
check("later steps are not attempted", switch.run(boom, "full-frame") == "full-frame" and len(calls) == 1)
check("a working step returns its value", ZoomSwitch(True).run(lambda: 7, None) == 7)
check("off from the start never runs a step", ZoomSwitch(False).run(boom, "x") == "x" and len(calls) == 1)
check("SHUTTL_ZOOM_PASS unset: on", zoom_enabled_from_env({}))
check("SHUTTL_ZOOM_PASS=0 / false / off: off",
      not any(zoom_enabled_from_env({"SHUTTL_ZOOM_PASS": v}) for v in ("0", "false", "off", "OFF")))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
