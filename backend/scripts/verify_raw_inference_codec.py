"""
Byte-level guard for the Python writer of the RawInference (RAWI) format.

Usage:
    python backend/scripts/verify_raw_inference_codec.py [--fixture PATH]

No GPU, no network. Run it after ANY change to:
    backend/raw_inference_codec.py

WHAT THIS DOES NOT DO. It does not decode its own output. The RAWI format has
exactly one reader, RawInferenceCodec.kt in badminton-rally-mobile, and a
Python decoder here would be a second reader capable of agreeing with a buggy
writer. The authoritative check is the fixture this script writes with
--fixture, which that Kotlin reader decodes in RawInferenceCodecTest.

So the checks here are the ones a decoder cannot help with: the magic, the
version, and the exact byte length computed from the layout rather than from
the encoder. A struct format string typo that widens a field shows up here.

Exits non-zero on the first failing assertion group.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from raw_inference_codec import (  # noqa: E402
    MAGIC,
    VERSION,
    RawBox,
    RawFrame,
    RawHeader,
    RawKeypoint,
    RawPerson,
    RawShuttle,
    encode,
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


# Distinct, non-round values everywhere, so a field-order swap cannot pass by
# two fields happening to hold the same number. Mirrors the sample in
# RawInferenceCodecTest.kt.
HEADER = RawHeader(
    fps=59.94,
    total_frames=1234,
    video_width=1920,
    video_height=1080,
    model_version="cloud:yolo11x-pose:abc123",
)

FRAMES = [
    RawFrame(
        frame=7,
        timestamp=1.0 / 3.0,
        shuttle=None,
        boxes=[],
        persons=[
            RawPerson(
                box=RawBox(class_id=1, confidence=0.9, x1=10.0, y1=20.0, x2=30.0, y2=40.0),
                keypoints=[RawKeypoint(i * 1.5, i * 2.5, i * 0.01) for i in range(17)],
            ),
            RawPerson(
                box=RawBox(class_id=1, confidence=0.7, x1=11.0, y1=21.0, x2=31.0, y2=41.0),
                keypoints=[RawKeypoint(i * 1.25, i * 2.25, i * 0.02) for i in range(17)],
            ),
        ],
    ),
    RawFrame(frame=8, timestamp=2.0 / 3.0, shuttle=None, boxes=[], persons=[]),
]

blob = encode(HEADER, FRAMES)

print("the stream identifies itself")
check("magic", blob[:4] == MAGIC, repr(blob[:4]))
check("version is little-endian at offset 4", struct.unpack_from("<i", blob, 4)[0] == VERSION)

print("the header is laid out exactly as the Kotlin reader expects")
# magic 4, version 4, fps 8, totalFrames 4, videoWidth 4, videoHeight 4,
# then an int32 length-prefixed UTF-8 modelVersion, then an int32 frame count.
check("fps at offset 8", struct.unpack_from("<d", blob, 8)[0] == HEADER.fps)
check("totalFrames at offset 16", struct.unpack_from("<i", blob, 16)[0] == HEADER.total_frames)
check("videoWidth at offset 20", struct.unpack_from("<i", blob, 20)[0] == HEADER.video_width)
check("videoHeight at offset 24", struct.unpack_from("<i", blob, 24)[0] == HEADER.video_height)
model_len = struct.unpack_from("<i", blob, 28)[0]
check("modelVersion length", model_len == len(HEADER.model_version.encode("utf-8")))
check(
    "modelVersion bytes",
    blob[32:32 + model_len] == HEADER.model_version.encode("utf-8"),
)
check("frame count follows the string", struct.unpack_from("<i", blob, 32 + model_len)[0] == 2)

print("the total length matches the layout, computed independently of the encoder")
BOX = 4 + 4 + 4 * 4          # classId i32, confidence f32, four f32 corners
KEYPOINT = 4 + 4 + 4         # x f32, y f32, confidence f32
PERSON = BOX + 4 + 17 * KEYPOINT
header_bytes = 4 + 4 + 8 + 4 + 4 + 4 + 4 + model_len + 4
frame_fixed = 4 + 8 + 1 + 4 + 4   # frame, timestamp, shuttleFlag, boxCount, personCount
expected = header_bytes + frame_fixed + 2 * PERSON + frame_fixed
check("byte length", len(blob) == expected, f"got {len(blob)}, expected {expected}")

print("a present shuttle adds exactly its own record")
with_shuttle = encode(
    HEADER,
    [RawFrame(frame=7, timestamp=0.0,
              shuttle=RawShuttle(x=1.0, y=2.0, confidence=0.5, visible=True))],
)
without = encode(HEADER, [RawFrame(frame=7, timestamp=0.0, shuttle=None)])
# x f32, y f32, confidence f32, visible i8
check("13 bytes", len(with_shuttle) - len(without) == 13,
      f"delta {len(with_shuttle) - len(without)}")

print("a non-ASCII model version is length-prefixed in BYTES, not characters")
# The Kotlin reader does bytes(int()).decodeToString(). A length in characters
# would truncate the string and then misread every byte after it.
wide = encode(
    RawHeader(fps=30.0, total_frames=1, video_width=2, video_height=3,
              model_version="cloud:pöse"),
    [],
)
check("byte length prefix", struct.unpack_from("<i", wide, 28)[0] == len("cloud:pöse".encode("utf-8")))

print("an empty stream is well formed")
empty = encode(HEADER, [])
check("frame count zero", struct.unpack_from("<i", empty, 32 + model_len)[0] == 0)
check("nothing after it", len(empty) == header_bytes)

print("skeleton frames project onto RawFrames without losing a player")
from modal_supabase_processor import raw_inference_from_skeleton_frames  # noqa: E402

SKELETON_FRAMES = [
    {
        "frame": 1,
        "timestamp": 0.0333,
        "players": [
            # The real shape Phase 2 builds at modal_supabase_processor.py:3272.
            # No bbox key: the box is derived, not stored.
            {
                "player_id": 0,
                "keypoints": [
                    {"name": f"k{i}", "x": float(i) + 1, "y": float(i) * 2 + 1, "confidence": 0.5}
                    for i in range(17)
                ],
                "center": {"x": 8.0, "y": 16.0},
                "current_speed": 12.4,
            },
            {
                "player_id": 1,
                "keypoints": [
                    {"name": f"k{i}", "x": float(i) + 100, "y": float(i) + 1, "confidence": 0.25}
                    for i in range(17)
                ],
                "center": {"x": 108.0, "y": 8.0},
                "current_speed": 3.1,
            },
        ],
    },
    # A frame the pose model produced nothing for. It must still appear, with
    # zero persons: selectPlayers counts framesWithPose from a non-empty
    # persons list, so dropping the frame and emitting an empty one are
    # different facts about coverage.
    {"frame": 2, "timestamp": 0.0667, "players": []},
]

hdr, frames = raw_inference_from_skeleton_frames(
    SKELETON_FRAMES,
    fps=30.0,
    total_frames=2,
    video_width=1920,
    video_height=1080,
    model_version="cloud:test",
)
check("every frame kept", len(frames) == 2, f"got {len(frames)}")
check("both players on the first frame", len(frames[0].persons) == 2)
check("frame index carried", frames[0].frame == 1)
check("presentation timestamp carried, not frame/fps", frames[0].timestamp == 0.0333)
check("17 keypoints", len(frames[0].persons[0].keypoints) == 17)
check("keypoint values carried", frames[0].persons[0].keypoints[3].x == 4.0)
check("keypoint confidence carried", frames[0].persons[0].keypoints[3].confidence == 0.5)

print("the box is derived from the keypoints, because Phase 2 stores none")
# player_data (modal_supabase_processor.py:3272) has no bbox key. The corners
# come from _bbox_from_player_keypoints, the same helper the thumbnail capture
# uses, so there is one derivation rather than two.
check("box spans the keypoint extent", frames[0].persons[0].box.x1 == 1.0)
check("box spans the keypoint extent", frames[0].persons[0].box.x2 == 17.0)

print("the box confidence RANKS candidates, so it must vary between them")
# NearPlayerSelector picks within one side with a strict `>` from negative
# infinity. Equal confidences would make the FIRST passing candidate win
# rather than the best, and a player and a line judge on the same side would
# resolve arbitrarily. The mean of the person's own keypoint confidences is
# model output, which is what is allowed to cross this boundary.
check("mean keypoint confidence", abs(frames[0].persons[0].box.confidence - 0.5) < 1e-6)
check("and it differs per person", abs(frames[0].persons[1].box.confidence - 0.25) < 1e-6)
check("an empty frame survives as an empty frame", frames[1].persons == [])

print("the artifact carries no shuttle and no boxes beyond the players")
# Spec 3: rally_clips is the rally truth for a cloud video. An artifact with a
# shuttle track is an artifact something could derive rallies from, and the
# invariant is structural rather than remembered.
check("no shuttle", all(f.shuttle is None for f in frames))
check("no loose boxes", all(f.boxes == [] for f in frames))

print("a player with the wrong keypoint count is dropped, not written short")
# A short keypoint list would decode into a PlayerPose that
# NearPlayerSelector.groundPoint rejects by size, silently costing coverage
# with nothing saying why. Dropped here, where it can be counted.
partial = [{"frame": 1, "timestamp": 0.0,
            "players": [{"player_id": 0,
                         "keypoints": [{"name": "k", "x": 1.0, "y": 2.0, "confidence": 0.5}],
                         "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "confidence": 0.5}}]}]
_, short_frames = raw_inference_from_skeleton_frames(
    partial, fps=30.0, total_frames=1, video_width=100, video_height=100,
    model_version="cloud:test",
)
check("dropped", short_frames[0].persons == [], str(short_frames[0].persons))

print("a player whose keypoints are too sparse to bound still contributes them")
# _bbox_from_player_keypoints needs five keypoints at positive coordinates and
# returns None below that. A player it cannot bound is still a player: the
# gates on the phone stand on the ankles, not on the box, so dropping it would
# cost coverage for a reason that has nothing to do with the player.
sparse = [{"frame": 1, "timestamp": 0.0,
           "players": [{"player_id": 0,
                        "keypoints": [
                            {"name": f"k{i}", "x": float(i) if i < 3 else 0.0,
                             "y": float(i) if i < 3 else 0.0, "confidence": 0.9}
                            for i in range(17)
                        ],
                        "center": {"x": 1.0, "y": 1.0}, "current_speed": 0.0}]}]
_, boxless = raw_inference_from_skeleton_frames(
    sparse, fps=30.0, total_frames=1, video_width=100, video_height=100,
    model_version="cloud:test",
)
check("kept", len(boxless[0].persons) == 1)
check("zero-extent box", boxless[0].persons[0].box.x1 == 0.0)
check("but a real confidence, so ranking still works", boxless[0].persons[0].box.confidence == 0.9)

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", help="write the two-person sample to this path for the Kotlin test")
args = parser.parse_args()
if args.fixture:
    Path(args.fixture).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fixture).write_bytes(blob)
    print(f"\nfixture written: {args.fixture} ({len(blob)} bytes)")

if FAILURES:
    print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("\nall checks passed")
