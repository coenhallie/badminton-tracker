"""The RawInference ("RAWI") wire format, as Python writes it.

Byte for byte the format defined in badminton-rally-mobile at
analysis/src/commonMain/kotlin/com/badmintontracker/analysis/raw/
RawInferenceCodec.kt. Little-endian, fixed width, no framing.

Three writers now exist and one reader. Swift writes this on iPhone, the
Android layer writes it on device, and this module writes it in the cloud;
RawInferenceCodec.kt reads all three. That asymmetry is the point of the
format: everything with a history of drifting between implementations lives
in :analysis on the phone, written once, and the only thing that crosses a
language boundary is model output.

There is no decoder here on purpose. A second reader could agree with a buggy
writer; the check that matters is scripts/verify_raw_inference_codec.py
--fixture, whose output the Kotlin reader decodes in its own test.

Layout (version 2; this module can no longer write version 1):
  magic        4 bytes  "RAWI"
  version      int32
  fps          float64
  totalFrames  int32
  videoWidth   int32
  videoHeight  int32
  modelVersion int32 length + UTF-8 bytes
  frameCount   int32
  per frame:   frame int32, timestamp float64, shuttleFlag int8,
               [shuttle], boxCount int32, [boxes], personCount int32,
               [persons]
  hasCamera    int8
  [camera]:    referenceTimestamp float64, count int32,
  per sample:  timestamp float64, 8 x float64
               (h00 h01 h02 h10 h11 h12 h20 h21; h22 is always 1.0 and is
               not written)

A stream with no camera ends with a single 0x00 byte after the frames. The
Kotlin reader (RawInferenceCodec.kt, Task 6) reads both version 1 and
version 2 and rejects trailing bytes; this module only ever writes version 2.

Coordinates are SOURCE-VIDEO pixels, matching RawInference.kt:12-16. Anything
in model input space has to be scaled back before it reaches this module.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from camera_motion import CameraMotion

MAGIC = b"RAWI"
VERSION = 2


@dataclass
class RawBox:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class RawKeypoint:
    x: float
    y: float
    confidence: float


@dataclass
class RawPerson:
    box: RawBox
    keypoints: List[RawKeypoint]


@dataclass
class RawShuttle:
    x: float
    y: float
    confidence: float
    visible: bool


@dataclass
class RawFrame:
    frame: int
    timestamp: float
    shuttle: Optional[RawShuttle] = None
    boxes: List[RawBox] = field(default_factory=list)
    persons: List[RawPerson] = field(default_factory=list)


@dataclass
class RawHeader:
    fps: float
    total_frames: int
    video_width: int
    video_height: int
    model_version: str


def _box(b: RawBox) -> bytes:
    return struct.pack("<ifffff", b.class_id, b.confidence, b.x1, b.y1, b.x2, b.y2)


def encode(
    header: RawHeader,
    frames: List[RawFrame],
    camera: Optional["CameraMotion"] = None,
) -> bytes:
    """The whole stream as bytes.

    Built in one bytearray rather than streamed: the caller writes this to a
    file in one go, and a 30-minute match is tens of megabytes, which is well
    inside the worker's 8 GB. If that ever stops being true, the shape to
    reach for is a generator of frame records, not a partial file.

    `camera` is a `camera_motion.CameraMotion` (or `None`, when the caller
    has not tracked the camera): only its `reference_timestamp` and
    `samples` attributes are used, so any duck-typed equivalent works too.
    This module deliberately has no runtime import of camera_motion, so it
    stays a standalone, dependency-free codec; the type is imported for type
    checking only.
    """
    out = bytearray()
    out += MAGIC
    out += struct.pack("<i", VERSION)
    out += struct.pack("<d", header.fps)
    out += struct.pack("<iii", header.total_frames, header.video_width, header.video_height)
    model = header.model_version.encode("utf-8")
    out += struct.pack("<i", len(model))
    out += model
    out += struct.pack("<i", len(frames))
    for f in frames:
        out += struct.pack("<i", f.frame)
        out += struct.pack("<d", f.timestamp)
        if f.shuttle is None:
            out += b"\x00"
        else:
            out += b"\x01"
            out += struct.pack("<fff", f.shuttle.x, f.shuttle.y, f.shuttle.confidence)
            out += b"\x01" if f.shuttle.visible else b"\x00"
        out += struct.pack("<i", len(f.boxes))
        for b in f.boxes:
            out += _box(b)
        out += struct.pack("<i", len(f.persons))
        for p in f.persons:
            out += _box(p.box)
            out += struct.pack("<i", len(p.keypoints))
            for k in p.keypoints:
                out += struct.pack("<fff", k.x, k.y, k.confidence)
    out += _encode_camera(camera)
    return bytes(out)


def _encode_camera(camera: Optional["CameraMotion"]) -> bytes:
    """The trailing camera section: a single 0x00 when there is none.

    `h22` is not written: the design's invariant (camera_motion.py:165,
    "every stored M has h22 exactly 1.0") is checked here rather than
    trusted silently, because a stream that quietly dropped a non-1.0 h22
    would be wrong in a way nothing downstream could detect.
    """
    if camera is None:
        return b"\x00"
    out = bytearray(b"\x01")
    out += struct.pack("<d", camera.reference_timestamp)
    out += struct.pack("<i", len(camera.samples))
    for timestamp, m in camera.samples:
        h22 = float(m[2][2])
        if abs(h22 - 1.0) > 1e-6:
            raise ValueError(f"camera sample at {timestamp} has h22={h22!r}, expected 1.0")
        out += struct.pack("<d", timestamp)
        out += struct.pack(
            "<8d",
            float(m[0][0]), float(m[0][1]), float(m[0][2]),
            float(m[1][0]), float(m[1][1]), float(m[1][2]),
            float(m[2][0]), float(m[2][1]),
        )
    return bytes(out)
