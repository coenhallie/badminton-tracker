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

Layout:
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

Coordinates are SOURCE-VIDEO pixels, matching RawInference.kt:12-16. Anything
in model input space has to be scaled back before it reaches this module.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

MAGIC = b"RAWI"
VERSION = 1


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


def encode(header: RawHeader, frames: List[RawFrame]) -> bytes:
    """The whole stream as bytes.

    Built in one bytearray rather than streamed: the caller writes this to a
    file in one go, and a 30-minute match is tens of megabytes, which is well
    inside the worker's 8 GB. If that ever stops being true, the shape to
    reach for is a generator of frame records, not a partial file.
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
    return bytes(out)
