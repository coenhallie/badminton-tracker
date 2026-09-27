"""
Run the Phase 2 pose loop on a local video, on this machine, without Modal.

Usage:
    python backend/scripts/run_phase2_locally.py
    python backend/scripts/run_phase2_locally.py --video PATH --marks PATH --out PATH

The cloud's own evidence for the camera-motion design
(badminton-rally-mobile docs/plans/2026-09-27-camera-motion-design.md
section 8.6) until the owner deploys the worker: `_run_full_yolo_loop` runs
here exactly as `_process_analytics_worker` calls it, and the pose artifact is
built the way the worker builds it, so the phone's codec and selector can read
the result (CloudLocalRunCorpusTest.kt in badminton-rally-mobile).

What is swapped out, and nothing else:

  * the worker's hard-coded `/cache` paths (the BoT-SORT config, the skeleton
    spill file, the tracker-metrics directory) point into a temp directory,
    through the module's own `Path` name, which is how the loop builds them;
  * `MODELS_PATH` points at a temp directory holding the detector as
    `badminton/best.pt`, and the working directory holds the pose weights
    under the bare name the loop loads (`yolo26m-pose.pt`), so ultralytics
    never downloads anything;
  * `send_log` prints and `send_status_update` does nothing, instead of
    writing to Supabase.

Phase 1's values come from a probe of the video, the same fallback the
worker uses when results.json lacks them: `fps` and `total_frames` from
OpenCV. Phase 1's shuttle positions are not available locally (TrackNet is
Phase 1's), so the loop runs with `tracknet_available=False`, as the worker
does when results.json has none.

Device: the loop does not take one. Ultralytics picks CUDA when it is
present and the CPU otherwise, so on a Mac this runs on the CPU (and in
fp32, since the loop only asks for half precision under CUDA).

Needs, beyond backend's own imports: opencv-python, numpy, torch,
ultralytics, and `lap` (BoT-SORT's assignment solver; ultralytics tries to
pip-install it on first use when it is missing).

Nothing here imports Supabase or calls Modal; importing the processor module
does import `modal` (for its app and image definitions), which is inert.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

CORPUS = Path.home() / "shuttl-corpus" / "IMG_5892"
WEIGHTS = Path("/Users/coenhallie/Desktop/projects/badminton-rally-mobile/tools/models/weights")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, default=CORPUS / "source.mov")
    parser.add_argument("--marks", type=Path, default=CORPUS / "marks_ideal.json",
                        help="JSON with the marks under 'manual_court_keypoints'")
    parser.add_argument("--pose-weights", type=Path, default=WEIGHTS / "yolo26m-pose.pt")
    parser.add_argument("--detector-weights", type=Path, default=WEIGHTS / "badminton.pt")
    parser.add_argument("--out", type=Path, default=CORPUS / "cloud-local" / "poses.raw")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many decoded frames (a long video's first stretch)")
    return parser.parse_args()


def _probe(video: Path):
    """fps, frame count and size, read the way the worker's probe reads them."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, width, height


def _limit_decoding(max_frames: int) -> None:
    """Make every VideoCapture opened from here on end after `max_frames`
    reads. The loop opens its own capture and reads until `read()` fails,
    so this is the one place a run can be cut short without touching it.
    """
    import cv2

    real_capture = cv2.VideoCapture

    class LimitedCapture:
        def __init__(self, *args):
            self._capture = real_capture(*args)
            self._left = max_frames

        def read(self):
            if self._left <= 0:
                return False, None
            self._left -= 1
            return self._capture.read()

        def __getattr__(self, name):
            return getattr(self._capture, name)

    cv2.VideoCapture = LimitedCapture


def main() -> int:
    import json

    args = _parse_args()
    for required in (args.video, args.marks, args.pose_weights, args.detector_weights):
        if not required.exists():
            raise SystemExit(f"missing input: {required}")

    import camera_motion
    import modal_supabase_processor as proc

    manual_court_keypoints = json.loads(args.marks.read_text())["manual_court_keypoints"]

    scratch_dir = tempfile.TemporaryDirectory(prefix="phase2-local-")
    scratch = Path(scratch_dir.name)
    cache_dir = scratch / "cache"
    cache_dir.mkdir()
    (scratch / "models" / "badminton").mkdir(parents=True)
    (scratch / "models" / "badminton" / "best.pt").symlink_to(args.detector_weights)
    (scratch / "work").mkdir()
    (scratch / "work" / proc.POSE_MODEL_NAME).symlink_to(args.pose_weights)

    real_path = proc.Path

    def local_path(*parts):
        path = real_path(*parts)
        text = str(path)
        if text == "/cache" or text.startswith("/cache/"):
            return real_path(str(cache_dir) + text[len("/cache"):])
        return path

    proc.Path = local_path
    proc.MODELS_PATH = str(scratch / "models")

    # The loop constructs its own tracker; recording the instance is how this
    # harness reads the accepted and dropped counts without the loop having
    # to export them.
    trackers = []

    class RecordingTracker(camera_motion.CameraMotionTracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            trackers.append(self)

    camera_motion.CameraMotionTracker = RecordingTracker

    async def send_log(message, level="info", category="processing", phase="phase2"):
        print(f"[log {level}/{category}] {message}", flush=True)

    async def send_status_update(status="processing_phase2", progress=0, current_frame=0,
                                 total_frames=0, error=None):
        pass

    probe_fps, total_frames, width, height = _probe(args.video)
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
        _limit_decoding(args.max_frames)
    fps, substituted = proc.normalize_fps(probe_fps)
    print(f"video: {args.video} {width}x{height}, fps {fps:.3f}"
          f"{' (substituted)' if substituted else ''}, {total_frames} frames"
          f"{' (limited by --max-frames)' if args.max_frames is not None else ''}")
    print("device: cpu (the loop takes no device; ultralytics uses CUDA only when present)")

    os.chdir(scratch / "work")
    started = time.time()
    skeleton_frames, tracking_state = asyncio.run(proc._run_full_yolo_loop(
        video_path=args.video,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        video_id="IMG_5892-local",
        manual_court_keypoints=manual_court_keypoints,
        tracknet_positions={},
        tracknet_available=False,
        send_log=send_log,
        send_status_update=send_status_update,
    ))
    elapsed = time.time() - started

    # The artifact, as _process_analytics_worker builds it; only the model
    # version's revision stamp differs, so a local file is never mistaken
    # for a deployed worker's.
    header, frames, camera = proc.raw_inference_from_skeleton_frames(
        skeleton_frames,
        fps=fps,
        total_frames=total_frames,
        video_width=width,
        video_height=height,
        model_version=f"cloud:{proc.POSE_MODEL_NAME}:local-harness",
        camera=tracking_state.get("camera_motion"),
    )
    blob = proc.encode_raw_inference(header, frames, camera=camera)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)

    stats = tracking_state.get("tracker_stats") or {}
    near_ids = [pid for pid in (0, 1) if stats.get(f"player_{pid}_side") == "bottom"]
    far_ids = [pid for pid in (0, 1) if stats.get(f"player_{pid}_side") == "top"]
    near_frames = sum(
        1 for entry in skeleton_frames
        if any(p.get("player_id") in near_ids for p in entry.get("players") or [])
    )
    far_frames = sum(
        1 for entry in skeleton_frames
        if any(p.get("player_id") in far_ids for p in entry.get("players") or [])
    )

    print()
    print(f"frames processed: {tracking_state.get('processed_count')} in {elapsed:.0f} s")
    for tracker in trackers:
        print(f"camera: accepted {tracker.accepted}, dropped {tracker.dropped}")
    if camera is not None:
        print(f"camera in the pose file: {len(camera.samples)} samples, "
              f"reference {camera.reference_timestamp:.3f} s")
    print(f"frames with a near-side player (player {near_ids}): {near_frames} of {len(skeleton_frames)}")
    print(f"frames with a far-side player (player {far_ids}): {far_frames} of {len(skeleton_frames)}")
    print(f"wrote {args.out} ({len(blob) / (1024 * 1024):.1f} MB, {len(frames)} frames)")
    scratch_dir.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
