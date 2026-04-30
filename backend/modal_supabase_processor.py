"""
Modal Video Processor for Supabase Integration

This module provides GPU-accelerated video processing that:
1. Downloads videos from Supabase Storage (signed URL in payload)
2. Processes with YOLO models (including pose estimation)
3. Writes progress updates directly to Supabase Postgres
4. Uploads results back to Supabase Storage
"""
from __future__ import annotations

import os
import sys
import asyncio
import json
import tempfile
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import time

import modal

try:
    # fastapi is installed in the Modal image; guarded so the module is still
    # importable for tooling/tests that don't need the Modal HTTP entry point.
    from fastapi import Request
except ImportError:  # pragma: no cover - only happens outside the Modal image
    Request = Any  # type: ignore[assignment,misc]


# --- Supabase + HMAC helpers --------------------------------------------------
# Shared with backend/modal_pdf_export.py via backend/supabase_helpers.py.
# Mounted into the Modal image below as /root/supabase_helpers.py.
from supabase_helpers import supabase_client, verify_hmac  # noqa: F401  (used in HMAC-protected endpoints)


# --- Rally clip generation ----------------------------------------------------
# Cut the source video into per-rally MP4 clips using ffmpeg stream copy
# (no re-encode), upload each to the 'clips' bucket, and insert a rally_clips
# row. Idempotent via UNIQUE (video_id, rally_index).

def cut_and_upload_rally_clips(video_path: str, rallies: list, video_id: str, owner_id: str):
    """
    For each detected rally, cut the source video using ffmpeg `-c copy`
    (stream copy — fast, no quality loss, cuts on nearest keyframe so clips
    may start up to ~2s before the requested timestamp), upload to the
    'clips' Supabase Storage bucket, and upsert a rally_clips DB row.

    A single bad rally (e.g., end_timestamp past actual video duration) logs
    a warning and continues — partial success is acceptable.
    """
    import subprocess

    sb = supabase_client()

    for rally in rallies:
        rally_id = rally.get("id")
        if rally_id is None:
            continue
        clip_local = f"/cache/{video_id}_rally_{rally_id}.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(rally["start_timestamp"]),
                    "-to", str(rally["end_timestamp"]),
                    "-i", video_path,
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    clip_local,
                ],
                check=True,
                capture_output=True,
                timeout=120,  # generous: stream copy of a 30s clip should be <1s
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = e.stderr.decode(errors="replace") if hasattr(e, "stderr") and e.stderr else str(e)
            try:
                sb.table("processing_logs").insert({
                    "video_id": video_id,
                    "owner_id": owner_id,
                    "message": f"clip generation failed for rally {rally_id}: {stderr[:200]}",
                    "level": "warning",
                    "category": "processing",
                }).execute()
            except Exception:
                pass  # don't let logging failure crash the pipeline
            # remove any partial clip file
            try:
                if os.path.exists(clip_local):
                    os.remove(clip_local)
            except OSError:
                pass
            continue

        storage_path = f"{owner_id}/{video_id}/rally_{rally_id}.mp4"
        try:
            with open(clip_local, "rb") as f:
                sb.storage.from_("clips").upload(
                    path=storage_path,
                    file=f.read(),
                    file_options={"content-type": "video/mp4", "upsert": "true"},
                )
            sb.table("rally_clips").upsert({
                "video_id":          video_id,
                "owner_id":          owner_id,
                "rally_index":       rally_id,
                "start_timestamp":   rally["start_timestamp"],
                "end_timestamp":     rally["end_timestamp"],
                "duration_seconds":  rally["duration_seconds"],
                "clip_storage_path": storage_path,
            }, on_conflict="video_id,rally_index").execute()
        except Exception as e:
            try:
                sb.table("processing_logs").insert({
                    "video_id": video_id,
                    "owner_id": owner_id,
                    "message": f"clip upload/insert failed for rally {rally_id}: {str(e)[:200]}",
                    "level": "warning",
                    "category": "processing",
                }).execute()
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(clip_local):
                    os.remove(clip_local)
            except OSError:
                pass


def compute_homography_matrix(corners, court_width=6.1, court_length=13.4):
    try:
        import cv2
        import numpy as np
        src = np.array(corners, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [court_width, 0],
            [court_width, court_length],
            [0, court_length],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return matrix
    except Exception:
        return None


# =============================================================================
# POSE CLASSIFICATION HELPERS
# =============================================================================
# These functions classify badminton-specific poses from YOLO keypoints.
# Based on pose_detection.py PoseAnalyzer class (simplified for Modal deployment)

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes should have x1, y1, x2, y2 keys.
    """
    # Calculate intersection
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def skeleton_bbox_from_keypoints(kpts) -> Optional[Dict]:
    """
    Calculate bounding box from skeleton keypoints.
    Returns None if insufficient keypoints are valid.
    """
    valid_points = [(float(pt[0]), float(pt[1])) for pt in kpts if pt[0] > 0 and pt[1] > 0]
    
    if len(valid_points) < 5:  # Need at least 5 valid keypoints
        return None
    
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    return {
        "x1": min(xs),
        "y1": min(ys),
        "x2": max(xs),
        "y2": max(ys)
    }


def _pick_best_two_player_frame(
    skeleton_frames: List[Dict],
    video_height: int,
    max_search_seconds: float,
    fps: float,
) -> Optional[Dict]:
    """
    Pick the frame best suited for player-identity thumbnails.

    Criteria (in priority order):
      1. Exactly 2 players detected in the frame.
      2. Both have valid center coordinates.
      3. Their center Y-coordinates are separated by >=25% of video_height
         (so bounding boxes don't overlap).
      4. Sum of keypoint confidences across both players is maximal.

    Prefer frames within the first max_search_seconds; fall back to
    "best anywhere" if no early frame qualifies.

    Returns the chosen frame dict or None if no qualifying frame exists.
    """
    min_sep_px = video_height * 0.25
    early_cutoff_frame = int(max_search_seconds * fps)

    def _center_xy(c):
        if c is None:
            return None
        if isinstance(c, dict):
            x = c.get("x")
            y = c.get("y")
            if x is None or y is None:
                return None
            return float(x), float(y)
        try:
            return float(c[0]), float(c[1])
        except (KeyError, IndexError, TypeError):
            return None

    def score(frame: Dict) -> float:
        players = frame.get("players") or []
        if len(players) != 2:
            return -1.0
        c0 = _center_xy(players[0].get("center"))
        c1 = _center_xy(players[1].get("center"))
        if not c0 or not c1:
            return -1.0
        if abs(c0[1] - c1[1]) < min_sep_px:
            return -1.0
        conf_sum = 0.0
        for p in players:
            for kp in p.get("keypoints") or []:
                if isinstance(kp, dict) and "confidence" in kp:
                    conf_sum += float(kp.get("confidence") or 0)
                elif isinstance(kp, (list, tuple)) and len(kp) >= 3:
                    conf_sum += float(kp[2])
                elif isinstance(kp, (list, tuple)) and len(kp) >= 2 and kp[0] and kp[1]:
                    conf_sum += 1.0
        return conf_sum

    best_early: Optional[Dict] = None
    best_early_score = -1.0
    best_any: Optional[Dict] = None
    best_any_score = -1.0
    for frame in skeleton_frames:
        s = score(frame)
        if s <= 0:
            continue
        frame_num = int(frame.get("frame") or 0)
        if frame_num <= early_cutoff_frame and s > best_early_score:
            best_early = frame
            best_early_score = s
        if s > best_any_score:
            best_any = frame
            best_any_score = s

    return best_early or best_any


def _bbox_from_player_keypoints(player: Dict) -> Optional[Dict]:
    """
    Compute a tight bbox around a player's keypoints.
    Handles both dict-form keypoints ({name, x, y, confidence}) and
    raw-array form ([x, y, ...]).

    Returns {x1, y1, x2, y2} or None when insufficient valid keypoints.
    """
    kpts = player.get("keypoints") or []
    xs: List[float] = []
    ys: List[float] = []
    for kp in kpts:
        if isinstance(kp, dict):
            x = kp.get("x")
            y = kp.get("y")
        elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
            x, y = kp[0], kp[1]
        else:
            continue
        if x is None or y is None:
            continue
        try:
            fx = float(x)
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if fx <= 0 or fy <= 0:
            continue
        xs.append(fx)
        ys.append(fy)
    if len(xs) < 5:
        return None
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def _capture_player_thumbnails(
    video_path: Path,
    frame_number: int,
    players: List[Dict],
    padding_ratio: float = 0.15,
) -> Optional[Tuple[bytes, bytes]]:
    """
    Decode the given frame from disk, crop each player's bounding box with
    padding, JPEG-encode at ~85% quality.

    Players may carry a bbox directly; otherwise one is derived from their
    keypoints.

    Returns (player_0_jpeg, player_1_jpeg) in the incoming list order -
    caller is responsible for matching that order to player_id. Returns
    None on any failure (disk read, empty bbox, encode error).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, img = cap.read()
        if not ok or img is None:
            return None
    finally:
        cap.release()

    h, w = img.shape[:2]
    out: List[bytes] = []
    for p in players:
        bbox = p.get("bbox")
        if not bbox:
            bbox = _bbox_from_player_keypoints(p)
        if not bbox:
            return None
        if isinstance(bbox, dict):
            x1 = bbox.get("x1")
            y1 = bbox.get("y1")
            x2 = bbox.get("x2")
            y2 = bbox.get("y2")
        else:
            try:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            except (IndexError, TypeError):
                return None
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        try:
            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        except (TypeError, ValueError):
            return None
        pad_x = (x2f - x1f) * padding_ratio
        pad_y = (y2f - y1f) * padding_ratio
        ix1 = max(0, int(x1f - pad_x))
        iy1 = max(0, int(y1f - pad_y))
        ix2 = min(w, int(x2f + pad_x))
        iy2 = min(h, int(y2f + pad_y))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = img[iy1:iy2, ix1:ix2]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        out.append(buf.tobytes())
    if len(out) != 2:
        return None
    return out[0], out[1]


def skeleton_center_from_keypoints(kpts) -> Optional[Tuple[float, float]]:
    """
    Calculate center position from skeleton keypoints (ankle/feet midpoint preferred).
    Uses feet position for accurate court positioning - this is where the player
    actually stands on the court, not where their torso is.
    
    Returns None if insufficient keypoints.
    """
    # COCO keypoint indices
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12
    
    # Primary: Try ankle midpoint first (most accurate for court position)
    left_ankle = kpts[LEFT_ANKLE_IDX] if len(kpts) > LEFT_ANKLE_IDX else None
    right_ankle = kpts[RIGHT_ANKLE_IDX] if len(kpts) > RIGHT_ANKLE_IDX else None
    
    if left_ankle is not None and right_ankle is not None:
        la_x, la_y = float(left_ankle[0]), float(left_ankle[1])
        ra_x, ra_y = float(right_ankle[0]), float(right_ankle[1])
        # Check if both ankles are valid (position > 0)
        if la_x > 0 and la_y > 0 and ra_x > 0 and ra_y > 0:
            return ((la_x + ra_x) / 2, (la_y + ra_y) / 2)
    
    # Fallback 1: Use hip midpoint if ankles not visible (far players)
    left_hip = kpts[LEFT_HIP_IDX] if len(kpts) > LEFT_HIP_IDX else None
    right_hip = kpts[RIGHT_HIP_IDX] if len(kpts) > RIGHT_HIP_IDX else None
    
    if left_hip is not None and right_hip is not None:
        lh_x, lh_y = float(left_hip[0]), float(left_hip[1])
        rh_x, rh_y = float(right_hip[0]), float(right_hip[1])
        if lh_x > 0 and lh_y > 0 and rh_x > 0 and rh_y > 0:
            return ((lh_x + rh_x) / 2, (lh_y + rh_y) / 2)
    
    # Fallback 2: center of all valid keypoints
    valid_points = [(float(pt[0]), float(pt[1])) for pt in kpts if pt[0] > 0 and pt[1] > 0]
    if len(valid_points) < 3:
        return None
    
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def calculate_movement(history: List[Dict], current_pos: Tuple[float, float]) -> float:
    """
    Calculate total movement over history of positions.
    Returns total displacement in pixels.
    """
    if not history:
        return 0.0
    
    total_movement = 0.0
    prev_pos = (history[0]["x"], history[0]["y"])
    
    for entry in history[1:]:
        curr = (entry["x"], entry["y"])
        dx = curr[0] - prev_pos[0]
        dy = curr[1] - prev_pos[1]
        total_movement += (dx**2 + dy**2)**0.5
        prev_pos = curr
    
    # Add movement from last history to current
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    total_movement += (dx**2 + dy**2)**0.5
    
    return total_movement


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate angle at p2 between lines p1-p2 and p2-p3 (in degrees)"""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def get_keypoint_position(keypoints: List[Dict], name: str, min_conf: float = 0.3) -> Optional[Tuple[float, float]]:
    """Get position of a named keypoint with sufficient confidence"""
    kp = next((k for k in keypoints if k["name"] == name), None)
    if kp and kp.get("confidence", 0) >= min_conf and kp.get("x") is not None and kp.get("y") is not None:
        return (kp["x"], kp["y"])
    return None


def calculate_body_angles(keypoints: List[Dict]) -> Dict[str, float]:
    """Calculate body angles from keypoints for pose classification"""
    angles = {
        "left_elbow": 0.0,
        "right_elbow": 0.0,
        "left_shoulder": 0.0,
        "right_shoulder": 0.0,
        "left_knee": 0.0,
        "right_knee": 0.0,
        "left_hip": 0.0,
        "right_hip": 0.0,
        "torso_lean": 0.0,
        "arm_raise": 0.0,
    }
    
    # Get joint positions (low confidence threshold for far players)
    ls = get_keypoint_position(keypoints, "left_shoulder", 0.1)
    rs = get_keypoint_position(keypoints, "right_shoulder", 0.1)
    le = get_keypoint_position(keypoints, "left_elbow", 0.1)
    re = get_keypoint_position(keypoints, "right_elbow", 0.1)
    lw = get_keypoint_position(keypoints, "left_wrist", 0.1)
    rw = get_keypoint_position(keypoints, "right_wrist", 0.1)
    lh = get_keypoint_position(keypoints, "left_hip", 0.1)
    rh = get_keypoint_position(keypoints, "right_hip", 0.1)
    lk = get_keypoint_position(keypoints, "left_knee", 0.1)
    rk = get_keypoint_position(keypoints, "right_knee", 0.1)
    la = get_keypoint_position(keypoints, "left_ankle", 0.1)
    ra = get_keypoint_position(keypoints, "right_ankle", 0.1)
    
    # Left elbow angle (shoulder-elbow-wrist)
    if ls and le and lw:
        angles["left_elbow"] = calculate_angle(ls, le, lw)
    
    # Right elbow angle
    if rs and re and rw:
        angles["right_elbow"] = calculate_angle(rs, re, rw)
    
    # Left shoulder angle (elbow-shoulder-hip)
    if le and ls and lh:
        angles["left_shoulder"] = calculate_angle(le, ls, lh)
    
    # Right shoulder angle
    if re and rs and rh:
        angles["right_shoulder"] = calculate_angle(re, rs, rh)
    
    # Left knee angle (hip-knee-ankle)
    if lh and lk and la:
        angles["left_knee"] = calculate_angle(lh, lk, la)
    
    # Right knee angle
    if rh and rk and ra:
        angles["right_knee"] = calculate_angle(rh, rk, ra)
    
    # Left hip angle (shoulder-hip-knee)
    if ls and lh and lk:
        angles["left_hip"] = calculate_angle(ls, lh, lk)
    
    # Right hip angle
    if rs and rh and rk:
        angles["right_hip"] = calculate_angle(rs, rh, rk)
    
    # Torso lean (vertical alignment of shoulders to hips)
    if ls and rs and lh and rh:
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx = shoulder_mid[0] - hip_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]  # Y increases downward
        angles["torso_lean"] = math.degrees(math.atan2(dx, dy))
    
    # Arm raise (average height of wrists relative to shoulders)
    if lw and rw and ls and rs:
        shoulder_y = (ls[1] + rs[1]) / 2
        wrist_y = (lw[1] + rw[1]) / 2
        angles["arm_raise"] = shoulder_y - wrist_y  # Positive = above shoulders
    
    return angles


def classify_pose(keypoints: List[Dict]) -> Dict[str, Any]:
    """
    Classify the pose type based on keypoints.
    
    Returns dict with pose_type and confidence.
    
    Badminton pose types:
    - smash: Arm high, body leaning back
    - overhead: Arm raised to hit above head
    - serving: Arm raised, upright stance
    - lunge: One leg significantly more bent
    - forehand/backhand: Arm extended to side
    - ready: Slight crouch, balanced stance
    - standing: Upright, relaxed
    - recovery: Moving back to center
    """
    angles = calculate_body_angles(keypoints)
    
    # Get key positions for classification
    nose = get_keypoint_position(keypoints, "nose", 0.1)
    lw = get_keypoint_position(keypoints, "left_wrist", 0.1)
    rw = get_keypoint_position(keypoints, "right_wrist", 0.1)
    ls = get_keypoint_position(keypoints, "left_shoulder", 0.1)
    rs = get_keypoint_position(keypoints, "right_shoulder", 0.1)
    
    # Calculate useful metrics
    wrist_above_head = False
    wrist_above_shoulder = False
    arm_extended_wide = False
    
    # Check wrist positions
    if nose:
        if lw and lw[1] < nose[1]:
            wrist_above_head = True
        if rw and rw[1] < nose[1]:
            wrist_above_head = True
    
    if ls and lw and lw[1] < ls[1]:
        wrist_above_shoulder = True
    if rs and rw and rw[1] < rs[1]:
        wrist_above_shoulder = True
    
    # Check arm extension
    if nose:
        body_center_x = nose[0]
        if lw and abs(lw[0] - body_center_x) > 80:
            arm_extended_wide = True
        if rw and abs(rw[0] - body_center_x) > 80:
            arm_extended_wide = True
    
    # Knee bend analysis
    left_knee = angles["left_knee"]
    right_knee = angles["right_knee"]
    left_knee_bent = left_knee < 150 if left_knee > 0 else False
    right_knee_bent = right_knee < 150 if right_knee > 0 else False
    deep_knee_bend = left_knee < 120 or right_knee < 120
    knee_diff = abs(left_knee - right_knee) if (left_knee > 0 and right_knee > 0) else 0
    
    torso_lean = angles["torso_lean"]
    arm_raise = angles["arm_raise"]
    
    # Classification (priority order)
    pose_type = "ready"  # Default
    confidence = 0.5
    
    # 1. SMASH - Arm high, body leaning back
    if wrist_above_head and arm_raise > 30:
        if torso_lean < -5:
            pose_type = "smash"
            confidence = 0.85
    
    # 2. OVERHEAD - Arm raised above head
    elif wrist_above_head:
        pose_type = "overhead"
        confidence = 0.8
    
    # 3. SERVING - Arm raised, upright
    elif wrist_above_shoulder and abs(torso_lean) < 20:
        if not arm_extended_wide:
            pose_type = "serving"
            confidence = 0.75
    
    # 4. LUNGE - Asymmetric leg bend
    elif knee_diff > 25 and deep_knee_bend:
        pose_type = "lunge"
        confidence = 0.8
    
    # 5. FOREHAND/BACKHAND - Arm extended to side
    elif arm_extended_wide and not wrist_above_shoulder:
        if lw and rw and nose:
            if abs(lw[0] - nose[0]) > abs(rw[0] - nose[0]):
                pose_type = "backhand"
            else:
                pose_type = "forehand"
            confidence = 0.7
        else:
            pose_type = "forehand"
            confidence = 0.6
    
    # 6. READY - Balanced crouch
    elif left_knee_bent and right_knee_bent and knee_diff < 20:
        if abs(torso_lean) < 25:
            pose_type = "ready"
            confidence = 0.75
    
    # 7. RECOVERY - Moving back to position
    elif 15 < abs(torso_lean) < 35 and not deep_knee_bend:
        pose_type = "recovery"
        confidence = 0.6
    
    # 8. STANDING - Upright, relaxed
    elif left_knee > 155 and right_knee > 155 and abs(torso_lean) < 15:
        pose_type = "standing"
        confidence = 0.7
    
    return {
        "pose_type": pose_type,
        "confidence": confidence,
        "body_angles": angles
    }

# =============================================================================
# =============================================================================
# RAW TRACKER METRICS ACCUMULATOR (BoT-SORT diagnostics)
# =============================================================================
# Records unsupervised tracking-quality metrics across a video — no ground
# truth needed. Relies on badminton priors: ~2 players, persistent IDs, smooth
# motion, ~static court positions. Writes a summary JSON + per-frame JSONL to
# /cache/tracker_metrics/ and echoes a single-line [tracker-metrics] record to
# stdout per run so Modal logs can be scraped for anomalies over time.

class TrackerMetricsAccumulator:
    """
    Per-video accumulator for raw-tracker diagnostics. Fed directly from the
    raw tracker output BEFORE the downstream PlayerIdentityTracker runs, so
    the numbers reflect the tracker's own quality, not post-processing.

    Headline metrics:
      - id_switches_per_min    (lower is better)
      - coverage_any_player    (higher is better, ~1.0 ideal)
      - unique_track_ids       (high value = fragmentation)
      - teleports_per_min      (lower is better)
    """

    def __init__(self, video_id: str, tracker_type: str, fps: float,
                 width: int, height: int, total_frames: int,
                 output_dir: str = "/cache/tracker_metrics"):
        self.video_id = video_id
        self.tracker_type = tracker_type
        self.fps = float(fps) if fps else 0.0
        self.width = int(width)
        self.height = int(height)
        self.total_frames = int(total_frames)
        self.started_at = time.time()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_path = self.output_dir / f"{video_id}_{tracker_type}_frames.jsonl"
        self._frames_file = None  # lazy

        # Frame-level counters
        self.frames_seen = 0
        self.frames_with_any_track = 0
        self.frames_with_2_tracks = 0
        self.frames_with_untracked_det = 0
        self.total_detections = 0
        self.untracked_detections = 0

        # Per-track aggregates
        self.track_first_frame: Dict[int, int] = {}
        self.track_last_frame: Dict[int, int] = {}
        self.track_frame_count: Dict[int, int] = {}
        self.track_bbox_heights: Dict[int, list] = {}
        self.track_step_distances: Dict[int, list] = {}
        self.track_side_counts: Dict[int, Dict[str, int]] = {}
        self.track_keypoint_jitter: Dict[int, list] = {}
        self._track_prev_kp: Dict[int, tuple] = {}

        # Anomaly event counters
        self.teleport_events = 0
        self.court_flip_events = 0
        self.id_swap_events = 0

        # Previous-frame state for swap / teleport / side-flip detection
        self._prev_centers: Dict[int, tuple] = {}
        self._prev_side: Dict[int, str] = {}

    def _ensure_frames_file(self):
        if self._frames_file is None:
            self._frames_file = open(self._frames_path, "w")

    def update(self, frame_idx: int, frame_pts: float,
               track_ids, boxes, kpts_xy=None, kpts_conf=None,
               kp_conf_thresh: float = 0.3):
        """
        Record raw tracker output for one frame. Never raises — on any error,
        logs and no-ops so metric collection can't kill the processing run.
        """
        try:
            self._update_impl(frame_idx, frame_pts, track_ids, boxes,
                              kpts_xy, kpts_conf, kp_conf_thresh)
        except Exception as e:
            print(f"[tracker-metrics] update failed @ frame {frame_idx}: {e}")

    def _update_impl(self, frame_idx, frame_pts, track_ids, boxes,
                     kpts_xy, kpts_conf, kp_conf_thresh):
        self.frames_seen += 1

        tids = list(track_ids) if track_ids is not None else []
        n_dets = len(tids)
        n_untracked = sum(1 for t in tids if t is None or int(t) < 0)
        n_tracked = n_dets - n_untracked

        self.total_detections += n_dets
        self.untracked_detections += n_untracked
        if n_tracked > 0:
            self.frames_with_any_track += 1
        if n_tracked == 2:
            self.frames_with_2_tracks += 1
        if n_untracked > 0:
            self.frames_with_untracked_det += 1

        current_centers: Dict[int, tuple] = {}
        current_sides: Dict[int, str] = {}
        per_track_rows = []
        midline_x = self.width / 2.0 if self.width > 0 else 0.0

        for i, tid in enumerate(tids):
            if tid is None:
                continue
            tid = int(tid)
            if tid < 0:
                continue
            if boxes is None or i >= len(boxes):
                continue
            bx = boxes[i]
            x1, y1, x2, y2 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bbox_h = max(0.0, y2 - y1)

            if tid not in self.track_first_frame:
                self.track_first_frame[tid] = frame_idx
                self.track_frame_count[tid] = 0
                self.track_bbox_heights[tid] = []
                self.track_step_distances[tid] = []
                self.track_side_counts[tid] = {"left": 0, "right": 0}
                self.track_keypoint_jitter[tid] = []
            self.track_last_frame[tid] = frame_idx
            self.track_frame_count[tid] += 1
            self.track_bbox_heights[tid].append(bbox_h)
            side = "left" if cx < midline_x else "right"
            self.track_side_counts[tid][side] += 1
            current_sides[tid] = side

            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                step = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                self.track_step_distances[tid].append(step)
                if self.width > 0 and step > self.width * 0.25:
                    self.teleport_events += 1
            if tid in self._prev_side and self._prev_side[tid] != side:
                self.court_flip_events += 1

            if kpts_xy is not None and kpts_conf is not None and i < len(kpts_xy):
                curr_xy = kpts_xy[i]
                curr_c = kpts_conf[i]
                prev = self._track_prev_kp.get(tid)
                if prev is not None:
                    prev_xy, prev_c = prev
                    jitters = []
                    for k in range(min(len(curr_xy), len(prev_xy))):
                        if (float(curr_c[k]) > kp_conf_thresh and
                                float(prev_c[k]) > kp_conf_thresh):
                            dx = float(curr_xy[k][0]) - float(prev_xy[k][0])
                            dy = float(curr_xy[k][1]) - float(prev_xy[k][1])
                            jitters.append((dx * dx + dy * dy) ** 0.5)
                    if jitters:
                        jitters.sort()
                        self.track_keypoint_jitter[tid].append(jitters[len(jitters) // 2])
                self._track_prev_kp[tid] = (curr_xy, curr_c)

            current_centers[tid] = (cx, cy)
            per_track_rows.append({
                "tid": tid,
                "cx": round(cx, 1), "cy": round(cy, 1),
                "h": round(bbox_h, 1),
                "side": side,
            })

        # ID-swap proxy: if curr_tid's nearest prev track is NOT curr_tid,
        # yet curr_tid existed in prev, flag a swap. Guarded by a max distance
        # so far-apart matches (both players are apart) don't count.
        if self.width > 0 and len(self._prev_centers) >= 2 and len(current_centers) >= 2:
            max_d = self.width * 0.1
            for curr_tid, (cx, cy) in current_centers.items():
                best_d = float("inf")
                best_prev = None
                for prev_tid, (px, py) in self._prev_centers.items():
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_prev = prev_tid
                if (best_prev is not None and best_prev != curr_tid and
                        curr_tid in self._prev_centers and best_d < max_d):
                    self.id_swap_events += 1

        self._prev_centers = current_centers
        self._prev_side = current_sides

        # Per-frame JSONL row (raw tracker slice only — keep it lean).
        self._ensure_frames_file()
        self._frames_file.write(json.dumps({
            "frame": frame_idx,
            "t": round(frame_pts, 3),
            "n_det": n_dets,
            "n_tracked": n_tracked,
            "n_untracked": n_untracked,
            "tracks": per_track_rows,
        }) + "\n")

    def finalize(self, identity_tracker_stats: Optional[Dict] = None) -> Dict:
        """Write summary JSON to disk and echo a single-line record to stdout."""
        try:
            if self._frames_file is not None:
                self._frames_file.flush()
                self._frames_file.close()
        except Exception:
            pass

        duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0

        tracks_out = []
        for tid, cnt in self.track_frame_count.items():
            steps = sorted(self.track_step_distances.get(tid, []))
            median_step = steps[len(steps) // 2] if steps else 0.0
            max_step = steps[-1] if steps else 0.0
            heights = self.track_bbox_heights.get(tid, [])
            min_h = min(heights) if heights else 0.0
            max_h = max(heights) if heights else 0.0
            sides = self.track_side_counts.get(tid, {"left": 0, "right": 0})
            tot = sides["left"] + sides["right"]
            side_consistency = max(sides["left"], sides["right"]) / tot if tot > 0 else 0
            jitter = sorted(self.track_keypoint_jitter.get(tid, []))
            median_jitter = jitter[len(jitter) // 2] if jitter else 0.0
            tracks_out.append({
                "track_id": tid,
                "frames": cnt,
                "first_frame": self.track_first_frame[tid],
                "last_frame": self.track_last_frame[tid],
                "lifetime_frames": self.track_last_frame[tid] - self.track_first_frame[tid] + 1,
                "median_step_px": round(median_step, 2),
                "max_step_px": round(max_step, 2),
                "bbox_h_min": round(min_h, 1),
                "bbox_h_max": round(max_h, 1),
                "bbox_h_range": round(max_h - min_h, 1),
                "side_consistency": round(side_consistency, 3),
                "majority_side": "left" if sides["left"] >= sides["right"] else "right",
                "median_keypoint_jitter_px": round(median_jitter, 2),
            })
        tracks_out.sort(key=lambda t: t["frames"], reverse=True)
        long_tracks = [t for t in tracks_out if t["frames"] >= 30]

        coverage_2 = self.frames_with_2_tracks / self.frames_seen if self.frames_seen > 0 else 0
        coverage_any = self.frames_with_any_track / self.frames_seen if self.frames_seen > 0 else 0
        untracked_pct = self.untracked_detections / self.total_detections if self.total_detections > 0 else 0

        summary = {
            "video_id": self.video_id,
            "tracker_type": self.tracker_type,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "frames_seen": self.frames_seen,
            "duration_sec": round(duration_sec, 2),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "id_switches": self.id_swap_events,
            "id_switches_per_min": round(self.id_swap_events / duration_min, 2) if duration_min > 0 else 0,
            "coverage_both_players": round(coverage_2, 4),
            "coverage_any_player": round(coverage_any, 4),
            "unique_track_ids": len(self.track_frame_count),
            "long_tracks_count": len(long_tracks),
            "teleport_events": self.teleport_events,
            "teleports_per_min": round(self.teleport_events / duration_min, 2) if duration_min > 0 else 0,
            "court_side_flips": self.court_flip_events,
            "untracked_detection_pct": round(untracked_pct, 4),
            "total_detections": self.total_detections,
            "untracked_detections": self.untracked_detections,
            "tracks": tracks_out,
            "identity_tracker": identity_tracker_stats or {},
            "frames_jsonl_path": str(self._frames_path),
        }

        summary_path = self.output_dir / f"{self.video_id}_{self.tracker_type}_summary.json"
        try:
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"[tracker-metrics] failed to write summary: {e}")

        headline = {k: v for k, v in summary.items()
                    if k not in ("tracks", "identity_tracker", "frames_jsonl_path")}
        print(f"[tracker-metrics] {json.dumps(headline)}")
        print(f"[tracker-metrics] summary_path={summary_path}")
        print(f"[tracker-metrics] frames_path={self._frames_path}")
        return summary


# =============================================================================
# ROBUST PLAYER IDENTITY TRACKER
# =============================================================================
# Prevents skeleton ID swapping by maintaining stable player-to-skeleton mapping
# using velocity prediction, court-side priors, appearance heuristics, and
# swap detection/correction.

class PlayerIdentityTracker:
    """
    Maintains consistent player_id ↔ skeleton mapping throughout a video session.

    Design principles:
    1. CALIBRATION PHASE: First frame locks initial assignment by Y-sort,
       subsequent calibration frames use track-ID continuity (no re-sorting)
    2. VELOCITY PREDICTION: Predict expected position using exponential moving average
    3. COMPOSITE COST MATCHING: Combine distance, velocity, court-side, area,
       track-ID, and assignment-stickiness costs
    4. MAJORITY-VOTE SMOOTHING: Only switch assignment when the new mapping
       dominates a sliding window (prevents single-frame flicker)
    5. SWAP DETECTION: Detect when both assignments violate court-side consistency
    6. SWAP CORRECTION: Re-associate when swap confirmed over 2 consecutive frames

    Convention:
    - Player 0 = top of video (far side of court, smaller Y)
    - Player 1 = bottom of video (near side of court, larger Y)
    """

    def __init__(
        self,
        video_height: float,
        fps: float,
        video_width: float = 1920.0,
        net_left: Optional[Tuple[float, float]] = None,
        net_right: Optional[Tuple[float, float]] = None,
    ):
        self.video_height = video_height
        self.video_width = video_width
        self.fps = fps
        self.MAX_INST_VEL = max(40.0, 0.035 * max(video_width, video_height))

        # Court-side split: when manual net-line keypoints are provided, use
        # the actual net as the top/bottom divider. Otherwise fall back to the
        # video's pixel centre — which only works for near-perfect overhead
        # shots and fails badly on any tilted camera.
        if net_left is not None and net_right is not None:
            self.net_line: Optional[Tuple[float, float, float, float]] = (
                net_left[0], net_left[1], net_right[0], net_right[1]
            )
            # Fallback midline used only when computing side for very low / very high x
            # that falls outside the net-endpoint range.
            self.court_midline_y = (net_left[1] + net_right[1]) / 2.0
        else:
            self.net_line = None
            self.court_midline_y = video_height / 2.0

        # Per-player state (indexed by player_id: 0 or 1)
        self.positions: Dict[int, List[Dict]] = {0: [], 1: []}  # history of {x, y, frame}
        self.velocities: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0), 1: (0.0, 0.0)}
        self.avg_areas: Dict[int, float] = {0: 0.0, 1: 0.0}  # running avg bbox area
        self.calibrated: Dict[int, bool] = {0: False, 1: False}
        self.last_track_ids: Dict[int, int] = {}  # player_id -> last YOLO track_id
        self.court_sides: Dict[int, str] = {0: "top", 1: "bottom"}  # locked court side

        # Calibration parameters
        self.CALIBRATION_FRAMES = 15  # Frames to establish baseline
        self.calibration_observations: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
        self.calibration_complete = False
        self.calibration_initial_locked = False  # True after first 2-player frame locks assignment
        self.frames_processed = 0

        # Tracking parameters
        self.VELOCITY_SMOOTHING = 0.3  # EMA alpha for velocity (lower = smoother)
        self.POSITION_HISTORY_LEN = 30  # Keep last N positions
        self.MAX_MATCH_DISTANCE = 250.0  # Max pixels for valid match
        self.AREA_SMOOTHING = 0.1  # EMA alpha for area

        # Swap detection parameters. Raised 2→6 (≈0.2 s @ 30 fps) to avoid
        # firing on brief glitches: a jump/dive that briefly puts a player on
        # the wrong side of the net, a 1-2 frame pose-keypoint failure, or a
        # transient skeleton-index reordering. Real swaps (players actually
        # trading halves) always persist well past 0.2 s, so raising the
        # threshold reduces oscillation without missing real events.
        self.SWAP_CONSECUTIVE_THRESHOLD = 6
        self.swap_violation_count = 0  # Running count of consecutive court-side violations
        self.total_swaps_corrected = 0

        # Cost weights for composite matching
        self.W_DISTANCE = 1.0       # Weight for Euclidean distance cost
        self.W_VELOCITY = 0.6       # Weight for velocity-predicted distance
        self.W_COURT_SIDE = 0.8     # Weight for court-side consistency
        self.W_AREA = 0.2           # Weight for bbox area similarity
        self.W_TRACK_ID = 0.15      # Weight for YOLO track ID continuity (low — far player IDs are unreliable)
        self.W_STICKINESS = 0.5     # Weight for keeping same assignment as previous frame (hysteresis)

        # Previous-frame assignment map: skeleton_index -> player_id
        # Used for stickiness cost and majority-vote smoothing
        self.prev_assignment: Dict[int, int] = {}  # skel_idx -> pid last frame
        self.prev_skel_centers: List[Tuple[float, float]] = []  # centres from previous frame

        # Majority-vote sliding window: list of recent assignment tuples
        # Each entry is a tuple of (pid_for_skel0, pid_for_skel1) — the "raw" best assignment
        self.VOTE_WINDOW_SIZE = 5
        self.assignment_vote_history: List[Tuple[int, int]] = []  # (pid_for_skel0, pid_for_skel1)
    
    def _predict_position(self, player_id: int) -> Optional[Tuple[float, float]]:
        """
        Predict next position for a player using last known position + velocity.
        Returns None if no history available.
        """
        if not self.positions[player_id]:
            return None
        last = self.positions[player_id][-1]
        vx, vy = self.velocities[player_id]
        return (last["x"] + vx, last["y"] + vy)
    
    def _update_velocity(self, player_id: int, new_x: float, new_y: float) -> None:
        """Update velocity estimate using Exponential Moving Average."""
        if not self.positions[player_id]:
            return
        last = self.positions[player_id][-1]
        dt_frames = max(1, self.frames_processed - last.get("frame", self.frames_processed - 1))
        # Instantaneous velocity (pixels per frame)
        inst_vx = (new_x - last["x"]) / dt_frames
        inst_vy = (new_y - last["y"]) / dt_frames
        speed = (inst_vx**2 + inst_vy**2)**0.5
        if speed > self.MAX_INST_VEL:
            scale = self.MAX_INST_VEL / speed
            inst_vx *= scale
            inst_vy *= scale
        # EMA update
        alpha = self.VELOCITY_SMOOTHING
        old_vx, old_vy = self.velocities[player_id]
        self.velocities[player_id] = (
            alpha * inst_vx + (1 - alpha) * old_vx,
            alpha * inst_vy + (1 - alpha) * old_vy
        )
    
    def _update_area(self, player_id: int, area: float) -> None:
        """Update running average of bounding box area."""
        if self.avg_areas[player_id] == 0.0:
            self.avg_areas[player_id] = area
        else:
            alpha = self.AREA_SMOOTHING
            self.avg_areas[player_id] = alpha * area + (1 - alpha) * self.avg_areas[player_id]
    
    def _add_position(self, player_id: int, x: float, y: float, frame: int) -> None:
        """Record a position and maintain history length."""
        self.positions[player_id].append({"x": x, "y": y, "frame": frame})
        if len(self.positions[player_id]) > self.POSITION_HISTORY_LEN:
            self.positions[player_id].pop(0)
    
    def _get_court_side(self, y: float, x: Optional[float] = None) -> str:
        """
        Determine which side of the court a point is on.

        When the manual net-line keypoints are available, use the actual
        net's y-at-x (linearly interpolated between net_left and net_right)
        so tilted-camera frames correctly identify which side of the NET a
        player is on — not which side of the arbitrary pixel midline.
        """
        if self.net_line is not None and x is not None:
            x0, y0, x1, y1 = self.net_line
            dx = x1 - x0
            if abs(dx) > 1e-6:
                # Linear interp of net line at this x, extrapolated when the
                # player is outside the net's x-range.
                t = (x - x0) / dx
                net_y = y0 + t * (y1 - y0)
                return "top" if y < net_y else "bottom"
        return "top" if y < self.court_midline_y else "bottom"
    
    def _compute_assignment_cost(
        self,
        player_id: int,
        skeleton: Dict,
        skeleton_center: Tuple[float, float],
        skeleton_index: int = -1,
    ) -> float:
        """
        Compute composite cost of assigning a skeleton to a player_id.
        Lower cost = better match.

        Components:
        1. Distance to last known position (or predicted position)
        2. Distance to velocity-predicted position
        3. Court-side consistency penalty
        4. Bounding box area similarity
        5. YOLO track ID continuity bonus (low weight — unreliable for far player)
        6. Assignment stickiness — bonus for keeping same pid↔skeleton as last frame
        """
        cost = 0.0
        sx, sy = skeleton_center

        # --- Component 1: Distance to last known position ---
        if self.positions[player_id]:
            last = self.positions[player_id][-1]
            dist = ((sx - last["x"])**2 + (sy - last["y"])**2)**0.5
            # Normalize by max match distance
            cost += self.W_DISTANCE * min(dist / self.MAX_MATCH_DISTANCE, 2.0)
        else:
            # No history - use court-side prior only
            cost += self.W_DISTANCE * 1.0  # Neutral cost

        # --- Component 2: Distance to velocity-predicted position ---
        predicted = self._predict_position(player_id)
        if predicted is not None:
            pred_dist = ((sx - predicted[0])**2 + (sy - predicted[1])**2)**0.5
            cost += self.W_VELOCITY * min(pred_dist / self.MAX_MATCH_DISTANCE, 2.0)

        # --- Component 3: Court-side consistency ---
        if self.calibrated[player_id]:
            expected_side = self.court_sides[player_id]
            actual_side = self._get_court_side(sy, sx)
            if actual_side != expected_side:
                cost += self.W_COURT_SIDE * 1.5  # Heavy penalty for wrong side
        else:
            # During calibration, use Y-position convention (Player 0 = top)
            if player_id == 0:
                # Player 0 should be near top (small Y) - penalize large Y
                cost += self.W_COURT_SIDE * (sy / self.video_height)
            else:
                # Player 1 should be near bottom (large Y) - penalize small Y
                cost += self.W_COURT_SIDE * (1.0 - sy / self.video_height)

        # --- Component 4: Bounding box area similarity ---
        if self.avg_areas[player_id] > 0 and skeleton.get("area", 0) > 0:
            area_ratio = skeleton["area"] / self.avg_areas[player_id]
            # Penalize large deviations from expected area
            area_diff = abs(1.0 - area_ratio)
            cost += self.W_AREA * min(area_diff, 2.0)

        # --- Component 5: YOLO track ID continuity ---
        if player_id in self.last_track_ids and skeleton.get("track_id", -1) >= 0:
            if skeleton["track_id"] == self.last_track_ids[player_id]:
                cost -= self.W_TRACK_ID * 0.5  # Bonus for same track ID
            # No penalty for different track ID (YOLO may reassign)

        # --- Component 6: Assignment stickiness (hysteresis) ---
        # Match this skeleton to a previous-frame skeleton by proximity,
        # then reward keeping the same player_id assignment.
        if skeleton_index >= 0 and self.prev_skel_centers:
            matched_prev_idx = self._match_to_prev_skeleton(sx, sy)
            if matched_prev_idx is not None and matched_prev_idx in self.prev_assignment:
                if self.prev_assignment[matched_prev_idx] == player_id:
                    cost -= self.W_STICKINESS  # Reward keeping same assignment
                else:
                    cost += self.W_STICKINESS * 0.3  # Small penalty for switching

        return cost

    def _match_to_prev_skeleton(self, sx: float, sy: float) -> Optional[int]:
        """Find the closest previous-frame skeleton to (sx, sy). Returns index or None."""
        if not self.prev_skel_centers:
            return None
        best_idx = None
        best_dist = float("inf")
        for idx, (px, py) in enumerate(self.prev_skel_centers):
            d = ((sx - px)**2 + (sy - py)**2)**0.5
            if d < best_dist:
                best_dist = d
                best_idx = idx
        # Only match if reasonably close (half the max match distance)
        if best_dist < self.MAX_MATCH_DISTANCE:
            return best_idx
        return None
    
    def _run_calibration(self, skeletons: List[Dict]) -> None:
        """
        During calibration phase, collect position observations and establish
        court-side assignments based on consistent Y-position patterns.

        IMPORTANT: The very first frame with >=2 skeletons uses Y-sort to lock
        the initial player_id -> track_id mapping. All subsequent calibration
        frames use track-ID / proximity continuity instead of re-sorting by Y,
        which prevents frame-to-frame flicker when players are at similar heights.
        """
        if len(skeletons) < 2:
            return

        if not self.calibration_initial_locked:
            # --- FIRST 2-PLAYER FRAME: sort by Y to establish initial mapping ---
            sorted_skels = sorted(skeletons[:2], key=lambda s: s["center"][1])
            for pid, skel in enumerate(sorted_skels):
                center = skel["center"]
                self.calibration_observations[pid].append(center)
                self._add_position(pid, center[0], center[1], self.frames_processed)
                self._update_area(pid, skel.get("area", 0))
                if skel.get("track_id", -1) >= 0:
                    self.last_track_ids[pid] = skel["track_id"]
            self.calibration_initial_locked = True
        else:
            # --- SUBSEQUENT CALIBRATION FRAMES: match by proximity to last known position ---
            # This avoids re-sorting by Y which causes flicker when players are close
            for pid in range(2):
                best_skel = None
                best_dist = float("inf")
                last_pos = self.positions[pid][-1] if self.positions[pid] else None

                for skel in skeletons[:4]:  # check up to 4 candidates
                    cx, cy = skel["center"]
                    # Prefer matching by YOLO track_id first
                    if (pid in self.last_track_ids and
                            skel.get("track_id", -1) >= 0 and
                            skel["track_id"] == self.last_track_ids[pid]):
                        best_skel = skel
                        break
                    # Fall back to proximity
                    if last_pos is not None:
                        d = ((cx - last_pos["x"])**2 + (cy - last_pos["y"])**2)**0.5
                        if d < best_dist:
                            best_dist = d
                            best_skel = skel

                if best_skel is not None:
                    center = best_skel["center"]
                    self.calibration_observations[pid].append(center)
                    self._add_position(pid, center[0], center[1], self.frames_processed)
                    self._update_area(pid, best_skel.get("area", 0))
                    if best_skel.get("track_id", -1) >= 0:
                        self.last_track_ids[pid] = best_skel["track_id"]

        # Check if calibration is complete
        if self.frames_processed >= self.CALIBRATION_FRAMES:
            for pid in [0, 1]:
                if self.calibration_observations[pid]:
                    obs = self.calibration_observations[pid]
                    avg_y = sum(o[1] for o in obs) / len(obs)
                    avg_x = sum(o[0] for o in obs) / len(obs)
                    self.court_sides[pid] = self._get_court_side(avg_y, avg_x)
                    self.calibrated[pid] = True

            # Refine court midline based on observed positions
            if self.calibration_observations[0] and self.calibration_observations[1]:
                avg_y_top = sum(o[1] for o in self.calibration_observations[0]) / len(self.calibration_observations[0])
                avg_y_bot = sum(o[1] for o in self.calibration_observations[1]) / len(self.calibration_observations[1])
                self.court_midline_y = (avg_y_top + avg_y_bot) / 2.0

            self.calibration_complete = True
    
    def _detect_and_correct_swap(self, assignments: List[Tuple[int, int]],
                                  skeletons: List[Dict]) -> List[Tuple[int, int]]:
        """
        Detect if a swap has occurred by checking court-side consistency.
        If both players are on wrong sides, swap their assignments.
        
        Args:
            assignments: List of (player_id, skeleton_index) tuples
            skeletons: The active skeletons list
            
        Returns:
            Corrected assignments list
        """
        if not self.calibration_complete or len(assignments) != 2:
            return assignments
        
        # Check if both assignments violate court-side expectations
        violations = 0
        for pid, skel_idx in assignments:
            if skel_idx < len(skeletons):
                skel_x, skel_y = skeletons[skel_idx]["center"]
                actual_side = self._get_court_side(skel_y, skel_x)
                expected_side = self.court_sides[pid]
                if actual_side != expected_side:
                    violations += 1
        
        if violations == 2:
            # Both players on wrong sides - this is a swap!
            self.swap_violation_count += 1
            
            if self.swap_violation_count >= self.SWAP_CONSECUTIVE_THRESHOLD:
                # Confirmed swap - correct it by swapping the assignments
                corrected = [(assignments[1][0], assignments[0][1]),
                             (assignments[0][0], assignments[1][1])]
                self.swap_violation_count = 0
                self.total_swaps_corrected += 1
                # Exchange every piece of per-player state that we keep
                # history for, so the next frame's cost matrix reasons
                # about the CORRECT player for pid 0 / pid 1. Without this
                # exchange, positions[0] still points to the other player's
                # trajectory and the cost matrix immediately wants to re-swap,
                # producing frame-to-frame oscillation in the player labels.
                # Court_sides[pid] stays fixed — those are the canonical
                # identity anchors we're correcting _towards_.
                self.positions[0], self.positions[1] = (
                    self.positions[1], self.positions[0]
                )
                self.velocities[0], self.velocities[1] = (
                    self.velocities[1], self.velocities[0]
                )
                self.avg_areas[0], self.avg_areas[1] = (
                    self.avg_areas[1], self.avg_areas[0]
                )
                if 0 in self.last_track_ids or 1 in self.last_track_ids:
                    t0 = self.last_track_ids.get(0)
                    t1 = self.last_track_ids.get(1)
                    if t0 is not None:
                        self.last_track_ids[1] = t0
                    else:
                        self.last_track_ids.pop(1, None)
                    if t1 is not None:
                        self.last_track_ids[0] = t1
                    else:
                        self.last_track_ids.pop(0, None)
                print(f"[TRACKER] Swap detected and corrected at frame {self.frames_processed} "
                      f"(total corrections: {self.total_swaps_corrected})")
                return corrected
        elif violations == 0:
            # No violations - reset the counter
            self.swap_violation_count = 0
        # Single violation (1) might be temporary - don't reset but don't correct
        
        return assignments
    
    def _apply_majority_vote(self, raw_assignments: List[Tuple[int, int]],
                             active_skeletons: List[Dict]) -> List[Tuple[int, int]]:
        """
        Majority-vote smoothing over a sliding window of recent assignments.
        Prevents single-frame cost fluctuations from flipping the assignment.

        We represent each assignment as a canonical tuple:
          (pid_assigned_to_skel0, pid_assigned_to_skel1)
        and pick whichever mapping appeared most often in the last VOTE_WINDOW_SIZE frames.
        """
        if len(raw_assignments) != 2 or len(active_skeletons) < 2:
            return raw_assignments

        # Build the current mapping tuple
        # Match each skeleton to a previous-frame skeleton by proximity so
        # "skel 0" / "skel 1" labels are spatially consistent across frames
        skel_centers = [s["center"] for s in active_skeletons[:2]]

        # Map current skel indices to "canonical" slots via proximity to
        # the first two prev_skel_centers (if available)
        if len(self.prev_skel_centers) >= 2:
            # Compute which ordering of current skeletons best matches
            # the previous frame's skeleton positions
            d00 = ((skel_centers[0][0] - self.prev_skel_centers[0][0])**2 +
                    (skel_centers[0][1] - self.prev_skel_centers[0][1])**2)
            d01 = ((skel_centers[0][0] - self.prev_skel_centers[1][0])**2 +
                    (skel_centers[0][1] - self.prev_skel_centers[1][1])**2)
            swapped_order = d01 < d00  # current skel 0 is closer to prev skel 1
        else:
            swapped_order = False

        # Build mapping: canonical_slot -> pid
        slot_to_pid: Dict[int, int] = {}
        for pid, skel_idx in raw_assignments:
            canonical_slot = skel_idx
            if swapped_order:
                canonical_slot = 1 - skel_idx if skel_idx < 2 else skel_idx
            slot_to_pid[canonical_slot] = pid

        vote_tuple = (slot_to_pid.get(0, 0), slot_to_pid.get(1, 1))

        # Append to history
        self.assignment_vote_history.append(vote_tuple)
        if len(self.assignment_vote_history) > self.VOTE_WINDOW_SIZE:
            self.assignment_vote_history.pop(0)

        # Count votes
        from collections import Counter
        counts = Counter(self.assignment_vote_history)
        winner = counts.most_common(1)[0][0]

        # If majority agrees with raw, keep raw. Otherwise override.
        if winner == vote_tuple:
            return raw_assignments

        # Reconstruct assignments from winner tuple, undoing the canonical swap
        result_slot_to_pid = {0: winner[0], 1: winner[1]}
        corrected = []
        for canonical_slot, pid in result_slot_to_pid.items():
            skel_idx = canonical_slot
            if swapped_order:
                skel_idx = 1 - canonical_slot if canonical_slot < 2 else canonical_slot
            corrected.append((pid, skel_idx))

        return corrected

    def match_skeletons(
        self,
        active_skeletons: List[Dict],
        frame_number: int
    ) -> List[Tuple[int, Dict, Any]]:
        """
        Main entry point: match active skeletons to player IDs.

        Returns list of (player_id, keypoints, confidences) tuples.
        Uses composite cost matching with velocity prediction, court-side priors,
        swap detection/correction, YOLO track ID continuity, assignment stickiness,
        and majority-vote smoothing.
        """
        self.frames_processed = frame_number

        if not active_skeletons:
            return []

        # --- CALIBRATION PHASE ---
        if not self.calibration_complete:
            self._run_calibration(active_skeletons)

            # During calibration, return assignments from calibration state
            # (which already used track-ID continuity after the first frame)
            result = []
            for pid in range(2):
                if self.positions[pid]:
                    last = self.positions[pid][-1]
                    # Find the skeleton closest to this player's last known position
                    best_skel = None
                    best_dist = float("inf")
                    for skel in active_skeletons:
                        cx, cy = skel["center"]
                        d = ((cx - last["x"])**2 + (cy - last["y"])**2)**0.5
                        if d < best_dist:
                            best_dist = d
                            best_skel = skel
                    if best_skel is not None:
                        result.append((pid, best_skel["kpts"], best_skel.get("conf")))

            # Update prev state for stickiness in next frame
            self.prev_skel_centers = [s["center"] for s in active_skeletons[:2]]
            self.prev_assignment = {}
            for pid, _, _ in result:
                for idx, skel in enumerate(active_skeletons[:2]):
                    if self.positions[pid]:
                        last = self.positions[pid][-1]
                        if abs(skel["center"][0] - last["x"]) < 1 and abs(skel["center"][1] - last["y"]) < 1:
                            self.prev_assignment[idx] = pid
                            break

            return result

        # --- MAIN MATCHING (post-calibration) ---
        n_skeletons = len(active_skeletons)

        if n_skeletons == 1:
            # Only one skeleton visible. Assign ONLY when we have a
            # high-confidence signal:
            #   1. Court side — once calibrated, each player is locked to
            #      a side of the net. A visible skeleton's side is a
            #      deterministic signal that does not drift with missed
            #      detections.
            #   2. YOLO26 track_id continuity — the tracker's own id says
            #      "this is the same object we were following".
            # If neither applies, return [] so the pid is not set on the
            # emitted bbox. The frontend renders it as unassigned rather
            # than showing a confidently-wrong label.
            skel = active_skeletons[0]
            center = skel["center"]
            sx, sy = center
            skel_tid = skel.get("track_id", -1)

            pid: Optional[int] = None

            if self.calibration_complete:
                actual_side = self._get_court_side(sy, sx)
                for candidate_pid in (0, 1):
                    if (self.calibrated[candidate_pid]
                            and self.court_sides[candidate_pid] == actual_side):
                        pid = candidate_pid
                        break

            if pid is None and skel_tid >= 0:
                for candidate_pid, last_tid in self.last_track_ids.items():
                    if last_tid == skel_tid:
                        pid = candidate_pid
                        break

            if pid is None:
                # No confident assignment. Don't guess — don't update
                # state either, since a wrong assignment would pollute
                # last_track_ids and court_sides for later frames.
                return []

            # Update state
            self._update_velocity(pid, center[0], center[1])
            self._add_position(pid, center[0], center[1], frame_number)
            self._update_area(pid, skel.get("area", 0))
            if skel.get("track_id", -1) >= 0:
                self.last_track_ids[pid] = skel["track_id"]

            # Update prev state
            self.prev_skel_centers = [center]
            self.prev_assignment = {0: pid}

            return [(pid, skel["kpts"], skel.get("conf"))]

        # Two or more skeletons: compute cost matrix and find optimal assignment
        # Build 2 x N cost matrix (2 players, N skeletons)
        cost_matrix = []
        for pid in range(2):
            row = []
            for skel_idx, skel in enumerate(active_skeletons):
                cost = self._compute_assignment_cost(pid, skel, skel["center"], skeleton_index=skel_idx)
                row.append(cost)
            cost_matrix.append(row)

        # Solve assignment (exhaustive for 2-player case)
        assignments = []  # List of (player_id, skeleton_index)

        if n_skeletons >= 2:
            if n_skeletons > 2:
                # Evaluate all pairs
                best_pair = None
                best_pair_cost = float("inf")
                for i in range(n_skeletons):
                    for j in range(n_skeletons):
                        if i == j:
                            continue
                        pair_cost = cost_matrix[0][i] + cost_matrix[1][j]
                        if pair_cost < best_pair_cost:
                            best_pair_cost = pair_cost
                            best_pair = (i, j)

                if best_pair:
                    assignments = [(0, best_pair[0]), (1, best_pair[1])]
            else:
                # Exactly 2 skeletons - compare both options
                cost_a = cost_matrix[0][0] + cost_matrix[1][1]
                cost_b = cost_matrix[0][1] + cost_matrix[1][0]

                if cost_a <= cost_b:
                    assignments = [(0, 0), (1, 1)]
                else:
                    assignments = [(0, 1), (1, 0)]

        # --- MAJORITY-VOTE SMOOTHING ---
        assignments = self._apply_majority_vote(assignments, active_skeletons)

        # --- SWAP DETECTION AND CORRECTION ---
        assignments = self._detect_and_correct_swap(assignments, active_skeletons)

        # --- BUILD RESULTS AND UPDATE STATE ---
        result = []
        new_prev_assignment: Dict[int, int] = {}
        for pid, skel_idx in assignments:
            if skel_idx < len(active_skeletons):
                skel = active_skeletons[skel_idx]
                center = skel["center"]

                # Validate match distance before updating state
                predicted = self._predict_position(pid)
                if predicted is not None:
                    dist = ((center[0] - predicted[0])**2 + (center[1] - predicted[1])**2)**0.5
                    if dist > self.MAX_MATCH_DISTANCE * 2:
                        print(f"[TRACKER] Large jump for Player {pid} at frame {frame_number}: "
                              f"{dist:.0f}px (keeping assignment but not updating velocity)")
                        self._add_position(pid, center[0], center[1], frame_number)
                    else:
                        # Normal update
                        self._update_velocity(pid, center[0], center[1])
                        self._add_position(pid, center[0], center[1], frame_number)
                else:
                    # First match after calibration
                    self._add_position(pid, center[0], center[1], frame_number)

                self._update_area(pid, skel.get("area", 0))
                if skel.get("track_id", -1) >= 0:
                    self.last_track_ids[pid] = skel["track_id"]

                new_prev_assignment[skel_idx] = pid
                result.append((pid, skel["kpts"], skel.get("conf")))

        # Update previous-frame state for next iteration
        self.prev_skel_centers = [s["center"] for s in active_skeletons[:max(new_prev_assignment.keys()) + 1]] if new_prev_assignment else []
        self.prev_assignment = new_prev_assignment

        return result
    
    def get_last_position(self, player_id: int) -> Optional[Dict]:
        """Get the last known position for a player (for speed calculation)."""
        if self.positions[player_id]:
            return self.positions[player_id][-1]
        return None
    
    def get_stats(self) -> Dict:
        """Get tracker statistics for logging."""
        return {
            "calibration_complete": self.calibration_complete,
            "total_swaps_corrected": self.total_swaps_corrected,
            "court_midline_y": self.court_midline_y,
            "player_0_side": self.court_sides[0],
            "player_1_side": self.court_sides[1],
            "player_0_positions": len(self.positions[0]),
            "player_1_positions": len(self.positions[1]),
        }


# Modal app configuration
app = modal.App("badminton-supabase-processor")

# Volume for temporary storage and model caching
vol = modal.Volume.from_name("badminton-processor-cache", create_if_missing=True)

# Volume for trained models (shared with modal_inference.py)
models_vol = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)
MODELS_PATH = "/models"

# Image with all dependencies
# Add local backend modules (tracknet, rally_detection) to the container
_backend_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .pip_install(
        "fastapi[standard]",
        "opencv-python-headless",
        "numpy",
        "ultralytics>=8.2.0",
        "httpx",
        "python-dotenv",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "supabase>=2.5.0",
    )
    .add_local_dir(str(_backend_dir / "tracknet"), remote_path="/root/tracknet")
    .add_local_file(str(_backend_dir / "rally_detection.py"), remote_path="/root/rally_detection.py")
    .add_local_file(str(_backend_dir / "shot_detection.py"), remote_path="/root/shot_detection.py")
    .add_local_file(str(_backend_dir / "rally_detection_shot_gap.py"), remote_path="/root/rally_detection_shot_gap.py")
    .add_local_file(str(_backend_dir / "speed_calc.py"), remote_path="/root/speed_calc.py")
    .add_local_file(str(_backend_dir / "supabase_helpers.py"), remote_path="/root/supabase_helpers.py")
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("supabase-secrets"),
        modal.Secret.from_name("modal-shared-secret"),
    ],
)
@modal.fastapi_endpoint(method="POST")
async def process_video(request: Request) -> Dict[str, Any]:
    """
    Lightweight endpoint that validates the request and spawns GPU
    processing in the background. Returns immediately so the Edge Function
    caller doesn't time out.

    Authenticates the caller via HMAC-SHA256 over the raw request body
    using MODAL_SHARED_SECRET (loaded from the 'modal-shared-secret'
    Modal Secret) and expects the hex digest in the X-Signature header.
    """
    from fastapi.responses import JSONResponse

    raw_body = await request.body()
    signature = request.headers.get("X-Signature")
    secret = os.environ.get("MODAL_SHARED_SECRET", "")
    if not verify_hmac(raw_body, signature, secret):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    payload = json.loads(raw_body)
    video_id = payload.get("video_id")
    owner_id = payload.get("owner_id")
    video_url = payload.get("video_url")
    manual_court_keypoints = payload.get("manual_court_keypoints")
    analysis_mode = payload.get("analysis_mode", "full")

    if not all([video_id, owner_id, video_url]):
        return JSONResponse(
            {"error": "Missing required fields: video_id, owner_id, video_url"},
            status_code=400,
        )

    # Spawn the GPU worker in the background — returns immediately.
    # The worker writes progress/logs/results directly to Supabase via
    # supabase_client(); no callback URL is needed.
    _process_video_worker.spawn(
        video_id=video_id,
        owner_id=owner_id,
        video_url=video_url,
        manual_court_keypoints=manual_court_keypoints,
        analysis_mode=analysis_mode,
    )

    return {"status": "accepted", "video_id": video_id, "owner_id": owner_id}


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("supabase-secrets"),
        modal.Secret.from_name("modal-shared-secret"),
    ],
    timeout=120,
)
@modal.fastapi_endpoint(method="POST")
async def recalculate_speeds(request: Request) -> Any:
    """
    Recompute player speeds from stored skeleton data, optionally with
    user-supplied manual court keypoints. HMAC-verified against
    MODAL_SHARED_SECRET (same scheme as `process_video`).

    Request body (JSON):
      {
        "video_id": str,
        "owner_id": str,
        "results_storage_path": str,
        "manual_court_keypoints": Optional[dict],
      }

    Response (JSON):
      {
        "video_id": str,
        "speed_data": <speed_calc result>,
        "manual_keypoints_used": bool,
        "detection_source": "modal",
        "status": "success",
      }
    """
    from fastapi.responses import JSONResponse

    raw_body = await request.body()
    signature = request.headers.get("X-Signature")
    secret = os.environ.get("MODAL_SHARED_SECRET", "")
    if not verify_hmac(raw_body, signature, secret):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    video_id = payload.get("video_id")
    owner_id = payload.get("owner_id")
    results_storage_path = payload.get("results_storage_path")
    manual_keypoints = payload.get("manual_court_keypoints")

    if not all([video_id, owner_id, results_storage_path]):
        return JSONResponse(
            {
                "error": "Missing required fields: "
                         "video_id, owner_id, results_storage_path"
            },
            status_code=400,
        )

    # Download results.json from Supabase Storage. supabase-py is sync; run
    # in a worker thread to avoid blocking the event loop.
    sb = supabase_client()
    try:
        blob = await asyncio.to_thread(
            lambda: sb.storage.from_("results").download(results_storage_path)
        )
    except Exception as e:  # noqa: BLE001 — surface the underlying message
        return JSONResponse(
            {"error": f"failed to fetch results: {str(e)[:200]}"},
            status_code=404,
        )
    if not blob:
        return JSONResponse({"error": "results not found"}, status_code=404)

    try:
        results = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return JSONResponse({"error": "results not valid JSON"}, status_code=500)

    # Late import — keeps cold-start cost out of unrelated function paths.
    from speed_calc import calculate_speeds_from_skeleton

    speed_data = calculate_speeds_from_skeleton(
        skeleton_data=results.get("skeleton_data", []),
        fps=float(results.get("fps", 30.0)),
        video_width=int(results.get("video_width", 1920)),
        video_height=int(results.get("video_height", 1080)),
        manual_court_keypoints=manual_keypoints,
    )

    return JSONResponse({
        "video_id": video_id,
        "speed_data": speed_data,
        "manual_keypoints_used": speed_data.get("manual_keypoints_used", False),
        "detection_source": "modal",
        "status": "success",
    })


@app.function(
    gpu="A10G",
    timeout=7200,  # 2 hours (long videos can exceed 60 min on T4; A10G should fit comfortably under 2 h)
    memory=8192,   # 8 GB RAM (skeleton data + JSON serialization for long videos)
    image=image,
    volumes={"/cache": vol, MODELS_PATH: models_vol},
    secrets=[modal.Secret.from_name("supabase-secrets")],
)
async def _process_video_worker(
    video_id: str,
    owner_id: str,
    video_url: str,
    manual_court_keypoints: Optional[Dict] = None,
    analysis_mode: str = "full",
) -> Dict[str, Any]:
    """
    GPU worker that does the actual video processing.
    Spawned in the background by process_video endpoint.

    Writes progress/logs/results directly to Supabase Postgres + Storage.
    """
    import httpx
    import cv2
    import numpy as np
    from ultralytics import YOLO

    import resource

    print(f"[MODAL] Starting processing for video: {video_id}")
    print(f"[MODAL] Owner: {owner_id}")
    print(f"[MODAL] Video URL: {video_url[:100]}...")

    def get_memory_mb() -> float:
        """Get current RSS memory usage in MB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    status_update_failures = 0

    async def send_status_update(
        status: str = "processing",
        progress: float = 0,
        current_frame: int = 0,
        total_frames: int = 0,
        error: Optional[str] = None
    ):
        """Update the videos row with the current processing progress.

        Runs the synchronous supabase-py call in a worker thread so the async
        event loop is not blocked while Postgres responds.
        """
        nonlocal status_update_failures
        update_payload: Dict[str, Any] = {
            "status": status,
            "progress": progress,
            "current_frame": current_frame,
            "total_frames": total_frames,
        }
        if error is not None:
            update_payload["error"] = error
        for attempt in range(3):
            try:
                await asyncio.to_thread(
                    lambda: supabase_client()
                    .table("videos")
                    .update(update_payload)
                    .eq("id", video_id)
                    .execute()
                )
                status_update_failures = 0
                return
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2.0 ** attempt)
                else:
                    status_update_failures += 1
                    print(f"[MODAL] Warning: Status update failed after 3 attempts: {e} "
                          f"(total failures: {status_update_failures})")

    async def send_log(
        message: str,
        level: str = "info",
        category: str = "processing"
    ):
        """Insert a row into processing_logs for this video."""
        log_row = {
            "video_id": video_id,
            "owner_id": owner_id,
            "message": message,
            "level": level,
            "category": category,
        }
        for attempt in range(2):
            try:
                await asyncio.to_thread(
                    lambda: supabase_client()
                    .table("processing_logs")
                    .insert(log_row)
                    .execute()
                )
                return
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(2.0)
                else:
                    print(f"[MODAL] Warning: Failed to send log: {e}")
    
    try:
        pipeline_start = time.time()
        phase_start = time.time()

        # Defensive cache cleanup: remove any stale files for this video_id from
        # a previous partial/failed run on the same Modal container. Each upload
        # gets a fresh video_id, so files from OTHER videos are already isolated
        # by path — this only protects against retry-on-same-id edge cases and
        # guarantees we never silently reuse cached artifacts between runs.
        stale_files = list(Path("/cache").glob(f"{video_id}*"))
        for stale_path in stale_files:
            try:
                stale_path.unlink()
            except Exception as cleanup_err:
                print(f"[MODAL] Warn: failed to remove stale {stale_path}: {cleanup_err}")
        if stale_files:
            print(f"[MODAL] Removed {len(stale_files)} stale cache file(s) for {video_id}")

        # Download video from Supabase Storage signed URL (streamed to disk
        # to handle large files). The signed URL is supplied by the caller
        # in the request payload and is good for ~1 hour.
        await send_log("Downloading video from storage...", "info", "processing")

        video_path = Path(f"/cache/{video_id}.mp4")

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0)) as client:
            async with client.stream("GET", video_url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                total_bytes = int(content_length) if content_length else None
                downloaded = 0
                last_log = time.time()

                with open(video_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log download progress every 5 seconds
                        now = time.time()
                        if now - last_log >= 5.0:
                            if total_bytes:
                                pct = downloaded / total_bytes * 100
                                await send_log(
                                    f"Downloading... {downloaded / (1024*1024):.0f}/{total_bytes / (1024*1024):.0f} MB ({pct:.0f}%)",
                                    "info", "processing"
                                )
                            else:
                                await send_log(
                                    f"Downloading... {downloaded / (1024*1024):.0f} MB",
                                    "info", "processing"
                                )
                            last_log = now

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        download_time = time.time() - phase_start
        await send_log(f"Video downloaded: {file_size_mb:.1f} MB in {download_time:.1f}s", "success", "processing")
        print(f"[MODAL] Video downloaded: {video_path} ({file_size_mb:.1f} MB) in {download_time:.1f}s")
        phase_start = time.time()
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        await send_log(f"Video loaded: {width}x{height} @ {fps:.1f}fps", "success", "processing")
        await send_log(f"Total frames: {total_frames} ({duration:.1f}s duration)", "info", "processing")

        sample_rate = 1  # Process every frame for full accuracy

        await send_status_update("processing", 0, 0, total_frames)

        # Load YOLO26 pose model (latest version)
        await send_log("Loading YOLO26m pose model (medium)...", "info", "model")
        pose_model = YOLO("yolo26m-pose.pt")  # Using YOLO26 small pose model (better far-player accuracy)

        # Load badminton detection model (for shuttlecock, racket detection)
        await send_log("Loading badminton detection model...", "info", "model")
        badminton_model_path = f"{MODELS_PATH}/badminton/best.pt"
        if os.path.exists(badminton_model_path):
            detection_model = YOLO(badminton_model_path)
            await send_log("Custom badminton model loaded", "success", "model")
        else:
            # Fallback to COCO model for general detection
            detection_model = YOLO("yolo11n.pt")
            await send_log("Using COCO model (custom model not found)", "info", "model")

        # =========================================================
        # TRACKER SETUP — BoT-SORT (Ultralytics built-in)
        # =========================================================
        # Scale time-based tracker buffers to actual FPS so 3-second
        # persistence holds whether the source is 24/30/60 fps.
        effective_fps = float(fps) if fps and fps > 0 else 30.0
        lost_buffer_frames = max(int(round(3.0 * effective_fps)), 30)

        # Custom BoT-SORT config tuned for badminton (2 players, static
        # broadcast camera, fast pivots, small far-court bboxes).
        # Key changes from defaults:
        # - track_buffer: scales with fps (3 seconds) instead of fixed 30 frames
        #   → Far player's track survives longer occlusions/low-confidence gaps
        # - track_high_thresh: 0.3 instead of 0.5
        #   → Far player (small, low-confidence) gets tracked more consistently
        # - new_track_thresh: 0.4 instead of 0.6
        #   → Faster track creation when far player reappears
        # - match_thresh: 0.9 instead of 0.8
        #   → More lenient matching to prevent track fragmentation
        # - with_reid: False
        #   → Appearance embeddings confuse identical uniforms; motion-only
        #     is cleaner for 2-player well-separated scenes.
        # Per-video YAML path (keyed on video_id) so concurrent workers on the
        # same Modal Volume can't race on a shared file, and so the config is
        # always re-written from scratch rather than reused from a prior run.
        tracker_config_path = Path(f"/cache/{video_id}_botsort.yaml")
        tracker_config_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_config_content = f"""# BoT-SORT tracker config optimized for badminton (2 players)
tracker_type: botsort
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: {lost_buffer_frames}
match_thresh: 0.9
fuse_score: True
# GMC (Global Motion Compensation) for camera movement
gmc_method: sparseOptFlow
# Proximity and appearance thresholds
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
        tracker_config_path.write_text(tracker_config_content)
        await send_log(
            f"BoT-SORT config written (track_buffer={lost_buffer_frames}f / 3.0s, track_high=0.3)",
            "info", "model"
        )

        # Tracking diagnostics accumulator — records raw tracker output
        # each frame, writes summary + per-frame JSONL to
        # /cache/tracker_metrics/ at end-of-run. Never throws; safe if
        # video dims are 0 or fps is 0.
        tracker_metrics = TrackerMetricsAccumulator(
            video_id=video_id,
            tracker_type="botsort",
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )
        await send_log(
            f"Tracker metrics accumulator armed (writing to /cache/tracker_metrics/{video_id}_botsort_*)",
            "info", "model"
        )

        # Warmup both models
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = pose_model(dummy_frame, verbose=False)
        _ = detection_model(dummy_frame, verbose=False)
        model_load_time = time.time() - phase_start
        mem_mb = get_memory_mb()
        await send_log(f"Models ready in {model_load_time:.1f}s (GPU accelerated, RAM: {mem_mb:.0f} MB)", "success", "model")
        phase_start = time.time()
        tracknet_positions = {}  # frame_num -> {"x", "y", "visible"}
        tracknet_available = False

        tracknet_model_path = f"{MODELS_PATH}/tracknet/TrackNet_best.pt"
        inpaintnet_model_path = f"{MODELS_PATH}/tracknet/InpaintNet_best.pt"

        if os.path.exists(tracknet_model_path):
            try:
                # Add tracknet module to path (mounted at /root)
                sys.path.insert(0, "/root")
                from tracknet.inference import TrackNetInference

                await send_log("Loading TrackNetV3 for shuttle tracking...", "info", "model")
                tracker = TrackNetInference(device="cuda")
                tracker.load_weights(
                    tracknet_model_path,
                    inpaintnet_model_path if os.path.exists(inpaintnet_model_path) else None,
                )
                await send_log("TrackNetV3 loaded, running shuttle tracking pass...", "info", "model")

                # Sync log callback for TrackNet progress (track_video is synchronous)
                _loop = asyncio.get_event_loop()
                def tracknet_log(msg: str):
                    _loop.run_until_complete(send_log(msg, "info", "model"))

                tracknet_positions = tracker.track_video(
                    str(video_path),
                    batch_size=16,
                    log_callback=tracknet_log,
                )
                tracknet_available = True

                visible_count = sum(1 for p in tracknet_positions.values() if p.get("visible"))
                total_tracked = len(tracknet_positions)
                pct = 100 * visible_count / max(total_tracked, 1)
                await send_log(
                    f"TrackNet: shuttle detected in {visible_count}/{total_tracked} frames ({pct:.1f}%)",
                    "success", "model"
                )
            except Exception as e:
                await send_log(f"TrackNet unavailable: {e}", "warning", "model")
                print(f"[MODAL] TrackNet error: {e}")
        else:
            await send_log(
                "TrackNet model not found - using YOLO shuttle detection only. "
                "Upload TrackNet weights to /tracknet/ for improved shuttle tracking.",
                "info", "model"
            )

        # Process frames
        tracknet_time = time.time() - phase_start
        mem_mb = get_memory_mb()
        await send_log(f"TrackNet phase complete in {tracknet_time:.1f}s (RAM: {mem_mb:.0f} MB)", "info", "processing")

        await send_log("Starting frame-by-frame analysis...", "info", "processing")
        phase_start = time.time()

        # Extract net-line endpoints from manual keypoints (if supplied) so
        # the identity tracker can do court-side checks against the real
        # net rather than the arbitrary pixel midline. For tilted cameras
        # these two can differ by hundreds of pixels.
        tracker_net_left: Optional[Tuple[float, float]] = None
        tracker_net_right: Optional[Tuple[float, float]] = None
        if manual_court_keypoints:
            nl = manual_court_keypoints.get("net_left")
            nr = manual_court_keypoints.get("net_right")
            if (
                isinstance(nl, (list, tuple)) and len(nl) >= 2 and
                isinstance(nr, (list, tuple)) and len(nr) >= 2
            ):
                tracker_net_left = (float(nl[0]), float(nl[1]))
                tracker_net_right = (float(nr[0]), float(nr[1]))

        # Initialize robust player identity tracker
        identity_tracker = PlayerIdentityTracker(
            video_height=float(height),
            fps=fps,
            video_width=float(width),
            net_left=tracker_net_left,
            net_right=tracker_net_right,
        )
        net_src = "manual net keypoints" if tracker_net_left is not None else "video-midline fallback"
        await send_log(
            f"Player identity tracker initialized ({net_src}; calibration phase: first 15 frames)",
            "info", "processing",
        )
        
        # Write skeleton frames incrementally to a temp file to avoid
        # accumulating hundreds of MB in RAM for long videos.
        # Each frame is written as a JSON line; read back after the loop.
        skeleton_frames_path = Path(f"/cache/{video_id}_skeleton.jsonl")
        skeleton_frames_file = open(skeleton_frames_path, "w")
        skeleton_frame_count = 0

        player_tracks: Dict[int, Dict] = {}
        # Player positions for summary metrics (aggregated from all frames)
        player_positions: Dict[int, list] = {0: [], 1: []}
        player_distances: Dict[int, float] = {0: 0.0, 1: 0.0}
        player_speeds: Dict[int, list] = {0: [], 1: []}
        # Sliding window for median filtering per player
        player_speed_windows: Dict[int, list] = {0: [], 1: []}
        SPEED_WINDOW_SIZE = 5  # Use median of last 5 readings
        frame_count = 0
        processed_count = 0
        
        # Movement-based filtering using YOLO tracking IDs
        # Track cumulative movement per track ID to distinguish players from stationary judges
        track_positions: Dict[int, Dict] = {}  # track_id -> {x, y, frame}
        track_cumulative_movement: Dict[int, float] = {}  # track_id -> total movement
        # Minimum cumulative movement (pixels) to be considered a real player
        # Lowered from 500 to 100 to improve far-side player detection
        # (far players are detected intermittently and may not accumulate much tracked movement)
        MIN_CUMULATIVE_MOVEMENT = 100
        
        court_polygon = None
        court_roi_active = False
        manual_court_corners = []

        if manual_court_keypoints:
            try:
                corners = []
                for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                    if key in manual_court_keypoints and manual_court_keypoints[key]:
                        pt = manual_court_keypoints[key]
                        corners.append([float(pt[0]), float(pt[1])])

                if len(corners) == 4:
                    manual_court_corners = corners
                    # Expand the polygon by 2% margin to include players at court edges
                    # REDUCED from 5% to 2% to prevent including judges sitting near the net
                    # 2% of court width (~6.1m) = ~0.12m which allows for lunges but not outside court
                    corners_np = np.array(corners, dtype=np.float32)
                    center = corners_np.mean(axis=0)
                    MARGIN_FACTOR = 1.02  # 2% margin (reduced from 5% to strictly filter judges)
                    expanded = center + (corners_np - center) * MARGIN_FACTOR
                    court_polygon = expanded.astype(np.int32)
                    court_roi_active = True
                    await send_log(f"Court ROI filter active (4-corner polygon + 2% margin)", "success", "court")
                    print(f"[MODAL] Court ROI polygon: {court_polygon.tolist()}")
                else:
                    await send_log(f"Manual keypoints incomplete ({len(corners)}/4 corners), using position filter", "warning", "court")
            except Exception as e:
                await send_log(f"Failed to initialize court ROI: {e}", "warning", "court")
                print(f"[MODAL] Court ROI init error: {e}")
        MOVEMENT_WARMUP_FRAMES = 45

        homography_matrix = None
        if manual_court_corners and len(manual_court_corners) >= 4:
            homography_matrix = compute_homography_matrix(manual_court_corners[:4])

        shuttle_static_clusters = []
        # A real shuttle is always moving; static detections are court markings/logos/lights
        shuttle_static_clusters = []  # List of {x, y, count} for positions that keep appearing
        # Scale movement thresholds by fps: per-frame pixel displacement is
        # inversely proportional to frame rate, so thresholds must shrink at
        # higher fps to avoid filtering out valid in-flight shuttle positions.
        _shuttle_fps_scale = 30.0 / fps
        SHUTTLE_STATIC_DIST_THRESHOLD = max(4, int(0.013 * max(width, height) * _shuttle_fps_scale))
        SHUTTLE_STATIC_COUNT_THRESHOLD = 3
        prev_shuttle_pos = None
        SHUTTLE_MIN_MOVEMENT = max(2, int(0.007 * max(width, height) * _shuttle_fps_scale))

        # Court ROI for shuttle filtering — expanded polygon to reject far-off detections (lights, etc.)
        # Horizontal: 40% expansion (shuttle rarely goes far past sidelines)
        # Vertical upward: expand to top of frame — shuttle can fly arbitrarily
        # high during clears/lobs, sometimes even off-screen
        shuttle_court_polygon = None
        if court_polygon is not None:
            court_center = court_polygon.astype(np.float32).mean(axis=0)
            expanded = court_polygon.astype(np.float32).copy()
            for i in range(len(expanded)):
                # Horizontal: 40% expansion from center
                expanded[i][0] = court_center[0] + (expanded[i][0] - court_center[0]) * 1.40
                # Vertical: 40% expansion downward, but extend to y=0 upward
                if expanded[i][1] < court_center[1]:
                    # Top points — extend to top of frame
                    expanded[i][1] = 0
                else:
                    # Bottom points — 40% expansion
                    expanded[i][1] = court_center[1] + (expanded[i][1] - court_center[1]) * 1.40
            shuttle_court_polygon = expanded.astype(np.int32)
            await send_log("Shuttle court ROI filter active (40% horizontal, full vertical upward)", "success", "court")

        def _shuttle_in_court(sx, sy):
            """Check if shuttle position is within the expanded court ROI."""
            if shuttle_court_polygon is None:
                return True  # No court info — allow all
            result = cv2.pointPolygonTest(shuttle_court_polygon, (float(sx), float(sy)), measureDist=False)
            return result >= 0

        def _is_static_cluster(sx, sy):
            """Check if position matches a known static false-positive cluster."""
            for cluster in shuttle_static_clusters:
                if math.sqrt((sx - cluster["x"])**2 + (sy - cluster["y"])**2) < SHUTTLE_STATIC_DIST_THRESHOLD:
                    cluster["count"] += 1
                    cluster["x"] = (cluster["x"] * (cluster["count"] - 1) + sx) / cluster["count"]
                    cluster["y"] = (cluster["y"] * (cluster["count"] - 1) + sy) / cluster["count"]
                    return True
            return False

        def _add_to_static(sx, sy):
            """Register position as potentially static."""
            for cluster in shuttle_static_clusters:
                if math.sqrt((sx - cluster["x"])**2 + (sy - cluster["y"])**2) < SHUTTLE_STATIC_DIST_THRESHOLD * 2:
                    cluster["count"] += 1
                    return
            shuttle_static_clusters.append({"x": sx, "y": sy, "count": 1})

        last_progress_update = time.time()

        while True:
            # Read actual presentation timestamp BEFORE reading the frame
            # This matches what the HTML video element's currentTime reports,
            # preventing skeleton drift on VFR videos or 29.97fps content
            frame_pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Sample rate (process every Nth frame)
            if frame_count % sample_rate != 0:
                continue
            
            processed_count += 1
            
            # Run pose estimation with integrated BoT-SORT tracking
            pose_results = pose_model.track(
                frame,
                persist=True,
                verbose=False,
                tracker=str(tracker_config_path),
                conf=0.15,
                iou=0.5,
                imgsz=960,
            )
            
            # Run object detection for shuttlecock, racket, etc. (NOT for players)
            detection_results = detection_model(frame, verbose=False)
            
            # Extract bounding box detections (shuttlecock, racket only - players come from pose model)
            badminton_detections = {
                "frame": frame_count,
                "players": [],  # Will be populated from pose model with tracking
                "shuttlecocks": [],
                "rackets": [],
                "other": []
            }
            
            if detection_results and len(detection_results) > 0:
                det_result = detection_results[0]
                if det_result.boxes is not None and len(det_result.boxes) > 0:
                    boxes = det_result.boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        xyxy = box.xyxy.cpu().numpy()[0]
                        
                        # Convert to center format
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2
                        box_width = xyxy[2] - xyxy[0]
                        box_height = xyxy[3] - xyxy[1]
                        
                        class_name = detection_model.names.get(cls_id, f"class_{cls_id}")
                        
                        det_entry = {
                            "class": class_name,
                            "confidence": conf,
                            "x": float(x_center),
                            "y": float(y_center),
                            "width": float(box_width),
                            "height": float(box_height),
                            "class_id": cls_id,
                            "detection_id": None
                        }
                        
                        # Only categorize known badminton objects - skip chairs, tables, etc.
                        # Players come from pose model, so we only need shuttle and racket here
                        class_lower = class_name.lower()
                        if class_lower in ["shuttle", "shuttlecock", "birdie", "ball"] or \
                           any(s in class_lower for s in ["shuttle", "birdie", "ball"]):
                            # Matches custom model names AND COCO's "sports ball"
                            badminton_detections["shuttlecocks"].append(det_entry)
                        elif class_lower in ["racket", "racquet"] or "racket" in class_lower or "racquet" in class_lower:
                            badminton_detections["rackets"].append(det_entry)
                        # Skip all other classes (chairs, tables, person, etc.)
                        # Person detections are also skipped as players come from pose model
            
            # Extract best shuttle position
            # Priority: TrackNet (continuous, high accuracy) > YOLO (sparse, fallback)
            shuttle_position = None

            # Try TrackNet first (if available) — with static + court filtering
            if tracknet_available and frame_count in tracknet_positions:
                tn_pos = tracknet_positions[frame_count]
                if tn_pos.get("visible"):
                    tx, ty = tn_pos["x"], tn_pos["y"]
                    # Reject if outside expanded court area (e.g. stadium lights)
                    if not _shuttle_in_court(tx, ty):
                        pass  # Skip — outside court bounds
                    elif _is_static_cluster(tx, ty):
                        pass  # Skip — known static false positive
                    elif prev_shuttle_pos is not None:
                        movement = math.sqrt((tx - prev_shuttle_pos["x"])**2 + (ty - prev_shuttle_pos["y"])**2)
                        if movement < SHUTTLE_MIN_MOVEMENT:
                            _add_to_static(tx, ty)
                        else:
                            shuttle_position = {"x": tx, "y": ty, "source": "tracknet"}
                    else:
                        shuttle_position = {"x": tx, "y": ty, "source": "tracknet"}

            # Fall back to YOLO detection with static false-positive filtering
            if shuttle_position is None and badminton_detections["shuttlecocks"]:
                candidates = sorted(badminton_detections["shuttlecocks"], key=lambda s: s["confidence"], reverse=True)

                for candidate in candidates:
                    cx, cy = candidate["x"], candidate["y"]

                    # Reject if outside expanded court area
                    if not _shuttle_in_court(cx, cy):
                        continue

                    # Check if this position is near a known static cluster
                    is_static = False
                    for cluster in shuttle_static_clusters:
                        if math.sqrt((cx - cluster["x"])**2 + (cy - cluster["y"])**2) < SHUTTLE_STATIC_DIST_THRESHOLD:
                            cluster["count"] += 1
                            cluster["x"] = (cluster["x"] * (cluster["count"] - 1) + cx) / cluster["count"]
                            cluster["y"] = (cluster["y"] * (cluster["count"] - 1) + cy) / cluster["count"]
                            is_static = True
                            break

                    if is_static:
                        continue

                    # Check if it moved relative to previous shuttle position
                    if prev_shuttle_pos is not None:
                        movement = math.sqrt((cx - prev_shuttle_pos["x"])**2 + (cy - prev_shuttle_pos["y"])**2)
                        if movement < SHUTTLE_MIN_MOVEMENT:
                            found_cluster = False
                            for cluster in shuttle_static_clusters:
                                if math.sqrt((cx - cluster["x"])**2 + (cy - cluster["y"])**2) < SHUTTLE_STATIC_DIST_THRESHOLD * 2:
                                    cluster["count"] += 1
                                    found_cluster = True
                                    break
                            if not found_cluster:
                                shuttle_static_clusters.append({"x": cx, "y": cy, "count": 1})
                            continue

                    shuttle_position = {"x": cx, "y": cy, "source": "yolo"}
                    break

                if shuttle_position:
                    prev_shuttle_pos = shuttle_position

            # Update prev_shuttle_pos for TrackNet detections too
            if shuttle_position and shuttle_position.get("source") == "tracknet":
                prev_shuttle_pos = shuttle_position

            # Prune: only keep confirmed static clusters
            shuttle_static_clusters = [
                c for c in shuttle_static_clusters
                if c["count"] >= SHUTTLE_STATIC_COUNT_THRESHOLD
            ]

            # Filter raw shuttlecock detections so the frontend doesn't draw false positives
            # Remove detections that are outside court ROI or match a static cluster
            if badminton_detections["shuttlecocks"]:
                filtered_shuttles = []
                for det in badminton_detections["shuttlecocks"]:
                    dx, dy = det["x"], det["y"]
                    if not _shuttle_in_court(dx, dy):
                        continue
                    is_known_static = False
                    for cluster in shuttle_static_clusters:
                        if math.sqrt((dx - cluster["x"])**2 + (dy - cluster["y"])**2) < SHUTTLE_STATIC_DIST_THRESHOLD:
                            is_known_static = True
                            break
                    if is_known_static:
                        continue
                    filtered_shuttles.append(det)
                badminton_detections["shuttlecocks"] = filtered_shuttles

            # Extract skeleton data from pose model with tracking
            frame_data = {
                "frame": frame_count,
                "timestamp": frame_pts,  # Use actual PTS from video container
                "players": [],
                "badminton_detections": badminton_detections,
                "shuttle_position": shuttle_position,
            }
            
            # COCO keypoint names
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            if pose_results and len(pose_results) > 0:
                result = pose_results[0]
                
                # Get tracking IDs from Ultralytics BoT-SORT output
                has_tracking = result.boxes is not None and result.boxes.is_track
                track_ids = result.boxes.id.int().cpu().tolist() if has_tracking and result.boxes.id is not None else None
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) > 0 else None

                # Feed raw tracker output to the metrics accumulator BEFORE
                # PlayerIdentityTracker post-processing runs.
                _mx_kp_xy = (result.keypoints.xy.cpu().numpy()
                             if result.keypoints is not None
                             and result.keypoints.xy is not None else None)
                _mx_kp_conf = (result.keypoints.conf.cpu().numpy()
                               if result.keypoints is not None
                               and result.keypoints.conf is not None else None)
                tracker_metrics.update(
                    frame_idx=frame_count,
                    frame_pts=frame_pts,
                    track_ids=track_ids,
                    boxes=boxes,
                    kpts_xy=_mx_kp_xy,
                    kpts_conf=_mx_kp_conf,
                )

                if result.keypoints is not None and result.keypoints.xy is not None:
                    kpts_data = result.keypoints.xy.cpu().numpy()
                    kpts_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                    
                    # Process each detected person
                    skeleton_data = []  # List of {track_id, center, kpts, conf, bbox, area}
                    
                    for person_idx in range(len(kpts_data)):
                        kpts = kpts_data[person_idx]
                        conf = kpts_conf[person_idx] if kpts_conf is not None else None
                        
                        # Get track ID (or use -1 if no tracking)
                        track_id = track_ids[person_idx] if track_ids and person_idx < len(track_ids) else -1
                        
                        # Calculate center position
                        center = skeleton_center_from_keypoints(kpts)
                        if center is None:
                            continue
                        
                        # Get bounding box from pose model (more accurate than calculating from keypoints)
                        bbox = None
                        area = 0
                        if boxes is not None and person_idx < len(boxes):
                            box = boxes[person_idx]
                            bbox = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
                            area = (box[2] - box[0]) * (box[3] - box[1])
                        else:
                            # Fallback to keypoint-based bbox
                            bbox = skeleton_bbox_from_keypoints(kpts)
                            if bbox:
                                area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
                        
                        if bbox is None:
                            continue
                        
                        # Track cumulative movement per track ID
                        if track_id >= 0:
                            if track_id in track_positions:
                                prev = track_positions[track_id]
                                dx = center[0] - prev["x"]
                                dy = center[1] - prev["y"]
                                movement = (dx**2 + dy**2)**0.5
                                track_cumulative_movement[track_id] = track_cumulative_movement.get(track_id, 0.0) + movement
                            
                            track_positions[track_id] = {"x": center[0], "y": center[1], "frame": frame_count}
                        
                        skeleton_data.append({
                            "track_id": track_id,
                            "center": center,
                            "kpts": kpts,
                            "conf": conf,
                            "bbox": bbox,
                            "area": area,
                            "cumulative_movement": track_cumulative_movement.get(track_id, 0.0) if track_id >= 0 else 0.0
                        })
                    
                    # POSITION-BASED FILTERING: Only include skeletons in the court region
                    # Use court polygon if available (from manual keypoints), otherwise fall back to rectangular
                    
                    # Helper function to detect if a person is sitting/crouching (likely a judge)
                    def is_sitting_pose(kpts, bbox, conf=None, min_conf=0.2) -> bool:
                        if bbox is None:
                            return False

                        bbox_height = bbox["y2"] - bbox["y1"]
                        bbox_width = bbox["x2"] - bbox["x1"]
                        if bbox_width > 0:
                            aspect_ratio = bbox_height / bbox_width
                            if aspect_ratio < 1.2:
                                return True

                        if len(kpts) >= 15:
                            def _valid(idx):
                                if idx >= len(kpts):
                                    return None
                                pt = kpts[idx]
                                if pt[0] <= 0:
                                    return None
                                if len(pt) > 2 and pt[2] <= min_conf:
                                    return None
                                return pt

                            left_hip = _valid(11)
                            right_hip = _valid(12)
                            left_knee = _valid(13)
                            right_knee = _valid(14)
                            left_ankle = _valid(15)
                            right_ankle = _valid(16)

                            if left_hip is not None and left_knee is not None:
                                hip_knee_diff = left_knee[1] - left_hip[1]
                                if hip_knee_diff < bbox_height * 0.15:
                                    return True

                            if right_hip is not None and right_knee is not None:
                                hip_knee_diff = right_knee[1] - right_hip[1]
                                if hip_knee_diff < bbox_height * 0.15:
                                    return True

                            if left_hip is not None and left_ankle is not None:
                                hip_ankle_diff = left_ankle[1] - left_hip[1]
                                if hip_ankle_diff < bbox_height * 0.3:
                                    return True

                            if right_hip is not None and right_ankle is not None:
                                hip_ankle_diff = right_ankle[1] - right_hip[1]
                                if hip_ankle_diff < bbox_height * 0.3:
                                    return True

                        return False
                    
                    if court_roi_active and court_polygon is not None:
                        # PRECISE POLYGON-BASED FILTERING using cv2.pointPolygonTest
                        # This uses the actual court boundaries from manual keypoints
                        in_court_skeletons = []
                        for s in skeleton_data:
                            # Check if center point (or feet position) is inside court polygon
                            # Use feet position (bottom of bbox) for more accurate court boundary check
                            bbox = s["bbox"]
                            feet_y = bbox["y2"] if bbox else s["center"][1]
                            check_point = (float(s["center"][0]), float(feet_y))
                            
                            # cv2.pointPolygonTest returns:
                            # > 0 if inside, = 0 if on edge, < 0 if outside
                            result = cv2.pointPolygonTest(court_polygon, check_point, measureDist=False)
                            if result >= 0:  # Inside or on edge of court polygon
                                # Additional filter: Skip sitting/crouching poses (likely judges)
                                if is_sitting_pose(s["kpts"], bbox, s.get("conf")):
                                    # Person is inside court but appears to be sitting (likely judge)
                                    # Still add them but mark as potential judge for later filtering
                                    s["is_sitting"] = True
                                else:
                                    s["is_sitting"] = False
                                in_court_skeletons.append(s)
                    else:
                        # FALLBACK: Simple rectangular filter (central 75% of frame)
                        # Badminton courts are typically in the central 75% of the frame (horizontally)
                        # Judges/line judges are on the edges
                        COURT_X_MIN = width * 0.12  # Left boundary (12% from left)
                        COURT_X_MAX = width * 0.88  # Right boundary (88% from left)
                        COURT_Y_MIN = height * 0.05  # Top boundary
                        COURT_Y_MAX = height * 0.95  # Bottom boundary
                        
                        in_court_skeletons = []
                        for s in skeleton_data:
                            if COURT_X_MIN <= s["center"][0] <= COURT_X_MAX and COURT_Y_MIN <= s["center"][1] <= COURT_Y_MAX:
                                # Mark sitting/standing for fallback too
                                s["is_sitting"] = is_sitting_pose(s["kpts"], s["bbox"], s.get("conf"))
                                in_court_skeletons.append(s)
                    
                    # Filter to active players based on cumulative movement AND pose type
                    # IMPROVED LOGIC: In badminton, we KNOW there are exactly 2 players
                    # Always prefer standing skeletons over sitting ones (judges sit, players stand)
                    
                    # First, separate standing vs sitting skeletons
                    standing_skeletons = [s for s in in_court_skeletons if not s.get("is_sitting", False)]
                    sitting_skeletons = [s for s in in_court_skeletons if s.get("is_sitting", False)]
                    
                    # Log if we filtered out sitting persons
                    if sitting_skeletons and processed_count <= 10:
                        print(f"[MODAL] Frame {frame_count}: Filtered {len(sitting_skeletons)} sitting person(s) (likely judges)")
                    
                    # Prefer standing skeletons - only use sitting ones if we don't have enough standing
                    if len(standing_skeletons) >= 2:
                        # We have enough standing players, ignore sitting persons entirely
                        candidate_skeletons = standing_skeletons
                    elif len(standing_skeletons) == 1 and len(sitting_skeletons) >= 1:
                        # Only 1 standing - check if sitting person has high movement (might be crouching player)
                        high_movement_sitting = [
                            s for s in sitting_skeletons
                            if s.get("cumulative_movement", 0) >= MIN_CUMULATIVE_MOVEMENT
                        ]
                        candidate_skeletons = standing_skeletons + high_movement_sitting
                    else:
                        # Fallback to all detected skeletons
                        candidate_skeletons = in_court_skeletons
                    
                    if len(candidate_skeletons) <= 2:
                        # If 2 or fewer candidates, keep all of them
                        # This ensures we don't filter out the far-side player who may have
                        # intermittent detections (low cumulative movement due to tracking gaps)
                        active_skeletons = sorted(
                            candidate_skeletons,
                            key=lambda s: s["area"],  # Sort by size (near player first)
                            reverse=True
                        )
                    elif frame_count > MOVEMENT_WARMUP_FRAMES:
                        # More than 2 candidates: filter by movement to remove stationary people
                        active_skeletons = sorted(
                            candidate_skeletons,
                            key=lambda s: s["cumulative_movement"],
                            reverse=True
                        )
                        
                        # Take top 2 that have sufficient movement
                        active_skeletons = [
                            s for s in active_skeletons
                            if s["cumulative_movement"] >= MIN_CUMULATIVE_MOVEMENT
                        ][:2]
                    else:
                        # During warmup with >2 candidates, use largest 2 by area
                        active_skeletons = sorted(
                            candidate_skeletons,
                            key=lambda s: s["area"],
                            reverse=True
                        )[:2]
                    
                    # Add validated player bounding boxes to badminton_detections
                    for skel in active_skeletons:
                        bbox = skel["bbox"]
                        badminton_detections["players"].append({
                            "class": "player",
                            "confidence": 0.9,
                            "x": float((bbox["x1"] + bbox["x2"]) / 2),
                            "y": float((bbox["y1"] + bbox["y2"]) / 2),
                            "width": float(bbox["x2"] - bbox["x1"]),
                            "height": float(bbox["y2"] - bbox["y1"]),
                            "class_id": 0,
                            "detection_id": skel["track_id"],
                        })
                    
                    # =========================================================
                    # ROBUST PLAYER-SKELETON MATCHING via PlayerIdentityTracker
                    # =========================================================
                    # Uses composite cost matching with:
                    # - Velocity prediction (EMA-smoothed)
                    # - Court-side consistency priors
                    # - YOLO track ID continuity
                    # - Bounding box area similarity
                    # - Swap detection and correction
                    matched_players = identity_tracker.match_skeletons(
                        active_skeletons, frame_count
                    )

                    # Attach canonical player_id to each emitted player bbox
                    # by mapping its track_id through the identity tracker's
                    # output. Lets the frontend render labels directly from
                    # YOLO26 + PlayerIdentityTracker instead of re-deriving
                    # via center-proximity. Matched_players preserves the
                    # keypoint-array object identity of the source skeleton,
                    # so we can link pid -> track_id here.
                    track_id_to_pid: Dict[Any, int] = {}
                    for _pid, _kpts, _ in matched_players:
                        for _skel in active_skeletons:
                            if _skel["kpts"] is _kpts:
                                _tid = _skel.get("track_id")
                                if _tid is not None and _tid >= 0:
                                    track_id_to_pid[_tid] = _pid
                                break
                    for _bbox in badminton_detections["players"]:
                        _tid = _bbox.get("detection_id")
                        if _tid is not None and _tid in track_id_to_pid:
                            _bbox["player_id"] = track_id_to_pid[_tid]

                    # Log tracker stats periodically
                    if frame_count == identity_tracker.CALIBRATION_FRAMES + 1:
                        stats = identity_tracker.get_stats()
                        await send_log(
                            f"Player identity calibration complete: "
                            f"midline_y={stats['court_midline_y']:.0f}, "
                            f"P0={stats['player_0_side']}, P1={stats['player_1_side']}",
                            "success", "processing"
                        )
                    if identity_tracker.total_swaps_corrected > 0 and frame_count % 100 == 0:
                        await send_log(
                            f"Identity tracker: {identity_tracker.total_swaps_corrected} swap(s) corrected so far",
                            "info", "processing"
                        )
                    
                    # Process matched players
                    for player_id, kpts, conf in matched_players:
                        player_data = {
                            "player_id": player_id,
                            "keypoints": [],
                            "center": {"x": 0.0, "y": 0.0},
                            "current_speed": 0.0,
                        }
                        
                        for kp_idx, (pt, c) in enumerate(zip(kpts, conf if conf is not None else [0.5] * len(kpts))):
                            if kp_idx < len(keypoint_names):
                                player_data["keypoints"].append({
                                    "name": keypoint_names[kp_idx],
                                    "x": float(pt[0]),
                                    "y": float(pt[1]),
                                    "confidence": float(c),
                                })
                        
                        # Calculate center position (ankle/feet midpoint for accurate court position)
                        # Feet position is where the player actually stands on the court
                        left_ankle = next((k for k in player_data["keypoints"] if k["name"] == "left_ankle"), None)
                        right_ankle = next((k for k in player_data["keypoints"] if k["name"] == "right_ankle"), None)
                        left_hip = next((k for k in player_data["keypoints"] if k["name"] == "left_hip"), None)
                        right_hip = next((k for k in player_data["keypoints"] if k["name"] == "right_hip"), None)
                        
                        center_x = None
                        center_y = None
                        
                        # Primary: Use ankle midpoint (most accurate for court position)
                        if left_ankle and right_ankle and left_ankle.get("x", 0) > 0 and right_ankle.get("x", 0) > 0:
                            center_x = (left_ankle["x"] + right_ankle["x"]) / 2
                            center_y = (left_ankle["y"] + right_ankle["y"]) / 2
                        # Fallback: Use hip midpoint if ankles not visible
                        elif left_hip and right_hip and left_hip.get("x", 0) > 0 and right_hip.get("x", 0) > 0:
                            center_x = (left_hip["x"] + right_hip["x"]) / 2
                            center_y = (left_hip["y"] + right_hip["y"]) / 2
                        
                        if center_x is not None and center_y is not None:
                            player_data["position"] = {
                                "x": center_x,
                                "y": center_y,
                            }
                            player_data["center"] = {
                                "x": center_x,
                                "y": center_y,
                            }
                            
                            # Add to player positions for summary metrics
                            if player_id in player_positions:
                                player_positions[player_id].append({
                                    "frame": frame_count,
                                    "x": center_x,
                                    "y": center_y,
                                })
                            
                            # Track player for speed calculation
                            track_id = player_id
                            current_speed = 0.0
                            is_valid_tracking = True  # Track if we should update player_tracks
                            
                            if track_id in player_tracks:
                                prev = player_tracks[track_id]
                                dt = (frame_count - prev["frame"]) / fps
                                
                                if dt > 0:
                                        dx = center_x - prev["x"]
                                        dy = center_y - prev["y"]
                                        distance_px = np.sqrt(dx**2 + dy**2)
                                        
                                        # PIXEL-BASED SANITY CHECK FIRST
                                        # Max reasonable pixel movement per frame at 30fps:
                                        # A fast player (~7m/s = 25km/h) on a 1080p video (~500px court width)
                                        # would move ~15px per frame.
                                        # REDUCED threshold from 150px to 80px to catch judge jumps
                                        # 80px allows for ~4 frames of fast movement without triggering
                                        # while catching single-frame jumps to judges sitting outside court
                                        MAX_PX_PER_FRAME = max(80, int(0.07 * max(width, height)))
                                        frames_elapsed = max(1, frame_count - prev["frame"])
                                        px_per_frame = distance_px / frames_elapsed
                                        
                                        if px_per_frame > MAX_PX_PER_FRAME:
                                            # Tracking error - likely ID swap, don't update tracking
                                            current_speed = 0.0
                                            is_valid_tracking = False
                                        else:
                                            if homography_matrix is not None:
                                                pt_cur = cv2.perspectiveTransform(
                                                    np.array([[[center_x, center_y]]], dtype=np.float32),
                                                    homography_matrix
                                                )
                                                pt_prev = cv2.perspectiveTransform(
                                                    np.array([[[prev["x"], prev["y"]]]], dtype=np.float32),
                                                    homography_matrix
                                                )
                                                dm_x = pt_cur[0][0][0] - pt_prev[0][0][0]
                                                dm_y = pt_cur[0][0][1] - pt_prev[0][0][1]
                                                distance_m = float(np.sqrt(dm_x**2 + dm_y**2))
                                            else:
                                                reference_dimension = max(width, height)
                                                meters_per_pixel = 13.4 / (reference_dimension * 0.8)
                                                distance_m = distance_px * meters_per_pixel
                                            speed_mps = distance_m / dt
                                            
                                            # PHYSIOLOGICAL SPEED LIMITS FOR BADMINTON
                                            # Research notes:
                                            # - Typical badminton movement: 1-4 m/s (4-15 km/h)
                                            # - Quick recoveries/lunges: 4-7 m/s (15-25 km/h)
                                            # - Maximum explosive burst (rare): 7-9 m/s (25-32 km/h)
                                            # - Speeds above 25 km/h should be RARE - likely tracking error
                                            
                                            # Threshold for rejecting datapoints entirely
                                            # Any speed above this is DISCARDED, not capped
                                            MAX_VALID_SPEED_MPS = 8.5
                                            
                                            # Per-frame distance check (accounts for dt)
                                            MAX_DISTANCE_PER_FRAME = 0.25  # meters (reduced for accuracy)
                                            distance_per_frame = distance_m / frames_elapsed
                                            
                                            # Determine if this is a valid measurement
                                            is_valid_measurement = True
                                            
                                            if distance_per_frame > MAX_DISTANCE_PER_FRAME:
                                                # Large position jump - likely tracking error/ID swap to judge
                                                is_valid_measurement = False
                                                is_valid_tracking = False  # Don't update position
                                                print(f"[MODAL] Player {track_id} frame {frame_count}: Rejected - distance jump {distance_per_frame:.3f}m/frame")
                                            elif speed_mps > MAX_VALID_SPEED_MPS:
                                                # Impossible speed - reject this datapoint entirely
                                                # This is likely a tracking jump to a judge
                                                is_valid_measurement = False
                                                is_valid_tracking = False  # Don't update position
                                                print(f"[MODAL] Player {track_id} frame {frame_count}: Rejected - speed {speed_mps*3.6:.1f} km/h > {MAX_VALID_SPEED_MPS*3.6:.1f} km/h limit")
                                            
                                            if is_valid_measurement:
                                                current_speed = speed_mps * 3.6  # Convert m/s to km/h
                                                
                                                # Apply median filter for additional outlier detection
                                                if track_id in player_speed_windows:
                                                    window = player_speed_windows[track_id]
                                                    
                                                    # Check against median BEFORE adding to window
                                                    if len(window) >= 3:
                                                        sorted_window = sorted(window)
                                                        median_speed = sorted_window[len(sorted_window) // 2]

                                                        # If current speed is >3x the recent median, reject entirely
                                                        # Use 3x (not 2x) to allow legitimate quick accelerations
                                                        # common in badminton (e.g., standing → lunge)
                                                        if current_speed > median_speed * 3.0 and median_speed > 2.0:
                                                            # This is an outlier spike - discard it
                                                            is_valid_measurement = False
                                                            is_valid_tracking = False
                                                            print(f"[MODAL] Player {track_id} frame {frame_count}: Rejected by median filter - {current_speed:.1f} km/h > 2x median {median_speed:.1f} km/h")
                                                            current_speed = 0.0
                                                        else:
                                                            # Valid reading - add to window
                                                            window.append(current_speed)
                                                            if len(window) > SPEED_WINDOW_SIZE:
                                                                window.pop(0)
                                                    else:
                                                        # Building up window - add this reading
                                                        window.append(current_speed)
                                            else:
                                                current_speed = 0.0  # Rejected measurement
                                            
                                            # Only record VALID measurements for statistics
                                            if is_valid_measurement and current_speed > 0:
                                                # Track total distance
                                                if track_id in player_distances:
                                                    player_distances[track_id] += distance_m
                                                
                                                # Track speeds for averaging
                                                if track_id in player_speeds:
                                                    player_speeds[track_id].append(current_speed)
                            
                            player_data["current_speed"] = current_speed
                            
                            # Only update tracking if this was valid (not a position jump/ID swap)
                            if is_valid_tracking:
                                player_tracks[track_id] = {
                                    "x": center_x,
                                    "y": center_y,
                                    "frame": frame_count,
                                }
                        
                        # Classify pose from keypoints
                        if player_data["keypoints"]:
                            pose_result = classify_pose(player_data["keypoints"])
                            player_data["pose"] = {
                                "pose_type": pose_result["pose_type"],
                                "confidence": pose_result["confidence"],
                                "body_angles": pose_result.get("body_angles"),
                            }
                        else:
                            player_data["pose"] = {
                                "pose_type": "unknown",
                                "confidence": 0.0,
                                "body_angles": None,
                            }
                        
                        frame_data["players"].append(player_data)
            
            # Sort players by player_id for consistent ordering in output
            # This ensures Player 0 always comes before Player 1 in the array,
            # preventing frontend rendering issues from inconsistent array ordering
            frame_data["players"].sort(key=lambda p: p["player_id"])
            
            skeleton_frames_file.write(json.dumps(frame_data) + "\n")
            skeleton_frame_count += 1

            # Send progress updates every 2 seconds
            now = time.time()
            if now - last_progress_update >= 2.0:
                progress = (frame_count / total_frames) * 100
                elapsed = now - phase_start
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                mem_mb = get_memory_mb()
                await send_status_update("processing", progress, frame_count, total_frames)
                # Log detailed diagnostics every 10 seconds
                if int(elapsed) % 10 < 3:
                    print(f"[MODAL] Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"{fps_actual:.1f} fps | RAM: {mem_mb:.0f} MB | "
                          f"status_failures: {status_update_failures}")
                last_progress_update = now
        
        cap.release()
        skeleton_frames_file.close()

        frame_loop_time = time.time() - phase_start
        mem_mb = get_memory_mb()
        fps_actual = processed_count / frame_loop_time if frame_loop_time > 0 else 0
        await send_log(
            f"Frame loop complete: {processed_count} frames in {frame_loop_time:.1f}s "
            f"({fps_actual:.1f} fps, RAM: {mem_mb:.0f} MB)",
            "success", "processing"
        )
        phase_start = time.time()

        # Read skeleton frames back from disk
        await send_log("Loading skeleton data for post-processing...", "info", "processing")
        skeleton_frames = []
        with open(skeleton_frames_path, "r") as f:
            for line in f:
                skeleton_frames.append(json.loads(line))
        print(f"[MODAL] Loaded {len(skeleton_frames)} skeleton frames from disk")

        await send_log(f"Processed {processed_count} frames", "success", "processing")
        
        # Log identity tracker summary
        tracker_stats = identity_tracker.get_stats()
        await send_log(
            f"Identity tracker: calibrated={tracker_stats['calibration_complete']}, "
            f"swaps_corrected={tracker_stats['total_swaps_corrected']}, "
            f"P0_positions={tracker_stats['player_0_positions']}, "
            f"P1_positions={tracker_stats['player_1_positions']}",
            "info", "processing"
        )
        if tracker_stats['total_swaps_corrected'] > 0:
            await send_log(
                f"⚠️ {tracker_stats['total_swaps_corrected']} skeleton ID swap(s) were detected and corrected",
                "warning", "processing"
            )

        # Finalize raw-tracker metrics: summary JSON + stdout headline.
        try:
            metrics_summary = tracker_metrics.finalize(identity_tracker_stats=tracker_stats)
            await send_log(
                f"Tracker metrics: id_switches={metrics_summary['id_switches']} "
                f"({metrics_summary['id_switches_per_min']}/min), "
                f"coverage_2p={metrics_summary['coverage_both_players']}, "
                f"unique_tids={metrics_summary['unique_track_ids']}, "
                f"teleports={metrics_summary['teleport_events']}",
                "info", "processing"
            )
        except Exception as _mex:
            print(f"[tracker-metrics] finalize failed: {_mex}")
        
        # Build player summary data from tracked positions
        # Use physiological limits based on badminton research:
        # - Typical footwork: 1-4 m/s (4-15 km/h)
        # - Quick lunges/recoveries: 4-7 m/s (15-25 km/h)
        # - Maximum burst (extremely rare): up to 7 m/s (25 km/h)
        # - Anything above 20 km/h sustained is SUSPICIOUS
        # - Anything above 25 km/h is REJECTED
        MAX_VALID_SPEED_KMH = 25.0       # Absolute max - matches per-frame filter
        TYPICAL_MAX_SPEED_KMH = 15.0     # Realistic average max
        SUSPICIOUS_SPEED_KMH = 20.0      # Above this is flagged as suspicious
        
        players_summary = []
        for player_id in range(2):
            speeds = player_speeds.get(player_id, [])
            positions = player_positions.get(player_id, [])
            distance = player_distances.get(player_id, 0.0)
            
            # STRICT filtering - reject all speeds above threshold, NO FALLBACK
            # Stage 1: Hard filter - remove ANYTHING above max valid speed
            filtered_speeds = [s for s in speeds if s <= MAX_VALID_SPEED_KMH]
            
            # Stage 2: If we have enough data, apply IQR-based outlier removal
            if len(filtered_speeds) >= 5:
                sorted_speeds = sorted(filtered_speeds)
                q1_idx = len(sorted_speeds) // 4
                q3_idx = 3 * len(sorted_speeds) // 4
                q1 = sorted_speeds[q1_idx]
                q3 = sorted_speeds[q3_idx]
                iqr = q3 - q1
                upper_bound = min(q3 + 1.5 * iqr, SUSPICIOUS_SPEED_KMH)  # Cap IQR bound
                
                # Filter to within IQR bounds
                filtered_speeds = [s for s in filtered_speeds if s <= upper_bound]
            
            # Stage 3: Remove top 5% as final safety measure
            if len(filtered_speeds) >= 5:
                sorted_filtered = sorted(filtered_speeds)
                cutoff_idx = int(len(sorted_filtered) * 0.95)
                if cutoff_idx > 0:
                    filtered_speeds = sorted_filtered[:cutoff_idx]
            
            # Calculate stats from filtered data only - NO FALLBACK to unfiltered
            avg_speed = sum(filtered_speeds) / len(filtered_speeds) if filtered_speeds else 0.0
            max_speed = max(filtered_speeds) if filtered_speeds else 0.0
            
            # Apply final physiological caps (safety net)
            avg_speed = min(avg_speed, TYPICAL_MAX_SPEED_KMH)
            max_speed = min(max_speed, MAX_VALID_SPEED_KMH)
            
            players_summary.append({
                "player_id": player_id,
                "avg_speed": round(avg_speed, 2),
                "max_speed": round(max_speed, 2),
                "total_distance": round(distance, 2),
                "positions": positions,  # Array of {frame, x, y}
                "keypoints_history": [],  # Empty for now, can be populated if needed
            })
        
        # player_id is 0-indexed internally, but display as 1-indexed (Player 1, Player 2)
        await send_log(f"Player 1: {len(player_positions[0])} positions, {player_distances[0]:.1f}m", "info", "processing")
        await send_log(f"Player 2: {len(player_positions[1])} positions, {player_distances[1]:.1f}m", "info", "processing")
        phase_start = time.time()
        await send_log("Starting rally detection...", "info", "processing")
        detected_rallies = []

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
            _RALLY_STATIC_COUNT = 3

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

                _rally_static_clusters = [c for c in _rally_static_clusters if c["count"] >= _RALLY_STATIC_COUNT]

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
                from rally_detection import detect_rallies

                detected_rallies = detect_rallies(
                    rally_shuttle_positions,
                    fps=fps,
                    total_frames=total_frames,
                )

                if detected_rallies:
                    await send_log(
                        f"Detected {len(detected_rallies)} rallies",
                        "success", "processing"
                    )
                else:
                    await send_log("No rallies detected from shuttle data", "warning", "processing")
            except Exception as e:
                await send_log(f"Rally detection error: {e}", "warning", "processing")
                print(f"[MODAL] Rally detection error: {e}")

        # Run shot-gap detector on the same skeleton data, then union with the
        # gradient-based detector for richer coverage. The shot-gap detector
        # mirrors the in-browser timeline (useAdvancedAnalytics.ts) so users
        # see the same rally set both in the timeline and as cut clips.
        try:
            sys.path.insert(0, "/root")
            from rally_detection_shot_gap import detect_rallies_from_shots, union_rallies
            shot_gap_rallies = detect_rallies_from_shots(skeleton_frames, fps)
            if shot_gap_rallies:
                await send_log(
                    f"Shot-gap detector found {len(shot_gap_rallies)} additional candidate rallies",
                    "info", "processing"
                )
            union = union_rallies(detected_rallies, shot_gap_rallies, fps=fps)
            if len(union) > len(detected_rallies):
                await send_log(
                    f"Combined detection: {len(detected_rallies)} (gradient) + {len(shot_gap_rallies)} (shot-gap) "
                    f"-> {len(union)} (union, dedup overlap >50%)",
                    "success", "processing"
                )
            detected_rallies = union  # use the union for the rest of the pipeline
        except Exception as e:
            await send_log(f"Shot-gap detector failed (continuing with gradient only): {e}", "warning", "processing")
            print(f"[MODAL] Shot-gap detector failed: {e}")

        rally_time = time.time() - phase_start
        mem_mb = get_memory_mb()
        await send_log(f"Rally detection complete in {rally_time:.1f}s (RAM: {mem_mb:.0f} MB)", "info", "processing")
        phase_start = time.time()

        # Build results
        results_data = {
            "video_id": video_id,
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "video_width": width,
            "video_height": height,
            "players": players_summary,
            "skeleton_data": skeleton_frames,
            "shuttle": None,
            "court_detection": None,
            "player_zone_analytics": None,
            "rallies": detected_rallies,
        }
        
        # Upload full results as a JSON file to Supabase Storage (the
        # 'results' bucket). Big payloads (long videos) easily exceed any
        # row-size limit, so we keep the heavy data out of Postgres.
        await send_log("Serializing results to JSON...", "info", "processing")
        serialize_start = time.time()
        results_json = json.dumps(results_data).encode("utf-8")
        serialize_time = time.time() - serialize_start
        results_mb = len(results_json) / (1024 * 1024)
        if results_mb > 50:
            await send_log(
                f"Warning: Results JSON is very large ({results_mb:.1f} MB). "
                f"Consider processing shorter video segments.",
                "warning", "processing"
            )
        mem_mb = get_memory_mb()
        await send_log(
            f"Uploading results ({results_mb:.1f} MB, serialized in {serialize_time:.1f}s, RAM: {mem_mb:.0f} MB)...",
            "info", "processing"
        )

        # Player-identity thumbnails - best-effort. Cosmetic UI feature;
        # never block the main analysis/results write on this.
        # Must run BEFORE `del skeleton_frames` so we still have the frame list.
        try:
            chosen_frame = _pick_best_two_player_frame(
                skeleton_frames,
                video_height=int(height),
                max_search_seconds=10.0,
                fps=fps,
            )
            if chosen_frame is None:
                await send_log(
                    "No qualifying 2-player frame for player thumbnails; skipping",
                    "info", "processing",
                )
            else:
                thumbs = _capture_player_thumbnails(
                    video_path,
                    frame_number=int(chosen_frame["frame"]),
                    players=chosen_frame["players"][:2],
                )
                if thumbs is None:
                    await send_log(
                        "Player thumbnail capture failed; skipping",
                        "warning", "processing",
                    )
                else:
                    def _upload_thumb(player_idx: int, jpeg_bytes: bytes) -> str:
                        sb = supabase_client()
                        path = f"{owner_id}/{video_id}/player_{player_idx}.jpg"
                        sb.storage.from_("thumbnails").upload(
                            path=path,
                            file=jpeg_bytes,
                            file_options={"content-type": "image/jpeg", "upsert": "true"},
                        )
                        return path

                    p0_path = await asyncio.to_thread(_upload_thumb, 0, thumbs[0])
                    p1_path = await asyncio.to_thread(_upload_thumb, 1, thumbs[1])

                    # Merge thumbnail paths into existing player_labels (preserve
                    # any user-set names/swaps that may already be on the row).
                    def _merge_labels():
                        sb = supabase_client()
                        existing = (
                            sb.table("videos")
                            .select("player_labels")
                            .eq("id", video_id)
                            .single()
                            .execute()
                        )
                        labels = (existing.data.get("player_labels") if existing.data else None) or {}
                        labels["player_0_thumbnail_path"] = p0_path
                        labels["player_1_thumbnail_path"] = p1_path
                        sb.table("videos").update({"player_labels": labels}).eq("id", video_id).execute()

                    await asyncio.to_thread(_merge_labels)
                    await send_log("Player thumbnails uploaded", "success", "processing")
        except Exception as thumb_err:
            print(f"[MODAL] Player thumbnail capture error: {thumb_err}")
            await send_log(
                f"Player thumbnail capture error (non-fatal): {thumb_err}",
                "warning", "processing",
            )

        # Free skeleton_frames from memory now that we have the JSON
        del skeleton_frames
        del results_data

        results_storage_path = f"{owner_id}/{video_id}/results.json"

        # Step 1: Upload the results JSON to Supabase Storage. The supabase-py
        # client is synchronous, so we run it in a thread to avoid blocking
        # the event loop while the upload streams.
        await asyncio.to_thread(
            lambda: supabase_client()
            .storage
            .from_("results")
            .upload(
                path=results_storage_path,
                file=results_json,
                file_options={
                    "content-type": "application/json",
                    "upsert": "true",
                },
            )
        )

        print(f"[MODAL] Results uploaded to storage: {results_storage_path}")

        # Step 2: Write metadata + storage path back into the videos row.
        # Status flip to 'completed' is deferred until after rally clip
        # generation; leave the existing 'processing' status untouched here.
        results_meta = {
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "player_count": 2,
            "has_court_detection": False,
            "has_rally_detection": len(detected_rallies) > 0,
            "rally_count": len(detected_rallies),
            "tracknet_used": tracknet_available,
        }

        await asyncio.to_thread(
            lambda: supabase_client()
            .table("videos")
            .update({
                "results_meta": results_meta,
                "results_storage_path": results_storage_path,
            })
            .eq("id", video_id)
            .execute()
        )

        # Cut per-rally clips and upload to the 'clips' bucket. Runs AFTER the
        # results JSON is durable on Storage, and BEFORE the status flip to
        # 'completed' so the UI sees "completed" only when clips are also
        # ready. Soft-failure: if clip generation throws, the video still
        # reaches 'completed' (the mobile rally library is just empty for
        # this video and a warning log row is written).
        if detected_rallies:
            await send_log(
                f"Cutting {len(detected_rallies)} rally clips...",
                "info", "processing"
            )
            try:
                await asyncio.to_thread(
                    lambda: cut_and_upload_rally_clips(
                        video_path=str(video_path),
                        rallies=detected_rallies,
                        video_id=video_id,
                        owner_id=owner_id,
                    )
                )
                await send_log(
                    f"Rally clips uploaded for {len(detected_rallies)} rallies",
                    "success", "processing"
                )
            except Exception as e:
                try:
                    await send_log(
                        f"clip generation pass failed: {str(e)[:200]}",
                        "warning", "processing",
                    )
                except Exception:
                    pass

        upload_time = time.time() - phase_start
        total_time = time.time() - pipeline_start
        await send_log(
            f"Analysis complete! Upload took {upload_time:.1f}s. "
            f"Total pipeline: {total_time:.1f}s ({total_time/60:.1f} min)",
            "success", "processing"
        )

        # Flip the videos row to 'completed' now that results JSON is durable
        # on Storage and per-rally clips have been cut + uploaded.
        await send_status_update(
            "completed",
            progress=1.0,
            current_frame=processed_count,
            total_frames=total_frames,
        )

        # Cleanup: remove every cache artifact we wrote for this video_id so the
        # /cache volume never serves stale data to a later run. Must run AFTER
        # clip cutting — the clip cutter reads video_path. Uses a glob so any
        # per-video file (mp4, skeleton jsonl, tracker yaml, etc.) is swept
        # without having to enumerate paths individually.
        for artifact in Path("/cache").glob(f"{video_id}*"):
            artifact.unlink(missing_ok=True)
        vol.commit()

        print(f"[MODAL] Processing complete for video: {video_id} in {total_time:.1f}s")

        return {
            "status": "completed",
            "video_id": video_id,
            "processed_frames": processed_count,
            "total_frames": total_frames,
        }

    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        elapsed = time.time() - pipeline_start if 'pipeline_start' in locals() else 0
        print(f"[MODAL] Error processing video {video_id} after {elapsed:.1f}s: {error_msg}")
        print(f"[MODAL] Traceback:\n{tb}")

        await send_status_update("failed", error=error_msg)
        await send_log(f"Processing failed after {elapsed:.1f}s: {error_msg}", "error", "processing")

        # Close skeleton file if still open, then sweep every cache artifact
        # tied to this video_id so a retry starts from scratch.
        if 'skeleton_frames_file' in locals() and not skeleton_frames_file.closed:
            skeleton_frames_file.close()
        for artifact in Path("/cache").glob(f"{video_id}*"):
            artifact.unlink(missing_ok=True)

        try:
            vol.commit()
        except Exception:
            pass

        return {
            "status": "failed",
            "video_id": video_id,
            "error": error_msg,
        }


if __name__ == "__main__":
    # Local testing
    print("Modal app ready for deployment")
    print("Deploy with: modal deploy backend/modal_supabase_processor.py")
