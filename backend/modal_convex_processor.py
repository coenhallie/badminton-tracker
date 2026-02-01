"""
Modal Video Processor for Convex Integration

This module provides GPU-accelerated video processing that:
1. Downloads videos from Convex storage
2. Processes with YOLO models (including pose estimation)
3. Sends progress updates to Convex HTTP endpoints
4. Uploads results back to Convex
"""

import os
import asyncio
import json
import tempfile
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import time

import modal


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

# Modal app configuration
app = modal.App("badminton-tracker-processor")

# Volume for temporary storage and model caching
vol = modal.Volume.from_name("badminton-processor-cache", create_if_missing=True)

# Volume for trained models (shared with modal_inference.py)
models_vol = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)
MODELS_PATH = "/models"

# Image with all dependencies
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
    )
)


@app.function(
    gpu="T4",
    timeout=1800,  # 30 minutes max
    image=image,
    volumes={"/cache": vol, MODELS_PATH: models_vol},
    secrets=[modal.Secret.from_name("convex-secrets")],
)
@modal.fastapi_endpoint(method="POST")
async def process_video(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a video from Convex storage.
    
    Request format:
    {
        "videoId": "...",
        "videoUrl": "https://...",  # Temporary URL from Convex
        "callbackUrl": "https://....convex.site"  # Convex HTTP endpoint base URL
        "manualCourtKeypoints": {  # Optional - for court ROI filtering
            "top_left": [x, y],
            "top_right": [x, y],
            "bottom_right": [x, y],
            "bottom_left": [x, y],
            ...
        }
    }
    """
    import httpx
    import cv2
    import numpy as np
    from ultralytics import YOLO
    
    video_id = request.get("videoId")
    video_url = request.get("videoUrl")
    callback_url = request.get("callbackUrl")
    manual_court_keypoints = request.get("manualCourtKeypoints")  # Optional court keypoints
    
    if not all([video_id, video_url, callback_url]):
        return {"error": "Missing required fields: videoId, videoUrl, callbackUrl"}
    
    print(f"[MODAL] Starting processing for video: {video_id}")
    print(f"[MODAL] Video URL: {video_url[:100]}...")
    print(f"[MODAL] Callback URL: {callback_url}")
    
    # Helper function to send updates to Convex
    async def send_status_update(
        status: str = "processing",
        progress: float = 0,
        current_frame: int = 0,
        total_frames: int = 0,
        error: Optional[str] = None
    ):
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{callback_url}/updateStatus",
                    json={
                        "videoId": video_id,
                        "status": status,
                        "progress": progress,
                        "currentFrame": current_frame,
                        "totalFrames": total_frames,
                        "error": error,
                    },
                    timeout=10.0,
                )
            except Exception as e:
                print(f"[MODAL] Warning: Failed to send status update: {e}")
    
    async def send_log(
        message: str,
        level: str = "info",
        category: str = "processing"
    ):
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{callback_url}/addLog",
                    json={
                        "videoId": video_id,
                        "message": message,
                        "level": level,
                        "category": category,
                    },
                    timeout=10.0,
                )
            except Exception as e:
                print(f"[MODAL] Warning: Failed to send log: {e}")
    
    try:
        # Download video from Convex
        await send_log("Downloading video from storage...", "info", "processing")
        
        video_path = Path(f"/cache/{video_id}.mp4")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(video_url)
            response.raise_for_status()
            
            with open(video_path, "wb") as f:
                f.write(response.content)
        
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        await send_log(f"Video downloaded: {file_size_mb:.1f} MB", "success", "processing")
        print(f"[MODAL] Video downloaded: {video_path} ({file_size_mb:.1f} MB)")
        
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
        
        await send_status_update("processing", 0, 0, total_frames)
        
        # Load YOLO26 pose model (latest version)
        await send_log("Loading YOLO26 pose model...", "info", "model")
        pose_model = YOLO("yolo26n-pose.pt")  # Using YOLO26 pose model
        
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
        
        # Warmup both models
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = pose_model(dummy_frame, verbose=False)
        _ = detection_model(dummy_frame, verbose=False)
        await send_log("Models ready (GPU accelerated)", "success", "model")
        
        # Process frames
        await send_log("Starting frame-by-frame analysis...", "info", "processing")
        
        skeleton_frames = []
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
        sample_rate = 1  # Process every frame
        
        # Movement-based filtering using YOLO tracking IDs
        # Track cumulative movement per track ID to distinguish players from stationary judges
        track_positions: Dict[int, Dict] = {}  # track_id -> {x, y, frame}
        track_cumulative_movement: Dict[int, float] = {}  # track_id -> total movement
        # Minimum cumulative movement (pixels) to be considered a real player
        # Lowered from 500 to 100 to improve far-side player detection
        # (far players are detected intermittently and may not accumulate much tracked movement)
        MIN_CUMULATIVE_MOVEMENT = 100
        
        # Court polygon ROI filter setup (if manual keypoints provided)
        court_polygon = None
        court_roi_active = False
        
        if manual_court_keypoints:
            try:
                # Extract the 4 corners for the court polygon
                corners = []
                for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                    if key in manual_court_keypoints and manual_court_keypoints[key]:
                        pt = manual_court_keypoints[key]
                        corners.append([float(pt[0]), float(pt[1])])
                
                if len(corners) == 4:
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
        MOVEMENT_WARMUP_FRAMES = 45  # Build movement history for 1.5 seconds before filtering
        
        last_progress_update = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample rate (process every Nth frame)
            if frame_count % sample_rate != 0:
                continue
            
            processed_count += 1
            
            # Run pose estimation WITH TRACKING for consistent person IDs
            pose_results = pose_model.track(frame, persist=True, verbose=False)
            
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
                        if class_lower in ["shuttle", "shuttlecock", "birdie", "ball"]:
                            badminton_detections["shuttlecocks"].append(det_entry)
                        elif class_lower in ["racket", "racquet"]:
                            badminton_detections["rackets"].append(det_entry)
                        # Skip all other classes (chairs, tables, sports ball, etc.)
                        # Person detections are also skipped as players come from pose model
            
            # Extract best shuttle position for tracking
            shuttle_position = None
            if badminton_detections["shuttlecocks"]:
                best_shuttle = max(
                    badminton_detections["shuttlecocks"],
                    key=lambda s: s["confidence"]
                )
                shuttle_position = {
                    "x": best_shuttle["x"],
                    "y": best_shuttle["y"]
                }
            
            # Extract skeleton data from pose model with tracking
            frame_data = {
                "frame": frame_count,
                "timestamp": frame_count / fps,
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
                
                # Get tracking IDs and boxes from pose model
                has_tracking = result.boxes is not None and result.boxes.is_track
                track_ids = result.boxes.id.int().cpu().tolist() if has_tracking and result.boxes.id is not None else None
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) > 0 else None
                
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
                    def is_sitting_pose(kpts, bbox, conf=None) -> bool:
                        """
                        Detect if a skeleton represents a sitting/crouching person (likely judge).
                        
                        Key indicators:
                        1. Low height-to-width bbox aspect ratio (<1.3 = sitting)
                        2. Knees at or above hip level
                        3. Low ankle-to-hip vertical distance (legs folded)
                        
                        Returns True if the pose appears to be sitting/crouching.
                        """
                        if bbox is None:
                            return False
                        
                        # Check 1: Bounding box aspect ratio
                        # Standing person: height/width > 1.5 typically
                        # Sitting person: height/width < 1.3 typically
                        bbox_height = bbox["y2"] - bbox["y1"]
                        bbox_width = bbox["x2"] - bbox["x1"]
                        if bbox_width > 0:
                            aspect_ratio = bbox_height / bbox_width
                            if aspect_ratio < 1.2:  # Very squat = definitely sitting
                                return True
                        
                        # Check 2: Keypoint-based sitting detection
                        # Get relevant keypoints (indices: 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee)
                        if len(kpts) >= 15:
                            left_hip = kpts[11] if kpts[11][0] > 0 else None
                            right_hip = kpts[12] if kpts[12][0] > 0 else None
                            left_knee = kpts[13] if kpts[13][0] > 0 else None
                            right_knee = kpts[14] if kpts[14][0] > 0 else None
                            left_ankle = kpts[15] if len(kpts) > 15 and kpts[15][0] > 0 else None
                            right_ankle = kpts[16] if len(kpts) > 16 and kpts[16][0] > 0 else None
                            
                            # Check if knees are at similar level or above hips (sitting indicator)
                            if left_hip is not None and left_knee is not None:
                                # In sitting pose, knee Y is similar to or less than hip Y
                                hip_knee_diff = left_knee[1] - left_hip[1]
                                # If knee is at same level or above hip, likely sitting
                                if hip_knee_diff < bbox_height * 0.15:  # Knee not much below hip
                                    return True
                            
                            if right_hip is not None and right_knee is not None:
                                hip_knee_diff = right_knee[1] - right_hip[1]
                                if hip_knee_diff < bbox_height * 0.15:
                                    return True
                            
                            # Check 3: Ankle vertical distance from hip is small (folded legs)
                            if left_hip is not None and left_ankle is not None:
                                hip_ankle_diff = left_ankle[1] - left_hip[1]
                                if hip_ankle_diff < bbox_height * 0.3:  # Legs very folded
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
                    
                    # Build matched_players with consistent player ID assignment
                    matched_players = []
                    
                    if active_skeletons:
                        # Get centers for all active skeletons
                        skeleton_centers = [s["center"] for s in active_skeletons]
                        
                        # Match to previous frame's player positions by minimum distance
                        # This prevents player ID swaps when a judge briefly appears
                        used_skeleton_indices = set()
                        
                        for player_id in range(2):  # For player 0 and 1
                            if player_id in player_tracks and skeleton_centers:
                                prev_pos = player_tracks[player_id]
                                prev_x, prev_y = prev_pos["x"], prev_pos["y"]
                                
                                # Find closest unused skeleton
                                best_idx = None
                                best_dist = float("inf")
                                
                                for idx, center in enumerate(skeleton_centers):
                                    if idx in used_skeleton_indices:
                                        continue
                                    dist = ((center[0] - prev_x)**2 + (center[1] - prev_y)**2)**0.5
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = idx
                                
                                # Only match if within reasonable distance (e.g., 200px per frame)
                                if best_idx is not None and best_dist < 200:
                                    used_skeleton_indices.add(best_idx)
                                    skeleton = active_skeletons[best_idx]
                                    matched_players.append((player_id, skeleton["kpts"], skeleton["conf"]))
                        
                        # Assign remaining skeletons to unused player IDs based on Y position
                        # This ensures consistent player assignment:
                        # - Player 0 = top of video (far side of court, smaller Y value)
                        # - Player 1 = bottom of video (near side of court, larger Y value)
                        remaining_skeletons = [
                            (idx, skeleton) for idx, skeleton in enumerate(active_skeletons)
                            if idx not in used_skeleton_indices
                        ]
                        
                        if remaining_skeletons:
                            # Sort remaining by Y position (ascending = top first)
                            remaining_skeletons.sort(key=lambda x: x[1]["center"][1])
                            
                            # Get unused player IDs, sorted (0, 1)
                            unused_pids = sorted([pid for pid in range(2) if not any(m[0] == pid for m in matched_players)])
                            
                            # Assign: smallest Y -> smallest unused ID (Player 0 = top/far)
                            for (idx, skeleton), pid in zip(remaining_skeletons, unused_pids):
                                matched_players.append((pid, skeleton["kpts"], skeleton["conf"]))
                    
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
                                        MAX_PX_PER_FRAME = 80  # Reduced from 150 to catch tracking errors
                                        frames_elapsed = max(1, frame_count - prev["frame"])
                                        px_per_frame = distance_px / frames_elapsed
                                        
                                        if px_per_frame > MAX_PX_PER_FRAME:
                                            # Tracking error - likely ID swap, don't update tracking
                                            current_speed = 0.0
                                            is_valid_tracking = False
                                        else:
                                            # Valid tracking - calculate speed
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
                                            MAX_VALID_SPEED_MPS = 7.0  # 25 km/h - above this, reject datapoint
                                            
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
                                                        
                                                        # If current speed is >2x the recent median, reject entirely
                                                        if current_speed > median_speed * 2.0 and median_speed > 1.0:
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
            
            skeleton_frames.append(frame_data)
            
            # Send progress updates every 2 seconds
            now = time.time()
            if now - last_progress_update >= 2.0:
                progress = (frame_count / total_frames) * 100
                await send_status_update("processing", progress, frame_count, total_frames)
                last_progress_update = now
        
        cap.release()
        
        await send_log(f"Processed {processed_count} frames", "success", "processing")
        
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
            "shuttle_analytics": None,
            "player_zone_analytics": None,
        }
        
        # Upload full results as JSON file to Convex storage (avoids 1MB document limit)
        await send_log("Uploading results to storage...", "info", "processing")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: Get upload URL from Convex
            upload_url_response = await client.post(
                f"{callback_url}/generateUploadUrl",
                json={},
            )
            upload_url_response.raise_for_status()
            upload_data = upload_url_response.json()
            upload_url = upload_data.get("uploadUrl")
            
            if not upload_url:
                raise Exception("Failed to get upload URL from Convex")
            
            # Step 2: Upload the results JSON to Convex storage
            results_json = json.dumps(results_data).encode("utf-8")
            storage_response = await client.post(
                upload_url,
                content=results_json,
                headers={"Content-Type": "application/json"},
            )
            storage_response.raise_for_status()
            storage_result = storage_response.json()
            results_storage_id = storage_result.get("storageId")
            
            if not results_storage_id:
                raise Exception("Failed to get storageId from upload response")
            
            print(f"[MODAL] Results uploaded to storage: {results_storage_id}")
            
            # Step 3: Send metadata + storage reference to updateResults
            await client.post(
                f"{callback_url}/updateResults",
                json={
                    "videoId": video_id,
                    "resultsMeta": {
                        "duration": duration,
                        "fps": fps,
                        "total_frames": total_frames,
                        "processed_frames": processed_count,
                        "player_count": 2,
                        "has_court_detection": False,
                        "has_shuttle_analytics": False,
                    },
                    "resultsStorageId": results_storage_id,
                },
            )
        
        await send_log("Analysis complete! Results ready.", "success", "processing")
        
        # Cleanup
        video_path.unlink(missing_ok=True)
        vol.commit()
        
        print(f"[MODAL] Processing complete for video: {video_id}")
        
        return {
            "status": "completed",
            "videoId": video_id,
            "processedFrames": processed_count,
            "totalFrames": total_frames,
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"[MODAL] Error processing video {video_id}: {error_msg}")
        
        await send_status_update("failed", error=error_msg)
        await send_log(f"Processing failed: {error_msg}", "error", "processing")
        
        return {
            "status": "failed",
            "videoId": video_id,
            "error": error_msg,
        }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    # Local testing
    print("Modal app ready for deployment")
    print("Deploy with: modal deploy backend/modal_convex_processor.py")
