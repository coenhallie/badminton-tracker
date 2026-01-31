"""
Badminton Tracker Backend - FastAPI Application
Processes videos with YOLOv26 pose estimation for badminton analysis
Enhanced with Roboflow court keypoint detection for accurate measurements
and custom Roboflow badminton model for player/shuttlecock detection

Performance Optimizations Applied:
- Async frame reading with prefetching buffer
- Adaptive frame skipping based on motion detection
- Object pooling for reduced memory allocations
- Frame timing metadata for synchronization
"""

import os
import json
import asyncio
import uuid
import subprocess
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from parent directory's .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Import court detection module
from court_detection import (
    get_court_detector, BadmintonCourtDetector, COURT_WIDTH_DOUBLES, COURT_LENGTH,
    ManualCourtKeypoints, set_manual_court_keypoints, clear_manual_court_keypoints,
    get_manual_keypoints_status, CourtDetection
)

# Import multi-model detector for running multiple YOLO models simultaneously
# This replaces the legacy badminton_detection module with a more capable detector
from multi_model_detector import get_multi_model_detector, MultiModelDetector, DetectionType

# Import pose detection module for pose classification
from pose_detection import (
    PoseAnalyzer, PoseType, Keypoint as PoseKeypoint, KeypointData,
    PlayerPose as PosePlayerPose, get_pose_detector, get_smoothed_pose_detector
)

# Import shuttle analytics module for advanced speed and trajectory analysis
from shuttle_analytics import (
    ShuttleTracker, PlayerPositionAnalyzer, create_analytics_summary,
    MatchAnalytics, ShotType, recalculate_zone_analytics_from_skeleton_data
)

# Import detection smoothing module for lag reduction
from detection_smoothing import (
    DetectionSmoother, FrameInterpolator, VideoPreprocessor,
    generate_smooth_skeleton_data, build_frame_index
)

# Import heatmap generator module for player position visualization
from heatmap_generator import (
    HeatmapGenerator, HeatmapConfig, HeatmapData,
    generate_heatmap_from_skeleton_data, get_or_create_generator,
    clear_generator_cache, HeatmapColormap
)

# Import PDF export module for analysis report generation
from pdf_export import (
    PDFReportGenerator, PDFExportConfig, generate_pdf_report
)

# Import speed analytics module for real-time speed graph visualization
from speed_analytics import (
    SpeedGraphManager, get_speed_manager,
    calculate_speed_from_skeleton_data,
    SpeedZone, SPEED_ZONE_THRESHOLDS, SPEED_ZONE_COLORS
)

# Import performance optimization modules (NEW - reduces processing lag)
from performance_optimizations import (
    AsyncFrameBuffer, AdaptiveFrameSkipper,
    ObjectPool, PreallocatedArrays,
    PerformanceProfiler, FrameSynchronizer
)

# Import Modal client for remote GPU inference (optional)
try:
    from modal_client import (
        get_modal_client, ModalInferenceClient,
        initialize_modal_client, HybridInferenceManager
    )
    MODAL_CLIENT_AVAILABLE = True
except ImportError:
    MODAL_CLIENT_AVAILABLE = False
    print("Modal client not available (httpx may not be installed)")

# Initialize FastAPI app
app = FastAPI(
    title="Badminton Tracker API",
    description="Video analysis API for badminton matches using YOLOv26 pose estimation",
    version="1.0.0"
)

# CORS configuration - centralized list of allowed origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]

# CORS middleware for Vue.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# CUSTOM EXCEPTION HANDLER FOR CORS
# =============================================================================
# FastAPI's CORSMiddleware doesn't always add CORS headers to error responses.
# This custom handler ensures CORS headers are present on ALL responses,
# including 500 Internal Server Errors.

from starlette.requests import Request
# Note: JSONResponse is already imported from fastapi.responses above

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that ensures CORS headers are present on error responses.
    
    This fixes the issue where a 500 error would trigger both:
    1. CORS error (no Access-Control-Allow-Origin header)
    2. The actual 500 error
    
    By handling exceptions here and returning a proper JSONResponse,
    the CORS middleware can add the correct headers.
    """
    # Get the origin from the request
    origin = request.headers.get("origin", "")
    
    # Log the error
    import traceback
    error_traceback = traceback.format_exc()
    print(f"[GLOBAL ERROR HANDLER] Request: {request.method} {request.url}")
    print(f"[GLOBAL ERROR HANDLER] Exception: {type(exc).__name__}: {exc}")
    print(f"[GLOBAL ERROR HANDLER] Traceback:\n{error_traceback}")
    
    # Build error response
    response = JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__
        }
    )
    
    # Manually add CORS headers if origin is in allowed list
    if origin in CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP exception handler that ensures CORS headers are present on HTTP error responses.
    """
    origin = request.headers.get("origin", "")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    # Manually add CORS headers if origin is in allowed list
    if origin in CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Create directories for uploads and processed videos
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/processed", StaticFiles(directory="processed"), name="processed")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Load YOLOv26 pose model (will download on first run)
print("Loading YOLOv26 pose model...")
pose_model = YOLO("yolo26n-pose.pt")  # Using YOLO26 nano pose model - NMS-free, optimized for edge
print("Model loaded successfully!")

# Initialize court detector (uses local trained model)
print("Initializing court detector...")
court_detector = get_court_detector()
print(f"Court detector initialized (model {'loaded' if court_detector.model else 'not found'})")

# Initialize multi-model detector (supports multiple models simultaneously)
print("Initializing multi-model detector...")
multi_detector = get_multi_model_detector()
if multi_detector.is_available:
    print(f"Multi-model detector initialized with {len(multi_detector.models)} model(s):")
    for name in multi_detector.available_models:
        print(f"  - {name}: {list(multi_detector.model_classes.get(name, {}).values())}")
else:
    print("Multi-model detector: No models loaded")

# Initialize pose analyzer for pose classification
print("Initializing pose analyzer...")
pose_analyzer = PoseAnalyzer()
print("Pose analyzer initialized (classifies poses from keypoints)")

# Initialize Modal inference client (optional - for GPU acceleration)
modal_client = None
modal_available = False
if MODAL_CLIENT_AVAILABLE:
    print("Initializing Modal GPU inference client...")
    _temp_client = get_modal_client()
    if _temp_client.is_enabled:
        print(f"  Modal endpoint: {_temp_client.config.endpoint_url}")
        print("  ⏳ Modal health check will be performed on first video analysis")
        modal_client = _temp_client
        modal_available = True
    else:
        print("  Modal inference disabled (USE_MODAL_INFERENCE=false in .env)")
else:
    print("Modal client not available (install httpx: pip3 install httpx)")


# =============================================================================
# PERFORMANCE OPTIMIZATION: Model Warmup
# =============================================================================
def warmup_models():
    """
    Warm up all models by running a single inference pass.
    
    PERFORMANCE OPTIMIZATION:
    - First inference on YOLO models is slower due to lazy initialization
    - Running a warmup pass ensures subsequent inferences are fast
    - Adds ~2-5 seconds to startup but saves time on first video
    """
    import numpy as np
    
    print("\nWarming up models for faster first inference...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    try:
        # Warm up pose model
        print("  Warming up pose model...")
        _ = pose_model(dummy_frame, verbose=False)
        print("  ✓ Pose model ready")
    except Exception as e:
        print(f"  Warning: Pose model warmup failed: {e}")
    
    try:
        # Warm up multi-model detector
        if multi_detector.is_available:
            print("  Warming up multi-model detector...")
            for model_name, model in multi_detector.models.items():
                _ = model(dummy_frame, verbose=False)
                print(f"    ✓ {model_name} model ready")
    except Exception as e:
        print(f"  Warning: Multi-model warmup failed: {e}")
    
    print("Model warmup complete!\n")


# Run warmup on startup
warmup_models()


# Store active WebSocket connections for progress updates
active_connections: dict[str, WebSocket] = {}

# Store analysis results
analysis_results: dict[str, dict] = {}


class AnalysisConfig(BaseModel):
    """Configuration for video analysis"""
    fps_sample_rate: int = 1  # Process every Nth frame
    confidence_threshold: float = 0.5
    track_shuttle: bool = True
    calculate_speeds: bool = True
    detect_court: bool = False  # DISABLED: Using manual keypoints only for court calibration
    court_detection_interval: int = 30  # Frames between court detection updates (not used when detect_court=False)
    use_badminton_detector: bool = True  # Use custom Roboflow model for player/shuttlecock detection
    
    # Court detection model selection - DEPRECATED (manual keypoints are now preferred)
    court_model: str = "none"  # "none" - automatic court detection disabled, use manual keypoints
    
    # Modal.com GPU inference options (NEW - ~10-50x faster processing)
    use_modal_inference: bool = True  # Use Modal GPU when available (auto-fallback to local)
    
    # Detection smoothing options (NEW - reduces lag during rapid movements)
    enable_smoothing: bool = True  # Enable Kalman filter smoothing for detections
    enable_interpolation: bool = True  # Enable interpolation between keyframes
    preprocess_video: bool = False  # Apply video preprocessing (deblur, enhance)
    player_smoothing_strength: float = 0.5  # Smoothing for player tracking (0=none, 1=max)
    shuttle_smoothing_strength: float = 0.3  # Smoothing for shuttle tracking (lower = more responsive)
    
    # Performance optimization options (NEW - reduces processing lag)
    use_async_frame_reading: bool = True  # Enable async frame prefetching
    async_buffer_size: int = 10  # Number of frames to prefetch
    use_adaptive_frame_skipping: bool = True  # Skip frames based on motion detection
    adaptive_skip_motion_threshold: float = 5.0  # Motion threshold for frame skipping
    enable_performance_profiling: bool = False  # Log timing for each processing stage
    
    # Delayed speed calculation option (NEW - improves accuracy with manual court keypoints)
    # When True, speed calculations are skipped during initial video processing.
    # Speeds are then calculated post-hoc via /api/speed/{video_id} endpoint
    # which uses manual court keypoints if set for accurate spatial calibration.
    delay_speed_calculation: bool = True  # Skip speed calculation during upload


class PlayerMetrics(BaseModel):
    """Metrics for a single player"""
    player_id: int
    total_distance: float  # meters
    avg_speed: float  # km/h
    max_speed: float  # km/h
    positions: list[dict]  # List of {frame, x, y}
    keypoints_history: list[dict]  # Skeleton keypoints per frame


class ShuttleMetrics(BaseModel):
    """Metrics for shuttle/shuttlecock"""
    avg_speed: float  # km/h
    max_speed: float  # km/h
    shots_detected: int
    shot_speeds: list[float]


class CourtDetectionResult(BaseModel):
    """Court detection results"""
    detected: bool
    confidence: float
    keypoints: list[dict]
    court_corners: Optional[list[list[float]]]
    court_dimensions: dict
    regions: Optional[list[dict]] = None  # Court region detections
    # For keypoint model (22 keypoints) - added for dual model support
    court_keypoints_22: Optional[dict] = None  # {keypoints: [[x,y,conf]...], bbox, confidence}
    court_model_used: str = "region"  # "region" or "keypoint"


class ShuttleAnalyticsResult(BaseModel):
    """Enhanced shuttle analytics from court keypoint tracking"""
    total_shots: int
    shot_types: dict  # {smash: count, clear: count, ...}
    speed_stats: dict  # {fastest_shot_kmh, avg_shot_speed_kmh, all_shot_speeds}
    trajectories: list[dict]  # Detailed trajectory data


class PlayerZoneAnalytics(BaseModel):
    """Player court zone coverage analytics"""
    player_id: int
    zone_coverage: dict  # {front, mid, back, left, center, right} percentages
    avg_distance_to_net_m: float
    heatmap: list[list[float]]  # 2D grid of position frequency
    position_count: int


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    video_id: str
    duration: float  # seconds
    fps: float
    total_frames: int
    processed_frames: int
    video_width: int  # Original video width in pixels (for coordinate mapping)
    video_height: int  # Original video height in pixels (for coordinate mapping)
    players: list[PlayerMetrics]
    shuttle: Optional[ShuttleMetrics]
    skeleton_data: list[dict]  # Per-frame skeleton data for visualization
    court_detection: Optional[CourtDetectionResult]  # Court keypoint detection results
    shuttle_analytics: Optional[dict] = None  # Enhanced shuttle tracking analytics
    player_zone_analytics: Optional[dict] = None  # Player court zone coverage


def calculate_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def pixel_to_meters(pixel_distance: float, court_width_pixels: float, court_width_meters: float = 6.1) -> float:
    """
    Convert pixel distance to meters using court dimensions
    Standard badminton court width is 6.1m (doubles) or 5.18m (singles)
    """
    if court_width_pixels == 0:
        return 0
    return (pixel_distance / court_width_pixels) * court_width_meters


# =============================================================================
# SPEED CALCULATION WITH PHYSIOLOGICAL LIMITS
# =============================================================================
# Maximum realistic human speed limits to filter out tracking errors
# NOTE: These values are aligned with speed_analytics.py to ensure consistency
MAX_HUMAN_SPEED_KMH = 54.0  # ~15 m/s - Usain Bolt peak with safety margin
MAX_BADMINTON_SPEED_KMH = 43.0  # ~12 m/s - Elite badminton player max
MAX_POSITION_JUMP_M = 1.0  # Max realistic position change per frame

# Logging configuration for speed calculations
import logging
speed_logger = logging.getLogger("speed_calculation")

# Track clamping statistics for debugging
_clamp_stats = {
    "total_calculations": 0,
    "clamped_count": 0,
    "max_raw_speed_seen": 0.0,
    "recent_clamps": []  # Store recent clamp events for debugging
}
MAX_RECENT_CLAMPS = 50  # How many recent clamp events to keep


def calculate_speed(
    distance_meters: float,
    time_seconds: float,
    clamp: bool = True,
    player_id: int = -1,
    frame_number: int = -1,
    log_clamp: bool = True
) -> float:
    """
    Calculate speed in km/h with optional physiological clamping.
    
    Args:
        distance_meters: Distance traveled in meters
        time_seconds: Time elapsed in seconds
        clamp: If True, clamp to MAX_BADMINTON_SPEED_KMH (default: True)
        player_id: Player ID for logging (optional)
        frame_number: Frame number for logging (optional)
        log_clamp: If True, log when speed is clamped (default: True)
    
    Returns:
        Speed in km/h, clamped to realistic limits if clamp=True
    """
    global _clamp_stats
    
    if time_seconds == 0:
        return 0.0
    
    raw_speed_kmh = (distance_meters / time_seconds) * 3.6  # Convert m/s to km/h
    _clamp_stats["total_calculations"] += 1
    
    # Track max raw speed seen
    if raw_speed_kmh > _clamp_stats["max_raw_speed_seen"]:
        _clamp_stats["max_raw_speed_seen"] = raw_speed_kmh
    
    if clamp:
        # Check if clamping is needed
        if raw_speed_kmh > MAX_BADMINTON_SPEED_KMH:
            _clamp_stats["clamped_count"] += 1
            
            # Log the clamp event
            clamp_event = {
                "player_id": player_id,
                "frame": frame_number,
                "raw_speed": round(raw_speed_kmh, 1),
                "clamped_to": MAX_BADMINTON_SPEED_KMH,
                "distance_m": round(distance_meters, 3),
                "time_s": round(time_seconds, 4)
            }
            _clamp_stats["recent_clamps"].append(clamp_event)
            
            # Keep only recent clamps
            if len(_clamp_stats["recent_clamps"]) > MAX_RECENT_CLAMPS:
                _clamp_stats["recent_clamps"] = _clamp_stats["recent_clamps"][-MAX_RECENT_CLAMPS:]
            
            if log_clamp:
                speed_logger.warning(
                    f"Speed clamped: Player {player_id} frame {frame_number} - "
                    f"raw {raw_speed_kmh:.1f} km/h -> {MAX_BADMINTON_SPEED_KMH} km/h "
                    f"(distance: {distance_meters:.3f}m in {time_seconds:.4f}s)"
                )
        
        speed_kmh = min(raw_speed_kmh, MAX_BADMINTON_SPEED_KMH)
        speed_kmh = max(0.0, speed_kmh)
    else:
        speed_kmh = raw_speed_kmh
    
    return speed_kmh


def get_speed_clamp_statistics() -> dict:
    """
    Get statistics about speed clamping for debugging.
    
    Returns:
        Dictionary with clamping statistics:
        - total_calculations: Total number of speed calculations
        - clamped_count: Number of times speed was clamped
        - clamp_rate: Percentage of calculations that were clamped
        - max_raw_speed_seen: Maximum raw speed before clamping
        - recent_clamps: List of recent clamp events
    """
    total = _clamp_stats["total_calculations"]
    clamped = _clamp_stats["clamped_count"]
    clamp_rate = (clamped / total * 100) if total > 0 else 0.0
    
    return {
        "total_calculations": total,
        "clamped_count": clamped,
        "clamp_rate_pct": round(clamp_rate, 2),
        "max_raw_speed_seen_kmh": round(_clamp_stats["max_raw_speed_seen"], 1),
        "max_allowed_kmh": MAX_BADMINTON_SPEED_KMH,
        "recent_clamps": _clamp_stats["recent_clamps"][-10:],  # Return last 10
        "interpretation": (
            "High clamp rates (>10%) suggest homography miscalibration or tracking errors. "
            "Check manual keypoint placement and ensure they define the correct court area."
            if clamp_rate > 10 else
            "Clamp rate is within normal range."
        )
    }


def reset_speed_clamp_statistics():
    """Reset the speed clamp statistics (e.g., when starting new video analysis)."""
    global _clamp_stats
    _clamp_stats = {
        "total_calculations": 0,
        "clamped_count": 0,
        "max_raw_speed_seen": 0.0,
        "recent_clamps": []
    }


def is_valid_position_jump(distance_meters: float, time_seconds: float, fps: float = 30.0) -> bool:
    """
    Check if a position jump is realistic (not a tracking error).
    
    Position jumps > MAX_POSITION_JUMP_M per frame indicate:
    - Track ID switch (player 1 detected as player 2)
    - Occlusion recovery (player reappears at different location)
    - Detection failure artifacts
    
    Args:
        distance_meters: Distance between consecutive positions
        time_seconds: Time elapsed between measurements
        fps: Video frame rate for scaling
    
    Returns:
        True if the position jump is realistic, False if likely an error
    """
    if time_seconds <= 0:
        return True
    
    # Calculate max realistic jump for this time delta
    # At 30fps, max 1.0m/frame. Scale for actual frame time.
    frames_elapsed = time_seconds * fps
    max_jump = MAX_POSITION_JUMP_M * max(1.0, frames_elapsed)
    
    return distance_meters <= max_jump


# COCO pose keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


def is_skeleton_inside_player_bbox(
    skeleton_center: tuple,
    player_bboxes: list,
    iou_threshold: float = 0.0,
    margin_pixels: float = 50.0
) -> bool:
    """
    Check if a skeleton's center point is inside any of the player bounding boxes.
    
    This is used to filter out skeletons for spectators, referees, or other people
    who are not the main 2 badminton players. Only skeletons whose centers fall
    within the detected player bounding boxes will be rendered.
    
    Args:
        skeleton_center: (x, y) center of the detected skeleton
        player_bboxes: List of (x, y, width, height) bounding boxes for detected players
                       where (x, y) is the center of the bbox
        iou_threshold: Not used currently (for future IoU-based filtering)
        margin_pixels: Extra margin around bbox to allow some tolerance
        
    Returns:
        True if skeleton center is inside any player bbox, False otherwise
    """
    if not player_bboxes:
        # No player detections - keep all skeletons (fallback behavior)
        return True
    
    sx, sy = skeleton_center
    
    for bbox in player_bboxes:
        bx, by, bw, bh = bbox
        # Convert center-based bbox to corner-based with margin
        x_min = bx - (bw / 2) - margin_pixels
        x_max = bx + (bw / 2) + margin_pixels
        y_min = by - (bh / 2) - margin_pixels
        y_max = by + (bh / 2) + margin_pixels
        
        # Check if skeleton center is inside this bbox
        if x_min <= sx <= x_max and y_min <= sy <= y_max:
            return True
    
    return False


# =============================================================================
# WEBSOCKET LOG HELPER
# =============================================================================

async def send_log_message(video_id: str, message: str, level: str = "info", category: str = "processing"):
    """
    Send a log message to the frontend via WebSocket.
    
    Args:
        video_id: The video ID for the WebSocket connection
        message: The log message text
        level: Log level - "info", "success", "warning", "error", "debug"
        category: Log category - "processing", "detection", "model", "court", "modal"
    """
    if video_id in active_connections:
        try:
            await active_connections[video_id].send_json({
                "type": "log",
                "message": message,
                "level": level,
                "category": category,
                "timestamp": time.time()
            })
        except Exception:
            pass  # Connection might be closed


async def process_video(video_path: Path, video_id: str, config: AnalysisConfig):
    """
    Process video with YOLOv8 pose estimation and court keypoint detection
    Sends progress updates via WebSocket
    
    Performance Optimizations Applied:
    - Async frame reading with prefetching buffer (reduces I/O wait)
    - Adaptive frame skipping based on motion (reduces CPU usage on static scenes)
    - Performance profiling for timing analysis
    - Modal GPU inference for ~10-50x faster processing (when enabled)
    """
    global modal_client, modal_available
    
    print(f"\n{'='*60}")
    print(f"STARTING VIDEO PROCESSING: {video_id}")
    print(f"Video path: {video_path}")
    print(f"Court calibration: Manual keypoints only (auto-detection disabled)")
    print(f"Modal inference requested: {config.use_modal_inference}")
    print(f"{'='*60}\n")
    
    # Send initial log to frontend
    await send_log_message(video_id, "Starting video analysis...", "info", "processing")
    await send_log_message(video_id, f"Loading video: {video_path.name}", "info", "processing")
    
    # Log delayed speed calculation status
    if config.delay_speed_calculation:
        await send_log_message(
            video_id,
            "Speed calculations delayed until court keypoints are confirmed and playback starts",
            "info",
            "processing"
        )
        print(f"[VIDEO PROCESSING] Speed calculations delayed for video {video_id}")
        print(f"  - Set court keypoints and start playback to trigger speed calculation")
    
    # Reset speed clamp statistics for fresh analysis
    reset_speed_clamp_statistics()
    speed_logger.info(f"Video analysis started for {video_id}, speed clamp statistics reset")
    
    # Initialize Modal inference if requested and available
    use_modal = False
    if config.use_modal_inference and modal_available and modal_client:
        print("🚀 Checking Modal GPU inference availability...")
        await send_log_message(video_id, "Checking Modal GPU availability...", "info", "modal")
        try:
            health = await modal_client.health_check()
            if health.get("status") == "healthy":
                use_modal = True
                gpu_name = health.get('gpu_name', 'Unknown GPU')
                print(f"  ✓ Modal GPU ready: {gpu_name}")
                print(f"  ✓ Pose model: {'loaded' if health.get('pose_model_loaded') else 'not loaded'}")
                print(f"  ✓ Detection model: {'loaded' if health.get('detection_model_loaded') else 'not loaded'}")
                await send_log_message(video_id, f"Modal GPU ready: {gpu_name}", "success", "modal")
                await send_log_message(video_id, "Using GPU acceleration for fast inference", "success", "modal")
            else:
                print(f"  ✗ Modal not healthy: {health.get('reason', 'Unknown')}")
                print("  → Falling back to local CPU inference")
                await send_log_message(video_id, f"Modal unavailable: {health.get('reason', 'Unknown')}", "warning", "modal")
                await send_log_message(video_id, "Falling back to local CPU inference", "warning", "modal")
        except Exception as e:
            print(f"  ✗ Modal health check failed: {e}")
            print("  → Falling back to local CPU inference")
            await send_log_message(video_id, f"Modal connection failed: {str(e)[:50]}", "warning", "modal")
            await send_log_message(video_id, "Using local CPU inference", "info", "modal")
    
    if use_modal:
        print(f"\n📡 INFERENCE MODE: Modal GPU (T4)")
        await send_log_message(video_id, "Inference mode: Modal GPU", "success", "model")
    else:
        print(f"\n💻 INFERENCE MODE: Local CPU")
        await send_log_message(video_id, "Inference mode: Local CPU", "info", "model")
    # Initialize performance profiler for timing analysis
    profiler = None
    if config.enable_performance_profiling:
        profiler = PerformanceProfiler()
    
    # Initialize async frame buffer for background frame reading
    frame_buffer = None
    cap = None
    
    await send_log_message(video_id, "Initializing video reader...", "info", "processing")
    
    if config.use_async_frame_reading:
        frame_buffer = AsyncFrameBuffer(
            str(video_path),
            buffer_size=config.async_buffer_size,
            skip_rate=1  # We'll handle skipping separately
        )
        frame_buffer.start()
        
        # Wait for video properties to be populated
        await asyncio.sleep(0.1)
        
        fps = frame_buffer.fps
        total_frames = frame_buffer.total_frames
        width = frame_buffer.width
        height = frame_buffer.height
        duration = total_frames / fps if fps > 0 else 0
    else:
        # Fall back to synchronous reading
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            await send_log_message(video_id, "Failed to open video file", "error", "processing")
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
    
    # Log video properties
    await send_log_message(video_id, f"Video loaded: {width}x{height} @ {fps:.1f}fps", "success", "processing")
    await send_log_message(video_id, f"Total frames: {total_frames} ({duration:.1f}s duration)", "info", "processing")
    
    # Initialize adaptive frame skipper for motion-based skipping
    adaptive_skipper = None
    if config.use_adaptive_frame_skipping:
        adaptive_skipper = AdaptiveFrameSkipper(
            base_skip_rate=config.fps_sample_rate,
            motion_threshold=config.adaptive_skip_motion_threshold,
            min_skip=1,
            max_skip=5
        )
    
    # Initialize court detection
    court_detection_result = None
    court_width_pixels = width * 0.8  # Default fallback
    use_homography = False  # Whether we have valid court detection for accurate measurements
    
    # Track Modal court keypoints (for keypoint model visualization)
    modal_court_keypoints = None  # Stores {keypoints: [[x,y,conf]...], bbox, confidence}
    modal_court_model_used = "region"  # Which court model was used
    
    # CRITICAL: If manual keypoints are set, create court_detection_result from them NOW
    # This ensures ALL speed calculations during processing use the manual calibration
    manual_kp_status = get_manual_keypoints_status()
    if manual_kp_status.get("has_manual_keypoints", False):
        manual_kp = court_detector.get_manual_keypoints()
        if manual_kp is not None:
            # Create CourtDetection from manual keypoints
            corners = manual_kp.to_numpy()
            homography = court_detector._calculate_homography(corners)
            
            if homography is not None:
                court_detection_result = CourtDetection(
                    regions=[],
                    keypoints=[],
                    court_corners=corners,
                    homography_matrix=homography,
                    confidence=1.0,
                    detected=True,
                    frame_width=width,
                    frame_height=height,
                    detection_source="manual",
                    manual_keypoints=manual_kp
                )
                use_homography = True
                court_width_pixels = court_detector.get_court_width_pixels(court_detection_result) or court_width_pixels
                
                print(f"[VIDEO PROCESSING] ✓ Using MANUAL court keypoints for speed calculation")
                print(f"  - Corners: {corners.tolist()}")
                print(f"  - Court width (px): {court_width_pixels:.1f}")
                await send_log_message(video_id, "Using manual court keypoints for accurate speed calculation", "success", "court")
    
    # Initialize tracking data
    player_tracks: dict[int, dict] = {}
    skeleton_frames: list[dict] = []
    shuttle_positions: list[tuple] = []
    
    # Initialize enhanced analytics
    shuttle_tracker = ShuttleTracker(court_detector, fps=fps)
    position_analyzer = PlayerPositionAnalyzer(court_detector)
    
    # Initialize detection smoother for lag reduction (NEW)
    detection_smoother = None
    video_preprocessor = None
    if config.enable_smoothing:
        await send_log_message(video_id, "Initializing detection smoothing...", "info", "model")
        # Convert smoothing strength (0-1) to noise parameters
        # Lower noise = more smoothing, Higher noise = more responsive
        player_process_noise = 0.1 + (1 - config.player_smoothing_strength) * 1.5
        player_measurement_noise = 0.5 + config.player_smoothing_strength * 1.5
        shuttle_process_noise = 0.5 + (1 - config.shuttle_smoothing_strength) * 3.0
        shuttle_measurement_noise = 0.2 + config.shuttle_smoothing_strength * 1.0
        
        detection_smoother = DetectionSmoother(
            fps=fps,
            player_process_noise=player_process_noise,
            player_measurement_noise=player_measurement_noise,
            shuttle_process_noise=shuttle_process_noise,
            shuttle_measurement_noise=shuttle_measurement_noise,
            max_prediction_frames=5  # Predict up to 5 frames ahead for fast objects
        )
        print(f"Detection smoother initialized (player_noise={player_process_noise:.2f}, shuttle_noise={shuttle_process_noise:.2f})")
        await send_log_message(video_id, "Detection smoothing enabled", "success", "model")
    
    # Initialize video preprocessor if enabled
    if config.preprocess_video:
        video_preprocessor = VideoPreprocessor(target_fps=fps)
        print("Video preprocessing enabled (deblur + enhancement)")
        await send_log_message(video_id, "Video preprocessing enabled (deblur + enhance)", "info", "model")
    
    # Log model status
    # NOTE: Court detection model logging removed - we now only use manual keypoints for court calibration
    if config.use_badminton_detector and multi_detector.is_available:
        await send_log_message(video_id, f"Object detection ready ({len(multi_detector.available_models)} models)", "success", "model")
    
    await send_log_message(video_id, "Starting frame-by-frame processing...", "info", "processing")
    
    # Output video with skeleton overlay (temporary file with mp4v codec)
    temp_output_path = PROCESSED_DIR / f"{video_id}_temp.mp4"
    output_path = PROCESSED_DIR / f"{video_id}_analyzed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    # Frame iterator - either from async buffer or synchronous cap.read()
    def sync_frame_iterator():
        """Synchronous frame iterator using cv2.VideoCapture"""
        nonlocal cap
        assert cap is not None, "VideoCapture not initialized"
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            yield frame, frame_num
    
    # Select frame source
    frame_source = frame_buffer if frame_buffer else sync_frame_iterator()
    
    try:
        for frame, frame_count in frame_source:
            # Profiling: measure frame read time (only for sync mode)
            if profiler and not frame_buffer:
                # Note: For async mode, I/O time is hidden in background thread
                pass
            
            # Use adaptive frame skipping if enabled
            if adaptive_skipper:
                if adaptive_skipper.should_skip(frame, frame_count):
                    out.write(frame)
                    continue
            elif frame_count % config.fps_sample_rate != 0:
                # Fall back to fixed skip rate
                out.write(frame)
                continue
            
            processed_count += 1
            
            # Start profiling this frame's processing
            if profiler:
                profiler_ctx = profiler.measure("frame_processing")
                profiler_ctx.__enter__()
            
            # Apply video preprocessing if enabled (deblur + enhancement for fast movements)
            if video_preprocessor is not None:
                frame = video_preprocessor.preprocess_frame(
                    frame,
                    apply_deblur=True,
                    apply_enhancement=True
                )
            
            # Run court detection periodically
            # PERFORMANCE OPTIMIZATION: Skip automatic detection if manual keypoints are set
            manual_kp_status = get_manual_keypoints_status()
            should_run_court_detection = (
                config.detect_court and
                court_detector.model and
                not manual_kp_status.get("has_manual_keypoints", False)
            )
            
            if should_run_court_detection:
                if frame_count == 1 or frame_count % config.court_detection_interval == 0:
                    print(f"\n[VIDEO PROCESSING] Running court detection on frame {frame_count}")
                    
                    court_detection = court_detector.detect_court(
                        frame,
                        frame_number=frame_count,
                        use_cache=True,
                        cache_interval=config.court_detection_interval
                    )
                    
                    if court_detection.detected:
                        court_detection_result = court_detection
                        # Update court width from detection
                        detected_width = court_detector.get_court_width_pixels(court_detection)
                        if detected_width:
                            court_width_pixels = detected_width
                            use_homography = court_detection.homography_matrix is not None
                        
                        print(f"[VIDEO PROCESSING] ✓ Court detected on frame {frame_count}:")
                        print(f"  - Regions: {len(court_detection.regions)}")
                        print(f"  - Keypoints: {len(court_detection.keypoints)}")
                        print(f"  - Corners: {court_detection.court_corners is not None}")
                        print(f"  - Homography: {court_detection.homography_matrix is not None}")
                        print(f"  - Court width (px): {detected_width:.1f}" if detected_width else "  - Court width: N/A")
                        print(f"  - Confidence: {court_detection.confidence:.3f}")
                        
                        if court_detection.court_corners is not None:
                            print(f"  - Court corners:")
                            corner_names = ["TL", "TR", "BR", "BL"]
                            for name, corner in zip(corner_names, court_detection.court_corners):
                                print(f"      {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
                    else:
                        print(f"[VIDEO PROCESSING] ✗ Court NOT detected on frame {frame_count}")
                        print(f"  - Regions found: {len(court_detection.regions)}")
            
            # Run multi-model detection for bounding boxes using YOLOv26 recommended plot() method
            badminton_detections = None
            shuttle_position_this_frame = None
            # Store player bounding boxes for filtering skeletons to only show main 2 players
            player_bboxes_for_skeleton_filter = []
            
            if config.use_badminton_detector and multi_detector.is_available:
                # Use the multi-model detect_and_annotate method
                # This uses results[0].plot() for optimized bounding box visualization
                badminton_detections, frame = multi_detector.detect_and_annotate(
                    frame,
                    frame_number=frame_count
                )
                
                # Extract player bounding boxes for skeleton filtering
                # This ensures only skeletons for the 2 main players (inside bboxes) are rendered
                # All other detected persons (spectators, refs, etc.) will NOT have skeletons
                if badminton_detections and badminton_detections.players:
                    # Keep only top 2 players by bbox area (main players are closer to camera = larger)
                    sorted_players = sorted(
                        badminton_detections.players,
                        key=lambda p: p.bbox.width * p.bbox.height,
                        reverse=True
                    )[:2]
                    player_bboxes_for_skeleton_filter = [
                        (p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height)
                        for p in sorted_players
                    ]
                
                # Track shuttlecock for enhanced analytics
                if badminton_detections and badminton_detections.shuttlecocks:
                    best_shuttle = max(badminton_detections.shuttlecocks, key=lambda s: s.confidence)
                    shuttle_position_this_frame = (best_shuttle.bbox.x, best_shuttle.bbox.y)
                    shuttle_tracker.update(
                        frame_number=frame_count,
                        shuttle_position=shuttle_position_this_frame,
                        confidence=best_shuttle.confidence,
                        court_detection=court_detection_result
                    )
            
            # Initialize results variable for local inference fallback
            results = None
            
            # Run pose estimation (Modal GPU or Local CPU)
            frame_skeleton_data = {
                "frame": frame_count,
                "timestamp": frame_count / fps,
                "players": [],
                "court_detected": court_detection_result is not None and court_detection_result.detected,
                "badminton_detections": badminton_detections.to_dict() if badminton_detections else None,
                "shuttle_position": shuttle_position_this_frame,
                "shuttle_speed_kmh": None,  # Will be filled if we have trajectory
                "inference_source": "modal" if use_modal else "local"
            }
            
            # Choose inference mode: Modal GPU or Local CPU
            if use_modal and modal_client:
                # ===== MODAL GPU INFERENCE =====
                # NOTE: Court detection has been removed from Modal - use manual keypoints instead
                # Manual keypoints are more accurate for speed/distance calculations
                try:
                    modal_result = await modal_client.process_frame(
                        frame=frame,
                        frame_number=frame_count,
                        run_pose=True,
                        run_detection=config.use_badminton_detector,
                        run_court=False,  # Court detection removed from Modal - always False
                        run_pose_classification=True,  # Run trained pose classification model
                        confidence_threshold=config.confidence_threshold,
                        court_model="none"  # Court detection removed from Modal
                    )
                    
                    # Convert Modal result to local format
                    results = None  # Not using local results
                    
                    # BUGFIX: Calculate inverse scale factor to map coordinates back to original frame size
                    # Modal client resizes frames to max 1280px for faster inference, so coordinates
                    # need to be scaled back up to the original video dimensions
                    inv_scale = 1.0 / modal_result.scale_factor if modal_result.scale_factor != 1.0 else 1.0
                    
                    # Process Modal pose results
                    for pose in modal_result.poses:
                        track_id = pose.get("track_id", 0)
                        keypoints_list = []
                        valid_keypoints = []
                        
                        for kp in pose.get("keypoints", []):
                            kp_name = kp.get("name")
                            kp_x_raw = kp.get("x")
                            kp_y_raw = kp.get("y")
                            kp_conf = kp.get("confidence", 0.0)
                            
                            # Apply inverse scale to map coordinates back to original frame size
                            kp_x = kp_x_raw * inv_scale if kp_x_raw is not None else None
                            kp_y = kp_y_raw * inv_scale if kp_y_raw is not None else None
                            
                            if kp_conf > config.confidence_threshold and kp_x is not None:
                                keypoints_list.append({
                                    "name": kp_name,
                                    "x": float(kp_x),
                                    "y": float(kp_y),
                                    "confidence": float(kp_conf)
                                })
                                valid_keypoints.append((kp_x, kp_y))
                                # Draw keypoint on frame (use original frame coordinates)
                                cv2.circle(frame, (int(kp_x), int(kp_y)), 5, (0, 255, 0), -1)
                            else:
                                keypoints_list.append({
                                    "name": kp_name,
                                    "x": None,
                                    "y": None,
                                    "confidence": float(kp_conf) if kp_conf else 0.0
                                })
                        
                        # Calculate center position (also apply scale)
                        center_x_raw = pose.get("center_x", 0)
                        center_y_raw = pose.get("center_y", 0)
                        center_x = center_x_raw * inv_scale
                        center_y = center_y_raw * inv_scale
                        
                        if not valid_keypoints:
                            continue
                        
                        # Filter skeleton: only keep if center is inside a detected player bbox
                        # This filters out spectators, referees, and other detected persons
                        if not is_skeleton_inside_player_bbox((center_x, center_y), player_bboxes_for_skeleton_filter):
                            continue  # Skip this skeleton - not a main player
                        
                        # Draw skeleton connections for Modal inference
                        for conn in SKELETON_CONNECTIONS:
                            kp1_idx, kp2_idx = conn
                            if kp1_idx < len(keypoints_list) and kp2_idx < len(keypoints_list):
                                kp1 = keypoints_list[kp1_idx]
                                kp2 = keypoints_list[kp2_idx]
                                if kp1["x"] is not None and kp2["x"] is not None:
                                    cv2.line(frame,
                                             (int(kp1["x"]), int(kp1["y"])),
                                             (int(kp2["x"]), int(kp2["y"])),
                                             (0, 255, 255), 2)
                        
                        # Initialize or update player tracking
                        if track_id not in player_tracks:
                            player_tracks[track_id] = {
                                "positions": deque(maxlen=100),
                                "speeds": [],
                                "keypoints_history": [],
                                "total_distance_pixels": 0
                            }
                        
                        track = player_tracks[track_id]
                        current_pos = (center_x, center_y)
                        
                        # Calculate speed if we have previous position
                        # DELAYED SPEED CALCULATION: Skip during initial upload if delay_speed_calculation is True
                        # Speed will be calculated post-hoc via /api/speed/{video_id} when:
                        # 1. Court keypoints have been confirmed
                        # 2. Video playback has been initiated
                        if len(track["positions"]) > 0 and not config.delay_speed_calculation:
                            prev_pos = track["positions"][-1]["pos"]
                            distance_pixels = calculate_distance(prev_pos, current_pos)
                            
                            if use_homography and court_detection_result:
                                distance_meters = court_detector.calculate_real_distance(
                                    prev_pos, current_pos, court_detection_result
                                )
                                if distance_meters is None:
                                    distance_meters = pixel_to_meters(distance_pixels, court_width_pixels)
                            else:
                                distance_meters = pixel_to_meters(distance_pixels, court_width_pixels)
                            
                            time_diff = config.fps_sample_rate / fps
                            speed = calculate_speed(
                                distance_meters,
                                time_diff,
                                player_id=int(track_id),
                                frame_number=frame_count
                            )
                            track["speeds"].append(speed)
                            track["total_distance_pixels"] += distance_pixels
                            
                            if "total_distance_meters" not in track:
                                track["total_distance_meters"] = 0
                            track["total_distance_meters"] += distance_meters
                        
                        track["positions"].append({
                            "frame": frame_count,
                            "pos": current_pos
                        })
                        track["keypoints_history"].append({
                            "frame": frame_count,
                            "keypoints": keypoints_list
                        })
                        
                        # Update player position analyzer
                        player_court_pos = position_analyzer.update(
                            frame_number=frame_count,
                            player_id=int(track_id),
                            pixel_position=(center_x, center_y),
                            court_detection=court_detection_result
                        )
                        
                        # Classify pose from keypoints (for Modal inference)
                        # Build keypoints dict in the format expected by PoseAnalyzer
                        pose_keypoints_dict = {}
                        for kp in keypoints_list:
                            if kp["x"] is not None and kp["y"] is not None:
                                try:
                                    kp_enum = PoseKeypoint[kp["name"].upper()]
                                    pose_keypoints_dict[kp_enum] = KeypointData(
                                        x=kp["x"],
                                        y=kp["y"],
                                        confidence=kp["confidence"],
                                        visible=kp["confidence"] > config.confidence_threshold
                                    )
                                except (KeyError, ValueError):
                                    pass
                        
                        # Create PlayerPose for classification
                        pose_data = PosePlayerPose(
                            keypoints=pose_keypoints_dict,
                            bbox_x=float(center_x),
                            bbox_y=float(center_y),
                            bbox_width=100.0,  # Approximate
                            bbox_height=200.0,  # Approximate
                            confidence=0.8,
                            player_id=int(track_id)
                        )
                        
                        # Calculate body angles and classify pose
                        pose_data.body_angles = pose_analyzer.calculate_body_angles(pose_data)
                        pose_type = pose_analyzer.classify_pose(pose_data)
                        pose_confidence = 0.8  # Approximate confidence
                        
                        # Add to frame skeleton data
                        frame_skeleton_data["players"].append({
                            "player_id": int(track_id),
                            "keypoints": keypoints_list,
                            "center": {"x": float(center_x), "y": float(center_y)},
                            "current_speed": float(track["speeds"][-1]) if track["speeds"] else 0.0,
                            "court_position": {
                                "x": player_court_pos.court_x,
                                "y": player_court_pos.court_y,
                                "zone": player_court_pos.zone,
                                "distance_to_net": player_court_pos.distance_to_net
                            } if player_court_pos.court_x is not None else None,
                            "pose": {
                                "pose_type": pose_type.value,
                                "confidence": float(pose_confidence)
                            }
                        })
                        
                        # Draw player info on frame
                        speed_text = f"P{track_id}: {track['speeds'][-1]:.1f} km/h" if track["speeds"] else f"P{track_id}"
                        cv2.putText(frame, speed_text, (int(center_x) - 30, int(center_y) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Use Modal inference time
                    frame_skeleton_data["inference_time_ms"] = modal_result.inference_time_ms
                    
                    # Store Modal court keypoints (for keypoint model)
                    # BUGFIX: Apply inverse scale to court keypoints, same as pose keypoints
                    # Modal client resizes frames to max 1280px, so keypoints need scaling back
                    if modal_result.court_keypoints:
                        modal_court_model_used = modal_result.court_model_used
                        
                        # Scale the court keypoints back to original frame size
                        raw_kps = modal_result.court_keypoints.get("keypoints", [])
                        if inv_scale != 1.0 and raw_kps:
                            scaled_kps = [
                                [kp[0] * inv_scale, kp[1] * inv_scale, kp[2] if len(kp) > 2 else 1.0]
                                for kp in raw_kps
                            ]
                            modal_court_keypoints = {
                                "keypoints": scaled_kps,
                                "bbox": modal_result.court_keypoints.get("bbox"),
                                "confidence": modal_result.court_keypoints.get("confidence", 0.0)
                            }
                            # Also scale the bbox if present
                            raw_bbox = modal_result.court_keypoints.get("bbox")
                            if raw_bbox and len(raw_bbox) >= 4:
                                modal_court_keypoints["bbox"] = [
                                    raw_bbox[0] * inv_scale,
                                    raw_bbox[1] * inv_scale,
                                    raw_bbox[2] * inv_scale,
                                    raw_bbox[3] * inv_scale
                                ]
                        else:
                            modal_court_keypoints = modal_result.court_keypoints
                        
                        # Also add to frame data for frontend access
                        frame_skeleton_data["court_keypoints"] = modal_court_keypoints
                        frame_skeleton_data["court_model_used"] = modal_result.court_model_used
                    
                    # Add trained pose classifications from Modal inference
                    if modal_result.pose_classifications:
                        # Scale pose classification bounding boxes to original frame size
                        scaled_pose_classifications = []
                        for pc in modal_result.pose_classifications:
                            scaled_pc = {
                                # Modal sends "pose_class", frontend expects "class_name"
                                "class_name": pc.get("pose_class"),
                                "confidence": pc.get("confidence"),
                                "bbox": {
                                    "x": pc.get("bbox", {}).get("x", 0) * inv_scale,
                                    "y": pc.get("bbox", {}).get("y", 0) * inv_scale,
                                    "width": pc.get("bbox", {}).get("width", 0) * inv_scale,
                                    "height": pc.get("bbox", {}).get("height", 0) * inv_scale,
                                } if pc.get("bbox") else None
                            }
                            scaled_pose_classifications.append(scaled_pc)
                        frame_skeleton_data["pose_classifications"] = scaled_pose_classifications
                    
                except Exception as e:
                    print(f"  ⚠️ Modal inference failed on frame {frame_count}: {e}")
                    # Fall back to local inference for this frame
                    results = pose_model.track(frame, persist=True, verbose=False)
            else:
                # ===== LOCAL CPU INFERENCE =====
                results = pose_model.track(frame, persist=True, verbose=False)
            
            # Add shuttle speed to frame data if available
            if shuttle_tracker.current_trajectory and len(shuttle_tracker.current_trajectory.positions) >= 2:
                positions = shuttle_tracker.current_trajectory.positions
                last_two = positions[-2:]
                if len(last_two) == 2:
                    _, speed_kmh = shuttle_tracker.calculate_speed_between_frames(
                        (last_two[0].pixel_x, last_two[0].pixel_y),
                        (last_two[1].pixel_x, last_two[1].pixel_y),
                        last_two[1].frame_number - last_two[0].frame_number,
                        court_detection_result
                    )
                    frame_skeleton_data["shuttle_speed_kmh"] = round(speed_kmh, 1)
            
            # Process detections (only for local inference - Modal already processed above)
            if results is not None and len(results) > 0:
                result = results[0]
                
                if result.keypoints is not None and len(result.keypoints) > 0:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    
                    # Get tracking IDs if available
                    track_ids = []
                    if result.boxes is not None and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = list(range(len(keypoints_data)))
                    
                    for idx, (kpts, track_id) in enumerate(zip(keypoints_data, track_ids)):
                        # Extract keypoint coordinates
                        keypoints_list = []
                        valid_keypoints = []
                        
                        for kp_idx, keypoint in enumerate(kpts):
                            x, y, conf = keypoint[0], keypoint[1], keypoint[2] if len(keypoint) > 2 else 1.0
                            
                            if conf > config.confidence_threshold:
                                keypoints_list.append({
                                    "name": KEYPOINT_NAMES[kp_idx],
                                    "x": float(x),
                                    "y": float(y),
                                    "confidence": float(conf)
                                })
                                valid_keypoints.append((x, y))
                                
                                # Draw keypoint on frame
                                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            else:
                                keypoints_list.append({
                                    "name": KEYPOINT_NAMES[kp_idx],
                                    "x": None,
                                    "y": None,
                                    "confidence": float(conf)
                                })
                        
                        # Draw skeleton connections
                        for conn in SKELETON_CONNECTIONS:
                            kp1, kp2 = conn
                            if kp1 < len(kpts) and kp2 < len(kpts):
                                x1, y1, c1 = kpts[kp1]
                                x2, y2, c2 = kpts[kp2]
                                if c1 > config.confidence_threshold and c2 > config.confidence_threshold:
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        
                        # Calculate center position (using ankle midpoint for accurate court positioning)
                        # Ankles represent where the player is actually standing on the court,
                        # which is more accurate than hip midpoint for mini court display
                        left_ankle_idx = KEYPOINT_NAMES.index("left_ankle")
                        right_ankle_idx = KEYPOINT_NAMES.index("right_ankle")
                        left_hip_idx = KEYPOINT_NAMES.index("left_hip")
                        right_hip_idx = KEYPOINT_NAMES.index("right_hip")
                        
                        if (kpts[left_ankle_idx][2] > config.confidence_threshold and
                            kpts[right_ankle_idx][2] > config.confidence_threshold):
                            # Primary: use ankle midpoint (most accurate for court position)
                            center_x = (kpts[left_ankle_idx][0] + kpts[right_ankle_idx][0]) / 2
                            center_y = (kpts[left_ankle_idx][1] + kpts[right_ankle_idx][1]) / 2
                        elif (kpts[left_hip_idx][2] > config.confidence_threshold and
                              kpts[right_hip_idx][2] > config.confidence_threshold):
                            # Fallback 1: use hip midpoint if ankles not visible
                            center_x = (kpts[left_hip_idx][0] + kpts[right_hip_idx][0]) / 2
                            center_y = (kpts[left_hip_idx][1] + kpts[right_hip_idx][1]) / 2
                        elif valid_keypoints:
                            # Fallback 2: mean of all valid keypoints
                            center_x = np.mean([kp[0] for kp in valid_keypoints])
                            center_y = np.mean([kp[1] for kp in valid_keypoints])
                        else:
                            continue
                        
                        # Filter skeleton: only keep if center is inside a detected player bbox
                        # This filters out spectators, referees, and other detected persons
                        if not is_skeleton_inside_player_bbox((center_x, center_y), player_bboxes_for_skeleton_filter):
                            continue  # Skip this skeleton - not a main player
                        
                        # Initialize or update player tracking
                        if track_id not in player_tracks:
                            player_tracks[track_id] = {
                                "positions": deque(maxlen=100),
                                "speeds": [],
                                "keypoints_history": [],
                                "total_distance_pixels": 0
                            }
                        
                        track = player_tracks[track_id]
                        current_pos = (center_x, center_y)
                        
                        # Calculate speed if we have previous position
                        # DELAYED SPEED CALCULATION: Skip during initial upload if delay_speed_calculation is True
                        # Speed will be calculated post-hoc via /api/speed/{video_id} when:
                        # 1. Court keypoints have been confirmed
                        # 2. Video playback has been initiated
                        if len(track["positions"]) > 0 and not config.delay_speed_calculation:
                            prev_pos = track["positions"][-1]["pos"]
                            distance_pixels = calculate_distance(prev_pos, current_pos)
                            
                            # Use accurate homography-based distance if court is detected
                            if use_homography and court_detection_result:
                                distance_meters = court_detector.calculate_real_distance(
                                    prev_pos, current_pos, court_detection_result
                                )
                                if distance_meters is None:
                                    distance_meters = pixel_to_meters(distance_pixels, court_width_pixels)
                            else:
                                distance_meters = pixel_to_meters(distance_pixels, court_width_pixels)
                            
                            time_diff = config.fps_sample_rate / fps
                            speed = calculate_speed(
                                distance_meters,
                                time_diff,
                                player_id=int(track_id),
                                frame_number=frame_count
                            )
                            track["speeds"].append(speed)
                            track["total_distance_pixels"] += distance_pixels
                            
                            # Store real distance for accurate totals
                            if "total_distance_meters" not in track:
                                track["total_distance_meters"] = 0
                            track["total_distance_meters"] += distance_meters
                        
                        track["positions"].append({
                            "frame": frame_count,
                            "pos": current_pos
                        })
                        track["keypoints_history"].append({
                            "frame": frame_count,
                            "keypoints": keypoints_list
                        })
                        
                        # Update player position analyzer for zone analytics
                        player_court_pos = position_analyzer.update(
                            frame_number=frame_count,
                            player_id=int(track_id),
                            pixel_position=(center_x, center_y),
                            court_detection=court_detection_result
                        )
                        
                        # Classify pose from keypoints
                        # Build keypoints dict in the format expected by PoseAnalyzer
                        pose_keypoints_dict = {}
                        for kp in keypoints_list:
                            if kp["x"] is not None and kp["y"] is not None:
                                try:
                                    kp_enum = PoseKeypoint[kp["name"].upper()]
                                    pose_keypoints_dict[kp_enum] = KeypointData(
                                        x=kp["x"],
                                        y=kp["y"],
                                        confidence=kp["confidence"],
                                        visible=kp["confidence"] > config.confidence_threshold
                                    )
                                except (KeyError, ValueError):
                                    pass
                        
                        # Create PlayerPose for classification
                        pose_data = PosePlayerPose(
                            keypoints=pose_keypoints_dict,
                            bbox_x=float(center_x),
                            bbox_y=float(center_y),
                            bbox_width=100.0,  # Approximate
                            bbox_height=200.0,  # Approximate
                            confidence=0.8,
                            player_id=int(track_id)
                        )
                        
                        # Calculate body angles and classify pose
                        pose_data.body_angles = pose_analyzer.calculate_body_angles(pose_data)
                        pose_type = pose_analyzer.classify_pose(pose_data)
                        pose_confidence = 0.8  # Approximate confidence
                        
                        # Add to frame skeleton data
                        frame_skeleton_data["players"].append({
                            "player_id": int(track_id),
                            "keypoints": keypoints_list,
                            "center": {"x": float(center_x), "y": float(center_y)},
                            "current_speed": float(track["speeds"][-1]) if track["speeds"] else 0.0,
                            "court_position": {
                                "x": player_court_pos.court_x,
                                "y": player_court_pos.court_y,
                                "zone": player_court_pos.zone,
                                "distance_to_net": player_court_pos.distance_to_net
                            } if player_court_pos.court_x is not None else None,
                            "pose": {
                                "pose_type": pose_type.value,
                                "confidence": float(pose_confidence)
                            }
                        })
                        
                        # Draw player info on frame
                        speed_text = f"P{track_id}: {track['speeds'][-1]:.1f} km/h" if track["speeds"] else f"P{track_id}"
                        cv2.putText(frame, speed_text, (int(center_x) - 30, int(center_y) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add smoothing metadata to frame
            frame_skeleton_data["smoothing_enabled"] = detection_smoother is not None
            frame_skeleton_data["interpolation_enabled"] = config.enable_interpolation
            
            skeleton_frames.append(frame_skeleton_data)
            
            # Draw court overlay if detected
            if court_detection_result and court_detection_result.detected:
                frame = court_detector.draw_court_overlay(
                    frame,
                    court_detection_result,
                    draw_keypoints=True,
                    draw_lines=True,
                    draw_corners=True
                )
            
            out.write(frame)
            
            # Send progress update via WebSocket
            if video_id in active_connections:
                progress = (frame_count / total_frames) * 100
                try:
                    await active_connections[video_id].send_json({
                        "type": "progress",
                        "progress": progress,
                        "frame": frame_count,
                        "total_frames": total_frames
                    })
                except Exception:
                    pass
            
            # Send periodic log messages for milestone updates
            if frame_count % 100 == 0 or frame_count == 1:
                # Log detection counts periodically
                player_count = len(frame_skeleton_data.get("players", []))
                shuttle_detected = frame_skeleton_data.get("shuttle_position") is not None
                
                if frame_count == 1:
                    await send_log_message(video_id, f"Processing first frame - {player_count} player(s) detected", "info", "detection")
                    if use_modal:
                        await send_log_message(video_id, "Running pose estimation on Modal GPU...", "info", "detection")
                    else:
                        await send_log_message(video_id, "Running pose estimation on local CPU...", "info", "detection")
                
                # Log milestone percentages
                progress_pct = int(progress)
                if progress_pct in [25, 50, 75] and frame_count > 1:
                    # Only log each milestone once (check if we just crossed it)
                    prev_progress = ((frame_count - 1) / total_frames) * 100
                    if int(prev_progress) < progress_pct:
                        await send_log_message(video_id, f"Processing {progress_pct}% complete ({frame_count}/{total_frames} frames)", "info", "processing")
                        if player_count > 0:
                            await send_log_message(video_id, f"Tracking {player_count} player(s)", "info", "detection")
            
            # Yield control to allow other async operations
            if frame_count % 10 == 0:
                await asyncio.sleep(0)
    
    finally:
        # Clean up resources based on which mode was used
        if frame_buffer:
            frame_buffer.stop()
        if cap:
            cap.release()
        out.release()
        
        # Print performance profiling report if enabled
        if profiler:
            print("\n=== PERFORMANCE PROFILING REPORT ===")
            profiler.print_report()
    
    await send_log_message(video_id, "Frame processing complete!", "success", "processing")
    await send_log_message(video_id, f"Processed {processed_count} frames", "info", "processing")
    await send_log_message(video_id, "Encoding output video (H.264)...", "info", "processing")
    
    # Re-encode video to H.264 for browser compatibility
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_output_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-movflags', '+faststart',
            str(output_path)
        ], check=True, capture_output=True)
        # Remove temporary file
        temp_output_path.unlink()
        await send_log_message(video_id, "Video encoded successfully (H.264)", "success", "processing")
    except subprocess.CalledProcessError as e:
        # If ffmpeg fails, fall back to the mp4v file
        print(f"FFmpeg re-encoding failed: {e.stderr.decode() if e.stderr else 'unknown error'}")
        temp_output_path.rename(output_path)
        await send_log_message(video_id, "FFmpeg encoding failed, using fallback codec", "warning", "processing")
    except FileNotFoundError:
        # ffmpeg not installed, use original file
        print("FFmpeg not found, using mp4v codec (may not play in browser)")
        temp_output_path.rename(output_path)
        await send_log_message(video_id, "FFmpeg not found, using fallback codec", "warning", "processing")
    
    await send_log_message(video_id, "Calculating player statistics...", "info", "processing")
    
    # Calculate final metrics for each player
    players_metrics = []
    
    # Filter to keep only the top 2 players by number of frames detected
    # This handles cases where YOLO tracker loses track and creates new IDs
    # All other detected people (spectators, refs, etc.) are excluded from output
    MAX_PLAYERS = 2
    if len(player_tracks) > MAX_PLAYERS:
        # Sort tracks by number of positions (frames where player was detected)
        sorted_tracks = sorted(
            player_tracks.items(),
            key=lambda x: len(x[1]["positions"]),
            reverse=True
        )
        # Keep only the top N players
        filtered_tracks = dict(sorted_tracks[:MAX_PLAYERS])
        excluded_count = len(player_tracks) - MAX_PLAYERS
        print(f"\n[PLAYER FILTERING] Detected {len(player_tracks)} unique track IDs")
        print(f"[PLAYER FILTERING] Keeping top {MAX_PLAYERS} players (by detection count)")
        print(f"[PLAYER FILTERING] Excluding {excluded_count} other detected people (spectators, refs, etc.)")
    else:
        filtered_tracks = player_tracks
        print(f"\n[PLAYER FILTERING] Detected {len(player_tracks)} players - no filtering needed")
    
    # Create mapping from old track IDs to new player IDs (1 and 2)
    # Sort by detection count to ensure consistent P1/P2 assignment
    sorted_main_players = sorted(filtered_tracks.keys(), key=lambda tid: len(filtered_tracks[tid]["positions"]), reverse=True)
    track_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted_main_players, start=1)}
    
    # Filter skeleton_frames to ONLY include the main 2 players (P1 and P2) PER FRAME
    #
    # IMPORTANT: We filter by bounding box size per frame, not by track ID.
    # This is because YOLO tracking can lose and regain tracks frequently,
    # causing track IDs to change throughout the video. Filtering by track ID
    # would result in empty player arrays for most frames.
    #
    # Instead, for each frame, we:
    # 1. Calculate the bounding box area for each detected player
    # 2. Keep only the 2 largest detections (actual players are usually larger than spectators)
    # 3. Assign player IDs based on vertical position (top of screen = P1, bottom = P2)
    #
    # This ensures spectators, referees, or other detected people are excluded
    # from the mini court display, pose overlay, and all statistics.
    
    print(f"[PLAYER FILTERING] Filtering each frame to keep only the 2 largest detections")
    total_filtered_out = 0
    
    for frame_data in skeleton_frames:
        players = frame_data["players"]
        
        if len(players) <= MAX_PLAYERS:
            # No filtering needed, just remap IDs based on position
            # Sort by Y position (lower Y = nearer to top of frame = P1)
            players.sort(key=lambda p: p["center"]["y"])
            for idx, player in enumerate(players):
                player["player_id"] = idx + 1  # Assign ID 1 or 2
        else:
            # Calculate bounding box area for each player (approximated from keypoints)
            def estimate_bbox_area(player_data):
                """Estimate bounding box area from keypoints spread"""
                keypoints = player_data.get("keypoints", [])
                valid_kps = [(kp["x"], kp["y"]) for kp in keypoints if kp["x"] is not None]
                if len(valid_kps) < 2:
                    return 0
                xs = [kp[0] for kp in valid_kps]
                ys = [kp[1] for kp in valid_kps]
                width = max(xs) - min(xs) if xs else 0
                height = max(ys) - min(ys) if ys else 0
                return width * height
            
            # Sort by estimated area (descending) and keep largest 2
            players_with_area = [(p, estimate_bbox_area(p)) for p in players]
            players_with_area.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the 2 largest
            filtered_players = [p for p, _ in players_with_area[:MAX_PLAYERS]]
            total_filtered_out += len(players) - len(filtered_players)
            
            # Sort remaining by Y position for consistent ID assignment
            filtered_players.sort(key=lambda p: p["center"]["y"])
            for idx, player in enumerate(filtered_players):
                player["player_id"] = idx + 1  # Assign ID 1 or 2
            
            frame_data["players"] = filtered_players
    
    print(f"[PLAYER FILTERING] Filtered out {total_filtered_out} non-player detections across all frames")
    
    # Reassign player IDs to be 1 and 2 (instead of arbitrary track IDs)
    for new_id, (track_id, track) in enumerate(sorted(filtered_tracks.items(), key=lambda x: len(x[1]["positions"]), reverse=True), start=1):
        # Use accurate distance if available, otherwise fall back to pixel-based calculation
        if "total_distance_meters" in track and track["total_distance_meters"] > 0:
            total_distance_meters = track["total_distance_meters"]
        else:
            total_distance_meters = pixel_to_meters(track["total_distance_pixels"], court_width_pixels)
        
        speeds = track["speeds"] if track["speeds"] else [0]
        
        players_metrics.append(PlayerMetrics(
            player_id=new_id,  # Use clean IDs (1, 2) instead of arbitrary track IDs
            total_distance=float(round(total_distance_meters, 2)),
            avg_speed=float(round(np.mean(speeds), 2)),
            max_speed=float(round(np.max(speeds), 2)),
            positions=[{"frame": p["frame"], "x": float(p["pos"][0]), "y": float(p["pos"][1])}
                      for p in track["positions"]],
            keypoints_history=track["keypoints_history"]
        ))
    
    # Shuttle detection (simplified - would need specific training for accurate detection)
    # For now, we'll estimate based on arm movements that indicate shots
    shuttle_metrics = None
    if config.track_shuttle and players_metrics:
        # Estimate shot speeds based on wrist velocity peaks
        shot_speeds = []
        for player in players_metrics:
            for i in range(1, len(player.keypoints_history)):
                prev_frame = player.keypoints_history[i-1]
                curr_frame = player.keypoints_history[i]
                
                # Get wrist positions
                prev_wrists = [(kp["x"], kp["y"]) for kp in prev_frame["keypoints"] 
                              if kp["name"] in ["left_wrist", "right_wrist"] and kp["x"] is not None]
                curr_wrists = [(kp["x"], kp["y"]) for kp in curr_frame["keypoints"] 
                              if kp["name"] in ["left_wrist", "right_wrist"] and kp["x"] is not None]
                
                if prev_wrists and curr_wrists:
                    # Calculate wrist velocity
                    for prev_w, curr_w in zip(prev_wrists, curr_wrists):
                        wrist_distance = calculate_distance(prev_w, curr_w)
                        wrist_dist_meters = pixel_to_meters(wrist_distance, court_width_pixels)
                        time_diff = config.fps_sample_rate / fps
                        wrist_speed = calculate_speed(wrist_dist_meters, time_diff)
                        
                        # If wrist speed is very high, it's likely a shot
                        if wrist_speed > 50:  # Threshold for shot detection
                            # Estimate shuttle speed (typically 1.5-3x wrist speed for smashes)
                            estimated_shuttle_speed = wrist_speed * 2.0
                            shot_speeds.append(float(min(estimated_shuttle_speed, 500)))  # Cap at 500 km/h
        
        if shot_speeds:
            shuttle_metrics = ShuttleMetrics(
                avg_speed=float(round(np.mean(shot_speeds), 2)),
                max_speed=float(round(np.max(shot_speeds), 2)),
                shots_detected=len(shot_speeds),
                shot_speeds=[float(s) for s in shot_speeds]
            )
    
    # Build court detection result for API response
    court_detection_api = None
    print(f"\n{'='*60}")
    print(f"VIDEO PROCESSING COMPLETE: {video_id}")
    print(f"{'='*60}")
    
    if court_detection_result and court_detection_result.detected:
        court_info = court_detector.get_detection_info(court_detection_result)
        court_detection_api = CourtDetectionResult(
            detected=court_info["detected"],
            confidence=court_info["confidence"],
            keypoints=court_info["keypoints"],
            court_corners=court_info["court_corners"],
            court_dimensions=court_info["court_dimensions"],
            regions=court_info.get("regions", []),  # Include regions for frontend rendering
            # Add Modal keypoint model data if available
            court_keypoints_22=modal_court_keypoints,
            court_model_used=modal_court_model_used
        )
    # If using Modal keypoint model but no local court detection, still create API response with Modal data
    elif modal_court_keypoints and modal_court_model_used == "keypoint":
        # Build court_detection_api from Modal keypoints only
        court_detection_api = CourtDetectionResult(
            detected=True,
            confidence=modal_court_keypoints.get("confidence", 0.0),
            keypoints=[],  # No region keypoints
            court_corners=None,  # Will be derived from 22 keypoints on frontend
            court_dimensions={
                "width_meters": COURT_WIDTH_DOUBLES,
                "length_meters": COURT_LENGTH
            },
            regions=[],
            court_keypoints_22=modal_court_keypoints,
            court_model_used=modal_court_model_used
        )
        print(f"  ✓ Using Modal keypoint model data (22 keypoints)")
    
    # Print final court detection result
    if court_detection_api:
        print(f"FINAL COURT DETECTION RESULT:")
        print(f"  - Detected: {court_detection_api.detected}")
        print(f"  - Confidence: {court_detection_api.confidence:.3f}")
        print(f"  - Court model used: {court_detection_api.court_model_used}")
        if court_detection_api.regions:
            print(f"  - Regions: {len(court_detection_api.regions)}")
            for region in court_detection_api.regions:
                print(f"      {region.get('name', 'unknown')}: bbox={region.get('bbox')}, conf={region.get('confidence', 0):.3f}")
        if court_detection_api.court_keypoints_22:
            kps = court_detection_api.court_keypoints_22.get("keypoints", [])
            print(f"  - Court keypoints (22): {len(kps)} keypoints")
            print(f"  - Keypoint confidence: {court_detection_api.court_keypoints_22.get('confidence', 0):.3f}")
        print(f"  - Keypoints (derived): {len(court_detection_api.keypoints)}")
        print(f"  - Court corners: {court_detection_api.court_corners}")
    else:
        print(f"FINAL COURT DETECTION RESULT: NOT DETECTED")
        if court_detection_result:
            print(f"  - Regions found: {len(court_detection_result.regions)}")
        else:
            print(f"  - No court detection result available")
    
    print(f"{'='*60}\n")
    
    # End any active shuttle trajectory and get enhanced analytics
    shuttle_tracker.end_trajectory()
    player_ids = [p.player_id for p in players_metrics]
    enhanced_analytics = create_analytics_summary(shuttle_tracker, position_analyzer, player_ids)
    
    # Apply frame interpolation if enabled (fills in gaps between processed frames)
    final_skeleton_frames = skeleton_frames
    if config.enable_interpolation and config.fps_sample_rate > 1 and len(skeleton_frames) >= 2:
        print(f"Applying frame interpolation (sample_rate={config.fps_sample_rate})...")
        interpolated_frames = []
        interpolator = FrameInterpolator()
        
        for i in range(len(skeleton_frames) - 1):
            current_frame = skeleton_frames[i]
            next_frame = skeleton_frames[i + 1]
            
            # Add current frame
            interpolated_frames.append(current_frame)
            
            # Calculate number of frames to interpolate
            frame_gap = next_frame["frame"] - current_frame["frame"]
            if frame_gap > 1:
                # Generate intermediate frames
                intermediate = interpolator.interpolate_frames(
                    current_frame,
                    next_frame,
                    frame_gap - 1
                )
                interpolated_frames.extend(intermediate)
        
        # Add last frame
        interpolated_frames.append(skeleton_frames[-1])
        final_skeleton_frames = interpolated_frames
        print(f"Interpolation complete: {len(skeleton_frames)} -> {len(final_skeleton_frames)} frames")
    
    # Build frame index for fast O(1) lookup
    frame_lookup_index = build_frame_index(final_skeleton_frames)
    
    # Store results (include video dimensions for accurate heatmap coordinate mapping)
    result = AnalysisResult(
        video_id=video_id,
        duration=round(duration, 2),
        fps=round(fps, 2),
        total_frames=total_frames,
        processed_frames=processed_count,
        video_width=width,  # Store actual video dimensions for heatmap/overlay coordinate mapping
        video_height=height,
        players=players_metrics,
        shuttle=shuttle_metrics,
        skeleton_data=final_skeleton_frames,
        court_detection=court_detection_api,
        shuttle_analytics=enhanced_analytics.get("shuttle_analytics"),
        player_zone_analytics=enhanced_analytics.get("player_analytics")
    )
    
    analysis_results[video_id] = result.model_dump()
    
    # Send final completion logs
    await send_log_message(video_id, f"Found {len(players_metrics)} players in video", "success", "detection")
    if shuttle_metrics:
        await send_log_message(video_id, f"Detected {shuttle_metrics.shots_detected} shots (max speed: {shuttle_metrics.max_speed:.1f} km/h)", "success", "detection")
    if court_detection_api and court_detection_api.detected:
        await send_log_message(video_id, "Court boundaries detected successfully", "success", "court")
    await send_log_message(video_id, "Analysis complete! Results ready.", "success", "processing")
    
    return result


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Badminton Tracker API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for analysis
    Returns a video_id for tracking the analysis
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix or ".mp4"
    file_path = UPLOAD_DIR / f"{video_id}{file_extension}"
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "size": len(contents),
        "status": "uploaded"
    }


@app.post("/api/analyze/{video_id}")
async def analyze_video(video_id: str, config: Optional[AnalysisConfig] = None):
    """
    Start video analysis for an uploaded video
    """
    if config is None:
        config = AnalysisConfig()
    
    # Find the video file
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video_files[0]
    
    # Start analysis in background
    try:
        result = await process_video(video_path, video_id, config)
        return {
            "video_id": video_id,
            "status": "completed",
            "result": result.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/results/{video_id}")
async def get_results(video_id: str):
    """Get analysis results for a video"""
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return analysis_results[video_id]


@app.get("/api/video/{video_id}")
async def get_processed_video(video_id: str):
    """Get the processed video with skeleton overlay"""
    video_path = PROCESSED_DIR / f"{video_id}_analyzed.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}_analyzed.mp4"
    )


@app.get("/api/original/{video_id}")
async def get_original_video(video_id: str):
    """Get the original uploaded video (browser-compatible)"""
    # Find the video file (may have different extensions)
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video_files[0]
    extension = video_path.suffix.lower()
    
    # Map extensions to MIME types
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".m4v": "video/mp4",
    }
    
    media_type = mime_types.get(extension, "video/mp4")
    
    return FileResponse(
        video_path,
        media_type=media_type,
        filename=f"{video_id}{extension}"
    )


@app.get("/api/skeleton/{video_id}")
async def get_skeleton_data(video_id: str):
    """Get skeleton overlay data as JSON for client-side rendering"""
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return {
        "video_id": video_id,
        "skeleton_data": analysis_results[video_id]["skeleton_data"]
    }


@app.get("/api/court-detection/status")
async def get_court_detection_status():
    """Get the status of the court detection module"""
    return {
        "enabled": court_detector.model is not None,
        "model_type": "local_yolo",
        "model_path": court_detector.model_path,
        "classes": ["frontcourt", "midcourt-down", "midcourt-up", "net",
                    "rearcourt-down", "rearcourt-up", "sideline-left", "sideline-right"],
        "court_dimensions": {
            "width_meters": COURT_WIDTH_DOUBLES,
            "length_meters": COURT_LENGTH
        }
    }


# =============================================================================
# MODAL GPU INFERENCE API ENDPOINTS
# =============================================================================

@app.get("/api/modal/status")
async def get_modal_inference_status():
    """
    Get the status of Modal.com GPU inference.
    
    Returns information about:
    - Whether Modal inference is enabled and available
    - Remote GPU health status
    - Loaded models on Modal
    
    See docs/MODAL_DEPLOYMENT_GUIDE.md for setup instructions.
    """
    if not MODAL_CLIENT_AVAILABLE:
        return {
            "available": False,
            "enabled": False,
            "reason": "Modal client not installed. Run: pip install httpx modal",
            "documentation": "docs/MODAL_DEPLOYMENT_GUIDE.md"
        }
    
    client = get_modal_client()
    
    if not client.is_enabled:
        return {
            "available": True,
            "enabled": False,
            "reason": "Modal inference disabled. Set USE_MODAL_INFERENCE=true in .env",
            "endpoint_url": client.config.endpoint_url or "(not configured)",
            "documentation": "docs/MODAL_DEPLOYMENT_GUIDE.md"
        }
    
    # Check health
    health = await client.health_check()
    
    return {
        "available": True,
        "enabled": True,
        "healthy": health.get("status") == "healthy",
        "endpoint_url": client.config.endpoint_url,
        "gpu": health.get("gpu_name", "Unknown"),
        "models_loaded": {
            "pose": health.get("pose_model_loaded", False),
            "detection": health.get("detection_model_loaded", False),
            "court": health.get("court_model_loaded", False)
        },
        "timeout": client.config.timeout,
        "max_retries": client.config.max_retries,
        "performance_note": (
            "GPU inference is ~10-50x faster than local CPU. "
            "Typical inference: 20-50ms per frame vs 200-500ms locally."
        ),
        "documentation": "docs/MODAL_DEPLOYMENT_GUIDE.md"
    }


@app.post("/api/modal/health-check")
async def check_modal_health():
    """
    Perform a health check on the Modal inference service.
    
    This will attempt to connect to Modal and verify GPU availability.
    Use this to diagnose connectivity issues.
    """
    if not MODAL_CLIENT_AVAILABLE:
        return {
            "status": "unavailable",
            "reason": "Modal client not installed"
        }
    
    client = get_modal_client()
    
    if not client.is_enabled:
        return {
            "status": "disabled",
            "reason": "Modal inference disabled in configuration"
        }
    
    try:
        health = await client.health_check()
        return {
            "status": health.get("status", "unknown"),
            "details": health
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e)
        }


@app.post("/api/modal/test-inference")
async def test_modal_inference(file: UploadFile = File(...)):
    """
    Test Modal inference on a single image.
    
    Upload an image and receive detection results from Modal's GPU.
    This is useful for verifying the inference pipeline is working.
    """
    if not MODAL_CLIENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Modal client not installed. Run: pip install httpx modal"
        )
    
    client = get_modal_client()
    
    if not client.is_enabled:
        raise HTTPException(
            status_code=503,
            detail="Modal inference disabled. Set USE_MODAL_INFERENCE=true in .env"
        )
    
    if not client.is_available:
        raise HTTPException(
            status_code=503,
            detail="Modal service not healthy. Run /api/modal/health-check to diagnose."
        )
    
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    try:
        # Run inference via Modal
        result = await client.process_frame(
            frame,
            frame_number=0,
            run_pose=True,
            run_detection=True,
            run_court=True,
            confidence_threshold=0.5
        )
        
        return {
            "status": "success",
            "inference_source": "modal",
            "inference_time_ms": result.inference_time_ms,
            "results": {
                "poses_detected": len(result.poses),
                "objects_detected": len(result.detections),
                "court_regions_detected": len(result.court_regions),
                "court_corners": result.court_corners
            },
            "skeleton_format": result.to_skeleton_format()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Modal inference failed: {str(e)}"
        )


@app.get("/api/badminton-detection/status")
async def get_badminton_detection_status():
    """
    Get the status of the badminton detection module.
    
    NOTE: This endpoint now uses the multi-model detector.
    For full model information, use /api/multi-model/status instead.
    """
    return {
        "enabled": multi_detector.is_available,
        "detection_mode": "multi_model",
        "models_loaded": multi_detector.available_models,
        "confidence_threshold": multi_detector.confidence_threshold,
        "player_classes": multi_detector.PLAYER_CLASSES,
        "shuttlecock_classes": multi_detector.SHUTTLECOCK_CLASSES,
        "racket_classes": multi_detector.RACKET_CLASSES,
        "annotation_method": "YOLOv26 plot() - recommended method",
        "note": "Legacy endpoint - prefer /api/multi-model/status for detailed model info"
    }


@app.post("/api/badminton-detection/update-classes")
async def update_badminton_detection_classes(
    player_classes: Optional[list[str]] = None,
    shuttlecock_classes: Optional[list[str]] = None
):
    """
    Update the class name mappings for player and shuttlecock detection.
    Use this if your custom model uses different class names.
    
    NOTE: This now updates the multi-model detector class mappings.
    """
    if player_classes:
        multi_detector.PLAYER_CLASSES = [c.lower() for c in player_classes]
    if shuttlecock_classes:
        multi_detector.SHUTTLECOCK_CLASSES = [c.lower() for c in shuttlecock_classes]
    
    return {
        "status": "updated",
        "player_classes": multi_detector.PLAYER_CLASSES,
        "shuttlecock_classes": multi_detector.SHUTTLECOCK_CLASSES
    }


@app.post("/api/badminton-detection/test")
async def test_badminton_detection(file: UploadFile = File(...)):
    """
    Test the badminton detection on a single image.
    Returns detected players, shuttlecocks, and rackets using YOLOv26 recommended method.
    
    NOTE: This endpoint now uses the multi-model detector.
    For more control, use /api/multi-model/test instead.
    """
    if not multi_detector.is_available:
        raise HTTPException(
            status_code=503,
            detail="No detection models configured. Check YOLO_BADMINTON_MODEL and YOLO_SHUTTLE_MODEL in .env"
        )
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Run detection using multi-model detector
    detections, annotated_frame = multi_detector.detect_and_annotate(frame, frame_number=0)
    
    return {
        "status": "success",
        "detection_mode": "multi_model",
        "models_used": list(detections.models.keys()),
        "annotation_method": "YOLOv26 plot() - recommended method",
        "detections": detections.to_dict(),
        "summary": {
            "players_detected": len(detections.players),
            "shuttlecocks_detected": len(detections.shuttlecocks),
            "rackets_detected": len(detections.rackets),
            "other_detected": len(detections.other)
        }
    }


@app.get("/api/court-detection/{video_id}")
async def get_court_detection(video_id: str):
    """Get court detection results for a video"""
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    court_data = analysis_results[video_id].get("court_detection")
    
    return {
        "video_id": video_id,
        "court_detection": court_data
    }


@app.get("/api/shuttle-analytics/{video_id}")
async def get_shuttle_analytics(video_id: str):
    """
    Get enhanced shuttle tracking analytics for a video.
    
    Returns detailed shuttle trajectory data including:
    - Shot speeds (instantaneous and average)
    - Shot type classification (smash, clear, drop, drive, net shot)
    - Trajectory positions in both pixel and court coordinates
    - Speed statistics
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    shuttle_analytics = analysis_results[video_id].get("shuttle_analytics")
    
    if not shuttle_analytics:
        return {
            "video_id": video_id,
            "shuttle_analytics": None,
            "message": "No shuttle analytics available. Ensure badminton detector is enabled and shuttlecock was detected."
        }
    
    return {
        "video_id": video_id,
        "shuttle_analytics": shuttle_analytics
    }


@app.get("/api/player-zone-analytics/{video_id}")
async def get_player_zone_analytics(video_id: str):
    """
    Get player court zone coverage analytics for a video.
    
    Returns zone coverage percentages and heatmap data for each player:
    - Time distribution across front/mid/back court
    - Time distribution across left/center/right sides
    - Average distance to net
    - Position heatmap grid
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    player_analytics = analysis_results[video_id].get("player_zone_analytics")
    
    if not player_analytics:
        return {
            "video_id": video_id,
            "player_zone_analytics": None,
            "message": "No player zone analytics available. Ensure court detection is enabled."
        }
    
    return {
        "video_id": video_id,
        "player_zone_analytics": player_analytics
    }


@app.get("/api/player-zone-analytics/{video_id}/recalculate")
async def get_recalculated_player_zone_analytics(video_id: str):
    """
    Get RECALCULATED player court zone coverage analytics using current manual keypoints.
    
    IMPORTANT: This endpoint recalculates zone coverage from skeleton data using
    the currently set manual court keypoints. This is critical because:
    1. Zone analytics are initially calculated during video processing
    2. If manual keypoints weren't set at that time, all zones show as 0%
    3. After user sets manual keypoints, this endpoint returns accurate zones
    
    Returns zone coverage percentages and heatmap data for each player:
    - Time distribution across front/mid/back court
    - Time distribution across left/center/right sides
    - Average distance to net
    - Position heatmap grid
    - manual_keypoints_used: Whether manual keypoints were used for calculation
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    video_width = result.get("video_width", 1920)
    video_height = result.get("video_height", 1080)
    
    if not skeleton_data:
        return {
            "video_id": video_id,
            "player_zone_analytics": None,
            "message": "No skeleton data available for zone recalculation"
        }
    
    # Check if stored analytics already have valid data
    stored_analytics = result.get("player_zone_analytics", {})
    has_valid_stored = False
    if stored_analytics:
        for player_id, analytics in stored_analytics.items():
            zone_coverage = analytics.get("zone_coverage", {})
            total = sum(zone_coverage.get(k, 0) for k in ["front", "mid", "back"])
            if total > 0:
                has_valid_stored = True
                break
    
    # Check if manual keypoints are set
    manual_kp_status = get_manual_keypoints_status()
    has_manual_keypoints = manual_kp_status.get("has_manual_keypoints", False)
    
    if has_valid_stored and not has_manual_keypoints:
        # Return stored analytics if they're valid and no manual keypoints to recalculate with
        return {
            "video_id": video_id,
            "player_zone_analytics": stored_analytics,
            "manual_keypoints_used": False,
            "recalculated": False,
            "message": "Using stored zone analytics (manual keypoints not set)"
        }
    
    # Recalculate zone analytics using manual keypoints
    try:
        recalculated = recalculate_zone_analytics_from_skeleton_data(
            skeleton_frames=skeleton_data,
            video_width=video_width,
            video_height=video_height
        )
        
        if recalculated:
            return {
                "video_id": video_id,
                "player_zone_analytics": recalculated,
                "manual_keypoints_used": has_manual_keypoints,
                "recalculated": True,
                "message": "Zone analytics recalculated using manual court keypoints"
            }
        else:
            # Recalculation returned empty - fall back to stored
            return {
                "video_id": video_id,
                "player_zone_analytics": stored_analytics or None,
                "manual_keypoints_used": False,
                "recalculated": False,
                "message": "Zone recalculation failed - manual keypoints may not be set"
            }
    except Exception as e:
        print(f"[ZONE ANALYTICS] Recalculation failed: {e}")
        return {
            "video_id": video_id,
            "player_zone_analytics": stored_analytics or None,
            "manual_keypoints_used": False,
            "recalculated": False,
            "error": str(e),
            "message": "Zone recalculation failed - using stored analytics"
        }


@app.get("/api/court-keypoints/info")
async def get_court_keypoints_info():
    """
    Get information about the court detection system.
    
    Returns details about:
    - Court region classes detected by the model
    - Standard badminton court dimensions
    - Homography transformation capabilities
    """
    from court_detection import COURT_WIDTH_DOUBLES, COURT_LENGTH, SERVICE_LINE_DISTANCE, BACK_BOUNDARY_SERVICE, COURT_CLASSES
    
    return {
        "court_dimensions": {
            "length_m": COURT_LENGTH,
            "width_doubles_m": COURT_WIDTH_DOUBLES,
            "width_singles_m": 5.18,
            "service_line_distance_m": SERVICE_LINE_DISTANCE,
            "back_boundary_service_m": BACK_BOUNDARY_SERVICE,
            "net_height_center_m": 1.524,
            "net_height_posts_m": 1.55
        },
        "detection_classes": COURT_CLASSES,
        "class_descriptions": {
            "frontcourt": "Front court zone near net",
            "midcourt-down": "Middle court zone (lower half)",
            "midcourt-up": "Middle court zone (upper half)",
            "net": "Net region at center of court",
            "rearcourt-down": "Rear court zone (lower half)",
            "rearcourt-up": "Rear court zone (upper half)",
            "sideline-left": "Left sideline boundary",
            "sideline-right": "Right sideline boundary"
        },
        "capabilities": {
            "homography_transform": "Converts pixel coordinates to real-world meters",
            "distance_calculation": "Accurate distance measurement using court reference",
            "speed_calculation": "Real-world speed calculation for shuttle and players",
            "zone_classification": "Classifies player position into court zones"
        },
        "model_info": {
            "model_type": "local_yolo",
            "model_path": "models/court/weights/best.pt",
            "detection_method": "Region-based detection (8 court zones)"
        }
    }


# =============================================================================
# MANUAL COURT KEYPOINTS API ENDPOINTS
# =============================================================================

class ManualKeypointsRequest(BaseModel):
    """Request body for setting manual court keypoints"""
    top_left: list[float]  # [x, y]
    top_right: list[float]  # [x, y]
    bottom_right: list[float]  # [x, y]
    bottom_left: list[float]  # [x, y]


@app.get("/api/court-keypoints/manual/status")
async def get_manual_keypoints_api_status():
    """
    Get the current status of manual court keypoints.
    
    RESEARCH: Manual keypoints vs Automatic Detection
    ================================================
    Manual keypoints provide significantly more accurate measurements because:
    1. User clicks on exact court corner positions (pixel-perfect)
    2. No bounding box interpolation errors from region detection
    3. Works even when automatic detection fails or is unreliable
    4. Consistent across all frames (no detection jitter)
    
    For maximum accuracy in player speed and distance calculations,
    manually specifying the four court corners is recommended.
    """
    return get_manual_keypoints_status()


@app.post("/api/court-keypoints/manual/set")
async def set_manual_keypoints_api(keypoints: ManualKeypointsRequest):
    """
    Set manual court keypoints to override automatic detection.
    
    This endpoint allows users to specify the four court corners manually
    for more accurate player speed and distance measurements.
    
    The keypoints should be the four corners of the visible court in pixel coordinates:
    - top_left: Top-left corner of the court [x, y]
    - top_right: Top-right corner of the court [x, y]
    - bottom_right: Bottom-right corner of the court [x, y]
    - bottom_left: Bottom-left corner of the court [x, y]
    
    USAGE: In the frontend, users can click on the four corners of the court
    in the video player to set these keypoints.
    
    Example request:
    {
        "top_left": [100, 50],
        "top_right": [500, 50],
        "bottom_right": [550, 400],
        "bottom_left": [50, 400]
    }
    """
    try:
        kp = set_manual_court_keypoints(
            top_left=tuple(keypoints.top_left),
            top_right=tuple(keypoints.top_right),
            bottom_right=tuple(keypoints.bottom_right),
            bottom_left=tuple(keypoints.bottom_left)
        )
        
        return {
            "status": "success",
            "message": "Manual court keypoints set successfully",
            "keypoints": kp.to_dict(),
            "mode": "manual",
            "recommendation": (
                "Manual keypoints are now active. Re-run video analysis "
                "for more accurate speed and distance measurements."
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set manual keypoints: {str(e)}"
        )


@app.delete("/api/court-keypoints/manual/clear")
async def clear_manual_keypoints_api():
    """
    Clear manual court keypoints and revert to automatic detection.
    
    After clearing, the system will use the YOLO model's automatic
    region detection to calculate court corners.
    """
    clear_manual_court_keypoints()
    
    return {
        "status": "success",
        "message": "Manual keypoints cleared. Using automatic detection.",
        "mode": "auto"
    }


class ManualKeypointsValidateRequest(BaseModel):
    """Request body for validating manual court keypoints"""
    top_left: list[float]  # [x, y]
    top_right: list[float]  # [x, y]
    bottom_right: list[float]  # [x, y]
    bottom_left: list[float]  # [x, y]
    frame_width: int
    frame_height: int


@app.post("/api/court-keypoints/manual/validate")
async def validate_manual_keypoints_api(request: ManualKeypointsValidateRequest):
    """
    Validate manual court keypoints BEFORE setting them.
    
    Use this endpoint to check if the clicked keypoints form a reasonable
    court shape that will produce accurate speed/distance calculations.
    
    This helps catch common errors such as:
    - Clicking on points that are too close together (causes inflated speeds)
    - Incorrect aspect ratio (clicked on half court instead of full court)
    - Points that don't form a proper quadrilateral
    
    Args:
        request: ManualKeypointsValidateRequest with corner coordinates and frame dimensions
    
    Returns:
        Validation results including:
        - valid: Whether the keypoints are acceptable
        - warnings: List of issues found
        - recommendations: Suggestions for fixing issues
        - speed_impact: How these keypoints would affect speed calculations
    
    Example request:
    {
        "top_left": [100, 50],
        "top_right": [500, 50],
        "bottom_right": [550, 400],
        "bottom_left": [50, 400],
        "frame_width": 1920,
        "frame_height": 1080
    }
    """
    try:
        # Create a temporary ManualCourtKeypoints object for validation
        keypoints = ManualCourtKeypoints(
            top_left=tuple(request.top_left),
            top_right=tuple(request.top_right),
            bottom_right=tuple(request.bottom_right),
            bottom_left=tuple(request.bottom_left)
        )
        
        # Validate using the court detector
        validation_result = court_detector.validate_manual_keypoints(
            keypoints=keypoints,
            frame_width=request.frame_width,
            frame_height=request.frame_height
        )
        
        return {
            "status": "validated",
            "keypoints": keypoints.to_dict(),
            "validation": validation_result,
            "summary": (
                "✓ Keypoints are valid for accurate speed calculation"
                if validation_result["valid"]
                else "⚠️ Issues detected - keypoints may cause inaccurate speed calculations"
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to validate keypoints: {str(e)}"
        )


@app.get("/api/speed/clamp-statistics")
async def get_speed_clamp_stats():
    """
    Get statistics about speed clamping during video analysis.
    
    This endpoint is useful for diagnosing issues where speeds seem wrong.
    If the clamp rate is high (>10%), it typically indicates:
    - Manual keypoints define too small an area (causing inflated speeds)
    - Tracking errors causing position jumps
    - Incorrect frame rate assumptions
    
    Returns:
        Dictionary with:
        - total_calculations: Number of speed calculations performed
        - clamped_count: How many were clamped
        - clamp_rate_pct: Percentage of speeds that were clamped
        - max_raw_speed_seen_kmh: Highest raw speed before clamping
        - recent_clamps: Last 10 clamp events with details
        - interpretation: Human-readable analysis of the statistics
    """
    return get_speed_clamp_statistics()


@app.post("/api/speed/clamp-statistics/reset")
async def reset_speed_clamp_stats():
    """
    Reset speed clamp statistics.
    
    Call this before starting a new video analysis to get fresh statistics.
    """
    reset_speed_clamp_statistics()
    return {
        "status": "reset",
        "message": "Speed clamp statistics have been reset"
    }


@app.get("/api/court-keypoints/accuracy-comparison")
async def get_accuracy_comparison():
    """
    Get information comparing manual vs automatic court keypoint accuracy.
    
    This endpoint provides research-oriented information about the
    trade-offs between manual and automatic court detection.
    """
    return {
        "comparison": {
            "automatic_detection": {
                "method": "YOLO region detection + corner interpolation",
                "pros": [
                    "No user interaction required",
                    "Works for any video automatically",
                    "Adapts to camera angle changes"
                ],
                "cons": [
                    "Corner positions are interpolated from bounding boxes",
                    "Detection may fail in poor lighting or unusual angles",
                    "Jitter between frames as detection varies",
                    "Sideline detection is crucial but can be unreliable"
                ],
                "accuracy": "Moderate - depends on model confidence and video quality",
                "best_for": "Quick analysis, videos where court is clearly visible"
            },
            "manual_keypoints": {
                "method": "User-specified corner coordinates",
                "pros": [
                    "Pixel-perfect corner positions",
                    "Consistent across all frames",
                    "Works regardless of detection model quality",
                    "Most accurate homography transformation"
                ],
                "cons": [
                    "Requires user interaction",
                    "Fixed positions (doesn't adapt to camera movement)",
                    "Must be reset for different videos"
                ],
                "accuracy": "High - limited only by user precision in clicking corners",
                "best_for": "Research analysis, accurate speed/distance measurements"
            }
        },
        "recommendation": (
            "For badminton analysis requiring accurate player speed and distance measurements, "
            "manual keypoints are recommended. Click on the four visible court corners in "
            "the first frame of your video for best results."
        ),
        "how_to_use": {
            "step_1": "Pause video on a frame where all four court corners are visible",
            "step_2": "Click on each corner in order: top-left, top-right, bottom-right, bottom-left",
            "step_3": "Submit the coordinates via POST /api/court-keypoints/manual/set",
            "step_4": "Re-run video analysis to use the new keypoints"
        }
    }


@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for real-time progress updates during analysis
    """
    await websocket.accept()
    active_connections[video_id] = websocket
    
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if video_id in active_connections:
            del active_connections[video_id]


@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and its analysis results"""
    # Delete uploaded file
    for file in UPLOAD_DIR.glob(f"{video_id}.*"):
        file.unlink()
    
    # Delete processed file
    processed_path = PROCESSED_DIR / f"{video_id}_analyzed.mp4"
    if processed_path.exists():
        processed_path.unlink()
    
    # Delete results
    if video_id in analysis_results:
        del analysis_results[video_id]
    
    return {"status": "deleted", "video_id": video_id}


# =============================================================================
# MULTI-MODEL DETECTION API ENDPOINTS
# =============================================================================

@app.get("/api/multi-model/status")
async def get_multi_model_status():
    """
    Get the status of all loaded detection models.
    
    Returns information about:
    - Available models and their classes
    - Enabled/disabled state for each model
    - Detection type toggles (players, shuttles, rackets)
    """
    return multi_detector.get_status()


@app.post("/api/multi-model/toggle-model/{model_name}")
async def toggle_model(model_name: str, enabled: bool = True):
    """
    Enable or disable a specific detection model.
    
    Args:
        model_name: Name of the model (e.g., 'badminton', 'shuttle')
        enabled: Whether to enable (true) or disable (false) the model
    """
    if model_name not in multi_detector.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {multi_detector.available_models}"
        )
    
    multi_detector.set_model_enabled(model_name, enabled)
    
    return {
        "status": "updated",
        "model": model_name,
        "enabled": enabled
    }


@app.post("/api/multi-model/toggle-detection-type/{detection_type}")
async def toggle_detection_type(detection_type: str, enabled: bool = True):
    """
    Enable or disable a specific detection type across all models.
    
    Args:
        detection_type: Type of detection ('player', 'shuttle', 'racket')
        enabled: Whether to enable (true) or disable (false)
    """
    try:
        dt = DetectionType(detection_type)
    except ValueError:
        valid_types = [t.value for t in DetectionType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid detection type. Valid types: {valid_types}"
        )
    
    multi_detector.set_type_enabled(dt, enabled)
    
    return {
        "status": "updated",
        "detection_type": detection_type,
        "enabled": enabled
    }


@app.post("/api/multi-model/test")
async def test_multi_model_detection(
    file: UploadFile = File(...),
    draw_players: bool = True,
    draw_shuttles: bool = True,
    draw_rackets: bool = True
):
    """
    Test multi-model detection on a single image.
    
    Returns detections from all enabled models with source tracking.
    """
    if not multi_detector.is_available:
        raise HTTPException(
            status_code=503,
            detail="No detection models available. Check YOLO_BADMINTON_MODEL and YOLO_SHUTTLE_MODEL in .env"
        )
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Configure drawing
    draw_config = {
        "players": draw_players,
        "shuttlecocks": draw_shuttles,
        "rackets": draw_rackets,
        "other": True
    }
    
    # Run multi-model detection
    detections, annotated_frame = multi_detector.detect_and_annotate(
        frame,
        frame_number=0,
        draw_config=draw_config
    )
    
    return {
        "status": "success",
        "models_used": list(detections.models.keys()),
        "detections": detections.to_dict(),
        "summary": {
            "players_detected": len(detections.players),
            "shuttlecocks_detected": len(detections.shuttlecocks),
            "rackets_detected": len(detections.rackets),
            "other_detected": len(detections.other)
        },
        "per_model": {
            name: {
                "detections_count": len(model_det.detections)
            }
            for name, model_det in detections.models.items()
        }
    }


@app.get("/api/detection-config")
async def get_detection_config():
    """
    Get the current detection configuration for the frontend.
    
    Returns toggleable options for the video player overlay.
    """
    return {
        "multi_model_available": multi_detector.is_available,
        "legacy_detector_available": False,  # Legacy detector removed - use multi_detector
        "models": {
            name: {
                "enabled": multi_detector.enabled_models.get(name, True),
                "classes": list(multi_detector.model_classes.get(name, {}).values()),
                "description": _get_model_description(name)
            }
            for name in multi_detector.available_models
        },
        "detection_types": {
            "players": {
                "enabled": multi_detector.enabled_types.get(DetectionType.PLAYER, True),
                "color": "#00FF00",
                "description": "Player detection and tracking"
            },
            "shuttles": {
                "enabled": multi_detector.enabled_types.get(DetectionType.SHUTTLE, True),
                "color": "#FFA500",
                "description": "Shuttlecock detection and trajectory"
            },
            "rackets": {
                "enabled": multi_detector.enabled_types.get(DetectionType.RACKET, True),
                "color": "#FF00FF",
                "description": "Racket detection"
            }
        },
        "overlays": {
            "skeleton": {
                "description": "Pose skeleton overlay (keypoints and connections)"
            },
            "bounding_boxes": {
                "description": "Object detection bounding boxes"
            },
            "court_lines": {
                "description": "Court boundary detection overlay"
            }
        }
    }


def _get_model_description(model_name: str) -> str:
    """Get human-readable description for a model"""
    descriptions = {
        "badminton": "General badminton detection (players, rackets, shuttles)",
        "shuttle": "Specialized shuttlecock detector (higher accuracy for shuttle tracking)",
        "court": "Court line and boundary detection",
        "pose": "Player pose estimation"
    }
    return descriptions.get(model_name, f"Detection model: {model_name}")


# =============================================================================
# SKELETON SMOOTHING API ENDPOINTS
# =============================================================================

# Global smoothed pose detector (initialized lazily)
_smoothed_detector = None


def _get_smoothed_detector():
    """Get or create the smoothed pose detector singleton"""
    global _smoothed_detector
    if _smoothed_detector is None:
        _smoothed_detector = get_smoothed_pose_detector(fps=30.0, preset="high_speed")
    return _smoothed_detector


@app.get("/api/smoothing/status")
async def get_smoothing_status():
    """
    Get the current smoothing configuration and status.
    
    Can be used by the frontend to show smoothing controls.
    """
    detector = _get_smoothed_detector()
    
    return {
        "enabled": detector.enable_smoothing,
        "available": detector.smoother is not None,
        "config": detector.get_smoothing_config(),
        "presets": {
            "high_speed": "Optimized for fast movements (badminton, tennis)",
            "stability": "Maximum smoothness, higher latency",
            "real_time": "Minimal latency, moderate smoothing",
            "auto": "Automatically adapts based on FPS"
        },
        "description": "Temporal smoothing reduces jerky skeleton movement using One Euro Filter"
    }


@app.post("/api/smoothing/toggle")
async def toggle_smoothing(enabled: bool = True):
    """
    Toggle smoothing on or off at runtime.
    
    This can be called from the frontend without restarting the backend.
    Smoothing state is preserved - when re-enabled, it resumes from
    existing filter states (no discontinuity in the skeleton).
    
    Args:
        enabled: Whether to enable (true) or disable (false) smoothing
    """
    detector = _get_smoothed_detector()
    success = detector.set_smoothing_enabled(enabled)
    
    return {
        "status": "updated" if success else "failed",
        "enabled": detector.enable_smoothing,
        "message": "Smoothing toggled successfully" if success else "Failed to toggle smoothing"
    }


@app.post("/api/smoothing/preset/{preset_name}")
async def set_smoothing_preset(preset_name: str):
    """
    Change the smoothing preset at runtime.
    
    Presets adjust the smoothing parameters for different use cases:
    - high_speed: Fast response for sports (recommended for badminton)
    - stability: Maximum smoothness at the cost of slight latency
    - real_time: Minimal computational overhead
    - none: Disables smoothing entirely
    
    Args:
        preset_name: One of "auto", "high_speed", "stability", "real_time", "none"
    """
    valid_presets = ["auto", "high_speed", "stability", "real_time", "none"]
    
    if preset_name not in valid_presets:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Invalid preset. Valid options: {valid_presets}"
        )
    
    detector = _get_smoothed_detector()
    success = detector.set_smoothing_preset(preset_name)
    
    return {
        "status": "updated" if success else "failed",
        "preset": preset_name,
        "enabled": detector.enable_smoothing,
        "config": detector.get_smoothing_config()
    }


@app.post("/api/smoothing/reset")
async def reset_smoothing():
    """
    Reset all smoothing state.
    
    Call this when starting analysis of a new video to clear
    any accumulated filter state from previous videos.
    """
    detector = _get_smoothed_detector()
    detector.reset_smoothing()
    
    return {
        "status": "reset",
        "message": "Smoothing state cleared"
    }


@app.get("/api/smoothing/benchmark")
async def benchmark_smoothing():
    """
    Run a quick performance benchmark of the smoothing algorithms.
    
    Returns timing information to verify real-time performance.
    """
    try:
        from skeleton_smoothing import benchmark_smoothing as run_benchmark
        results = run_benchmark(iterations=500)
        
        # Check if all algorithms are fast enough for 60fps
        fps_60_target_ms = 1000 / 60  # ~16.67ms
        
        return {
            "status": "success",
            "results": {
                name: {
                    "ms_per_frame": round(ms, 4),
                    "suitable_for_60fps": ms < fps_60_target_ms,
                    "suitable_for_120fps": ms < (1000 / 120)
                }
                for name, ms in results.items()
            },
            "recommendation": "All algorithms suitable for real-time processing"
        }
    except ImportError:
        return {
            "status": "unavailable",
            "message": "skeleton_smoothing module not available"
        }


# =============================================================================
# HEATMAP API ENDPOINTS
# =============================================================================

class HeatmapConfigRequest(BaseModel):
    """Request body for heatmap configuration"""
    colormap: Optional[str] = "turbo"  # turbo, parula, inferno, viridis, plasma, hot, jet, cool
    kernel_size: Optional[int] = 31  # Gaussian kernel size (odd number)
    sigma: Optional[float] = 10.0  # Gaussian spread
    decay_rate: Optional[float] = 0.995  # Per-frame decay rate
    intensity_scale: Optional[float] = 1.0  # Heat intensity multiplier


class HeatmapGenerateRequest(BaseModel):
    """Request body for heatmap generation"""
    config: Optional[HeatmapConfigRequest] = None
    player_id: Optional[int] = None  # None = all players combined
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


@app.get("/api/heatmap/{video_id}")
async def get_heatmap_data(video_id: str, player_id: Optional[int] = None):
    """
    Get pre-generated heatmap data for a video.
    
    This returns the heatmap accumulation matrix that can be rendered
    on the frontend as an overlay on the video.
    
    Args:
        video_id: The video ID to get heatmap for
        player_id: Optional player ID to filter (1 or 2). If None, returns combined heatmap.
    
    Returns:
        Heatmap data including the normalized heat matrix and configuration.
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found. Run analysis first.")
    
    # Check if heatmap data exists
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    
    if not skeleton_data:
        raise HTTPException(status_code=404, detail="No skeleton data available for heatmap generation")
    
    # Get video dimensions from the analysis result (stored during processing)
    # Fallback: read from original video file if not in results (for backwards compatibility)
    video_width = result.get("video_width")
    video_height = result.get("video_height")
    
    if video_width is None or video_height is None:
        # Try to read dimensions from original video file
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if video_files:
            cap = cv2.VideoCapture(str(video_files[0]))
            if cap.isOpened():
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"[HEATMAP] Read video dimensions from file: {video_width}x{video_height}")
            else:
                video_width = 1920
                video_height = 1080
                print(f"[HEATMAP] Using default dimensions: {video_width}x{video_height}")
        else:
            video_width = 1920
            video_height = 1080
            print(f"[HEATMAP] Video file not found, using default: {video_width}x{video_height}")
    else:
        print(f"[HEATMAP] Using stored video dimensions: {video_width}x{video_height}")
    
    # Generate heatmap from skeleton data
    try:
        # Filter for specific player if requested
        if player_id is not None:
            filtered_skeleton = []
            for frame in skeleton_data:
                filtered_frame = frame.copy()
                filtered_frame["players"] = [
                    p for p in frame.get("players", [])
                    if p.get("player_id") == player_id
                ]
                filtered_skeleton.append(filtered_frame)
        else:
            filtered_skeleton = skeleton_data
        
        heatmap_data = generate_heatmap_from_skeleton_data(
            skeleton_frames=filtered_skeleton,
            video_width=video_width,
            video_height=video_height,
            video_id=video_id
        )
        
        return {
            "video_id": video_id,
            "player_id": player_id,
            "heatmap": heatmap_data.to_dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")


@app.post("/api/heatmap/{video_id}/generate")
async def generate_heatmap(video_id: str, request: HeatmapGenerateRequest):
    """
    Generate a new heatmap from skeleton data with custom configuration.
    
    This allows customizing the heatmap appearance:
    - colormap: Color scheme (turbo, parula, inferno, etc.)
    - kernel_size: Gaussian blur kernel size for smooth heat spread
    - sigma: Gaussian spread parameter
    - decay_rate: Temporal decay for fading effect
    - intensity_scale: Heat intensity multiplier
    
    Args:
        video_id: The video ID to generate heatmap for
        request: Generation configuration
    
    Returns:
        Generated heatmap data with the specified configuration.
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found. Run analysis first.")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    
    if not skeleton_data:
        raise HTTPException(status_code=404, detail="No skeleton data available for heatmap generation")
    
    # Get video dimensions (with fallback for backwards compatibility)
    video_width = result.get("video_width")
    video_height = result.get("video_height")
    
    if video_width is None or video_height is None:
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if video_files:
            cap = cv2.VideoCapture(str(video_files[0]))
            if cap.isOpened():
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        if video_width is None:
            video_width, video_height = 1920, 1080
    print(f"[HEATMAP GENERATE] Using video dimensions: {video_width}x{video_height}")
    
    # Build config from request
    config = None
    if request.config:
        # Validate colormap
        valid_colormaps = ["turbo", "parula", "inferno", "viridis", "plasma", "hot", "jet", "cool"]
        colormap = request.config.colormap.lower() if request.config.colormap else "turbo"
        if colormap not in valid_colormaps:
            colormap = "turbo"
        
        config = HeatmapConfig(
            colormap=colormap,
            kernel_size=request.config.kernel_size or 31,
            sigma=request.config.sigma or 10.0,
            decay_rate=request.config.decay_rate or 0.995,
            intensity_scale=request.config.intensity_scale or 1.0
        )
    
    # Filter frames if specified
    filtered_skeleton_data = skeleton_data
    if request.start_frame is not None or request.end_frame is not None:
        start = request.start_frame or 0
        end = request.end_frame or float('inf')
        filtered_skeleton_data = [
            frame for frame in skeleton_data
            if start <= frame.get("frame", 0) <= end
        ]
    
    # Filter for specific player if requested
    if request.player_id is not None:
        player_filtered = []
        for frame in filtered_skeleton_data:
            filtered_frame = frame.copy()
            filtered_frame["players"] = [
                p for p in frame.get("players", [])
                if p.get("player_id") == request.player_id
            ]
            player_filtered.append(filtered_frame)
        filtered_skeleton_data = player_filtered
    
    try:
        heatmap_data = generate_heatmap_from_skeleton_data(
            skeleton_frames=filtered_skeleton_data,
            video_width=video_width,
            video_height=video_height,
            video_id=video_id,
            config=config
        )
        
        return {
            "video_id": video_id,
            "player_id": request.player_id,
            "frame_range": {
                "start": request.start_frame,
                "end": request.end_frame,
                "frames_processed": len(filtered_skeleton_data)
            },
            "heatmap": heatmap_data.to_dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")


@app.get("/api/heatmap/colormaps")
async def get_available_colormaps():
    """
    Get list of available colormap options for heatmap visualization.
    
    Returns colormap names and their descriptions for the frontend UI.
    """
    return {
        "colormaps": [
            {
                "name": "turbo",
                "display_name": "Turbo",
                "description": "High-contrast rainbow colormap (recommended)",
                "opencv_code": "cv2.COLORMAP_TURBO"
            },
            {
                "name": "parula",
                "display_name": "Parula",
                "description": "Blue-to-yellow perceptually uniform colormap",
                "opencv_code": "cv2.COLORMAP_PARULA"
            },
            {
                "name": "inferno",
                "display_name": "Inferno",
                "description": "Black-red-yellow colormap for intensity",
                "opencv_code": "cv2.COLORMAP_INFERNO"
            },
            {
                "name": "magma",
                "display_name": "Magma",
                "description": "Black-purple-orange-white colormap",
                "opencv_code": "cv2.COLORMAP_MAGMA"
            },
            {
                "name": "viridis",
                "display_name": "Viridis",
                "description": "Green-blue-yellow perceptually uniform",
                "opencv_code": "cv2.COLORMAP_VIRIDIS"
            },
            {
                "name": "plasma",
                "display_name": "Plasma",
                "description": "Blue-purple-orange-yellow colormap",
                "opencv_code": "cv2.COLORMAP_PLASMA"
            },
            {
                "name": "hot",
                "display_name": "Hot",
                "description": "Classic black-red-yellow-white heat",
                "opencv_code": "cv2.COLORMAP_HOT"
            }
        ],
        "default": "turbo",
        "recommended_for_badminton": "turbo"
    }


@app.get("/api/heatmap/{video_id}/frame/{frame_number}")
async def get_heatmap_at_frame(
    video_id: str,
    frame_number: int,
    player_id: Optional[int] = None,
    colormap: str = "turbo",
    decay_rate: float = 0.995
):
    """
    Get heatmap data accumulated up to a specific frame.
    
    This is useful for:
    - Showing progressive heatmap buildup during video playback
    - Creating time-lapse visualization of player movement
    
    Args:
        video_id: The video ID
        frame_number: The frame number to accumulate heatmap up to
        player_id: Optional player filter
        colormap: Colormap to use (turbo, parula, inferno, viridis, plasma, hot, jet, cool)
        decay_rate: Temporal decay rate (0.9-1.0)
    
    Returns:
        Heatmap data accumulated from start to the specified frame.
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    
    if not skeleton_data:
        raise HTTPException(status_code=404, detail="No skeleton data available")
    
    # Filter to frames up to the specified frame number
    filtered_data = [
        frame for frame in skeleton_data
        if frame.get("frame", 0) <= frame_number
    ]
    
    if not filtered_data:
        return {
            "video_id": video_id,
            "frame_number": frame_number,
            "player_id": player_id,
            "heatmap": None,
            "message": "No data for specified frame range"
        }
    
    # Filter for specific player if requested
    if player_id is not None:
        player_filtered = []
        for frame in filtered_data:
            filtered_frame = frame.copy()
            filtered_frame["players"] = [
                p for p in frame.get("players", [])
                if p.get("player_id") == player_id
            ]
            player_filtered.append(filtered_frame)
        filtered_data = player_filtered
    
    # Validate colormap
    valid_colormaps = ["turbo", "parula", "inferno", "viridis", "plasma", "hot", "jet", "cool"]
    colormap_str = colormap.lower() if colormap else "turbo"
    if colormap_str not in valid_colormaps:
        colormap_str = "turbo"
    
    config = HeatmapConfig(
        colormap=colormap_str,
        decay_rate=decay_rate
    )
    
    # Get video dimensions (with fallback for backwards compatibility)
    video_width = result.get("video_width")
    video_height = result.get("video_height")
    
    if video_width is None or video_height is None:
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if video_files:
            cap = cv2.VideoCapture(str(video_files[0]))
            if cap.isOpened():
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        if video_width is None:
            video_width, video_height = 1920, 1080
    print(f"[HEATMAP FRAME] Using video dimensions: {video_width}x{video_height}")
    
    try:
        heatmap_data = generate_heatmap_from_skeleton_data(
            skeleton_frames=filtered_data,
            video_width=video_width,
            video_height=video_height,
            video_id=video_id,
            config=config
        )
        
        return {
            "video_id": video_id,
            "frame_number": frame_number,
            "player_id": player_id,
            "frames_accumulated": len(filtered_data),
            "heatmap": heatmap_data.to_dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")


@app.delete("/api/heatmap/cache/clear")
async def clear_heatmap_cache():
    """
    Clear the heatmap generator cache.
    
    Call this to free memory if heatmaps are no longer needed,
    or to force regeneration with fresh state.
    """
    clear_generator_cache()
    
    return {
        "status": "success",
        "message": "Heatmap cache cleared"
    }


# =============================================================================
# SPEED ANALYTICS API ENDPOINTS
# =============================================================================

@app.get("/api/speed/{video_id}")
async def get_speed_graph_data(video_id: str, window_seconds: float = 60.0):
    """
    Get speed graph data for a video.
    
    IMPORTANT: This endpoint now recalculates speeds using manual court keypoints
    if they have been set. This means speeds will be accurate even if:
    1. Manual keypoints were set AFTER video processing
    2. Automatic court detection failed during processing
    
    Returns speed data formatted for Chart.js visualization including:
    - Per-player speed history (sliding window for real-time, full history for post-match)
    - Statistics (current, max, average speeds) per player
    - Speed zone thresholds and colors for reference bands
    - Time range for X-axis scaling
    
    Args:
        video_id: The video ID to get speed data for
        window_seconds: Sliding window size for display (30-60 recommended)
    
    Returns:
        Speed graph data dictionary with:
        - players: Dict of player_id -> {window_data, full_history}
        - statistics: Dict of player_id -> speed stats
        - zone_thresholds: Speed zone reference data
        - time_range: Min/max timestamps in data
        - manual_keypoints_used: Whether manual keypoints were used for calculation
        - detection_source: "manual" or "auto"
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found. Run analysis first.")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    fps = result.get("fps", 30.0)
    
    # Get video dimensions for homography calculation with manual keypoints
    video_width = result.get("video_width", 1920)
    video_height = result.get("video_height", 1080)
    
    if not skeleton_data:
        raise HTTPException(status_code=404, detail="No skeleton data available for speed calculation")
    
    try:
        # Calculate speed data from skeleton frames
        # NOTE: This now properly uses manual court keypoints if set
        speed_data = calculate_speed_from_skeleton_data(
            skeleton_frames=skeleton_data,
            fps=fps,
            window_seconds=window_seconds,
            video_width=video_width,
            video_height=video_height
        )
        
        return {
            "video_id": video_id,
            "fps": fps,
            "speed_data": speed_data,
            "status": "success",
            "manual_keypoints_used": speed_data.get("manual_keypoints_used", False),
            "detection_source": speed_data.get("detection_source", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate speed data: {str(e)}")


@app.get("/api/speed/{video_id}/statistics")
async def get_speed_statistics(video_id: str):
    """
    Get aggregated speed statistics for all players in a video.
    
    IMPORTANT: This endpoint now recalculates speeds using manual court keypoints
    if they have been set, ensuring accurate statistics even after manual calibration.
    
    Returns summary statistics without the full speed history,
    useful for dashboard displays and performance summaries.
    
    Returns:
        - Per-player stats: current, max, avg speed (m/s and km/h)
        - Total distance traveled (meters)
        - Time distribution across speed zones (percentage)
        - manual_keypoints_used: Whether manual keypoints were used
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    fps = result.get("fps", 30.0)
    
    # Get video dimensions for homography calculation
    video_width = result.get("video_width", 1920)
    video_height = result.get("video_height", 1080)
    
    if not skeleton_data:
        return {
            "video_id": video_id,
            "statistics": {},
            "message": "No skeleton data available"
        }
    
    try:
        speed_data = calculate_speed_from_skeleton_data(
            skeleton_frames=skeleton_data,
            fps=fps,
            window_seconds=60.0,
            video_width=video_width,
            video_height=video_height
        )
        
        return {
            "video_id": video_id,
            "statistics": speed_data.get("statistics", {}),
            "zone_thresholds": speed_data.get("zone_thresholds", []),
            "manual_keypoints_used": speed_data.get("manual_keypoints_used", False),
            "detection_source": speed_data.get("detection_source", "unknown"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")


@app.get("/api/speed/zones/info")
async def get_speed_zones_info():
    """
    Get information about speed zone classifications.
    
    Returns speed zone thresholds, colors, and descriptions
    for the frontend to display speed zone legends and indicators.
    
    Speed zones are based on realistic badminton player movement research:
    - Standing: < 0.5 m/s (stationary, preparing for shot)
    - Walking: 0.5 - 1.5 m/s (repositioning between points)
    - Active: 1.5 - 3.0 m/s (normal footwork during rallies)
    - Running: 3.0 - 5.0 m/s (quick court coverage)
    - Sprinting: 5.0 - 7.0 m/s (fast recovery moves)
    - Explosive: > 7.0 m/s (lunges, dives - should be RARE)
    
    NOTE: Most badminton movement should be in Standing/Active/Running zones.
    Sustained Sprinting/Explosive readings indicate tracking issues.
    """
    zones = []
    for zone in SpeedZone:
        min_mps, max_mps = SPEED_ZONE_THRESHOLDS[zone]
        zones.append({
            "zone": zone.value,
            "display_name": zone.value.capitalize(),
            "min_mps": min_mps,
            "max_mps": max_mps if max_mps != float('inf') else None,
            "min_kmh": round(min_mps * 3.6, 1),
            "max_kmh": round(max_mps * 3.6, 1) if max_mps != float('inf') else None,
            "color": SPEED_ZONE_COLORS[zone],
            "description": _get_zone_description(zone)
        })
    
    return {
        "zones": zones,
        "unit_info": {
            "mps": "meters per second",
            "kmh": "kilometers per hour",
            "conversion": "km/h = m/s × 3.6"
        },
        "research_note": (
            "Speed zone thresholds are based on realistic badminton player movement patterns. "
            "Most movement is 5-18 km/h. Speeds above 25 km/h should be rare, brief moments. "
            "If sustained high speeds occur, check tracking/calibration."
        )
    }


def _get_zone_description(zone: SpeedZone) -> str:
    """Get human-readable description for a speed zone"""
    descriptions = {
        SpeedZone.STANDING: "Stationary, preparing for next shot",
        SpeedZone.WALKING: "Casual repositioning between points",
        SpeedZone.ACTIVE: "Normal rally footwork, positioning",
        SpeedZone.RUNNING: "Quick court coverage, active play",
        SpeedZone.SPRINTING: "Fast recovery, chasing difficult shots",
        SpeedZone.EXPLOSIVE: "Lunges, dives, explosive bursts (should be rare)"
    }
    return descriptions.get(zone, "Movement speed zone")


@app.get("/api/speed/{video_id}/frame/{frame_number}")
async def get_speed_at_frame(video_id: str, frame_number: int):
    """
    Get speed data at a specific frame.
    
    Useful for synchronized display with video playback.
    Returns the speed of each player at the specified frame.
    
    Args:
        video_id: The video ID
        frame_number: The frame number to query
    
    Returns:
        Speed data for each player at the specified frame
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    fps = result.get("fps", 30.0)
    
    # Find the frame in skeleton data
    frame_data = None
    for frame in skeleton_data:
        if frame.get("frame") == frame_number:
            frame_data = frame
            break
    
    if not frame_data:
        # Try to find nearest frame
        closest_frame = None
        min_diff = float('inf')
        for frame in skeleton_data:
            diff = abs(frame.get("frame", 0) - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame
        frame_data = closest_frame
    
    if not frame_data:
        return {
            "video_id": video_id,
            "frame_number": frame_number,
            "players": [],
            "message": "No data for specified frame"
        }
    
    # Extract player speeds from skeleton data
    players_speed = []
    for player in frame_data.get("players", []):
        speed_kmh = player.get("current_speed", 0.0)
        speed_mps = speed_kmh / 3.6
        
        # Classify into zone
        zone = SpeedZone.STANDING
        for z, (min_s, max_s) in SPEED_ZONE_THRESHOLDS.items():
            if min_s <= speed_mps < max_s:
                zone = z
                break
        
        players_speed.append({
            "player_id": player.get("player_id"),
            "speed_mps": round(speed_mps, 3),
            "speed_kmh": round(speed_kmh, 2),
            "zone": zone.value,
            "zone_color": SPEED_ZONE_COLORS[zone],
            "position": player.get("center")
        })
    
    return {
        "video_id": video_id,
        "frame_number": frame_data.get("frame"),
        "timestamp": frame_data.get("timestamp"),
        "players": players_speed,
        "status": "success"
    }


@app.get("/api/speed/{video_id}/timeline")
async def get_speed_timeline(
    video_id: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    sample_rate: int = 1
):
    """
    Get speed timeline data for a range of frames.
    
    IMPORTANT: This endpoint now RECALCULATES speeds using manual court keypoints
    if they have been set. This is critical because:
    1. During initial video processing, speeds are set to 0 (delay_speed_calculation=True)
    2. After manual keypoints are set, this endpoint returns accurate speeds
    
    This endpoint is optimized for chart rendering - returns
    arrays of timestamps and speeds for efficient plotting.
    
    Args:
        video_id: The video ID
        start_frame: Optional start frame (default: beginning)
        end_frame: Optional end frame (default: end)
        sample_rate: Only include every Nth frame (for performance with long videos)
    
    Returns:
        Timeline data with timestamps and speed arrays per player
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found")
    
    result = analysis_results[video_id]
    skeleton_data = result.get("skeleton_data", [])
    fps = result.get("fps", 30.0)
    
    # Get video dimensions for homography calculation with manual keypoints
    video_width = result.get("video_width", 1920)
    video_height = result.get("video_height", 1080)
    
    if not skeleton_data:
        return {
            "video_id": video_id,
            "timeline": {},
            "message": "No skeleton data available"
        }
    
    # CRITICAL FIX: Recalculate speeds using manual court keypoints
    # The stored skeleton_data has current_speed=0 when delay_speed_calculation=True
    # We need to recalculate speeds post-hoc using the speed analytics module
    print(f"[SPEED TIMELINE] Recalculating speeds for video {video_id}")
    
    try:
        speed_data = calculate_speed_from_skeleton_data(
            skeleton_frames=skeleton_data,
            fps=fps,
            window_seconds=60.0,  # Full history
            video_width=video_width,
            video_height=video_height
        )
        
        print(f"[SPEED TIMELINE] Speed calculation complete")
        print(f"  - Manual keypoints used: {speed_data.get('manual_keypoints_used', False)}")
        print(f"  - Detection source: {speed_data.get('detection_source', 'unknown')}")
        
        # Build per-frame speed lookup from calculated data
        # The speed_data contains full_history arrays per player
        player_speeds_by_frame: Dict[int, Dict[int, float]] = {}  # player_id -> {frame: speed_kmh}
        
        for player_id_str, player_data in speed_data.get("players", {}).items():
            player_id = int(player_id_str)
            player_speeds_by_frame[player_id] = {}
            
            # Build frame -> speed mapping from full_history
            full_history = player_data.get("full_history", [])
            for entry in full_history:
                # Map timestamp back to frame number
                frame_num = int(entry.get("timestamp", 0) * fps)
                speed_kmh = entry.get("speed_kmh", 0.0)
                player_speeds_by_frame[player_id][frame_num] = speed_kmh
        
    except Exception as e:
        print(f"[SPEED TIMELINE] Speed calculation failed: {e}")
        # Fall back to stored speeds (will be 0 if delay_speed_calculation was True)
        player_speeds_by_frame = {}
        speed_data = {"manual_keypoints_used": False, "detection_source": "fallback"}
    
    # Filter frame range
    filtered_frames = skeleton_data
    if start_frame is not None:
        filtered_frames = [f for f in filtered_frames if f.get("frame", 0) >= start_frame]
    if end_frame is not None:
        filtered_frames = [f for f in filtered_frames if f.get("frame", 0) <= end_frame]
    
    # Apply sample rate
    if sample_rate > 1:
        filtered_frames = filtered_frames[::sample_rate]
    
    # Build timeline per player using recalculated speeds
    players_timeline: Dict[int, Dict] = {}
    
    for frame_data in filtered_frames:
        frame_num = frame_data.get("frame", 0)
        timestamp = frame_data.get("timestamp", frame_num / fps)
        
        for player in frame_data.get("players", []):
            player_id = player.get("player_id", 0)
            
            # Use recalculated speed if available, otherwise fall back to stored value
            if player_id in player_speeds_by_frame and frame_num in player_speeds_by_frame[player_id]:
                speed_kmh = player_speeds_by_frame[player_id][frame_num]
            else:
                # Try to find closest frame in calculated data
                if player_id in player_speeds_by_frame:
                    available_frames = list(player_speeds_by_frame[player_id].keys())
                    if available_frames:
                        closest_frame = min(available_frames, key=lambda f: abs(f - frame_num))
                        if abs(closest_frame - frame_num) < 5:  # Within 5 frames
                            speed_kmh = player_speeds_by_frame[player_id][closest_frame]
                        else:
                            speed_kmh = player.get("current_speed", 0.0)  # Stored fallback
                    else:
                        speed_kmh = player.get("current_speed", 0.0)
                else:
                    speed_kmh = player.get("current_speed", 0.0)
            
            speed_mps = speed_kmh / 3.6
            
            if player_id not in players_timeline:
                players_timeline[player_id] = {
                    "frames": [],
                    "timestamps": [],
                    "speeds_mps": [],
                    "speeds_kmh": [],
                    "zones": []
                }
            
            # Classify zone
            zone = SpeedZone.STANDING
            for z, (min_s, max_s) in SPEED_ZONE_THRESHOLDS.items():
                if min_s <= speed_mps < max_s:
                    zone = z
                    break
            
            players_timeline[player_id]["frames"].append(frame_num)
            players_timeline[player_id]["timestamps"].append(round(timestamp, 3))
            players_timeline[player_id]["speeds_mps"].append(round(speed_mps, 3))
            players_timeline[player_id]["speeds_kmh"].append(round(speed_kmh, 2))
            players_timeline[player_id]["zones"].append(zone.value)
    
    # Calculate statistics per player
    for player_id, timeline in players_timeline.items():
        speeds = timeline["speeds_mps"]
        if speeds:
            timeline["stats"] = {
                "max_mps": round(max(speeds), 3),
                "max_kmh": round(max(speeds) * 3.6, 2),
                "avg_mps": round(sum(speeds) / len(speeds), 3),
                "avg_kmh": round((sum(speeds) / len(speeds)) * 3.6, 2),
                "data_points": len(speeds)
            }
    
    return {
        "video_id": video_id,
        "frame_range": {
            "start": filtered_frames[0].get("frame") if filtered_frames else None,
            "end": filtered_frames[-1].get("frame") if filtered_frames else None,
            "total_frames": len(filtered_frames)
        },
        "sample_rate": sample_rate,
        "fps": fps,
        "players": players_timeline,
        "zone_colors": {zone.value: color for zone, color in SPEED_ZONE_COLORS.items()},
        "manual_keypoints_used": speed_data.get("manual_keypoints_used", False),
        "detection_source": speed_data.get("detection_source", "unknown"),
        "status": "success"
    }


# =============================================================================
# PDF EXPORT API ENDPOINTS
# =============================================================================

class PDFExportRequest(BaseModel):
    """Request body for PDF export configuration"""
    frame_number: Optional[int] = None  # Frame to use for heatmap visualization (None = middle frame)
    include_heatmap: bool = True
    heatmap_colormap: str = "turbo"  # turbo, parula, inferno, viridis, plasma, hot
    heatmap_alpha: float = 0.6  # Heatmap overlay opacity (0-1)
    include_player_stats: bool = True
    include_shuttle_stats: bool = True
    include_court_info: bool = True
    include_speed_stats: bool = True
    title: str = "Badminton Video Analysis Report"


class PDFExportWithDataRequest(BaseModel):
    """Request body for PDF export with frontend data included"""
    # Configuration options
    frame_number: Optional[int] = None
    include_heatmap: bool = True
    heatmap_colormap: str = "turbo"
    heatmap_alpha: float = 0.6
    title: str = "Badminton Video Analysis Report"
    
    # Data from frontend (same structure as AnalysisResult)
    duration: float = 0
    fps: float = 30
    total_frames: int = 0
    processed_frames: int = 0
    video_width: int = 1920
    video_height: int = 1080
    players: List[Dict[str, Any]] = []
    shuttle: Optional[Dict[str, Any]] = None
    court_detection: Optional[Dict[str, Any]] = None
    shuttle_analytics: Optional[Dict[str, Any]] = None
    player_zone_analytics: Optional[Dict[str, Any]] = None


@app.get("/api/export/pdf/{video_id}")
async def export_pdf_report(
    video_id: str,
    frame_number: Optional[int] = None,
    include_heatmap: bool = True,
    heatmap_colormap: str = "turbo",
    heatmap_alpha: float = 0.6
):
    """
    Generate and download a PDF report for the video analysis.
    
    This endpoint creates a professional PDF report containing:
    - Video frame with heatmap overlay showing player positions
    - Video summary (duration, frames, players detected, FPS)
    - Movement analysis (total distance, average/max speed)
    - Per-player statistics
    - Shuttle/shot analytics (if available)
    - Court detection information (if available)
    - Player zone coverage analytics (if available)
    
    Args:
        video_id: The video ID to generate report for
        frame_number: Specific frame to use for heatmap visualization (default: middle frame)
        include_heatmap: Whether to include the heatmap visualization
        heatmap_colormap: Colormap for heatmap (turbo, parula, inferno, viridis, plasma, hot)
        heatmap_alpha: Opacity of heatmap overlay (0.0-1.0)
    
    Returns:
        PDF file download
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found. Run analysis first.")
    
    # Find the video file
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Original video file not found")
    
    video_path = video_files[0]
    
    # Get analysis result
    result = analysis_results[video_id]
    
    # Create PDF export configuration
    config = PDFExportConfig(
        include_heatmap=include_heatmap,
        heatmap_colormap=heatmap_colormap,
        heatmap_alpha=heatmap_alpha,
        frame_number=frame_number,
        include_player_stats=True,
        include_shuttle_stats=True,
        include_court_info=True,
        include_speed_stats=True,
        title="Badminton Video Analysis Report"
    )
    
    try:
        # Generate PDF
        print(f"[PDF EXPORT] Generating PDF report for video {video_id}")
        pdf_bytes = generate_pdf_report(result, video_path, config)
        print(f"[PDF EXPORT] PDF generated successfully ({len(pdf_bytes)} bytes)")
        
        # Return as file download
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=badminton_analysis_{video_id[:8]}.pdf"
            }
        )
    except Exception as e:
        print(f"[PDF EXPORT] Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@app.post("/api/export/pdf/{video_id}")
async def export_pdf_report_with_config(video_id: str, request: PDFExportRequest):
    """
    Generate and download a PDF report with custom configuration.
    
    This endpoint accepts a JSON body with detailed configuration options
    for PDF generation. Use GET /api/export/pdf/{video_id} for quick export
    with default settings.
    
    Args:
        video_id: The video ID to generate report for
        request: PDF export configuration
    
    Returns:
        PDF file download
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found. Run analysis first.")
    
    # Find the video file
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Original video file not found")
    
    video_path = video_files[0]
    
    # Get analysis result
    result = analysis_results[video_id]
    
    # Create PDF export configuration from request
    config = PDFExportConfig(
        include_heatmap=request.include_heatmap,
        heatmap_colormap=request.heatmap_colormap,
        heatmap_alpha=request.heatmap_alpha,
        frame_number=request.frame_number,
        include_player_stats=request.include_player_stats,
        include_shuttle_stats=request.include_shuttle_stats,
        include_court_info=request.include_court_info,
        include_speed_stats=request.include_speed_stats,
        title=request.title
    )
    
    try:
        # Generate PDF
        print(f"[PDF EXPORT] Generating PDF report for video {video_id} (custom config)")
        pdf_bytes = generate_pdf_report(result, video_path, config)
        print(f"[PDF EXPORT] PDF generated successfully ({len(pdf_bytes)} bytes)")
        
        # Return as file download
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=badminton_analysis_{video_id[:8]}.pdf"
            }
        )
    except Exception as e:
        print(f"[PDF EXPORT] Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@app.post("/api/export/pdf/{video_id}/with-data")
async def export_pdf_report_with_frontend_data(video_id: str, request: PDFExportWithDataRequest):
    """
    Generate PDF report using data provided by the frontend.
    
    This endpoint is designed to export exactly what the frontend is displaying,
    which may include recalculated speeds that are more accurate than the
    initially stored values.
    
    Args:
        video_id: The video ID (used to find the video file for heatmap)
        request: PDF export configuration with frontend data
    
    Returns:
        PDF file download
    """
    # Find the video file (still needed for heatmap frame extraction)
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Original video file not found")
    
    video_path = video_files[0]
    
    # Build result dictionary from frontend data
    # This overrides stored data with what the frontend is showing
    result = {
        "video_id": video_id,
        "duration": request.duration,
        "fps": request.fps,
        "total_frames": request.total_frames,
        "processed_frames": request.processed_frames,
        "video_width": request.video_width,
        "video_height": request.video_height,
        "players": request.players,
        "shuttle": request.shuttle,
        "court_detection": request.court_detection,
        "shuttle_analytics": request.shuttle_analytics,
        "player_zone_analytics": request.player_zone_analytics,
        # Get skeleton_data from stored results for heatmap (frontend doesn't have this)
        "skeleton_data": analysis_results.get(video_id, {}).get("skeleton_data", []) if video_id in analysis_results else []
    }
    
    # Create PDF export configuration from request
    config = PDFExportConfig(
        include_heatmap=request.include_heatmap,
        heatmap_colormap=request.heatmap_colormap,
        heatmap_alpha=request.heatmap_alpha,
        frame_number=request.frame_number,
        include_player_stats=True,
        include_shuttle_stats=True,
        include_court_info=True,
        include_speed_stats=True,
        title=request.title
    )
    
    try:
        # Generate PDF using frontend data
        print(f"[PDF EXPORT] Generating PDF report for video {video_id} (with frontend data)")
        print(f"  - Duration: {request.duration}s")
        print(f"  - Players: {len(request.players)}")
        for i, p in enumerate(request.players):
            print(f"    - Player {p.get('player_id', i+1)}: distance={p.get('total_distance', 0):.1f}m, avg_speed={p.get('avg_speed', 0):.1f}km/h")
        
        pdf_bytes = generate_pdf_report(result, video_path, config)
        print(f"[PDF EXPORT] PDF generated successfully ({len(pdf_bytes)} bytes)")
        
        # Return as file download
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=badminton_analysis_{video_id[:8]}.pdf"
            }
        )
    except Exception as e:
        print(f"[PDF EXPORT] Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@app.get("/api/export/pdf/preview/{video_id}")
async def get_pdf_export_preview(video_id: str):
    """
    Get a preview of what data will be included in the PDF export.
    
    This is useful for the frontend to show a preview dialog before
    triggering the actual PDF generation.
    
    Args:
        video_id: The video ID to preview
    
    Returns:
        Summary of data that will be included in the PDF
    """
    if video_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Video results not found")
    
    result = analysis_results[video_id]
    
    # Calculate summary stats
    players = result.get("players", [])
    total_distance = sum(p.get("total_distance", 0) for p in players)
    avg_speed = sum(p.get("avg_speed", 0) for p in players) / max(len(players), 1)
    max_speed = max((p.get("max_speed", 0) for p in players), default=0)
    
    shuttle = result.get("shuttle")
    court_detection = result.get("court_detection")
    shuttle_analytics = result.get("shuttle_analytics")
    player_zone_analytics = result.get("player_zone_analytics")
    
    return {
        "video_id": video_id,
        "preview": {
            "video_summary": {
                "duration_seconds": result.get("duration", 0),
                "total_frames": result.get("total_frames", 0),
                "processed_frames": result.get("processed_frames", 0),
                "fps": result.get("fps", 30),
                "video_dimensions": f"{result.get('video_width', 0)}x{result.get('video_height', 0)}"
            },
            "players_detected": len(players),
            "movement_summary": {
                "total_distance_m": round(total_distance, 2),
                "avg_speed_kmh": round(avg_speed, 1),
                "max_speed_kmh": round(max_speed, 1)
            },
            "shuttle_data_available": shuttle is not None,
            "shuttle_analytics_available": shuttle_analytics is not None,
            "court_detected": court_detection is not None and court_detection.get("detected", False),
            "zone_analytics_available": player_zone_analytics is not None,
            "skeleton_frames_available": len(result.get("skeleton_data", [])) > 0
        },
        "export_options": {
            "colormaps": ["turbo", "parula", "inferno", "viridis", "plasma", "hot"],
            "recommended_colormap": "turbo",
            "default_heatmap_alpha": 0.6
        },
        "status": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
