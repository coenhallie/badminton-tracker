"""
Modal Client for Remote YOLO Inference
======================================

This module provides a client class to call Modal's serverless GPU
inference endpoints from the local FastAPI backend.

NOTE: Court detection has been REMOVED from Modal inference.
Manual keypoints provide more accurate court calibration for speed/distance
calculations. Use the local court_detection.py module instead.

ARCHITECTURE:
-------------
Local Backend (FastAPI)          Modal (Serverless GPU)
        |                               |
        |  1. Encode frame (base64)     |
        | ----------------------------> |
        |                               | 2. Run YOLO inference on GPU
        |  3. Return detections         |    (pose + object detection only)
        | <---------------------------- |
        |                               |
        | 4. Post-process locally       |
        |    (smoothing, analytics,     |
        |     court detection)          |

CONFIGURATION:
--------------
Set these environment variables in .env:

    # Enable Modal inference (set to "true" to use Modal, "false" for local)
    USE_MODAL_INFERENCE=true
    
    # Modal endpoint URL (get from `modal deploy` output)
    MODAL_ENDPOINT_URL=https://your-workspace--badminton-tracker-inference.modal.run
    
    # Optional: API key if you set one
    MODAL_API_KEY=

USAGE:
------
    from modal_client import get_modal_client, ModalInferenceClient
    
    client = get_modal_client()
    
    if client.is_available:
        # Process a single frame
        result = await client.process_frame(frame, frame_number=0)
        
        # Or process multiple frames in batch
        results = await client.process_frames_batch(frames)
"""

import os
import base64
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import httpx

# Configure logging
logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (seconds)
DEFAULT_TIMEOUT = 30.0

# Max frames to process in a single batch request
MAX_BATCH_SIZE = 8  # Reduced to 8 for better GPU memory usage

# Frame encoding optimization settings
INFERENCE_JPEG_QUALITY = 75  # Reduced from 85 for faster transfer (minimal quality impact for inference)
INFERENCE_MAX_DIMENSION = 1280  # Resize frames larger than this for faster inference


@dataclass
class ModalConfig:
    """Configuration for Modal inference client"""
    enabled: bool = False
    endpoint_url: str = ""
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "ModalConfig":
        """Load configuration from environment variables"""
        return cls(
            enabled=os.getenv("USE_MODAL_INFERENCE", "false").lower() == "true",
            endpoint_url=os.getenv("MODAL_ENDPOINT_URL", ""),
            api_key=os.getenv("MODAL_API_KEY", ""),
            timeout=float(os.getenv("MODAL_TIMEOUT", str(DEFAULT_TIMEOUT))),
            max_retries=int(os.getenv("MODAL_MAX_RETRIES", "3")),
        )


@dataclass
class InferenceResult:
    """
    Result from Modal inference.
    
    NOTE: Court detection has been REMOVED from Modal inference.
    court_regions, court_keypoints, court_corners are kept for backwards
    compatibility but will always be empty/None.
    """
    frame_number: int
    poses: List[Dict[str, Any]]
    detections: List[Dict[str, Any]]
    court_regions: List[Dict[str, Any]]  # DEPRECATED - always empty
    court_keypoints: Optional[Dict[str, Any]]  # DEPRECATED - always None
    court_corners: Optional[List[List[float]]]  # DEPRECATED - always None
    court_model_used: str  # DEPRECATED - always "none"
    inference_time_ms: float
    scale_factor: float = 1.0  # For coordinate remapping when frame was resized
    
    @classmethod
    def from_dict(cls, data: Dict, scale_factor: float = 1.0) -> "InferenceResult":
        """Create from API response dictionary"""
        return cls(
            frame_number=data.get("frame_number", 0),
            poses=data.get("poses", []),
            detections=data.get("detections", []),
            court_regions=[],  # Court detection removed - always empty
            court_keypoints=None,  # Court detection removed - always None
            court_corners=None,  # Court detection removed - always None
            court_model_used="none",  # Court detection removed
            inference_time_ms=data.get("inference_time_ms", 0.0),
            scale_factor=scale_factor,
        )
    
    def _scale_coordinate(self, value: Optional[float]) -> Optional[float]:
        """Scale a coordinate back to original frame size"""
        if value is None or self.scale_factor == 1.0:
            return value
        return value / self.scale_factor
    
    def to_skeleton_format(self) -> Dict[str, Any]:
        """
        Convert to the skeleton_data format used by the frontend.
        
        This transforms the Modal response into the same structure
        that the local YOLO processing produces.
        
        OPTIMIZATION: Applies coordinate scaling if frame was resized before inference.
        """
        players = []
        inv_scale = 1.0 / self.scale_factor if self.scale_factor != 1.0 else 1.0
        
        for pose in self.poses:
            # Convert keypoints from Modal format to local format with scaling
            keypoints = []
            for kp in pose.get("keypoints", []):
                x = kp.get("x")
                y = kp.get("y")
                # Scale coordinates back to original frame size
                if inv_scale != 1.0 and x is not None:
                    x = x * inv_scale
                if inv_scale != 1.0 and y is not None:
                    y = y * inv_scale
                keypoints.append({
                    "name": kp.get("name"),
                    "x": x,
                    "y": y,
                    "confidence": kp.get("confidence", 0.0)
                })
            
            # Scale center coordinates
            center_x = pose.get("center_x", 0)
            center_y = pose.get("center_y", 0)
            if inv_scale != 1.0:
                center_x = center_x * inv_scale
                center_y = center_y * inv_scale
            
            players.append({
                "player_id": pose.get("track_id", 0),
                "keypoints": keypoints,
                "center": {
                    "x": center_x,
                    "y": center_y
                },
                "current_speed": 0.0,  # Calculated later in post-processing
                "court_position": None,  # Calculated later with court detection
                "pose": {
                    "pose_type": "standing",  # Classified later
                    "confidence": 0.8
                }
            })
        
        # Convert detections to badminton_detections format with scaling
        badminton_detections = {
            "players": [],
            "shuttlecocks": [],
            "rackets": [],
            "other": []
        }
        
        for det in self.detections:
            # Scale detection coordinates
            x = det.get("x", 0)
            y = det.get("y", 0)
            w = det.get("width", 0)
            h = det.get("height", 0)
            if inv_scale != 1.0:
                x = x * inv_scale
                y = y * inv_scale
                w = w * inv_scale
                h = h * inv_scale
            
            det_entry = {
                "class": det.get("class_name", ""),
                "confidence": det.get("confidence", 0.0),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
            
            det_type = det.get("detection_type", "other")
            if det_type == "player":
                badminton_detections["players"].append(det_entry)
            elif det_type == "shuttle":
                badminton_detections["shuttlecocks"].append(det_entry)
            elif det_type == "racket":
                badminton_detections["rackets"].append(det_entry)
            else:
                badminton_detections["other"].append(det_entry)
        
        # Find best shuttle position for tracking
        shuttle_pos = None
        if badminton_detections["shuttlecocks"]:
            best_shuttle = max(
                badminton_detections["shuttlecocks"],
                key=lambda s: s["confidence"]
            )
            shuttle_pos = (best_shuttle["x"], best_shuttle["y"])
        
        # Court detection removed from Modal - use manual keypoints locally
        return {
            "frame": self.frame_number,
            "timestamp": 0.0,  # Set by caller based on fps
            "players": players,
            "court_detected": False,  # Court detection done locally, not via Modal
            "court_keypoints": None,  # Court detection removed from Modal
            "court_model_used": "none",  # Court detection removed from Modal
            "badminton_detections": badminton_detections,
            "shuttle_position": shuttle_pos,
            "shuttle_speed_kmh": None,
            "inference_time_ms": self.inference_time_ms,
            "inference_source": "modal"
        }


class ModalInferenceClient:
    """
    Client for calling Modal's serverless YOLO inference.
    
    This class handles:
    - Frame encoding and decoding
    - HTTP requests to Modal endpoints
    - Error handling and retries
    - Batch processing for efficiency
    
    Usage:
        client = ModalInferenceClient(config)
        if client.is_available:
            result = await client.process_frame(frame)
    """
    
    def __init__(self, config: Optional[ModalConfig] = None):
        """
        Initialize Modal inference client.
        
        Args:
            config: Modal configuration. If None, loads from environment.
        """
        self.config = config or ModalConfig.from_env()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._health_checked = False
        self._is_healthy = False
    
    @property
    def is_available(self) -> bool:
        """Check if Modal inference is configured and enabled"""
        return (
            self.config.enabled and
            bool(self.config.endpoint_url) and
            self._is_healthy
        )
    
    @property
    def is_enabled(self) -> bool:
        """Check if Modal is enabled in config (may not be healthy yet)"""
        return self.config.enabled and bool(self.config.endpoint_url)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the HTTP client with optimized settings.
        
        OPTIMIZATIONS:
        - HTTP/2 for connection multiplexing
        - Connection pooling for reuse
        - Aggressive timeouts for fast failure
        """
        if self._http_client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,  # Fast connect timeout
                    read=self.config.timeout,
                    write=10.0,
                    pool=5.0
                ),
                headers=headers,
                http2=True,  # Enable HTTP/2 for multiplexing
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0
                )
            )
        return self._http_client
    
    async def close(self):
        """Close the HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Modal inference service is healthy.
        
        Returns health status including GPU availability, loaded models, etc.
        """
        if not self.config.enabled or not self.config.endpoint_url:
            return {"status": "disabled", "reason": "Modal not configured"}
        
        try:
            client = await self._get_client()
            # Modal URL format: {base}-{function_name}.modal.run
            health_url = f"{self.config.endpoint_url}-health.modal.run"
            response = await client.get(health_url)
            
            if response.status_code == 200:
                data = response.json()
                self._is_healthy = data.get("status") == "healthy"
                self._health_checked = True
                return data
            else:
                self._is_healthy = False
                return {
                    "status": "error",
                    "reason": f"HTTP {response.status_code}",
                    "detail": response.text
                }
        except httpx.TimeoutException:
            self._is_healthy = False
            return {"status": "timeout", "reason": "Health check timed out"}
        except Exception as e:
            self._is_healthy = False
            return {"status": "error", "reason": str(e)}
    
    def _resize_for_inference(self, frame: np.ndarray) -> tuple:
        """
        Resize frame if larger than max dimension for faster transfer.
        
        PERFORMANCE: Reduces data transfer by up to 4x for 4K video.
        Returns (resized_frame, scale_factor) for coordinate remapping.
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= INFERENCE_MAX_DIMENSION:
            return frame, 1.0
        
        scale = INFERENCE_MAX_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use INTER_AREA for downscaling (best quality for neural networks)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def _encode_frame(
        self,
        frame: np.ndarray,
        quality: Optional[int] = None,
        resize: bool = True
    ) -> tuple:
        """
        Encode frame to base64 JPEG with optimizations.
        
        OPTIMIZATIONS:
        - Resize large frames to reduce transfer size (up to 4x smaller for 4K)
        - Lower JPEG quality for inference (75 vs 85 - minimal impact on detection)
        - Returns scale factor for coordinate remapping
        
        Args:
            frame: BGR image
            quality: JPEG quality (uses INFERENCE_JPEG_QUALITY if None)
            resize: Whether to resize large frames
            
        Returns:
            (base64_string, scale_factor)
        """
        if quality is None:
            quality = INFERENCE_JPEG_QUALITY
        
        scale = 1.0
        if resize:
            frame, scale = self._resize_for_inference(frame)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8'), scale
    
    async def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        run_pose: bool = True,
        run_detection: bool = True,
        run_court: bool = False,  # DEPRECATED - court detection removed from Modal
        confidence_threshold: float = 0.5,
        resize_frame: bool = True,
        court_model: str = "none"  # DEPRECATED - kept for backwards compatibility
    ) -> InferenceResult:
        """
        Process a single frame via Modal inference.
        
        NOTE: Court detection has been REMOVED from Modal inference.
        Use manual keypoints locally for more accurate court calibration.
        The run_court and court_model parameters are deprecated and ignored.
        
        OPTIMIZATIONS:
        - Frames are resized if larger than 1280px for faster transfer
        - JPEG quality reduced to 75 for smaller payload
        - Coordinates are automatically scaled back to original frame size
        
        Args:
            frame: BGR image (numpy array from cv2)
            frame_number: Frame number in video
            run_pose: Whether to run pose estimation
            run_detection: Whether to run object detection
            run_court: DEPRECATED - ignored, court detection removed from Modal
            confidence_threshold: Minimum detection confidence
            resize_frame: Whether to resize large frames (default: True)
            court_model: DEPRECATED - ignored, court detection removed from Modal
            
        Returns:
            InferenceResult with pose and detection results (coordinates in original frame space)
        """
        if not self.is_enabled:
            raise RuntimeError("Modal inference is not enabled")
        
        # Encode frame with optimizations (resize + compress)
        frame_b64, scale_factor = self._encode_frame(frame, resize=resize_frame)
        
        # Build request - court detection disabled
        request_data = {
            "frame_base64": frame_b64,
            "frame_number": frame_number,
            "run_pose": run_pose,
            "run_detection": run_detection,
            "run_court": False,  # Court detection removed from Modal
            "confidence_threshold": confidence_threshold,
            "court_model": "none"  # Court detection removed from Modal
        }
        
        # Send to Modal - URL format: {base}-{function_name}.modal.run
        client = await self._get_client()
        infer_url = f"{self.config.endpoint_url}-infer-frame.modal.run"
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(
                    infer_url,
                    json=request_data
                )
                
                if response.status_code == 200:
                    # Pass scale_factor to enable coordinate remapping
                    return InferenceResult.from_dict(response.json(), scale_factor=scale_factor)
                elif response.status_code >= 500:
                    # Server error, retry
                    logger.warning(
                        f"Modal server error (attempt {attempt + 1}): {response.status_code}"
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    # Client error, don't retry
                    raise RuntimeError(
                        f"Modal inference failed: HTTP {response.status_code} - {response.text}"
                    )
            except httpx.TimeoutException:
                logger.warning(f"Modal timeout (attempt {attempt + 1})")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (attempt + 1))
        
        raise RuntimeError("Modal inference failed after all retries")
    
    async def process_frames_batch(
        self,
        frames: List[np.ndarray],
        start_frame_number: int = 0,
        **kwargs
    ) -> List[InferenceResult]:
        """
        Process multiple frames in parallel.
        
        This sends all frames concurrently to Modal for faster processing.
        
        Args:
            frames: List of BGR images
            start_frame_number: Frame number of the first frame
            **kwargs: Additional arguments passed to process_frame
            
        Returns:
            List of InferenceResult in same order as input frames
        """
        if not frames:
            return []
        
        # Process in batches to avoid overwhelming Modal
        results = []
        for i in range(0, len(frames), MAX_BATCH_SIZE):
            batch = frames[i:i + MAX_BATCH_SIZE]
            batch_start = start_frame_number + i
            
            # Create tasks for concurrent processing
            tasks = [
                self.process_frame(
                    frame,
                    frame_number=batch_start + j,
                    **kwargs
                )
                for j, frame in enumerate(batch)
            ]
            
            # Wait for all to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Frame {batch_start + j} failed: {result}")
                    # Create empty result for failed frames
                    results.append(InferenceResult(
                        frame_number=batch_start + j,
                        poses=[],
                        detections=[],
                        court_regions=[],  # Always empty (court detection removed)
                        court_keypoints=None,  # Always None (court detection removed)
                        court_corners=None,  # Always None (court detection removed)
                        court_model_used="none",  # Court detection removed
                        inference_time_ms=0.0
                    ))
                else:
                    results.append(result)
        
        return results


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_modal_client: Optional[ModalInferenceClient] = None


def get_modal_client() -> ModalInferenceClient:
    """Get or create the Modal client singleton"""
    global _modal_client
    
    if _modal_client is None:
        _modal_client = ModalInferenceClient()
    
    return _modal_client


async def initialize_modal_client() -> bool:
    """
    Initialize and health-check the Modal client.
    
    Call this at application startup to verify Modal connectivity.
    
    Returns:
        True if Modal is available and healthy, False otherwise
    """
    client = get_modal_client()
    
    if not client.is_enabled:
        logger.info("Modal inference is disabled (USE_MODAL_INFERENCE=false)")
        return False
    
    logger.info(f"Checking Modal health at {client.config.endpoint_url}...")
    health = await client.health_check()
    
    if health.get("status") == "healthy":
        logger.info("Modal inference is healthy and ready")
        logger.info(f"  GPU: {health.get('gpu_name', 'Unknown')}")
        logger.info(f"  Pose model: {'loaded' if health.get('pose_model_loaded') else 'not loaded'}")
        logger.info(f"  Detection model: {'loaded' if health.get('detection_model_loaded') else 'not loaded'}")
        logger.info(f"  Note: Court detection handled locally via manual keypoints")
        return True
    else:
        logger.warning(f"Modal health check failed: {health}")
        return False


# =============================================================================
# HYBRID INFERENCE HELPER
# =============================================================================

class HybridInferenceManager:
    """
    Manages switching between local and Modal inference.
    
    This class provides a unified interface that:
    - Uses Modal when available for GPU acceleration
    - Falls back to local CPU inference when Modal is unavailable
    - Handles errors gracefully with automatic fallback
    
    Usage:
        manager = HybridInferenceManager()
        await manager.initialize()
        
        # Uses Modal if available, otherwise local
        result = await manager.process_frame(frame)
    """
    
    def __init__(self):
        self.modal_client = get_modal_client()
        self.use_modal = False
        self._local_pose_model = None
        self._local_detection_model = None
    
    async def initialize(self) -> str:
        """
        Initialize inference backend.
        
        Returns:
            "modal" if using Modal, "local" if using local inference
        """
        if self.modal_client.is_enabled:
            modal_healthy = await initialize_modal_client()
            if modal_healthy:
                self.use_modal = True
                return "modal"
        
        # Fall back to local inference
        logger.info("Using local inference (Modal not available)")
        self._load_local_models()
        return "local"
    
    def _load_local_models(self):
        """Load local YOLO models for fallback"""
        from ultralytics import YOLO
        
        logger.info("Loading local YOLO models...")
        self._local_pose_model = YOLO("yolo26n-pose.pt")
        self._local_detection_model = YOLO("yolov8n.pt")
        logger.info("Local models loaded")
    
    async def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        court_model: str = "none",  # DEPRECATED - kept for compatibility
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a frame using best available backend.
        
        NOTE: Court detection is done locally, not via Modal.
        The court_model parameter is deprecated and ignored.
        
        Args:
            frame: BGR image (numpy array from cv2)
            frame_number: Frame number in video
            court_model: DEPRECATED - court detection done locally
            **kwargs: Additional arguments for inference
        
        Returns skeleton_data format compatible with frontend.
        """
        if self.use_modal and self.modal_client.is_available:
            try:
                result = await self.modal_client.process_frame(
                    frame, frame_number, **kwargs
                )
                return result.to_skeleton_format()
            except Exception as e:
                logger.warning(f"Modal inference failed, falling back to local: {e}")
        
        # Local inference fallback
        return self._process_local(frame, frame_number, **kwargs)
    
    def _process_local(
        self,
        frame: np.ndarray,
        frame_number: int,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Process frame using local YOLO models"""
        if self._local_pose_model is None:
            self._load_local_models()
        
        # Run pose estimation
        pose_results = self._local_pose_model.track(
            frame, persist=True, verbose=False, conf=confidence_threshold
        )
        
        # COCO keypoint indices
        LEFT_ANKLE_IDX = 15
        RIGHT_ANKLE_IDX = 16
        LEFT_HIP_IDX = 11
        RIGHT_HIP_IDX = 12
        
        players = []
        if pose_results and len(pose_results) > 0:
            result = pose_results[0]
            if result.keypoints is not None:
                for idx, kpts in enumerate(result.keypoints.data.cpu().numpy()):
                    keypoints = []
                    for kp_idx, kp in enumerate(kpts):
                        keypoints.append({
                            "name": f"kp_{kp_idx}",
                            "x": float(kp[0]) if kp[2] > confidence_threshold else None,
                            "y": float(kp[1]) if kp[2] > confidence_threshold else None,
                            "confidence": float(kp[2])
                        })
                    
                    valid_kps = [(kp[0], kp[1]) for kp in kpts if kp[2] > confidence_threshold]
                    if not valid_kps:
                        continue
                    
                    # Calculate center using ankle midpoint (feet) as primary
                    # This matches modal_inference.py for consistent court positioning
                    center_x = None
                    center_y = None
                    
                    if (len(kpts) > RIGHT_ANKLE_IDX and
                        kpts[LEFT_ANKLE_IDX][2] > confidence_threshold and
                        kpts[RIGHT_ANKLE_IDX][2] > confidence_threshold):
                        # Primary: use ankle midpoint (most accurate for court position)
                        center_x = (kpts[LEFT_ANKLE_IDX][0] + kpts[RIGHT_ANKLE_IDX][0]) / 2
                        center_y = (kpts[LEFT_ANKLE_IDX][1] + kpts[RIGHT_ANKLE_IDX][1]) / 2
                    elif (len(kpts) > RIGHT_HIP_IDX and
                          kpts[LEFT_HIP_IDX][2] > confidence_threshold and
                          kpts[RIGHT_HIP_IDX][2] > confidence_threshold):
                        # Fallback 1: use hip midpoint if ankles not visible
                        center_x = (kpts[LEFT_HIP_IDX][0] + kpts[RIGHT_HIP_IDX][0]) / 2
                        center_y = (kpts[LEFT_HIP_IDX][1] + kpts[RIGHT_HIP_IDX][1]) / 2
                    else:
                        # Fallback 2: mean of all valid keypoints
                        center_x = sum(k[0] for k in valid_kps) / len(valid_kps)
                        center_y = sum(k[1] for k in valid_kps) / len(valid_kps)
                    
                    players.append({
                        "player_id": idx,
                        "keypoints": keypoints,
                        "center": {"x": float(center_x), "y": float(center_y)},
                        "current_speed": 0.0,
                        "court_position": None,
                        "pose": {"pose_type": "standing", "confidence": 0.8}
                    })
        
        return {
            "frame": frame_number,
            "timestamp": 0.0,
            "players": players,
            "court_detected": False,
            "badminton_detections": None,
            "shuttle_position": None,
            "shuttle_speed_kmh": None,
            "inference_time_ms": 0.0,
            "inference_source": "local"
        }
    
    @property
    def inference_mode(self) -> str:
        """Get current inference mode"""
        return "modal" if self.use_modal else "local"
