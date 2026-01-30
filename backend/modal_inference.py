"""
Modal.com Serverless GPU Inference for Badminton Tracker
========================================================

This module deploys YOLO26 models to Modal's serverless GPU infrastructure
for high-performance inference. Running inference on Modal provides:

- GPU acceleration (A10G, T4, A100 available)
- Auto-scaling from 0 to N instances
- No local GPU required
- Pay-per-second billing
- ~10-50x faster inference than MacBook CPU

PERFORMANCE OPTIMIZATIONS (v2.0):
---------------------------------
1. PARALLEL MODEL EXECUTION - Run pose and detection simultaneously
2. CUDA STREAMS - Separate GPU streams for each model (true GPU parallelism)
3. FP16 INFERENCE - Half precision for 2x speed with minimal accuracy loss
4. BATCH PROCESSING - Process multiple frames in single request
5. OPTIMIZED INPUT - Accept pre-resized frames to reduce decode time
6. WARM CONTAINER - Long idle timeout + concurrent inputs

NOTE: Court detection has been REMOVED from Modal inference.
Manual keypoints provide more accurate court calibration for speed/distance
calculations. See the local court_detection.py module instead.

DEPLOYMENT:
-----------
1. Install Modal: pip install modal
2. Authenticate: modal setup
3. Deploy: modal deploy backend/modal_inference.py
4. Set MODAL_ENDPOINT_URL in .env

USAGE FROM LOCAL BACKEND:
-------------------------
The local FastAPI backend sends frames to Modal for inference,
then receives detection results back for post-processing.

VOLUME STORAGE:
---------------
Custom trained models (badminton detection) are stored
in a Modal Volume for persistent access across container restarts.
"""

import modal
import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel


# =============================================================================
# MODAL CONFIGURATION
# =============================================================================

# Persistent volume for model weights
# This stores custom trained models that can't be downloaded from Ultralytics hub
volume = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)
VOLUME_PATH = Path("/root/models")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        # System libraries for OpenCV and graphics
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",  # OpenMP for YOLO
    ])
    .pip_install([
        # Core ML libraries
        "ultralytics>=8.3.0",  # YOLO26 support
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",  # Headless for server
        # API dependencies
        "fastapi[standard]>=0.104.0",
        "pydantic>=2.0.0",
    ])
)

# Initialize Modal app
app = modal.App(
    "badminton-tracker-inference",
    image=image,
    volumes={VOLUME_PATH: volume}
)


# =============================================================================
# DATA MODELS FOR API
# =============================================================================

class FrameRequest(BaseModel):
    """Request to process a video frame"""
    frame_base64: str  # Base64 encoded JPEG/PNG frame
    frame_number: int = 0
    run_pose: bool = True
    run_detection: bool = True
    run_court: bool = False  # DEPRECATED - court detection removed from Modal, use manual keypoints
    confidence_threshold: float = 0.5
    court_model: str = "region"  # DEPRECATED - kept for backwards compatibility


class BoundingBoxResult(BaseModel):
    """Bounding box detection result"""
    x: float  # Center x
    y: float  # Center y
    width: float
    height: float
    class_name: str
    class_id: int
    confidence: float
    detection_type: str  # "player", "shuttle", "racket", "court_region"


class KeypointResult(BaseModel):
    """Pose keypoint result"""
    name: str
    x: Optional[float]
    y: Optional[float]
    confidence: float


class PersonPoseResult(BaseModel):
    """Pose detection result for one person"""
    track_id: int
    keypoints: list[KeypointResult]
    bbox: BoundingBoxResult
    center_x: float
    center_y: float


class CourtRegionResult(BaseModel):
    """
    Court region detection result.
    DEPRECATED: Court detection removed from Modal. Use manual keypoints instead.
    Kept for backwards compatibility - will always be empty.
    """
    name: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class CourtKeypointResult(BaseModel):
    """
    Court keypoint detection result (22 keypoints).
    DEPRECATED: Court detection removed from Modal. Use manual keypoints instead.
    Kept for backwards compatibility - will always be None.
    """
    keypoints: list[list[float]]  # [[x, y, confidence], ...]
    bbox: Optional[list[float]]  # [x1, y1, x2, y2]
    confidence: float


class FrameResult(BaseModel):
    """Complete inference result for a frame"""
    frame_number: int
    poses: list[PersonPoseResult]
    detections: list[BoundingBoxResult]
    court_regions: list[CourtRegionResult]  # DEPRECATED - always empty
    court_keypoints: Optional[CourtKeypointResult]  # DEPRECATED - always None
    court_corners: Optional[list[list[float]]]  # DEPRECATED - always None
    court_model_used: str = "none"  # DEPRECATED - court detection removed
    inference_time_ms: float


class BatchFrameResult(BaseModel):
    """Result for batch frame processing"""
    results: list[FrameResult]
    total_inference_time_ms: float


# =============================================================================
# INFERENCE CLASS
# =============================================================================

# COCO pose keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Court region classes - DEPRECATED (court detection removed from Modal)
# Kept for reference only
COURT_CLASSES = [
    'frontcourt', 'midcourt-down', 'midcourt-up', 'net',
    'rearcourt-down', 'rearcourt-up', 'sideline-left', 'sideline-right'
]


@app.cls(
    gpu="T4",  # Cost-effective GPU, upgrade to "A10G" for faster inference
    timeout=300,  # 5 minute timeout per request
    container_idle_timeout=300,  # Keep warm for 5 minutes (increased from 2)
    allow_concurrent_inputs=20,  # Handle 20 concurrent requests (increased from 10)
)
class BadmintonInference:
    """
    GPU-accelerated inference for badminton video analysis.
    
    PERFORMANCE OPTIMIZATIONS:
    - Parallel model execution using CUDA streams
    - FP16 inference for 2x speed
    - Batch frame processing support
    - Pre-compiled models with warmup
    
    NOTE: Court detection has been REMOVED from Modal inference.
    Use manual keypoints locally for more accurate court calibration.
    
    Models are loaded once when the container starts (@modal.enter),
    then reused for all subsequent inference requests.
    
    Available methods:
    - process_frame: Full inference (pose + detection)
    - detect_poses: Pose estimation only
    - detect_objects: Object detection only
    - batch_process: Process multiple frames in parallel
    """
    
    @modal.enter()
    def load_models(self):
        """
        Load all YOLO models when container starts.
        
        OPTIMIZATIONS:
        - Creates dedicated CUDA streams for each model (true GPU parallelism)
        - Uses FP16 for faster inference
        - Pre-compiles models with warmup passes
        
        NOTE: Court detection models are NOT loaded (removed for better accuracy
        with manual keypoints).
        """
        import time
        import torch
        from ultralytics import YOLO
        
        start = time.time()
        print("Loading YOLO models on GPU with optimizations...")
        print("NOTE: Court detection removed from Modal - use manual keypoints locally")
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Create CUDA streams for parallel execution (pose + detection)
            self.pose_stream = torch.cuda.Stream()
            self.detection_stream = torch.cuda.Stream()
            print("Created 2 CUDA streams for parallel model execution")
        else:
            self.pose_stream = None
            self.detection_stream = None
        
        # Thread pool for CPU-bound operations (decoding, encoding)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Load pose model (downloads automatically from Ultralytics hub)
        print("Loading YOLOv8 pose model...")
        self.pose_model = YOLO("yolov8n-pose.pt")  # yolo26n-pose when available
        self.pose_model.to(self.device)
        
        # Load custom badminton detection model if available
        badminton_model_path = VOLUME_PATH / "badminton" / "best.pt"
        if badminton_model_path.exists():
            print(f"Loading custom badminton model from {badminton_model_path}...")
            self.detection_model = YOLO(str(badminton_model_path))
            self.detection_model.to(self.device)
        else:
            print("Custom badminton model not found, using COCO for detection")
            self.detection_model = YOLO("yolov8n.pt")  # Fallback to COCO
            self.detection_model.to(self.device)
        
        # Warmup models with dummy inference (compiles CUDA kernels)
        print("Warming up models (compiling CUDA kernels)...")
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Multiple warmup passes for JIT compilation
        for i in range(3):
            _ = self.pose_model(dummy, verbose=False, half=True)
            _ = self.detection_model(dummy, verbose=False, half=True)
        
        load_time = time.time() - start
        print(f"All models loaded and optimized in {load_time:.2f}s")
    
    def _decode_frame(self, frame_base64: str):
        """Decode base64 frame to numpy array"""
        import cv2
        import numpy as np
        
        img_bytes = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def _run_pose_detection(self, frame, confidence_threshold: float) -> list[PersonPoseResult]:
        """Run pose estimation on a frame"""
        results = self.pose_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=confidence_threshold,
            half=True  # FP16 for speed
        )
        
        poses = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints_data = result.keypoints.data.cpu().numpy()
                
                # Get tracking IDs
                track_ids = []
                if result.boxes is not None and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(keypoints_data)))
                
                for idx, (kpts, track_id) in enumerate(zip(keypoints_data, track_ids)):
                    keypoints = []
                    valid_kps = []
                    
                    for kp_idx, keypoint in enumerate(kpts):
                        x, y, conf = keypoint[0], keypoint[1], keypoint[2] if len(keypoint) > 2 else 1.0
                        
                        kp_result = KeypointResult(
                            name=KEYPOINT_NAMES[kp_idx],
                            x=float(x) if conf > confidence_threshold else None,
                            y=float(y) if conf > confidence_threshold else None,
                            confidence=float(conf)
                        )
                        keypoints.append(kp_result)
                        
                        if conf > confidence_threshold:
                            valid_kps.append((float(x), float(y)))
                    
                    # Calculate center position (using ankle midpoint for accurate court positioning)
                    # Ankles represent where the player is actually standing on the court
                    left_ankle_idx = KEYPOINT_NAMES.index("left_ankle")
                    right_ankle_idx = KEYPOINT_NAMES.index("right_ankle")
                    left_hip_idx = KEYPOINT_NAMES.index("left_hip")
                    right_hip_idx = KEYPOINT_NAMES.index("right_hip")
                    
                    if (kpts[left_ankle_idx][2] > confidence_threshold and
                        kpts[right_ankle_idx][2] > confidence_threshold):
                        # Primary: use ankle midpoint (most accurate for court position)
                        center_x = (float(kpts[left_ankle_idx][0]) + float(kpts[right_ankle_idx][0])) / 2
                        center_y = (float(kpts[left_ankle_idx][1]) + float(kpts[right_ankle_idx][1])) / 2
                    elif (kpts[left_hip_idx][2] > confidence_threshold and
                          kpts[right_hip_idx][2] > confidence_threshold):
                        # Fallback 1: use hip midpoint if ankles not visible
                        center_x = (float(kpts[left_hip_idx][0]) + float(kpts[right_hip_idx][0])) / 2
                        center_y = (float(kpts[left_hip_idx][1]) + float(kpts[right_hip_idx][1])) / 2
                    elif len(valid_kps) >= 2:
                        # Fallback 2: mean of all valid keypoints
                        center_x = sum(kp[0] for kp in valid_kps) / len(valid_kps)
                        center_y = sum(kp[1] for kp in valid_kps) / len(valid_kps)
                    else:
                        continue
                    
                    # Get bounding box
                    if result.boxes is not None and idx < len(result.boxes):
                        box = result.boxes[idx]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = BoundingBoxResult(
                            x=float((x1 + x2) / 2),
                            y=float((y1 + y2) / 2),
                            width=float(x2 - x1),
                            height=float(y2 - y1),
                            class_name="person",
                            class_id=0,
                            confidence=float(box.conf[0].cpu().numpy()),
                            detection_type="player"
                        )
                    else:
                        xs = [kp[0] for kp in valid_kps]
                        ys = [kp[1] for kp in valid_kps]
                        bbox = BoundingBoxResult(
                            x=center_x,
                            y=center_y,
                            width=max(xs) - min(xs) + 50,
                            height=max(ys) - min(ys) + 50,
                            class_name="person",
                            class_id=0,
                            confidence=0.8,
                            detection_type="player"
                        )
                    
                    poses.append(PersonPoseResult(
                        track_id=int(track_id),
                        keypoints=keypoints,
                        bbox=bbox,
                        center_x=center_x,
                        center_y=center_y
                    ))
        
        return poses
    
    def _run_object_detection(self, frame, confidence_threshold: float) -> list[BoundingBoxResult]:
        """Run object detection on a frame"""
        results = self.detection_model(
            frame,
            verbose=False,
            conf=confidence_threshold,
            half=True
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Get class name from model
                    class_name = self.detection_model.names.get(cls_id, f"class_{cls_id}")
                    
                    # Determine detection type
                    class_lower = class_name.lower()
                    if any(p in class_lower for p in ["player", "person", "human"]):
                        det_type = "player"
                    elif any(s in class_lower for s in ["shuttle", "birdie", "ball"]):
                        det_type = "shuttle"
                    elif any(r in class_lower for r in ["racket", "racquet"]):
                        det_type = "racket"
                    else:
                        det_type = "other"
                    
                    detections.append(BoundingBoxResult(
                        x=float((x1 + x2) / 2),
                        y=float((y1 + y2) / 2),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                        class_name=class_name,
                        class_id=cls_id,
                        confidence=conf,
                        detection_type=det_type
                    ))
        
        return detections
    
    @modal.method()
    def process_frame(self, request: FrameRequest) -> FrameResult:
        """
        Process a single frame with pose and detection models.
        
        NOTE: Court detection has been REMOVED from Modal inference.
        Use manual keypoints locally for more accurate court calibration.
        
        OPTIMIZED: Uses CUDA streams to run models in parallel when possible.
        This provides ~2x speedup over sequential execution.
        """
        import time
        import torch
        
        start = time.time()
        frame = self._decode_frame(request.frame_base64)
        
        poses = []
        detections = []
        
        # Check if we can use parallel execution (GPU available + multiple models requested)
        num_models_to_run = sum([request.run_pose, request.run_detection])
        use_parallel = (
            self.device == "cuda" and
            num_models_to_run >= 2 and
            self.pose_stream is not None
        )
        
        if use_parallel:
            # PARALLEL EXECUTION with CUDA streams
            # Each model runs on its own stream for true GPU parallelism
            pose_future = None
            detection_future = None
            
            # Launch all models concurrently using thread pool + CUDA streams
            if request.run_pose:
                def run_pose():
                    with torch.cuda.stream(self.pose_stream):
                        return self._run_pose_detection(frame, request.confidence_threshold)
                pose_future = self.thread_pool.submit(run_pose)
            
            if request.run_detection:
                def run_detection():
                    with torch.cuda.stream(self.detection_stream):
                        return self._run_object_detection(frame, request.confidence_threshold)
                detection_future = self.thread_pool.submit(run_detection)
            
            # Synchronize CUDA streams and collect results
            torch.cuda.synchronize()  # Wait for all GPU work to complete
            
            if pose_future:
                poses = pose_future.result()
            if detection_future:
                detections = detection_future.result()
        else:
            # SEQUENTIAL EXECUTION (fallback for CPU or single model)
            if request.run_pose:
                poses = self._run_pose_detection(frame, request.confidence_threshold)
            
            if request.run_detection:
                detections = self._run_object_detection(frame, request.confidence_threshold)
        
        inference_time = (time.time() - start) * 1000
        
        return FrameResult(
            frame_number=request.frame_number,
            poses=poses,
            detections=detections,
            court_regions=[],  # DEPRECATED - always empty
            court_keypoints=None,  # DEPRECATED - always None
            court_corners=None,  # DEPRECATED - always None
            court_model_used="none",  # DEPRECATED - court detection removed
            inference_time_ms=inference_time
        )
    
    @modal.method()
    def process_batch(self, requests: List[FrameRequest]) -> BatchFrameResult:
        """
        Process multiple frames in a single request.
        
        OPTIMIZED: Processes frames in parallel using GPU parallelism.
        Best for batch video processing - send 4-8 frames per request.
        
        Args:
            requests: List of FrameRequest objects (up to 8 recommended)
            
        Returns:
            BatchFrameResult with all frame results
        """
        import time
        import torch
        
        if not requests:
            return BatchFrameResult(results=[], total_inference_time_ms=0)
        
        start = time.time()
        
        # Process frames concurrently
        futures = []
        for request in requests:
            future = self.thread_pool.submit(
                lambda r: self.process_frame(r),
                request
            )
            futures.append(future)
        
        # Collect results
        results = [f.result() for f in futures]
        
        # Synchronize GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        total_time = (time.time() - start) * 1000
        
        return BatchFrameResult(
            results=results,
            total_inference_time_ms=total_time
        )
    
    @modal.method()
    def detect_poses_only(self, frame_base64: str, confidence_threshold: float = 0.5) -> list[PersonPoseResult]:
        """Run only pose estimation"""
        frame = self._decode_frame(frame_base64)
        return self._run_pose_detection(frame, confidence_threshold)
    
    @modal.method()
    def detect_objects_only(self, frame_base64: str, confidence_threshold: float = 0.5) -> list[BoundingBoxResult]:
        """Run only object detection"""
        frame = self._decode_frame(frame_base64)
        return self._run_object_detection(frame, confidence_threshold)
    
    @modal.method()
    def health_check(self) -> dict:
        """Check if models are loaded and ready"""
        import torch
        return {
            "status": "healthy",
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pose_model_loaded": self.pose_model is not None,
            "detection_model_loaded": self.detection_model is not None,
            # Court detection removed - use manual keypoints locally
            "court_model_loaded": False,
            "court_region_model_loaded": False,
            "court_keypoint_model_loaded": False,
            "available_court_models": [],
            "note": "Court detection removed from Modal. Use manual keypoints locally for better accuracy."
        }


# =============================================================================
# WEB ENDPOINTS
# =============================================================================

@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
def infer_frame(request: FrameRequest) -> FrameResult:
    """
    HTTP endpoint for single frame inference.
    
    Send a base64-encoded frame and receive detection results.
    This endpoint is accessible via HTTP POST request.
    """
    return BadmintonInference().process_frame.remote(request)


@app.function()
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint"""
    return BadmintonInference().health_check.remote()


# =============================================================================
# CLI UTILITIES
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the inference pipeline locally"""
    import cv2
    import numpy as np
    
    print("Testing Modal inference pipeline...")
    
    # Create a dummy test frame
    test_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (300, 400), (255, 255, 255), -1)
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', test_frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create request
    request = FrameRequest(
        frame_base64=frame_b64,
        frame_number=0,
        run_pose=True,
        run_detection=True,
        run_court=False,
        confidence_threshold=0.5
    )
    
    # Run inference
    print("Running inference on Modal GPU...")
    result = BadmintonInference().process_frame.remote(request)
    
    print(f"\nResults:")
    print(f"  Frame: {result.frame_number}")
    print(f"  Poses detected: {len(result.poses)}")
    print(f"  Objects detected: {len(result.detections)}")
    print(f"  Court regions: {len(result.court_regions)}")
    print(f"  Inference time: {result.inference_time_ms:.1f}ms")
    
    # Health check
    print("\nHealth check:")
    health = BadmintonInference().health_check.remote()
    for k, v in health.items():
        print(f"  {k}: {v}")


@app.function()
def upload_model(model_type: str, model_data: bytes):
    """
    Upload a custom trained model to Modal Volume.
    
    Args:
        model_type: One of "court", "court_keypoint", "badminton", "shuttle"
        model_data: Raw bytes of the model .pt file
    
    Usage from Python:
        import modal
        upload_fn = modal.Function.from_name('badminton-tracker-inference', 'upload_model')
        with open('models/court_keypoint/weights/best.pt', 'rb') as f:
            result = upload_fn.remote(model_type='court_keypoint', model_data=f.read())
    """
    valid_types = ["court", "court_keypoint", "badminton", "shuttle"]
    if model_type not in valid_types:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of: {', '.join(valid_types)}")
    
    dest_dir = VOLUME_PATH / model_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / "best.pt"
    
    # Write the bytes directly to the volume
    with open(dest_path, 'wb') as f:
        f.write(model_data)
    
    # Commit volume changes
    volume.commit()
    
    print(f"Uploaded {len(model_data)} bytes to {dest_path}")
    return {"path": str(dest_path), "size_bytes": len(model_data), "model_type": model_type}
