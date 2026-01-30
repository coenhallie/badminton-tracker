"""
Badminton Court Detection Module
Uses locally trained YOLO model for court region detection

PERFORMANCE NOTES:
- Video size normalization is applied before inference for consistent detection
- Manual court keypoints can be provided for more accurate measurements
- The model was trained on 640x640 images; input is resized accordingly
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set to DEBUG for verbose court detection logging
DEBUG_COURT_DETECTION = os.environ.get("DEBUG_COURT_DETECTION", "1") == "1"
if DEBUG_COURT_DETECTION:
    logger.setLevel(logging.DEBUG)

# Standard badminton court dimensions (in meters)
COURT_LENGTH = 13.4  # Full court length
COURT_WIDTH_DOUBLES = 6.1  # Doubles court width
COURT_WIDTH_SINGLES = 5.18  # Singles court width
SERVICE_LINE_DISTANCE = 1.98  # Distance from net to short service line
BACK_BOUNDARY_SERVICE = 0.76  # Distance from back line to long service line (doubles)

# Local model configuration
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "court", "weights", "best.pt")

# Class names from our trained model (8 regions)
COURT_CLASSES = [
    'frontcourt',      # 0
    'midcourt-down',   # 1
    'midcourt-up',     # 2
    'net',             # 3
    'rearcourt-down',  # 4
    'rearcourt-up',    # 5
    'sideline-left',   # 6
    'sideline-right'   # 7
]


@dataclass
class CourtRegion:
    """Represents a detected court region"""
    name: str
    class_id: int
    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y
    confidence: float
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class CourtKeypoint:
    """Represents a derived court keypoint (for API compatibility)"""
    name: str
    x: float
    y: float
    confidence: float
    class_id: int


@dataclass
class ManualCourtKeypoints:
    """
    Manual court keypoints for accurate measurements.
    
    When provided, these override the automatic detection for homography calculation.
    This allows users to click on the four corners of the court for precise
    perspective transformation, resulting in more accurate speed/distance measurements.
    
    All coordinates are in pixel space of the ORIGINAL video frame.
    """
    top_left: Tuple[float, float]  # (x, y) of top-left court corner
    top_right: Tuple[float, float]  # (x, y) of top-right court corner
    bottom_right: Tuple[float, float]  # (x, y) of bottom-right court corner
    bottom_left: Tuple[float, float]  # (x, y) of bottom-left court corner
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for homography calculation"""
        return np.array([
            list(self.top_left),
            list(self.top_right),
            list(self.bottom_right),
            list(self.bottom_left)
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left": list(self.bottom_left)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ManualCourtKeypoints":
        """Create from dictionary (e.g., from API request)"""
        return cls(
            top_left=tuple(data["top_left"]),
            top_right=tuple(data["top_right"]),
            bottom_right=tuple(data["bottom_right"]),
            bottom_left=tuple(data["bottom_left"])
        )


@dataclass
class CourtDetection:
    """Contains complete court detection results"""
    regions: List[CourtRegion]
    keypoints: List[CourtKeypoint]  # Derived from regions for compatibility
    court_corners: Optional[np.ndarray]  # 4 corner points for perspective transform
    homography_matrix: Optional[np.ndarray]  # For pixel-to-court coordinate conversion
    confidence: float
    detected: bool
    # New fields for tracking detection source and frame dimensions
    frame_width: int = 0  # Original frame width
    frame_height: int = 0  # Original frame height
    detection_source: str = "auto"  # "auto" or "manual"
    manual_keypoints: Optional[ManualCourtKeypoints] = None


# Default inference size for YOLO model (matches training size)
DEFAULT_INFERENCE_SIZE = 640


class BadmintonCourtDetector:
    """
    Detects badminton court regions and derives keypoints/corners
    for accurate player position and distance measurements.
    
    Uses locally trained YOLO model that detects:
    - frontcourt, midcourt-down, midcourt-up (court zones)
    - rearcourt-down, rearcourt-up (back court zones)
    - sideline-left, sideline-right (court boundaries)
    - net (center divider)
    
    VIDEO SIZE HANDLING:
    - The model was trained on 640x640 images
    - Input frames are resized to the inference size before detection
    - All coordinates are automatically scaled back to original frame dimensions
    - This ensures consistent detection quality regardless of input resolution
    
    MANUAL KEYPOINTS:
    - Users can provide manual court corner keypoints for precise measurements
    - Manual keypoints override automatic detection for homography calculation
    - This is recommended when automatic detection is unreliable or for maximum accuracy
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        inference_size: int = DEFAULT_INFERENCE_SIZE
    ):
        """
        Initialize the court detector.
        
        Args:
            model_path: Path to trained YOLO model. If None, uses default path.
            confidence_threshold: Minimum confidence for detections
            inference_size: Size to resize frames to before inference (default: 640)
                           Set to None to use original frame size (may affect detection quality)
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.inference_size = inference_size
        self.model = None
        self._last_detection: Optional[CourtDetection] = None
        self._detection_cache: dict[int, CourtDetection] = {}
        
        # Manual keypoints for overriding automatic detection
        self._manual_keypoints: Optional[ManualCourtKeypoints] = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"Court detector initialized with local model: {self.model_path}")
            except Exception as e:
                print(f"Error loading court model: {e}")
                self.model = None
        else:
            print(f"Warning: Court model not found at {self.model_path}")
            print("Run 'python train_court_model.py' to train the model first.")
            self.model = None
    
    def set_manual_keypoints(self, keypoints: Optional[ManualCourtKeypoints]) -> None:
        """
        Set manual court keypoints to override automatic detection.
        
        When manual keypoints are set, the homography calculation will use
        these precise corner positions instead of deriving them from detected regions.
        This results in more accurate player speed and distance measurements.
        
        Args:
            keypoints: ManualCourtKeypoints with the four court corners,
                      or None to clear manual keypoints and use auto-detection
        
        Example:
            detector.set_manual_keypoints(ManualCourtKeypoints(
                top_left=(100, 50),
                top_right=(500, 50),
                bottom_right=(550, 400),
                bottom_left=(50, 400)
            ))
        """
        self._manual_keypoints = keypoints
        
        # Clear cache when keypoints change (need to recalculate homography)
        self._detection_cache.clear()
        self._last_detection = None
        
        if keypoints:
            logger.info(f"Manual keypoints set: {keypoints.to_dict()}")
        else:
            logger.info("Manual keypoints cleared - using auto-detection")
    
    def get_manual_keypoints(self) -> Optional[ManualCourtKeypoints]:
        """Get the currently set manual keypoints, if any."""
        return self._manual_keypoints
    
    def has_manual_keypoints(self) -> bool:
        """Check if manual keypoints are set."""
        return self._manual_keypoints is not None
    
    def validate_manual_keypoints(
        self,
        keypoints: ManualCourtKeypoints,
        frame_width: int,
        frame_height: int
    ) -> Dict:
        """
        Validate manual keypoints for reasonable court shape and size.
        
        This helps detect incorrectly placed keypoints that would cause
        inaccurate speed/distance calculations due to homography miscalibration.
        
        A standard badminton court has:
        - Aspect ratio ~2.2:1 (length/width = 13.4m / 6.1m)
        - Should occupy a significant portion of the video frame
        
        Args:
            keypoints: ManualCourtKeypoints with the four corners
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            
        Returns:
            Dictionary with validation results:
            - valid: bool - whether keypoints are acceptable
            - aspect_ratio: calculated aspect ratio (height/width)
            - expected_ratio: expected ratio (~2.2)
            - ratio_error_pct: percentage error in aspect ratio
            - area_ratio: court area as fraction of frame area
            - pixel_width: court width in pixels
            - pixel_height: court height in pixels
            - warnings: list of warning messages
            - recommendations: list of recommendations for fixing issues
        """
        corners = keypoints.to_numpy()
        
        # Calculate pixel dimensions using corner distances
        # Top edge width
        width_top = np.sqrt(
            (corners[1][0] - corners[0][0])**2 +
            (corners[1][1] - corners[0][1])**2
        )
        # Bottom edge width
        width_bottom = np.sqrt(
            (corners[2][0] - corners[3][0])**2 +
            (corners[2][1] - corners[3][1])**2
        )
        # Left edge height
        height_left = np.sqrt(
            (corners[3][0] - corners[0][0])**2 +
            (corners[3][1] - corners[0][1])**2
        )
        # Right edge height
        height_right = np.sqrt(
            (corners[2][0] - corners[1][0])**2 +
            (corners[2][1] - corners[1][1])**2
        )
        
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        
        # Calculate aspect ratio (should be ~2.2 for badminton court)
        aspect_ratio = avg_height / avg_width if avg_width > 0 else 0
        expected_ratio = COURT_LENGTH / COURT_WIDTH_DOUBLES  # 13.4 / 6.1 ≈ 2.2
        ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else 1.0
        
        # Calculate area ratio (court area vs frame area)
        court_area = avg_width * avg_height
        frame_area = frame_width * frame_height
        area_ratio = court_area / frame_area if frame_area > 0 else 0
        
        # Calculate meters-per-pixel to estimate speed scaling
        meters_per_pixel_width = COURT_WIDTH_DOUBLES / avg_width if avg_width > 0 else 0
        meters_per_pixel_height = COURT_LENGTH / avg_height if avg_height > 0 else 0
        
        # Collect warnings and recommendations
        warnings = []
        recommendations = []
        
        # Check aspect ratio
        ASPECT_TOLERANCE = 0.35  # Allow 35% deviation
        if ratio_error > ASPECT_TOLERANCE:
            warnings.append(
                f"Court aspect ratio ({aspect_ratio:.2f}) differs significantly from "
                f"expected ({expected_ratio:.2f}). Error: {ratio_error*100:.1f}%"
            )
            if aspect_ratio < expected_ratio:
                recommendations.append(
                    "The court appears too wide/short. Ensure you clicked on the full court "
                    "length (baseline to baseline), not just half court."
                )
            else:
                recommendations.append(
                    "The court appears too narrow/tall. Ensure you clicked on the full court "
                    "width (sideline to sideline)."
                )
        
        # Check if court is too small (would cause inflated speeds)
        MIN_AREA_RATIO = 0.05  # Court should be at least 5% of frame
        if area_ratio < MIN_AREA_RATIO:
            warnings.append(
                f"Court area is very small ({area_ratio*100:.1f}% of frame). "
                "This will cause inflated speed calculations."
            )
            recommendations.append(
                "The clicked points are too close together. Make sure to click on the "
                "actual outer corners of the full court, not an inner service box."
            )
        
        # Check if court is too large (might extend beyond visible area)
        MAX_AREA_RATIO = 0.95
        if area_ratio > MAX_AREA_RATIO:
            warnings.append(
                f"Court area is very large ({area_ratio*100:.1f}% of frame). "
                "Some keypoints might be outside the court."
            )
            recommendations.append(
                "The clicked points span almost the entire frame. Ensure they are "
                "on the actual court corners, not outside the court."
            )
        
        # Check for extreme scaling that would cause speed issues
        # At 30fps, moving 1 pixel should not translate to > 0.1m (3 m/s = 10.8 km/h per pixel)
        MAX_METERS_PER_PIXEL = 0.15  # Conservative limit
        if meters_per_pixel_width > MAX_METERS_PER_PIXEL or meters_per_pixel_height > MAX_METERS_PER_PIXEL:
            warnings.append(
                f"Pixel-to-meter scale is very high (width: {meters_per_pixel_width:.4f} m/px, "
                f"height: {meters_per_pixel_height:.4f} m/px). "
                "This will cause severely inflated speed calculations."
            )
            recommendations.append(
                f"The court corners are too close together in pixel space. "
                f"Expected court width is at least {COURT_WIDTH_DOUBLES / MAX_METERS_PER_PIXEL:.0f} pixels, "
                f"but detected only {avg_width:.0f} pixels."
            )
        
        # Check minimum pixel dimensions
        MIN_PIXEL_WIDTH = 100  # Court should be at least 100 pixels wide
        MIN_PIXEL_HEIGHT = 200  # Court should be at least 200 pixels tall
        if avg_width < MIN_PIXEL_WIDTH:
            warnings.append(
                f"Court width ({avg_width:.0f}px) is below minimum ({MIN_PIXEL_WIDTH}px)."
            )
            recommendations.append("Click on court corners that are farther apart horizontally.")
        if avg_height < MIN_PIXEL_HEIGHT:
            warnings.append(
                f"Court height ({avg_height:.0f}px) is below minimum ({MIN_PIXEL_HEIGHT}px)."
            )
            recommendations.append("Click on court corners that are farther apart vertically.")
        
        # Determine overall validity
        is_valid = (
            ratio_error <= ASPECT_TOLERANCE and
            area_ratio >= MIN_AREA_RATIO and
            area_ratio <= MAX_AREA_RATIO and
            avg_width >= MIN_PIXEL_WIDTH and
            avg_height >= MIN_PIXEL_HEIGHT and
            meters_per_pixel_width <= MAX_METERS_PER_PIXEL and
            meters_per_pixel_height <= MAX_METERS_PER_PIXEL
        )
        
        result = {
            "valid": is_valid,
            "aspect_ratio": round(aspect_ratio, 3),
            "expected_ratio": round(expected_ratio, 3),
            "ratio_error_pct": round(ratio_error * 100, 1),
            "area_ratio": round(area_ratio, 4),
            "pixel_width": round(avg_width, 1),
            "pixel_height": round(avg_height, 1),
            "meters_per_pixel_width": round(meters_per_pixel_width, 5),
            "meters_per_pixel_height": round(meters_per_pixel_height, 5),
            "frame_dimensions": {"width": frame_width, "height": frame_height},
            "warnings": [w for w in warnings if w],
            "recommendations": [r for r in recommendations if r],
            "speed_impact": {
                "estimated_1pixel_speed_kmh": round(meters_per_pixel_width * 30 * 3.6, 1),  # 30fps
                "description": (
                    "Moving 1 pixel per frame at 30fps would register as "
                    f"{meters_per_pixel_width * 30 * 3.6:.1f} km/h"
                )
            }
        }
        
        logger.info(f"Manual keypoint validation: valid={is_valid}, "
                   f"aspect_ratio={aspect_ratio:.2f}, area_ratio={area_ratio:.3f}")
        if warnings:
            for w in warnings:
                logger.warning(f"Keypoint validation warning: {w}")
        
        return result
    
    def create_detection_from_manual_keypoints(
        self,
        frame: np.ndarray,
        frame_number: int = 0
    ) -> CourtDetection:
        """
        Create a CourtDetection from manual keypoints.
        
        This bypasses the YOLO model and creates a detection directly from
        the user-provided corner coordinates.
        
        Args:
            frame: The video frame (used for dimensions)
            frame_number: Current frame number
            
        Returns:
            CourtDetection with manually specified corners
        """
        if self._manual_keypoints is None:
            logger.warning("No manual keypoints set")
            return self._empty_detection()
        
        height, width = frame.shape[:2]
        corners = self._manual_keypoints.to_numpy()
        homography = self._calculate_homography(corners)
        
        detection = CourtDetection(
            regions=[],  # No automatic region detection
            keypoints=[],  # No automatic keypoints
            court_corners=corners,
            homography_matrix=homography,
            confidence=1.0,  # Manual = 100% confidence
            detected=True,
            frame_width=width,
            frame_height=height,
            detection_source="manual",
            manual_keypoints=self._manual_keypoints
        )
        
        logger.info(f"[Frame {frame_number}] Created detection from manual keypoints")
        return detection
    
    def detect_court(self, frame: np.ndarray, frame_number: int = 0,
                     use_cache: bool = True, cache_interval: int = 30) -> CourtDetection:
        """
        Detect court regions in a video frame.
        
        VIDEO SIZE HANDLING:
        - If inference_size is set, the frame is resized before detection
        - All coordinates are automatically scaled back to original frame dimensions
        - This ensures consistent detection quality regardless of input resolution
        
        MANUAL KEYPOINTS:
        - If manual keypoints are set, they are used for homography calculation
        - Automatic detection still runs to provide region/keypoint data
        - The homography uses manual corners for maximum accuracy
        
        Args:
            frame: BGR image frame from video
            frame_number: Current frame number for caching
            use_cache: Whether to use cached detection for nearby frames
            cache_interval: Number of frames to reuse cached detection
            
        Returns:
            CourtDetection object with regions, keypoints and transformation matrix
        """
        # Get original frame dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # If manual keypoints are set and we want to skip model inference
        if self._manual_keypoints is not None and self.model is None:
            return self.create_detection_from_manual_keypoints(frame, frame_number)
        
        if self.model is None:
            # No model, but check if we have manual keypoints
            if self._manual_keypoints is not None:
                return self.create_detection_from_manual_keypoints(frame, frame_number)
            logger.warning(f"[Frame {frame_number}] Court model is None - returning empty detection")
            return self._empty_detection()
        
        # Check cache for recent detection
        if use_cache and self._last_detection and self._last_detection.detected:
            cache_frame = max(self._detection_cache.keys()) if self._detection_cache else -999
            if abs(frame_number - cache_frame) < cache_interval:
                logger.debug(f"[Frame {frame_number}] Using cached detection from frame {cache_frame}")
                return self._last_detection
        
        logger.debug(f"[Frame {frame_number}] Running court detection on frame {frame.shape}")
        
        # Calculate scale factors for coordinate conversion
        scale_x = 1.0
        scale_y = 1.0
        
        try:
            # Run inference with explicit image size parameter
            # PERFORMANCE OPTIMIZATION: Using imgsz ensures consistent detection quality
            # regardless of input video resolution
            if self.inference_size:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    verbose=False,
                    imgsz=self.inference_size  # Normalize to training size
                )
                logger.debug(f"[Frame {frame_number}] Inference at size {self.inference_size} "
                           f"(original: {orig_width}x{orig_height})")
            else:
                # Use original frame size (may have variable detection quality)
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                logger.debug(f"[Frame {frame_number}] Inference at original size {orig_width}x{orig_height}")
            
            # Parse regions from results
            # Note: YOLO automatically scales coordinates back to original frame size
            # when using imgsz parameter, so no manual scaling is needed
            regions = self._parse_regions(results)
            
            logger.info(f"[Frame {frame_number}] Detected {len(regions)} court regions:")
            for region in regions:
                logger.info(f"  - {region.name}: bbox=({region.x1:.1f}, {region.y1:.1f}, {region.x2:.1f}, {region.y2:.1f}), "
                           f"center=({region.center_x:.1f}, {region.center_y:.1f}), conf={region.confidence:.3f}")
            
            if len(regions) >= 2:  # Need at least 2 regions to estimate court
                # Derive keypoints from regions
                keypoints = self._derive_keypoints(regions)
                
                # Determine corners: use manual keypoints if available, otherwise auto-detect
                detection_source = "auto"
                if self._manual_keypoints is not None:
                    # Use manual keypoints for precise homography
                    corners = self._manual_keypoints.to_numpy()
                    detection_source = "manual"
                    logger.info(f"[Frame {frame_number}] Using MANUAL court corners:")
                    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                    for name, corner in zip(corner_names, corners):
                        logger.info(f"  - {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
                else:
                    # Calculate court corners from detected regions
                    corners = self._extract_court_corners(regions)
                    
                    if corners is not None:
                        logger.info(f"[Frame {frame_number}] Extracted auto-detected court corners:")
                        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                        for name, corner in zip(corner_names, corners):
                            logger.info(f"  - {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
                    else:
                        logger.warning(f"[Frame {frame_number}] Failed to extract court corners")
                
                # Calculate homography if we have corners
                homography = self._calculate_homography(corners) if corners is not None else None
                
                if homography is not None:
                    logger.debug(f"[Frame {frame_number}] Homography matrix calculated successfully")
                else:
                    logger.warning(f"[Frame {frame_number}] Failed to calculate homography matrix")
                
                # Calculate average confidence
                avg_confidence = np.mean([r.confidence for r in regions])
                
                detection = CourtDetection(
                    regions=regions,
                    keypoints=keypoints,
                    court_corners=corners,
                    homography_matrix=homography,
                    confidence=float(avg_confidence),
                    detected=True,
                    frame_width=orig_width,
                    frame_height=orig_height,
                    detection_source=detection_source,
                    manual_keypoints=self._manual_keypoints
                )
                
                logger.info(f"[Frame {frame_number}] ✓ Court detection successful "
                           f"(confidence: {avg_confidence:.3f}, source: {detection_source})")
            else:
                logger.warning(f"[Frame {frame_number}] Only {len(regions)} regions detected (need at least 2)")
                
                # Even with few regions, if we have manual keypoints, create a valid detection
                if self._manual_keypoints is not None:
                    corners = self._manual_keypoints.to_numpy()
                    homography = self._calculate_homography(corners)
                    detection = CourtDetection(
                        regions=regions,
                        keypoints=[],
                        court_corners=corners,
                        homography_matrix=homography,
                        confidence=0.5,  # Lower confidence since auto-detection failed
                        detected=True,
                        frame_width=orig_width,
                        frame_height=orig_height,
                        detection_source="manual",
                        manual_keypoints=self._manual_keypoints
                    )
                    logger.info(f"[Frame {frame_number}] Using manual keypoints as fallback")
                else:
                    detection = self._empty_detection(orig_width, orig_height)
                    detection.regions = regions
            
            # Update cache
            self._last_detection = detection
            self._detection_cache[frame_number] = detection
            
            # Keep cache size manageable
            if len(self._detection_cache) > 100:
                oldest_keys = sorted(self._detection_cache.keys())[:-50]
                for k in oldest_keys:
                    del self._detection_cache[k]
            
            return detection
            
        except Exception as e:
            logger.error(f"[Frame {frame_number}] Court detection error: {e}", exc_info=True)
            return self._empty_detection(orig_width, orig_height)
    
    def _empty_detection(self, frame_width: int = 0, frame_height: int = 0) -> CourtDetection:
        """Return an empty detection result"""
        return CourtDetection(
            regions=[],
            keypoints=[],
            court_corners=None,
            homography_matrix=None,
            confidence=0.0,
            detected=False,
            frame_width=frame_width,
            frame_height=frame_height,
            detection_source="none",
            manual_keypoints=None
        )
    
    def _parse_regions(self, results) -> List[CourtRegion]:
        """Parse detected regions from YOLO results"""
        regions = []
        
        if results and len(results) > 0:
            result = results[0]  # Get first (and usually only) result
            
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    region = CourtRegion(
                        name=COURT_CLASSES[cls_id] if cls_id < len(COURT_CLASSES) else f"class_{cls_id}",
                        class_id=cls_id,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf
                    )
                    regions.append(region)
        
        return regions
    
    def _derive_keypoints(self, regions: List[CourtRegion]) -> List[CourtKeypoint]:
        """
        Derive keypoints from detected regions for API compatibility.
        Creates corner and boundary points from the region bounding boxes.
        """
        keypoints = []
        
        # Group regions by type
        regions_by_name = {r.name: r for r in regions}
        
        # Derive keypoints from sidelines (primary for corners)
        sideline_left = regions_by_name.get('sideline-left')
        sideline_right = regions_by_name.get('sideline-right')
        
        if sideline_left:
            keypoints.extend([
                CourtKeypoint("tl-sideline", sideline_left.x1, sideline_left.y1, sideline_left.confidence, 0),
                CourtKeypoint("bl-sideline", sideline_left.x1, sideline_left.y2, sideline_left.confidence, 1),
            ])
        
        if sideline_right:
            keypoints.extend([
                CourtKeypoint("tr-sideline", sideline_right.x2, sideline_right.y1, sideline_right.confidence, 2),
                CourtKeypoint("br-sideline", sideline_right.x2, sideline_right.y2, sideline_right.confidence, 3),
            ])
        
        # Derive net line position
        net = regions_by_name.get('net')
        if net:
            keypoints.extend([
                CourtKeypoint("net-left", net.x1, net.center_y, net.confidence, 4),
                CourtKeypoint("net-center", net.center_x, net.center_y, net.confidence, 5),
                CourtKeypoint("net-right", net.x2, net.center_y, net.confidence, 6),
            ])
        
        # Derive service line positions from court zones
        frontcourt = regions_by_name.get('frontcourt')
        if frontcourt:
            keypoints.append(
                CourtKeypoint("service-line", frontcourt.center_x, frontcourt.y2, frontcourt.confidence, 7)
            )
        
        # Add center points of each detected region
        for region in regions:
            keypoints.append(
                CourtKeypoint(f"{region.name}-center", region.center_x, region.center_y, region.confidence, region.class_id + 10)
            )
        
        return keypoints
    
    def _extract_court_corners(self, regions: List[CourtRegion]) -> Optional[np.ndarray]:
        """
        Extract the four main court corners for perspective transform.
        Uses sidelines and court zones to determine boundaries.
        
        Key insight: Sidelines are bounding boxes around the LINE itself.
        - Left sideline's INNER edge (facing court) is x2 (right edge of box)
        - Right sideline's INNER edge (facing court) is x1 (left edge of box)
        
        IMPORTANT: The model may mislabel sidelines (left as right, etc.)
        We determine actual left/right by comparing X positions, not class names.
        """
        if not regions:
            logger.warning("_extract_court_corners: No regions provided")
            return None
        
        # Collect ALL sideline detections (both left and right labeled)
        # The model sometimes mislabels them, so we sort by X position
        all_sidelines = [r for r in regions if 'sideline' in r.name.lower()]
        
        logger.debug(f"_extract_court_corners: Found {len(all_sidelines)} sideline detections")
        for s in all_sidelines:
            logger.debug(f"  {s.name}: x1={s.x1:.1f}, x2={s.x2:.1f}, center_x={s.center_x:.1f}, conf={s.confidence:.3f}")
        
        # Sort by center X position to determine actual left vs right
        if len(all_sidelines) >= 2:
            # Remove duplicate/overlapping sidelines (keep highest confidence)
            unique_sidelines = []
            for s in sorted(all_sidelines, key=lambda x: x.confidence, reverse=True):
                is_duplicate = False
                for existing in unique_sidelines:
                    # Check if centers are within 20% of each other (likely same line)
                    if abs(s.center_x - existing.center_x) < (s.width + existing.width):
                        is_duplicate = True
                        logger.debug(f"  Skipping duplicate {s.name} (overlaps with {existing.name})")
                        break
                if not is_duplicate:
                    unique_sidelines.append(s)
            
            logger.debug(f"  Unique sidelines after dedup: {len(unique_sidelines)}")
            
            if len(unique_sidelines) >= 2:
                # Sort by X position: leftmost first
                unique_sidelines.sort(key=lambda x: x.center_x)
                sideline_left = unique_sidelines[0]  # Leftmost is TRUE left
                sideline_right = unique_sidelines[-1]  # Rightmost is TRUE right
                logger.info(f"  ACTUAL left sideline: {sideline_left.name} at x={sideline_left.center_x:.1f}")
                logger.info(f"  ACTUAL right sideline: {sideline_right.name} at x={sideline_right.center_x:.1f}")
            elif len(unique_sidelines) == 1:
                # Only one sideline - determine if left or right by position in frame
                s = unique_sidelines[0]
                # Assume frame center is approximately at center_x of court zones
                court_zones = [r for r in regions if 'court' in r.name.lower() or 'midcourt' in r.name.lower()]
                if court_zones:
                    avg_zone_x = sum(z.center_x for z in court_zones) / len(court_zones)
                    if s.center_x < avg_zone_x:
                        sideline_left = s
                        sideline_right = None
                    else:
                        sideline_left = None
                        sideline_right = s
                else:
                    sideline_left = s
                    sideline_right = None
            else:
                sideline_left = None
                sideline_right = None
        elif len(all_sidelines) == 1:
            # Single sideline detected
            s = all_sidelines[0]
            # Use court zones to determine which side
            court_zones = [r for r in regions if 'court' in r.name.lower()]
            if court_zones:
                avg_zone_x = sum(z.center_x for z in court_zones) / len(court_zones)
                if s.center_x < avg_zone_x:
                    sideline_left = s
                    sideline_right = None
                else:
                    sideline_left = None
                    sideline_right = s
            else:
                sideline_left = s
                sideline_right = None
        else:
            sideline_left = None
            sideline_right = None
        
        # Group other regions by type
        regions_by_name = {r.name: r for r in regions}
        
        # Log final sideline assignments
        if sideline_left:
            logger.debug(f"  Final sideline-left: x1={sideline_left.x1:.1f}, x2={sideline_left.x2:.1f}, "
                        f"y1={sideline_left.y1:.1f}, y2={sideline_left.y2:.1f}")
        else:
            logger.debug("  Final sideline-left: NOT AVAILABLE")
            
        if sideline_right:
            logger.debug(f"  Final sideline-right: x1={sideline_right.x1:.1f}, x2={sideline_right.x2:.1f}, "
                        f"y1={sideline_right.y1:.1f}, y2={sideline_right.y2:.1f}")
        else:
            logger.debug("  Final sideline-right: NOT AVAILABLE")
        
        # Get court zones for vertical extent
        frontcourt = regions_by_name.get('frontcourt')
        midcourt_up = regions_by_name.get('midcourt-up')
        midcourt_down = regions_by_name.get('midcourt-down')
        rearcourt_up = regions_by_name.get('rearcourt-up')
        rearcourt_down = regions_by_name.get('rearcourt-down')
        net = regions_by_name.get('net')
        
        # Determine left/right boundaries using INNER edges of sidelines
        # (the edge that touches the playing area)
        boundary_method = "unknown"
        if sideline_left and sideline_right:
            # Inner edge of left sideline is x2, inner edge of right sideline is x1
            left_x = sideline_left.x2  # Inner edge
            right_x = sideline_right.x1  # Inner edge
            boundary_method = "both_sidelines (inner edges: left.x2, right.x1)"
            logger.info(f"  Using BOTH sidelines: left_x={left_x:.1f} (from left.x2), right_x={right_x:.1f} (from right.x1)")
        elif sideline_left:
            # Only left sideline detected - use outer edge and estimate court width
            left_x = sideline_left.x2  # Inner edge
            # Estimate court width based on typical badminton aspect ratio
            estimated_width = sideline_left.height * 0.45  # Approx court width/length ratio
            right_x = left_x + estimated_width
            boundary_method = f"left_sideline_only (estimated width={estimated_width:.1f})"
            logger.info(f"  Using LEFT sideline only: left_x={left_x:.1f}, estimated right_x={right_x:.1f}")
        elif sideline_right:
            # Only right sideline detected
            right_x = sideline_right.x1  # Inner edge
            estimated_width = sideline_right.height * 0.45
            left_x = right_x - estimated_width
            boundary_method = f"right_sideline_only (estimated width={estimated_width:.1f})"
            logger.info(f"  Using RIGHT sideline only: estimated left_x={left_x:.1f}, right_x={right_x:.1f}")
        else:
            # No sidelines - use bounding box of all detections
            all_x = [r.x1 for r in regions] + [r.x2 for r in regions]
            left_x = min(all_x)
            right_x = max(all_x)
            boundary_method = "all_regions_bbox"
            logger.warning(f"  NO sidelines detected - using bbox of all regions: left_x={left_x:.1f}, right_x={right_x:.1f}")
        
        # Determine top/bottom boundaries from sidelines first (most accurate)
        vertical_method = "unknown"
        if sideline_left and sideline_right:
            top_y = min(sideline_left.y1, sideline_right.y1)
            bottom_y = max(sideline_left.y2, sideline_right.y2)
            vertical_method = "both_sidelines"
            logger.info(f"  Vertical from both sidelines: top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
        elif sideline_left:
            top_y = sideline_left.y1
            bottom_y = sideline_left.y2
            vertical_method = "left_sideline_only"
            logger.info(f"  Vertical from left sideline: top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
        elif sideline_right:
            top_y = sideline_right.y1
            bottom_y = sideline_right.y2
            vertical_method = "right_sideline_only"
            logger.info(f"  Vertical from right sideline: top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
        else:
            # Use court zones for vertical extent
            all_y = []
            for zone in [frontcourt, midcourt_up, midcourt_down, rearcourt_up, rearcourt_down, net]:
                if zone:
                    all_y.extend([zone.y1, zone.y2])
            if all_y:
                top_y = min(all_y)
                bottom_y = max(all_y)
                vertical_method = "court_zones"
                logger.info(f"  Vertical from court zones: top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
            else:
                # Last resort: use all regions
                all_y = [r.y1 for r in regions] + [r.y2 for r in regions]
                top_y = min(all_y)
                bottom_y = max(all_y)
                vertical_method = "all_regions_bbox"
                logger.warning(f"  Vertical from all regions bbox: top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
        
        # Refine vertical boundaries with court zones if sidelines not found or partial
        if not (sideline_left and sideline_right):
            if frontcourt:
                old_top = top_y
                top_y = min(top_y, frontcourt.y1)
                if old_top != top_y:
                    logger.debug(f"  Refined top_y with frontcourt: {old_top:.1f} -> {top_y:.1f}")
            if rearcourt_down:
                old_bottom = bottom_y
                bottom_y = max(bottom_y, rearcourt_down.y2)
                if old_bottom != bottom_y:
                    logger.debug(f"  Refined bottom_y with rearcourt_down: {old_bottom:.1f} -> {bottom_y:.1f}")
            elif rearcourt_up:
                old_bottom = bottom_y
                bottom_y = max(bottom_y, rearcourt_up.y2)
                if old_bottom != bottom_y:
                    logger.debug(f"  Refined bottom_y with rearcourt_up: {old_bottom:.1f} -> {bottom_y:.1f}")
        
        # Calculate court dimensions
        court_width = right_x - left_x
        court_height = bottom_y - top_y
        aspect_ratio = court_height / court_width if court_width > 0 else 0
        
        logger.info(f"  FINAL court bounds: left_x={left_x:.1f}, right_x={right_x:.1f}, top_y={top_y:.1f}, bottom_y={bottom_y:.1f}")
        logger.info(f"  Court dimensions: width={court_width:.1f}px, height={court_height:.1f}px, aspect_ratio={aspect_ratio:.2f}")
        logger.info(f"  Methods used: horizontal={boundary_method}, vertical={vertical_method}")
        
        # Build corners: [top-left, top-right, bottom-right, bottom-left]
        corners = np.array([
            [left_x, top_y],      # Top-left
            [right_x, top_y],     # Top-right
            [right_x, bottom_y],  # Bottom-right
            [left_x, bottom_y],   # Bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def _calculate_homography(self, image_corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate homography matrix for pixel to court coordinate transformation.
        
        Args:
            image_corners: 4 corner points in image coordinates [tl, tr, br, bl]
            
        Returns:
            3x3 homography matrix or None if calculation fails
        """
        if image_corners is None or len(image_corners) != 4:
            return None
        
        # Define court corners in real-world coordinates (meters)
        # Origin at top-left corner, x increases to the right, y increases downward
        court_corners = np.array([
            [0, 0],                          # Top-left
            [COURT_WIDTH_DOUBLES, 0],        # Top-right
            [COURT_WIDTH_DOUBLES, COURT_LENGTH],  # Bottom-right
            [0, COURT_LENGTH],               # Bottom-left
        ], dtype=np.float32)
        
        try:
            homography, _ = cv2.findHomography(image_corners, court_corners)
            return homography
        except Exception as e:
            print(f"Homography calculation failed: {e}")
            return None
    
    def pixel_to_court_coords(self, pixel_x: float, pixel_y: float, 
                              detection: Optional[CourtDetection] = None) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to court coordinates (in meters).
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            detection: CourtDetection to use, or None to use last detection
            
        Returns:
            Tuple of (court_x, court_y) in meters, or None if no valid homography
        """
        det = detection or self._last_detection
        
        if det is None or det.homography_matrix is None:
            return None
        
        # Apply homography transformation
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, det.homography_matrix)
        
        court_x, court_y = transformed[0][0]
        return (float(court_x), float(court_y))
    
    def calculate_real_distance(self, p1_pixel: Tuple[float, float], 
                                p2_pixel: Tuple[float, float],
                                detection: Optional[CourtDetection] = None) -> Optional[float]:
        """
        Calculate real-world distance between two pixel points.
        
        Args:
            p1_pixel: First point (x, y) in pixels
            p2_pixel: Second point (x, y) in pixels
            detection: CourtDetection to use, or None to use last detection
            
        Returns:
            Distance in meters, or None if court detection unavailable
        """
        c1 = self.pixel_to_court_coords(p1_pixel[0], p1_pixel[1], detection)
        c2 = self.pixel_to_court_coords(p2_pixel[0], p2_pixel[1], detection)
        
        if c1 is None or c2 is None:
            return None
        
        distance = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        return float(distance)
    
    def get_court_width_pixels(self, detection: Optional[CourtDetection] = None) -> Optional[float]:
        """
        Get the court width in pixels from detection.
        Useful for fallback distance calculations.
        """
        det = detection or self._last_detection
        
        if det is None or det.court_corners is None:
            return None
        
        # Calculate width from top corners
        corners = det.court_corners
        width = np.sqrt((corners[1][0] - corners[0][0])**2 + 
                       (corners[1][1] - corners[0][1])**2)
        return float(width)
    
    def draw_court_overlay(self, frame: np.ndarray, 
                           detection: Optional[CourtDetection] = None,
                           draw_regions: bool = True,
                           draw_keypoints: bool = True,
                           draw_lines: bool = True,
                           draw_corners: bool = True) -> np.ndarray:
        """
        Draw court detection overlay on frame.
        
        Args:
            frame: BGR image frame
            detection: CourtDetection to use, or None to use last detection
            draw_regions: Whether to draw detected region boxes
            draw_keypoints: Whether to draw derived keypoints
            draw_lines: Whether to draw court boundary lines
            draw_corners: Whether to highlight corner points
            
        Returns:
            Frame with overlay drawn
        """
        det = detection or self._last_detection
        
        if det is None or not det.detected:
            return frame
        
        output = frame.copy()
        
        # Define colors for different region types
        region_colors = {
            'frontcourt': (0, 255, 0),      # Green
            'midcourt-down': (0, 200, 100), # Light green
            'midcourt-up': (0, 200, 100),   # Light green
            'net': (0, 0, 255),             # Red
            'rearcourt-down': (255, 165, 0), # Orange
            'rearcourt-up': (255, 165, 0),   # Orange
            'sideline-left': (255, 0, 255),  # Magenta
            'sideline-right': (255, 0, 255), # Magenta
        }
        
        # Draw detected regions
        if draw_regions:
            for region in det.regions:
                color = region_colors.get(region.name, (128, 128, 128))
                
                # Draw semi-transparent filled rectangle
                overlay = output.copy()
                cv2.rectangle(overlay, 
                             (int(region.x1), int(region.y1)), 
                             (int(region.x2), int(region.y2)), 
                             color, -1)
                cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
                
                # Draw rectangle border
                cv2.rectangle(output, 
                             (int(region.x1), int(region.y1)), 
                             (int(region.x2), int(region.y2)), 
                             color, 2)
                
                # Draw label
                label = f"{region.name} ({region.confidence:.2f})"
                label_y = int(region.y1) - 5 if region.y1 > 20 else int(region.y2) + 15
                cv2.putText(output, label, (int(region.x1), label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw keypoints
        if draw_keypoints:
            for kp in det.keypoints:
                cv2.circle(output, (int(kp.x), int(kp.y)), 6, (0, 255, 255), -1)
                cv2.circle(output, (int(kp.x), int(kp.y)), 8, (0, 0, 0), 2)
        
        # Draw court boundary
        if draw_lines and det.court_corners is not None:
            corners = det.court_corners.astype(np.int32)
            
            # Draw filled polygon with transparency
            overlay = output.copy()
            cv2.fillPoly(overlay, [corners], (0, 255, 0))
            cv2.addWeighted(overlay, 0.1, output, 0.9, 0, output)
            
            # Draw boundary lines
            cv2.polylines(output, [corners], True, (0, 255, 0), 2)
            
            # Draw diagonal lines to show perspective
            cv2.line(output, tuple(corners[0]), tuple(corners[2]), (0, 200, 0), 1)
            cv2.line(output, tuple(corners[1]), tuple(corners[3]), (0, 200, 0), 1)
        
        # Highlight corners
        if draw_corners and det.court_corners is not None:
            corner_labels = ["TL", "TR", "BR", "BL"]
            for corner, label in zip(det.court_corners, corner_labels):
                cv2.circle(output, (int(corner[0]), int(corner[1])), 10, (255, 0, 255), -1)
                cv2.putText(output, label, (int(corner[0]) - 10, int(corner[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw confidence indicator
        conf_text = f"Court Detection: {det.confidence:.1%}"
        cv2.putText(output, conf_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output
    
    def get_detection_info(self, detection: Optional[CourtDetection] = None) -> dict:
        """Get detection information as a dictionary for API response"""
        det = detection or self._last_detection
        
        if det is None:
            return {
                "detected": False,
                "regions": [],
                "keypoints": [],
                "confidence": 0.0,
                "frame_dimensions": None,
                "detection_source": "none",
                "manual_keypoints": None
            }
        
        return {
            "detected": det.detected,
            "confidence": det.confidence,
            "regions": [
                {
                    "name": region.name,
                    "class_id": region.class_id,
                    "bbox": [region.x1, region.y1, region.x2, region.y2],
                    "center": [region.center_x, region.center_y],
                    "confidence": region.confidence
                }
                for region in det.regions
            ],
            "keypoints": [
                {
                    "name": kp.name,
                    "x": kp.x,
                    "y": kp.y,
                    "confidence": kp.confidence
                }
                for kp in det.keypoints
            ],
            "court_corners": det.court_corners.tolist() if det.court_corners is not None else None,
            "court_dimensions": {
                "width_meters": COURT_WIDTH_DOUBLES,
                "length_meters": COURT_LENGTH
            },
            # New fields for video size and manual keypoints
            "frame_dimensions": {
                "width": det.frame_width,
                "height": det.frame_height
            } if det.frame_width > 0 and det.frame_height > 0 else None,
            "detection_source": det.detection_source,
            "manual_keypoints": det.manual_keypoints.to_dict() if det.manual_keypoints else None,
            "inference_size": self.inference_size
        }


# Singleton instance for reuse
_detector_instance: Optional[BadmintonCourtDetector] = None


def get_court_detector(
    model_path: Optional[str] = None,
    inference_size: int = DEFAULT_INFERENCE_SIZE
) -> BadmintonCourtDetector:
    """
    Get or create the court detector singleton.
    
    Args:
        model_path: Path to YOLO model (optional)
        inference_size: Size for inference normalization (default: 640)
        
    Returns:
        BadmintonCourtDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = BadmintonCourtDetector(
            model_path,
            inference_size=inference_size
        )
    
    return _detector_instance


def set_manual_court_keypoints(
    top_left: Tuple[float, float],
    top_right: Tuple[float, float],
    bottom_right: Tuple[float, float],
    bottom_left: Tuple[float, float]
) -> ManualCourtKeypoints:
    """
    Convenience function to set manual court keypoints on the global detector.
    
    This is useful for the API to allow users to manually specify court corners
    for more accurate player speed and distance measurements.
    
    RESEARCH BENEFIT:
    Manual keypoints provide significantly more accurate homography transformations
    compared to automatically detected region-based corners. This is because:
    1. User clicks on exact court corner positions
    2. No bounding box interpolation errors
    3. Works even when automatic detection fails
    4. Consistent across all frames (no detection jitter)
    
    Args:
        top_left: (x, y) pixel coordinates of top-left court corner
        top_right: (x, y) pixel coordinates of top-right court corner
        bottom_right: (x, y) pixel coordinates of bottom-right court corner
        bottom_left: (x, y) pixel coordinates of bottom-left court corner
        
    Returns:
        ManualCourtKeypoints object that was set
        
    Example:
        # User clicks on four corners in the frontend
        set_manual_court_keypoints(
            top_left=(100, 50),
            top_right=(500, 50),
            bottom_right=(550, 400),
            bottom_left=(50, 400)
        )
    """
    detector = get_court_detector()
    keypoints = ManualCourtKeypoints(
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
        bottom_left=bottom_left
    )
    detector.set_manual_keypoints(keypoints)
    return keypoints


def clear_manual_court_keypoints() -> None:
    """Clear manual keypoints and revert to automatic detection."""
    detector = get_court_detector()
    detector.set_manual_keypoints(None)


def get_manual_keypoints_status() -> Dict:
    """
    Get the current status of manual keypoints.
    
    Returns:
        Dictionary with manual keypoint status information
    """
    detector = get_court_detector()
    manual_kp = detector.get_manual_keypoints()
    
    return {
        "has_manual_keypoints": detector.has_manual_keypoints(),
        "manual_keypoints": manual_kp.to_dict() if manual_kp else None,
        "detection_mode": "manual" if detector.has_manual_keypoints() else "auto",
        "inference_size": detector.inference_size,
        "model_loaded": detector.model is not None,
        "recommendation": (
            "Manual keypoints provide the most accurate measurements. "
            "Click on the four court corners in the video to set them."
            if not detector.has_manual_keypoints()
            else "Manual keypoints are active. Measurements will use user-defined corners."
        )
    }
