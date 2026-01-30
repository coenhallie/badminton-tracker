"""
Pose Detection Module for Badminton Player Analysis
Uses YOLOv26-pose for human keypoint detection to analyze player poses.

Features:
- 17-point skeleton detection (COCO format)
- Badminton-specific pose classification (smash, serve, lunge, etc.)
- Body angle calculations for movement analysis
- Player tracking with pose data
- **Smooth temporal filtering** to eliminate jerky movement between frames

Smoothing Algorithms:
- One Euro Filter: Adaptive low-pass filter (O(1) per keypoint)
- Kalman Filter: Predictive tracking for bounding boxes
- EMA: Exponential moving average for simple smoothing

Usage:
    detector = PoseDetector(enable_smoothing=True)
    poses = detector.detect(frame)
    for pose in poses.players:
        print(f"Player pose: {pose.pose_type}, angles: {pose.body_angles}")
"""

import os
import math
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None

# Import smoothing module
try:
    from skeleton_smoothing import (
        SmoothingConfig,
        SmoothingMethod,
        MultiPlayerSmoother,
        KeypointSmoother,
        BoundingBoxSmoother,
        create_smoother_from_config,
        SmoothedKeypoint,
        SmoothedBoundingBox
    )
    HAS_SMOOTHING = True
except ImportError:
    HAS_SMOOTHING = False
    SmoothingConfig = None
    MultiPlayerSmoother = None


class PoseType(str, Enum):
    """Badminton-specific pose classifications"""
    STANDING = "standing"
    READY = "ready"
    SERVING = "serving"
    SMASH = "smash"
    OVERHEAD = "overhead"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    LUNGE = "lunge"
    JUMP = "jump"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


# COCO 17-keypoint format indices
class Keypoint(int, Enum):
    """COCO keypoint indices for human pose"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Head
    (Keypoint.NOSE, Keypoint.LEFT_EYE),
    (Keypoint.NOSE, Keypoint.RIGHT_EYE),
    (Keypoint.LEFT_EYE, Keypoint.LEFT_EAR),
    (Keypoint.RIGHT_EYE, Keypoint.RIGHT_EAR),
    # Torso
    (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP),
    # Left arm
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW),
    (Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST),
    # Right arm
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW),
    (Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST),
    # Left leg
    (Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE),
    (Keypoint.LEFT_KNEE, Keypoint.LEFT_ANKLE),
    # Right leg
    (Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE),
    (Keypoint.RIGHT_KNEE, Keypoint.RIGHT_ANKLE),
]


@dataclass
class KeypointData:
    """Single keypoint with position and confidence"""
    x: float
    y: float
    confidence: float
    visible: bool = True
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with native Python types"""
        return {
            "x": float(self.x) if self.x is not None else None,
            "y": float(self.y) if self.y is not None else None,
            "confidence": float(self.confidence),
            "visible": bool(self.visible)
        }


@dataclass
class BodyAngles:
    """Calculated body angles for pose analysis"""
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    torso_lean: float = 0.0  # Forward/backward lean angle
    arm_raise: float = 0.0   # How high arms are raised
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with native Python types"""
        return {
            "left_elbow": float(self.left_elbow),
            "right_elbow": float(self.right_elbow),
            "left_shoulder": float(self.left_shoulder),
            "right_shoulder": float(self.right_shoulder),
            "left_knee": float(self.left_knee),
            "right_knee": float(self.right_knee),
            "left_hip": float(self.left_hip),
            "right_hip": float(self.right_hip),
            "torso_lean": float(self.torso_lean),
            "arm_raise": float(self.arm_raise)
        }


@dataclass
class PlayerPose:
    """Complete pose data for a single player"""
    keypoints: Dict[Keypoint, KeypointData]
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    pose_type: PoseType = PoseType.UNKNOWN
    body_angles: Optional[BodyAngles] = None
    player_id: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center of the bounding box"""
        return (self.bbox_x, self.bbox_y)
    
    def get_keypoint(self, kp: Keypoint) -> Optional[KeypointData]:
        """Get a specific keypoint"""
        return self.keypoints.get(kp)
    
    def get_wrist_position(self, dominant_hand: str = "right") -> Optional[Tuple[float, float]]:
        """Get the position of the dominant wrist (for racket tracking)"""
        kp = Keypoint.RIGHT_WRIST if dominant_hand == "right" else Keypoint.LEFT_WRIST
        keypoint = self.keypoints.get(kp)
        if keypoint and keypoint.visible:
            return keypoint.position
        return None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with native Python types"""
        return {
            "keypoints": {str(kp.name): kd.to_dict() for kp, kd in self.keypoints.items()},
            "bbox": {
                "x": float(self.bbox_x),
                "y": float(self.bbox_y),
                "width": float(self.bbox_width),
                "height": float(self.bbox_height)
            },
            "confidence": float(self.confidence),
            "pose_type": str(self.pose_type.value),
            "body_angles": self.body_angles.to_dict() if self.body_angles else None,
            "player_id": int(self.player_id)
        }


@dataclass
class FramePoses:
    """All pose detections for a single frame"""
    frame_number: int
    players: List[PlayerPose] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict with native Python types"""
        return {
            "frame": int(self.frame_number),
            "player_count": int(len(self.players)),
            "players": [p.to_dict() for p in self.players]
        }
    
    def filter_to_main_players(self, max_players: int = 2, player_bboxes: Optional[List[Tuple[float, float, float, float]]] = None) -> 'FramePoses':
        """
        Filter poses to only include the main players.
        
        Args:
            max_players: Maximum number of players to keep (default: 2)
            player_bboxes: Optional list of (x, y, width, height) bounding boxes
                          from player detection to match poses to specific players.
                          If provided, only poses overlapping with these boxes are kept.
        
        Returns:
            New FramePoses with only the main player poses
        """
        if len(self.players) <= max_players and player_bboxes is None:
            return self
        
        filtered_players = []
        
        if player_bboxes is not None and len(player_bboxes) > 0:
            # Match poses to player bounding boxes using IoU
            for pose in self.players:
                best_iou = 0.0
                for bbox in player_bboxes:
                    iou = self._calculate_bbox_iou(pose, bbox)
                    best_iou = max(best_iou, iou)
                
                # Keep pose if it has good overlap with any player bbox
                if best_iou > 0.3:
                    filtered_players.append(pose)
        else:
            # No bboxes provided - keep largest/most confident poses
            # Sort by bounding box area (larger = more likely main player, not spectator)
            sorted_poses = sorted(
                self.players,
                key=lambda p: p.bbox_width * p.bbox_height,
                reverse=True
            )
            filtered_players = sorted_poses[:max_players]
        
        # Reassign player IDs to be 0 and 1 (P1 and P2)
        for idx, pose in enumerate(filtered_players):
            pose.player_id = idx
        
        return FramePoses(
            frame_number=self.frame_number,
            players=filtered_players
        )
    
    def _calculate_bbox_iou(self, pose: PlayerPose, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between a pose's bbox and a detection bbox"""
        # Pose bbox (center-based)
        p_x1 = pose.bbox_x - pose.bbox_width / 2
        p_y1 = pose.bbox_y - pose.bbox_height / 2
        p_x2 = pose.bbox_x + pose.bbox_width / 2
        p_y2 = pose.bbox_y + pose.bbox_height / 2
        
        # Detection bbox (center-based: x, y, width, height)
        d_x, d_y, d_w, d_h = bbox
        d_x1 = d_x - d_w / 2
        d_y1 = d_y - d_h / 2
        d_x2 = d_x + d_w / 2
        d_y2 = d_y + d_h / 2
        
        # Calculate intersection
        x1 = max(p_x1, d_x1)
        y1 = max(p_y1, d_y1)
        x2 = min(p_x2, d_x2)
        y2 = min(p_y2, d_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        pose_area = pose.bbox_width * pose.bbox_height
        det_area = d_w * d_h
        union = pose_area + det_area - intersection
        
        return intersection / union if union > 0 else 0.0


class PoseAnalyzer:
    """Analyzes poses to classify badminton-specific movements"""
    
    @staticmethod
    def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        Returns angle in degrees (0-180).
        """
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))
    
    @staticmethod
    def calculate_body_angles(pose: PlayerPose) -> BodyAngles:
        """Calculate all relevant body angles from keypoints"""
        angles = BodyAngles()
        
        def get_pos(kp: Keypoint) -> Optional[Tuple[float, float]]:
            keypoint = pose.keypoints.get(kp)
            if keypoint and keypoint.visible and keypoint.confidence > 0.3:
                return keypoint.position
            return None
        
        # Left elbow angle (shoulder-elbow-wrist)
        ls, le, lw = get_pos(Keypoint.LEFT_SHOULDER), get_pos(Keypoint.LEFT_ELBOW), get_pos(Keypoint.LEFT_WRIST)
        if ls and le and lw:
            angles.left_elbow = PoseAnalyzer.calculate_angle(ls, le, lw)
        
        # Right elbow angle
        rs, re, rw = get_pos(Keypoint.RIGHT_SHOULDER), get_pos(Keypoint.RIGHT_ELBOW), get_pos(Keypoint.RIGHT_WRIST)
        if rs and re and rw:
            angles.right_elbow = PoseAnalyzer.calculate_angle(rs, re, rw)
        
        # Left shoulder angle (elbow-shoulder-hip)
        lh = get_pos(Keypoint.LEFT_HIP)
        if le and ls and lh:
            angles.left_shoulder = PoseAnalyzer.calculate_angle(le, ls, lh)
        
        # Right shoulder angle
        rh = get_pos(Keypoint.RIGHT_HIP)
        if re and rs and rh:
            angles.right_shoulder = PoseAnalyzer.calculate_angle(re, rs, rh)
        
        # Left knee angle (hip-knee-ankle)
        lk, la = get_pos(Keypoint.LEFT_KNEE), get_pos(Keypoint.LEFT_ANKLE)
        if lh and lk and la:
            angles.left_knee = PoseAnalyzer.calculate_angle(lh, lk, la)
        
        # Right knee angle
        rk, ra = get_pos(Keypoint.RIGHT_KNEE), get_pos(Keypoint.RIGHT_ANKLE)
        if rh and rk and ra:
            angles.right_knee = PoseAnalyzer.calculate_angle(rh, rk, ra)
        
        # Left hip angle (shoulder-hip-knee)
        if ls and lh and lk:
            angles.left_hip = PoseAnalyzer.calculate_angle(ls, lh, lk)
        
        # Right hip angle
        if rs and rh and rk:
            angles.right_hip = PoseAnalyzer.calculate_angle(rs, rh, rk)
        
        # Torso lean (vertical alignment of shoulders to hips)
        if ls and rs and lh and rh:
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            # Angle from vertical (negative = leaning forward, positive = backward)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]  # Inverted because y increases downward
            angles.torso_lean = math.degrees(math.atan2(dx, dy))
        
        # Arm raise (average height of wrists relative to shoulders)
        if lw and rw and ls and rs:
            shoulder_y = (ls[1] + rs[1]) / 2
            wrist_y = (lw[1] + rw[1]) / 2
            # Positive = arms above shoulders, negative = below
            angles.arm_raise = shoulder_y - wrist_y
        
        return angles
    
    @staticmethod
    def classify_pose(pose: PlayerPose) -> PoseType:
        """
        Classify the pose type based on keypoints and angles.
        
        Badminton pose classification rules:
        - SMASH: Arm raised high above head, body leaning back, explosive position
        - OVERHEAD: Arm raised, hitting above head level (clear, drop)
        - SERVING: Arm raised, relatively upright stance, pre-serve position
        - LUNGE: Deep forward/side step with one leg bent significantly
        - FOREHAND: Arm extended to dominant side, rotation towards ball
        - BACKHAND: Arm extended to non-dominant side, rotation away
        - READY: Slightly crouched, balanced stance, waiting for shuttle
        - STANDING: Upright, relaxed, between points
        - JUMP: Both feet off ground (detected by hip height and ankle positions)
        - RECOVERY: Moving back to center after shot
        """
        angles = pose.body_angles
        if angles is None:
            return PoseType.STANDING  # Default to standing if no angles
        
        # Get key keypoints
        nose = pose.get_keypoint(Keypoint.NOSE)
        left_shoulder = pose.get_keypoint(Keypoint.LEFT_SHOULDER)
        right_shoulder = pose.get_keypoint(Keypoint.RIGHT_SHOULDER)
        left_wrist = pose.get_keypoint(Keypoint.LEFT_WRIST)
        right_wrist = pose.get_keypoint(Keypoint.RIGHT_WRIST)
        left_hip = pose.get_keypoint(Keypoint.LEFT_HIP)
        right_hip = pose.get_keypoint(Keypoint.RIGHT_HIP)
        left_knee = pose.get_keypoint(Keypoint.LEFT_KNEE)
        right_knee = pose.get_keypoint(Keypoint.RIGHT_KNEE)
        
        # Calculate useful metrics
        wrist_above_head = False
        wrist_above_shoulder = False
        arm_extended_wide = False
        
        # Check wrist positions relative to head/shoulders
        if nose and nose.visible:
            if left_wrist and left_wrist.visible:
                if left_wrist.y < nose.y:  # Y increases downward
                    wrist_above_head = True
            if right_wrist and right_wrist.visible:
                if right_wrist.y < nose.y:
                    wrist_above_head = True
        
        if left_shoulder and left_shoulder.visible:
            if left_wrist and left_wrist.visible:
                if left_wrist.y < left_shoulder.y:
                    wrist_above_shoulder = True
        if right_shoulder and right_shoulder.visible:
            if right_wrist and right_wrist.visible:
                if right_wrist.y < right_shoulder.y:
                    wrist_above_shoulder = True
        
        # Check arm extension (wrist far from center body)
        if nose and nose.visible:
            body_center_x = nose.x
            if left_wrist and left_wrist.visible:
                if abs(left_wrist.x - body_center_x) > 80:
                    arm_extended_wide = True
            if right_wrist and right_wrist.visible:
                if abs(right_wrist.x - body_center_x) > 80:
                    arm_extended_wide = True
        
        # Calculate knee bend levels
        left_knee_bent = angles.left_knee < 150 if angles.left_knee > 0 else False
        right_knee_bent = angles.right_knee < 150 if angles.right_knee > 0 else False
        deep_knee_bend = angles.left_knee < 120 or angles.right_knee < 120
        knee_diff = abs(angles.left_knee - angles.right_knee) if (angles.left_knee > 0 and angles.right_knee > 0) else 0
        
        # ==========================================
        # POSE CLASSIFICATION (priority order)
        # ==========================================
        
        # 1. SMASH - Arm high, body leaning back/coiled
        if wrist_above_head and angles.arm_raise > 30:
            if angles.torso_lean < -5:  # Leaning back for power
                return PoseType.SMASH
        
        # 2. OVERHEAD - Arm raised to hit above head (clear, drop shot)
        if wrist_above_head:
            return PoseType.OVERHEAD
        
        # 3. SERVING - Arm raised, preparing to serve
        if wrist_above_shoulder and abs(angles.torso_lean) < 20:
            if not arm_extended_wide:  # Arm up but not fully swung
                return PoseType.SERVING
        
        # 4. LUNGE - One leg significantly more bent than other
        if knee_diff > 25 and deep_knee_bend:
            return PoseType.LUNGE
        
        # 5. FOREHAND/BACKHAND - Arm extended to side, hitting shot
        if arm_extended_wide and not wrist_above_shoulder:
            # Determine forehand vs backhand by which arm is extended
            if left_wrist and right_wrist and left_wrist.visible and right_wrist.visible:
                if left_wrist.x < right_wrist.x:
                    # Left arm is more to the left (backhand for right-handed)
                    if nose and abs(left_wrist.x - nose.x) > abs(right_wrist.x - nose.x):
                        return PoseType.BACKHAND
                    else:
                        return PoseType.FOREHAND
                else:
                    if nose and abs(right_wrist.x - nose.x) > abs(left_wrist.x - nose.x):
                        return PoseType.FOREHAND
                    else:
                        return PoseType.BACKHAND
            elif arm_extended_wide:
                return PoseType.FOREHAND  # Default to forehand
        
        # 6. READY - Slight crouch, balanced stance
        if left_knee_bent and right_knee_bent:
            if knee_diff < 20:  # Balanced
                if abs(angles.torso_lean) < 25:
                    return PoseType.READY
        
        # 7. RECOVERY - Moving back to ready position (slight lean)
        if abs(angles.torso_lean) > 15 and abs(angles.torso_lean) < 35:
            if not deep_knee_bend:
                return PoseType.RECOVERY
        
        # 8. STANDING - Upright, relaxed
        if angles.left_knee > 155 and angles.right_knee > 155:
            if abs(angles.torso_lean) < 15:
                return PoseType.STANDING
        
        # Default: READY position is most common during play
        return PoseType.READY


class PoseDetector:
    """
    Detects human poses using YOLOv26-pose model.
    
    Provides 17-keypoint skeleton detection with badminton-specific
    pose classification and body angle analysis.
    
    Optional temporal smoothing eliminates jerky movement between frames
    using One Euro Filter (for keypoints) and Kalman Filter (for bounding boxes).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        keypoint_confidence: float = 0.3,
        enable_smoothing: bool = False,
        smoothing_preset: str = "auto",
        fps: float = 30.0
    ):
        """
        Initialize the pose detector.
        
        Args:
            model_path: Path to pose model (default: yolo26n-pose.pt)
            confidence_threshold: Minimum confidence for person detection
            keypoint_confidence: Minimum confidence for keypoint visibility
            enable_smoothing: Enable temporal smoothing for smooth skeleton output
            smoothing_preset: Smoothing preset - "auto", "high_speed", "stability", "real_time", "none"
            fps: Video frame rate for smoothing calculations
        """
        self.confidence_threshold = confidence_threshold
        self.keypoint_confidence = keypoint_confidence
        self.model = None
        self.analyzer = PoseAnalyzer()
        
        # Smoothing configuration
        self.enable_smoothing = enable_smoothing and HAS_SMOOTHING
        self.fps = fps
        self.smoother: Optional[Any] = None  # MultiPlayerSmoother when enabled
        
        if self.enable_smoothing and HAS_SMOOTHING:
            config = create_smoother_from_config(smoothing_preset, fps)
            self.smoother = MultiPlayerSmoother(config, fps)
            print(f"Pose smoothing enabled with preset: {smoothing_preset}")
        elif enable_smoothing and not HAS_SMOOTHING:
            print("Warning: Smoothing requested but skeleton_smoothing module not available")
        
        # Get model path from parameter or environment
        model_file = model_path or os.getenv("YOLO_POSE_MODEL", "yolo26n-pose.pt")
        
        self._load_model(model_file)
    
    def _load_model(self, model_path: str):
        """Load the YOLO pose model"""
        if not HAS_ULTRALYTICS:
            print("Error: ultralytics package not installed")
            print("Install with: pip install ultralytics")
            return
        
        try:
            # Handle relative paths - check if it's a standard model name
            if model_path in ["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", 
                             "yolo26l-pose.pt", "yolo26x-pose.pt"]:
                # Ultralytics will auto-download standard models
                self.model = YOLO(model_path)
            else:
                # Custom model path
                if not os.path.isabs(model_path):
                    backend_dir = os.path.dirname(__file__)
                    full_path = os.path.join(backend_dir, model_path)
                else:
                    full_path = model_path
                
                if os.path.exists(full_path):
                    self.model = YOLO(full_path)
                else:
                    # Fall back to auto-download
                    print(f"Model not found at {full_path}, attempting to download {model_path}")
                    self.model = YOLO(model_path)
            
            print(f"Pose detector initialized with model: {model_path}")
        except Exception as e:
            print(f"Error loading pose model: {e}")
            print("Attempting to load default yolo26n-pose.pt...")
            try:
                self.model = YOLO("yolo26n-pose.pt")
                print("Successfully loaded default pose model")
            except Exception as e2:
                print(f"Failed to load default model: {e2}")
    
    @property
    def is_available(self) -> bool:
        """Check if pose detection is available"""
        return self.model is not None
    
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: Optional[float] = None
    ) -> FramePoses:
        """
        Detect poses in a frame with optional temporal smoothing.
        
        Args:
            frame: BGR image frame
            frame_number: Current frame number
            timestamp: Optional timestamp in seconds (for smoothing)
            
        Returns:
            FramePoses with all detected player poses (smoothed if enabled)
        """
        if not self.is_available:
            return FramePoses(frame_number=frame_number)
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            frame_poses = self._parse_results(results, frame_number)
            
            # Apply smoothing if enabled
            if self.enable_smoothing and self.smoother is not None:
                frame_poses = self._apply_smoothing(frame_poses, timestamp)
            
            return frame_poses
        except Exception as e:
            print(f"Pose detection error on frame {frame_number}: {e}")
            return FramePoses(frame_number=frame_number)
    
    def _apply_smoothing(
        self,
        frame_poses: FramePoses,
        timestamp: Optional[float] = None
    ) -> FramePoses:
        """
        Apply temporal smoothing to detected poses.
        
        Uses One Euro Filter for keypoints (adaptive smoothing based on movement speed)
        and Kalman Filter for bounding boxes (predictive tracking).
        
        Args:
            frame_poses: Raw detected poses
            timestamp: Optional timestamp for smoothing
            
        Returns:
            FramePoses with smoothed keypoints and bboxes
        """
        if not self.smoother:
            return frame_poses
        
        smoothed_players = []
        
        for player in frame_poses.players:
            # Convert keypoints to list format for smoother
            kp_list = []
            for kp_idx in range(17):
                kp_enum = Keypoint(kp_idx)
                kd = player.keypoints.get(kp_enum)
                if kd:
                    kp_list.append({
                        'x': kd.x,
                        'y': kd.y,
                        'confidence': kd.confidence
                    })
                else:
                    kp_list.append({'x': 0, 'y': 0, 'confidence': 0})
            
            # Update smoother
            smoothed_kps, smoothed_bbox = self.smoother.update(
                player_id=player.player_id,
                keypoints=kp_list,
                bbox_x=player.bbox_x,
                bbox_y=player.bbox_y,
                bbox_width=player.bbox_width,
                bbox_height=player.bbox_height,
                confidence=player.confidence,
                frame_number=frame_poses.frame_number,
                timestamp=timestamp
            )
            
            # Convert smoothed keypoints back to KeypointData
            smoothed_keypoints = {}
            for kp_idx, skp in enumerate(smoothed_kps):
                kp_enum = Keypoint(kp_idx)
                smoothed_keypoints[kp_enum] = KeypointData(
                    x=skp.x,
                    y=skp.y,
                    confidence=skp.confidence,
                    visible=skp.confidence > self.keypoint_confidence
                )
            
            # Create smoothed player pose
            smoothed_pose = PlayerPose(
                keypoints=smoothed_keypoints,
                bbox_x=smoothed_bbox.x,
                bbox_y=smoothed_bbox.y,
                bbox_width=smoothed_bbox.width,
                bbox_height=smoothed_bbox.height,
                confidence=smoothed_bbox.confidence,
                player_id=player.player_id
            )
            
            # Recalculate body angles with smoothed keypoints
            smoothed_pose.body_angles = self.analyzer.calculate_body_angles(smoothed_pose)
            
            # Reclassify pose type
            smoothed_pose.pose_type = self.analyzer.classify_pose(smoothed_pose)
            
            smoothed_players.append(smoothed_pose)
        
        return FramePoses(
            frame_number=frame_poses.frame_number,
            players=smoothed_players
        )
    
    def predict_poses(self, frame_number: int) -> Optional[FramePoses]:
        """
        Predict poses for a frame without detection (for dropout handling).
        
        When detection fails or is skipped, this method uses the smoother's
        prediction capability to estimate poses based on motion history.
        
        Args:
            frame_number: Frame number to predict for
            
        Returns:
            Predicted FramePoses or None if prediction not available
        """
        if not self.enable_smoothing or not self.smoother:
            return None
        
        predictions = self.smoother.predict_all(frame_number)
        
        if not predictions:
            return None
        
        predicted_players = []
        
        for player_id, (smoothed_kps, smoothed_bbox) in predictions.items():
            # Convert smoothed keypoints to KeypointData
            keypoints = {}
            for kp_idx, skp in enumerate(smoothed_kps):
                kp_enum = Keypoint(kp_idx)
                keypoints[kp_enum] = KeypointData(
                    x=skp.x,
                    y=skp.y,
                    confidence=skp.confidence,
                    visible=skp.confidence > self.keypoint_confidence
                )
            
            # Create predicted player pose
            predicted_pose = PlayerPose(
                keypoints=keypoints,
                bbox_x=smoothed_bbox.x,
                bbox_y=smoothed_bbox.y,
                bbox_width=smoothed_bbox.width,
                bbox_height=smoothed_bbox.height,
                confidence=smoothed_bbox.confidence,
                player_id=player_id
            )
            
            # Calculate body angles
            predicted_pose.body_angles = self.analyzer.calculate_body_angles(predicted_pose)
            predicted_pose.pose_type = self.analyzer.classify_pose(predicted_pose)
            
            predicted_players.append(predicted_pose)
        
        return FramePoses(
            frame_number=frame_number,
            players=predicted_players
        )
    
    def reset_smoothing(self):
        """Reset the smoothing state (e.g., when starting a new video)"""
        if self.smoother:
            self.smoother.reset()
    
    def set_smoothing_enabled(self, enabled: bool):
        """
        Enable or disable smoothing at runtime.
        
        This can be toggled from the frontend without restarting the backend.
        Previous smoothing state is preserved - when re-enabled, it resumes
        from the existing filter states.
        
        Args:
            enabled: Whether to enable (True) or disable (False) smoothing
        """
        if enabled and not HAS_SMOOTHING:
            print("Warning: Cannot enable smoothing - skeleton_smoothing module not available")
            return False
        
        if enabled and self.smoother is None:
            # Initialize smoother if not already created
            config = create_smoother_from_config("high_speed", self.fps)
            self.smoother = MultiPlayerSmoother(config, self.fps)
            print("Smoothing enabled at runtime")
        
        self.enable_smoothing = enabled
        return True
    
    def set_smoothing_preset(self, preset: str):
        """
        Change the smoothing preset at runtime.
        
        Args:
            preset: One of "auto", "high_speed", "stability", "real_time", "none"
        """
        if not HAS_SMOOTHING:
            return False
        
        if preset == "none":
            self.enable_smoothing = False
            return True
        
        config = create_smoother_from_config(preset, self.fps)
        self.smoother = MultiPlayerSmoother(config, self.fps)
        self.enable_smoothing = True
        return True
    
    def get_smoothing_config(self) -> dict:
        """Get current smoothing configuration for API response"""
        if not self.enable_smoothing or not self.smoother:
            return {
                "enabled": False,
                "method": None,
                "active_players": []
            }
        
        return {
            "enabled": True,
            "method": "one_euro_filter",
            "active_players": self.smoother.get_active_players(),
            "config": {
                "fps": self.fps,
                "max_players": self.smoother.max_players
            }
        }
    
    def _parse_results(self, results, frame_number: int) -> FramePoses:
        """Parse YOLO pose results into FramePoses"""
        frame_poses = FramePoses(frame_number=frame_number)
        
        for result in results:
            keypoints_data = result.keypoints
            boxes = result.boxes
            
            if keypoints_data is None or boxes is None:
                continue
            
            # Get keypoint tensor: [num_persons, 17, 3] (x, y, conf)
            kpts = keypoints_data.data.cpu().numpy()
            
            for person_idx in range(len(kpts)):
                person_kpts = kpts[person_idx]
                
                # Build keypoints dictionary
                keypoints = {}
                for kp_idx in range(17):
                    x, y, conf = person_kpts[kp_idx]
                    kp_enum = Keypoint(kp_idx)
                    keypoints[kp_enum] = KeypointData(
                        x=float(x),
                        y=float(y),
                        confidence=float(conf),
                        visible=conf > self.keypoint_confidence
                    )
                
                # Get bounding box
                if person_idx < len(boxes):
                    box = boxes[person_idx]
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    x1, y1, x2, y2 = [float(c) for c in xyxy]
                    width = x2 - x1
                    height = y2 - y1
                    cx = x1 + width / 2
                    cy = y1 + height / 2
                else:
                    # Estimate bbox from keypoints
                    valid_kps = [(kd.x, kd.y) for kd in keypoints.values() if kd.visible]
                    if valid_kps:
                        xs, ys = zip(*valid_kps)
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        width = x2 - x1
                        height = y2 - y1
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        conf = 0.5
                    else:
                        continue
                
                # Create player pose
                player_pose = PlayerPose(
                    keypoints=keypoints,
                    bbox_x=cx,
                    bbox_y=cy,
                    bbox_width=width,
                    bbox_height=height,
                    confidence=conf,
                    player_id=person_idx
                )
                
                # Calculate body angles
                player_pose.body_angles = self.analyzer.calculate_body_angles(player_pose)
                
                # Classify pose type
                player_pose.pose_type = self.analyzer.classify_pose(player_pose)
                
                frame_poses.players.append(player_pose)
        
        return frame_poses
    
    def detect_and_annotate(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True,
        draw_pose_label: bool = True,
        draw_angles: bool = False
    ) -> Tuple[FramePoses, np.ndarray]:
        """
        Detect poses and return annotated frame.
        
        Args:
            frame: BGR image frame
            frame_number: Current frame number
            draw_skeleton: Draw skeleton connections
            draw_keypoints: Draw keypoint circles
            draw_bbox: Draw bounding boxes
            draw_pose_label: Draw pose type label
            draw_angles: Draw angle values on joints
            
        Returns:
            Tuple of (FramePoses, annotated_frame)
        """
        import cv2
        
        poses = self.detect(frame, frame_number)
        annotated = frame.copy()
        
        # Colors
        SKELETON_COLOR = (0, 255, 255)  # Yellow
        KEYPOINT_COLOR = (0, 255, 0)    # Green
        BBOX_COLOR = (255, 165, 0)      # Orange
        TEXT_COLOR = (255, 255, 255)    # White
        
        for player in poses.players:
            # Draw bounding box
            if draw_bbox:
                x1 = int(player.bbox_x - player.bbox_width / 2)
                y1 = int(player.bbox_y - player.bbox_height / 2)
                x2 = int(player.bbox_x + player.bbox_width / 2)
                y2 = int(player.bbox_y + player.bbox_height / 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), BBOX_COLOR, 2)
            
            # Draw skeleton connections
            if draw_skeleton:
                for kp1, kp2 in SKELETON_CONNECTIONS:
                    pt1 = player.keypoints.get(kp1)
                    pt2 = player.keypoints.get(kp2)
                    
                    if pt1 and pt2 and pt1.visible and pt2.visible:
                        p1 = (int(pt1.x), int(pt1.y))
                        p2 = (int(pt2.x), int(pt2.y))
                        cv2.line(annotated, p1, p2, SKELETON_COLOR, 2)
            
            # Draw keypoints
            if draw_keypoints:
                for kp, kd in player.keypoints.items():
                    if kd.visible:
                        pt = (int(kd.x), int(kd.y))
                        # Different sizes for different keypoints
                        radius = 5 if kp.value < 5 else 4  # Larger for head keypoints
                        cv2.circle(annotated, pt, radius, KEYPOINT_COLOR, -1)
                        cv2.circle(annotated, pt, radius + 1, (0, 0, 0), 1)
            
            # Draw pose label
            if draw_pose_label:
                label = f"P{player.player_id + 1}: {player.pose_type.value}"
                x1 = int(player.bbox_x - player.bbox_width / 2)
                y1 = int(player.bbox_y - player.bbox_height / 2)
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), BBOX_COLOR, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw angles (optional, can be verbose)
            if draw_angles and player.body_angles:
                angles = player.body_angles
                # Draw elbow angles
                for kp, angle_val in [
                    (Keypoint.LEFT_ELBOW, angles.left_elbow),
                    (Keypoint.RIGHT_ELBOW, angles.right_elbow)
                ]:
                    kd = player.keypoints.get(kp)
                    if kd and kd.visible and angle_val > 0:
                        pt = (int(kd.x) + 10, int(kd.y))
                        cv2.putText(annotated, f"{angle_val:.0f}°", pt,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        
        return poses, annotated
    
    def get_status(self) -> dict:
        """Get detector status including smoothing configuration"""
        status = {
            "available": self.is_available,
            "model": self.model.model_name if self.model else None,
            "confidence_threshold": self.confidence_threshold,
            "keypoint_confidence": self.keypoint_confidence,
            "smoothing_enabled": self.enable_smoothing,
            "fps": self.fps
        }
        
        if self.enable_smoothing and self.smoother:
            status["smoothing"] = {
                "method": "one_euro_filter",
                "active_players": self.smoother.get_active_players(),
                "description": "Adaptive low-pass filter for smooth skeleton tracking"
            }
        
        return status


# Singleton instance
_pose_detector_instance: Optional[PoseDetector] = None


def get_pose_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    keypoint_confidence: float = 0.3,
    enable_smoothing: bool = True,
    smoothing_preset: str = "auto",
    fps: float = 30.0
) -> PoseDetector:
    """
    Get or create the pose detector singleton.
    
    Args:
        model_path: Path to YOLO pose model
        confidence_threshold: Minimum confidence for person detection
        keypoint_confidence: Minimum confidence for keypoint visibility
        enable_smoothing: Enable temporal smoothing (recommended)
        smoothing_preset: Smoothing preset - "auto", "high_speed", "stability", "real_time", "none"
        fps: Video frame rate for smoothing calculations
        
    Returns:
        PoseDetector instance with optional smoothing enabled
    """
    global _pose_detector_instance
    
    if _pose_detector_instance is None:
        conf = float(os.getenv("POSE_CONFIDENCE_THRESHOLD", confidence_threshold))
        kp_conf = float(os.getenv("POSE_KEYPOINT_CONFIDENCE", keypoint_confidence))
        smooth = os.getenv("POSE_ENABLE_SMOOTHING", str(enable_smoothing)).lower() == "true"
        preset = os.getenv("POSE_SMOOTHING_PRESET", smoothing_preset)
        
        _pose_detector_instance = PoseDetector(
            model_path=model_path,
            confidence_threshold=conf,
            keypoint_confidence=kp_conf,
            enable_smoothing=smooth,
            smoothing_preset=preset,
            fps=fps
        )
    
    return _pose_detector_instance


def reset_pose_detector():
    """Reset the singleton instance (also resets smoothing state)"""
    global _pose_detector_instance
    if _pose_detector_instance and _pose_detector_instance.smoother:
        _pose_detector_instance.reset_smoothing()
    _pose_detector_instance = None


def get_smoothed_pose_detector(
    fps: float = 30.0,
    preset: str = "high_speed"
) -> PoseDetector:
    """
    Convenience function to get a pose detector with smoothing enabled.
    
    Optimized for badminton video analysis with high-speed movements.
    
    Args:
        fps: Video frame rate
        preset: Smoothing preset (default: "high_speed" for sports)
        
    Returns:
        PoseDetector with smoothing enabled
    """
    return get_pose_detector(
        enable_smoothing=True,
        smoothing_preset=preset,
        fps=fps
    )
