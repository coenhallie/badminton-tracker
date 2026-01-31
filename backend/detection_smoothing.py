"""
Detection Smoothing and Temporal Interpolation Module

Provides smooth tracking of detected objects using:
1. Kalman filtering for noise reduction and prediction
2. Temporal interpolation for frames without detections
3. Motion prediction for fast-moving objects (shuttlecock)
4. Static detection filtering for false positive removal

This module addresses the lag between detections and video frames
during rapid movements in badminton matches.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import cv2


@dataclass
class SmoothPosition:
    """Smoothed position with prediction capability"""
    x: float
    y: float
    vx: float = 0.0  # velocity x
    vy: float = 0.0  # velocity y
    confidence: float = 1.0
    frame_number: int = 0
    is_predicted: bool = False


@dataclass
class StaticDetectionZone:
    """Represents a zone where static false positives are detected"""
    x: float
    y: float
    radius: float  # Detection radius in pixels
    detection_count: int = 0
    last_frame: int = 0
    first_frame: int = 0
    
    def contains(self, px: float, py: float, tolerance: float = 0) -> bool:
        """Check if a point is within this zone"""
        distance = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return distance <= (self.radius + tolerance)
    
    def update(self, px: float, py: float, frame_number: int):
        """Update zone center with a new detection (running average)"""
        # Smooth update of position
        alpha = 0.1  # Learning rate for position update
        self.x = self.x * (1 - alpha) + px * alpha
        self.y = self.y * (1 - alpha) + py * alpha
        self.detection_count += 1
        self.last_frame = frame_number


class StaticDetectionFilter:
    """
    Filters out static false positive detections by tracking positions
    that remain stationary across multiple frames.
    
    Static objects (like scoring card points, UI elements, or artifacts)
    will be detected in the same location repeatedly without movement.
    Real shuttlecocks are always moving during active play.
    
    This filter:
    1. Tracks detection positions across frames
    2. Identifies zones where detections appear consistently without movement
    3. Filters out detections in these static zones
    """
    
    def __init__(
        self,
        static_threshold_frames: int = 30,
        position_tolerance: float = 25.0,
        min_movement_per_frame: float = 2.0,
        zone_expiry_frames: int = 300,
        min_detections_for_static: int = 15
    ):
        """
        Initialize the static detection filter.
        
        Args:
            static_threshold_frames: Number of consecutive frames a detection
                must appear without significant movement to be considered static
            position_tolerance: Maximum pixel distance to consider detections
                as the same position (default: 25px)
            min_movement_per_frame: Minimum expected movement (pixels/frame)
                for a real shuttle
            zone_expiry_frames: Frames after which an unused static zone expires
            min_detections_for_static: Minimum detections needed to mark a zone as static
        """
        self.static_threshold_frames = static_threshold_frames
        self.position_tolerance = position_tolerance
        self.min_movement_per_frame = min_movement_per_frame
        self.zone_expiry_frames = zone_expiry_frames
        self.min_detections_for_static = min_detections_for_static
        
        # Track static zones (areas with false positives)
        self.static_zones: List[StaticDetectionZone] = []
        
        # Track recent detection positions for movement analysis
        self.recent_positions: deque = deque(maxlen=static_threshold_frames)
        
        # Candidate zones (being evaluated)
        self.candidate_zones: List[StaticDetectionZone] = []
        
        # Current frame number
        self.current_frame = 0
        
        # Statistics
        self.total_filtered: int = 0
        self.total_passed: int = 0
    
    def _find_matching_zone(
        self,
        x: float,
        y: float,
        zones: List[StaticDetectionZone],
        tolerance_multiplier: float = 1.0
    ) -> Optional[StaticDetectionZone]:
        """Find a zone that contains the given position"""
        for zone in zones:
            if zone.contains(x, y, self.position_tolerance * tolerance_multiplier):
                return zone
        return None
    
    def _calculate_recent_movement(self) -> float:
        """Calculate average movement from recent positions"""
        if len(self.recent_positions) < 2:
            return float('inf')  # Not enough data
        
        positions = list(self.recent_positions)
        total_movement = 0.0
        
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            frame_diff = p2[2] - p1[2]  # Frame difference
            if frame_diff > 0:
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                movement_per_frame = distance / frame_diff
                total_movement += movement_per_frame
        
        return total_movement / (len(positions) - 1) if len(positions) > 1 else float('inf')
    
    def _cleanup_expired_zones(self, current_frame: int):
        """Remove zones that haven't been seen recently"""
        self.static_zones = [
            zone for zone in self.static_zones
            if (current_frame - zone.last_frame) < self.zone_expiry_frames
        ]
        self.candidate_zones = [
            zone for zone in self.candidate_zones
            if (current_frame - zone.last_frame) < self.static_threshold_frames * 2
        ]
    
    def _promote_candidate_to_static(self, candidate: StaticDetectionZone):
        """Promote a candidate zone to confirmed static zone"""
        # Check if there's already a similar static zone
        existing = self._find_matching_zone(candidate.x, candidate.y, self.static_zones)
        if existing:
            existing.update(candidate.x, candidate.y, candidate.last_frame)
        else:
            self.static_zones.append(StaticDetectionZone(
                x=candidate.x,
                y=candidate.y,
                radius=self.position_tolerance,
                detection_count=candidate.detection_count,
                last_frame=candidate.last_frame,
                first_frame=candidate.first_frame
            ))
            print(f"[StaticFilter] New static zone detected at ({candidate.x:.1f}, {candidate.y:.1f}) - likely false positive")
    
    def filter_detections(
        self,
        detections: List[Dict],
        frame_number: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter detections, removing those in static zones.
        
        Args:
            detections: List of detection dicts with 'x' and 'y' keys
            frame_number: Current frame number
            
        Returns:
            Tuple of (valid_detections, filtered_detections)
        """
        self.current_frame = frame_number
        self._cleanup_expired_zones(frame_number)
        
        valid = []
        filtered = []
        
        for det in detections:
            x = det.get("x", 0)
            y = det.get("y", 0)
            
            # Check if detection is in a known static zone
            static_zone = self._find_matching_zone(x, y, self.static_zones)
            if static_zone:
                static_zone.update(x, y, frame_number)
                filtered.append(det)
                self.total_filtered += 1
                continue
            
            # Check if detection matches a candidate zone
            candidate_zone = self._find_matching_zone(x, y, self.candidate_zones)
            if candidate_zone:
                candidate_zone.update(x, y, frame_number)
                
                # Check if candidate should be promoted
                frames_span = frame_number - candidate_zone.first_frame
                if (candidate_zone.detection_count >= self.min_detections_for_static and
                    frames_span >= self.static_threshold_frames):
                    self._promote_candidate_to_static(candidate_zone)
                    self.candidate_zones.remove(candidate_zone)
                    filtered.append(det)
                    self.total_filtered += 1
                    continue
            else:
                # Create new candidate zone
                self.candidate_zones.append(StaticDetectionZone(
                    x=x,
                    y=y,
                    radius=self.position_tolerance,
                    detection_count=1,
                    last_frame=frame_number,
                    first_frame=frame_number
                ))
            
            # Track position for movement analysis
            self.recent_positions.append((x, y, frame_number))
            
            valid.append(det)
            self.total_passed += 1
        
        return valid, filtered
    
    def is_static_position(self, x: float, y: float) -> bool:
        """Check if a position is within a known static zone"""
        return self._find_matching_zone(x, y, self.static_zones) is not None
    
    def add_manual_exclusion_zone(self, x: float, y: float, radius: float = 50.0):
        """
        Manually add an exclusion zone (useful for known problem areas).
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            radius: Exclusion radius in pixels
        """
        self.static_zones.append(StaticDetectionZone(
            x=x,
            y=y,
            radius=radius,
            detection_count=1,
            last_frame=self.current_frame,
            first_frame=self.current_frame
        ))
        print(f"[StaticFilter] Manual exclusion zone added at ({x:.1f}, {y:.1f}), radius={radius:.1f}px")
    
    def get_static_zones(self) -> List[Dict]:
        """Get list of current static zones for debugging/visualization"""
        return [
            {
                "x": zone.x,
                "y": zone.y,
                "radius": zone.radius,
                "detection_count": zone.detection_count,
                "first_frame": zone.first_frame,
                "last_frame": zone.last_frame
            }
            for zone in self.static_zones
        ]
    
    def get_stats(self) -> Dict:
        """Get filter statistics"""
        return {
            "total_filtered": self.total_filtered,
            "total_passed": self.total_passed,
            "static_zones_count": len(self.static_zones),
            "candidate_zones_count": len(self.candidate_zones),
            "filter_rate": self.total_filtered / max(1, self.total_filtered + self.total_passed)
        }
    
    def reset(self):
        """Reset all zones and statistics"""
        self.static_zones.clear()
        self.candidate_zones.clear()
        self.recent_positions.clear()
        self.total_filtered = 0
        self.total_passed = 0
        self.current_frame = 0


@dataclass
class InterpolatedFrame:
    """Frame data with interpolated detections"""
    frame_number: int
    timestamp: float
    players: List[Dict[str, Any]] = field(default_factory=list)
    shuttlecocks: List[Dict[str, Any]] = field(default_factory=list)
    rackets: List[Dict[str, Any]] = field(default_factory=list)
    keypoints_data: List[Dict[str, Any]] = field(default_factory=list)
    is_interpolated: bool = False


class KalmanTracker:
    """
    Kalman filter for smooth object tracking.
    
    Models constant velocity motion with position and velocity state.
    Provides prediction for fast-moving objects.
    """
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        prediction_frames: int = 3
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: How much we expect the motion to vary (higher = more responsive)
            measurement_noise: How noisy our measurements are (higher = more smoothing)
            prediction_frames: How many frames ahead we can predict
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Error covariance
        self.P = np.eye(4)
        
        self.initialized = False
        self.prediction_frames = prediction_frames
        self.last_update_frame = 0
    
    def initialize(self, x: float, y: float, frame_number: int = 0):
        """Initialize the filter with a first measurement"""
        self.state = np.array([x, y, 0, 0], dtype=np.float32)
        self.P = np.eye(4) * 10  # High initial uncertainty
        self.initialized = True
        self.last_update_frame = frame_number
    
    def predict(self) -> Tuple[float, float, float, float]:
        """
        Predict the next state.
        
        Returns:
            Tuple of (x, y, vx, vy)
        """
        if not self.initialized:
            return (0, 0, 0, 0)
        
        # State prediction
        self.state = self.F @ self.state
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return tuple(self.state)
    
    def update(self, x: float, y: float, frame_number: int = 0) -> Tuple[float, float, float, float]:
        """
        Update the filter with a new measurement.
        
        Args:
            x: Measured x position
            y: Measured y position
            frame_number: Current frame number
            
        Returns:
            Tuple of smoothed (x, y, vx, vy)
        """
        if not self.initialized:
            self.initialize(x, y, frame_number)
            return (x, y, 0, 0)
        
        # Account for skipped frames
        frames_skipped = frame_number - self.last_update_frame
        if frames_skipped > 1:
            # Run multiple predictions for skipped frames
            for _ in range(frames_skipped - 1):
                self.predict()
        
        # Regular prediction step
        self.predict()
        
        # Measurement
        z = np.array([x, y], dtype=np.float32)
        
        # Innovation (measurement residual)
        y_residual = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + K @ y_residual
        
        # Covariance update
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.last_update_frame = frame_number
        
        return tuple(self.state)
    
    def predict_ahead(self, num_frames: int = 1) -> List[Tuple[float, float]]:
        """
        Predict positions for multiple frames ahead without updating state.
        
        Args:
            num_frames: Number of frames to predict ahead
            
        Returns:
            List of (x, y) predicted positions
        """
        if not self.initialized:
            return []
        
        predictions = []
        state = self.state.copy()
        
        for _ in range(num_frames):
            state = self.F @ state
            predictions.append((state[0], state[1]))
        
        return predictions
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        if not self.initialized:
            return (0, 0)
        return (float(self.state[2]), float(self.state[3]))
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate"""
        if not self.initialized:
            return (0, 0)
        return (float(self.state[0]), float(self.state[1]))


class DetectionSmoother:
    """
    Smooths detection results across frames.
    
    Maintains separate Kalman trackers for each detected object
    and provides interpolation for frames without detections.
    Also filters out static false positive detections (e.g., scoring card points).
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        player_process_noise: float = 0.5,
        player_measurement_noise: float = 1.0,
        shuttle_process_noise: float = 2.0,
        shuttle_measurement_noise: float = 0.5,
        max_prediction_frames: int = 5,
        enable_static_filter: bool = True,
        static_filter_threshold_frames: int = 30
    ):
        """
        Initialize the detection smoother.
        
        Args:
            fps: Video frames per second
            player_process_noise: Process noise for player tracking (lower = smoother)
            player_measurement_noise: Measurement noise for players
            shuttle_process_noise: Process noise for shuttle (higher = more responsive)
            shuttle_measurement_noise: Measurement noise for shuttle
            max_prediction_frames: Maximum frames to predict without detection
            enable_static_filter: Enable filtering of static false positives
            static_filter_threshold_frames: Frames needed to classify as static
        """
        self.fps = fps
        self.max_prediction_frames = max_prediction_frames
        
        # Separate trackers for each object type
        self.player_trackers: Dict[int, KalmanTracker] = {}
        self.shuttle_tracker: Optional[KalmanTracker] = None
        self.racket_trackers: Dict[int, KalmanTracker] = {}
        
        # Noise parameters
        self.player_process_noise = player_process_noise
        self.player_measurement_noise = player_measurement_noise
        self.shuttle_process_noise = shuttle_process_noise
        self.shuttle_measurement_noise = shuttle_measurement_noise
        
        # Static detection filter for shuttle (filters false positives like scoring cards)
        self.enable_static_filter = enable_static_filter
        self.static_filter: Optional[StaticDetectionFilter] = None
        if enable_static_filter:
            self.static_filter = StaticDetectionFilter(
                static_threshold_frames=static_filter_threshold_frames,
                position_tolerance=25.0,  # 25px tolerance for same position
                min_detections_for_static=15  # Need 15 detections to confirm static
            )
        
        # History for interpolation
        self.frame_history: deque = deque(maxlen=100)
        self.last_detections: Dict[str, Any] = {}
    
    def process_frame(
        self,
        frame_number: int,
        players: Optional[List[Dict]] = None,
        shuttlecocks: Optional[List[Dict]] = None,
        rackets: Optional[List[Dict]] = None,
        keypoints: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process detections for a single frame with smoothing.
        
        Args:
            frame_number: Current frame number
            players: List of player detections with bbox
            shuttlecocks: List of shuttle detections
            rackets: List of racket detections
            keypoints: Raw keypoint data for pose
            
        Returns:
            Dict with smoothed detections
        """
        result = {
            "frame_number": frame_number,
            "timestamp": frame_number / self.fps,
            "players": [],
            "shuttlecocks": [],
            "rackets": [],
            "keypoints": keypoints or [],
            "smoothed": True
        }
        
        # Process player detections
        if players:
            for player in players:
                player_id = player.get("player_id", 0)
                x = player.get("x", 0)
                y = player.get("y", 0)
                
                if player_id not in self.player_trackers:
                    self.player_trackers[player_id] = KalmanTracker(
                        process_noise=self.player_process_noise,
                        measurement_noise=self.player_measurement_noise
                    )
                
                tracker = self.player_trackers[player_id]
                sx, sy, vx, vy = tracker.update(x, y, frame_number)
                
                smoothed_player = player.copy()
                smoothed_player["x"] = float(sx)
                smoothed_player["y"] = float(sy)
                smoothed_player["velocity_x"] = float(vx)
                smoothed_player["velocity_y"] = float(vy)
                smoothed_player["is_smoothed"] = True
                result["players"].append(smoothed_player)
        else:
            # Predict player positions if no detections
            for player_id, tracker in self.player_trackers.items():
                if tracker.initialized:
                    frames_since_update = frame_number - tracker.last_update_frame
                    if frames_since_update <= self.max_prediction_frames:
                        x, y, vx, vy = tracker.predict()
                        result["players"].append({
                            "player_id": player_id,
                            "x": float(x),
                            "y": float(y),
                            "velocity_x": float(vx),
                            "velocity_y": float(vy),
                            "is_predicted": True,
                            "confidence": max(0.3, 1.0 - frames_since_update * 0.2)
                        })
        
        # Process shuttle detections with static filter
        filtered_shuttlecocks = shuttlecocks
        if shuttlecocks and len(shuttlecocks) > 0:
            # Apply static detection filter to remove false positives (e.g., scoring card points)
            if self.enable_static_filter and self.static_filter:
                filtered_shuttlecocks, removed = self.static_filter.filter_detections(
                    shuttlecocks, frame_number
                )
                if removed:
                    # Store info about filtered detections for debugging
                    result["filtered_static_detections"] = len(removed)
        
        if filtered_shuttlecocks and len(filtered_shuttlecocks) > 0:
            # Use highest confidence shuttle from filtered list
            best_shuttle = max(filtered_shuttlecocks, key=lambda s: s.get("confidence", 0))
            x = best_shuttle.get("x", 0)
            y = best_shuttle.get("y", 0)
            
            if self.shuttle_tracker is None:
                self.shuttle_tracker = KalmanTracker(
                    process_noise=self.shuttle_process_noise,
                    measurement_noise=self.shuttle_measurement_noise
                )
            
            sx, sy, vx, vy = self.shuttle_tracker.update(x, y, frame_number)
            
            smoothed_shuttle = best_shuttle.copy()
            smoothed_shuttle["x"] = float(sx)
            smoothed_shuttle["y"] = float(sy)
            smoothed_shuttle["velocity_x"] = float(vx)
            smoothed_shuttle["velocity_y"] = float(vy)
            smoothed_shuttle["speed_pixels_per_frame"] = float(np.sqrt(vx**2 + vy**2))
            smoothed_shuttle["is_smoothed"] = True
            result["shuttlecocks"].append(smoothed_shuttle)
        elif self.shuttle_tracker and self.shuttle_tracker.initialized:
            # Predict shuttle position - shuttles move fast, so predict aggressively
            frames_since_update = frame_number - self.shuttle_tracker.last_update_frame
            if frames_since_update <= self.max_prediction_frames:
                x, y, vx, vy = self.shuttle_tracker.predict()
                speed = float(np.sqrt(vx**2 + vy**2))
                result["shuttlecocks"].append({
                    "x": float(x),
                    "y": float(y),
                    "velocity_x": float(vx),
                    "velocity_y": float(vy),
                    "speed_pixels_per_frame": speed,
                    "is_predicted": True,
                    "confidence": max(0.2, 1.0 - frames_since_update * 0.3)
                })
        
        # Process racket detections
        if rackets:
            for i, racket in enumerate(rackets):
                if i not in self.racket_trackers:
                    self.racket_trackers[i] = KalmanTracker(
                        process_noise=1.0,
                        measurement_noise=0.5
                    )
                
                x = racket.get("x", 0)
                y = racket.get("y", 0)
                
                sx, sy, vx, vy = self.racket_trackers[i].update(x, y, frame_number)
                smoothed_racket = racket.copy()
                smoothed_racket["x"] = float(sx)
                smoothed_racket["y"] = float(sy)
                smoothed_racket["is_smoothed"] = True
                result["rackets"].append(smoothed_racket)
        
        # Store in history for interpolation
        self.frame_history.append(result)
        
        return result
    
    def reset(self):
        """Reset all trackers and static filter"""
        self.player_trackers.clear()
        self.shuttle_tracker = None
        self.racket_trackers.clear()
        self.frame_history.clear()
        if self.static_filter:
            self.static_filter.reset()
    
    def add_exclusion_zone(self, x: float, y: float, radius: float = 50.0):
        """
        Add a manual exclusion zone for shuttlecock detection.
        Useful for areas with known false positives like scoring cards.
        
        Args:
            x: Center x coordinate in pixels
            y: Center y coordinate in pixels
            radius: Exclusion radius in pixels (default: 50)
        """
        if self.static_filter:
            self.static_filter.add_manual_exclusion_zone(x, y, radius)
    
    def get_static_filter_stats(self) -> Dict:
        """Get statistics about filtered detections"""
        if self.static_filter:
            return self.static_filter.get_stats()
        return {"enabled": False}
    
    def get_static_zones(self) -> List[Dict]:
        """Get list of identified static zones (false positive areas)"""
        if self.static_filter:
            return self.static_filter.get_static_zones()
        return []


class FrameInterpolator:
    """
    Interpolates skeleton and detection data between keyframes.
    
    Used when frame sampling rate is > 1, this generates
    smooth intermediate frames.
    """
    
    @staticmethod
    def interpolate_position(
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
        t: float
    ) -> Tuple[float, float]:
        """
        Linear interpolation between two positions.
        
        Args:
            pos1: Start position (x, y)
            pos2: End position (x, y)
            t: Interpolation factor (0 = pos1, 1 = pos2)
        """
        return (
            pos1[0] + (pos2[0] - pos1[0]) * t,
            pos1[1] + (pos2[1] - pos1[1]) * t
        )
    
    @staticmethod
    def interpolate_keypoints(
        kp1: List[Dict],
        kp2: List[Dict],
        t: float
    ) -> List[Dict]:
        """
        Interpolate between two keypoint sets.
        
        Args:
            kp1: First keypoint list
            kp2: Second keypoint list
            t: Interpolation factor
        """
        if len(kp1) != len(kp2):
            # Can't interpolate mismatched keypoints
            return kp1 if t < 0.5 else kp2
        
        result = []
        for i in range(len(kp1)):
            k1 = kp1[i]
            k2 = kp2[i]
            
            interp_kp = {
                "name": k1.get("name", ""),
                "confidence": k1.get("confidence", 0) * (1 - t) + k2.get("confidence", 0) * t
            }
            
            # Only interpolate if both have valid positions
            if k1.get("x") is not None and k2.get("x") is not None:
                interp_kp["x"] = k1["x"] + (k2["x"] - k1["x"]) * t
                interp_kp["y"] = k1["y"] + (k2["y"] - k1["y"]) * t
            elif k1.get("x") is not None:
                interp_kp["x"] = k1["x"]
                interp_kp["y"] = k1["y"]
            elif k2.get("x") is not None:
                interp_kp["x"] = k2["x"]
                interp_kp["y"] = k2["y"]
            else:
                interp_kp["x"] = None
                interp_kp["y"] = None
            
            result.append(interp_kp)
        
        return result
    
    @staticmethod
    def interpolate_bounding_box(
        box1: Dict,
        box2: Dict,
        t: float
    ) -> Dict:
        """
        Interpolate between two bounding boxes.
        """
        return {
            "x": box1.get("x", 0) + (box2.get("x", 0) - box1.get("x", 0)) * t,
            "y": box1.get("y", 0) + (box2.get("y", 0) - box1.get("y", 0)) * t,
            "width": box1.get("width", 0) + (box2.get("width", 0) - box1.get("width", 0)) * t,
            "height": box1.get("height", 0) + (box2.get("height", 0) - box1.get("height", 0)) * t,
            "confidence": (box1.get("confidence", 0) + box2.get("confidence", 0)) / 2,
            "is_interpolated": True
        }
    
    @classmethod
    def interpolate_frames(
        cls,
        frame1: Dict,
        frame2: Dict,
        num_intermediate: int
    ) -> List[Dict]:
        """
        Generate intermediate frames between two keyframes.
        
        Args:
            frame1: First keyframe data
            frame2: Second keyframe data
            num_intermediate: Number of intermediate frames to generate
            
        Returns:
            List of interpolated frame data
        """
        if num_intermediate <= 0:
            return []
        
        results = []
        frame_num_start = frame1.get("frame", 0)
        frame_num_end = frame2.get("frame", 0)
        time_start = frame1.get("timestamp", 0)
        time_end = frame2.get("timestamp", 0)
        
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            
            interp_frame = {
                "frame": int(frame_num_start + (frame_num_end - frame_num_start) * t),
                "timestamp": time_start + (time_end - time_start) * t,
                "is_interpolated": True,
                "players": [],
                "shuttlecocks": [],
                "rackets": []
            }
            
            # Interpolate players (match by player_id)
            players1 = {p.get("player_id"): p for p in frame1.get("players", [])}
            players2 = {p.get("player_id"): p for p in frame2.get("players", [])}
            
            all_player_ids = set(players1.keys()) | set(players2.keys())
            for pid in all_player_ids:
                p1 = players1.get(pid)
                p2 = players2.get(pid)
                
                if p1 and p2:
                    # Interpolate center position
                    center1 = p1.get("center", {"x": p1.get("x", 0), "y": p1.get("y", 0)})
                    center2 = p2.get("center", {"x": p2.get("x", 0), "y": p2.get("y", 0)})
                    
                    interp_player = {
                        "player_id": pid,
                        "center": {
                            "x": center1["x"] + (center2["x"] - center1["x"]) * t,
                            "y": center1["y"] + (center2["y"] - center1["y"]) * t
                        },
                        "current_speed": p1.get("current_speed", 0) + (p2.get("current_speed", 0) - p1.get("current_speed", 0)) * t,
                        "is_interpolated": True
                    }
                    
                    # Interpolate keypoints if available
                    kp1 = p1.get("keypoints", [])
                    kp2 = p2.get("keypoints", [])
                    if kp1 and kp2:
                        interp_player["keypoints"] = cls.interpolate_keypoints(kp1, kp2, t)
                    elif kp1:
                        interp_player["keypoints"] = kp1
                    elif kp2:
                        interp_player["keypoints"] = kp2
                    
                    interp_frame["players"].append(interp_player)
                elif p1:
                    interp_frame["players"].append({**p1, "is_extrapolated": True})
                elif p2:
                    interp_frame["players"].append({**p2, "is_extrapolated": True})
            
            # Interpolate shuttlecocks
            shuttles1 = frame1.get("shuttlecocks", [])
            shuttles2 = frame2.get("shuttlecocks", [])
            
            if shuttles1 and shuttles2:
                # Interpolate the best shuttle from each
                for s1, s2 in zip(shuttles1[:1], shuttles2[:1]):
                    interp_frame["shuttlecocks"].append(cls.interpolate_bounding_box(s1, s2, t))
            elif shuttles1:
                interp_frame["shuttlecocks"] = [s.copy() for s in shuttles1]
            elif shuttles2:
                interp_frame["shuttlecocks"] = [s.copy() for s in shuttles2]
            
            results.append(interp_frame)
        
        return results


def generate_smooth_skeleton_data(
    raw_skeleton_data: List[Dict],
    target_fps: float,
    original_fps: float,
    sample_rate: int = 1
) -> List[Dict]:
    """
    Generate smooth skeleton data by interpolating between processed frames.
    
    Args:
        raw_skeleton_data: Original skeleton data from processing
        target_fps: Desired output frame rate
        original_fps: Original video frame rate
        sample_rate: Processing sample rate (every Nth frame)
        
    Returns:
        List of interpolated skeleton frames at target FPS
    """
    if not raw_skeleton_data or len(raw_skeleton_data) < 2:
        return raw_skeleton_data
    
    result = []
    smoother = DetectionSmoother(fps=original_fps)
    interpolator = FrameInterpolator()
    
    for i in range(len(raw_skeleton_data) - 1):
        current_frame = raw_skeleton_data[i]
        next_frame = raw_skeleton_data[i + 1]
        
        # Add current frame (with smoothing)
        current_frame_num = current_frame.get("frame", i)
        next_frame_num = next_frame.get("frame", i + 1)
        
        # Apply Kalman smoothing to current frame
        smoothed_current = smoother.process_frame(
            frame_number=current_frame_num,
            players=[{"player_id": p.get("player_id"), "x": p.get("center", {}).get("x"), "y": p.get("center", {}).get("y")} for p in current_frame.get("players", [])],
            shuttlecocks=current_frame.get("badminton_detections", {}).get("shuttlecocks", []) if current_frame.get("badminton_detections") else [],
            keypoints=[p.get("keypoints", []) for p in current_frame.get("players", [])]
        )
        
        # Merge smoothed data back into frame
        enhanced_current = current_frame.copy()
        enhanced_current["smoothed"] = True
        result.append(enhanced_current)
        
        # Generate intermediate frames
        frames_gap = next_frame_num - current_frame_num
        if frames_gap > 1:
            intermediate = interpolator.interpolate_frames(
                current_frame,
                next_frame,
                frames_gap - 1
            )
            result.extend(intermediate)
    
    # Add last frame
    if raw_skeleton_data:
        result.append(raw_skeleton_data[-1])
    
    return result


class VideoPreprocessor:
    """
    Preprocesses video for optimal YOLO detection performance.
    
    Techniques:
    1. Motion deblur for fast movements
    2. Frame rate normalization
    3. Sharpening for small object detection (shuttlecock)
    """
    
    def __init__(self, target_fps: float = 60):
        """
        Initialize preprocessor.
        
        Args:
            target_fps: Target frame rate for processing
        """
        self.target_fps = target_fps
    
    def deblur_frame(self, frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply motion deblur to reduce blur from fast movements.
        
        Uses unsharp masking technique.
        """
        # Create a Gaussian blurred version
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        
        # Unsharp mask: original + (original - blurred) * amount
        sharpened = cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def enhance_small_objects(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance visibility of small objects like shuttlecock.
        
        Uses contrast enhancement and edge sharpening.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        apply_deblur: bool = True,
        apply_enhancement: bool = True
    ) -> np.ndarray:
        """
        Apply all preprocessing steps to a frame.
        """
        result = frame.copy()
        
        if apply_deblur:
            result = self.deblur_frame(result)
        
        if apply_enhancement:
            result = self.enhance_small_objects(result)
        
        return result


# Utility function for frame-accurate lookup
def binary_search_frame(
    skeleton_data: List[Dict],
    target_timestamp: float
) -> int:
    """
    Binary search to find the closest frame to a target timestamp.
    
    Much faster than linear search for large datasets.
    O(log n) vs O(n).
    
    Args:
        skeleton_data: Sorted list of frame data
        target_timestamp: Target timestamp in seconds
        
    Returns:
        Index of the closest frame
    """
    if not skeleton_data:
        return 0
    
    left, right = 0, len(skeleton_data) - 1
    
    while left < right:
        mid = (left + right) // 2
        mid_time = skeleton_data[mid].get("timestamp", 0)
        
        if mid_time < target_timestamp:
            left = mid + 1
        else:
            right = mid
    
    # Check if previous frame is closer
    if left > 0:
        prev_time = skeleton_data[left - 1].get("timestamp", 0)
        curr_time = skeleton_data[left].get("timestamp", 0)
        
        if abs(prev_time - target_timestamp) < abs(curr_time - target_timestamp):
            return left - 1
    
    return left


# Build frame index for O(1) lookup
def build_frame_index(skeleton_data: List[Dict]) -> Dict[int, int]:
    """
    Build an index mapping frame numbers to array indices.
    
    Enables O(1) frame lookup.
    
    Args:
        skeleton_data: List of frame data
        
    Returns:
        Dict mapping frame number -> array index
    """
    return {
        frame.get("frame", i): i
        for i, frame in enumerate(skeleton_data)
    }
