"""
Pose Analytics Module for Badminton Analysis
Provides advanced analysis of player poses over time.

Features:
- Track pose changes across frames
- Detect specific badminton actions (smash, serve, lunge)
- Calculate movement metrics (speed, range of motion)
- Generate pose statistics and summaries
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import math

from pose_detection import (
    FramePoses, PlayerPose, PoseType, Keypoint, 
    BodyAngles, KeypointData, PoseAnalyzer
)


@dataclass
class PoseEvent:
    """Represents a detected pose event (action)"""
    event_type: str  # smash, serve, lunge, jump, etc.
    frame_number: int
    player_id: int
    confidence: float
    duration_frames: int = 1
    peak_frame: int = 0
    body_angles: Optional[BodyAngles] = None
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "frame_number": self.frame_number,
            "player_id": self.player_id,
            "confidence": self.confidence,
            "duration_frames": self.duration_frames,
            "peak_frame": self.peak_frame,
            "body_angles": self.body_angles.to_dict() if self.body_angles else None
        }


@dataclass
class PlayerMovementMetrics:
    """Movement metrics for a single player"""
    player_id: int
    total_distance: float = 0.0  # Total distance traveled (pixels)
    avg_speed: float = 0.0  # Average speed (pixels/frame)
    max_speed: float = 0.0  # Maximum speed 
    range_of_motion: Dict[str, float] = field(default_factory=dict)  # ROM for each joint
    pose_distribution: Dict[str, int] = field(default_factory=dict)  # Count of each pose type
    action_count: Dict[str, int] = field(default_factory=dict)  # Count of detected actions
    
    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "total_distance": self.total_distance,
            "avg_speed": self.avg_speed,
            "max_speed": self.max_speed,
            "range_of_motion": self.range_of_motion,
            "pose_distribution": self.pose_distribution,
            "action_count": self.action_count
        }


@dataclass
class PoseTracker:
    """Tracks pose history for a single player"""
    player_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=60))  # ~2 seconds at 30fps
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def add_pose(self, pose: PlayerPose, frame_number: int):
        """Add a pose to the history"""
        self.history.append((frame_number, pose))
        self.positions.append((pose.bbox_x, pose.bbox_y))
    
    def get_recent_poses(self, n: int = 10) -> List[Tuple[int, PlayerPose]]:
        """Get the n most recent poses"""
        return list(self.history)[-n:]
    
    def calculate_speed(self) -> float:
        """Calculate current speed based on recent positions"""
        if len(self.positions) < 2:
            return 0.0
        
        # Calculate distance over last few frames
        positions = list(self.positions)[-5:]
        total_dist = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_dist += math.sqrt(dx*dx + dy*dy)
        
        return total_dist / (len(positions) - 1) if len(positions) > 1 else 0.0


class PoseActionDetector:
    """Detects specific badminton actions from pose sequences"""
    
    # Action detection thresholds
    SMASH_ARM_ANGLE_MIN = 150  # Nearly straight arm
    SMASH_ARM_HEIGHT_MIN = 100  # Arm well above shoulder
    SERVE_DETECTION_FRAMES = 10
    LUNGE_KNEE_ANGLE_MAX = 100  # Deep knee bend
    JUMP_VERTICAL_THRESHOLD = 30  # Significant vertical movement
    
    def __init__(self):
        self.action_buffer: Dict[int, List[Tuple[int, PoseType, BodyAngles]]] = {}
    
    def update(self, poses: FramePoses) -> List[PoseEvent]:
        """
        Process frame poses and detect actions.
        
        Args:
            poses: Frame poses from pose detector
            
        Returns:
            List of detected pose events
        """
        events = []
        
        for player in poses.players:
            player_id = player.player_id
            
            # Initialize buffer for new players
            if player_id not in self.action_buffer:
                self.action_buffer[player_id] = []
            
            # Add current pose to buffer
            if player.body_angles:
                self.action_buffer[player_id].append(
                    (poses.frame_number, player.pose_type, player.body_angles)
                )
            
            # Keep buffer size manageable
            if len(self.action_buffer[player_id]) > 30:
                self.action_buffer[player_id] = self.action_buffer[player_id][-30:]
            
            # Detect actions based on pose type and angles
            event = self._detect_action(player, poses.frame_number)
            if event:
                events.append(event)
        
        return events
    
    def _detect_action(self, player: PlayerPose, frame_number: int) -> Optional[PoseEvent]:
        """Detect specific action from player pose"""
        if not player.body_angles:
            return None
        
        angles = player.body_angles
        pose_type = player.pose_type
        
        # Smash detection
        if pose_type == PoseType.SMASH:
            return PoseEvent(
                event_type="smash",
                frame_number=frame_number,
                player_id=player.player_id,
                confidence=self._calculate_smash_confidence(player),
                body_angles=angles
            )
        
        # Serve detection
        if pose_type == PoseType.SERVING:
            return PoseEvent(
                event_type="serve",
                frame_number=frame_number,
                player_id=player.player_id,
                confidence=self._calculate_serve_confidence(player),
                body_angles=angles
            )
        
        # Lunge detection
        if pose_type == PoseType.LUNGE:
            return PoseEvent(
                event_type="lunge",
                frame_number=frame_number,
                player_id=player.player_id,
                confidence=self._calculate_lunge_confidence(player),
                body_angles=angles
            )
        
        # Overhead shot detection
        if pose_type == PoseType.OVERHEAD:
            return PoseEvent(
                event_type="overhead",
                frame_number=frame_number,
                player_id=player.player_id,
                confidence=0.7,
                body_angles=angles
            )
        
        return None
    
    def _calculate_smash_confidence(self, player: PlayerPose) -> float:
        """Calculate confidence score for smash detection"""
        if not player.body_angles:
            return 0.0
        
        confidence = 0.5  # Base confidence from pose classification
        angles = player.body_angles
        
        # Higher arm raise increases confidence
        if angles.arm_raise > 100:
            confidence += 0.2
        elif angles.arm_raise > 50:
            confidence += 0.1
        
        # Forward lean increases confidence
        if angles.torso_lean < -15:
            confidence += 0.15
        
        # Extended arm (elbow angle > 150) increases confidence
        if max(angles.left_elbow, angles.right_elbow) > 150:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _calculate_serve_confidence(self, player: PlayerPose) -> float:
        """Calculate confidence score for serve detection"""
        if not player.body_angles:
            return 0.0
        
        confidence = 0.5
        angles = player.body_angles
        
        # Arm raised for serve
        if angles.arm_raise > 30:
            confidence += 0.2
        
        # Upright posture
        if abs(angles.torso_lean) < 10:
            confidence += 0.15
        
        # One arm extended
        if max(angles.left_elbow, angles.right_elbow) > 140:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _calculate_lunge_confidence(self, player: PlayerPose) -> float:
        """Calculate confidence score for lunge detection"""
        if not player.body_angles:
            return 0.0
        
        confidence = 0.5
        angles = player.body_angles
        
        # Deep knee bend
        min_knee = min(angles.left_knee, angles.right_knee)
        if min_knee < 90:
            confidence += 0.25
        elif min_knee < 110:
            confidence += 0.15
        
        # Asymmetric knee angles (one bent, one extended)
        knee_diff = abs(angles.left_knee - angles.right_knee)
        if knee_diff > 50:
            confidence += 0.25
        elif knee_diff > 30:
            confidence += 0.1
        
        return min(confidence, 1.0)


class PoseAnalytics:
    """
    Main analytics class for tracking and analyzing player poses.
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize pose analytics.
        
        Args:
            fps: Video frame rate for time-based calculations
        """
        self.fps = fps
        self.player_trackers: Dict[int, PoseTracker] = {}
        self.player_metrics: Dict[int, PlayerMovementMetrics] = {}
        self.action_detector = PoseActionDetector()
        self.events: List[PoseEvent] = []
        self.frame_count = 0
    
    def process_frame(self, poses: FramePoses) -> Dict[str, Any]:
        """
        Process a frame of pose data and update analytics.
        
        Args:
            poses: Frame poses from pose detector
            
        Returns:
            Dict with frame analysis results
        """
        self.frame_count += 1
        frame_results = {
            "frame": poses.frame_number,
            "player_count": len(poses.players),
            "events": [],
            "player_states": []
        }
        
        for player in poses.players:
            player_id = player.player_id
            
            # Initialize tracker and metrics for new players
            if player_id not in self.player_trackers:
                self.player_trackers[player_id] = PoseTracker(player_id=player_id)
                self.player_metrics[player_id] = PlayerMovementMetrics(player_id=player_id)
            
            # Update tracker
            tracker = self.player_trackers[player_id]
            metrics = self.player_metrics[player_id]
            
            # Calculate movement from previous position
            if tracker.positions:
                last_pos = tracker.positions[-1]
                dx = player.bbox_x - last_pos[0]
                dy = player.bbox_y - last_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                metrics.total_distance += distance
                
                speed = tracker.calculate_speed()
                metrics.max_speed = max(metrics.max_speed, speed)
            
            tracker.add_pose(player, poses.frame_number)
            
            # Update pose distribution
            pose_name = player.pose_type.value
            metrics.pose_distribution[pose_name] = metrics.pose_distribution.get(pose_name, 0) + 1
            
            # Add player state to results
            frame_results["player_states"].append({
                "player_id": player_id,
                "pose_type": pose_name,
                "position": (player.bbox_x, player.bbox_y),
                "speed": tracker.calculate_speed(),
                "body_angles": player.body_angles.to_dict() if player.body_angles else None
            })
        
        # Detect actions
        new_events = self.action_detector.update(poses)
        for event in new_events:
            self.events.append(event)
            
            # Update action counts
            player_id = event.player_id
            if player_id in self.player_metrics:
                metrics = self.player_metrics[player_id]
                metrics.action_count[event.event_type] = metrics.action_count.get(event.event_type, 0) + 1
        
        frame_results["events"] = [e.to_dict() for e in new_events]
        
        return frame_results
    
    def get_player_metrics(self, player_id: int) -> Optional[PlayerMovementMetrics]:
        """Get metrics for a specific player"""
        metrics = self.player_metrics.get(player_id)
        if metrics and self.frame_count > 0:
            metrics.avg_speed = metrics.total_distance / self.frame_count
        return metrics
    
    def get_all_metrics(self) -> Dict[int, dict]:
        """Get metrics for all players"""
        results = {}
        for player_id in self.player_metrics:
            metrics = self.get_player_metrics(player_id)
            if metrics:
                results[player_id] = metrics.to_dict()
        return results
    
    def get_events(self, event_type: Optional[str] = None, player_id: Optional[int] = None) -> List[PoseEvent]:
        """
        Get detected events with optional filtering.
        
        Args:
            event_type: Filter by event type (smash, serve, lunge, etc.)
            player_id: Filter by player ID
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if player_id is not None:
            events = [e for e in events if e.player_id == player_id]
        
        return events
    
    def get_summary(self) -> dict:
        """Get overall analytics summary"""
        total_events = len(self.events)
        event_counts = {}
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        return {
            "total_frames": self.frame_count,
            "player_count": len(self.player_trackers),
            "total_events": total_events,
            "event_breakdown": event_counts,
            "player_metrics": self.get_all_metrics()
        }
    
    def reset(self):
        """Reset all analytics data"""
        self.player_trackers.clear()
        self.player_metrics.clear()
        self.action_detector = PoseActionDetector()
        self.events.clear()
        self.frame_count = 0


# Singleton instance
_analytics_instance: Optional[PoseAnalytics] = None


def get_pose_analytics(fps: float = 30.0) -> PoseAnalytics:
    """Get or create the pose analytics singleton"""
    global _analytics_instance
    
    if _analytics_instance is None:
        _analytics_instance = PoseAnalytics(fps=fps)
    
    return _analytics_instance


def reset_pose_analytics():
    """Reset the analytics instance"""
    global _analytics_instance
    if _analytics_instance:
        _analytics_instance.reset()
    _analytics_instance = None
