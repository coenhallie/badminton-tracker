"""
Shuttle Analytics Module
Tracks shuttlecock trajectory and calculates real-world speed using court keypoints.
Provides comprehensive analytics for badminton game analysis.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from court_detection import BadmintonCourtDetector, CourtDetection, COURT_WIDTH_DOUBLES, COURT_LENGTH


class ShotType(Enum):
    """Classification of shot types based on trajectory and speed"""
    SMASH = "smash"            # High speed, downward trajectory
    CLEAR = "clear"            # High arc, back of court
    DROP = "drop"              # Slow, steep downward from net
    DRIVE = "drive"            # Flat, fast horizontal
    NET_SHOT = "net_shot"      # Near net, low speed
    LOB = "lob"                # Defensive high return
    SERVE = "serve"            # Initial shot
    UNKNOWN = "unknown"


@dataclass
class ShuttlePosition:
    """Single shuttle position in a frame"""
    frame_number: int
    timestamp: float  # seconds
    pixel_x: float
    pixel_y: float
    court_x: Optional[float] = None  # Real-world position in meters
    court_y: Optional[float] = None
    confidence: float = 0.0


@dataclass
class ShuttleTrajectory:
    """Complete trajectory of shuttle between shots"""
    trajectory_id: int
    positions: List[ShuttlePosition] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    shot_type: ShotType = ShotType.UNKNOWN
    
    # Calculated metrics
    peak_speed_mps: float = 0.0  # meters per second
    peak_speed_kmh: float = 0.0  # km/h
    avg_speed_kmh: float = 0.0
    total_distance_m: float = 0.0
    max_height_change: float = 0.0
    direction_angle: float = 0.0  # degrees from court centerline


@dataclass
class PlayerCourtPosition:
    """Player position relative to court zones"""
    frame_number: int
    player_id: int
    pixel_x: float
    pixel_y: float
    court_x: Optional[float] = None  # meters from left boundary
    court_y: Optional[float] = None  # meters from back line
    zone: str = "unknown"  # front/mid/back + left/center/right
    distance_to_net: Optional[float] = None
    distance_to_center: Optional[float] = None


@dataclass
class CourtZoneAnalytics:
    """Analytics for court zone coverage"""
    player_id: int
    time_in_front: float = 0.0    # % time in front court (0-4.47m from net)
    time_in_mid: float = 0.0      # % time in mid court
    time_in_back: float = 0.0     # % time in back court
    time_in_left: float = 0.0     # % time on left side
    time_in_center: float = 0.0   # % time in center
    time_in_right: float = 0.0    # % time on right side
    avg_distance_to_net: float = 0.0
    heatmap_data: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MatchAnalytics:
    """Complete match analytics with court-based measurements"""
    total_shots: int = 0
    smash_count: int = 0
    clear_count: int = 0
    drop_count: int = 0
    drive_count: int = 0
    net_shot_count: int = 0
    
    fastest_shot_kmh: float = 0.0
    avg_shot_speed_kmh: float = 0.0
    shot_speeds: List[float] = field(default_factory=list)
    
    rally_lengths: List[int] = field(default_factory=list)
    avg_rally_length: float = 0.0
    
    player_analytics: Dict[int, CourtZoneAnalytics] = field(default_factory=dict)
    shuttle_trajectories: List[ShuttleTrajectory] = field(default_factory=list)


class ShuttleTracker:
    """
    Tracks shuttle positions across frames and calculates real-world speeds.
    Uses court keypoint detection for accurate pixel-to-meter conversion.
    """
    
    # Speed thresholds for shot classification (km/h)
    SMASH_SPEED_THRESHOLD = 150
    DRIVE_SPEED_THRESHOLD = 100
    CLEAR_SPEED_THRESHOLD = 80
    DROP_SPEED_THRESHOLD = 60
    
    # Court zone boundaries (meters from back line)
    BACK_COURT_END = 4.47  # 1/3 of half court
    MID_COURT_END = 8.94   # 2/3 of half court
    
    def __init__(self, court_detector: BadmintonCourtDetector, fps: float = 30.0):
        """
        Initialize shuttle tracker.
        
        Args:
            court_detector: BadmintonCourtDetector instance for coordinate conversion
            fps: Video frames per second for time calculations
        """
        self.court_detector = court_detector
        self.fps = fps
        self.current_trajectory = None
        self.trajectories: List[ShuttleTrajectory] = []
        self.trajectory_counter = 0
        
        # Position history for smoothing and interpolation
        self.position_buffer: deque = deque(maxlen=30)
        
        # Missing frame interpolation settings
        self.max_interpolation_frames = 5
        self.last_detection_frame = -1
    
    def update(self, frame_number: int, shuttle_position: Optional[Tuple[float, float]],
               confidence: float, court_detection: Optional[CourtDetection] = None) -> Optional[ShuttlePosition]:
        """
        Update tracker with new shuttle position.
        
        Args:
            frame_number: Current frame number
            shuttle_position: (x, y) pixel coordinates or None if not detected
            confidence: Detection confidence (0-1)
            court_detection: Current court detection for coordinate conversion
            
        Returns:
            ShuttlePosition object with real-world coordinates, or None
        """
        timestamp = frame_number / self.fps
        
        if shuttle_position is None:
            # Handle missing detection - try interpolation
            if self._should_interpolate(frame_number):
                return self._interpolate_position(frame_number, timestamp)
            return None
        
        pixel_x, pixel_y = shuttle_position
        
        # Convert to court coordinates if possible
        court_x, court_y = None, None
        if court_detection and court_detection.detected:
            court_coords = self.court_detector.pixel_to_court_coords(
                pixel_x, pixel_y, court_detection
            )
            if court_coords:
                court_x, court_y = court_coords
        
        position = ShuttlePosition(
            frame_number=frame_number,
            timestamp=timestamp,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            court_x=court_x,
            court_y=court_y,
            confidence=confidence
        )
        
        self.position_buffer.append(position)
        self.last_detection_frame = frame_number
        
        # Add to current trajectory
        if self.current_trajectory is None:
            self._start_new_trajectory(position)
        else:
            self.current_trajectory.positions.append(position)
            self.current_trajectory.end_frame = frame_number
        
        return position
    
    def _should_interpolate(self, frame_number: int) -> bool:
        """Check if interpolation should be attempted for missing detection"""
        if self.last_detection_frame < 0:
            return False
        frames_since_detection = frame_number - self.last_detection_frame
        return 0 < frames_since_detection <= self.max_interpolation_frames
    
    def _interpolate_position(self, frame_number: int, timestamp: float) -> Optional[ShuttlePosition]:
        """Interpolate shuttle position from recent detections"""
        if len(self.position_buffer) < 2:
            return None
        
        # Use last two positions for linear interpolation
        p1, p2 = list(self.position_buffer)[-2:]
        
        if p2.frame_number == p1.frame_number:
            return None
        
        # Linear interpolation factor
        t = (frame_number - p1.frame_number) / (p2.frame_number - p1.frame_number)
        
        # Interpolate pixel coordinates
        pixel_x = p1.pixel_x + t * (p2.pixel_x - p1.pixel_x)
        pixel_y = p1.pixel_y + t * (p2.pixel_y - p1.pixel_y)
        
        # Interpolate court coordinates if available
        court_x, court_y = None, None
        if p1.court_x is not None and p2.court_x is not None:
            court_x = p1.court_x + t * (p2.court_x - p1.court_x)
            court_y = p1.court_y + t * (p2.court_y - p1.court_y)
        
        # Lower confidence for interpolated positions
        confidence = min(p1.confidence, p2.confidence) * 0.5
        
        position = ShuttlePosition(
            frame_number=frame_number,
            timestamp=timestamp,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            court_x=court_x,
            court_y=court_y,
            confidence=confidence
        )
        
        if self.current_trajectory:
            self.current_trajectory.positions.append(position)
        
        return position
    
    def _start_new_trajectory(self, position: ShuttlePosition):
        """Start a new shuttle trajectory"""
        self.trajectory_counter += 1
        self.current_trajectory = ShuttleTrajectory(
            trajectory_id=self.trajectory_counter,
            positions=[position],
            start_frame=position.frame_number,
            end_frame=position.frame_number
        )
    
    def end_trajectory(self) -> Optional[ShuttleTrajectory]:
        """
        End current trajectory and calculate metrics.
        Call this when shuttle leaves frame or rally ends.
        
        Returns:
            Completed trajectory with calculated metrics
        """
        if self.current_trajectory is None or len(self.current_trajectory.positions) < 2:
            self.current_trajectory = None
            return None
        
        trajectory = self.current_trajectory
        self._calculate_trajectory_metrics(trajectory)
        self.trajectories.append(trajectory)
        self.current_trajectory = None
        
        return trajectory
    
    def _calculate_trajectory_metrics(self, trajectory: ShuttleTrajectory):
        """Calculate speed, distance, and classify shot type"""
        positions = trajectory.positions
        
        if len(positions) < 2:
            return
        
        speeds = []
        total_distance = 0.0
        max_height_change = 0.0
        
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            
            # Calculate distance
            if (p1.court_x is not None and p2.court_x is not None and
                p1.court_y is not None and p2.court_y is not None):
                # Use real-world coordinates
                dx = p2.court_x - p1.court_x
                dy = p2.court_y - p1.court_y
                distance = np.sqrt(dx**2 + dy**2)
            else:
                # Fallback to pixel distance (less accurate)
                dx = p2.pixel_x - p1.pixel_x
                dy = p2.pixel_y - p1.pixel_y
                # Estimate meters (assume 640px ≈ 6.1m court width)
                pixel_to_meter = COURT_WIDTH_DOUBLES / 640
                distance = np.sqrt(dx**2 + dy**2) * pixel_to_meter
            
            total_distance += distance
            
            # Calculate time difference
            dt = p2.timestamp - p1.timestamp
            if dt > 0:
                speed_mps = distance / dt
                speed_kmh = speed_mps * 3.6
                speeds.append(speed_kmh)
            
            # Track height changes (y-axis in court coords represents depth, not height)
            # For actual height, we'd need 3D reconstruction
            # Here we use pixel y-difference as proxy
            height_change = abs(p2.pixel_y - p1.pixel_y)
            max_height_change = max(max_height_change, height_change)
        
        if speeds:
            trajectory.peak_speed_kmh = max(speeds)
            trajectory.peak_speed_mps = trajectory.peak_speed_kmh / 3.6
            trajectory.avg_speed_kmh = np.mean(speeds)
        
        trajectory.total_distance_m = total_distance
        trajectory.max_height_change = max_height_change
        
        # Calculate direction angle
        if len(positions) >= 2:
            p_start, p_end = positions[0], positions[-1]
            if (p_start.court_x is not None and p_end.court_x is not None and
                p_start.court_y is not None and p_end.court_y is not None):
                dx = p_end.court_x - p_start.court_x
                dy = p_end.court_y - p_start.court_y
            else:
                dx = p_end.pixel_x - p_start.pixel_x
                dy = p_end.pixel_y - p_start.pixel_y
            trajectory.direction_angle = np.degrees(np.arctan2(dy, dx))
        
        # Classify shot type
        trajectory.shot_type = self._classify_shot(trajectory)
    
    def _classify_shot(self, trajectory: ShuttleTrajectory) -> ShotType:
        """Classify shot type based on speed and trajectory characteristics"""
        speed = trajectory.peak_speed_kmh
        height_change = trajectory.max_height_change
        
        # Speed-based classification
        if speed >= self.SMASH_SPEED_THRESHOLD:
            return ShotType.SMASH
        elif speed >= self.DRIVE_SPEED_THRESHOLD:
            return ShotType.DRIVE
        elif speed >= self.CLEAR_SPEED_THRESHOLD:
            # Distinguish between clear and lob based on trajectory
            if height_change > 100:  # High arc
                return ShotType.CLEAR
            return ShotType.DRIVE
        elif speed >= self.DROP_SPEED_THRESHOLD:
            return ShotType.DROP
        else:
            # Low speed near net
            if trajectory.total_distance_m < 3:
                return ShotType.NET_SHOT
            return ShotType.LOB
    
    def calculate_speed_between_frames(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float],
        frame_diff: int,
        court_detection: Optional[CourtDetection] = None
    ) -> Tuple[float, float]:
        """
        Calculate speed between two positions.
        
        Args:
            p1: First position (pixel_x, pixel_y)
            p2: Second position (pixel_x, pixel_y)
            frame_diff: Number of frames between positions
            court_detection: Court detection for accurate measurement
            
        Returns:
            Tuple of (speed_mps, speed_kmh)
        """
        # Get real-world distance
        if court_detection and court_detection.detected:
            distance = self.court_detector.calculate_real_distance(p1, p2, court_detection)
        else:
            # Fallback estimate
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            pixel_distance = np.sqrt(dx**2 + dy**2)
            # Rough estimate: assume 640px width = 6.1m
            distance = pixel_distance * (COURT_WIDTH_DOUBLES / 640)
        
        if distance is None:
            return 0.0, 0.0
        
        # Calculate time
        time_seconds = frame_diff / self.fps
        
        if time_seconds <= 0:
            return 0.0, 0.0
        
        speed_mps = distance / time_seconds
        speed_kmh = speed_mps * 3.6
        
        return speed_mps, speed_kmh
    
    def get_analytics(self) -> MatchAnalytics:
        """
        Get comprehensive match analytics.
        
        Returns:
            MatchAnalytics object with all calculated statistics
        """
        analytics = MatchAnalytics()
        
        # Include current trajectory if exists
        all_trajectories = self.trajectories.copy()
        if self.current_trajectory and len(self.current_trajectory.positions) >= 2:
            self._calculate_trajectory_metrics(self.current_trajectory)
            all_trajectories.append(self.current_trajectory)
        
        analytics.shuttle_trajectories = all_trajectories
        analytics.total_shots = len(all_trajectories)
        
        for traj in all_trajectories:
            # Count shot types
            if traj.shot_type == ShotType.SMASH:
                analytics.smash_count += 1
            elif traj.shot_type == ShotType.CLEAR:
                analytics.clear_count += 1
            elif traj.shot_type == ShotType.DROP:
                analytics.drop_count += 1
            elif traj.shot_type == ShotType.DRIVE:
                analytics.drive_count += 1
            elif traj.shot_type == ShotType.NET_SHOT:
                analytics.net_shot_count += 1
            
            # Track speeds
            if traj.peak_speed_kmh > 0:
                analytics.shot_speeds.append(traj.peak_speed_kmh)
        
        if analytics.shot_speeds:
            analytics.fastest_shot_kmh = max(analytics.shot_speeds)
            analytics.avg_shot_speed_kmh = np.mean(analytics.shot_speeds)
        
        return analytics


class PlayerPositionAnalyzer:
    """
    Analyzes player positions relative to court zones.
    Provides court coverage heatmaps and zone statistics.
    """
    
    # Court zone boundaries (meters)
    FRONT_COURT_DEPTH = COURT_LENGTH / 2 / 3  # ~2.23m from net
    MID_COURT_DEPTH = COURT_LENGTH / 2 * 2 / 3  # ~4.47m from net
    
    LEFT_BOUNDARY = COURT_WIDTH_DOUBLES / 3
    RIGHT_BOUNDARY = COURT_WIDTH_DOUBLES * 2 / 3
    
    def __init__(self, court_detector: BadmintonCourtDetector):
        """
        Initialize position analyzer.
        
        Args:
            court_detector: BadmintonCourtDetector for coordinate conversion
        """
        self.court_detector = court_detector
        self.player_positions: Dict[int, List[PlayerCourtPosition]] = {}
    
    def update(self, frame_number: int, player_id: int,
               pixel_position: Tuple[float, float],
               court_detection: Optional[CourtDetection] = None) -> PlayerCourtPosition:
        """
        Update player position tracking.
        
        Args:
            frame_number: Current frame number
            player_id: ID of the player
            pixel_position: (x, y) pixel coordinates
            court_detection: Current court detection
            
        Returns:
            PlayerCourtPosition with zone classification
        """
        pixel_x, pixel_y = pixel_position
        
        # Convert to court coordinates
        court_x, court_y = None, None
        distance_to_net = None
        distance_to_center = None
        
        if court_detection and court_detection.detected:
            coords = self.court_detector.pixel_to_court_coords(pixel_x, pixel_y, court_detection)
            if coords:
                court_x, court_y = coords
                
                # Calculate distances
                # Net is at y = COURT_LENGTH / 2
                net_y = COURT_LENGTH / 2
                distance_to_net = abs(court_y - net_y)
                
                # Center line is at x = COURT_WIDTH_DOUBLES / 2
                center_x = COURT_WIDTH_DOUBLES / 2
                distance_to_center = abs(court_x - center_x)
        
        # Determine zone
        zone = self._classify_zone(court_x, court_y) if court_x is not None and court_y is not None else "unknown"
        
        position = PlayerCourtPosition(
            frame_number=frame_number,
            player_id=player_id,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            court_x=court_x,
            court_y=court_y,
            zone=zone,
            distance_to_net=distance_to_net,
            distance_to_center=distance_to_center
        )
        
        # Store position
        if player_id not in self.player_positions:
            self.player_positions[player_id] = []
        self.player_positions[player_id].append(position)
        
        return position
    
    def _classify_zone(self, court_x: float, court_y: float) -> str:
        """Classify position into court zone"""
        if court_x is None or court_y is None:
            return "unknown"
        
        # Determine front/mid/back (relative to player's side)
        # Assuming player is on bottom half (y > COURT_LENGTH/2)
        net_y = COURT_LENGTH / 2
        
        if court_y > net_y:  # Player's side (bottom half)
            dist_from_net = court_y - net_y
        else:  # Opponent's side (top half)
            dist_from_net = net_y - court_y
        
        if dist_from_net < self.FRONT_COURT_DEPTH:
            depth = "front"
        elif dist_from_net < self.MID_COURT_DEPTH:
            depth = "mid"
        else:
            depth = "back"
        
        # Determine left/center/right
        if court_x < self.LEFT_BOUNDARY:
            side = "left"
        elif court_x > self.RIGHT_BOUNDARY:
            side = "right"
        else:
            side = "center"
        
        return f"{depth}_{side}"
    
    def get_zone_analytics(self, player_id: int) -> CourtZoneAnalytics:
        """
        Calculate zone coverage analytics for a player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            CourtZoneAnalytics with zone time percentages and heatmap
        """
        if player_id not in self.player_positions:
            return CourtZoneAnalytics(player_id=player_id)
        
        positions = self.player_positions[player_id]
        
        if not positions:
            return CourtZoneAnalytics(player_id=player_id)
        
        analytics = CourtZoneAnalytics(player_id=player_id)
        
        # Count zone occurrences
        zone_counts = {
            "front": 0, "mid": 0, "back": 0,
            "left": 0, "center": 0, "right": 0
        }
        
        total_distance_to_net = 0.0
        valid_net_distances = 0
        
        for pos in positions:
            zone = pos.zone
            if zone != "unknown":
                depth, side = zone.split("_")
                zone_counts[depth] += 1
                zone_counts[side] += 1
            
            if pos.distance_to_net is not None:
                total_distance_to_net += pos.distance_to_net
                valid_net_distances += 1
            
            # Collect heatmap data (court coordinates)
            if pos.court_x is not None and pos.court_y is not None:
                analytics.heatmap_data.append((pos.court_x, pos.court_y))
        
        total = len(positions)
        
        # Calculate percentages
        analytics.time_in_front = (zone_counts["front"] / total) * 100
        analytics.time_in_mid = (zone_counts["mid"] / total) * 100
        analytics.time_in_back = (zone_counts["back"] / total) * 100
        analytics.time_in_left = (zone_counts["left"] / total) * 100
        analytics.time_in_center = (zone_counts["center"] / total) * 100
        analytics.time_in_right = (zone_counts["right"] / total) * 100
        
        if valid_net_distances > 0:
            analytics.avg_distance_to_net = total_distance_to_net / valid_net_distances
        
        return analytics
    
    def generate_heatmap_grid(self, player_id: int, grid_size: int = 20) -> np.ndarray:
        """
        Generate a heatmap grid for player positioning.
        
        Args:
            player_id: ID of the player
            grid_size: Number of cells in each dimension
            
        Returns:
            2D numpy array representing position frequency
        """
        if player_id not in self.player_positions:
            return np.zeros((grid_size, grid_size))
        
        positions = self.player_positions[player_id]
        heatmap = np.zeros((grid_size, grid_size))
        
        for pos in positions:
            if pos.court_x is not None and pos.court_y is not None:
                # Normalize to grid coordinates
                grid_x = int((pos.court_x / COURT_WIDTH_DOUBLES) * (grid_size - 1))
                grid_y = int((pos.court_y / COURT_LENGTH) * (grid_size - 1))
                
                # Clamp to valid range
                grid_x = max(0, min(grid_size - 1, grid_x))
                grid_y = max(0, min(grid_size - 1, grid_y))
                
                heatmap[grid_y, grid_x] += 1
        
        # Normalize
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap


def create_analytics_summary(
    shuttle_tracker: ShuttleTracker,
    position_analyzer: PlayerPositionAnalyzer,
    player_ids: List[int]
) -> Dict:
    """
    Create comprehensive analytics summary for API response.
    
    Args:
        shuttle_tracker: Shuttle tracker instance
        position_analyzer: Player position analyzer instance
        player_ids: List of player IDs
        
    Returns:
        Dictionary with complete analytics data
    """
    match_analytics = shuttle_tracker.get_analytics()
    
    summary = {
        "shuttle_analytics": {
            "total_shots": match_analytics.total_shots,
            "shot_types": {
                "smash": match_analytics.smash_count,
                "clear": match_analytics.clear_count,
                "drop": match_analytics.drop_count,
                "drive": match_analytics.drive_count,
                "net_shot": match_analytics.net_shot_count
            },
            "speed_stats": {
                "fastest_shot_kmh": round(match_analytics.fastest_shot_kmh, 1),
                "avg_shot_speed_kmh": round(match_analytics.avg_shot_speed_kmh, 1),
                "all_shot_speeds": [round(s, 1) for s in match_analytics.shot_speeds]
            },
            "trajectories": [
                {
                    "id": t.trajectory_id,
                    "shot_type": t.shot_type.value,
                    "peak_speed_kmh": round(t.peak_speed_kmh, 1),
                    "avg_speed_kmh": round(t.avg_speed_kmh, 1),
                    "distance_m": round(t.total_distance_m, 2),
                    "direction_angle": round(t.direction_angle, 1),
                    "positions": [
                        {
                            "frame": p.frame_number,
                            "pixel": {"x": p.pixel_x, "y": p.pixel_y},
                            "court": {"x": p.court_x, "y": p.court_y} if p.court_x else None,
                            "confidence": round(p.confidence, 2)
                        }
                        for p in t.positions
                    ]
                }
                for t in match_analytics.shuttle_trajectories
            ]
        },
        "player_analytics": {}
    }
    
    for player_id in player_ids:
        zone_analytics = position_analyzer.get_zone_analytics(player_id)
        heatmap = position_analyzer.generate_heatmap_grid(player_id)
        
        summary["player_analytics"][str(player_id)] = {
            "zone_coverage": {
                "front": round(zone_analytics.time_in_front, 1),
                "mid": round(zone_analytics.time_in_mid, 1),
                "back": round(zone_analytics.time_in_back, 1),
                "left": round(zone_analytics.time_in_left, 1),
                "center": round(zone_analytics.time_in_center, 1),
                "right": round(zone_analytics.time_in_right, 1)
            },
            "avg_distance_to_net_m": round(zone_analytics.avg_distance_to_net, 2),
            "heatmap": heatmap.tolist(),
            "position_count": len(zone_analytics.heatmap_data)
        }
    
    return summary


def recalculate_zone_analytics_from_skeleton_data(
    skeleton_frames: List[Dict],
    video_width: int,
    video_height: int
) -> Dict:
    """
    Recalculate player zone analytics from skeleton data using current manual keypoints.
    
    This function allows zone analytics to be recalculated after manual court keypoints
    have been set, similar to how speeds are recalculated. This is important because:
    1. Zone analytics are initially calculated during video processing
    2. If manual keypoints weren't set at that time, all zones are "unknown" (0%)
    3. After user sets manual keypoints, we need to recalculate from pixel positions
    
    Args:
        skeleton_frames: List of skeleton frame data from analysis results
        video_width: Video frame width in pixels
        video_height: Video frame height in pixels
        
    Returns:
        Dictionary with recalculated zone analytics per player, using same format
        as create_analytics_summary's player_analytics section
    """
    from court_detection import (
        get_court_detector, CourtDetection, ManualCourtKeypoints,
        COURT_WIDTH_DOUBLES, COURT_LENGTH
    )
    
    court_detector = get_court_detector()
    
    # Check if manual keypoints are available
    manual_kp = court_detector.get_manual_keypoints()
    if manual_kp is None:
        print("[ZONE ANALYTICS] No manual keypoints set - cannot recalculate zone coverage")
        return {}
    
    # Create court detection from manual keypoints
    import numpy as np
    corners = manual_kp.to_numpy()
    homography = court_detector._calculate_homography(corners)
    
    if homography is None:
        print("[ZONE ANALYTICS] Failed to calculate homography from manual keypoints")
        return {}
    
    court_detection = CourtDetection(
        regions=[],
        keypoints=[],
        court_corners=corners,
        homography_matrix=homography,
        confidence=1.0,
        detected=True,
        frame_width=video_width,
        frame_height=video_height,
        detection_source="manual",
        manual_keypoints=manual_kp
    )
    
    print(f"[ZONE ANALYTICS] Recalculating zone coverage using manual keypoints")
    print(f"  - Video dimensions: {video_width}x{video_height}")
    print(f"  - Court corners: {corners.tolist()}")
    
    # Create a new position analyzer for recalculation
    position_analyzer = PlayerPositionAnalyzer(court_detector)
    
    # Process all skeleton frames to build position history
    for frame_data in skeleton_frames:
        frame_number = frame_data.get("frame", 0)
        
        for player in frame_data.get("players", []):
            player_id = player.get("player_id", 0)
            center = player.get("center", {})
            
            pixel_x = center.get("x")
            pixel_y = center.get("y")
            
            if pixel_x is not None and pixel_y is not None:
                # Update position with court detection for zone classification
                position_analyzer.update(
                    frame_number=frame_number,
                    player_id=player_id,
                    pixel_position=(pixel_x, pixel_y),
                    court_detection=court_detection
                )
    
    # Calculate zone analytics for each player
    player_analytics = {}
    player_ids = list(position_analyzer.player_positions.keys())
    
    print(f"[ZONE ANALYTICS] Found {len(player_ids)} players with position data")
    
    for player_id in player_ids:
        zone_analytics = position_analyzer.get_zone_analytics(player_id)
        heatmap = position_analyzer.generate_heatmap_grid(player_id)
        
        print(f"  - Player {player_id}: {len(zone_analytics.heatmap_data)} positions, "
              f"front={zone_analytics.time_in_front:.1f}%, mid={zone_analytics.time_in_mid:.1f}%, "
              f"back={zone_analytics.time_in_back:.1f}%")
        
        player_analytics[str(player_id)] = {
            "zone_coverage": {
                "front": round(zone_analytics.time_in_front, 1),
                "mid": round(zone_analytics.time_in_mid, 1),
                "back": round(zone_analytics.time_in_back, 1),
                "left": round(zone_analytics.time_in_left, 1),
                "center": round(zone_analytics.time_in_center, 1),
                "right": round(zone_analytics.time_in_right, 1)
            },
            "avg_distance_to_net_m": round(zone_analytics.avg_distance_to_net, 2),
            "heatmap": heatmap.tolist(),
            "position_count": len(zone_analytics.heatmap_data)
        }
    
    return player_analytics
