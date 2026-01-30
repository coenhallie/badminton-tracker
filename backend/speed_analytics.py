"""
Speed Analytics Module for Badminton Player Movement Analysis

This module calculates real-time running speed for players using:
- Court-based spatial calibration for accurate pixel-to-meter conversion
- Frame-to-frame position tracking
- Exponential smoothing to reduce detection jitter
- Speed zone classification for athletic performance insights
- **ROBUST OUTLIER DETECTION** to filter unrealistic speeds
- **KALMAN FILTERING** for smooth velocity estimation
- **MEDIAN FILTERING** to reject spike outliers
- **PHYSIOLOGICAL SPEED LIMITS** as sanity checks

Standard badminton court dimensions:
- Doubles: 13.4m x 6.1m
- Singles: 13.4m x 5.18m

RESEARCH NOTES on Human Running Speeds:
- Usain Bolt peak speed: 12.4 m/s (44.7 km/h) - fastest human ever
- Usain Bolt average over 100m: 10.4 m/s (37.5 km/h)
- Average human sprint: 6.7 m/s (24 km/h) - 100m in 15 seconds
- Non-professional athlete max: 6.7-8 m/s (24-29 km/h)

BADMINTON-SPECIFIC SPEED DATA:
- Badminton involves SHORT movements on a small court (13.4m x 6.1m)
- Players do NOT sprint continuously - they shuffle, lunge, and recover
- Typical footwork movement: 1-4 m/s (4-15 km/h)
- Quick recoveries/lunges: 4-7 m/s (15-25 km/h)
- Maximum explosive burst (rare): 7-9 m/s (25-32 km/h)
- Absolute max (diving saves): ~9-10 m/s (32-36 km/h) in SHORT bursts

IMPORTANT: Speeds above 25-30 km/h should be RARE in badminton!
Average speeds during a rally should be 5-15 km/h (constant repositioning).
If you're seeing sustained 30+ km/h, it's likely a tracking/calibration error.

If calculated speed exceeds these thresholds, it indicates:
1. Position tracking error (ID switch, occlusion recovery)
2. Coordinate system calibration error (incorrect pixel-to-meter ratio)
3. Frame timing error
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque
from collections import deque
from enum import Enum
import logging
import statistics

# Import court detection for homography-based measurements
from court_detection import (
    get_court_detector, BadmintonCourtDetector, CourtDetection,
    COURT_LENGTH, COURT_WIDTH_DOUBLES, COURT_WIDTH_SINGLES
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSIOLOGICAL SPEED LIMITS (Sanity Checks)
# =============================================================================
# These thresholds prevent impossible speed values from corrupting analytics
#
# IMPORTANT: These values are aligned with main.py to ensure consistent behavior
# across the entire application. If you change these, also update main.py.

# Maximum human running speed (m/s) - Usain Bolt peak was ~12.4 m/s
MAX_HUMAN_SPEED_MPS = 12.5  # 45 km/h - absolute physical limit for any human

# Maximum expected badminton player speed (m/s)
# Aligned with main.py MAX_BADMINTON_SPEED_KMH = 43.0 km/h = ~12 m/s
# This is generous to allow for brief explosive movements while filtering errors
MAX_BADMINTON_SPEED_MPS = 12.0  # 43.2 km/h - aligned with main.py

# Typical maximum sustained speed during active play
# Most movement is 3-6 m/s (11-22 km/h) with bursts to 7-8 m/s
TYPICAL_MAX_BADMINTON_SPEED_MPS = 8.0  # 29 km/h - normal "sprinting" on court

# Maximum position jump per frame (meters) - detects tracking loss
# At 30fps, 10 m/s max speed = 0.33m/frame max realistic movement
# Setting to 0.5m allows for some detection jitter while catching errors
MAX_POSITION_JUMP_M = 0.5  # 0.5 meter per frame is likely a tracking error

# Minimum time interval for speed calculation (seconds)
# Avoids division by very small numbers
MIN_TIME_INTERVAL = 0.001  # 1 millisecond

# Median filter window size (odd number)
MEDIAN_FILTER_WINDOW = 5  # Use median of last 5 speed readings


# =============================================================================
# ROBUST SPEED FILTERING CLASSES
# =============================================================================

class MedianFilter:
    """
    Sliding window median filter for spike outlier rejection.
    
    Unlike exponential smoothing, median filter completely rejects outliers
    rather than dampening them. A spike of 1000 m/s in a window of [2, 3, 1000, 2, 3]
    produces median = 3, not a dampened high value.
    
    This is essential for rejecting tracking loss artifacts.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize median filter.
        
        Args:
            window_size: Odd number for window size (default: 5)
        """
        # Ensure odd window size for true median
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.buffer: Deque[float] = deque(maxlen=self.window_size)
    
    def filter(self, value: float) -> float:
        """
        Apply median filter to a value.
        
        Args:
            value: Raw value to filter
            
        Returns:
            Median of the buffer (including new value)
        """
        self.buffer.append(value)
        
        if len(self.buffer) < 3:
            # Not enough data for robust median yet
            return value
        
        return statistics.median(self.buffer)
    
    def reset(self):
        """Reset the filter state."""
        self.buffer.clear()


class VelocityKalmanFilter:
    """
    Kalman filter for smooth velocity estimation with prediction capability.
    
    This provides physically plausible velocity estimates by modeling
    player movement with acceleration constraints. Unlike simple smoothing,
    Kalman filtering:
    1. Can predict velocity when detection drops out
    2. Naturally handles variable measurement noise
    3. Produces smooth velocity curves that respect physics
    
    State vector: [speed, acceleration]
    Observation: [speed]
    """
    
    def __init__(
        self,
        process_noise: float = 2.0,  # How much acceleration can change
        measurement_noise: float = 5.0,  # How noisy speed measurements are
        initial_speed: float = 0.0
    ):
        """
        Initialize velocity Kalman filter.
        
        Args:
            process_noise: Expected acceleration variance (m/s^2)
            measurement_noise: Expected measurement variance (m/s)
            initial_speed: Starting speed estimate
        """
        # State: [velocity, acceleration]
        self.state = np.array([initial_speed, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 10.0  # Initial uncertainty
        
        # Process noise covariance (how much acceleration varies)
        self.Q = np.array([
            [0.1, 0.0],
            [0.0, process_noise]
        ])
        
        # Measurement noise (variance of speed measurements)
        self.R = np.array([[measurement_noise]])
        
        # Measurement matrix (we observe speed only)
        self.H = np.array([[1.0, 0.0]])
        
        self.last_update_time = 0.0
        self.initialized = False
    
    def predict(self, dt: float) -> float:
        """
        Predict velocity at next timestep.
        
        Args:
            dt: Time delta in seconds
            
        Returns:
            Predicted velocity
        """
        if not self.initialized:
            return 0.0
        
        # State transition matrix: v_new = v_old + a * dt
        F = np.array([
            [1.0, dt],
            [0.0, 1.0 - 0.5 * dt]  # Acceleration decays towards 0
        ])
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
        return float(self.state[0])
    
    def update(self, measured_speed: float, dt: float) -> float:
        """
        Update filter with new speed measurement.
        
        Args:
            measured_speed: Observed speed in m/s
            dt: Time delta since last update in seconds
            
        Returns:
            Filtered velocity estimate
        """
        if not self.initialized:
            self.state = np.array([measured_speed, 0.0])
            self.initialized = True
            return measured_speed
        
        # Predict step
        self.predict(dt)
        
        # Measurement update
        z = np.array([[measured_speed]])
        
        # Innovation (measurement residual)
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + (K @ y).flatten()
        
        # Update covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return float(self.state[0])
    
    def get_velocity(self) -> float:
        """Get current velocity estimate."""
        return float(self.state[0]) if self.initialized else 0.0
    
    def get_acceleration(self) -> float:
        """Get current acceleration estimate."""
        return float(self.state[1]) if self.initialized else 0.0
    
    def reset(self):
        """Reset filter state."""
        self.state = np.array([0.0, 0.0])
        self.P = np.eye(2) * 10.0
        self.initialized = False


class RobustSpeedFilter:
    """
    Comprehensive speed filtering pipeline that combines multiple techniques
    to ensure physiologically plausible speed readings.
    
    Pipeline:
    1. Sanity Check: Reject speeds > MAX_HUMAN_SPEED_MPS
    2. Position Jump Detection: Detect tracking loss from large position jumps
    3. Median Filter: Reject spike outliers
    4. Kalman Filter: Smooth velocity with physics-aware prediction
    5. Final Clamp: Ensure output is within bounds
    
    This pipeline ensures that even severe tracking errors (1300+ km/h spikes)
    are filtered down to realistic values.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        use_kalman: bool = True,
        use_median: bool = True,
        max_speed_mps: float = MAX_BADMINTON_SPEED_MPS
    ):
        """
        Initialize robust speed filter.
        
        Args:
            fps: Video frame rate for timing calculations
            use_kalman: Whether to use Kalman filtering
            use_median: Whether to use median filtering
            max_speed_mps: Maximum allowed speed (sanity check threshold)
        """
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.max_speed = max_speed_mps
        
        # Filter components
        self.median_filter = MedianFilter(MEDIAN_FILTER_WINDOW) if use_median else None
        self.kalman_filter = VelocityKalmanFilter() if use_kalman else None
        
        # Tracking for position jump detection
        self.last_valid_speed = 0.0
        self.consecutive_outliers = 0
        self.max_consecutive_outliers = 3  # After 3 outliers, accept new baseline
        
        # Statistics for debugging
        self.total_samples = 0
        self.rejected_samples = 0
        self.clamped_samples = 0
    
    def filter(
        self,
        raw_speed_mps: float,
        distance_m: float,
        time_delta: float
    ) -> Tuple[float, bool, str]:
        """
        Apply robust filtering to a raw speed measurement.
        
        Args:
            raw_speed_mps: Raw calculated speed in m/s
            distance_m: Distance traveled in meters (for jump detection)
            time_delta: Time between frames in seconds
            
        Returns:
            Tuple of (filtered_speed_mps, is_valid, rejection_reason)
            - filtered_speed_mps: Filtered speed value
            - is_valid: Whether the raw value was considered valid
            - rejection_reason: If invalid, explains why (for debugging)
        """
        self.total_samples += 1
        rejection_reason = ""
        is_valid = True
        
        # =========================================
        # STEP 1: Position jump detection
        # =========================================
        # Large jumps indicate tracking loss (ID switch, occlusion recovery)
        max_jump = MAX_POSITION_JUMP_M * (time_delta * self.fps)  # Scale by frames elapsed
        
        if distance_m > max_jump and self.total_samples > 1:
            # This is likely a tracking error
            is_valid = False
            rejection_reason = f"Position jump ({distance_m:.2f}m > {max_jump:.2f}m max)"
            self.rejected_samples += 1
            self.consecutive_outliers += 1
            
            # If many consecutive outliers, player may have genuinely teleported
            # (e.g., camera cut, scene change) - accept new baseline
            if self.consecutive_outliers >= self.max_consecutive_outliers:
                logger.debug(f"Accepting new baseline after {self.consecutive_outliers} outliers")
                self.consecutive_outliers = 0
                raw_speed_mps = 0.0  # Reset to stationary
            else:
                # Use last valid speed instead of the spike
                raw_speed_mps = self.last_valid_speed
        else:
            self.consecutive_outliers = 0
        
        # =========================================
        # STEP 2: Hard sanity check
        # =========================================
        if raw_speed_mps > MAX_HUMAN_SPEED_MPS:
            is_valid = False
            rejection_reason = f"Exceeds human limit ({raw_speed_mps:.1f} > {MAX_HUMAN_SPEED_MPS} m/s)"
            self.clamped_samples += 1
            raw_speed_mps = self.last_valid_speed  # Reject completely
        
        # =========================================
        # STEP 3: Median filter (spike rejection)
        # =========================================
        if self.median_filter is not None:
            raw_speed_mps = self.median_filter.filter(raw_speed_mps)
        
        # =========================================
        # STEP 4: Kalman filter (smooth estimation)
        # =========================================
        if self.kalman_filter is not None:
            raw_speed_mps = self.kalman_filter.update(raw_speed_mps, max(time_delta, MIN_TIME_INTERVAL))
        
        # =========================================
        # STEP 5: Final clamp to max speed
        # =========================================
        final_speed = min(raw_speed_mps, self.max_speed)
        final_speed = max(0.0, final_speed)  # No negative speeds
        
        # Update last valid speed
        if is_valid:
            self.last_valid_speed = final_speed
        
        return final_speed, is_valid, rejection_reason
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics for debugging."""
        return {
            "total_samples": self.total_samples,
            "rejected_samples": self.rejected_samples,
            "clamped_samples": self.clamped_samples,
            "rejection_rate": self.rejected_samples / max(1, self.total_samples),
            "last_valid_speed": self.last_valid_speed
        }
    
    def reset(self):
        """Reset all filter states."""
        if self.median_filter:
            self.median_filter.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self.last_valid_speed = 0.0
        self.consecutive_outliers = 0
        self.total_samples = 0
        self.rejected_samples = 0
        self.clamped_samples = 0


class SpeedZone(Enum):
    """
    Speed zones based on realistic badminton player movement patterns.
    
    Thresholds based on research on badminton player movement speeds:
    - Standing: < 0.5 m/s (stationary, preparing for next shot)
    - Walking: 0.5 - 1.5 m/s (1.8 - 5.4 km/h) - repositioning between points
    - Active: 1.5 - 3.0 m/s (5.4 - 11 km/h) - normal footwork during rallies
    - Running: 3.0 - 5.0 m/s (11 - 18 km/h) - quick court coverage
    - Sprinting: 5.0 - 7.0 m/s (18 - 25 km/h) - fast recovery moves
    - Explosive: > 7.0 m/s (> 25 km/h) - lunges, dives, explosive bursts (RARE)
    
    NOTE: Most badminton movement should be in Standing/Active/Running zones.
    Sprinting and Explosive should be rare, brief moments. If you see
    sustained Explosive readings, it indicates a tracking/calibration issue.
    """
    STANDING = "standing"
    WALKING = "walking"
    ACTIVE = "active"  # Renamed from JOGGING - more appropriate for badminton
    RUNNING = "running"
    SPRINTING = "sprinting"
    EXPLOSIVE = "explosive"


# Speed zone thresholds in m/s
# Based on realistic badminton movement patterns
SPEED_ZONE_THRESHOLDS = {
    SpeedZone.STANDING: (0.0, 0.5),    # 0 - 1.8 km/h
    SpeedZone.WALKING: (0.5, 1.5),     # 1.8 - 5.4 km/h
    SpeedZone.ACTIVE: (1.5, 3.0),      # 5.4 - 11 km/h - typical footwork
    SpeedZone.RUNNING: (3.0, 5.0),     # 11 - 18 km/h - quick movement
    SpeedZone.SPRINTING: (5.0, 7.0),   # 18 - 25 km/h - fast recovery
    SpeedZone.EXPLOSIVE: (7.0, float('inf'))  # > 25 km/h - lunges/dives (RARE)
}

# Speed zone colors (for visualization)
SPEED_ZONE_COLORS = {
    SpeedZone.STANDING: "#94A3B8",   # Slate gray
    SpeedZone.WALKING: "#22C55E",    # Green
    SpeedZone.ACTIVE: "#3B82F6",     # Blue
    SpeedZone.RUNNING: "#F59E0B",    # Amber
    SpeedZone.SPRINTING: "#EF4444",  # Red
    SpeedZone.EXPLOSIVE: "#9333EA"   # Purple
}


def classify_speed_zone(speed_mps: float) -> SpeedZone:
    """
    Classify a speed value into a SpeedZone.
    
    Args:
        speed_mps: Speed in meters per second
        
    Returns:
        SpeedZone enum value
    """
    for zone, (min_speed, max_speed) in SPEED_ZONE_THRESHOLDS.items():
        if min_speed <= speed_mps < max_speed:
            return zone
    return SpeedZone.STANDING


@dataclass
class SpeedDataPoint:
    """Single speed measurement for a player at a specific time."""
    frame_number: int
    timestamp: float  # seconds from video start
    speed_mps: float  # meters per second
    speed_kmh: float  # kilometers per hour
    position_x: float  # pixel x position
    position_y: float  # pixel y position
    court_x: Optional[float] = None  # court x in meters (if available)
    court_y: Optional[float] = None  # court y in meters (if available)
    zone: SpeedZone = field(default=SpeedZone.STANDING)
    smoothed: bool = False  # whether this value is smoothed
    
    def to_dict(self) -> Dict:
        return {
            "frame": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "speed_mps": round(self.speed_mps, 3),
            "speed_kmh": round(self.speed_kmh, 2),
            "position": {"x": self.position_x, "y": self.position_y},
            "court_position": {"x": self.court_x, "y": self.court_y} if self.court_x is not None else None,
            "zone": self.zone.value,
            "zone_color": SPEED_ZONE_COLORS[self.zone],
            "smoothed": self.smoothed
        }


@dataclass
class PlayerSpeedStats:
    """Aggregate speed statistics for a player."""
    player_id: int
    current_speed_mps: float = 0.0
    current_speed_kmh: float = 0.0
    max_speed_mps: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_mps: float = 0.0
    avg_speed_kmh: float = 0.0
    total_distance_m: float = 0.0
    current_zone: SpeedZone = SpeedZone.STANDING
    
    # Zone time distribution (percentage) - keyed by SpeedZone enum
    zone_time_distribution: Dict[SpeedZone, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "current": {
                "speed_mps": round(self.current_speed_mps, 3),
                "speed_kmh": round(self.current_speed_kmh, 2),
                "zone": self.current_zone.value,
                "zone_color": SPEED_ZONE_COLORS[self.current_zone]
            },
            "max": {
                "speed_mps": round(self.max_speed_mps, 3),
                "speed_kmh": round(self.max_speed_kmh, 2)
            },
            "avg": {
                "speed_mps": round(self.avg_speed_mps, 3),
                "speed_kmh": round(self.avg_speed_kmh, 2)
            },
            "total_distance_m": round(self.total_distance_m, 2),
            "zone_distribution": {
                zone.value: round(pct, 1) 
                for zone, pct in self.zone_time_distribution.items()
            }
        }


class ExponentialSmoother:
    """
    Exponential moving average smoother for reducing noise in speed data.
    
    Uses exponential smoothing to reduce the impact of detection jitter
    while preserving genuine acceleration/deceleration patterns.
    
    Formula: smoothed_value = alpha * current + (1 - alpha) * previous_smoothed
    
    Higher alpha (0.7-0.9): More responsive, less smoothing
    Lower alpha (0.2-0.5): More smoothing, slower response
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Initialize the smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing.
        """
        self.alpha = alpha
        self.previous_value: Optional[float] = None
        self.initialized = False
    
    def smooth(self, value: float) -> float:
        """
        Apply exponential smoothing to a value.
        
        Args:
            value: Raw value to smooth
            
        Returns:
            Smoothed value
        """
        if not self.initialized:
            self.previous_value = value
            self.initialized = True
            return value
        
        smoothed = self.alpha * value + (1 - self.alpha) * (self.previous_value or 0)
        self.previous_value = smoothed
        return smoothed
    
    def reset(self):
        """Reset the smoother state."""
        self.previous_value = None
        self.initialized = False


class AdaptiveSmoother:
    """
    Adaptive smoother that adjusts smoothing based on movement detection.
    
    When the player is stationary or moving slowly, applies more smoothing
    to reduce jitter. When moving quickly, applies less smoothing to
    preserve genuine speed changes.
    
    This prevents the "lag" effect during rapid movements while maintaining
    stability during slow movements.
    """
    
    def __init__(
        self,
        min_alpha: float = 0.2,
        max_alpha: float = 0.8,
        speed_threshold_mps: float = 2.0
    ):
        """
        Initialize adaptive smoother.
        
        Args:
            min_alpha: Smoothing factor for stationary/slow movement
            max_alpha: Smoothing factor for fast movement
            speed_threshold_mps: Speed at which to use max_alpha
        """
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.speed_threshold = speed_threshold_mps
        self.previous_value: Optional[float] = None
        self.initialized = False
    
    def smooth(self, value: float) -> float:
        """
        Apply adaptive smoothing based on current speed.
        """
        if not self.initialized:
            self.previous_value = value
            self.initialized = True
            return value
        
        # Calculate adaptive alpha based on speed
        # Fast movement = high alpha (less smoothing)
        # Slow movement = low alpha (more smoothing)
        speed_ratio = min(1.0, abs(value) / self.speed_threshold)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * speed_ratio
        
        smoothed = alpha * value + (1 - alpha) * (self.previous_value or 0)
        self.previous_value = smoothed
        return smoothed
    
    def reset(self):
        """Reset the smoother state."""
        self.previous_value = None
        self.initialized = False


class PlayerSpeedTracker:
    """
    Tracks speed data for a single player with sliding window history.
    
    Features:
    - Maintains historical speed data for graphing
    - Calculates real-world speed using court-based calibration
    - Applies smoothing to reduce detection noise
    - Tracks statistics (current, max, average speeds)
    - Classifies movement into speed zones
    """
    
    def __init__(
        self,
        player_id: int,
        fps: float = 30.0,
        window_seconds: float = 60.0,
        smoothing_alpha: float = 0.4,
        use_adaptive_smoothing: bool = True,
        use_robust_filtering: bool = True
    ):
        """
        Initialize the speed tracker.
        
        Args:
            player_id: Unique player identifier
            fps: Video frame rate
            window_seconds: Sliding window size in seconds (30-60 recommended)
            smoothing_alpha: Base smoothing factor
            use_adaptive_smoothing: Whether to use adaptive smoothing (if not using robust)
            use_robust_filtering: Whether to use robust filtering (recommended, overrides adaptive)
        """
        self.player_id = player_id
        self.fps = fps
        self.window_seconds = window_seconds
        self.max_data_points = int(window_seconds * fps)
        self.use_robust_filtering = use_robust_filtering
        
        # Speed history (sliding window)
        self.speed_history: Deque[SpeedDataPoint] = deque(maxlen=self.max_data_points)
        
        # Full history for post-match analysis
        self.full_history: List[SpeedDataPoint] = []
        
        # Position tracking
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_frame: int = -1
        self.last_timestamp: float = 0.0
        
        # Court position tracking
        self.last_court_position: Optional[Tuple[float, float]] = None
        
        # Filtering: Use robust filter (recommended) or fallback to legacy smoothing
        if use_robust_filtering:
            # RobustSpeedFilter combines: position jump detection, sanity checks,
            # median filter, Kalman filter, and final clamping
            self.robust_filter = RobustSpeedFilter(
                fps=fps,
                use_kalman=True,
                use_median=True,
                max_speed_mps=MAX_BADMINTON_SPEED_MPS
            )
            self.smoother = None  # Not used with robust filtering
        else:
            # Legacy smoothing (kept for comparison/debugging)
            self.robust_filter = None
            if use_adaptive_smoothing:
                self.smoother = AdaptiveSmoother(min_alpha=0.3, max_alpha=0.8)
            else:
                self.smoother = ExponentialSmoother(alpha=smoothing_alpha)
        
        # Statistics tracking
        self.total_distance_m = 0.0
        self.max_speed_mps = 0.0
        self.zone_frame_counts: Dict[SpeedZone, int] = {zone: 0 for zone in SpeedZone}
        
        # Debugging: Track rejection statistics
        self.rejected_count = 0
        self.total_updates = 0
        
        # Court detector reference
        self.court_detector = get_court_detector()
    
    def update(
        self,
        frame_number: int,
        pixel_position: Tuple[float, float],
        court_detection: Optional[CourtDetection] = None,
        fallback_court_width_pixels: float = 800.0
    ) -> SpeedDataPoint:
        """
        Update speed tracker with new position data.
        
        Args:
            frame_number: Current frame number
            pixel_position: (x, y) position in pixels
            court_detection: Court detection result for accurate measurements
            fallback_court_width_pixels: Fallback court width if no detection
            
        Returns:
            SpeedDataPoint with calculated speed
        """
        timestamp = frame_number / self.fps
        
        # Calculate speed
        speed_mps = 0.0
        distance_m = 0.0
        court_x, court_y = None, None
        
        if self.last_position is not None and self.last_frame >= 0:
            # Calculate time difference
            time_diff = timestamp - self.last_timestamp
            
            if time_diff > 0:
                # Calculate distance
                if court_detection and court_detection.homography_matrix is not None:
                    # Use homography for accurate measurement
                    distance_m = self.court_detector.calculate_real_distance(
                        self.last_position, pixel_position, court_detection
                    )
                    if distance_m is None:
                        distance_m = self._fallback_distance(
                            self.last_position, pixel_position, fallback_court_width_pixels
                        )
                    
                    # Get court coordinates
                    court_coords = self.court_detector.pixel_to_court_coords(
                        pixel_position[0], pixel_position[1], court_detection
                    )
                    if court_coords:
                        court_x, court_y = court_coords
                else:
                    # Fallback to pixel-based calculation
                    distance_m = self._fallback_distance(
                        self.last_position, pixel_position, fallback_court_width_pixels
                    )
                
                # Calculate raw speed (m/s)
                raw_speed_mps = distance_m / time_diff
                self.total_updates += 1
                
                # Apply filtering (robust or legacy)
                if self.robust_filter is not None:
                    # Use robust filtering pipeline (recommended)
                    speed_mps, is_valid, rejection_reason = self.robust_filter.filter(
                        raw_speed_mps=raw_speed_mps,
                        distance_m=distance_m,
                        time_delta=time_diff
                    )
                    
                    if not is_valid:
                        self.rejected_count += 1
                        logger.debug(
                            f"Player {self.player_id} frame {frame_number}: "
                            f"Raw {raw_speed_mps:.1f} m/s -> Filtered {speed_mps:.1f} m/s "
                            f"({rejection_reason})"
                        )
                elif self.smoother is not None:
                    # Legacy smoothing (fallback)
                    speed_mps = self.smoother.smooth(raw_speed_mps)
                else:
                    speed_mps = raw_speed_mps
                
                # Accumulate distance (only for valid movements)
                if distance_m < MAX_POSITION_JUMP_M * (time_diff * self.fps):
                    self.total_distance_m += distance_m
        
        # Update max speed
        if speed_mps > self.max_speed_mps:
            self.max_speed_mps = speed_mps
        
        # Classify zone
        zone = classify_speed_zone(speed_mps)
        self.zone_frame_counts[zone] += 1
        
        # Create data point
        data_point = SpeedDataPoint(
            frame_number=frame_number,
            timestamp=timestamp,
            speed_mps=speed_mps,
            speed_kmh=speed_mps * 3.6,  # Convert m/s to km/h
            position_x=pixel_position[0],
            position_y=pixel_position[1],
            court_x=court_x,
            court_y=court_y,
            zone=zone,
            smoothed=True
        )
        
        # Store in history
        self.speed_history.append(data_point)
        self.full_history.append(data_point)
        
        # Update last position
        self.last_position = pixel_position
        self.last_frame = frame_number
        self.last_timestamp = timestamp
        # Only set court position if both coordinates are available
        if court_x is not None and court_y is not None:
            self.last_court_position = (court_x, court_y)
        else:
            self.last_court_position = None
        
        return data_point
    
    def _fallback_distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        court_width_pixels: float
    ) -> float:
        """
        Calculate distance using simple pixel-to-meter conversion.
        
        Args:
            p1: First position (pixels)
            p2: Second position (pixels)
            court_width_pixels: Estimated court width in pixels
            
        Returns:
            Distance in meters
        """
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        meters_per_pixel = COURT_WIDTH_DOUBLES / court_width_pixels
        return pixel_distance * meters_per_pixel
    
    def get_window_data(self) -> List[Dict]:
        """
        Get speed data for the sliding window (for real-time graphing).
        
        Returns:
            List of speed data points as dictionaries
        """
        return [dp.to_dict() for dp in self.speed_history]
    
    def get_full_history(self) -> List[Dict]:
        """
        Get complete speed history (for post-match analysis).
        
        Returns:
            List of all speed data points as dictionaries
        """
        return [dp.to_dict() for dp in self.full_history]
    
    def get_statistics(self) -> PlayerSpeedStats:
        """
        Calculate current speed statistics.
        
        Returns:
            PlayerSpeedStats with current metrics
        """
        current_speed_mps = 0.0
        current_zone = SpeedZone.STANDING
        
        if self.speed_history:
            current_point = self.speed_history[-1]
            current_speed_mps = current_point.speed_mps
            current_zone = current_point.zone
        
        # Calculate averages
        total_frames = sum(self.zone_frame_counts.values())
        zone_distribution = {}
        if total_frames > 0:
            for zone, count in self.zone_frame_counts.items():
                zone_distribution[zone] = (count / total_frames) * 100
        
        avg_speed_mps = 0.0
        if self.full_history:
            avg_speed_mps = sum(dp.speed_mps for dp in self.full_history) / len(self.full_history)
        
        return PlayerSpeedStats(
            player_id=self.player_id,
            current_speed_mps=current_speed_mps,
            current_speed_kmh=current_speed_mps * 3.6,
            max_speed_mps=self.max_speed_mps,
            max_speed_kmh=self.max_speed_mps * 3.6,
            avg_speed_mps=avg_speed_mps,
            avg_speed_kmh=avg_speed_mps * 3.6,
            total_distance_m=self.total_distance_m,
            current_zone=current_zone,
            zone_time_distribution=zone_distribution
        )
    
    def reset(self):
        """Reset all tracking state."""
        self.speed_history.clear()
        self.full_history.clear()
        self.last_position = None
        self.last_frame = -1
        self.last_timestamp = 0.0
        self.last_court_position = None
        self.total_distance_m = 0.0
        self.max_speed_mps = 0.0
        self.zone_frame_counts = {zone: 0 for zone in SpeedZone}
        self.rejected_count = 0
        self.total_updates = 0
        
        # Reset the appropriate filter
        if self.robust_filter is not None:
            self.robust_filter.reset()
        elif self.smoother is not None:
            self.smoother.reset()
    
    def get_filter_statistics(self) -> Dict:
        """
        Get filtering statistics for debugging aberrant speed values.
        
        Returns:
            Dictionary with rejection rate and filter state info
        """
        stats = {
            "player_id": self.player_id,
            "total_updates": self.total_updates,
            "rejected_count": self.rejected_count,
            "rejection_rate": self.rejected_count / max(1, self.total_updates),
            "filter_type": "robust" if self.robust_filter else "legacy"
        }
        
        if self.robust_filter:
            stats["robust_filter_stats"] = self.robust_filter.get_statistics()
        
        return stats


class SpeedGraphManager:
    """
    Manages speed tracking for multiple players and provides graph data.
    
    This is the main interface for the speed visualization system.
    It handles:
    - Multi-player speed tracking
    - Graph data generation for the frontend
    - Statistics aggregation
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 60.0,
        smoothing_alpha: float = 0.4
    ):
        """
        Initialize the speed graph manager.
        
        Args:
            fps: Video frame rate
            window_seconds: Sliding window size for graph display
            smoothing_alpha: Smoothing factor for speed calculations
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.smoothing_alpha = smoothing_alpha
        
        # Player trackers
        self.players: Dict[int, PlayerSpeedTracker] = {}
        
        # Court detector
        self.court_detector = get_court_detector()
        
        # Latest court detection
        self.current_court_detection: Optional[CourtDetection] = None
    
    def update_player(
        self,
        player_id: int,
        frame_number: int,
        pixel_position: Tuple[float, float],
        court_detection: Optional[CourtDetection] = None
    ) -> SpeedDataPoint:
        """
        Update speed data for a player.
        
        Args:
            player_id: Player identifier (1 or 2)
            frame_number: Current frame number
            pixel_position: Player position in pixels
            court_detection: Optional court detection for accuracy
            
        Returns:
            SpeedDataPoint with calculated speed
        """
        # Create tracker if needed
        if player_id not in self.players:
            self.players[player_id] = PlayerSpeedTracker(
                player_id=player_id,
                fps=self.fps,
                window_seconds=self.window_seconds,
                smoothing_alpha=self.smoothing_alpha
            )
        
        # Store court detection
        if court_detection:
            self.current_court_detection = court_detection
        
        # Update tracker
        return self.players[player_id].update(
            frame_number=frame_number,
            pixel_position=pixel_position,
            court_detection=court_detection or self.current_court_detection
        )
    
    def get_graph_data(self) -> Dict:
        """
        Get data formatted for the speed graph visualization.
        
        Returns:
            Dictionary with:
            - players: Dict of player_id -> speed history
            - statistics: Dict of player_id -> stats
            - zone_thresholds: Speed zone threshold data
            - time_range: Current time window range
        """
        players_data = {}
        statistics = {}
        
        for player_id, tracker in self.players.items():
            players_data[player_id] = {
                "window_data": tracker.get_window_data(),
                "full_history": tracker.get_full_history()
            }
            statistics[player_id] = tracker.get_statistics().to_dict()
        
        # Build zone threshold data for graph reference lines
        # NOTE: JSON doesn't support float('inf'), so convert to None for serialization
        zone_thresholds = [
            {
                "zone": zone.value,
                "min_mps": thresholds[0],
                "max_mps": thresholds[1] if thresholds[1] != float('inf') else None,
                "min_kmh": thresholds[0] * 3.6,
                "max_kmh": thresholds[1] * 3.6 if thresholds[1] != float('inf') else None,
                "color": SPEED_ZONE_COLORS[zone]
            }
            for zone, thresholds in SPEED_ZONE_THRESHOLDS.items()
        ]
        
        # Calculate time range
        time_range = None
        all_times = []
        for tracker in self.players.values():
            if tracker.speed_history:
                all_times.extend([dp.timestamp for dp in tracker.speed_history])
        
        if all_times:
            time_range = {
                "min": min(all_times),
                "max": max(all_times)
            }
        
        return {
            "players": players_data,
            "statistics": statistics,
            "zone_thresholds": zone_thresholds,
            "time_range": time_range,
            "window_seconds": self.window_seconds,
            "fps": self.fps
        }
    
    def get_current_speeds(self) -> Dict[int, Dict]:
        """
        Get current speed for all players (for real-time display).
        
        Returns:
            Dict of player_id -> current speed data
        """
        return {
            player_id: {
                "speed_mps": tracker.get_statistics().current_speed_mps,
                "speed_kmh": tracker.get_statistics().current_speed_kmh,
                "zone": tracker.get_statistics().current_zone.value,
                "zone_color": SPEED_ZONE_COLORS[tracker.get_statistics().current_zone]
            }
            for player_id, tracker in self.players.items()
        }
    
    def reset(self):
        """Reset all player trackers."""
        for tracker in self.players.values():
            tracker.reset()
        self.players.clear()
        self.current_court_detection = None


# Global manager instance
_speed_manager: Optional[SpeedGraphManager] = None


def get_speed_manager(
    fps: float = 30.0,
    window_seconds: float = 60.0,
    reset: bool = False
) -> SpeedGraphManager:
    """
    Get or create the global speed manager instance.
    
    Args:
        fps: Video frame rate
        window_seconds: Sliding window size
        reset: If True, reset the manager
        
    Returns:
        SpeedGraphManager instance
    """
    global _speed_manager
    
    if _speed_manager is None or reset:
        _speed_manager = SpeedGraphManager(
            fps=fps,
            window_seconds=window_seconds
        )
    
    return _speed_manager


def calculate_speed_from_skeleton_data(
    skeleton_frames: List[Dict],
    fps: float = 30.0,
    window_seconds: float = 60.0,
    video_width: int = 1920,
    video_height: int = 1080
) -> Dict:
    """
    Calculate speed data from pre-computed skeleton frames.
    
    This is useful for post-processing already analyzed videos.
    
    IMPORTANT: This function now properly uses manual court keypoints if set.
    Manual keypoints provide more accurate homography for speed calculations
    compared to the automatic detection that was used during video processing.
    
    Args:
        skeleton_frames: List of skeleton frame dictionaries
        fps: Video frame rate
        window_seconds: Window size for output data
        video_width: Video frame width (for creating detection from manual keypoints)
        video_height: Video frame height (for creating detection from manual keypoints)
        
    Returns:
        Speed graph data dictionary
    """
    manager = SpeedGraphManager(fps=fps, window_seconds=window_seconds)
    court_detector = get_court_detector()
    
    # CRITICAL FIX: Check if manual keypoints are set on the court detector
    # If so, create a CourtDetection object from them to use for ALL frames
    # This ensures that speeds are recalculated using the user-specified court corners
    manual_court_detection = None
    manual_keypoints = court_detector.get_manual_keypoints()
    
    if manual_keypoints is not None:
        logger.info(f"Using manual keypoints for speed recalculation: {manual_keypoints.to_dict()}")
        
        # Create a CourtDetection from manual keypoints
        # We need to calculate the homography matrix from the manual corners
        corners = manual_keypoints.to_numpy()
        homography = court_detector._calculate_homography(corners)
        
        if homography is not None:
            manual_court_detection = CourtDetection(
                regions=[],  # No automatic region detection
                keypoints=[],  # No automatic keypoints
                court_corners=corners,
                homography_matrix=homography,
                confidence=1.0,  # Manual = 100% confidence
                detected=True,
                frame_width=video_width,
                frame_height=video_height,
                detection_source="manual",
                manual_keypoints=manual_keypoints
            )
            logger.info(f"Created CourtDetection from manual keypoints with homography")
        else:
            logger.warning("Failed to calculate homography from manual keypoints")
    
    for frame_data in skeleton_frames:
        frame_number = frame_data.get("frame", 0)
        players = frame_data.get("players", [])
        
        # Determine court detection to use for this frame:
        # 1. If manual keypoints are set, ALWAYS use them (most accurate)
        # 2. Otherwise, check if frame had court detection during processing
        if manual_court_detection is not None:
            # Use manual keypoints for all frames
            court_detection = manual_court_detection
        elif frame_data.get("court_detected", False):
            # Fall back to cached court detection if available
            court_detection = manager.current_court_detection
        else:
            court_detection = None
        
        for player in players:
            player_id = player.get("player_id", 1)
            center = player.get("center")
            
            if center and center.get("x") is not None:
                manager.update_player(
                    player_id=player_id,
                    frame_number=frame_number,
                    pixel_position=(center["x"], center["y"]),
                    court_detection=court_detection
                )
    
    # Add metadata about whether manual keypoints were used
    graph_data = manager.get_graph_data()
    graph_data["manual_keypoints_used"] = manual_court_detection is not None
    graph_data["detection_source"] = "manual" if manual_court_detection else "auto"
    
    return graph_data
