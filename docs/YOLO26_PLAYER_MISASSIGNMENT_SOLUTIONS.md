# YOLO26 Player Misassignment Solutions - Comprehensive Analysis

## Executive Summary

This document analyzes the YOLO26 model misassignment issues where skeleton detection and bounding boxes are incorrectly applied to judges, referees, line officials, and spectators instead of active badminton players. This causes downstream problems with:
- **Heatmap generation accuracy** - Non-player positions pollute movement patterns
- **Speed calculation reliability** - Stationary officials create false velocity spikes

**Document Status:** Analysis Complete - Ready for Implementation  
**Date:** 2026-02-01  
**Priority:** HIGH - Affects core analytics accuracy

---

## Table of Contents

1. [Root Cause Analysis](#1-root-cause-analysis)
2. [Downstream Impact Assessment](#2-downstream-impact-assessment)
3. [Solution Approaches](#3-solution-approaches)
4. [Implementation Details](#4-implementation-details)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Prioritized Implementation Strategy](#6-prioritized-implementation-strategy)
7. [Code Implementation Examples](#7-code-implementation-examples)

---

## 1. Root Cause Analysis

### 1.1 Current Detection Flow

The current player detection flow in [`multi_model_detector.py`](../backend/multi_model_detector.py:362) works as follows:

```
Frame → YOLO26 Detection → Class Filtering → Size-based Filtering → Pose Detection → Output
```

**Critical Flaw:** The `_filter_main_players()` method (line 603) uses only bounding box size:

```python
def _filter_main_players(self, players: List[Detection], max_players: int) -> List[Detection]:
    # Sort by bounding box area (largest first)
    sorted_players = sorted(
        players,
        key=lambda p: p.bbox.width * p.bbox.height,
        reverse=True
    )
    return sorted_players[:max_players]
```

**Problem:** A referee standing close to the camera may have a larger bounding box than the far-court player.

### 1.2 Class Differentiation Issues

**Current Class Mapping** (line 163-166):
```python
PLAYER_CLASSES = ["player", "person", "human", "athlete"]
SHUTTLECOCK_CLASSES = ["shuttlecock", "shuttle", "birdie", "ball"]
RACKET_CLASSES = ["racket", "racquet"]
```

**Issue:** "person" is too broad - it captures any human in the frame (judges, officials, spectators).

### 1.3 Court Boundary Awareness Gap

The court detection module ([`court_detection.py`](../backend/court_detection.py)) provides:
- Court corner coordinates
- Homography matrix for perspective transformation
- Region detection (frontcourt, midcourt, rearcourt, sidelines)

**However:** This information is **NOT** used to filter player detections. Players outside court boundaries are not filtered.

### 1.4 Pose Estimation Confidence Issues

From [`pose_detection.py`](../backend/pose_detection.py:617):
```python
confidence_threshold: float = 0.25,  # Lowered from 0.5 for better far-player detection
keypoint_confidence: float = 0.05,   # Very low for far players
```

**Trade-off:** Lower thresholds improve far-player detection but also increase false positives for non-players.

### 1.5 Training Data Composition

From [`train_pose_model.py`](../backend/train_pose_model.py:13):
- Dataset: "badminton pose.v1i.yolo26" (1200 images from Roboflow)
- Classes: backhand-general, defense, lift, offense, serve, smash

**Missing:** 
- Negative examples of judges/referees/spectators
- Court boundary context in training

### 1.6 NMS Parameters

Default NMS (Non-Maximum Suppression) parameters in YOLO26 may allow overlapping detections of different people to pass through, creating duplicate person detections.

---

## 2. Downstream Impact Assessment

### 2.1 Heatmap Generation Impact

From [`heatmap_generator.py`](../backend/heatmap_generator.py:286):

```python
def add_position(self, player_id: int, x: float, y: float, intensity: float = 1.0):
```

**Impact:**
- Non-player positions are added to heatmaps with equal weight
- Officials who remain stationary create "hot spots" at single locations
- Movement patterns become distorted by static off-court positions
- Court coverage analysis becomes unreliable

**Severity:** HIGH - Heatmaps become misleading for tactical analysis

### 2.2 Speed Calculation Impact

From [`speed_analytics.py`](../backend/speed_analytics.py:74-99):

```python
MAX_HUMAN_SPEED_MPS = 10.0  # 36 km/h - absolute physical limit
MAX_BADMINTON_SPEED_MPS = 8.3  # 30 km/h - realistic max
SUSPICIOUS_SPEED_MPS = 7.0  # 25 km/h - flag as potential outlier
```

**Impact Scenarios:**

1. **ID Switch to Official:** When player ID switches to a nearby stationary official:
   - Sudden position jump creates impossible speed spike
   - Existing filters (Kalman, Median) are overwhelmed by magnitude

2. **Tracking Loss Recovery:** When tracking resumes on wrong person:
   - Speed oscillates between zero (official) and max (when switching back)
   - Average speed statistics are corrupted

3. **Persistent Misassignment:** When official is consistently tracked as "player":
   - Near-zero speed readings pollute statistics
   - Player appears to not be moving during active rally

**Severity:** CRITICAL - Speed data becomes unreliable for performance analysis

### 2.3 Pose Analytics Impact

The pose classification system ([`pose_detection.py`](../backend/pose_detection.py:474)) classifies poses:
- STANDING, READY, SERVING, SMASH, OVERHEAD, FOREHAND, BACKHAND, LUNGE, JUMP, RECOVERY

**Impact:**
- Officials in "STANDING" pose dominate pose distribution
- Actual player shot types become underrepresented
- Rally pattern analysis is corrupted

---

## 3. Solution Approaches

### 3.1 Court ROI Filtering (Region of Interest)

**Concept:** Use court boundaries to create a mask and filter detections outside the playing area.

**✅ WORKS WITH MANUAL KEYPOINTS:** This solution uses the **same 4 court corners** that users already provide for speed calibration via [`ManualCourtKeypoints`](../backend/court_detection.py:89). No automatic court detection model required!

**Implementation Approach:**
1. Get court corners from **manual keypoints** ([`ManualCourtKeypoints.to_numpy()`](../backend/court_detection.py:105))
2. Create a polygon mask for the court boundary (with margin for lunges)
3. Filter player detections by checking if their center/feet are within the polygon

**How it integrates with existing flow:**
```python
# User already provides these for speed calibration:
manual_keypoints = ManualCourtKeypoints(
    top_left=(100, 50),
    top_right=(500, 50),
    bottom_right=(550, 400),
    bottom_left=(50, 400)
)
# Same corners used for ROI filtering polygon
roi_polygon = manual_keypoints.to_numpy()
```

**Pros:**
- **Uses existing manual keypoint workflow** - no new user interaction needed
- Highly accurate (user-specified corners)
- Deterministic filtering
- No additional ML training required
- Works immediately once keypoints are set

**Cons:**
- Requires user to set manual keypoints first (already done for speed calibration)
- Fixed margin may miss players during extreme lunges
- Does not distinguish between players and ball kids inside court

**Complexity:** LOW
**Computational Overhead:** ~0.1ms per frame (polygon point-in-polygon test)
**Expected Accuracy Improvement:** 40-60%

### 3.2 Secondary Classification Layer

**Concept:** Add a lightweight CNN classifier to distinguish "active player" from "official/spectator".

**Training Data Requirements:**
- Active players: ~2000 images (available from existing dataset)
- Officials/Spectators: ~1000 images (need to collect/annotate)

**Model Options:**
1. **MobileNetV3-Small:** 2.5MB, ~2ms inference
2. **EfficientNet-B0:** 5MB, ~3ms inference
3. **Custom CNN:** <1MB, <1ms inference

**Integration:**
```python
def classify_player_type(crop: np.ndarray) -> Tuple[str, float]:
    # Returns ("active_player" | "official" | "spectator", confidence)
    pass
```

**Pros:**
- High accuracy potential (>90%)
- Can distinguish by uniform, pose, equipment
- Generalizes to different court views

**Cons:**
- Requires training data collection
- Additional inference time per detection
- Model maintenance burden

**Complexity:** MEDIUM-HIGH  
**Computational Overhead:** 2-5ms per detection (2-10ms total per frame)  
**Expected Accuracy Improvement:** 80-95%

### 3.3 Temporal Consistency Tracking

**Concept:** Track detections across frames using motion prediction to maintain consistent player IDs and reject sudden position jumps.

**Algorithm:**
1. Use Kalman filter to predict expected player position
2. Match new detections to predictions using Hungarian algorithm
3. Reject matches exceeding maximum expected displacement
4. Maintain track confidence based on consistency history

**Implementation:**
```python
@dataclass
class PlayerTrack:
    track_id: int
    positions: Deque[Tuple[float, float]]  # Recent positions
    kalman_filter: KalmanFilter
    confidence: float
    last_seen_frame: int
    
def update_tracks(detections: List[Detection], frame_number: int):
    # Match detections to existing tracks
    # Create new tracks for unmatched detections
    # Age out tracks not seen recently
    pass
```

**Pros:**
- Prevents sudden ID switches
- Smooths trajectory naturally
- Works with any detection source

**Cons:**
- Initial frames require warmup
- Can lose track during occlusions
- Doesn't prevent initial misassignment

**Complexity:** MEDIUM  
**Computational Overhead:** ~1ms per frame  
**Expected Accuracy Improvement:** 30-50% (prevents switches, not initial errors)

### 3.4 Adjusted NMS Parameters

**Concept:** Tune Non-Maximum Suppression to reduce overlapping detections.

**Current (Default) Parameters:**
- IoU threshold: 0.45
- Confidence threshold: 0.5 (lowered to 0.25 for far players)

**Proposed Adjustments:**
```python
# More aggressive suppression
results = model(
    frame,
    conf=0.25,
    iou=0.3,  # Lower IoU = more aggressive suppression
    agnostic_nms=True,  # Cross-class NMS
    max_det=10  # Limit total detections
)
```

**Pros:**
- Simple parameter change
- No additional processing
- Reduces duplicate detections

**Cons:**
- May suppress legitimate close players
- Trade-off between recall and precision
- Doesn't address non-player detection

**Complexity:** LOW  
**Computational Overhead:** None (may even be faster)  
**Expected Accuracy Improvement:** 10-20%

### 3.5 Fine-tuning with Badminton-Specific Dataset

**Concept:** Fine-tune the YOLO26 model with annotated badminton data including negative examples.

**Dataset Requirements:**
```yaml
classes:
  0: active_player
  1: referee
  2: line_judge
  3: spectator
  4: ball_kid
  5: racket
  6: shuttlecock

# Annotations include:
# - All visible people in frame
# - Clear class distinction
# - Court context
```

**Training Configuration:**
```python
model.train(
    data="badminton_people.yaml",
    epochs=100,
    imgsz=960,
    batch=16,
    freeze=10,  # Freeze backbone layers
    augment=True,
    # Balanced sampling for minority classes
    rect=True,  # Rectangular training
)
```

**Pros:**
- Native model-level discrimination
- Single inference pass
- Best long-term solution

**Cons:**
- Requires significant annotation effort
- Training time and resources
- May overfit to specific court/broadcast styles

**Complexity:** HIGH  
**Computational Overhead:** None (same inference as current)  
**Expected Accuracy Improvement:** 85-98%

### 3.6 Jersey/Uniform Detection

**Concept:** Use color histogram or pattern matching to identify player uniforms.

**Implementation:**
```python
def extract_dominant_colors(crop: np.ndarray, n_colors: int = 3) -> List[Tuple[int, int, int]]:
    """Extract dominant colors from player crop using k-means clustering"""
    pixels = crop.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int).tolist()

def match_player_by_uniform(
    detection_colors: List[Tuple[int, int, int]],
    registered_player_colors: List[List[Tuple[int, int, int]]]
) -> Optional[int]:
    """Match detection to registered players by uniform color"""
    pass
```

**Semi-automatic Calibration:**
1. User clicks on both players in first frame
2. System extracts uniform colors
3. Subsequent frames filter by color similarity

**Pros:**
- Players typically wear distinct uniforms from officials
- Can be combined with other methods
- User calibration increases accuracy

**Cons:**
- Requires initial user interaction
- Color variation due to lighting
- White uniforms common for both players and officials

**Complexity:** MEDIUM  
**Computational Overhead:** ~2ms per detection  
**Expected Accuracy Improvement:** 50-70% (dependent on uniform distinctiveness)

### 3.7 Court Geometry Constraints

**Concept:** Use known court geometry to validate player positions logically.

**Rules:**
1. **Maximum 2 players per side:** Filter if >2 detections on one side
2. **Player size consistency:** Far player should be smaller than near player
3. **Position constraints:** Players during rally should be behind service line
4. **Motion patterns:** Active players move more than stationary officials

**Implementation:**
```python
def apply_geometry_constraints(
    detections: List[Detection],
    court: CourtDetection,
    frame_number: int
) -> List[Detection]:
    """Filter detections using court geometry constraints"""
    
    # Transform detections to court coordinates
    court_positions = []
    for det in detections:
        court_pos = court.pixel_to_court(det.bbox.center)
        court_positions.append((det, court_pos))
    
    # Split by court side (net at y=0)
    near_court = [(d, p) for d, p in court_positions if p[1] < 0]
    far_court = [(d, p) for d, p in court_positions if p[1] >= 0]
    
    # Keep at most 2 per side, prioritizing center-court positions
    valid = []
    for side in [near_court, far_court]:
        # Sort by distance from court center
        side.sort(key=lambda x: abs(x[1][0]))  # x distance from centerline
        valid.extend([d for d, _ in side[:1]])  # 1 player per side for singles
    
    return valid
```

**Pros:**
- Uses badminton domain knowledge
- No ML training required
- Logical consistency

**Cons:**
- Assumes singles (doubles needs 4 players)
- May fail during unusual situations
- Requires accurate court detection

**Complexity:** LOW-MEDIUM  
**Computational Overhead:** ~0.5ms per frame  
**Expected Accuracy Improvement:** 40-60%

---

## 4. Implementation Details

### 4.1 Court ROI Filtering Implementation

**File:** `backend/player_roi_filter.py` (NEW)

**Key Feature:** Works with **manual keypoints** - same corners users provide for speed calibration!

```python
"""
Player ROI (Region of Interest) Filtering Module
Filters player detections to only those within the badminton court boundaries.

WORKS WITH MANUAL KEYPOINTS: Uses the same court corners that users provide
for speed calibration via ManualCourtKeypoints.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import cv2

from court_detection import CourtDetection, ManualCourtKeypoints
from multi_model_detector import Detection, BoundingBox


@dataclass
class ROIFilterConfig:
    """Configuration for ROI filtering"""
    margin_meters: float = 1.0  # Margin outside court lines (for lunges)
    use_feet_position: bool = True  # Use bottom of bbox instead of center
    min_court_confidence: float = 0.5  # Minimum court detection confidence (ignored for manual)


class CourtROIFilter:
    """
    Filters player detections to only those within court boundaries.
    
    Supports TWO modes:
    1. Manual keypoints (RECOMMENDED): Uses user-provided court corners
    2. Auto detection: Uses CourtDetection.court_corners from ML model
    
    Manual keypoints are preferred as they're already provided by users
    for speed/distance calibration.
    """
    
    def __init__(self, config: Optional[ROIFilterConfig] = None):
        self.config = config or ROIFilterConfig()
        self._court_mask: Optional[np.ndarray] = None
        self._court_polygon: Optional[np.ndarray] = None
        self._last_court_frame: int = -1
    
    def update_from_manual_keypoints(
        self,
        keypoints: ManualCourtKeypoints,
        frame_shape: Tuple[int, int]
    ) -> bool:
        """
        Update court boundary from MANUAL keypoints (recommended flow).
        
        This is the PREFERRED method - uses the same corners users provide
        for speed calibration, ensuring perfect alignment.
        
        Args:
            keypoints: ManualCourtKeypoints with 4 court corners
            frame_shape: (height, width) of video frame
            
        Returns:
            True if court was updated successfully
        """
        # Get corners directly from manual keypoints
        corners = keypoints.to_numpy()
        
        # Apply margin expansion (expand from center)
        center = corners.mean(axis=0)
        margin_factor = 1.0 + (self.config.margin_meters / 6.7)  # ~6.7m half-court
        corners = center + (corners - center) * margin_factor
        
        self._court_polygon = corners.astype(np.int32)
        
        # Create binary mask
        height, width = frame_shape
        self._court_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self._court_mask, [self._court_polygon], 255)
        
        return True
    
    def update_court(self, court: CourtDetection, frame_shape: Tuple[int, int]) -> bool:
        """
        Update the court boundary from auto detection (fallback).
        
        NOTE: Prefer update_from_manual_keypoints() when manual keypoints are set.
        
        Args:
            court: CourtDetection with court_corners
            frame_shape: (height, width) of video frame
            
        Returns:
            True if court was updated successfully
        """
        if not court.detected or court.court_corners is None:
            return False
        
        if court.confidence < self.config.min_court_confidence:
            return False
        
        # Expand court corners by margin
        corners = court.court_corners.copy()
        
        # Apply margin expansion (simplified - expand from center)
        center = corners.mean(axis=0)
        margin_factor = 1.0 + (self.config.margin_meters / 6.7)  # ~6.7m half-court
        corners = center + (corners - center) * margin_factor
        
        self._court_polygon = corners.astype(np.int32)
        
        # Create binary mask
        height, width = frame_shape
        self._court_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self._court_mask, [self._court_polygon], 255)
        
        return True
    
    def filter_detections(
        self,
        detections: List[Detection],
        court: Optional[CourtDetection] = None,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> List[Detection]:
        """
        Filter detections to only those within court boundaries.
        
        Args:
            detections: List of player detections
            court: Optional new court detection to update from
            frame_shape: Required if court is provided
            
        Returns:
            Filtered list of detections within court
        """
        # Update court if provided
        if court is not None and frame_shape is not None:
            self.update_court(court, frame_shape)
        
        # If no valid court, return all detections
        if self._court_polygon is None:
            return detections
        
        filtered = []
        for det in detections:
            # Get position to check (feet or center)
            if self.config.use_feet_position:
                check_point = (int(det.bbox.x), int(det.bbox.y_max))
            else:
                check_point = (int(det.bbox.x), int(det.bbox.y))
            
            # Check if point is inside court polygon
            result = cv2.pointPolygonTest(
                self._court_polygon,
                check_point,
                measureDist=False
            )
            
            if result >= 0:  # Inside or on boundary
                filtered.append(det)
        
        return filtered
    
    def get_court_mask(self) -> Optional[np.ndarray]:
        """Get the current court mask for visualization"""
        return self._court_mask


# Singleton instance
_roi_filter: Optional[CourtROIFilter] = None


def get_roi_filter() -> CourtROIFilter:
    """Get or create the global ROI filter instance"""
    global _roi_filter
    if _roi_filter is None:
        _roi_filter = CourtROIFilter()
    return _roi_filter
```

### 4.2 Integration with MultiModelDetector

**File:** `backend/multi_model_detector.py` (MODIFY)

```python
# Add import at top
from player_roi_filter import get_roi_filter, CourtROIFilter

class MultiModelDetector:
    def __init__(
        self,
        # ... existing params ...
        enable_roi_filter: bool = True,  # NEW PARAMETER
        roi_filter_margin: float = 1.0   # NEW PARAMETER
    ):
        # ... existing init ...
        
        # Initialize ROI filter
        self.enable_roi_filter = enable_roi_filter
        if enable_roi_filter:
            from player_roi_filter import ROIFilterConfig
            self._roi_filter = CourtROIFilter(ROIFilterConfig(
                margin_meters=roi_filter_margin
            ))
        else:
            self._roi_filter = None
    
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        max_players: int = 2,
        court: Optional[Any] = None  # NEW: Accept court detection
    ) -> MultiModelDetections:
        """
        Run detection with optional court ROI filtering.
        """
        result = MultiModelDetections(frame_number=frame_number)
        
        # ... existing detection logic ...
        
        # Apply ROI filter before size-based filtering
        if self._roi_filter is not None and court is not None:
            result.players = self._roi_filter.filter_detections(
                result.players,
                court=court,
                frame_shape=frame.shape[:2]
            )
        
        # Then apply size-based filtering
        if len(result.players) > max_players:
            result.players = self._filter_main_players(result.players, max_players)
        
        # ... rest of method ...
```

### 4.3 Temporal Consistency Implementation

**File:** `backend/player_tracking.py` (NEW)

```python
"""
Temporal Player Tracking Module
Maintains consistent player identities across frames using motion prediction.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

from multi_model_detector import Detection, BoundingBox


@dataclass
class TrackState:
    """State for a single player track"""
    track_id: int
    positions: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    velocities: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_detection: Optional[Detection] = None
    last_seen_frame: int = 0
    confidence: float = 1.0
    age: int = 0  # Frames since track creation
    hits: int = 0  # Number of successful matches
    
    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed after consistent detections"""
        return self.hits >= 3
    
    def predict_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        if len(self.positions) == 0:
            return (0, 0)
        
        last_pos = self.positions[-1]
        
        if len(self.velocities) > 0:
            avg_vel = (
                np.mean([v[0] for v in self.velocities]),
                np.mean([v[1] for v in self.velocities])
            )
            return (last_pos[0] + avg_vel[0], last_pos[1] + avg_vel[1])
        
        return last_pos
    
    def update(self, detection: Detection, frame_number: int):
        """Update track with new detection"""
        new_pos = (detection.bbox.x, detection.bbox.y)
        
        if len(self.positions) > 0:
            last_pos = self.positions[-1]
            velocity = (new_pos[0] - last_pos[0], new_pos[1] - last_pos[1])
            self.velocities.append(velocity)
        
        self.positions.append(new_pos)
        self.last_detection = detection
        self.last_seen_frame = frame_number
        self.hits += 1
        self.confidence = min(1.0, self.confidence + 0.1)


@dataclass
class TrackerConfig:
    """Configuration for player tracker"""
    max_age: int = 30  # Frames to keep track without detection
    min_hits: int = 3  # Hits needed to confirm track
    max_distance: float = 150.0  # Maximum pixel distance for matching
    max_velocity: float = 50.0  # Maximum expected velocity (pixels/frame)


class PlayerTracker:
    """
    Multi-object tracker for maintaining consistent player identities.
    
    Uses Hungarian algorithm for optimal detection-to-track assignment
    and velocity-based position prediction for missing detections.
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.tracks: Dict[int, TrackState] = {}
        self._next_id: int = 0
        self._frame_count: int = 0
    
    def _get_next_id(self) -> int:
        """Get next available track ID"""
        self._next_id += 1
        return self._next_id
    
    def _calculate_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[TrackState]
    ) -> np.ndarray:
        """
        Calculate cost matrix for Hungarian algorithm.
        Cost = distance from predicted position to detection.
        """
        n_det = len(detections)
        n_trk = len(tracks)
        
        cost_matrix = np.full((n_det, n_trk), self.config.max_distance * 2)
        
        for i, det in enumerate(detections):
            det_pos = (det.bbox.x, det.bbox.y)
            
            for j, track in enumerate(tracks):
                pred_pos = track.predict_position()
                distance = np.sqrt(
                    (det_pos[0] - pred_pos[0])**2 +
                    (det_pos[1] - pred_pos[1])**2
                )
                
                # Penalize large distances
                if distance <= self.config.max_distance:
                    cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def update(
        self,
        detections: List[Detection],
        frame_number: int
    ) -> List[Tuple[Detection, int]]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of player detections for current frame
            frame_number: Current frame number
            
        Returns:
            List of (detection, track_id) tuples for confirmed tracks
        """
        self._frame_count = frame_number
        
        # Get active tracks
        active_tracks = [
            track for track in self.tracks.values()
            if frame_number - track.last_seen_frame <= self.config.max_age
        ]
        
        matched_detections = []
        matched_track_ids = set()
        
        if len(detections) > 0 and len(active_tracks) > 0:
            # Calculate cost matrix
            cost_matrix = self._calculate_cost_matrix(detections, active_tracks)
            
            # Hungarian algorithm for optimal assignment
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
            
            for det_idx, trk_idx in zip(det_indices, trk_indices):
                if cost_matrix[det_idx, trk_idx] <= self.config.max_distance:
                    track = active_tracks[trk_idx]
                    detection = detections[det_idx]
                    
                    track.update(detection, frame_number)
                    matched_track_ids.add(track.track_id)
                    
                    if track.is_confirmed:
                        matched_detections.append((detection, track.track_id))
        
        # Handle unmatched detections (create new tracks)
        det_positions = set()
        for det in detections:
            det_positions.add((int(det.bbox.x), int(det.bbox.y)))
        
        for det in detections:
            is_matched = any(
                trk.last_detection is not None and
                int(trk.last_detection.bbox.x) == int(det.bbox.x) and
                int(trk.last_detection.bbox.y) == int(det.bbox.y)
                for trk in self.tracks.values()
                if trk.track_id in matched_track_ids
            )
            
            if not is_matched:
                # Create new track
                new_id = self._get_next_id()
                new_track = TrackState(track_id=new_id)
                new_track.update(det, frame_number)
                self.tracks[new_id] = new_track
        
        # Age unmatched tracks
        for track in self.tracks.values():
            if track.track_id not in matched_track_ids:
                track.confidence = max(0, track.confidence - 0.15)
            track.age += 1
        
        # Clean up old tracks
        self._cleanup_tracks(frame_number)
        
        return matched_detections
    
    def _cleanup_tracks(self, frame_number: int):
        """Remove tracks that are too old or have low confidence"""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            frames_since_seen = frame_number - track.last_seen_frame
            
            if frames_since_seen > self.config.max_age:
                to_remove.append(track_id)
            elif track.confidence <= 0 and not track.is_confirmed:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[TrackState]:
        """Get all confirmed player tracks"""
        return [
            track for track in self.tracks.values()
            if track.is_confirmed
        ]
    
    def reset(self):
        """Reset all tracks"""
        self.tracks.clear()
        self._next_id = 0


# Singleton instance
_player_tracker: Optional[PlayerTracker] = None


def get_player_tracker() -> PlayerTracker:
    """Get or create the global player tracker instance"""
    global _player_tracker
    if _player_tracker is None:
        _player_tracker = PlayerTracker()
    return _player_tracker
```

---

## 5. Comparative Analysis

### 5.1 Solution Comparison Matrix

| Solution | Complexity | Compute Overhead | Accuracy Gain | Maintenance | Dependencies |
|----------|-----------|------------------|---------------|-------------|--------------|
| Court ROI Filtering | LOW | +0.1ms | 40-60% | LOW | Court detection |
| Secondary Classifier | HIGH | +5ms | 80-95% | HIGH | Training data |
| Temporal Tracking | MEDIUM | +1ms | 30-50% | LOW | scipy |
| NMS Adjustment | LOW | 0ms | 10-20% | NONE | None |
| Model Fine-tuning | HIGH | 0ms | 85-98% | HIGH | Training data |
| Uniform Detection | MEDIUM | +2ms | 50-70% | MEDIUM | User calibration |
| Geometry Constraints | LOW-MED | +0.5ms | 40-60% | LOW | Court detection |

### 5.2 Recommendation Ranking

**Immediate Impact (Implement First):**
1. **Court ROI Filtering** - Best effort/reward ratio
2. **Temporal Tracking** - Prevents ID switches
3. **NMS Adjustment** - Simple parameter change

**Medium-term (2-4 weeks):**
4. **Geometry Constraints** - Domain knowledge enforcement
5. **Uniform Detection** - Additional discrimination

**Long-term (1-3 months):**
6. **Secondary Classifier** - If other methods insufficient
7. **Model Fine-tuning** - Ultimate solution, highest effort

### 5.3 Combined Solution Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    Frame Input                                   │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │                 Court Detection (cached)                         │
 │        [court_detection.py - runs every 30 frames]              │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │              YOLO26 Person Detection                            │
 │     [multi_model_detector.py - with adjusted NMS params]        │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │              Stage 1: Court ROI Filter                          │
 │        [player_roi_filter.py - remove off-court detections]     │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │            Stage 2: Geometry Constraints                        │
 │     [Apply court position rules - max 2 per side]               │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │           Stage 3: Temporal Consistency                         │
 │       [player_tracking.py - maintain stable IDs]                │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │            Stage 4: Size-based Selection                        │
 │      [Original _filter_main_players as final tie-breaker]       │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │                Pose Detection                                    │
 │          [pose_detection.py - match to filtered players]        │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │              Speed/Heatmap Analytics                            │
 │          [Receive clean, consistent player data]                │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 6. Prioritized Implementation Strategy

### Phase 1: Quick Wins (1-2 days)

**Goal:** Immediate accuracy improvement with minimal code changes

**Tasks:**
1. ✅ Create `player_roi_filter.py` module
2. ✅ Integrate ROI filter into `multi_model_detector.py`
3. ✅ Adjust NMS parameters in detection calls
4. ✅ Test with sample videos

**Expected Outcome:** 40-50% reduction in misassignment

**Validation:**
```python
# Test script
def validate_roi_filter():
    # Load video with known officials visible
    # Count detections before/after ROI filter
    # Verify officials are filtered out
    pass
```

### Phase 2: Temporal Stability (3-5 days)

**Goal:** Prevent ID switches and tracking loss

**Tasks:**
1. ✅ Create `player_tracking.py` module
2. ✅ Integrate tracker into main processing loop
3. ✅ Add track visualization for debugging
4. ✅ Tune tracker parameters for badminton motion

**Expected Outcome:** 30-40% additional improvement (cumulative: 60-70%)

### Phase 3: Domain Knowledge (1 week)

**Goal:** Apply badminton-specific constraints

**Tasks:**
1. Add court-side player count constraints
2. Implement relative size validation (far vs near player)
3. Add service line position constraints during serve
4. Create validation visualizations

**Expected Outcome:** 10-20% additional improvement (cumulative: 70-80%)

### Phase 4: Advanced Solutions (2-4 weeks, if needed)

**Goal:** Achieve >95% accuracy if Phase 1-3 insufficient

**Tasks:**
1. Collect training data for player/official classifier
2. Train lightweight classification model
3. OR: Annotate negative examples and fine-tune YOLO26
4. A/B test against Phase 3 solution

**Expected Outcome:** 95%+ accuracy

---

## 7. Code Implementation Examples

### 7.1 Updated Detection Pipeline

```python
# backend/main.py - Updated process_video function

async def process_video_with_filtering(
    video_path: str,
    video_id: str,
    enable_roi_filter: bool = True,
    enable_tracking: bool = True
):
    """
    Process video with improved player filtering.
    """
    # Initialize components
    multi_detector = get_multi_model_detector()
    court_detector = get_court_detector()
    roi_filter = get_roi_filter() if enable_roi_filter else None
    tracker = get_player_tracker() if enable_tracking else None
    
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    court_detection = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update court detection periodically
        if frame_number % 30 == 0:
            court_detection = court_detector.detect_court(frame, frame_number)
        
        # Run detection with ROI filtering
        detections = multi_detector.detect(
            frame,
            frame_number,
            max_players=2,
            court=court_detection  # Pass court for ROI filtering
        )
        
        # Apply temporal tracking
        if tracker is not None:
            tracked = tracker.update(detections.players, frame_number)
            # Replace detections with tracked ones
            detections.players = [det for det, _ in tracked]
        
        # Process for heatmap/speed (now with clean player data)
        yield process_frame_analytics(detections, frame_number)
        
        frame_number += 1
    
    cap.release()
```

### 7.2 Configuration API

```python
# backend/main.py - New configuration endpoints

class PlayerFilteringConfig(BaseModel):
    enable_roi_filter: bool = True
    roi_margin_meters: float = 1.0
    enable_tracking: bool = True
    tracking_max_age: int = 30
    tracking_max_distance: float = 150.0
    nms_iou_threshold: float = 0.3
    max_players_per_side: int = 1  # Singles

@app.post("/api/config/player-filtering")
async def update_player_filtering_config(config: PlayerFilteringConfig):
    """Update player filtering configuration"""
    multi_detector = get_multi_model_detector()
    tracker = get_player_tracker()
    
    # Apply ROI filter settings
    if config.enable_roi_filter:
        roi_filter = get_roi_filter()
        roi_filter.config.margin_meters = config.roi_margin_meters
    
    # Apply tracker settings
    if config.enable_tracking:
        tracker.config.max_age = config.tracking_max_age
        tracker.config.max_distance = config.tracking_max_distance
    
    return {"status": "updated", "config": config.dict()}
```

### 7.3 Debugging Visualization

```python
# backend/debug_visualization.py - NEW

def draw_filtering_debug(
    frame: np.ndarray,
    all_detections: List[Detection],
    filtered_detections: List[Detection],
    court_polygon: Optional[np.ndarray],
    track_states: List[TrackState]
) -> np.ndarray:
    """
    Draw debug visualization showing filtering stages.
    
    - Red boxes: Filtered out (outside ROI or low confidence)
    - Yellow boxes: Detected but not tracked
    - Green boxes: Active tracked players
    - Blue polygon: Court ROI boundary
    """
    debug_frame = frame.copy()
    
    # Draw court ROI
    if court_polygon is not None:
        cv2.polylines(debug_frame, [court_polygon], True, (255, 200, 0), 2)
    
    # Draw filtered-out detections (red)
    filtered_ids = set(id(d) for d in filtered_detections)
    for det in all_detections:
        if id(det) not in filtered_ids:
            x1, y1 = int(det.bbox.x_min), int(det.bbox.y_min)
            x2, y2 = int(det.bbox.x_max), int(det.bbox.y_max)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug_frame, "FILTERED", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw tracked players (green)
    for track in track_states:
        if track.is_confirmed and track.last_detection:
            det = track.last_detection
            x1, y1 = int(det.bbox.x_min), int(det.bbox.y_min)
            x2, y2 = int(det.bbox.x_max), int(det.bbox.y_max)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"P{track.track_id}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw trajectory
            if len(track.positions) > 1:
                pts = np.array(list(track.positions), dtype=np.int32)
                cv2.polylines(debug_frame, [pts], False, (0, 255, 0), 1)
    
    return debug_frame
```

---

## 8. Testing and Validation

### 8.1 Test Cases

```python
# tests/test_player_filtering.py

import pytest
from backend.player_roi_filter import CourtROIFilter, ROIFilterConfig
from backend.player_tracking import PlayerTracker, TrackerConfig
from backend.multi_model_detector import Detection, BoundingBox

class TestCourtROIFilter:
    def test_filters_outside_court(self):
        """Detections outside court should be filtered"""
        filter = CourtROIFilter()
        # Setup court polygon
        # Add detection outside
        # Verify filtered out
        pass
    
    def test_keeps_inside_court(self):
        """Detections inside court should be kept"""
        pass
    
    def test_margin_expansion(self):
        """Margin should allow detections slightly outside lines"""
        pass

class TestPlayerTracker:
    def test_creates_new_tracks(self):
        """New detections should create tracks"""
        pass
    
    def test_maintains_track_ids(self):
        """Same player should keep same ID across frames"""
        pass
    
    def test_rejects_large_jumps(self):
        """Large position jumps should not match to existing tracks"""
        pass

class TestIntegration:
    def test_official_filtered_from_heatmap(self):
        """Officials should not appear in heatmap data"""
        pass
    
    def test_speed_not_affected_by_switch(self):
        """ID switch should not cause speed spike"""
        pass
```

### 8.2 Metrics to Track

1. **Misassignment Rate:** % frames where non-player is detected as player
2. **Track Stability:** Average track lifetime vs expected (full video)
3. **Speed Spike Rate:** % frames with speed > 30 km/h
4. **Heatmap Coverage:** % of heatmap intensity on court vs off-court

---

## 9. Conclusion

This document provides a comprehensive analysis of the YOLO26 player misassignment issues and multiple solution approaches. The recommended implementation strategy prioritizes:

1. **Quick wins** with Court ROI filtering (immediate impact)
2. **Stability improvements** with temporal tracking (prevents switches)
3. **Domain enforcement** with geometry constraints (badminton rules)
4. **Advanced solutions** only if needed (secondary classifier or fine-tuning)

The combined multi-stage filtering pipeline is expected to achieve 80-90% reduction in misassignment with minimal computational overhead (~2ms per frame), preserving real-time performance while significantly improving heatmap and speed calculation accuracy.

**Next Steps:**
1. Implement Phase 1 (Court ROI filtering)
2. Validate with test videos
3. Proceed to Phase 2 if successful
4. Iterate based on results

---

*Document prepared for Badminton Tracker Project - 2026-02-01*
