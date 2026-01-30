# Skeleton Smoothing and Temporal Filtering Guide

This document explains the motion smoothing implementation for the badminton tracker's YOLO-based pose estimation and bounding box tracking system.

## Overview

The smoothing system eliminates jerky, staggered movement between frames by applying temporal filtering to detected keypoints and bounding boxes. It uses a combination of algorithms optimized for real-time performance with minimal latency.

## Implemented Algorithms

### 1. One Euro Filter (Recommended for Keypoints)

**Reference:** Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012)

The One Euro Filter is an adaptive low-pass filter that automatically adjusts its cutoff frequency based on the rate of change of the signal:

- **At low speeds (stationary pose):** Uses low cutoff frequency → Maximum smoothing to reduce jitter
- **At high speeds (fast movements):** Uses high cutoff frequency → Minimum lag to track rapid motion

```python
# Adaptive cutoff calculation
cutoff = min_cutoff + beta * abs(derivative)
```

**Parameters:**
- `min_cutoff` (Hz): Minimum cutoff frequency. Lower = more smoothing at rest. Default: 1.0
- `beta`: Speed coefficient. Higher = less lag during fast movements. Default: 0.007
- `d_cutoff` (Hz): Cutoff for derivative estimation. Default: 1.0

**Computational Complexity:** O(1) per keypoint
**Typical Latency:** ~0.5-1.5 frames depending on movement speed

### 2. Kalman Filter (For Bounding Boxes)

Based on Ultralytics YOLO's BOT-SORT/ByteTrack implementation. Uses a constant velocity motion model with state:

```
State: [x, y, w, h, vx, vy, vw, vh]
Measurement: [x, y, w, h]
```

**Features:**
- Predictive tracking for occlusion handling
- Velocity estimation for motion continuity
- Adaptive noise based on object size

**Computational Complexity:** O(n²) where n=8 (fixed state size) = O(1)

### 3. Exponential Moving Average (EMA)

Simple and efficient smoothing for real-time applications:

```python
smooth_value = alpha * new_value + (1 - alpha) * prev_value
```

As used in YOLO's BOTrack.update_features() for appearance feature smoothing.

**Parameters:**
- `alpha`: Smoothing factor (0-1). Higher = more responsive, lower = smoother.

**Computational Complexity:** O(1)

## Usage

### Basic Usage

```python
from pose_detection import get_pose_detector, get_smoothed_pose_detector

# Option 1: Enable smoothing via parameter
detector = get_pose_detector(
    enable_smoothing=True,
    smoothing_preset="high_speed",  # Optimized for badminton
    fps=30.0
)

# Option 2: Use convenience function
detector = get_smoothed_pose_detector(fps=30.0)

# Detect with automatic smoothing
poses = detector.detect(frame, frame_number=i)
```

### Smoothing Presets

| Preset | Use Case | Min Cutoff | Beta | Description |
|--------|----------|------------|------|-------------|
| `auto` | General use | Based on FPS | Auto | Automatically selects based on video FPS |
| `high_speed` | Badminton, tennis | 1.5 | 0.01 | Optimized for fast-moving sports |
| `stability` | Slow movements | 0.5 | 0.003 | Maximum smoothness, higher latency |
| `real_time` | Low latency | EMA 0.5 | - | Uses EMA for minimal processing |
| `none` | Raw output | - | - | Disables all smoothing |

### Custom Configuration

```python
from skeleton_smoothing import SmoothingConfig, SmoothingMethod

config = SmoothingConfig(
    method=SmoothingMethod.ONE_EURO,
    one_euro_min_cutoff=1.2,    # Adjust for your use case
    one_euro_beta=0.008,         # Higher = less lag
    one_euro_d_cutoff=1.0,
    kalman_process_noise=0.05,
    kalman_measurement_noise=0.1,
    max_prediction_frames=5,     # Max frames to predict without detection
    confidence_threshold=0.3,
    velocity_damping=0.9
)

from skeleton_smoothing import MultiPlayerSmoother
smoother = MultiPlayerSmoother(config, fps=30.0)
```

### Handling Detection Dropouts

When detections are missed (occlusion, momentary detection failure), the system can predict poses:

```python
# Detection failed this frame - use prediction
poses = detector.predict_poses(frame_number=current_frame)

if poses is None:
    # No prediction available (too many frames without detection)
    # Handle gracefully
    pass
```

## Performance Benchmarks

Tested on typical system (M1 Mac, Python 3.9):

| Algorithm | Per-Frame Time | Overhead at 30 FPS |
|-----------|---------------|-------------------|
| One Euro Filter (17 keypoints) | ~0.03 ms | +0.03 ms |
| EMA (17 keypoints) | ~0.01 ms | +0.01 ms |
| Kalman Filter (bbox) | ~0.02 ms | +0.02 ms |
| **Total (One Euro + Kalman)** | ~0.05 ms | **<0.1 ms** |

**Memory overhead:** ~2KB per tracked player

All algorithms suitable for real-time processing (target: <1ms for 60fps).

## Algorithm Selection Guide

### Choose One Euro Filter When:
- ✅ Tracking human poses/keypoints
- ✅ Movement speed varies (stationary to fast)
- ✅ Need to minimize both jitter AND lag
- ✅ Real-time visualization is important

### Choose Kalman Filter When:
- ✅ Tracking bounding boxes/object positions
- ✅ Need velocity estimation
- ✅ Prediction capability for occlusions
- ✅ Consistent motion patterns expected

### Choose EMA When:
- ✅ Minimal computational overhead required
- ✅ Simple smoothing sufficient
- ✅ Very high frame rates (120+ fps)
- ✅ Feature vectors/non-positional data

## Body Part-Specific Tuning

Different body parts move at different speeds. The implementation applies per-keypoint beta multipliers:

| Body Part | Beta Multiplier | Reason |
|-----------|-----------------|--------|
| Wrists | 1.5x | Fastest moving during shots |
| Ankles | 1.3x | Quick footwork movements |
| Elbows | 1.2x | Arm swing dynamics |
| Knees | 1.1x | Lunge/jump movements |
| Torso | 1.0x | Core stability |
| Head | 1.0x | Generally stable |

## Trade-offs

### Smoothness vs. Latency

```
More Smoothing (lower min_cutoff, lower beta):
+ Eliminates jitter
+ Stable visualization  
- Higher latency
- Delayed reaction to fast movements

Less Smoothing (higher min_cutoff, higher beta):
+ Low latency
+ Immediate response
- More jitter visible
- Less stable visualization
```

### Recommended Settings by Use Case

| Use Case | Min Cutoff | Beta | Notes |
|----------|------------|------|-------|
| Real-time display | 1.0-1.5 | 0.007-0.01 | Balance smoothness/responsiveness |
| Video analysis (post) | 0.5-0.8 | 0.003-0.005 | Can tolerate more latency |
| Shot detection | 1.5-2.0 | 0.01-0.015 | Minimize lag for timing accuracy |
| Movement metrics | 0.8-1.2 | 0.005-0.008 | Smooth for velocity calculation |

## Integration with Existing Pipeline

The smoothing is integrated directly into `PoseDetector.detect()`:

```python
# In pose_detection.py
def detect(self, frame, frame_number=0, timestamp=None):
    # 1. Run YOLO pose detection
    results = self.model(frame)
    frame_poses = self._parse_results(results)
    
    # 2. Apply smoothing if enabled
    if self.enable_smoothing:
        frame_poses = self._apply_smoothing(frame_poses, timestamp)
    
    return frame_poses
```

No changes required to downstream code - smoothed poses have the same interface.

## Resetting State

When starting a new video or scene:

```python
detector.reset_smoothing()
```

This clears all filter states and tracking history.

## Environment Variables

Configure smoothing via environment:

```bash
POSE_ENABLE_SMOOTHING=true       # Enable/disable smoothing
POSE_SMOOTHING_PRESET=high_speed # Preset name
```

## Troubleshooting

### Skeleton still appears jerky
- Increase smoothing: lower `min_cutoff` (try 0.5-0.8)
- Check if smoothing is enabled: `detector.get_status()`

### Skeleton lags behind actual movement
- Decrease smoothing: higher `min_cutoff` (try 1.5-2.0)
- Increase `beta` (try 0.01-0.02)

### Keypoints jump when detection resumes after dropout
- Increase `max_prediction_frames` to bridge longer gaps
- Ensure frame numbers are passed correctly

### Poor performance
- Run benchmark: `python skeleton_smoothing.py`
- Consider using `real_time` preset (EMA-based)

## References

1. Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. CHI '12.

2. Ultralytics YOLO Trackers: BOT-SORT and ByteTrack implementations
   - https://docs.ultralytics.com/reference/trackers/bot_sort/
   - https://docs.ultralytics.com/reference/trackers/byte_tracker/

3. Kalman Filter theory and implementation
   - Welch & Bishop, "An Introduction to the Kalman Filter"
