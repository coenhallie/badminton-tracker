# Multi-Signal Rally Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-signal gradient-only rally classification with a multi-signal fusion approach that combines shuttle movement, player pose state, and player movement intensity into a per-frame confidence score.

**Architecture:** Each frame gets a rally confidence score (0.0-1.0) from three weighted signals: shuttle gradient (0.50), player pose (0.30), player movement (0.20). A windowed average of scores is thresholded to classify frames as active/idle. All existing post-processing (direction-change validation, boundary trimming, carried-shuttle filter) is preserved.

**Tech Stack:** Python (backend/rally_detection.py), Modal processor (backend/modal_convex_processor.py)

---

### Task 1: Add pose and speed signal computation to rally_detection.py

**Files:**
- Modify: `backend/rally_detection.py` — add `_compute_pose_signal()` and `_compute_movement_signal()` functions

**Step 1: Add `_compute_pose_signal` function**

Add after `_compute_gradients()` (after line 159):

```python
# Hitting poses that indicate active rally play
_HITTING_POSES = {
    "smash", "overhead", "serving", "serve",
    "forehand", "backhand", "lunge", "offense",
}

# Idle poses that indicate between-rally state
_IDLE_POSES = {"standing", "ready", "recovery", "unknown"}


def _compute_pose_signal(
    pose_data: Dict[int, List[Dict]],
    total_frames: int,
) -> List[float]:
    """
    Compute per-frame pose signal (0.0-1.0) from player pose classifications.

    Returns 1.0 when any player is in a hitting pose, 0.0 when all players
    are idle, and interpolated values for mixed/uncertain states.

    pose_data: frame_num -> list of {"pose_type": str, "confidence": float}
    """
    signal = [0.0] * total_frames

    for frame in range(total_frames):
        players = pose_data.get(frame)
        if not players:
            continue

        max_score = 0.0
        for player in players:
            pose = player.get("pose_type", "unknown")
            conf = player.get("confidence", 0.0)

            if pose in _HITTING_POSES and conf >= 0.4:
                # Scale by confidence — a high-confidence smash is stronger
                # signal than a low-confidence forehand
                max_score = max(max_score, min(conf, 1.0))
            # Idle poses contribute 0.0 (default)

        signal[frame] = max_score

    return signal


def _compute_movement_signal(
    speed_data: Dict[int, List[float]],
    total_frames: int,
) -> List[float]:
    """
    Compute per-frame movement signal (0.0-1.0) from player speeds.

    During rallies, players move 5-25 km/h (lunging, recovering).
    Between rallies, players walk at 2-4 km/h or stand still.

    Uses max speed across all players in each frame.
    Normalized: 0 at 2 km/h (walking), 1.0 at 12+ km/h (fast movement).

    speed_data: frame_num -> list of speeds (km/h) per player
    """
    signal = [0.0] * total_frames
    WALK_SPEED = 2.0   # km/h — below this is standing/slow walk
    RALLY_SPEED = 12.0  # km/h — above this is definitely rally movement

    for frame in range(total_frames):
        speeds = speed_data.get(frame)
        if not speeds:
            continue

        max_speed = max(speeds) if speeds else 0.0

        if max_speed <= WALK_SPEED:
            signal[frame] = 0.0
        elif max_speed >= RALLY_SPEED:
            signal[frame] = 1.0
        else:
            # Linear interpolation between walk and rally speed
            signal[frame] = (max_speed - WALK_SPEED) / (RALLY_SPEED - WALK_SPEED)

    return signal
```

**Step 2: Run quick sanity check**

```bash
python3 -c "
from backend.rally_detection import _compute_pose_signal, _compute_movement_signal

# Test pose signal
pose_data = {
    0: [{'pose_type': 'smash', 'confidence': 0.9}],
    1: [{'pose_type': 'standing', 'confidence': 0.8}],
    2: [{'pose_type': 'lunge', 'confidence': 0.5}],
}
sig = _compute_pose_signal(pose_data, 5)
assert sig[0] == 0.9, f'Expected 0.9, got {sig[0]}'
assert sig[1] == 0.0, f'Expected 0.0, got {sig[1]}'
assert sig[2] == 0.5, f'Expected 0.5, got {sig[2]}'
assert sig[3] == 0.0  # No data
print('Pose signal: OK')

# Test movement signal
speed_data = {
    0: [0.0, 1.0],   # Standing
    1: [5.0, 3.0],    # Moderate
    2: [15.0, 8.0],   # Fast
    3: [2.0, 2.0],    # Walking threshold
}
sig = _compute_movement_signal(speed_data, 5)
assert sig[0] == 0.0
assert 0.2 < sig[1] < 0.4  # 5 km/h normalized
assert sig[2] == 1.0        # 15 km/h capped
assert sig[3] == 0.0        # At walk threshold
print('Movement signal: OK')
"
```
Expected: Both tests pass.

**Step 3: Commit**

```bash
git add backend/rally_detection.py
git commit -m "add pose and movement signal computation for rally detection"
```

---

### Task 2: Replace binary classification with multi-signal fusion

**Files:**
- Modify: `backend/rally_detection.py` — update `detect_rallies()` to accept new data and use fused scoring, replace `_classify_frames` usage

**Step 1: Update `detect_rallies` signature and classification logic**

Update the function signature (line 26) to accept new parameters:

```python
def detect_rallies(
    shuttle_positions: Dict[int, Dict],
    fps: float,
    total_frames: int,
    player_positions: Optional[Dict[int, List[Dict]]] = None,
    pose_data: Optional[Dict[int, List[Dict]]] = None,
    speed_data: Optional[Dict[int, List[float]]] = None,
    min_rally_duration_s: float = 2.0,
    min_gap_duration_s: float = 3.0,
    zero_gradient_window: int = 0,
    zero_gradient_ratio: float = 0.80,
    frame_width: int = 0,
    frame_height: int = 0,
) -> List[Dict]:
```

Replace the frame classification block (Steps 1-2, lines 75-102). The new logic:

```python
    # Step 1: Compute all signals
    gradients = _compute_gradients(shuttle_positions, total_frames)

    max_dim = max(frame_width, frame_height) if frame_width > 0 else 0
    zero_threshold = max(2.0, 0.003 * max_dim) * (30.0 / fps) if max_dim > 0 else 2.0 * (30.0 / fps)

    # Step 2: Multi-signal fusion classification
    if pose_data or speed_data:
        # Fused approach: combine shuttle gradient + pose + movement signals
        frame_active = _classify_frames_fused(
            gradients, total_frames, zero_gradient_window,
            zero_threshold, fps,
            pose_data=pose_data,
            speed_data=speed_data,
        )
    else:
        # Fallback to gradient-only (backwards compatible)
        frame_active = _classify_frames(
            gradients, total_frames, zero_gradient_window,
            zero_gradient_ratio, fps,
            zero_threshold=zero_threshold,
        )
```

Keep the rest of the pipeline (carried-shuttle, extract, post-filter, validate, trim) unchanged.

**Step 2: Add `_classify_frames_fused` function**

Add after `_classify_frames` (after line 347):

```python
def _classify_frames_fused(
    gradients: List[float],
    total_frames: int,
    window_size: int,
    zero_threshold: float,
    fps: float,
    pose_data: Optional[Dict[int, List[Dict]]] = None,
    speed_data: Optional[Dict[int, List[float]]] = None,
    score_threshold: float = 0.35,
) -> List[bool]:
    """
    Classify frames using multi-signal fusion.

    Computes a per-frame rally confidence score (0-1) from three signals:
    - Shuttle gradient (weight 0.50): normalized movement intensity
    - Player pose (weight 0.30): hitting vs idle pose classification
    - Player movement (weight 0.20): player speed intensity

    The windowed average of scores is compared against score_threshold
    to classify each frame as active or idle.
    """
    W_SHUTTLE = 0.50
    W_POSE = 0.30
    W_MOVEMENT = 0.20

    # --- Shuttle signal: normalize gradient to 0-1 ---
    # Use a soft normalization: 0 at zero_threshold, 1.0 at 4x threshold
    high_gradient = zero_threshold * 4.0
    shuttle_signal = [0.0] * total_frames
    for i in range(total_frames):
        g = gradients[i]
        if g < zero_threshold:
            shuttle_signal[i] = 0.0
        elif g >= high_gradient:
            shuttle_signal[i] = 1.0
        else:
            shuttle_signal[i] = (g - zero_threshold) / (high_gradient - zero_threshold)

    # --- Pose signal ---
    pose_signal = _compute_pose_signal(pose_data or {}, total_frames)

    # --- Movement signal ---
    movement_signal = _compute_movement_signal(speed_data or {}, total_frames)

    # --- Fused per-frame score ---
    raw_scores = [0.0] * total_frames
    for i in range(total_frames):
        raw_scores[i] = (
            W_SHUTTLE * shuttle_signal[i]
            + W_POSE * pose_signal[i]
            + W_MOVEMENT * movement_signal[i]
        )

    # --- Windowed average ---
    half_w = window_size // 2
    frame_active = [False] * total_frames
    window_sum = sum(raw_scores[0:min(window_size, total_frames)])

    for center in range(total_frames):
        win_start = max(0, center - half_w)
        win_end = min(total_frames, center + half_w + 1)
        actual_window = win_end - win_start

        if center == 0 or win_start == 0:
            window_sum = sum(raw_scores[win_start:win_end])
        else:
            old_start = max(0, (center - 1) - half_w)
            if old_start < win_start and old_start < total_frames:
                window_sum -= raw_scores[old_start]
            new_end = min(total_frames, center + half_w + 1) - 1
            old_end = min(total_frames, (center - 1) + half_w + 1) - 1
            if new_end > old_end and new_end < total_frames:
                window_sum += raw_scores[new_end]

        avg_score = window_sum / max(actual_window, 1)
        frame_active[center] = avg_score >= score_threshold

    return frame_active
```

**Step 3: Run end-to-end test**

```bash
python3 -c "
from backend.rally_detection import detect_rallies
import math, random
random.seed(42)

fps = 30; total = 900; positions = {}; pose_data = {}; speed_data = {}

# Rally 1: frames 30-270
for f in range(30, 270):
    t = (f - 30) / fps
    x = 500 + 200 * math.sin(t * 3.0) + random.uniform(-2, 2)
    y = 300 + 100 * math.cos(t * 2.5) + random.uniform(-2, 2)
    positions[f] = {'x': x, 'y': y, 'visible': True}
    pose_data[f] = [{'pose_type': random.choice(['smash','forehand','lunge','overhead']), 'confidence': 0.7}]
    speed_data[f] = [random.uniform(5, 18)]

# Idle: frames 270-570
for f in range(270, 570):
    positions[f] = {'x': 650 + random.uniform(-1,1), 'y': 400 + random.uniform(-1,1), 'visible': random.random() > 0.5}
    pose_data[f] = [{'pose_type': 'standing', 'confidence': 0.8}]
    speed_data[f] = [random.uniform(0, 3)]

# Rally 2: frames 570-870
for f in range(570, 870):
    t = (f - 570) / fps
    x = 700 + 180 * math.sin(t * 2.8) + random.uniform(-2, 2)
    y = 350 + 120 * math.cos(t * 3.2) + random.uniform(-2, 2)
    positions[f] = {'x': x, 'y': y, 'visible': True}
    pose_data[f] = [{'pose_type': random.choice(['backhand','forehand','smash']), 'confidence': 0.65}]
    speed_data[f] = [random.uniform(6, 20)]

for f in range(total):
    if f not in positions:
        positions[f] = {'x': 0, 'y': 0, 'visible': False}

# With fusion
rallies = detect_rallies(positions, fps=fps, total_frames=total,
    pose_data=pose_data, speed_data=speed_data,
    frame_width=1920, frame_height=1080)
print(f'Fused: {len(rallies)} rallies')
for r in rallies:
    print(f'  Rally {r[\"id\"]}: {r[\"start_timestamp\"]:.1f}s - {r[\"end_timestamp\"]:.1f}s ({r[\"duration_seconds\"]:.1f}s)')

# Without fusion (backwards compat)
rallies2 = detect_rallies(positions, fps=fps, total_frames=total,
    frame_width=1920, frame_height=1080)
print(f'Gradient-only: {len(rallies2)} rallies')
assert len(rallies) == 2, f'Expected 2 rallies, got {len(rallies)}'
print('PASS')
"
```

Expected: 2 rallies detected with fused approach, boundaries close to ground truth.

**Step 4: Commit**

```bash
git add backend/rally_detection.py
git commit -m "add multi-signal fusion classification for rally detection"
```

---

### Task 3: Pass pose and speed data from processor to rally detection

**Files:**
- Modify: `backend/modal_convex_processor.py` — extract pose and speed data from skeleton_frames and pass to `detect_rallies()`

**Step 1: Build pose_data and speed_data dicts from skeleton_frames**

In `modal_convex_processor.py`, after the existing `player_wrist_data` extraction block (after line ~2311), add:

```python
        # Build pose and speed data for multi-signal rally detection.
        rally_pose_data = {}
        rally_speed_data = {}
        for sf in skeleton_frames:
            fn = sf["frame"]
            poses_in_frame = []
            speeds_in_frame = []
            for p in sf.get("players", []):
                pose = p.get("pose")
                if pose:
                    poses_in_frame.append({
                        "pose_type": pose.get("pose_type", "unknown"),
                        "confidence": pose.get("confidence", 0.0),
                    })
                speed = p.get("current_speed", 0.0)
                if speed is not None:
                    speeds_in_frame.append(float(speed))
            if poses_in_frame:
                rally_pose_data[fn] = poses_in_frame
            if speeds_in_frame:
                rally_speed_data[fn] = speeds_in_frame
```

**Step 2: Pass the new data to `detect_rallies()`**

Update the `detect_rallies()` call:

```python
                detected_rallies = detect_rallies(
                    rally_shuttle_positions,
                    fps=fps,
                    total_frames=total_frames,
                    player_positions=player_wrist_data if player_wrist_data else None,
                    pose_data=rally_pose_data if rally_pose_data else None,
                    speed_data=rally_speed_data if rally_speed_data else None,
                    frame_width=width,
                    frame_height=height,
                )
```

**Step 3: Commit**

```bash
git add backend/modal_convex_processor.py
git commit -m "pass pose and speed data to multi-signal rally detection"
```

---

### Task 4: Deploy and verify

**Files:**
- No code changes — deployment and verification

**Step 1: Deploy to Modal**

```bash
python3 -m modal deploy backend/modal_convex_processor.py
```

Expected: Successful deployment with rally_detection.py mounted.

**Step 2: Verify backwards compatibility**

The fused classification is only used when `pose_data` or `speed_data` is provided. In `rally_only` mode (where `skeleton_frames = []`), the data dicts are empty and the old gradient-only path runs. No regression.

**Step 3: Commit all changes**

```bash
git add -A
git commit -m "multi-signal rally detection: fuse shuttle gradient + player pose + movement intensity"
```
