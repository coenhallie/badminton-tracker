# 🏸 Badminton Tracker

An AI-powered badminton match analysis application that uses YOLOv8 pose estimation to track player movements, calculate speeds, and provide detailed performance metrics.

![Badminton Tracker](https://img.shields.io/badge/Vue.js-3.5-4FC08D?logo=vue.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript)

## ✨ Features

### Video Analysis
- **Upload and Process** - Support for MP4, MPEG, MOV, AVI, and WebM video formats
- **Real-time Progress** - WebSocket-based progress updates during analysis
- **Frame-by-Frame Analysis** - Detailed pose estimation on each video frame

### Pose Estimation
- **Skeleton Overlay** - Visual skeleton tracking using YOLOv11 pose model
- **17 Keypoint Detection** - Full body keypoint tracking (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Multi-Player Tracking** - Simultaneous tracking of multiple players on court

### 🆕 Court Detection (Roboflow)
- **Court Keypoint Detection** - Automatic detection of badminton court lines using [Roboflow's Badminton Court Keypoint Dataset](https://universe.roboflow.com/learning-9i34b/badminton-court-keypoint-dataset)
- **Perspective Calibration** - Homography-based coordinate transformation for accurate measurements
- **Court Boundary Visualization** - Visual overlay of detected court boundaries
- **Accurate Distance Calculations** - Real-world distance measurements using detected court dimensions

### 🆕 Custom Badminton Model (Roboflow)
- **Custom Model Support** - Use your own trained Roboflow model for player/shuttlecock detection
- **Cloud or Self-Hosted** - Use Roboflow cloud API or self-host locally with Docker
- **No File Size Limits** - Self-hosted mode bypasses the 20MB cloud limit
- **Player Detection** - Detect and track players using custom-trained models
- **Shuttlecock Detection** - Detect shuttlecock positions for more accurate shot analysis
- **Configurable Class Mappings** - Map your model's class names to players/shuttlecocks

### Speed Metrics
- **Player Movement Speed** - Real-time and average speed calculation in km/h
- **Maximum Speed Detection** - Track peak movement speeds during rallies
- **Total Distance Covered** - Cumulative distance traveled by each player
- **Court-Calibrated Measurements** - Accurate real-world distances using court keypoint detection

### 🆕 Enhanced Shuttle Analytics
- **Direct Shuttle Tracking** - Track shuttlecock position across frames using detection
- **Real-Time Speed Calculation** - Calculate shuttle speed using court homography for accurate measurements
- **Shot Type Classification** - Automatic classification of shots (smash, clear, drop, drive, net shot, lob)
- **Trajectory Analysis** - Track complete shuttle trajectories with positions in both pixel and court coordinates
- **Shot Statistics** - Fastest shot, average speed, total shots, and shot type distribution

### 🆕 Player Zone Analytics
- **Court Zone Coverage** - Track time spent in front/mid/back and left/center/right zones
- **Position Heatmaps** - Generate heatmap data for player movement visualization
- **Distance to Net** - Track average distance from the net for each player
- **Real-World Positioning** - Convert pixel positions to court coordinates using homography

### 🆕 Detection Smoothing & Lag Reduction
- **Kalman Filter Tracking** - Smooth player and shuttle detection to reduce jitter and lag
- **Temporal Interpolation** - Fill in gaps between processed frames for smoother playback
- **Binary Search Frame Lookup** - O(log n) frame lookup for instant skeleton data access
- **Motion Prediction** - Predict positions for fast-moving objects (shuttlecock)
- **Video Preprocessing** - Optional deblur and enhancement for rapid movement frames

### Legacy Shuttle Analysis
- **Shot Detection** - Automatic detection of shuttle shots based on wrist velocity
- **Shot Speed Estimation** - Estimated shuttle speed for detected shots
- **Shot Distribution** - Visual chart of shot speeds throughout the match

### Results Dashboard
- **Comprehensive Metrics** - Duration, FPS, frames analyzed, players detected
- **Per-Player Statistics** - Individual stats for each tracked player
- **Court Detection Status** - Display of court keypoint detection confidence
- **Video Playback** - Processed video with skeleton and court overlay
- **Interactive Controls** - Play, pause, seek, playback speed, fullscreen

## 🏗️ Architecture

```
badminton-tracker/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # API endpoints and video processing
│   ├── court_detection.py     # Court keypoint detection module
│   ├── badminton_detection.py # Custom Roboflow model integration
│   ├── shuttle_analytics.py   # Enhanced shuttle tracking & speed analysis
│   ├── detection_smoothing.py # 🆕 Kalman filtering & temporal interpolation
│   ├── multi_model_detector.py # Multi-model YOLO detection manager
│   ├── requirements.txt       # Python dependencies
│   ├── uploads/               # Uploaded video storage
│   └── processed/             # Processed video storage
│
├── src/                       # Vue.js frontend
│   ├── components/
│   │   ├── VideoUpload.vue    # Drag-and-drop video upload
│   │   ├── VideoPlayer.vue    # Video player with skeleton overlay
│   │   ├── AnalysisProgress.vue # Real-time progress display
│   │   └── ResultsDashboard.vue # Analysis results display
│   ├── services/
│   │   └── api.ts             # API service and WebSocket client
│   ├── types/
│   │   └── analysis.ts        # TypeScript type definitions
│   ├── App.vue                # Main application component
│   └── main.ts                # Application entry point
│
├── .env.example               # Environment variables template
├── package.json               # Frontend dependencies
├── vite.config.ts             # Vite configuration
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** v20.19.0+ or v22.12.0+
- **Python** 3.11+
- **pip** (Python package manager)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python3 main.py
   ```
   
   Or use uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Copy the environment file and configure if needed:
   ```bash
   cp .env.example .env
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

## 📡 API Endpoints

### Upload Video
```http
POST /api/upload
Content-Type: multipart/form-data

file: <video_file>
```

### Start Analysis
```http
POST /api/analyze/{video_id}
Content-Type: application/json

{
  "fps_sample_rate": 1,
  "confidence_threshold": 0.5,
  "track_shuttle": true,
  "calculate_speeds": true
}
```

### Get Results
```http
GET /api/results/{video_id}
```

### Get Processed Video
```http
GET /api/video/{video_id}
```

### Get Skeleton Data
```http
GET /api/skeleton/{video_id}
```

### 🆕 Get Shuttle Analytics
```http
GET /api/shuttle-analytics/{video_id}
```

Returns:
- Total shots detected
- Shot type counts (smash, clear, drop, drive, net shot)
- Speed statistics (fastest, average, all shot speeds)
- Detailed trajectory data with positions in pixel and court coordinates

### 🆕 Get Player Zone Analytics
```http
GET /api/player-zone-analytics/{video_id}
```

Returns per player:
- Zone coverage percentages (front/mid/back, left/center/right)
- Average distance to net in meters
- Position heatmap grid (normalized 0-1)
- Total position count

### 🆕 Get Court Keypoints Info
```http
GET /api/court-keypoints/info
```

Returns:
- Standard badminton court dimensions
- Court keypoint names and descriptions
- Capabilities (homography transform, distance calculation, etc.)

### WebSocket Progress Updates
```
WS /ws/{video_id}
```

## 🔧 Configuration

### Analysis Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fps_sample_rate` | int | 1 | Process every Nth frame |
| `confidence_threshold` | float | 0.5 | Keypoint confidence threshold |
| `track_shuttle` | bool | true | Enable shuttle detection |
| `calculate_speeds` | bool | true | Enable speed calculations |
| `detect_court` | bool | true | Enable court keypoint detection |
| `court_detection_interval` | int | 30 | Frames between court detection updates |
| `use_badminton_detector` | bool | true | Use custom Roboflow model for player/shuttlecock detection |
| `enable_smoothing` | bool | true | **NEW** Enable Kalman filter smoothing to reduce detection lag |
| `enable_interpolation` | bool | true | **NEW** Enable temporal interpolation between keyframes |
| `preprocess_video` | bool | false | **NEW** Apply deblur/enhancement for rapid movements |
| `player_smoothing_strength` | float | 0.5 | **NEW** Smoothing strength for players (0-1, higher = smoother) |
| `shuttle_smoothing_strength` | float | 0.3 | **NEW** Smoothing for shuttle (lower = more responsive) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |
| `ROBOFLOW_API_KEY` | (none) | Roboflow API key for court & badminton detection |
| `ROBOFLOW_BADMINTON_MODEL` | `badminton-hehp8-347a9-gmjzk/1` | Custom Roboflow model ID |
| `ROBOFLOW_CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence threshold (0-1) |
| `ROBOFLOW_LOCAL_SERVER` | `false` | Use self-hosted Docker server (bypasses 20MB limit) |

### Setting up Court Detection

To enable accurate court-calibrated measurements:

1. Create a free account at [Roboflow](https://app.roboflow.com/)
2. Get your API key from [Settings → API](https://app.roboflow.com/settings/api)
3. Set the environment variable:
   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   ```
   Or add it to your `.env` file.

The court detection uses the [Badminton Court Keypoint Dataset](https://universe.roboflow.com/learning-9i34b/badminton-court-keypoint-dataset) model to detect court lines and corners, enabling precise real-world distance calculations.

### Setting up Custom Badminton Model

To use your own trained Roboflow model for player/shuttlecock detection:

1. Train a model on [Roboflow](https://app.roboflow.com/) with classes for players and/or shuttlecocks
2. Deploy your model to Roboflow's Serverless API
3. Get your model ID from the deployment page (format: `project-id/version`)
4. Set the environment variables:
   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   export ROBOFLOW_BADMINTON_MODEL=your-model-id/1
   ```

**Testing your model:**

```bash
# Test detection on a single image
curl -X POST "http://localhost:8000/api/badminton-detection/test" \
  -F "file=@test_image.jpg"
```

**Updating class mappings:**

If your model uses different class names, update them via the API:

```bash
curl -X POST "http://localhost:8000/api/badminton-detection/update-classes" \
  -H "Content-Type: application/json" \
  -d '{"player_classes": ["player", "athlete"], "shuttlecock_classes": ["shuttle", "birdie"]}'
```

**Check detector status:**

```bash
curl "http://localhost:8000/api/badminton-detection/status"
```

### 🐳 Self-Hosted Inference (No Size Limits)

Roboflow's cloud API has a **20MB file size limit**. For larger videos or unlimited processing, run the inference server locally using Docker.

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) installed and running

**Option 1: Using Roboflow CLI (Recommended)**

```bash
# Install the inference CLI
pip install inference-cli

# Start the local server
inference server start
```

**Option 2: Using Docker directly (CPU)**

```bash
docker run -d \
    --name roboflow-inference \
    -p 9001:9001 \
    -v ~/.inference/cache:/tmp:rw \
    roboflow/roboflow-inference-server-cpu:latest
```

**Option 3: Using Docker with GPU (NVIDIA)**

```bash
docker run -d \
    --name roboflow-inference \
    --gpus all \
    -p 9001:9001 \
    -v ~/.inference/cache:/tmp:rw \
    roboflow/roboflow-inference-server-gpu:latest
```

**Enable local server in your app:**

1. Add to your `.env` file:
   ```
   ROBOFLOW_LOCAL_SERVER=true
   ```

2. Restart the backend server

3. You should see:
   ```
   Badminton detector initialized [LOCAL (self-hosted)]: badminton-hehp8-347a9-gmjzk/1
     API URL: http://localhost:9001
     Note: Make sure Docker inference server is running on port 9001
   ```

**Benefits of self-hosting:**
- ✅ No file size limits
- ✅ Faster inference (no network latency)
- ✅ Model weights cached locally after first download
- ✅ Works offline after model is cached
- ✅ Same API interface - just change one environment variable

## 🧠 How It Works

### Pose Estimation Pipeline

1. **Video Upload** - User uploads a badminton match video
2. **Frame Extraction** - Video is split into individual frames
3. **YOLO Processing** - YOLOv11 pose model detects 17 body keypoints per person
4. **Tracking** - Multi-object tracking maintains player identities across frames
5. **Skeleton Drawing** - Keypoints are connected to form skeleton visualization
6. **Speed Calculation** - Position changes converted to real-world speeds using court dimensions

### Speed Calculation Method

Player speed is calculated by:
1. Tracking hip midpoint position across frames
2. Converting pixel displacement to meters using standard court dimensions (6.1m width)
3. Applying time delta based on video FPS
4. Converting m/s to km/h

### 🆕 Enhanced Shuttle Speed Calculation

When court keypoints are detected, shuttle speed is calculated using real-world measurements:
1. **Shuttle Detection** - Track shuttlecock position using the badminton detection model
2. **Court Homography** - Use detected court keypoints to create a perspective transformation matrix
3. **Coordinate Conversion** - Convert pixel positions to real-world court coordinates (meters)
4. **Speed Calculation** - Calculate distance traveled between frames in meters
5. **Time Normalization** - Apply frame rate to convert to meters/second, then km/h

This provides accurate shuttle speeds based on real court dimensions (6.1m × 13.4m).

### Shot Type Classification

Shots are classified based on speed and trajectory:
| Shot Type | Speed Threshold | Characteristics |
|-----------|-----------------|-----------------|
| Smash | ≥150 km/h | High speed, downward trajectory |
| Drive | ≥100 km/h | Fast, flat horizontal |
| Clear | ≥80 km/h | High arc, back of court |
| Drop | ≥60 km/h | Slow, steep downward |
| Net Shot | <60 km/h | Near net, low speed |
| Lob | <60 km/h | Defensive high return |

### Legacy Shuttle Speed Estimation

When direct shuttle tracking is not available, speeds are estimated by:
1. Detecting rapid wrist movements (velocity peaks)
2. Applying a multiplier factor (typically 2x wrist speed for smashes)
3. Filtering results above 50 km/h threshold

### 🆕 Detection Smoothing Pipeline

The smoothing system addresses the common issue of detection lag during rapid movements:

1. **Kalman Filtering** - Each player and shuttlecock has a dedicated Kalman filter that:
   - Tracks position and velocity state
   - Smooths noisy detections
   - Predicts position when detection is lost (up to 5 frames)

2. **Temporal Interpolation** - When `fps_sample_rate > 1`:
   - Generates intermediate frames between processed keyframes
   - Uses linear interpolation for positions and keypoints
   - Provides smooth 60fps-like playback even with lower processing rate

3. **Binary Search Optimization** - Frontend frame lookup uses:
   - O(log n) binary search instead of O(n) linear search
   - Pre-built frame index for O(1) direct lookup
   - Significant performance improvement for long videos

4. **Video Preprocessing** (optional):
   - Unsharp masking to reduce motion blur
   - CLAHE enhancement for small object visibility (shuttlecock)
   - Applied before detection for improved accuracy

**Recommended settings for fast-paced matches:**
```json
{
  "fps_sample_rate": 1,
  "enable_smoothing": true,
  "enable_interpolation": true,
  "player_smoothing_strength": 0.5,
  "shuttle_smoothing_strength": 0.3,
  "preprocess_video": false
}
```

## 🎯 Tips for Best Results

1. **Video Quality** - Use 720p or higher resolution for accurate keypoint detection
2. **Camera Angle** - Side or elevated angle captures full body movements better
3. **Lighting** - Even lighting improves pose detection accuracy
4. **Video Length** - Keep videos under 5 minutes for faster processing
5. **Player Count** - Works best with 2-4 players visible on court

## 🛠️ Development

### Type Checking
```bash
npm run type-check
```

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Code Formatting
```bash
npm run format
```

## 📄 License

This project is for educational and personal use. The YOLOv8/v11 models are provided by [Ultralytics](https://ultralytics.com/) under the AGPL-3.0 license.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8/v11 pose estimation models
- [Vue.js](https://vuejs.org/) for the reactive frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance Python API
- [OpenCV](https://opencv.org/) for video processing capabilities
