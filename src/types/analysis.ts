/**
 * Type definitions for badminton video analysis
 */

export interface Keypoint {
  name: string
  x: number | null
  y: number | null
  confidence: number
}

export interface PlayerPosition {
  frame: number
  x: number
  y: number
}

export interface KeypointsFrame {
  frame: number
  keypoints: Keypoint[]
}

export interface PlayerMetrics {
  player_id: number
  total_distance: number // meters
  avg_speed: number // km/h
  max_speed: number // km/h
  positions: PlayerPosition[]
  keypoints_history: KeypointsFrame[]
}

export interface ShuttleMetrics {
  avg_speed: number // km/h
  max_speed: number // km/h
  shots_detected: number
  shot_speeds: number[]
}

export interface CourtKeypoint {
  name: string
  x: number
  y: number
  confidence: number
}

export interface CourtDimensions {
  width_meters: number
  length_meters: number
}

// Court keypoints from the 22-keypoint model
export interface CourtKeypoints22 {
  keypoints: number[][]  // [[x, y, confidence], ...] - 22 keypoints
  bbox?: number[] | null  // [x1, y1, x2, y2]
  confidence: number
}

export interface CourtDetection {
  detected: boolean
  confidence: number
  keypoints: CourtKeypoint[]
  court_corners: number[][] | null
  court_dimensions: CourtDimensions
  regions?: CourtRegion[]  // For region model (8 zones)
  court_keypoints_22?: CourtKeypoints22 | null  // For keypoint model (22 keypoints)
  court_model_used?: string  // "region" or "keypoint"
}

// Court region from region-based detection
export interface CourtRegion {
  name: string
  class_id: number
  bbox: number[]  // [x1, y1, x2, y2]
  center: number[]  // [x, y]
  confidence: number
}

export interface CourtDetectionStatus {
  enabled: boolean
  model_id: string
  api_url: string
  court_dimensions: CourtDimensions
}

export interface CourtPosition {
  x: number | null
  y: number | null
  zone: string
  distance_to_net: number | null
}

// Pose types for player pose classification
export type PoseType =
  | 'standing'
  | 'ready'
  | 'serving'
  | 'smash'
  | 'overhead'
  | 'forehand'
  | 'backhand'
  | 'lunge'
  | 'jump'
  | 'recovery'
  | 'unknown'

// Body angles for pose analysis
export interface BodyAngles {
  left_elbow: number | null
  right_elbow: number | null
  left_shoulder: number | null
  right_shoulder: number | null
  left_knee: number | null
  right_knee: number | null
  left_hip: number | null
  right_hip: number | null
  torso_lean: number | null
}

// Player pose information
export interface PlayerPose {
  pose_type: PoseType
  confidence: number
  body_angles?: BodyAngles | null
}

export interface FramePlayer {
  player_id: number
  keypoints: Keypoint[]
  center: { x: number; y: number }
  current_speed: number
  court_position?: CourtPosition | null
  pose?: PlayerPose | null
}

export interface BoundingBoxDetection {
  class: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
  class_id: number
  detection_id: string | null
}

export interface BadmintonDetections {
  frame: number
  players: BoundingBoxDetection[]
  shuttlecocks: BoundingBoxDetection[]
  rackets: BoundingBoxDetection[]
  other: BoundingBoxDetection[]
}

export interface SkeletonFrame {
  frame: number
  timestamp: number
  players: FramePlayer[]
  court_detected?: boolean
  badminton_detections?: BadmintonDetections | null
  shuttle_position?: { x: number; y: number } | null
  shuttle_speed_kmh?: number | null
}

// Shot type classification
export type ShotType = 'smash' | 'clear' | 'drop' | 'drive' | 'net_shot' | 'lob' | 'serve' | 'unknown'

// Shuttle trajectory position
export interface ShuttleTrajectoryPosition {
  frame: number
  pixel: { x: number; y: number }
  court: { x: number | null; y: number | null } | null
  confidence: number
}

// Single shuttle trajectory (one shot)
export interface ShuttleTrajectory {
  id: number
  shot_type: ShotType
  peak_speed_kmh: number
  avg_speed_kmh: number
  distance_m: number
  direction_angle: number
  positions: ShuttleTrajectoryPosition[]
}

// Shot type counts
export interface ShotTypeCounts {
  smash: number
  clear: number
  drop: number
  drive: number
  net_shot: number
}

// Speed statistics
export interface SpeedStats {
  fastest_shot_kmh: number
  avg_shot_speed_kmh: number
  all_shot_speeds: number[]
}

// Enhanced shuttle analytics
export interface ShuttleAnalytics {
  total_shots: number
  shot_types: ShotTypeCounts
  speed_stats: SpeedStats
  trajectories: ShuttleTrajectory[]
}

// Player zone coverage
export interface ZoneCoverage {
  front: number  // percentage
  mid: number
  back: number
  left: number
  center: number
  right: number
}

// Player zone analytics
export interface PlayerZoneAnalytics {
  zone_coverage: ZoneCoverage
  avg_distance_to_net_m: number
  heatmap: number[][]  // 2D grid of position frequency (normalized 0-1)
  position_count: number
}

// All player analytics
export interface PlayersZoneAnalytics {
  [playerId: string]: PlayerZoneAnalytics
}

export interface AnalysisResult {
  video_id: string
  duration: number // seconds
  fps: number
  total_frames: number
  processed_frames: number
  players: PlayerMetrics[]
  shuttle: ShuttleMetrics | null
  skeleton_data: SkeletonFrame[]
  court_detection: CourtDetection | null
  shuttle_analytics?: ShuttleAnalytics | null
  player_zone_analytics?: PlayersZoneAnalytics | null
}

export interface UploadResponse {
  video_id: string
  filename: string
  size: number
  status: string
}

export interface AnalyzeResponse {
  video_id: string
  status: string
  result: AnalysisResult
}

export interface ProgressUpdate {
  type: 'progress' | 'pong' | 'log'
  progress?: number
  frame?: number
  total_frames?: number
  // Log message fields (when type === 'log')
  message?: string
  level?: LogLevel
  category?: LogCategory
  timestamp?: number
}

// Log levels for processing log messages
export type LogLevel = 'info' | 'success' | 'warning' | 'error' | 'debug'

// Log categories for grouping messages
export type LogCategory = 'processing' | 'detection' | 'model' | 'court' | 'modal'

// Processing log entry
export interface ProcessingLog {
  id: number
  message: string
  level: LogLevel
  category: LogCategory
  timestamp: number
}

// Court detection model types
export type CourtModelType = 'region' | 'keypoint'

export interface AnalysisConfig {
  fps_sample_rate: number
  confidence_threshold: number
  track_shuttle: boolean
  calculate_speeds: boolean
  detect_court?: boolean
  court_detection_interval?: number
  court_model?: CourtModelType  // "region" (bounding box) or "keypoint" (22 keypoints)
}

// Skeleton connection indices for drawing
export const SKELETON_CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],  // Face
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // Arms
  [5, 11], [6, 12], [11, 12],  // Torso
  [11, 13], [13, 15], [12, 14], [14, 16]  // Legs
]

export const KEYPOINT_COLORS: Record<string, string> = {
  // Face keypoints
  nose: '#FF6B6B',
  left_eye: '#4ECDC4',
  right_eye: '#4ECDC4',
  left_ear: '#45B7D1',
  right_ear: '#45B7D1',
  // Upper body
  left_shoulder: '#96CEB4',
  right_shoulder: '#96CEB4',
  left_elbow: '#FFEAA7',
  right_elbow: '#FFEAA7',
  left_wrist: '#DDA0DD',
  right_wrist: '#DDA0DD',
  // Lower body
  left_hip: '#98D8C8',
  right_hip: '#98D8C8',
  left_knee: '#F7DC6F',
  right_knee: '#F7DC6F',
  left_ankle: '#BB8FCE',
  right_ankle: '#BB8FCE'
}

export const PLAYER_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Cyan
  '#45B7D1', // Blue
  '#96CEB4', // Green
]

// Court zone colors for heatmap visualization
export const COURT_ZONE_COLORS: Record<string, string> = {
  front_left: '#FF6B6B',
  front_center: '#FF8E8E',
  front_right: '#FFB0B0',
  mid_left: '#4ECDC4',
  mid_center: '#70D7D0',
  mid_right: '#92E1DC',
  back_left: '#45B7D1',
  back_center: '#67C5DB',
  back_right: '#89D3E5',
}

// Shot type colors for visualization
export const SHOT_TYPE_COLORS: Record<ShotType, string> = {
  smash: '#FF4444',      // Red - aggressive
  clear: '#4488FF',      // Blue - defensive
  drop: '#44FF88',       // Green - deceptive
  drive: '#FF8844',      // Orange - fast
  net_shot: '#8844FF',   // Purple - finesse
  lob: '#44FFFF',        // Cyan - high
  serve: '#FFFF44',      // Yellow - start
  unknown: '#888888',    // Gray
}

// Pose type colors for visualization
export const POSE_TYPE_COLORS: Record<PoseType, string> = {
  standing: '#A0AEC0',   // Gray - neutral
  ready: '#48BB78',      // Green - prepared
  serving: '#ECC94B',    // Yellow - start
  smash: '#F56565',      // Red - aggressive
  overhead: '#ED8936',   // Orange - high reach
  forehand: '#4299E1',   // Blue - lateral
  backhand: '#9F7AEA',   // Purple - lateral
  lunge: '#38B2AC',      // Teal - movement
  jump: '#F687B3',       // Pink - aerial
  recovery: '#68D391',   // Light green - reset
  unknown: '#718096',    // Dark gray
}

// Pose type display names
export const POSE_TYPE_NAMES: Record<PoseType, string> = {
  standing: 'Standing',
  ready: 'Ready Position',
  serving: 'Serving',
  smash: 'Smash',
  overhead: 'Overhead',
  forehand: 'Forehand',
  backhand: 'Backhand',
  lunge: 'Lunge',
  jump: 'Jump',
  recovery: 'Recovery',
  unknown: 'Unknown',
}

// Pose type icons (emoji)
export const POSE_TYPE_ICONS: Record<PoseType, string> = {
  standing: '🧍',
  ready: '🏸',
  serving: '🎯',
  smash: '💥',
  overhead: '🙌',
  forehand: '👉',
  backhand: '👈',
  lunge: '🦵',
  jump: '🦘',
  recovery: '🔄',
  unknown: '❓',
}

// Court keypoint names
export const COURT_KEYPOINT_NAMES = [
  'tr-1', 'tr-2',           // Top-right corner
  'tl-1', 'tl-2', 'tl-3', 'tl-4',  // Top-left corner
  'tm-1', 'tm-2',           // Top-middle (centerline)
  'bm-1', 'bm-2',           // Bottom-middle (centerline)
  'br-1', 'br-2',           // Bottom-right corner
  'bl-1', 'bl-2', 'bl-3', 'bl-4',  // Bottom-left corner
] as const

// Standard badminton court dimensions (in meters)
export const COURT_DIMENSIONS = {
  length: 13.4,              // Full court length
  width_doubles: 6.1,        // Doubles court width
  width_singles: 5.18,       // Singles court width
  service_line: 1.98,        // Distance from net to short service line
  back_boundary_service: 0.76, // Distance from back line to long service line (doubles)
  net_height_center: 1.524,  // Net height at center
  net_height_posts: 1.55,    // Net height at posts
} as const

// =============================================================================
// SPEED ANALYTICS TYPES
// =============================================================================

// Speed zone classification
export type SpeedZone = 'standing' | 'walking' | 'jogging' | 'running' | 'sprinting' | 'explosive'

// Speed zone thresholds (m/s)
export const SPEED_ZONE_THRESHOLDS: Record<SpeedZone, { min: number; max: number | null }> = {
  standing: { min: 0.0, max: 0.5 },
  walking: { min: 0.5, max: 1.5 },
  jogging: { min: 1.5, max: 3.5 },
  running: { min: 3.5, max: 5.5 },
  sprinting: { min: 5.5, max: 7.5 },
  explosive: { min: 7.5, max: null },  // null = infinity
}

// Speed zone colors for visualization
export const SPEED_ZONE_COLORS: Record<SpeedZone, string> = {
  standing: '#94A3B8',   // Slate gray
  walking: '#22C55E',    // Green
  jogging: '#3B82F6',    // Blue
  running: '#F59E0B',    // Amber
  sprinting: '#EF4444',  // Red
  explosive: '#9333EA',  // Purple
}

// Speed zone display names
export const SPEED_ZONE_NAMES: Record<SpeedZone, string> = {
  standing: 'Standing',
  walking: 'Walking',
  jogging: 'Jogging',
  running: 'Running',
  sprinting: 'Sprinting',
  explosive: 'Explosive',
}

// Single speed data point
export interface SpeedDataPoint {
  frame: number
  timestamp: number
  speed_mps: number
  speed_kmh: number
  zone: SpeedZone
  zone_color: string
  position?: { x: number; y: number }
  court_position?: { x: number | null; y: number | null } | null
  smoothed: boolean
}

// Player speed statistics
export interface PlayerSpeedStats {
  player_id: number
  current: {
    speed_mps: number
    speed_kmh: number
    zone: SpeedZone
    zone_color: string
  }
  max: {
    speed_mps: number
    speed_kmh: number
  }
  avg: {
    speed_mps: number
    speed_kmh: number
  }
  total_distance_m: number
  zone_distribution: Record<SpeedZone, number>  // percentage per zone
}

// Speed graph data for a single player
export interface PlayerSpeedData {
  window_data: SpeedDataPoint[]  // Sliding window for real-time display
  full_history: SpeedDataPoint[]  // Complete history for post-match analysis
}

// Zone threshold information
export interface ZoneThreshold {
  zone: SpeedZone
  min_mps: number
  max_mps: number | null
  min_kmh: number
  max_kmh: number | null
  color: string
}

// Time range for graph X-axis
export interface TimeRange {
  min: number
  max: number
}

// Complete speed graph data from API
export interface SpeedGraphData {
  players: Record<number, PlayerSpeedData>
  statistics: Record<number, PlayerSpeedStats>
  zone_thresholds: ZoneThreshold[]
  time_range: TimeRange | null
  window_seconds: number
  fps: number
}

// Speed API response
export interface SpeedApiResponse {
  video_id: string
  fps: number
  speed_data: SpeedGraphData
  status: string
}

// Speed timeline data (optimized for charts)
export interface SpeedTimelinePlayer {
  frames: number[]
  timestamps: number[]
  speeds_mps: number[]
  speeds_kmh: number[]
  zones: SpeedZone[]
  stats?: {
    max_mps: number
    max_kmh: number
    avg_mps: number
    avg_kmh: number
    data_points: number
  }
}

// Timeline API response
export interface SpeedTimelineResponse {
  video_id: string
  frame_range: {
    start: number | null
    end: number | null
    total_frames: number
  }
  sample_rate: number
  fps: number
  players: Record<number, SpeedTimelinePlayer>
  zone_colors: Record<SpeedZone, string>
  status: string
}

// Player colors for speed graph
export const PLAYER_SPEED_COLORS = [
  '#3B82F6',  // Player 1 - Blue
  '#EF4444',  // Player 2 - Red
  '#22C55E',  // Player 3 - Green (if more players)
  '#F59E0B',  // Player 4 - Amber
]
