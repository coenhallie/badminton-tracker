/**
 * API Service for Badminton Tracker (Supabase backend)
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Debounce / throttle utilities
 * - Response caching for repeated requests
 */

import type {
  UploadResponse,
  AnalyzeResponse,
  AnalysisResult,
  AnalysisConfig,
  ProgressUpdate,
  SkeletonFrame,
} from '@/types/analysis'
import { supabase } from '@/lib/supabase'
import { computeHeatmap } from '@/composables/useHeatmap'
import { computeZoneAnalytics } from '@/composables/useZoneAnalytics'

// =============================================================================
// PERFORMANCE OPTIMIZATION: Utility functions
// =============================================================================

/**
 * Create a debounced version of a function
 * Delays execution until after wait milliseconds have elapsed since the last call
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function (this: unknown, ...args: Parameters<T>) {
    if (timeoutId !== null) {
      clearTimeout(timeoutId)
    }
    timeoutId = setTimeout(() => {
      timeoutId = null
      fn.apply(this, args)
    }, wait)
  }
}

/**
 * Create a throttled version of a function
 * Executes at most once per wait milliseconds
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  wait: number
): (...args: Parameters<T>) => void {
  let lastTime = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function (this: unknown, ...args: Parameters<T>) {
    const now = Date.now()
    const remaining = wait - (now - lastTime)

    if (remaining <= 0) {
      if (timeoutId !== null) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      lastTime = now
      fn.apply(this, args)
    } else if (timeoutId === null) {
      timeoutId = setTimeout(() => {
        timeoutId = null
        lastTime = Date.now()
        fn.apply(this, args)
      }, remaining)
    }
  }
}

/**
 * Simple cache with TTL (time-to-live)
 * Useful for caching API responses
 */
export class SimpleCache<T> {
  private cache = new Map<string, { value: T; expiry: number }>()
  private defaultTTL: number

  constructor(defaultTTLMs: number = 30000) {
    this.defaultTTL = defaultTTLMs
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key)
    if (!entry) return undefined
    if (Date.now() > entry.expiry) {
      this.cache.delete(key)
      return undefined
    }
    return entry.value
  }

  set(key: string, value: T, ttl: number = this.defaultTTL): void {
    this.cache.set(key, { value, expiry: Date.now() + ttl })
  }

  clear(): void {
    this.cache.clear()
  }

  delete(key: string): void {
    this.cache.delete(key)
  }
}

// Shared cache instance for API responses
export const apiCache = new SimpleCache<unknown>(60000) // 1 minute default TTL

// Re-export type imports so external consumers can still use them via this module if needed
export type { UploadResponse, AnalyzeResponse, AnalysisResult, AnalysisConfig, ProgressUpdate, SkeletonFrame }

// =============================================================================
// Video URL
// =============================================================================

/**
 * Fetch a signed URL for the original uploaded video.
 */
export async function fetchVideoUrl(videoId: string): Promise<string> {
  const { data: row, error } = await supabase
    .from('videos')
    .select('storage_path')
    .eq('id', videoId)
    .single()
  if (error || !row) throw error ?? new Error('Video not found')
  const { data: signed, error: e2 } = await supabase
    .storage.from('videos')
    .createSignedUrl(row.storage_path, 3600)
  if (e2 || !signed) throw e2 ?? new Error('Could not sign URL')
  return signed.signedUrl
}

// =============================================================================
// Health check
// =============================================================================

/**
 * Health check response interface
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy'
  timestamp: number
  version: string
  backend: 'local' | 'convex' | 'supabase'
  components: {
    pose_model?: string
    court_detector?: string
    multi_model_detector?: string
    modal_inference?: string
    convex?: string
    database?: string
  }
  system?: {
    cpu_percent: number
    memory_percent: number
    memory_available_gb: number
  }
  stats?: {
    total_videos: number
    processing: number
    videos_by_status: Record<string, number>
  }
  active_analyses?: number
  error?: string
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const { error } = await supabase.from('videos').select('id').limit(1)
    return !error
  } catch {
    return false
  }
}

export async function getApiHealthDetails(): Promise<HealthCheckResponse | null> {
  // Trimmed down — UI only needs the boolean from checkApiHealth.
  const ok = await checkApiHealth()
  return ok
    ? {
        status: 'healthy',
        timestamp: Date.now(),
        version: 'supabase',
        backend: 'supabase',
        components: {},
      }
    : null
}

// =============================================================================
// Manual Court Keypoints API
// =============================================================================

export interface ManualKeypointsRequest {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  net_left?: number[]
  net_right?: number[]
  service_line_near_left?: number[]
  service_line_near_right?: number[]
  service_line_far_left?: number[]
  service_line_far_right?: number[]
  doubles_left_near?: number[]
  doubles_right_near?: number[]
}

export interface ManualKeypointsStatus {
  has_manual_keypoints: boolean
  keypoints: ManualKeypointsRequest | null
}

export interface ManualKeypointsResponse {
  status: string
  message: string
  keypoints: ManualKeypointsRequest
}

export async function getManualKeypointsStatus(videoId: string): Promise<ManualKeypointsStatus> {
  const { data, error } = await supabase
    .from('videos')
    .select('manual_court_keypoints')
    .eq('id', videoId)
    .single()
  if (error || !data) return { has_manual_keypoints: false, keypoints: null }
  return {
    has_manual_keypoints: !!data.manual_court_keypoints,
    keypoints: data.manual_court_keypoints,
  }
}

export async function setManualCourtKeypoints(
  keypoints: ManualKeypointsRequest,
  videoId: string
): Promise<ManualKeypointsResponse> {
  const { error } = await supabase
    .from('videos')
    .update({ manual_court_keypoints: keypoints })
    .eq('id', videoId)
  if (error) throw error
  return { status: 'success', message: 'ok', keypoints }
}

// =============================================================================
// Heatmap API
// =============================================================================

/**
 * Heatmap data structure matching backend response
 */
export interface HeatmapData {
  video_id: string
  width: number
  height: number
  colormap: string
  combined_heatmap?: number[][] // 2D array of intensity values (0-255)
  player_heatmaps?: Record<string, number[][]> // Per-player heatmaps
  total_frames: number
  player_position_counts: Record<string, number>
  video_width: number
  video_height: number
  court_corners?: number[][]
}

export interface HeatmapResponse {
  video_id: string
  player_id: number | null
  heatmap: HeatmapData
  status: string
}

// Heatmap cache with longer TTL (heatmaps are expensive to compute)
const heatmapCache = new SimpleCache<HeatmapData>(300000) // 5 minute TTL

export async function getHeatmap(
  videoId: string,
  playerId?: number
): Promise<HeatmapResponse> {
  const cacheKey = `heatmap:${videoId}:${playerId ?? 'all'}`
  const cached = heatmapCache.get(cacheKey) as HeatmapData | undefined
  if (cached) {
    return {
      video_id: videoId,
      player_id: playerId ?? null,
      heatmap: cached,
      status: 'success',
    }
  }
  const heatmap = await computeHeatmap(videoId, playerId)
  heatmapCache.set(cacheKey, heatmap)
  return {
    video_id: videoId,
    player_id: playerId ?? null,
    heatmap,
    status: 'success',
  }
}

export async function preloadHeatmap(videoId: string): Promise<void> {
  try {
    await getHeatmap(videoId)
    console.log(`[Heatmap] Preloaded heatmap for video ${videoId}`)
  } catch (e) {
    console.warn('[Heatmap] preload failed', e)
  }
}

// =============================================================================
// Speed Analytics API
// =============================================================================

export interface PlayerSpeedStatsResponse {
  player_id: number
  current: {
    speed_mps: number
    speed_kmh: number
    zone: string
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
  zone_distribution: Record<string, number>
}

export interface SpeedDataResponse {
  video_id: string
  fps: number
  speed_data: {
    players: Record<
      string,
      {
        window_data: Array<{ timestamp: number; speed_mps: number; speed_kmh: number }>
        full_history: Array<{ timestamp: number; speed_mps: number; speed_kmh: number }>
      }
    >
    statistics: Record<string, PlayerSpeedStatsResponse>
    zone_thresholds: Array<{
      zone: string
      min_mps: number
      max_mps: number | null
      color: string
    }>
    time_range: {
      min: number
      max: number
    }
    manual_keypoints_used: boolean
    detection_source: string
  }
  status: string
  manual_keypoints_used: boolean
  detection_source: string
}

// Speed data cache with moderate TTL
const speedCache = new SimpleCache<SpeedDataResponse['speed_data']>(60000)

export async function getSpeedData(
  videoId: string,
  windowSeconds: number = 60.0,
  forceRefresh: boolean = false
): Promise<SpeedDataResponse> {
  const cacheKey = `speed:${videoId}:${windowSeconds}`
  if (!forceRefresh) {
    const cached = speedCache.get(cacheKey)
    if (cached) {
      return {
        video_id: videoId,
        fps: 30,
        speed_data: cached,
        status: 'success',
        manual_keypoints_used: cached.manual_keypoints_used ?? false,
        detection_source: cached.detection_source ?? 'modal',
      }
    }
  }
  const { data, error } = await supabase.functions.invoke('recalculate-speeds', {
    body: { video_id: videoId },
  })
  if (error) throw error
  const resp = data as SpeedDataResponse
  if (resp.speed_data) speedCache.set(cacheKey, resp.speed_data)
  return resp
}

export function clearSpeedCache(videoId?: string): void {
  if (videoId) {
    speedCache.delete(`speed:${videoId}:30`)
    speedCache.delete(`speed:${videoId}:60`)
    speedCache.delete(`speed:${videoId}:120`)
  } else {
    speedCache.clear()
  }
}

export async function triggerSpeedRecalculation(videoId: string): Promise<SpeedDataResponse> {
  console.log(`[Speed] Triggering speed recalculation for video ${videoId}`)
  clearSpeedCache(videoId)
  const data = await getSpeedData(videoId, 60.0, true)
  console.log(`[Speed] Recalculation complete - manual_keypoints_used: ${data.manual_keypoints_used}`)
  return data
}

/**
 * Manual court keypoints for speed calculation
 */
export interface ManualCourtKeypoints {
  top_left: { x: number; y: number }
  top_right: { x: number; y: number }
  bottom_left: { x: number; y: number }
  bottom_right: { x: number; y: number }
}

/**
 * Request body for recalculating speeds from skeleton data
 */
export interface RecalculateSpeedsRequest {
  skeleton_data: SkeletonFrame[]
  fps: number
  video_width: number
  video_height: number
  manual_court_keypoints?: ManualCourtKeypoints
}

/**
 * Response from speed recalculation endpoint
 */
export interface RecalculateSpeedsResponse {
  status: string
  players: Array<{
    player_id: number
    avg_speed: number
    max_speed: number
    total_distance: number
    current_speed: number
    zone_distribution: Record<string, number>
  }>
  speed_data: SpeedDataResponse['speed_data']
  frames_processed: number
  fps: number
  manual_keypoints_used: boolean
  detection_source: string
  methodology: {
    description: string
    max_speed_cap_kmh: number
    uses_homography: boolean
  }
}

// Speed timeline response type for per-frame speed data
export interface SpeedTimelineResponse {
  video_id: string
  frame_range: {
    start: number | null
    end: number | null
    total_frames: number
  }
  sample_rate: number
  fps: number
  players: Record<
    string,
    {
      frames: number[]
      timestamps: number[]
      speeds_mps: number[]
      speeds_kmh: number[]
      zones: string[]
      stats?: {
        max_mps: number
        max_kmh: number
        avg_mps: number
        avg_kmh: number
        data_points: number
      }
    }
  >
  zone_colors: Record<string, string>
  manual_keypoints_used: boolean
  detection_source: string
  status: string
}

export async function getSpeedTimeline(
  videoId: string,
  sampleRate: number = 1
): Promise<SpeedTimelineResponse> {
  const { data, error } = await supabase.functions.invoke('recalculate-speeds', {
    body: { video_id: videoId },
  })
  if (error) throw error

  // The Edge Function returns the same shape the Convex /api/speed endpoint did:
  //   { video_id, speed_data: { frame_data, statistics, ... }, manual_keypoints_used, detection_source }
  // Reshape into SpeedTimelineResponse.
  const payload = data as {
    video_id?: string
    speed_data?: {
      frame_data?: Array<{ frame: number; players: Array<{ player_id: number; speed_kmh: number }> }>
      manual_keypoints_used?: boolean
    }
    manual_keypoints_used?: boolean
    detection_source?: string
  }

  const speedData = payload.speed_data
  const players: Record<
    string,
    { frames: number[]; timestamps: number[]; speeds_mps: number[]; speeds_kmh: number[]; zones: string[] }
  > = {}

  if (speedData?.frame_data) {
    for (const frame of speedData.frame_data) {
      for (const player of frame.players) {
        const pid = player.player_id.toString()
        if (!players[pid]) {
          players[pid] = { frames: [], timestamps: [], speeds_mps: [], speeds_kmh: [], zones: [] }
        }
        players[pid].frames.push(frame.frame)
        players[pid].timestamps.push(frame.frame / 30)
        players[pid].speeds_kmh.push(player.speed_kmh)
        players[pid].speeds_mps.push(player.speed_kmh / 3.6)
        players[pid].zones.push(
          player.speed_kmh > 15 ? 'sprint' : player.speed_kmh > 8 ? 'run' : 'walk'
        )
      }
    }
  }

  const manualKeypointsUsed =
    payload.manual_keypoints_used ?? speedData?.manual_keypoints_used ?? false
  const detectionSource = payload.detection_source ?? 'supabase'

  return {
    video_id: videoId,
    frame_range: { start: 0, end: null, total_frames: speedData?.frame_data?.length ?? 0 },
    sample_rate: sampleRate,
    fps: 30,
    players,
    zone_colors: { walk: '#4CAF50', run: '#FFC107', sprint: '#F44336' },
    manual_keypoints_used: manualKeypointsUsed,
    detection_source: detectionSource,
    status: 'success',
  }
}

// =============================================================================
// PDF Export API
// =============================================================================

/**
 * PDF export configuration options
 */
export interface PDFExportConfig {
  frame_number?: number | null
  include_heatmap?: boolean
  heatmap_colormap?: string
  heatmap_alpha?: number
  include_player_stats?: boolean
  include_shuttle_stats?: boolean
  include_court_info?: boolean
  include_speed_stats?: boolean
  title?: string
}

// PDF export with frontend data - includes analysis result for accurate display
export interface PDFExportWithDataConfig {
  frame_number?: number | null
  include_heatmap?: boolean
  heatmap_colormap?: string
  heatmap_alpha?: number
  title?: string
  duration: number
  fps: number
  total_frames: number
  processed_frames: number
  video_width: number
  video_height: number
  players: Array<{
    player_id: number
    total_distance: number
    avg_speed: number
    max_speed: number
    positions: Array<{ frame: number; x: number; y: number }>
  }>
  shuttle?: {
    avg_speed: number
    max_speed: number
    shots_detected: number
    shot_speeds: number[]
  } | null
  court_detection?: {
    detected: boolean
    confidence: number
    court_dimensions?: { width_meters: number; length_meters: number }
    court_corners?: number[][] | null
  } | null
  shuttle_analytics?: Record<string, unknown> | null
  player_zone_analytics?: Record<string, unknown> | null
}

/**
 * PDF export preview response
 */
export interface PDFExportPreview {
  video_id: string
  preview: {
    video_summary: {
      duration_seconds: number
      total_frames: number
      processed_frames: number
      fps: number
      video_dimensions: string
    }
    players_detected: number
    movement_summary: {
      total_distance_m: number
      avg_speed_kmh: number
      max_speed_kmh: number
    }
    shuttle_data_available: boolean
    shuttle_analytics_available: boolean
    court_detected: boolean
    zone_analytics_available: boolean
    skeleton_frames_available: boolean
  }
  export_options: {
    colormaps: string[]
    recommended_colormap: string
    default_heatmap_alpha: number
  }
  status: string
}

export async function downloadPDFExport(
  _videoId: string,
  _options?: {
    frame_number?: number
    include_heatmap?: boolean
    heatmap_colormap?: string
    heatmap_alpha?: number
  }
): Promise<void> {
  throw new Error('not yet migrated')
}

export async function exportPDFWithFrontendData(
  _videoId: string,
  _data: PDFExportWithDataConfig
): Promise<void> {
  throw new Error('not yet migrated')
}

// =============================================================================
// Zone Analytics API
// =============================================================================

/**
 * Zone coverage data for a single player
 */
export interface PlayerZoneData {
  zone_coverage: {
    front: number
    mid: number
    back: number
    left: number
    center: number
    right: number
  }
  avg_distance_to_net_m: number
  position_count: number
}

/**
 * Response from recalculate zone analytics endpoint
 */
export interface RecalculatedZoneAnalyticsResponse {
  video_id: string
  player_zone_analytics: Record<string, PlayerZoneData>
  recalculated: boolean
  manual_keypoints_used: boolean
  total_skeleton_frames: number
  status: string
  message: string
}

// Zone analytics cache
const zoneAnalyticsCache = new SimpleCache<RecalculatedZoneAnalyticsResponse>(120000) // 2 minute TTL

export async function getRecalculatedZoneAnalytics(
  videoId: string,
  forceRefresh: boolean = false
): Promise<RecalculatedZoneAnalyticsResponse> {
  const cacheKey = `zone-analytics:${videoId}`
  if (!forceRefresh) {
    const cached = zoneAnalyticsCache.get(cacheKey)
    if (cached) return cached
  }
  const data = await computeZoneAnalytics(videoId)
  zoneAnalyticsCache.set(cacheKey, data)
  return data
}

/**
 * Clear zone analytics cache (call when keypoints change)
 */
export function clearZoneAnalyticsCache(videoId?: string): void {
  if (videoId) {
    zoneAnalyticsCache.delete(`zone-analytics:${videoId}`)
  } else {
    zoneAnalyticsCache.clear()
  }
}

// Internal exports of caches for use by composables / wrappers
export { heatmapCache, speedCache, zoneAnalyticsCache }
