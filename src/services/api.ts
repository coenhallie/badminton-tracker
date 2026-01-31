/**
 * API Service for Badminton Tracker Backend
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Debounce utility for throttling frequent operations
 * - Response caching for repeated requests
 * - Throttled callbacks for WebSocket progress updates
 */

import type {
  UploadResponse,
  AnalyzeResponse,
  AnalysisResult,
  AnalysisConfig,
  ProgressUpdate,
  SkeletonFrame
} from '@/types/analysis'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_BASE_URL = API_BASE_URL.replace('http', 'ws')

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
  
  return function(this: unknown, ...args: Parameters<T>) {
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
  
  return function(this: unknown, ...args: Parameters<T>) {
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

/**
 * Upload a video file to the server
 */
export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to upload video')
  }

  return response.json()
}

/**
 * Start video analysis
 */
export async function analyzeVideo(
  videoId: string,
  config?: Partial<AnalysisConfig>
): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/api/analyze/${videoId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: config ? JSON.stringify(config) : undefined
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to analyze video')
  }

  return response.json()
}

/**
 * Get analysis results for a video
 */
export async function getResults(videoId: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/api/results/${videoId}`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get results')
  }

  return response.json()
}

/**
 * Get skeleton overlay data for client-side rendering
 */
export async function getSkeletonData(
  videoId: string
): Promise<{ video_id: string; skeleton_data: SkeletonFrame[] }> {
  const response = await fetch(`${API_BASE_URL}/api/skeleton/${videoId}`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get skeleton data')
  }

  return response.json()
}

/**
 * Get processed video URL (with skeleton overlay baked in)
 */
export function getProcessedVideoUrl(videoId: string): string {
  return `${API_BASE_URL}/api/video/${videoId}`
}

/**
 * Get original uploaded video URL (browser-compatible streaming)
 */
export function getOriginalVideoUrl(videoId: string): string {
  return `${API_BASE_URL}/api/original/${videoId}`
}

/**
 * Get original uploaded video URL (legacy - uses static file serving)
 * @deprecated Use getOriginalVideoUrl instead
 */
export function getUploadedVideoUrl(videoId: string, extension: string = '.mp4'): string {
  return `${API_BASE_URL}/uploads/${videoId}${extension}`
}

/**
 * Delete a video and its analysis
 */
export async function deleteVideo(videoId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/video/${videoId}`, {
    method: 'DELETE'
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to delete video')
  }
}

/**
 * WebSocket connection for real-time progress updates
 */
export class AnalysisProgressSocket {
  private ws: WebSocket | null = null
  private videoId: string
  private onProgress: (update: ProgressUpdate) => void
  private onError: (error: Error) => void
  private onComplete: () => void
  private pingInterval: number | null = null

  constructor(
    videoId: string,
    onProgress: (update: ProgressUpdate) => void,
    onError: (error: Error) => void,
    onComplete: () => void
  ) {
    this.videoId = videoId
    this.onProgress = onProgress
    this.onError = onError
    this.onComplete = onComplete
  }

  connect(): void {
    this.ws = new WebSocket(`${WS_BASE_URL}/ws/${this.videoId}`)

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      // Send ping every 30 seconds to keep connection alive
      this.pingInterval = window.setInterval(() => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send('ping')
        }
      }, 30000)
    }

    this.ws.onmessage = (event) => {
      try {
        const data: ProgressUpdate = JSON.parse(event.data)
        if (data.type === 'progress') {
          this.onProgress(data)
          if (data.progress && data.progress >= 100) {
            this.onComplete()
          }
        } else if (data.type === 'log') {
          // Pass log messages to the progress handler
          this.onProgress(data)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    this.ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      this.onError(new Error('WebSocket connection error'))
    }

    this.ws.onclose = () => {
      console.log('WebSocket closed')
      if (this.pingInterval) {
        clearInterval(this.pingInterval)
        this.pingInterval = null
      }
    }
  }

  disconnect(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval)
      this.pingInterval = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

/**
 * Check API health
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`)
    return response.ok
  } catch {
    return false
  }
}

// ============================================================================
// Manual Court Keypoints API
// ============================================================================

export interface ManualKeypointsRequest {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
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

/**
 * Get manual court keypoints status
 */
export async function getManualKeypointsStatus(): Promise<ManualKeypointsStatus> {
  const response = await fetch(`${API_BASE_URL}/api/court-keypoints/manual/status`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get manual keypoints status')
  }

  return response.json()
}

/**
 * Set manual court keypoints
 */
export async function setManualCourtKeypoints(keypoints: ManualKeypointsRequest): Promise<ManualKeypointsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/court-keypoints/manual/set`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(keypoints)
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to set manual keypoints')
  }

  return response.json()
}

/**
 * Clear manual court keypoints
 */
export async function clearManualCourtKeypoints(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/court-keypoints/manual/clear`, {
    method: 'DELETE'
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to clear manual keypoints')
  }

  return response.json()
}

// ============================================================================
// Heatmap API
// ============================================================================

/**
 * Heatmap data structure matching backend response
 */
export interface HeatmapData {
  video_id: string
  width: number
  height: number
  colormap: string
  combined_heatmap?: number[][]  // 2D array of intensity values (0-255)
  player_heatmaps?: Record<string, number[][]>  // Per-player heatmaps
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

export interface HeatmapConfigRequest {
  colormap?: string  // turbo, parula, inferno, viridis, plasma, hot, jet, cool
  kernel_size?: number  // Gaussian kernel size
  sigma?: number  // Gaussian spread
  decay_rate?: number  // Per-frame decay rate
  intensity_scale?: number  // Heat intensity multiplier
}

export interface HeatmapGenerateRequest {
  config?: HeatmapConfigRequest
  player_id?: number  // null = all players
  start_frame?: number
  end_frame?: number
}

export interface HeatmapGenerateResponse {
  video_id: string
  player_id: number | null
  frame_range: {
    start: number | null
    end: number | null
    frames_processed: number
  }
  heatmap: HeatmapData
  status: string
}

export interface ColorMapInfo {
  name: string
  display_name: string
  description: string
  opencv_code: string
}

export interface ColormapsResponse {
  colormaps: ColorMapInfo[]
  default: string
  recommended_for_badminton: string
}

// Heatmap cache with longer TTL (heatmaps are expensive to compute)
const heatmapCache = new SimpleCache<HeatmapData>(300000) // 5 minute TTL

/**
 * Get heatmap data for a video
 * Uses caching to avoid repeated expensive computations
 */
export async function getHeatmap(
  videoId: string,
  playerId?: number
): Promise<HeatmapResponse> {
  // Check cache first
  const cacheKey = `heatmap:${videoId}:${playerId ?? 'all'}`
  const cached = heatmapCache.get(cacheKey) as HeatmapData | undefined
  if (cached) {
    return {
      video_id: videoId,
      player_id: playerId ?? null,
      heatmap: cached,
      status: 'success'
    }
  }
  
  // Build URL with optional player_id
  let url = `${API_BASE_URL}/api/heatmap/${videoId}`
  if (playerId !== undefined) {
    url += `?player_id=${playerId}`
  }

  const response = await fetch(url)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get heatmap data')
  }

  const data: HeatmapResponse = await response.json()
  
  // Cache the result
  if (data.heatmap) {
    heatmapCache.set(cacheKey, data.heatmap)
  }
  
  return data
}

/**
 * Generate heatmap with custom configuration
 */
export async function generateHeatmap(
  videoId: string,
  request: HeatmapGenerateRequest
): Promise<HeatmapGenerateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/heatmap/${videoId}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to generate heatmap')
  }

  const data: HeatmapGenerateResponse = await response.json()
  
  // Update cache with new data
  const cacheKey = `heatmap:${videoId}:${request.player_id ?? 'all'}`
  if (data.heatmap) {
    heatmapCache.set(cacheKey, data.heatmap)
  }
  
  return data
}

/**
 * Get heatmap data up to a specific frame (for progressive rendering)
 */
export async function getHeatmapAtFrame(
  videoId: string,
  frameNumber: number,
  playerId?: number,
  colormap: string = 'turbo',
  decayRate: number = 0.995
): Promise<HeatmapResponse> {
  let url = `${API_BASE_URL}/api/heatmap/${videoId}/frame/${frameNumber}`
  url += `?colormap=${colormap}&decay_rate=${decayRate}`
  if (playerId !== undefined) {
    url += `&player_id=${playerId}`
  }

  const response = await fetch(url)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get heatmap at frame')
  }

  return response.json()
}

/**
 * Get available colormap options
 */
export async function getHeatmapColormaps(): Promise<ColormapsResponse> {
  // Use shared API cache
  const cacheKey = 'heatmap:colormaps'
  const cached = apiCache.get(cacheKey) as ColormapsResponse | undefined
  if (cached) {
    return cached
  }

  const response = await fetch(`${API_BASE_URL}/api/heatmap/colormaps`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get colormaps')
  }

  const data: ColormapsResponse = await response.json()
  apiCache.set(cacheKey, data, 3600000) // Cache for 1 hour
  
  return data
}

/**
 * Clear heatmap cache (for a specific video or all)
 */
export async function clearHeatmapCache(videoId?: string): Promise<void> {
  // Clear local cache
  if (videoId) {
    heatmapCache.delete(`heatmap:${videoId}:all`)
    heatmapCache.delete(`heatmap:${videoId}:1`)
    heatmapCache.delete(`heatmap:${videoId}:2`)
  } else {
    heatmapCache.clear()
  }
  
  // Also clear on backend
  const response = await fetch(`${API_BASE_URL}/api/heatmap/cache/clear`, {
    method: 'DELETE'
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to clear heatmap cache')
  }
}

/**
 * Preload heatmap data for a video (call after analysis complete)
 * This warms the cache for instant heatmap display
 */
export async function preloadHeatmap(videoId: string): Promise<void> {
  try {
    // Preload combined heatmap
    await getHeatmap(videoId)
    console.log(`[Heatmap] Preloaded heatmap for video ${videoId}`)
  } catch (error) {
    console.warn(`[Heatmap] Failed to preload heatmap:`, error)
  }
}

// ============================================================================
// Speed Analytics API
// ============================================================================

/**
 * Speed data response structure from the backend
 * Contains per-player speed history and statistics
 */
/**
 * Player speed statistics matching backend PlayerSpeedStats.to_dict()
 * Uses nested structure: { current: { speed_kmh }, max: { speed_kmh }, avg: { speed_kmh } }
 */
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
    players: Record<string, {
      window_data: Array<{ timestamp: number; speed_mps: number; speed_kmh: number }>
      full_history: Array<{ timestamp: number; speed_mps: number; speed_kmh: number }>
    }>
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
const speedCache = new SimpleCache<SpeedDataResponse['speed_data']>(60000) // 1 minute TTL

/**
 * Get speed data for a video
 *
 * IMPORTANT: This endpoint recalculates speeds from skeleton data using
 * manual court keypoints if they have been set. This provides accurate
 * speed calculations even when called after video processing is complete.
 *
 * @param videoId - The video ID to get speed data for
 * @param windowSeconds - Sliding window size for display (default: 60)
 * @param forceRefresh - If true, bypasses cache and fetches fresh data
 */
export async function getSpeedData(
  videoId: string,
  windowSeconds: number = 60.0,
  forceRefresh: boolean = false
): Promise<SpeedDataResponse> {
  // Check cache first (unless force refresh)
  const cacheKey = `speed:${videoId}:${windowSeconds}`
  if (!forceRefresh) {
    const cached = speedCache.get(cacheKey)
    if (cached) {
      return {
        video_id: videoId,
        fps: 30, // Approximate, actual comes from server
        speed_data: cached,
        status: 'success',
        manual_keypoints_used: cached.manual_keypoints_used,
        detection_source: cached.detection_source
      }
    }
  }

  const url = `${API_BASE_URL}/api/speed/${videoId}?window_seconds=${windowSeconds}`
  const response = await fetch(url)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get speed data')
  }

  const data: SpeedDataResponse = await response.json()
  
  // Cache the result
  if (data.speed_data) {
    speedCache.set(cacheKey, data.speed_data)
  }
  
  return data
}

/**
 * Clear speed data cache (call when keypoints change)
 */
export function clearSpeedCache(videoId?: string): void {
  if (videoId) {
    // Clear all window sizes for this video
    speedCache.delete(`speed:${videoId}:30`)
    speedCache.delete(`speed:${videoId}:60`)
    speedCache.delete(`speed:${videoId}:120`)
  } else {
    speedCache.clear()
  }
}

/**
 * Trigger speed recalculation after court keypoints are set
 * This is called when manual keypoints are confirmed and video playback starts
 *
 * @param videoId - The video ID to recalculate speeds for
 * @returns Promise with the recalculated speed data
 */
export async function triggerSpeedRecalculation(videoId: string): Promise<SpeedDataResponse> {
  console.log(`[Speed] Triggering speed recalculation for video ${videoId}`)
  
  // Clear cache to force fresh calculation
  clearSpeedCache(videoId)
  
  // Fetch fresh speed data (backend will use manual keypoints if set)
  const data = await getSpeedData(videoId, 60.0, true)
  
  console.log(`[Speed] Recalculation complete - manual_keypoints_used: ${data.manual_keypoints_used}`)
  
  return data
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
  players: Record<string, {
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
  }>
  zone_colors: Record<string, string>
  manual_keypoints_used: boolean
  detection_source: string
  status: string
}

/**
 * Get per-frame speed timeline data
 * Used to update skeleton_data with calculated speeds after delayed calculation
 *
 * @param videoId - Video ID to get timeline for
 * @param sampleRate - Only include every Nth frame (default: 1 = all frames)
 */
export async function getSpeedTimeline(
  videoId: string,
  sampleRate: number = 1
): Promise<SpeedTimelineResponse> {
  const url = `${API_BASE_URL}/api/speed/${videoId}/timeline?sample_rate=${sampleRate}`
  const response = await fetch(url)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get speed timeline')
  }
  
  return response.json()
}

// ============================================================================
// PDF Export API
// ============================================================================

/**
 * PDF export configuration options
 */
export interface PDFExportConfig {
  frame_number?: number | null  // Frame to use for heatmap visualization
  include_heatmap?: boolean
  heatmap_colormap?: string  // turbo, parula, inferno, viridis, plasma, hot
  heatmap_alpha?: number  // 0.0 - 1.0
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
  // Data from frontend AnalysisResult
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

/**
 * Get PDF export URL for direct download
 * Use this to create a download link or trigger browser download
 *
 * @param videoId - Video ID to export
 * @param options - Optional configuration (appended as query params)
 */
export function getPDFExportUrl(
  videoId: string,
  options?: {
    frame_number?: number
    include_heatmap?: boolean
    heatmap_colormap?: string
    heatmap_alpha?: number
  }
): string {
  let url = `${API_BASE_URL}/api/export/pdf/${videoId}`
  
  const params = new URLSearchParams()
  if (options?.frame_number !== undefined) {
    params.append('frame_number', options.frame_number.toString())
  }
  if (options?.include_heatmap !== undefined) {
    params.append('include_heatmap', options.include_heatmap.toString())
  }
  if (options?.heatmap_colormap !== undefined) {
    params.append('heatmap_colormap', options.heatmap_colormap)
  }
  if (options?.heatmap_alpha !== undefined) {
    params.append('heatmap_alpha', options.heatmap_alpha.toString())
  }
  
  const queryString = params.toString()
  if (queryString) {
    url += `?${queryString}`
  }
  
  return url
}

/**
 * Download PDF export directly
 * This triggers an immediate download of the PDF file
 *
 * @param videoId - Video ID to export
 * @param options - Optional configuration
 */
export async function downloadPDFExport(
  videoId: string,
  options?: {
    frame_number?: number
    include_heatmap?: boolean
    heatmap_colormap?: string
    heatmap_alpha?: number
  }
): Promise<void> {
  const url = getPDFExportUrl(videoId, options)
  
  try {
    const response = await fetch(url)
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to generate PDF')
    }
    
    // Get the blob and create download link
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    
    // Create temporary link and trigger download
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `badminton_analysis_${videoId.slice(0, 8)}.pdf`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // Clean up object URL
    window.URL.revokeObjectURL(downloadUrl)
  } catch (error) {
    console.error('[PDF Export] Download failed:', error)
    throw error
  }
}

/**
 * Export PDF with custom configuration (POST endpoint)
 * Provides more configuration options than the GET endpoint
 *
 * @param videoId - Video ID to export
 * @param config - Full configuration options
 */
export async function exportPDFWithConfig(
  videoId: string,
  config: PDFExportConfig
): Promise<void> {
  const url = `${API_BASE_URL}/api/export/pdf/${videoId}`
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to generate PDF')
    }
    
    // Get the blob and create download link
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    
    // Create temporary link and trigger download
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `badminton_analysis_${videoId.slice(0, 8)}.pdf`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // Clean up object URL
    window.URL.revokeObjectURL(downloadUrl)
  } catch (error) {
    console.error('[PDF Export] Download failed:', error)
    throw error
  }
}

/**
 * Get preview of PDF export data
 * Shows what will be included in the PDF before generating
 *
 * @param videoId - Video ID to preview
 */
export async function getPDFExportPreview(videoId: string): Promise<PDFExportPreview> {
  const url = `${API_BASE_URL}/api/export/pdf/preview/${videoId}`
  
  const response = await fetch(url)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get PDF preview')
  }
  
  return response.json()
}

/**
 * Export PDF with frontend data
 * This sends the analysis result data directly from the frontend to ensure
 * the PDF contains exactly what the user sees on screen.
 *
 * @param videoId - Video ID (used to find video file for heatmap)
 * @param data - Full configuration with frontend data
 */
export async function exportPDFWithFrontendData(
  videoId: string,
  data: PDFExportWithDataConfig
): Promise<void> {
  const url = `${API_BASE_URL}/api/export/pdf/${videoId}/with-data`
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to generate PDF')
    }
    
    // Get the blob and create download link
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    
    // Create temporary link and trigger download
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `badminton_analysis_${videoId.slice(0, 8)}.pdf`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // Clean up object URL
    window.URL.revokeObjectURL(downloadUrl)
    
    console.log('[PDF Export] Successfully exported PDF with frontend data')
  } catch (error) {
    console.error('[PDF Export] Export with frontend data failed:', error)
    throw error
  }
}

// ============================================================================
// Zone Analytics API
// ============================================================================

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

/**
 * Get recalculated player zone analytics
 * This endpoint recalculates zone coverage using manual court keypoints
 * if they have been set, even after video processing is complete.
 *
 * @param videoId - Video ID to get zone analytics for
 * @param forceRefresh - If true, bypasses cache and fetches fresh data
 */
export async function getRecalculatedZoneAnalytics(
  videoId: string,
  forceRefresh: boolean = false
): Promise<RecalculatedZoneAnalyticsResponse> {
  // Check cache first (unless force refresh)
  const cacheKey = `zone-analytics:${videoId}`
  if (!forceRefresh) {
    const cached = zoneAnalyticsCache.get(cacheKey)
    if (cached) {
      return cached
    }
  }

  const url = `${API_BASE_URL}/api/player-zone-analytics/${videoId}/recalculate`
  const response = await fetch(url)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get zone analytics')
  }

  const data: RecalculatedZoneAnalyticsResponse = await response.json()
  
  // Cache the result
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
