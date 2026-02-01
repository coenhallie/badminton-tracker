import { httpRouter } from "convex/server"
import { httpAction } from "./_generated/server"
import { api } from "./_generated/api"
import type { Id } from "./_generated/dataModel"

const http = httpRouter()

/**
 * HTTP endpoint for Modal to update processing status
 * POST /updateStatus
 */
http.route({
  path: "/updateStatus",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      
      const { videoId, status, progress, currentFrame, totalFrames, error } = body
      
      if (!videoId) {
        return new Response(JSON.stringify({ error: "Missing videoId" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        })
      }
      
      // Filter out null values - Convex v.optional() doesn't accept null
      await ctx.runMutation(api.videos.updateStatus, {
        videoId: videoId as Id<"videos">,
        status,
        ...(progress != null ? { progress } : {}),
        ...(currentFrame != null ? { currentFrame } : {}),
        ...(totalFrames != null ? { totalFrames } : {}),
        ...(error != null ? { error } : {}),
      })
      
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    } catch (error) {
      console.error("updateStatus error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      )
    }
  }),
})

/**
 * HTTP endpoint for Modal to add processing logs
 * POST /addLog
 */
http.route({
  path: "/addLog",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      
      const { videoId, message, level, category } = body
      
      if (!videoId || !message) {
        return new Response(JSON.stringify({ error: "Missing required fields" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        })
      }
      
      await ctx.runMutation(api.videos.addLog, {
        videoId: videoId as Id<"videos">,
        message,
        level: level || "info",
        category: category || "processing",
      })
      
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    } catch (error) {
      console.error("addLog error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      )
    }
  }),
})

/**
 * HTTP endpoint for Modal to submit final results
 * POST /updateResults
 *
 * Since results can be large (>1MB), Modal should:
 * 1. Upload the full results JSON to Convex storage (using /generateUploadUrl)
 * 2. Send only metadata + storage ID here
 */
http.route({
  path: "/updateResults",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      
      const {
        videoId,
        resultsMeta,
        resultsStorageId,
        processedVideoStorageId,
        skeletonDataStorageId
      } = body
      
      if (!videoId) {
        return new Response(JSON.stringify({ error: "Missing videoId" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        })
      }
      
      await ctx.runMutation(api.videos.updateResults, {
        videoId: videoId as Id<"videos">,
        ...(resultsMeta != null ? { resultsMeta } : {}),
        ...(resultsStorageId != null ? { resultsStorageId: resultsStorageId as Id<"_storage"> } : {}),
        ...(processedVideoStorageId != null ? { processedVideoStorageId: processedVideoStorageId as Id<"_storage"> } : {}),
        ...(skeletonDataStorageId != null ? { skeletonDataStorageId: skeletonDataStorageId as Id<"_storage"> } : {}),
      })
      
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    } catch (error) {
      console.error("updateResults error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      )
    }
  }),
})

/**
 * HTTP endpoint for Modal to generate an upload URL (for processed videos)
 * POST /generateUploadUrl
 */
http.route({
  path: "/generateUploadUrl",
  method: "POST",
  handler: httpAction(async (ctx, _request) => {
    try {
      const uploadUrl = await ctx.storage.generateUploadUrl()
      
      return new Response(JSON.stringify({ uploadUrl }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    } catch (error) {
      console.error("generateUploadUrl error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      )
    }
  }),
})

/**
 * Serve video files from storage
 * GET /video?storageId=xxx
 */
http.route({
  path: "/video",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const { searchParams } = new URL(request.url)
    const storageId = searchParams.get("storageId")
    
    if (!storageId) {
      return new Response("Missing storageId", { status: 400 })
    }
    
    const blob = await ctx.storage.get(storageId as Id<"_storage">)
    if (blob === null) {
      return new Response("Video not found", { status: 404 })
    }
    
    return new Response(blob, {
      headers: {
        "Content-Type": "video/mp4",
        "Cache-Control": "public, max-age=3600",
      },
    })
  }),
})

/**
 * HTTP endpoint to set manual court keypoints
 * POST /api/court-keypoints/manual/set
 */
http.route({
  path: "/api/court-keypoints/manual/set",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      const { videoId, keypoints } = body
      
      if (!videoId || !keypoints) {
        return new Response(JSON.stringify({
          error: "Missing videoId or keypoints"
        }), {
          status: 400,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Validate keypoints has required fields
      if (!keypoints.top_left || !keypoints.top_right ||
          !keypoints.bottom_right || !keypoints.bottom_left) {
        return new Response(JSON.stringify({
          error: "Missing required keypoints (top_left, top_right, bottom_right, bottom_left)"
        }), {
          status: 400,
          headers: corsHeaders("application/json"),
        })
      }
      
      await ctx.runMutation(api.videos.setManualCourtKeypoints, {
        videoId: videoId as Id<"videos">,
        keypoints,
      })
      
      return new Response(JSON.stringify({
        status: "success",
        message: "Manual court keypoints saved",
        keypoints,
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("setManualCourtKeypoints error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

/**
 * HTTP endpoint to get manual court keypoints status
 * GET /api/court-keypoints/manual/status?videoId=xxx
 */
http.route({
  path: "/api/court-keypoints/manual/status",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    try {
      const { searchParams } = new URL(request.url)
      const videoId = searchParams.get("videoId")
      
      if (!videoId) {
        return new Response(JSON.stringify({
          error: "Missing videoId"
        }), {
          status: 400,
          headers: corsHeaders("application/json"),
        })
      }
      
      const result = await ctx.runQuery(api.videos.getManualCourtKeypoints, {
        videoId: videoId as Id<"videos">,
      })
      
      return new Response(JSON.stringify(result || {
        has_manual_keypoints: false,
        keypoints: null,
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("getManualCourtKeypoints error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

// Handle OPTIONS preflight for manual keypoints endpoints
http.route({
  path: "/api/court-keypoints/manual/set",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: corsHeaders("application/json"),
    })
  }),
})

http.route({
  path: "/api/court-keypoints/manual/status",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: corsHeaders("application/json"),
    })
  }),
})

/**
 * HTTP endpoint for heatmap generation
 * GET /api/heatmap?videoId=xxx
 *
 * Generates a heatmap from skeleton_data stored in Convex
 */
http.route({
  path: "/api/heatmap",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const { searchParams } = new URL(request.url)
    const videoId = searchParams.get("videoId")
    const width = parseInt(searchParams.get("width") || "64")
    const height = parseInt(searchParams.get("height") || "48")
    const playerId = searchParams.get("player_id")
    
    if (!videoId) {
      return new Response(JSON.stringify({ error: "Missing videoId" }), {
        status: 400,
        headers: corsHeaders("application/json"),
      })
    }
    
    try {
      // Get video with results URL
      const video = await ctx.runQuery(api.videos.getVideo, {
        videoId: videoId as Id<"videos">
      })
      
      if (!video || !video.resultsUrl) {
        return new Response(JSON.stringify({ error: "Video or results not found" }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Fetch results from storage
      const response = await fetch(video.resultsUrl)
      if (!response.ok) {
        throw new Error("Failed to fetch results")
      }
      const results = await response.json()
      
      const skeleton_data = results.skeleton_data || []
      const video_width = results.video_width || 1920
      const video_height = results.video_height || 1080
      
      // Generate heatmap from skeleton data
      const heatmap = generateHeatmapFromSkeleton(
        skeleton_data,
        video_width,
        video_height,
        width,
        height,
        playerId ? parseInt(playerId) : undefined
      )
      
      return new Response(JSON.stringify({
        video_id: videoId,
        width,
        height,
        colormap: "turbo",
        combined_heatmap: heatmap.combined,
        player_heatmaps: heatmap.byPlayer,
        total_frames: skeleton_data.length,
        video_width,
        video_height,
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("Heatmap generation error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

/**
 * HTTP endpoint for zone analytics
 * GET /api/zone-analytics?videoId=xxx
 *
 * Calculates zone coverage from skeleton_data
 */
http.route({
  path: "/api/zone-analytics",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const { searchParams } = new URL(request.url)
    const videoId = searchParams.get("videoId")
    
    if (!videoId) {
      return new Response(JSON.stringify({ error: "Missing videoId" }), {
        status: 400,
        headers: corsHeaders("application/json"),
      })
    }
    
    try {
      // Get video with results URL
      const video = await ctx.runQuery(api.videos.getVideo, {
        videoId: videoId as Id<"videos">
      })
      
      if (!video || !video.resultsUrl) {
        return new Response(JSON.stringify({ error: "Video or results not found" }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Fetch results from storage
      const response = await fetch(video.resultsUrl)
      if (!response.ok) {
        throw new Error("Failed to fetch results")
      }
      const results = await response.json()
      
      const skeleton_data = results.skeleton_data || []
      const video_width = results.video_width || 1920
      const video_height = results.video_height || 1080
      
      // Calculate zone analytics
      const zoneAnalytics = calculateZoneAnalytics(
        skeleton_data,
        video_width,
        video_height
      )
      
      return new Response(JSON.stringify({
        video_id: videoId,
        player_zone_analytics: zoneAnalytics,
        manual_keypoints_used: false,
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("Zone analytics error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

/**
 * HTTP endpoint for speed recalculation
 * GET /api/speed?videoId=xxx
 *
 * Uses manual court keypoints when available for accurate pixel-to-meter conversion.
 * Caps speeds to realistic values (max 50 km/h) to filter tracking errors.
 */
http.route({
  path: "/api/speed",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const { searchParams } = new URL(request.url)
    const videoId = searchParams.get("videoId")
    
    if (!videoId) {
      return new Response(JSON.stringify({ error: "Missing videoId" }), {
        status: 400,
        headers: corsHeaders("application/json"),
      })
    }
    
    try {
      // Get video with results URL and manual keypoints
      const video = await ctx.runQuery(api.videos.getVideo, {
        videoId: videoId as Id<"videos">
      })
      
      if (!video || !video.resultsUrl) {
        return new Response(JSON.stringify({ error: "Video or results not found" }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Fetch results from storage
      const response = await fetch(video.resultsUrl)
      if (!response.ok) {
        throw new Error("Failed to fetch results")
      }
      const results = await response.json()
      
      const skeleton_data = results.skeleton_data || []
      const fps = results.fps || 30
      const video_width = results.video_width || 1920
      const video_height = results.video_height || 1080
      
      // Get manual court keypoints for accurate pixel-to-meter conversion
      const manualKeypoints = video.manualCourtKeypoints || null
      const hasManualKeypoints = !!manualKeypoints?.top_left && !!manualKeypoints?.bottom_left
      
      // Calculate speed data with keypoints for calibration
      const speedData = calculateSpeedFromSkeleton(
        skeleton_data,
        fps,
        video_width,
        video_height,
        manualKeypoints
      )
      
      return new Response(JSON.stringify({
        video_id: videoId,
        speed_data: speedData,
        manual_keypoints_used: hasManualKeypoints,
        detection_source: "convex",
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("Speed calculation error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

// =============================================================================
// CORS headers helper
// =============================================================================
function corsHeaders(contentType: string) {
  return {
    "Content-Type": contentType,
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  }
}

// =============================================================================
// Heatmap generation from skeleton data
// =============================================================================
interface SkeletonPlayer {
  player_id: number
  center?: { x: number; y: number }
  position?: { x: number; y: number }
  keypoints?: Array<{ x: number | null; y: number | null; confidence: number }>
}

interface SkeletonFrameData {
  frame: number
  players: SkeletonPlayer[]
}

function generateHeatmapFromSkeleton(
  skeletonData: SkeletonFrameData[],
  videoWidth: number,
  videoHeight: number,
  heatmapWidth: number,
  heatmapHeight: number,
  playerId?: number
): { combined: number[][]; byPlayer: Record<string, number[][]> } {
  // Initialize heatmaps
  const combined: number[][] = Array(heatmapHeight).fill(null).map(() => Array(heatmapWidth).fill(0))
  const byPlayer: Record<string, number[][]> = {}
  
  // Gaussian blur radius in heatmap cells
  const radius = 2
  
  for (const frame of skeletonData) {
    for (const player of frame.players) {
      // Filter by player ID if specified
      if (playerId !== undefined && player.player_id !== playerId) continue
      
      // Get player position
      let x: number | null = null
      let y: number | null = null
      
      if (player.center) {
        x = player.center.x
        y = player.center.y
      } else if (player.position) {
        x = player.position.x
        y = player.position.y
      }
      
      if (x === null || y === null) continue
      
      // Convert to heatmap coordinates
      const hx = Math.floor((x / videoWidth) * heatmapWidth)
      const hy = Math.floor((y / videoHeight) * heatmapHeight)
      
      // Initialize player heatmap if needed
      const pid = player.player_id.toString()
      if (!byPlayer[pid]) {
        byPlayer[pid] = Array(heatmapHeight).fill(null).map(() => Array(heatmapWidth).fill(0))
      }
      
      // Add Gaussian blur around the point
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = hx + dx
          const py = hy + dy
          
          if (px >= 0 && px < heatmapWidth && py >= 0 && py < heatmapHeight) {
            const dist = Math.sqrt(dx * dx + dy * dy)
            const weight = Math.exp(-(dist * dist) / (2 * radius * radius))
            
            const combinedRow = combined[py]
            const playerData = byPlayer[pid]
            if (combinedRow && playerData) {
              combinedRow[px] = (combinedRow[px] ?? 0) + weight
              const playerRow = playerData[py]
              if (playerRow) {
                playerRow[px] = (playerRow[px] ?? 0) + weight
              }
            }
          }
        }
      }
    }
  }
  
  // Normalize to 0-255 range
  const flatCombined = combined.flat().filter((v): v is number => v !== undefined)
  const maxCombined = flatCombined.length > 0 ? Math.max(...flatCombined) : 0
  if (maxCombined > 0) {
    for (let y = 0; y < heatmapHeight; y++) {
      const row = combined[y]
      if (row) {
        for (let x = 0; x < heatmapWidth; x++) {
          row[x] = Math.round(((row[x] ?? 0) / maxCombined) * 255)
        }
      }
    }
  }
  
  for (const pid of Object.keys(byPlayer)) {
    const playerData = byPlayer[pid]
    if (!playerData) continue
    const flatPlayer = playerData.flat().filter((v): v is number => v !== undefined)
    const maxPlayer = flatPlayer.length > 0 ? Math.max(...flatPlayer) : 0
    if (maxPlayer > 0) {
      for (let y = 0; y < heatmapHeight; y++) {
        const row = playerData[y]
        if (row) {
          for (let x = 0; x < heatmapWidth; x++) {
            row[x] = Math.round(((row[x] ?? 0) / maxPlayer) * 255)
          }
        }
      }
    }
  }
  
  return { combined, byPlayer }
}

// =============================================================================
// Zone analytics calculation
// =============================================================================
interface ZoneCoverage {
  front: number
  mid: number
  back: number
  left: number
  center: number
  right: number
}

interface PlayerZoneData {
  zone_coverage: ZoneCoverage
  total_positions: number
}

function calculateZoneAnalytics(
  skeletonData: SkeletonFrameData[],
  videoWidth: number,
  videoHeight: number
): Record<string, PlayerZoneData> {
  const playerZones: Record<string, { zones: ZoneCoverage; total: number }> = {}
  
  // Define zones as percentages of court dimensions
  // Front/Mid/Back zones (vertical thirds)
  // Left/Center/Right zones (horizontal thirds)
  
  for (const frame of skeletonData) {
    for (const player of frame.players) {
      let x: number | null = null
      let y: number | null = null
      
      if (player.center) {
        x = player.center.x
        y = player.center.y
      } else if (player.position) {
        x = player.position.x
        y = player.position.y
      }
      
      if (x === null || y === null) continue
      
      const pid = player.player_id.toString()
      if (!playerZones[pid]) {
        playerZones[pid] = {
          zones: { front: 0, mid: 0, back: 0, left: 0, center: 0, right: 0 },
          total: 0,
        }
      }
      
      // Normalize position
      const nx = x / videoWidth
      const ny = y / videoHeight
      
      // Determine vertical zone (front/mid/back)
      if (ny < 0.33) {
        playerZones[pid].zones.front++
      } else if (ny < 0.67) {
        playerZones[pid].zones.mid++
      } else {
        playerZones[pid].zones.back++
      }
      
      // Determine horizontal zone (left/center/right)
      if (nx < 0.33) {
        playerZones[pid].zones.left++
      } else if (nx < 0.67) {
        playerZones[pid].zones.center++
      } else {
        playerZones[pid].zones.right++
      }
      
      playerZones[pid].total++
    }
  }
  
  // Convert to percentages
  const result: Record<string, PlayerZoneData> = {}
  for (const [pid, data] of Object.entries(playerZones)) {
    const total = data.total || 1
    result[pid] = {
      zone_coverage: {
        front: (data.zones.front / total) * 100,
        mid: (data.zones.mid / total) * 100,
        back: (data.zones.back / total) * 100,
        left: (data.zones.left / total) * 100,
        center: (data.zones.center / total) * 100,
        right: (data.zones.right / total) * 100,
      },
      total_positions: data.total,
    }
  }
  
  return result
}

// =============================================================================
// Speed calculation from skeleton data
// =============================================================================

// Maximum realistic speed for a badminton player (km/h)
// Speeds above this are filtered as tracking errors
const MAX_REALISTIC_SPEED_KMH = 50.0

// Maximum realistic movement per frame (pixels at 30fps)
// ~0.5 meters per frame max -> at typical calibration, this is ~50-100px
const MAX_FRAME_JUMP_PIXELS = 200

interface SpeedStatistics {
  avg: { speed_kmh: number }
  max: { speed_kmh: number }
  total_distance_m: number
}

interface SpeedData {
  frame_data: Array<{ frame: number; players: Array<{ player_id: number; speed_kmh: number }> }>
  statistics: Record<string, SpeedStatistics>
  manual_keypoints_used?: boolean
}

// Manual court keypoints type (matches schema)
interface ManualCourtKeypoints {
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  center_near?: number[]
  center_far?: number[]
  net_left?: number[]
  net_right?: number[]
  service_line_near_left?: number[]
  service_line_near_right?: number[]
  service_line_far_left?: number[]
  service_line_far_right?: number[]
}

/**
 * Calculate meters per pixel from manual court keypoints.
 * Uses the known court dimensions (13.4m x 6.1m for doubles) and
 * the pixel distances between corners to compute accurate calibration.
 */
function calculateMetersPerPixel(
  keypoints: ManualCourtKeypoints,
  videoWidth: number,
  videoHeight: number
): number {
  // Badminton court dimensions (meters)
  const COURT_LENGTH = 13.4  // Full length
  const COURT_WIDTH = 6.1   // Doubles width
  
  // Get corner coordinates
  const topLeft = keypoints.top_left
  const topRight = keypoints.top_right
  const bottomLeft = keypoints.bottom_left
  const bottomRight = keypoints.bottom_right
  
  // Calculate pixel distances for horizontal edges (width of court)
  const topEdgePx = Math.sqrt(
    Math.pow(topRight[0] - topLeft[0], 2) +
    Math.pow(topRight[1] - topLeft[1], 2)
  )
  const bottomEdgePx = Math.sqrt(
    Math.pow(bottomRight[0] - bottomLeft[0], 2) +
    Math.pow(bottomRight[1] - bottomLeft[1], 2)
  )
  
  // Calculate pixel distances for vertical edges (length of court)
  const leftEdgePx = Math.sqrt(
    Math.pow(bottomLeft[0] - topLeft[0], 2) +
    Math.pow(bottomLeft[1] - topLeft[1], 2)
  )
  const rightEdgePx = Math.sqrt(
    Math.pow(bottomRight[0] - topRight[0], 2) +
    Math.pow(bottomRight[1] - topRight[1], 2)
  )
  
  // Average the edge lengths (accounts for perspective)
  const avgWidthPx = (topEdgePx + bottomEdgePx) / 2
  const avgLengthPx = (leftEdgePx + rightEdgePx) / 2
  
  // Calculate meters per pixel for both dimensions
  const mppWidth = COURT_WIDTH / avgWidthPx
  const mppLength = COURT_LENGTH / avgLengthPx
  
  // Use weighted average (length typically more visible)
  const metersPerPixel = (mppWidth + mppLength) / 2
  
  console.log(`[Speed] Calibration: avgWidth=${avgWidthPx.toFixed(1)}px, avgLength=${avgLengthPx.toFixed(1)}px, mpp=${metersPerPixel.toFixed(6)}`)
  
  return metersPerPixel
}

function calculateSpeedFromSkeleton(
  skeletonData: SkeletonFrameData[],
  fps: number,
  videoWidth: number,
  videoHeight: number,
  manualKeypoints?: ManualCourtKeypoints | null
): SpeedData {
  // Calculate meters per pixel - use manual keypoints if available
  let metersPerPixel: number
  const hasManualKeypoints = !!manualKeypoints?.top_left && !!manualKeypoints?.bottom_left
  
  if (hasManualKeypoints && manualKeypoints) {
    metersPerPixel = calculateMetersPerPixel(manualKeypoints, videoWidth, videoHeight)
  } else {
    // Fallback: approximate based on typical court visibility in video
    // Assume court length (13.4m) covers about 60% of the max dimension
    metersPerPixel = 13.4 / (Math.max(videoWidth, videoHeight) * 0.6)
  }
  
  const frameData: SpeedData["frame_data"] = []
  const playerStats: Record<string, {
    speeds: number[]
    distance: number
    prevPos: { x: number; y: number; frame: number } | null
  }> = {}
  
  let filteredOutCount = 0
  
  for (const frame of skeletonData) {
    const frameSpeeds: Array<{ player_id: number; speed_kmh: number }> = []
    
    for (const player of frame.players) {
      let x: number | null = null
      let y: number | null = null
      
      if (player.center) {
        x = player.center.x
        y = player.center.y
      } else if (player.position) {
        x = player.position.x
        y = player.position.y
      }
      
      if (x === null || y === null) continue
      
      const pid = player.player_id.toString()
      if (!playerStats[pid]) {
        playerStats[pid] = { speeds: [], distance: 0, prevPos: null }
      }
      
      let speed_kmh = 0
      if (playerStats[pid].prevPos) {
        const prev = playerStats[pid].prevPos!
        const dx = x - prev.x
        const dy = y - prev.y
        const distPx = Math.sqrt(dx * dx + dy * dy)
        const distM = distPx * metersPerPixel
        const dt = (frame.frame - prev.frame) / fps
        
        if (dt > 0) {
          const speedMs = distM / dt
          speed_kmh = speedMs * 3.6
          
          // Filter out unrealistic speeds (tracking errors, ID swaps)
          // Also check for unrealistic position jumps (potential tracking loss)
          if (speed_kmh > MAX_REALISTIC_SPEED_KMH || distPx > MAX_FRAME_JUMP_PIXELS * dt * fps) {
            // Treat as tracking error - zero speed for this frame
            filteredOutCount++
            speed_kmh = 0
          } else {
            // Valid speed - add to stats
            playerStats[pid].speeds.push(speed_kmh)
            playerStats[pid].distance += distM
          }
        }
      }
      
      playerStats[pid].prevPos = { x, y, frame: frame.frame }
      frameSpeeds.push({ player_id: player.player_id, speed_kmh })
    }
    
    if (frameSpeeds.length > 0) {
      frameData.push({ frame: frame.frame, players: frameSpeeds })
    }
  }
  
  if (filteredOutCount > 0) {
    console.log(`[Speed] Filtered out ${filteredOutCount} unrealistic speed values (>${MAX_REALISTIC_SPEED_KMH} km/h)`)
  }
  
  // Calculate statistics per player
  const statistics: Record<string, SpeedStatistics> = {}
  for (const [pid, stats] of Object.entries(playerStats)) {
    const speeds = stats.speeds.filter(s => s <= MAX_REALISTIC_SPEED_KMH)
    const avgSpeed = speeds.length > 0 ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0
    const maxSpeed = speeds.length > 0 ? Math.min(Math.max(...speeds), MAX_REALISTIC_SPEED_KMH) : 0
    
    statistics[pid] = {
      avg: { speed_kmh: avgSpeed },
      max: { speed_kmh: maxSpeed },
      total_distance_m: stats.distance,
    }
  }
  
  return {
    frame_data: frameData,
    statistics,
    manual_keypoints_used: hasManualKeypoints
  }
}

// =============================================================================
// Video URL endpoint - Get video URL from Convex storage
// =============================================================================
http.route({
  path: "/api/video-url",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const url = new URL(request.url)
    const videoId = url.searchParams.get("videoId")
    
    if (!videoId) {
      return new Response(JSON.stringify({ error: "Missing videoId parameter" }), {
        status: 400,
        headers: corsHeaders("application/json"),
      })
    }
    
    try {
      const video = await ctx.runQuery(api.videos.getVideo, { videoId: videoId as Id<"videos"> })
      
      if (!video) {
        return new Response(JSON.stringify({ error: "Video not found" }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Get the URL for the original video from storage
      const videoUrl = await ctx.storage.getUrl(video.storageId)
      
      if (!videoUrl) {
        return new Response(JSON.stringify({ error: "Video file not found in storage" }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      return new Response(JSON.stringify({ url: videoUrl }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("Video URL error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

// OPTIONS handler for CORS preflight for /api/video-url
http.route({
  path: "/api/video-url",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: corsHeaders("application/json"),
    })
  }),
})

/**
 * Health check endpoint
 * GET /health
 *
 * Returns backend status including Convex and Modal availability
 */
http.route({
  path: "/health",
  method: "GET",
  handler: httpAction(async (ctx) => {
    try {
      // Check if Modal endpoint is configured
      const modalEndpoint = process.env.MODAL_ENDPOINT_URL
      const modalConfigured = !!modalEndpoint
      
      // Get some database stats
      const videos = await ctx.runQuery(api.videos.listVideos, {})
      
      // Count videos by status
      const statusCounts: Record<string, number> = {}
      for (const video of videos || []) {
        const status = video.status || "unknown"
        statusCounts[status] = (statusCounts[status] || 0) + 1
      }
      
      // Check for any currently processing videos
      const processingCount = statusCounts["processing"] || 0
      
      return new Response(JSON.stringify({
        status: "healthy",
        timestamp: Date.now() / 1000,
        version: "1.0.0",
        backend: "convex",
        components: {
          convex: "connected",
          modal_inference: modalConfigured ? "configured" : "not configured",
          database: "connected"
        },
        stats: {
          total_videos: videos?.length || 0,
          processing: processingCount,
          videos_by_status: statusCounts
        }
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
    } catch (error) {
      console.error("Health check error:", error)
      return new Response(JSON.stringify({
        status: "unhealthy",
        timestamp: Date.now() / 1000,
        error: error instanceof Error ? error.message : "Unknown error",
        backend: "convex"
      }), {
        status: 503,
        headers: corsHeaders("application/json"),
      })
    }
  }),
})

// OPTIONS handler for CORS preflight for /health
http.route({
  path: "/health",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: corsHeaders("application/json"),
    })
  }),
})

/**
 * HTTP endpoint for PDF export
 * POST /api/export/pdf
 *
 * Calls Modal PDF export service to generate a professional analysis report.
 * Returns the PDF as a base64 encoded string.
 */
http.route({
  path: "/api/export/pdf",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    try {
      const body = await request.json()
      const { videoId, config } = body
      
      if (!videoId) {
        return new Response(JSON.stringify({
          error: "Missing videoId"
        }), {
          status: 400,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Get video from database
      const video = await ctx.runQuery(api.videos.getVideo, {
        videoId: videoId as Id<"videos">,
      })
      
      if (!video) {
        return new Response(JSON.stringify({
          error: "Video not found"
        }), {
          status: 404,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Get storage URLs for video and results
      const videoUrl = video.storageId ? await ctx.storage.getUrl(video.storageId) : null
      const resultsUrl = video.resultsStorageId ? await ctx.storage.getUrl(video.resultsStorageId) : null
      
      if (!videoUrl || !resultsUrl) {
        return new Response(JSON.stringify({
          error: "Video or results not available in storage"
        }), {
          status: 400,
          headers: corsHeaders("application/json"),
        })
      }
      
      // Call Modal PDF export service
      const modalPdfEndpoint = "https://coenhallie--badminton-tracker-pdf-export-generate-pdf.modal.run"
      
      console.log("[PDF EXPORT] Calling Modal PDF service...")
      const modalResponse = await fetch(modalPdfEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          videoUrl,
          resultsUrl,
          config: config || {
            title: `Badminton Analysis - ${video.filename || videoId}`,
            include_heatmap: true,
            heatmap_colormap: "turbo",
            heatmap_alpha: 0.6,
          },
        }),
      })
      
      if (!modalResponse.ok) {
        const errorText = await modalResponse.text()
        console.error("[PDF EXPORT] Modal service error:", errorText)
        return new Response(JSON.stringify({
          error: `PDF generation failed: ${errorText}`
        }), {
          status: 500,
          headers: corsHeaders("application/json"),
        })
      }
      
      const modalResult = await modalResponse.json()
      
      if (!modalResult.success) {
        return new Response(JSON.stringify({
          error: modalResult.error || "PDF generation failed"
        }), {
          status: 500,
          headers: corsHeaders("application/json"),
        })
      }
      
      console.log(`[PDF EXPORT] PDF generated successfully (${modalResult.size} bytes)`)
      
      // Return the base64 encoded PDF
      return new Response(JSON.stringify({
        success: true,
        pdfBase64: modalResult.pdfBase64,
        size: modalResult.size,
        filename: `badminton_analysis_${videoId.slice(0, 8)}.pdf`,
      }), {
        status: 200,
        headers: corsHeaders("application/json"),
      })
      
    } catch (error) {
      console.error("[PDF EXPORT] Error:", error)
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
        { status: 500, headers: corsHeaders("application/json") }
      )
    }
  }),
})

// OPTIONS handler for CORS preflight for /api/export/pdf
http.route({
  path: "/api/export/pdf",
  method: "OPTIONS",
  handler: httpAction(async () => {
    return new Response(null, {
      status: 204,
      headers: corsHeaders("application/json"),
    })
  }),
})

export default http
