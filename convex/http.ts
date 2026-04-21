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

// Manual court keypoints are now set/read via the Convex mutation
// `api.videos.setManualCourtKeypoints` and query `getManualCourtKeypoints`
// directly from the client — the previous HTTP wrappers that just
// validated payload shapes and forwarded to those same mutation/query
// have been removed. Validation is handled by Convex's argument validators
// on the mutation itself.

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
// Research: Badminton players rarely exceed 25 km/h even in explosive movements.
// Usain Bolt's peak is 44.7 km/h — 50 km/h is physically impossible in badminton.
// UNIFIED with backend modal_convex_processor.py (MAX_VALID_SPEED_MPS = 7.0 = 25 km/h)
const MAX_REALISTIC_SPEED_KMH = 25.0

// Maximum realistic movement per frame (pixels at 30fps)
// At 7 m/s max and typical court calibration (~0.008 m/px), that's ~29px/frame.
// Allow 80px to account for calibration variance — matches backend MAX_PX_PER_FRAME.
const MAX_FRAME_JUMP_PIXELS = 80

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

// Standard badminton court dimensions (meters)
const COURT_LENGTH = 13.4
const COURT_WIDTH = 6.1
const SERVICE_LINE = 1.98 // Distance from net to short service line

// Standard court keypoint positions in meters (matches 12-point calibration system)
// Order: TL, TR, BR, BL, NL, NR, SNL, SNR, SFL, SFR, CTN, CTF
const COURT_KEYPOINT_POSITIONS: number[][] = [
  [0, 0],                                             // 0: TL
  [COURT_WIDTH, 0],                                   // 1: TR
  [COURT_WIDTH, COURT_LENGTH],                        // 2: BR
  [0, COURT_LENGTH],                                  // 3: BL
  [0, COURT_LENGTH / 2],                              // 4: NL - Net left
  [COURT_WIDTH, COURT_LENGTH / 2],                    // 5: NR - Net right
  [0, COURT_LENGTH / 2 - SERVICE_LINE],               // 6: SNL - Service near left
  [COURT_WIDTH, COURT_LENGTH / 2 - SERVICE_LINE],     // 7: SNR - Service near right
  [0, COURT_LENGTH / 2 + SERVICE_LINE],               // 8: SFL - Service far left
  [COURT_WIDTH, COURT_LENGTH / 2 + SERVICE_LINE],     // 9: SFR - Service far right
  [COURT_WIDTH / 2, COURT_LENGTH / 2 - SERVICE_LINE], // 10: CTN - Center near
  [COURT_WIDTH / 2, COURT_LENGTH / 2 + SERVICE_LINE], // 11: CTF - Center far
]

// =============================================================================
// Homography computation with Hartley normalization (supports 4-12 point DLT)
// =============================================================================

/**
 * Normalize a set of 2D points for numerical stability (Hartley normalization).
 * Translates centroid to origin and scales so average distance = sqrt(2).
 */
function normalizePoints(points: number[][]): { normalized: number[][]; T: number[][] } {
  let sumX = 0, sumY = 0, count = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      sumX += p[0]!
      sumY += p[1]!
      count++
    }
  }
  if (count === 0) return { normalized: points, T: [[1,0,0],[0,1,0],[0,0,1]] }

  const meanX = sumX / count
  const meanY = sumY / count

  let sumDist = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      const dx = p[0]! - meanX
      const dy = p[1]! - meanY
      sumDist += Math.sqrt(dx * dx + dy * dy)
    }
  }
  const avgDist = sumDist / count
  const scale = avgDist > 0 ? Math.sqrt(2) / avgDist : 1

  const normalized: number[][] = points.map(p =>
    (p && p.length >= 2) ? [(p[0]! - meanX) * scale, (p[1]! - meanY) * scale] : [0, 0]
  )

  const T: number[][] = [
    [scale, 0, -scale * meanX],
    [0, scale, -scale * meanY],
    [0, 0, 1]
  ]

  return { normalized, T }
}

/** Invert a 3x3 matrix */
function invertMatrix3x3(M: number[][]): number[][] | null {
  const a = M[0]![0]!, b = M[0]![1]!, c = M[0]![2]!
  const d = M[1]![0]!, e = M[1]![1]!, f = M[1]![2]!
  const g = M[2]![0]!, h = M[2]![1]!, i = M[2]![2]!
  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
  if (Math.abs(det) < 1e-12) return null
  const inv = 1.0 / det
  return [
    [(e*i - f*h)*inv, (c*h - b*i)*inv, (b*f - c*e)*inv],
    [(f*g - d*i)*inv, (a*i - c*g)*inv, (c*d - a*f)*inv],
    [(d*h - e*g)*inv, (b*g - a*h)*inv, (a*e - b*d)*inv]
  ]
}

/** Multiply two 3x3 matrices */
function multiplyMatrix3x3(A: number[][], B: number[][]): number[][] {
  const R: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0
      for (let k = 0; k < 3; k++) sum += A[i]![k]! * B[k]![j]!
      R[i]![j] = sum
    }
  }
  return R
}

/** Solve linear system Ax = b via Gaussian elimination with partial pivoting */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length
  const aug: number[][] = A.map((row, i) => [...row, b[i]!])

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row]![col]!) > Math.abs(aug[maxRow]![col]!)) maxRow = row
    }
    if (Math.abs(aug[maxRow]![col]!) < 1e-10) return null
    ;[aug[col], aug[maxRow]] = [aug[maxRow]!, aug[col]!]

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row]![col]! / aug[col]![col]!
      for (let j = col; j <= n; j++) aug[row]![j]! -= factor * aug[col]![j]!
    }
  }

  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = aug[i]![n]!
    for (let j = i + 1; j < n; j++) x[i]! -= aug[i]![j]! * x[j]!
    x[i]! /= aug[i]![i]!
  }
  return x
}

/** Solve overdetermined system via normal equations: x = (A^T A)^-1 A^T b */
function solveLeastSquares(A: number[][], b: number[]): number[] | null {
  const m = A.length, n = A[0]?.length ?? 0
  if (n === 0) return null

  const AtA: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0
      for (let k = 0; k < m; k++) sum += A[k]![i]! * A[k]![j]!
      AtA[i]![j] = sum
    }
  }

  const Atb = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let sum = 0
    for (let k = 0; k < m; k++) sum += A[k]![i]! * b[k]!
    Atb[i] = sum
  }

  return solveLinearSystem(AtA, Atb)
}

/**
 * Compute a 3x3 homography matrix using DLT with Hartley normalization.
 * Supports 4 points (exact) or more (least-squares for better accuracy).
 * Returns a 9-element flat array [h0..h8] representing the 3x3 matrix.
 */
function computeHomography(
  srcPts: number[][],
  dstPts: number[][]
): number[] | null {
  const count = Math.min(srcPts.length, dstPts.length)
  if (count < 4) return null

  // Filter valid point pairs
  const validSrc: number[][] = []
  const validDst: number[][] = []
  for (let i = 0; i < count; i++) {
    if (srcPts[i]?.length! >= 2 && dstPts[i]?.length! >= 2) {
      validSrc.push(srcPts[i]!)
      validDst.push(dstPts[i]!)
    }
  }
  if (validSrc.length < 4) return null

  // Normalize coordinates for numerical stability
  const { normalized: normSrc, T: T_src } = normalizePoints(validSrc)
  const { normalized: normDst, T: T_dst } = normalizePoints(validDst)

  // Build DLT matrix equation
  const A: number[][] = []
  const b: number[] = []
  for (let i = 0; i < normSrc.length; i++) {
    const sx = normSrc[i]![0]!, sy = normSrc[i]![1]!
    const dx = normDst[i]![0]!, dy = normDst[i]![1]!
    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    b.push(dx)
    b.push(dy)
  }
  if (A.length < 8) return null

  // Solve: exact for 4 points, least-squares for overdetermined
  const h = A.length === 8 ? solveLinearSystem(A, b) : solveLeastSquares(A, b)
  if (!h) return null

  // Construct normalized homography and denormalize: H = T_dst^-1 * H_norm * T_src
  const H_norm: number[][] = [
    [h[0]!, h[1]!, h[2]!],
    [h[3]!, h[4]!, h[5]!],
    [h[6]!, h[7]!, 1]
  ]
  const T_dst_inv = invertMatrix3x3(T_dst)
  if (!T_dst_inv) return null

  const H = multiplyMatrix3x3(T_dst_inv, multiplyMatrix3x3(H_norm, T_src))

  // Flatten to 9-element array
  return [H[0]![0]!, H[0]![1]!, H[0]![2]!, H[1]![0]!, H[1]![1]!, H[1]![2]!, H[2]![0]!, H[2]![1]!, H[2]![2]!]
}

/**
 * Apply a 3x3 homography to transform a pixel point to court-meter coordinates.
 * H is a 9-element flat array [h0..h8].
 */
function applyHomography(H: number[], px: number, py: number): [number, number] | null {
  const w = H[6]! * px + H[7]! * py + H[8]!
  if (Math.abs(w) < 1e-12) return null
  const cx = (H[0]! * px + H[1]! * py + H[2]!) / w
  const cy = (H[3]! * px + H[4]! * py + H[5]!) / w
  return [cx, cy]
}

/**
 * Build a homography matrix from manual court keypoints.
 * Uses ALL available keypoints (up to 12) for maximum accuracy via least-squares DLT.
 * Falls back gracefully to 4-point if optional keypoints are missing.
 *
 * Court coordinate system:
 *   Origin (0,0) = top-left corner
 *   X-axis = court width (0 to 6.1m)
 *   Y-axis = court length (0 to 13.4m)
 */
function buildHomographyFromKeypoints(
  keypoints: ManualCourtKeypoints
): number[] | null {
  // Required 4 corners
  const requiredPts = [
    keypoints.top_left,
    keypoints.top_right,
    keypoints.bottom_right,
    keypoints.bottom_left,
  ]
  for (const pt of requiredPts) {
    if (!pt || pt.length < 2) return null
  }

  // Build source (pixel) and destination (court meters) point arrays
  // using all available keypoints for better accuracy
  const srcPts: number[][] = [...requiredPts]
  const dstPts: number[][] = [
    COURT_KEYPOINT_POSITIONS[0]!, // TL
    COURT_KEYPOINT_POSITIONS[1]!, // TR
    COURT_KEYPOINT_POSITIONS[2]!, // BR
    COURT_KEYPOINT_POSITIONS[3]!, // BL
  ]

  // Optional keypoints - add each one that exists
  const optionalMappings: Array<{ field: keyof ManualCourtKeypoints; idx: number }> = [
    { field: "net_left", idx: 4 },
    { field: "net_right", idx: 5 },
    { field: "service_line_near_left", idx: 6 },
    { field: "service_line_near_right", idx: 7 },
    { field: "service_line_far_left", idx: 8 },
    { field: "service_line_far_right", idx: 9 },
    { field: "center_near", idx: 10 },
    { field: "center_far", idx: 11 },
  ]

  for (const { field, idx } of optionalMappings) {
    const pt = keypoints[field]
    if (pt && Array.isArray(pt) && pt.length >= 2) {
      srcPts.push(pt)
      dstPts.push(COURT_KEYPOINT_POSITIONS[idx]!)
    }
  }

  console.log(`[Speed] Building homography from ${srcPts.length} keypoints (4 required + ${srcPts.length - 4} optional)`)
  return computeHomography(srcPts, dstPts)
}

// Median filter window size
const SPEED_MEDIAN_WINDOW = 5

function calculateSpeedFromSkeleton(
  skeletonData: SkeletonFrameData[],
  fps: number,
  videoWidth: number,
  videoHeight: number,
  manualKeypoints?: ManualCourtKeypoints | null
): SpeedData {
  const hasManualKeypoints = !!manualKeypoints?.top_left && !!manualKeypoints?.bottom_left
  
  // --- Compute homography for perspective-correct distance calculation ---
  // When manual keypoints are available, use homography to transform pixel
  // coords to court-meter coords. This correctly handles perspective distortion
  // (near-camera players appear larger / move more pixels per meter).
  let homography: number[] | null = null
  let fallbackMpp = 0  // fallback meters-per-pixel when no keypoints
  
  if (hasManualKeypoints && manualKeypoints) {
    homography = buildHomographyFromKeypoints(manualKeypoints)
    if (homography) {
      console.log("[Speed] Using homography-based court calibration (perspective-correct)")
    } else {
      console.log("[Speed] Homography computation failed, falling back to simple mpp")
    }
  }
  
  if (!homography) {
    // Fallback: approximate based on typical court visibility in video
    // Use 80% of max dimension (matches modal_convex_processor.py)
    fallbackMpp = COURT_LENGTH / (Math.max(videoWidth, videoHeight) * 0.8)
    console.log(`[Speed] Using fallback meters-per-pixel: ${fallbackMpp.toFixed(6)}`)
  }
  
  // Helper: convert pixel position to court-meter position
  function pixelToMeters(px: number, py: number): [number, number] | null {
    if (homography) {
      const result = applyHomography(homography, px, py)
      if (!result) return null
      // Reject points far outside the court (bad detections)
      const [cx, cy] = result
      if (cx < -2 || cx > COURT_WIDTH + 2 || cy < -2 || cy > COURT_LENGTH + 2) {
        return null  // Outside court with 2m margin
      }
      return result
    }
    // Fallback: scale linearly (inaccurate but better than nothing)
    return [px * fallbackMpp, py * fallbackMpp]
  }
  
  // Maximum distance per frame in meters (at 30fps, 7 m/s max => 0.23m/frame)
  const MAX_DISTANCE_PER_FRAME_M = 0.25
  
  const frameData: SpeedData["frame_data"] = []
  const playerStats: Record<string, {
    speeds: number[]
    distance: number
    prevCourtPos: { cx: number; cy: number; px: number; py: number; frame: number } | null
    speedWindow: number[]  // Sliding window for median filtering
  }> = {}
  
  let filteredOutCount = 0
  let medianFilteredCount = 0
  
  for (const frame of skeletonData) {
    const frameSpeeds: Array<{ player_id: number; speed_kmh: number }> = []
    
    for (const player of frame.players) {
      let px: number | null = null
      let py: number | null = null
      
      if (player.center) {
        px = player.center.x
        py = player.center.y
      } else if (player.position) {
        px = player.position.x
        py = player.position.y
      }
      
      if (px === null || py === null) continue
      
      // Transform pixel position to court meters
      const courtPos = pixelToMeters(px, py)
      if (!courtPos) continue
      
      const [cx, cy] = courtPos
      
      const pid = player.player_id.toString()
      if (!playerStats[pid]) {
        playerStats[pid] = { speeds: [], distance: 0, prevCourtPos: null, speedWindow: [] }
      }
      
      let speed_kmh = 0
      let isValidMeasurement = true
      
      if (playerStats[pid].prevCourtPos) {
        const prev = playerStats[pid].prevCourtPos!
        
        // Calculate distance in COURT METERS (perspective-corrected)
        const dxM = cx - prev.cx
        const dyM = cy - prev.cy
        const distM = Math.sqrt(dxM * dxM + dyM * dyM)
        
        // Also check pixel distance for tracking jump detection
        const dxPx = px - prev.px
        const dyPx = py - prev.py
        const distPx = Math.sqrt(dxPx * dxPx + dyPx * dyPx)
        
        const framesElapsed = Math.max(1, frame.frame - prev.frame)
        const dt = framesElapsed / fps
        const distPerFrame = distM / framesElapsed
        const pxPerFrame = distPx / framesElapsed
        
        if (dt > 0) {
          const speedMs = distM / dt
          speed_kmh = speedMs * 3.6
          
          // STEP 1: Pixel-based jump detection (catches tracking ID swaps)
          if (pxPerFrame > MAX_FRAME_JUMP_PIXELS) {
            filteredOutCount++
            speed_kmh = 0
            isValidMeasurement = false
          }
          // STEP 2: Distance-per-frame check in meters (catches position jumps)
          else if (distPerFrame > MAX_DISTANCE_PER_FRAME_M) {
            filteredOutCount++
            speed_kmh = 0
            isValidMeasurement = false
          }
          // STEP 3: Hard speed cap (physiological limit)
          else if (speed_kmh > MAX_REALISTIC_SPEED_KMH) {
            filteredOutCount++
            speed_kmh = 0
            isValidMeasurement = false
          }
          // STEP 4: Median filter — reject outlier spikes
          // Use 3x threshold (not 2x) to allow legitimate quick accelerations
          // common in badminton (e.g., standing → lunge)
          else {
            const window = playerStats[pid].speedWindow
            if (window.length >= 3) {
              const sortedWindow = [...window].sort((a, b) => a - b)
              const medianSpeed = sortedWindow[Math.floor(sortedWindow.length / 2)]!
              if (medianSpeed > 2.0 && speed_kmh > medianSpeed * 3.0) {
                medianFilteredCount++
                speed_kmh = 0
                isValidMeasurement = false
              }
            }
          }
          
          if (isValidMeasurement && speed_kmh > 0) {
            // Add to median filter window
            playerStats[pid].speedWindow.push(speed_kmh)
            if (playerStats[pid].speedWindow.length > SPEED_MEDIAN_WINDOW) {
              playerStats[pid].speedWindow.shift()
            }
            // Valid speed - add to stats
            playerStats[pid].speeds.push(speed_kmh)
            playerStats[pid].distance += distM
          }
        }
      }
      
      // Only update tracking position if measurement was valid (avoid propagating jumps)
      if (isValidMeasurement) {
        playerStats[pid].prevCourtPos = { cx, cy, px, py, frame: frame.frame }
      }
      frameSpeeds.push({ player_id: player.player_id, speed_kmh })
    }
    
    if (frameSpeeds.length > 0) {
      frameData.push({ frame: frame.frame, players: frameSpeeds })
    }
  }
  
  if (filteredOutCount > 0) {
    console.log(`[Speed] Filtered out ${filteredOutCount} unrealistic speed values (jump/cap filter)`)
  }
  if (medianFilteredCount > 0) {
    console.log(`[Speed] Filtered out ${medianFilteredCount} speed values by median filter`)
  }
  
  // Calculate statistics per player
  // Per-frame filtering already handles outliers; only apply hard cap here
  const statistics: Record<string, SpeedStatistics> = {}
  for (const [pid, stats] of Object.entries(playerStats)) {
    const filtered = stats.speeds.filter(s => s > 0 && s <= MAX_REALISTIC_SPEED_KMH)

    const avgSpeed = filtered.length > 0
      ? filtered.reduce((a, b) => a + b, 0) / filtered.length
      : 0
    const maxSpeed = filtered.length > 0
      ? Math.max(...filtered)
      : 0

    statistics[pid] = {
      avg: { speed_kmh: avgSpeed },
      max: { speed_kmh: Math.min(maxSpeed, MAX_REALISTIC_SPEED_KMH) },
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
      
      // Build config, ensuring frontend player data is included so the PDF
      // shows the same calibrated speed values as the dashboard.
      const pdfConfig = config || {
        title: `Badminton Analysis - ${video.filename || videoId}`,
        include_heatmap: true,
        heatmap_colormap: "turbo",
        heatmap_alpha: 0.6,
      }
      
      console.log("[PDF EXPORT] Calling Modal PDF service...")
      console.log("[PDF EXPORT] Frontend player data included:", !!pdfConfig.players)
      const modalResponse = await fetch(modalPdfEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          videoUrl,
          resultsUrl,
          config: pdfConfig,
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
