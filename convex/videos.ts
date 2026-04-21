import { action, mutation, query, internalMutation } from "./_generated/server"
import { v } from "convex/values"
import type { Id } from "./_generated/dataModel"
import { api, internal } from "./_generated/api"

// =============================================================================
// QUERIES
// =============================================================================

/**
 * Get a video by ID (real-time query - will auto-update in UI)
 */
export const getVideo = query({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return null
    
    // Get the video URL for playback
    const videoUrl = video.storageId
      ? await ctx.storage.getUrl(video.storageId)
      : null
    
    // Get processed video URL if available
    const processedVideoUrl = video.processedVideoStorageId
      ? await ctx.storage.getUrl(video.processedVideoStorageId)
      : null
    
    // Get results JSON URL if available
    const resultsUrl = video.resultsStorageId
      ? await ctx.storage.getUrl(video.resultsStorageId)
      : null
    
    return {
      ...video,
      videoUrl,
      processedVideoUrl,
      resultsUrl,
    }
  },
})

/**
 * Get all videos (for listing)
 */
export const listVideos = query({
  args: {},
  handler: async (ctx) => {
    const videos = await ctx.db
      .query("videos")
      .withIndex("by_createdAt")
      .order("desc")
      .take(50)
    
    return videos
  },
})

/**
 * Get processing logs for a video (real-time)
 */
export const getProcessingLogs = query({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    return await ctx.db
      .query("processingLogs")
      .withIndex("by_videoId_timestamp", (q) => q.eq("videoId", videoId))
      .order("asc")
      .take(100)
  },
})

// =============================================================================
// MUTATIONS
// =============================================================================

/**
 * Generate a temporary URL for uploading a video
 */
export const generateUploadUrl = mutation({
  args: {},
  handler: async (ctx) => {
    return await ctx.storage.generateUploadUrl()
  },
})

/**
 * Create a video record after upload
 */
export const createVideo = mutation({
  args: {
    storageId: v.id("_storage"),
    filename: v.string(),
    size: v.number(),
    analysisMode: v.optional(v.union(v.literal("rally_only"), v.literal("full"))),
    cameraAngle: v.optional(v.union(v.literal("overhead"), v.literal("corner"))),
    trackerType: v.optional(v.union(v.literal("botsort"), v.literal("ocsort"))),
  },
  handler: async (ctx, args) => {
    const videoId = await ctx.db.insert("videos", {
      storageId: args.storageId,
      filename: args.filename,
      size: args.size,
      status: "uploaded",
      analysisMode: args.analysisMode ?? "full",
      cameraAngle: args.cameraAngle ?? "overhead",
      trackerType: args.trackerType ?? "botsort",
      createdAt: Date.now(),
    })
    
    // Add initial log
    await ctx.db.insert("processingLogs", {
      videoId,
      message: `Video "${args.filename}" uploaded successfully`,
      level: "success",
      category: "processing",
      timestamp: Date.now(),
    })
    
    return videoId
  },
})

/**
 * Update video processing status
 */
export const updateStatus = mutation({
  args: {
    videoId: v.id("videos"),
    status: v.union(
      v.literal("uploaded"),
      v.literal("processing"),
      v.literal("completed"),
      v.literal("failed")
    ),
    progress: v.optional(v.number()),
    currentFrame: v.optional(v.number()),
    totalFrames: v.optional(v.number()),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { videoId, ...updates } = args
    
    const patchData: Record<string, unknown> = { status: updates.status }
    if (updates.progress !== undefined) patchData.progress = updates.progress
    if (updates.currentFrame !== undefined) patchData.currentFrame = updates.currentFrame
    if (updates.totalFrames !== undefined) patchData.totalFrames = updates.totalFrames
    if (updates.error !== undefined) patchData.error = updates.error
    
    if (updates.status === "processing" && !patchData.processingStartedAt) {
      patchData.processingStartedAt = Date.now()
    }
    
    await ctx.db.patch(videoId, patchData)
  },
})

/**
 * Update with final results
 *
 * Due to Convex's 1MB document limit, large data like skeleton_data
 * should be stored as separate files (using resultsStorageId), while
 * only essential metadata is stored directly in the document.
 */
export const updateResults = mutation({
  args: {
    videoId: v.id("videos"),
    // Small metadata only (duration, fps, player counts, etc.)
    resultsMeta: v.optional(v.object({
      duration: v.number(),
      fps: v.number(),
      total_frames: v.number(),
      processed_frames: v.number(),
      player_count: v.optional(v.number()),
      has_court_detection: v.optional(v.boolean()),
      has_shuttle_analytics: v.optional(v.boolean()),
      has_rally_detection: v.optional(v.boolean()),
      rally_count: v.optional(v.number()),
      tracknet_used: v.optional(v.boolean()),
    })),
    // Storage ID for the full results JSON file
    resultsStorageId: v.optional(v.id("_storage")),
    processedVideoStorageId: v.optional(v.id("_storage")),
    skeletonDataStorageId: v.optional(v.id("_storage")),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.videoId, {
      status: "completed",
      progress: 100,
      resultsMeta: args.resultsMeta,
      resultsStorageId: args.resultsStorageId,
      processedVideoStorageId: args.processedVideoStorageId,
      skeletonDataStorageId: args.skeletonDataStorageId,
      completedAt: Date.now(),
    })
    
    // Add completion log
    await ctx.db.insert("processingLogs", {
      videoId: args.videoId,
      message: "Analysis complete! Results ready.",
      level: "success",
      category: "processing",
      timestamp: Date.now(),
    })
  },
})

/**
 * Add a processing log entry
 */
export const addLog = mutation({
  args: {
    videoId: v.id("videos"),
    message: v.string(),
    level: v.union(
      v.literal("info"),
      v.literal("success"),
      v.literal("warning"),
      v.literal("error"),
      v.literal("debug")
    ),
    category: v.union(
      v.literal("processing"),
      v.literal("detection"),
      v.literal("model"),
      v.literal("court"),
      v.literal("modal")
    ),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("processingLogs", {
      videoId: args.videoId,
      message: args.message,
      level: args.level,
      category: args.category,
      timestamp: Date.now(),
    })
  },
})

/**
 * Delete a BOUNDED batch of log rows for a video. Used by deleteAllVideos
 * to avoid tripping Convex's 4096-read-per-mutation cap on videos with
 * thousands of accumulated logs. Returns how many rows were deleted so
 * the caller can loop until 0.
 */
export const deleteVideoLogsBatch = mutation({
  args: { videoId: v.id("videos"), limit: v.number() },
  handler: async (ctx, { videoId, limit }) => {
    const logs = await ctx.db
      .query("processingLogs")
      .withIndex("by_videoId", (q) => q.eq("videoId", videoId))
      .take(limit)
    for (const log of logs) await ctx.db.delete(log._id)
    return logs.length
  },
})

/**
 * Delete a video's storage blobs + the record itself. ASSUMES logs are
 * already drained by deleteVideoLogsBatch (bulk path) — for the per-video
 * UI delete, this still works when log count is small (well under 4096).
 *
 * Bug fix: previously missed resultsStorageId — the JSON file that holds
 * skeleton + shuttle + rally output. Without this, deleting a video
 * orphaned its results blob in Convex storage.
 */
export const deleteVideo = mutation({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return

    // Delete every storage blob this record references. Each field is
    // independently optional, and a blob may already be gone from a
    // prior partial sweep — tolerate per-blob failures so a single stale
    // id can't prevent the record itself from being deleted.
    const blobIds = [
      video.storageId,
      video.resultsStorageId,
      video.processedVideoStorageId,
      video.skeletonDataStorageId,
      video.playerLabels?.player_0_thumbnail,
      video.playerLabels?.player_1_thumbnail,
    ].filter((id): id is Id<"_storage"> => id != null)

    await Promise.allSettled(blobIds.map(id => ctx.storage.delete(id)))

    // Drain logs. For normal per-video deletes log count is small; when
    // the bulk deleteAllVideos action is running it has already drained
    // logs via deleteVideoLogsBatch, so .take(1000) here is a safety net.
    const logs = await ctx.db
      .query("processingLogs")
      .withIndex("by_videoId", (q) => q.eq("videoId", videoId))
      .take(1000)

    await Promise.all(logs.map(log => ctx.db.delete(log._id)))

    // Delete video record
    await ctx.db.delete(videoId)
  },
})

/**
 * Internal query: list every video's ID so the deleteAllVideos action can
 * iterate without blowing past the per-mutation read cap.
 */
export const listAllVideoIds = query({
  args: {},
  handler: async (ctx) => {
    const videos = await ctx.db.query("videos").collect()
    return videos.map(v => v._id)
  },
})

/**
 * Delete EVERY video + its storage blobs + its logs.
 *
 * Destructive, irreversible. Runs as an action so each per-video delete
 * executes in its own mutation transaction — Convex caps reads at 4096
 * per function call, and a single video can have thousands of log rows,
 * so a single-mutation sweep is not viable at scale.
 */
export const deleteAllVideos = action({
  args: {},
  handler: async (ctx): Promise<{ deleted: number; failed: number }> => {
    const ids: Id<"videos">[] = await ctx.runQuery(api.videos.listAllVideoIds, {})
    let deleted = 0
    let failed = 0
    const LOG_BATCH = 1000
    for (const id of ids) {
      try {
        // Drain logs in batches first — per-video log count can exceed
        // Convex's 4096-read mutation cap.
        for (;;) {
          const n: number = await ctx.runMutation(
            api.videos.deleteVideoLogsBatch,
            { videoId: id, limit: LOG_BATCH },
          )
          if (n < LOG_BATCH) break
        }
        await ctx.runMutation(api.videos.deleteVideo, { videoId: id })
        deleted++
      } catch (err) {
        console.error(`[deleteAllVideos] failed to delete ${id}:`, err)
        failed++
      }
    }
    return { deleted, failed }
  },
})

/**
 * Set manual court keypoints for a video
 * Used for spatial calibration and homography calculation
 */
export const setManualCourtKeypoints = mutation({
  args: {
    videoId: v.id("videos"),
    keypoints: v.object({
      // 4 outer corners (required)
      top_left: v.array(v.number()),
      top_right: v.array(v.number()),
      bottom_right: v.array(v.number()),
      bottom_left: v.array(v.number()),
      // Net intersections (optional)
      net_left: v.optional(v.array(v.number())),
      net_right: v.optional(v.array(v.number())),
      // Service line corners (optional)
      service_line_near_left: v.optional(v.array(v.number())),
      service_line_near_right: v.optional(v.array(v.number())),
      service_line_far_left: v.optional(v.array(v.number())),
      service_line_far_right: v.optional(v.array(v.number())),
      // Center line endpoints (optional) - for precise homography
      center_near: v.optional(v.array(v.number())),
      center_far: v.optional(v.array(v.number())),
    }),
  },
  handler: async (ctx, args) => {
    const video = await ctx.db.get(args.videoId)
    if (!video) {
      throw new Error("Video not found")
    }
    
    await ctx.db.patch(args.videoId, {
      manualCourtKeypoints: args.keypoints,
    })
    
    // Add log entry
    await ctx.db.insert("processingLogs", {
      videoId: args.videoId,
      message: "Manual court keypoints saved",
      level: "info",
      category: "court",
      timestamp: Date.now(),
    })
    
    return { success: true, keypoints: args.keypoints }
  },
})

/**
 * Get manual court keypoints for a video
 */
export const getManualCourtKeypoints = query({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return null

    return {
      has_manual_keypoints: !!video.manualCourtKeypoints,
      keypoints: video.manualCourtKeypoints || null,
    }
  },
})

export const getPlayerLabels = query({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return null
    return video.playerLabels ?? null
  },
})

/**
 * Resolve a Convex storage id to a signed, time-limited URL suitable
 * for <img src>. Used by the Player Identity panel to render player
 * thumbnail blobs stored at `playerLabels.player_{0,1}_thumbnail`.
 */
export const getStorageUrl = query({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, { storageId }) => {
    return await ctx.storage.getUrl(storageId)
  },
})

export const updatePlayerLabels = mutation({
  args: {
    videoId: v.id("videos"),
    swapped: v.optional(v.boolean()),
    player_0_name: v.optional(v.string()),
    player_1_name: v.optional(v.string()),
  },
  handler: async (ctx, { videoId, swapped, player_0_name, player_1_name }) => {
    const video = await ctx.db.get(videoId)
    if (!video) throw new Error("Video not found")

    const next = { ...(video.playerLabels ?? {}) }
    if (swapped !== undefined) next.swapped = swapped
    if (player_0_name !== undefined) next.player_0_name = player_0_name
    if (player_1_name !== undefined) next.player_1_name = player_1_name

    await ctx.db.patch(videoId, { playerLabels: next })
    return { success: true, playerLabels: next }
  },
})

export const setPlayerThumbnails = mutation({
  args: {
    videoId: v.id("videos"),
    player_0_thumbnail: v.id("_storage"),
    player_1_thumbnail: v.id("_storage"),
  },
  handler: async (ctx, { videoId, player_0_thumbnail, player_1_thumbnail }) => {
    const video = await ctx.db.get(videoId)
    if (!video) throw new Error("Video not found")

    const next = {
      ...(video.playerLabels ?? {}),
      player_0_thumbnail,
      player_1_thumbnail,
    }
    await ctx.db.patch(videoId, { playerLabels: next })
    return { success: true }
  },
})

// =============================================================================
// ACTIONS (for external API calls)
// =============================================================================

/**
 * Trigger Modal processing for a video
 */
export const processVideo = action({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    // Get video record
    const video = await ctx.runQuery(api.videos.getVideo, { videoId })
    if (!video) {
      throw new Error("Video not found")
    }
    
    // Get temporary URL for Modal to download
    if (!video.videoUrl) {
      throw new Error("Failed to get video URL")
    }
    
    // Get analysis mode (default to "full" for backwards compatibility)
    const analysisMode = video.analysisMode ?? "full"
    const cameraAngle = video.cameraAngle ?? "overhead"
    const trackerType = video.trackerType ?? "botsort"

    // Get manual court keypoints if available (for ROI filtering)
    const keypointsData = await ctx.runQuery(api.videos.getManualCourtKeypoints, { videoId })
    
    // Update status to processing
    await ctx.runMutation(api.videos.updateStatus, {
      videoId,
      status: "processing",
      progress: 0,
    })
    
    // Add log
    await ctx.runMutation(api.videos.addLog, {
      videoId,
      message: "Starting video analysis...",
      level: "info",
      category: "processing",
    })
    
    // Get Modal endpoint URL from environment
    const modalUrl = process.env.MODAL_ENDPOINT_URL
    if (!modalUrl) {
      await ctx.runMutation(api.videos.updateStatus, {
        videoId,
        status: "failed",
        error: "Modal endpoint not configured",
      })
      throw new Error("MODAL_ENDPOINT_URL not configured")
    }
    
    // Get Convex HTTP endpoint URL for callbacks
    const convexSiteUrl = process.env.CONVEX_SITE_URL
    if (!convexSiteUrl) {
      await ctx.runMutation(api.videos.updateStatus, {
        videoId,
        status: "failed",
        error: "Convex site URL not configured",
      })
      throw new Error("CONVEX_SITE_URL not configured")
    }
    
    try {
      // Call Modal endpoint
      await ctx.runMutation(api.videos.addLog, {
        videoId,
        message: "Sending video to Modal GPU for processing...",
        level: "info",
        category: "modal",
      })
      
      // Include manual court keypoints if available for ROI filtering
      const hasCourtKeypoints = keypointsData?.has_manual_keypoints && keypointsData.keypoints
      if (hasCourtKeypoints) {
        await ctx.runMutation(api.videos.addLog, {
          videoId,
          message: "Court ROI filter will be active (manual keypoints provided)",
          level: "info",
          category: "court",
        })
      }
      
      // Modal endpoint returns immediately after spawning GPU worker.
      // Actual processing happens in background with HTTP callbacks for progress.
      const response = await fetch(modalUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoId,
          videoUrl: video.videoUrl,
          callbackUrl: convexSiteUrl,
          // Pass manual keypoints for court ROI filtering
          manualCourtKeypoints: hasCourtKeypoints ? keypointsData.keypoints : null,
          analysisMode,
          cameraAngle,
          trackerType,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Modal processing failed: ${errorText}`)
      }

      await ctx.runMutation(api.videos.addLog, {
        videoId,
        message: "Video processing started on Modal GPU",
        level: "success",
        category: "modal",
      })
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error"
      
      await ctx.runMutation(api.videos.updateStatus, {
        videoId,
        status: "failed",
        error: errorMessage,
      })
      
      await ctx.runMutation(api.videos.addLog, {
        videoId,
        message: `Processing failed: ${errorMessage}`,
        level: "error",
        category: "modal",
      })
      
      throw error
    }
  },
})
