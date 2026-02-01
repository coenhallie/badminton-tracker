import { action, mutation, query, internalMutation } from "./_generated/server"
import { v } from "convex/values"
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
    
    return Promise.all(
      videos.map(async (video) => ({
        ...video,
        videoUrl: video.storageId 
          ? await ctx.storage.getUrl(video.storageId)
          : null,
      }))
    )
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
  },
  handler: async (ctx, args) => {
    const videoId = await ctx.db.insert("videos", {
      storageId: args.storageId,
      filename: args.filename,
      size: args.size,
      status: "uploaded",
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
 * Delete a video and its associated data
 */
export const deleteVideo = mutation({
  args: { videoId: v.id("videos") },
  handler: async (ctx, { videoId }) => {
    const video = await ctx.db.get(videoId)
    if (!video) return
    
    // Delete storage files
    if (video.storageId) {
      await ctx.storage.delete(video.storageId)
    }
    if (video.processedVideoStorageId) {
      await ctx.storage.delete(video.processedVideoStorageId)
    }
    if (video.skeletonDataStorageId) {
      await ctx.storage.delete(video.skeletonDataStorageId)
    }
    
    // Delete logs
    const logs = await ctx.db
      .query("processingLogs")
      .withIndex("by_videoId", (q) => q.eq("videoId", videoId))
      .collect()
    
    for (const log of logs) {
      await ctx.db.delete(log._id)
    }
    
    // Delete video record
    await ctx.db.delete(videoId)
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
      
      // Modal endpoint URL already includes the full path
      const response = await fetch(modalUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoId,
          videoUrl: video.videoUrl,
          callbackUrl: convexSiteUrl,
          // Pass manual keypoints for court ROI filtering
          manualCourtKeypoints: hasCourtKeypoints ? keypointsData.keypoints : null,
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
