import { defineSchema, defineTable } from "convex/server"
import { v } from "convex/values"

export default defineSchema({
  videos: defineTable({
    // File storage
    storageId: v.id("_storage"),
    filename: v.string(),
    size: v.number(),
    
    // Processing status
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
    
    // Analysis results metadata (small data stored directly)
    resultsMeta: v.optional(v.object({
      duration: v.number(),
      fps: v.number(),
      total_frames: v.number(),
      processed_frames: v.number(),
      player_count: v.optional(v.number()),
      has_court_detection: v.optional(v.boolean()),
      has_shuttle_analytics: v.optional(v.boolean()),
    })),
    
    // Full results stored as file (for large data like skeleton_data)
    resultsStorageId: v.optional(v.id("_storage")),
    
    // Processed video storage (optional - for videos with overlays)
    processedVideoStorageId: v.optional(v.id("_storage")),
    
    // Skeleton data storage ID (for large skeleton data)
    skeletonDataStorageId: v.optional(v.id("_storage")),
    
    // Manual court keypoints (12-point system for homography)
    manualCourtKeypoints: v.optional(v.object({
      top_left: v.array(v.number()),
      top_right: v.array(v.number()),
      bottom_right: v.array(v.number()),
      bottom_left: v.array(v.number()),
      center_near: v.optional(v.array(v.number())),
      center_far: v.optional(v.array(v.number())),
      net_left: v.optional(v.array(v.number())),
      net_right: v.optional(v.array(v.number())),
      service_line_near_left: v.optional(v.array(v.number())),
      service_line_near_right: v.optional(v.array(v.number())),
      service_line_far_left: v.optional(v.array(v.number())),
      service_line_far_right: v.optional(v.array(v.number())),
      doubles_left_near: v.optional(v.array(v.number())),
      doubles_right_near: v.optional(v.array(v.number())),
    })),
    
    // Metadata
    createdAt: v.number(),
    completedAt: v.optional(v.number()),
    processingStartedAt: v.optional(v.number()),
  })
    .index("by_status", ["status"])
    .index("by_createdAt", ["createdAt"]),
  
  // Processing logs for real-time log display
  processingLogs: defineTable({
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
    timestamp: v.number(),
  })
    .index("by_videoId", ["videoId"])
    .index("by_videoId_timestamp", ["videoId", "timestamp"]),
})
