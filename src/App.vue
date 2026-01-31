<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import VideoUpload from '@/components/VideoUpload.vue'
import VideoPlayer from '@/components/VideoPlayer.vue'
import ResultsDashboard from '@/components/ResultsDashboard.vue'
import AnalysisProgress from '@/components/AnalysisProgress.vue'
import MiniCourt from '@/components/MiniCourt.vue'
import SpeedGraph from '@/components/SpeedGraph.vue'
import {
  checkApiHealth, getOriginalVideoUrl, setManualCourtKeypoints, getManualKeypointsStatus,
  getHeatmap, preloadHeatmap, triggerSpeedRecalculation, clearSpeedCache, getSpeedTimeline
} from '@/services/api'
import type { UploadResponse, AnalysisResult, SkeletonFrame, CourtModelType } from '@/types/analysis'
import type { HeatmapData, SpeedDataResponse, SpeedTimelineResponse } from '@/services/api'

type AppState = 'upload' | 'analyzing' | 'results'

const currentState = ref<AppState>('upload')
const uploadedVideo = ref<UploadResponse | null>(null)
const analysisResult = ref<AnalysisResult | null>(null)
const errorMessage = ref('')
const isApiConnected = ref(false)

// Overlay visibility toggles
const showSkeleton = ref(true)
const showBoundingBoxes = ref(true)
const showPoseOverlay = ref(true)
// Pose source: 'skeleton' (YOLO-pose keypoints), 'trained' (custom AI model), 'both'
const poseSource = ref<'skeleton' | 'trained' | 'both'>('both')
// NOTE: Court overlay removed - automatic court detection disabled, using manual keypoints only
const showHeatmap = ref(false)  // Heatmap off by default - toggle to show player position heatmap

// Heatmap data storage
const heatmapData = ref<HeatmapData | null>(null)
const isHeatmapLoading = ref(false)

// Detection type visibility toggles (within bounding boxes)
const showPlayers = ref(true)
const showShuttles = ref(true)
const showRackets = ref(true)

// Mini court visibility toggle
const showMiniCourt = ref(true)

// Speed graph visibility toggle
const showSpeedGraph = ref(false)

// Settings panel visibility toggle
const showSettingsPanel = ref(false)

// Video container height tracking for MiniCourt sync
const videoSectionRef = ref<HTMLElement | null>(null)
const videoContainerHeight = ref(440) // Default fallback
let resizeObserver: ResizeObserver | null = null

// Keypoint selection state (from VideoPlayer)
const isKeypointSelectionActive = ref(false)
const keypointSelectionCount = ref(0)

// Court model selection removed - using manual keypoints only
// Note: The CourtModelType is kept for backwards compatibility but not used in UI

// Current video time and frame tracking for MiniCourt sync
const currentVideoTime = ref(0)
const currentFrame = ref(0)

// Extended court keypoints type for 12-point system
interface ExtendedCourtKeypoints {
  // 4 outer corners
  top_left: number[]
  top_right: number[]
  bottom_right: number[]
  bottom_left: number[]
  // Net intersections
  net_left: number[]
  net_right: number[]
  // Service line corners (near court - top half)
  service_near_left: number[]
  service_near_right: number[]
  // Service line corners (far court - bottom half)
  service_far_left: number[]
  service_far_right: number[]
  // Center line endpoints
  center_near: number[]
  center_far: number[]
}

// Manual court keypoints storage (for mini court when manually set)
const manualCourtKeypoints = ref<ExtendedCourtKeypoints | null>(null)

// ============================================================================
// DELAYED SPEED CALCULATION STATE
// ============================================================================
// Speed calculation is delayed until BOTH conditions are met:
// 1. Court keypoints have been confirmed by the user
// 2. Video playback has been initiated
//
// This ensures accurate speed measurements using proper spatial calibration.

// Track whether video playback has been initiated
const videoPlaybackStarted = ref(false)

// Track whether speed calculation has been triggered (to avoid duplicate calls)
const speedCalculationTriggered = ref(false)

// Track speed calculation status
const isSpeedCalculating = ref(false)

// Store calculated speed data
const calculatedSpeedData = ref<SpeedDataResponse['speed_data'] | null>(null)

// Computed: Check if both conditions for speed calculation are met
const canCalculateSpeed = computed(() => {
  return manualCourtKeypoints.value !== null &&
         videoPlaybackStarted.value &&
         analysisResult.value !== null
})

// Computed: Get current skeleton frame based on video time
const currentSkeletonFrame = computed(() => {
  if (!analysisResult.value?.skeleton_data || analysisResult.value.skeleton_data.length === 0) {
    return null
  }
  
  const targetTime = currentVideoTime.value
  const frames = analysisResult.value.skeleton_data
  
  // Binary search for closest frame
  let left = 0
  let right = frames.length - 1
  
  while (left < right) {
    const mid = Math.floor((left + right) / 2)
    const midFrame = frames[mid]
    if (midFrame && midFrame.timestamp < targetTime) {
      left = mid + 1
    } else {
      right = mid
    }
  }
  
  // Check if previous frame is closer
  if (left > 0) {
    const prevFrame = frames[left - 1]
    const currFrame = frames[left]
    if (prevFrame && currFrame) {
      const prevDiff = Math.abs(prevFrame.timestamp - targetTime)
      const currDiff = Math.abs(currFrame.timestamp - targetTime)
      if (prevDiff < currDiff) {
        return prevFrame
      }
    }
  }
  
  return frames[left] ?? null
})

// Computed: Get all court keypoints for MiniCourt (12 points when available, else 4 corners)
const courtCornersForMiniCourt = computed(() => {
  // Prefer manual keypoints if set (full 12-point set)
  if (manualCourtKeypoints.value) {
    const kp = manualCourtKeypoints.value
    return [
      kp.top_left,           // 0: TL corner
      kp.top_right,          // 1: TR corner
      kp.bottom_right,       // 2: BR corner
      kp.bottom_left,        // 3: BL corner
      kp.net_left,           // 4: Net left
      kp.net_right,          // 5: Net right
      kp.service_near_left,  // 6: Service near left
      kp.service_near_right, // 7: Service near right
      kp.service_far_left,   // 8: Service far left
      kp.service_far_right,  // 9: Service far right
      kp.center_near,        // 10: Center line near
      kp.center_far          // 11: Center line far
    ]
  }
  // Otherwise use auto-detected court corners (4 points)
  return analysisResult.value?.court_detection?.court_corners ?? null
})

// Computed: Get current players for MiniCourt
const currentPlayers = computed(() => {
  return currentSkeletonFrame.value?.players ?? []
})

// Computed: Get current shuttle position for MiniCourt
const currentShuttlePosition = computed(() => {
  return currentSkeletonFrame.value?.shuttle_position ?? null
})

// Handle video time updates from VideoPlayer
function handleTimeUpdate(time: number) {
  currentVideoTime.value = time
}

function handleFrameUpdate(frame: number) {
  currentFrame.value = frame
}

// Handle video play event - sets videoPlaybackStarted to true
function handleVideoPlay() {
  if (!videoPlaybackStarted.value) {
    console.log('[App] Video playback initiated')
    videoPlaybackStarted.value = true
  }
}

// Handle keypoint selection mode changes from VideoPlayer
function handleKeypointSelectionChange(isActive: boolean, count: number) {
  isKeypointSelectionActive.value = isActive
  keypointSelectionCount.value = count
}

// Trigger speed recalculation when both conditions are met
async function triggerDelayedSpeedCalculation() {
  if (speedCalculationTriggered.value || !analysisResult.value) return
  
  speedCalculationTriggered.value = true
  isSpeedCalculating.value = true
  
  console.log('[App] Both conditions met - triggering speed calculation')
  console.log('[App]   - Court keypoints confirmed: ✓')
  console.log('[App]   - Video playback initiated: ✓')
  
  try {
    const videoId = analysisResult.value.video_id
    
    // Trigger speed recalculation (clears cache and recalculates with manual keypoints)
    const speedResponse = await triggerSpeedRecalculation(videoId)
    calculatedSpeedData.value = speedResponse.speed_data
    
    console.log('[App] Speed calculation complete')
    console.log('[App]   - Manual keypoints used:', speedResponse.manual_keypoints_used)
    console.log('[App]   - Detection source:', speedResponse.detection_source)
    
    // Fetch per-frame timeline data to update skeleton_data
    console.log('[App] Fetching per-frame speed timeline...')
    const timelineResponse = await getSpeedTimeline(videoId)
    
    // Debug: Log timeline response summary
    console.log('[App] Timeline response received:')
    console.log('  - Players in response:', Object.keys(timelineResponse.players || {}).length)
    console.log('  - Manual keypoints used:', timelineResponse.manual_keypoints_used)
    console.log('  - Detection source:', timelineResponse.detection_source)
    
    // Debug: Log sample speeds for each player
    for (const [playerId, playerTimeline] of Object.entries(timelineResponse.players || {})) {
      const speeds = playerTimeline.speeds_kmh || []
      const nonZeroSpeeds = speeds.filter((s: number) => s > 0)
      const maxSpeed = speeds.length > 0 ? Math.max(...speeds) : 0
      console.log(`  - Player ${playerId}: ${speeds.length} frames, ` +
        `${nonZeroSpeeds.length} non-zero speeds, max: ${maxSpeed.toFixed(1)} km/h`)
    }
    
    // Update skeleton_data with calculated speeds from timeline
    if (analysisResult.value?.skeleton_data && timelineResponse.players) {
      console.log('[App] Updating skeleton_data with calculated speeds...')
      
      // Build a frame-to-speed lookup for each player
      const playerSpeedsByFrame: Record<number, Record<number, number>> = {}
      
      for (const [playerId, playerTimeline] of Object.entries(timelineResponse.players)) {
        const pid = parseInt(playerId)
        playerTimeline.frames.forEach((frameNum: number, idx: number) => {
          if (!playerSpeedsByFrame[frameNum]) {
            playerSpeedsByFrame[frameNum] = {}
          }
          playerSpeedsByFrame[frameNum][pid] = playerTimeline.speeds_kmh[idx] ?? 0
        })
      }
      
      // Debug: Check frame lookup coverage
      const frameNumbers = Object.keys(playerSpeedsByFrame).map(Number)
      console.log(`[App] Frame speed lookup: ${frameNumbers.length} frames covered, ` +
        `range: ${Math.min(...frameNumbers)}-${Math.max(...frameNumbers)}`)
      
      // Update each skeleton frame's player current_speed
      // Create new objects to ensure Vue reactivity triggers
      let updatedCount = 0
      let nonZeroUpdates = 0
      const updatedSkeletonData = analysisResult.value.skeleton_data.map(frame => {
        const frameSpeeds = playerSpeedsByFrame[frame.frame]
        if (frameSpeeds) {
          const updatedPlayers = frame.players.map(player => {
            const speed = frameSpeeds[player.player_id]
            if (speed !== undefined) {
              updatedCount++
              if (speed > 0) nonZeroUpdates++
              return { ...player, current_speed: speed }
            }
            return player
          })
          return { ...frame, players: updatedPlayers }
        }
        return frame
      })
      
      // Replace skeleton_data with new array to trigger reactivity
      analysisResult.value.skeleton_data = updatedSkeletonData
      
      console.log(`[App] Updated ${updatedCount} player speed entries (${nonZeroUpdates} non-zero)`)
      
      // Debug: Verify a sample of the updated data
      const sampleFrames = updatedSkeletonData.slice(0, 5)
      for (const frame of sampleFrames) {
        for (const player of frame.players) {
          console.log(`[App] Sample - Frame ${frame.frame}, Player ${player.player_id}: ${player.current_speed?.toFixed(1)} km/h`)
        }
      }
    }
    
    // Update player summary statistics from speed response for ResultsDashboard
    // The statistics object has per-player avg_speed, max_speed data with NESTED structure:
    // statistics[player_id] = { avg: { speed_kmh }, max: { speed_kmh }, total_distance_m }
    if (analysisResult.value?.players && speedResponse.speed_data?.statistics) {
      console.log('[App] Updating player summary statistics...')
      
      const statistics = speedResponse.speed_data.statistics
      
      for (const player of analysisResult.value.players) {
        // Statistics are keyed by player_id as string
        const playerStats = statistics[String(player.player_id)]
        if (playerStats) {
          // Update player metrics with calculated speeds (in km/h)
          // Backend returns nested structure: { avg: { speed_kmh }, max: { speed_kmh }, total_distance_m }
          player.avg_speed = playerStats.avg?.speed_kmh ?? 0
          player.max_speed = playerStats.max?.speed_kmh ?? 0
          player.total_distance = playerStats.total_distance_m ?? 0
          
          console.log(`[App]   Player ${player.player_id}: avg=${player.avg_speed.toFixed(1)} km/h, max=${player.max_speed.toFixed(1)} km/h, dist=${player.total_distance.toFixed(1)}m`)
        }
      }
    }
    
    // Force Vue reactivity by creating a new reference
    analysisResult.value = { ...analysisResult.value }
    
    // Clear any error message
    errorMessage.value = ''
  } catch (e) {
    console.error('[App] Speed calculation failed:', e)
    errorMessage.value = 'Failed to calculate player speeds'
    speedCalculationTriggered.value = false // Allow retry
  } finally {
    isSpeedCalculating.value = false
  }
}

// Use original video URL for playback (browser-compatible)
// Skeleton overlay is rendered client-side via canvas
const videoUrl = computed(() => {
  if (!analysisResult.value) return ''
  return getOriginalVideoUrl(analysisResult.value.video_id)
})

function handleUploadComplete(response: UploadResponse) {
  uploadedVideo.value = response
  currentState.value = 'analyzing'
  errorMessage.value = ''
}

function handleUploadError(message: string) {
  errorMessage.value = message
}

function handleAnalysisComplete(result: AnalysisResult) {
  analysisResult.value = result
  currentState.value = 'results'
  
  // Preload heatmap data in background for instant toggle
  preloadHeatmapData(result.video_id)
}

function handleAnalysisError(message: string) {
  errorMessage.value = message
  currentState.value = 'upload'
}

function handleAnalysisCancel() {
  currentState.value = 'upload'
  uploadedVideo.value = null
}

function startNewAnalysis() {
  // Save video_id before resetting (for cache clearing)
  const previousVideoId = analysisResult.value?.video_id
  
  currentState.value = 'upload'
  uploadedVideo.value = null
  analysisResult.value = null
  errorMessage.value = ''
  
  // Reset delayed speed calculation state
  manualCourtKeypoints.value = null
  videoPlaybackStarted.value = false
  speedCalculationTriggered.value = false
  isSpeedCalculating.value = false
  calculatedSpeedData.value = null
  
  // Clear speed cache for clean start
  if (previousVideoId) {
    clearSpeedCache(previousVideoId)
  }
}

function dismissError() {
  errorMessage.value = ''
}

// Heatmap functions
async function preloadHeatmapData(videoId: string) {
  try {
    isHeatmapLoading.value = true
    const response = await getHeatmap(videoId)
    heatmapData.value = response.heatmap
    console.log('[Heatmap] Preloaded heatmap data for video', videoId)
  } catch (e) {
    console.warn('[Heatmap] Failed to preload heatmap:', e)
    // Don't show error - heatmap is optional
  } finally {
    isHeatmapLoading.value = false
  }
}

async function loadHeatmapData() {
  if (!analysisResult.value) return
  if (heatmapData.value) return // Already loaded
  
  try {
    isHeatmapLoading.value = true
    const response = await getHeatmap(analysisResult.value.video_id)
    heatmapData.value = response.heatmap
  } catch (e) {
    console.error('[Heatmap] Failed to load heatmap:', e)
    errorMessage.value = 'Failed to load heatmap data'
    showHeatmap.value = false // Disable toggle on error
  } finally {
    isHeatmapLoading.value = false
  }
}

// Handle manual court keypoints from VideoPlayer (12-point system)
async function handleCourtKeypointsSet(keypoints: ExtendedCourtKeypoints) {
  try {
    console.log('[App] Received 12 court keypoints:', keypoints)
    
    // Store locally for MiniCourt component
    manualCourtKeypoints.value = keypoints
    
    // Backend API still uses 4-corner format for basic court detection
    // Send the 4 corners for backward compatibility
    const fourCornerFormat = {
      top_left: keypoints.top_left,
      top_right: keypoints.top_right,
      bottom_right: keypoints.bottom_right,
      bottom_left: keypoints.bottom_left
    }
    const response = await setManualCourtKeypoints(fourCornerFormat)
    console.log('[App] Manual keypoints set successfully:', response)
    // Show success feedback (optional)
    errorMessage.value = '' // Clear any previous error
  } catch (e) {
    console.error('Failed to set manual court keypoints:', e)
    errorMessage.value = 'Failed to set manual court keypoints'
  }
}

// Watch for heatmap toggle - load data when enabled
watch(showHeatmap, (enabled) => {
  if (enabled && !heatmapData.value && analysisResult.value) {
    loadHeatmapData()
  }
})

// Watch for conditions to trigger delayed speed calculation
// Speed is calculated when BOTH conditions are met:
// 1. Manual court keypoints have been set (provides spatial calibration)
// 2. Video playback has been initiated (confirms user is ready)
watch(canCalculateSpeed, (canCalculate) => {
  if (canCalculate && !speedCalculationTriggered.value) {
    triggerDelayedSpeedCalculation()
  }
})

// Setup ResizeObserver for video container height tracking
function setupResizeObserver() {
  if (videoSectionRef.value && !resizeObserver) {
    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        // Get the height of the video container
        const height = entry.contentRect.height
        if (height > 0) {
          videoContainerHeight.value = Math.round(height)
        }
      }
    })
    resizeObserver.observe(videoSectionRef.value)
  }
}

onMounted(async () => {
  isApiConnected.value = await checkApiHealth()
  // Setup resize observer after mount
  setupResizeObserver()
})

onUnmounted(() => {
  // Cleanup resize observer
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
})

// Watch for video section ref changes (e.g., when results view is shown)
watch(videoSectionRef, () => {
  setupResizeObserver()
})
</script>

<template>
  <div class="app">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <div class="logo">
          <h1>SHUTTL.</h1>
        </div>

        <nav class="nav">
          <span class="api-status" :class="{ connected: isApiConnected }">
            <span class="status-dot" />
            {{ isApiConnected ? 'API Connected' : 'API Disconnected' }}
          </span>
        </nav>
      </div>
    </header>

    <!-- Error Message -->
    <Transition name="fade">
      <div v-if="errorMessage" class="error-banner">
        <div class="error-content">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{{ errorMessage }}</span>
          <button @click="dismissError" class="dismiss-btn">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>
    </Transition>

    <!-- Main Content -->
    <main class="main">
      <!-- Upload State -->
      <Transition name="slide" mode="out-in">
        <div v-if="currentState === 'upload'" key="upload" class="content-section">
          <div class="hero">
            <h2>Analyze Your Badminton Match</h2>
            <p>
              Upload a video of your badminton match and get detailed analysis including
              player movement patterns, speed metrics, and shot analysis powered by AI.
            </p>
          </div>

          <VideoUpload
            @uploaded="handleUploadComplete"
            @error="handleUploadError"
          />

          <div class="features">
            <div class="feature">
              <div class="feature-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              </div>
              <h3>Pose Estimation</h3>
              <p>Track player body positions frame by frame with skeleton overlay</p>
            </div>

            <div class="feature">
              <div class="feature-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                </svg>
              </div>
              <h3>Speed Analysis</h3>
              <p>Calculate player movement speed and shot velocities</p>
            </div>

            <div class="feature">
              <div class="feature-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="20" x2="18" y2="10" />
                  <line x1="12" y1="20" x2="12" y2="4" />
                  <line x1="6" y1="20" x2="6" y2="14" />
                </svg>
              </div>
              <h3>Performance Metrics</h3>
              <p>Get detailed statistics on distance covered and rally patterns</p>
            </div>
          </div>
        </div>

        <!-- Analyzing State -->
        <div v-else-if="currentState === 'analyzing' && uploadedVideo" key="analyzing" class="content-section">
          <AnalysisProgress
            :video-id="uploadedVideo.video_id"
            :filename="uploadedVideo.filename"
            @complete="handleAnalysisComplete"
            @error="handleAnalysisError"
            @cancel="handleAnalysisCancel"
          />
        </div>

        <!-- Results State -->
        <div v-else-if="currentState === 'results' && analysisResult" key="results" class="content-section results-view">
          <div class="results-header">
            <button class="back-btn" @click="startNewAnalysis">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="19" y1="12" x2="5" y2="12" />
                <polyline points="12 19 5 12 12 5" />
              </svg>
              New Analysis
            </button>

            <!-- Settings Toggle Button -->
            <button
              class="settings-toggle-btn"
              :class="{ active: showSettingsPanel }"
              @click="showSettingsPanel = !showSettingsPanel"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
              <span>Display Settings</span>
              <svg class="chevron" :class="{ rotated: showSettingsPanel }" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </div>

          <!-- Collapsible Settings Panel -->
          <Transition name="settings-slide">
            <div v-if="showSettingsPanel" class="settings-panel">
              <div class="settings-panel-content">
                <!-- Overlays Section -->
                <div class="settings-section">
                  <h4 class="settings-section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <line x1="3" y1="9" x2="21" y2="9" />
                      <line x1="9" y1="21" x2="9" y2="9" />
                    </svg>
                    Overlays
                  </h4>
                  <div class="settings-grid">
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showSkeleton" />
                        <span class="toggle-slider" />
                      </label>
                      <span>Skeleton</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showBoundingBoxes" />
                        <span class="toggle-slider" />
                      </label>
                      <span>Bounding Boxes</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showPoseOverlay" />
                        <span class="toggle-slider pose-toggle" />
                      </label>
                      <span>Pose Info</span>
                    </div>
                    <!-- Pose Source Selector (only visible when Pose Info is enabled) -->
                    <div class="toggle-item pose-source-selector" v-if="showPoseOverlay">
                      <span class="source-label">Source:</span>
                      <select v-model="poseSource" class="pose-source-select">
                        <option value="both">Both (Skeleton + AI)</option>
                        <option value="skeleton">Skeleton Only</option>
                        <option value="trained">AI Model Only</option>
                      </select>
                    </div>
                    <!-- Court Lines toggle removed - automatic detection disabled, using manual keypoints only -->
                    <div class="toggle-item" :class="{ loading: isHeatmapLoading }">
                      <label class="toggle">
                        <input type="checkbox" v-model="showHeatmap" :disabled="isHeatmapLoading" />
                        <span class="toggle-slider heatmap-toggle" />
                      </label>
                      <span>Position Heatmap</span>
                      <span v-if="isHeatmapLoading" class="loading-indicator">
                        <svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-linecap="round" />
                        </svg>
                      </span>
                    </div>
                  </div>
                </div>

                <!-- Detections Section -->
                <div class="settings-section">
                  <h4 class="settings-section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                    Object Detection
                  </h4>
                  <p class="settings-section-hint" v-if="!showBoundingBoxes">Enable "Bounding Boxes" to see detections</p>
                  <div class="settings-grid" :class="{ disabled: !showBoundingBoxes }">
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showPlayers" :disabled="!showBoundingBoxes" />
                        <span class="toggle-slider player-toggle" />
                      </label>
                      <span>Players</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showShuttles" :disabled="!showBoundingBoxes" />
                        <span class="toggle-slider shuttle-toggle" />
                      </label>
                      <span>Shuttlecock</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showRackets" :disabled="!showBoundingBoxes" />
                        <span class="toggle-slider racket-toggle" />
                      </label>
                      <span>Rackets</span>
                    </div>
                  </div>
                </div>

                <!-- Views Section -->
                <div class="settings-section">
                  <h4 class="settings-section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect x="3" y="3" width="7" height="7" />
                      <rect x="14" y="3" width="7" height="7" />
                      <rect x="14" y="14" width="7" height="7" />
                      <rect x="3" y="14" width="7" height="7" />
                    </svg>
                    Additional Views
                  </h4>
                  <div class="settings-grid">
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showMiniCourt" />
                        <span class="toggle-slider minicourt-toggle" />
                      </label>
                      <span>Mini Court Map</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showSpeedGraph" />
                        <span class="toggle-slider speedgraph-toggle" />
                      </label>
                      <span>Speed Graph</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Transition>

          <div class="results-content" :class="{ 'with-minicourt': showMiniCourt }">
            <div class="video-with-minicourt">
              <div ref="videoSectionRef" class="video-section">
                <VideoPlayer
                  :video-url="videoUrl"
                  :skeleton-data="analysisResult.skeleton_data"
                  :heatmap-data="heatmapData"
                  :show-skeleton="showSkeleton"
                  :show-bounding-boxes="showBoundingBoxes"
                  :show-players="showPlayers"
                  :show-shuttles="showShuttles"
                  :show-rackets="showRackets"
                  :show-pose-overlay="showPoseOverlay"
                  :pose-source="poseSource"
                  :show-heatmap="showHeatmap"
                  @court-keypoints-set="handleCourtKeypointsSet"
                  @time-update="handleTimeUpdate"
                  @frame-update="handleFrameUpdate"
                  @play="handleVideoPlay"
                  @keypoint-selection-change="handleKeypointSelectionChange"
                />
              </div>
              
              <!-- Mini Court Panel -->
              <Transition name="slide-fade">
                <div v-if="showMiniCourt || isKeypointSelectionActive" class="minicourt-section">
                  <MiniCourt
                    :court-corners="courtCornersForMiniCourt"
                    :players="currentPlayers"
                    :shuttle-position="currentShuttlePosition"
                    :width="240"
                    :height="videoContainerHeight"
                    :show-grid="true"
                    :show-labels="true"
                    :show-shuttle="showShuttles"
                    :show-trails="false"
                    :is-keypoint-selection-mode="isKeypointSelectionActive"
                    :keypoint-selection-count="keypointSelectionCount"
                  />
                </div>
              </Transition>
            </div>
            
            <!-- Speed Graph Panel -->
            <Transition name="slide-fade">
              <div v-if="showSpeedGraph && analysisResult" class="speedgraph-section">
                <SpeedGraph
                  :skeleton-data="analysisResult.skeleton_data"
                  :fps="analysisResult.fps"
                  :current-frame="currentFrame"
                  :visible="showSpeedGraph"
                  :is-speed-calculating="isSpeedCalculating"
                  :speed-calculated="speedCalculationTriggered && !isSpeedCalculating"
                  :court-keypoints-set="manualCourtKeypoints !== null"
                  :video-started="videoPlaybackStarted"
                  @close="showSpeedGraph = false"
                />
              </div>
            </Transition>

            <div class="dashboard-section">
              <ResultsDashboard
                :result="analysisResult"
                :manual-keypoints-set="manualCourtKeypoints !== null"
              />
            </div>
          </div>
        </div>
      </Transition>
    </main>

    <!-- Footer -->
    <footer class="footer">
    </footer>
  </div>
</template>

<style>
/* Global Styles - Minimalist Theme */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0d0d0d;
  color: #e2e8f0;
  min-height: 100vh;
}

a {
  color: #22c55e;
  text-decoration: none;
  transition: color 0.2s ease;
}

a:hover {
  color: #4ade80;
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.slide-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}
</style>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
.header {
  padding: 16px 24px;
  border-bottom: 1px solid #222;
  position: sticky;
  top: 0;
  z-index: 100;
  background: #0d0d0d;
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo svg {
  width: 32px;
  height: 32px;
  color: #22c55e;
}

.logo h1 {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffffff;
}

.nav {
  display: flex;
  align-items: center;
  gap: 16px;
}

.api-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  font-size: 0.75rem;
  color: #ef4444;
}

.api-status.connected {
  background: #1a1a1a;
  border-color: #22c55e;
  color: #22c55e;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.nav-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 0;
  color: #888;
  transition: all 0.2s ease;
}

.nav-link:hover {
  background: #1a1a1a;
  color: #fff;
}

.nav-link svg {
  width: 20px;
  height: 20px;
}

/* Error Banner */
.error-banner {
  background: #1a0000;
  border-bottom: 1px solid #ef4444;
}

.error-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #ef4444;
}

.error-content svg {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.error-content span {
  flex: 1;
  font-size: 0.875rem;
}

.dismiss-btn {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: 0;
  color: #ef4444;
  cursor: pointer;
  transition: background 0.2s ease;
}

.dismiss-btn:hover {
  background: #2a0000;
}

.dismiss-btn svg {
  width: 16px;
  height: 16px;
}

/* Main */
.main {
  flex: 1;
  padding: 48px 24px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.content-section {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Hero */
.hero {
  text-align: center;
  margin-bottom: 48px;
}

.hero h2 {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 16px;
  color: #ffffff;
}

.hero p {
  max-width: 600px;
  margin: 0 auto;
  color: #888;
  font-size: 1.125rem;
  line-height: 1.7;
}

/* Features */
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 48px;
}

.feature {
  padding: 24px;
  background: #141414;
  border: 1px solid #222;
  border-radius: 0;
  text-align: center;
  transition: all 0.2s ease;
}

.feature:hover {
  border-color: #22c55e;
}

.feature-icon {
  width: 56px;
  height: 56px;
  margin: 0 auto 16px;
  padding: 14px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  color: #22c55e;
}

.feature-icon svg {
  width: 100%;
  height: 100%;
}

.feature h3 {
  color: #fff;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 8px;
}

.feature p {
  color: #666;
  font-size: 0.875rem;
  line-height: 1.6;
}

/* Results View */
.results-view {
  max-width: 100%;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  gap: 16px;
  flex-wrap: wrap;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: #141414;
  border: 1px solid #333;
  border-radius: 0;
  color: #888;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.back-btn:hover {
  background: #1a1a1a;
  border-color: #22c55e;
  color: #fff;
}

.back-btn svg {
  width: 18px;
  height: 18px;
}

/* Settings Toggle Button */
.settings-toggle-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 18px;
  background: #141414;
  border: 1px solid #333;
  border-radius: 0;
  color: #fff;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-toggle-btn:hover {
  background: #1a1a1a;
  border-color: #22c55e;
}

.settings-toggle-btn.active {
  background: #1a1a1a;
  border-color: #22c55e;
}

.settings-toggle-btn svg:first-child {
  width: 18px;
  height: 18px;
  color: #22c55e;
}

.settings-toggle-btn .chevron {
  width: 16px;
  height: 16px;
  color: #888;
  transition: transform 0.3s ease;
}

.settings-toggle-btn .chevron.rotated {
  transform: rotate(180deg);
}

/* Settings Panel */
.settings-panel {
  background: #141414;
  border: 1px solid #222;
  border-radius: 0;
  margin-bottom: 24px;
  overflow: hidden;
}

.settings-panel-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 24px;
  padding: 24px;
}

.settings-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.settings-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #fff;
  font-size: 0.9rem;
  font-weight: 600;
  padding-bottom: 8px;
  border-bottom: 1px solid #222;
}

.settings-section-title svg {
  width: 18px;
  height: 18px;
  color: #22c55e;
}

.settings-section-hint {
  color: #666;
  font-size: 0.75rem;
  font-style: italic;
}

.settings-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.settings-grid.disabled {
  opacity: 0.4;
  pointer-events: none;
}

/* Settings slide transition */
.settings-slide-enter-active {
  transition: all 0.3s ease-out;
}

.settings-slide-leave-active {
  transition: all 0.2s ease-in;
}

.settings-slide-enter-from {
  opacity: 0;
  max-height: 0;
  transform: translateY(-10px);
}

.settings-slide-leave-to {
  opacity: 0;
  max-height: 0;
  transform: translateY(-10px);
}

.settings-slide-enter-to,
.settings-slide-leave-from {
  opacity: 1;
  max-height: 500px;
}

.toggle-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #888;
  font-size: 0.875rem;
  transition: opacity 0.2s ease;
}

.toggle-item.disabled {
  opacity: 0.4;
  pointer-events: none;
}

.toggle {
  position: relative;
  display: inline-block;
  width: 48px;
  height: 26px;
}

.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  transition: 0.3s;
}

.toggle-slider::before {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 2px;
  bottom: 2px;
  background-color: #444;
  border-radius: 0;
  transition: 0.3s;
}

.toggle input:checked + .toggle-slider {
  background: #22c55e;
  border-color: #22c55e;
}

.toggle input:checked + .toggle-slider.player-toggle {
  background: #22c55e;
  border-color: #22c55e;
}

.toggle input:checked + .toggle-slider.shuttle-toggle {
  background: #f97316;
  border-color: #f97316;
}

.toggle input:checked + .toggle-slider.racket-toggle {
  background: #d946ef;
  border-color: #d946ef;
}

.toggle input:checked + .toggle-slider.pose-toggle {
  background: #ec4899;
  border-color: #ec4899;
}

/* Pose source selector */
.pose-source-selector {
  padding-left: 16px;
  border-left: 2px solid rgba(236, 72, 153, 0.3);
  margin-left: 4px;
}

.pose-source-selector .source-label {
  color: #888;
  font-size: 0.8rem;
  margin-right: 8px;
}

.pose-source-select {
  padding: 6px 12px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0;
  color: #fff;
  font-size: 0.8rem;
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.pose-source-select:hover {
  border-color: #ec4899;
}

.pose-source-select:focus {
  outline: none;
  border-color: #ec4899;
}

.pose-source-select option {
  background: #1a1a1a;
  color: #fff;
}

/* Court toggle removed - automatic detection disabled, using manual keypoints only */

.toggle input:checked + .toggle-slider.heatmap-toggle {
  background: #f97316;
  border-color: #f97316;
}

.toggle input:checked + .toggle-slider::before {
  transform: translateX(22px);
  background-color: #fff;
}

/* Loading state for smoothing toggle */
.toggle-item.loading {
  opacity: 0.7;
}

.loading-indicator {
  display: inline-flex;
  align-items: center;
  margin-left: 4px;
}

.spinner {
  width: 14px;
  height: 14px;
  animation: spin 1s linear infinite;
  color: #22c55e;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.results-content {
  display: grid;
  gap: 32px;
}

.results-content.with-minicourt .video-with-minicourt {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.video-with-minicourt {
  width: 100%;
}

.video-section {
  flex: 1;
  min-width: 0;
}

.minicourt-section {
  flex-shrink: 0;
  position: sticky;
  top: 88px;
}

.dashboard-section {
  width: 100%;
}

/* Mini court toggle style */
.toggle input:checked + .toggle-slider.minicourt-toggle {
  background: #22c55e;
  border-color: #22c55e;
}

/* Speed graph toggle style */
.toggle input:checked + .toggle-slider.speedgraph-toggle {
  background: #3B82F6;
  border-color: #3B82F6;
}

/* Speed graph section */
.speedgraph-section {
  width: 100%;
  margin-top: 16px;
  max-height: 380px;
}

/* Slide-fade transition for mini court */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.3s ease-in;
}

.slide-fade-enter-from {
  transform: translateX(20px);
  opacity: 0;
}

.slide-fade-leave-to {
  transform: translateX(20px);
  opacity: 0;
}

/* Footer */
.footer {
  padding: 24px;
  text-align: center;
  border-top: 1px solid #222;
  color: #666;
  font-size: 0.875rem;
}

/* Responsive */
@media (max-width: 1200px) {
  .results-content.with-minicourt .video-with-minicourt {
    flex-direction: column;
  }
  
  .minicourt-section {
    position: static;
    width: 100%;
    display: flex;
    justify-content: center;
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    gap: 16px;
  }

  .hero h2 {
    font-size: 1.75rem;
  }

  .hero p {
    font-size: 1rem;
  }

  .results-header {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }
  
  .settings-toggle-btn {
    justify-content: center;
  }
  
  .settings-panel-content {
    grid-template-columns: 1fr;
    padding: 16px;
  }
  
  .court-model-indicator {
    justify-content: center;
  }
}

</style>
