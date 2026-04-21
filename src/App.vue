<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useTheme } from '@/composables/useTheme'
import VideoUpload from '@/components/VideoUpload.vue'
import VideoPlayer from '@/components/VideoPlayer.vue'
import ResultsDashboard from '@/components/ResultsDashboard.vue'
import AnalysisProgress from '@/components/AnalysisProgress.vue'
import CourtSetup from '@/components/CourtSetup.vue'
import MiniCourt from '@/components/MiniCourt.vue'
import PlayerLabelsProvider from '@/components/PlayerLabelsProvider.vue'
import SpeedGraph from '@/components/SpeedGraph.vue'
import ShotSpeedList from '@/components/ShotSpeedList.vue'
import AdvancedAnalytics from '@/components/AdvancedAnalytics.vue'
import RallyTimeline from '@/components/RallyTimeline.vue'
import { useAdvancedAnalytics } from '@/composables/useAdvancedAnalytics'
import { useShotSegments, type ShotMovementSegmentWithPeaks } from '@/composables/useShotSegments'
import { computeHomographyFromKeypoints } from '@/utils/homography'
import {
  checkApiHealth, getApiHealthDetails, getApiBaseUrl, isUsingConvex, getOriginalVideoUrl, fetchVideoUrl,
  triggerSpeedRecalculation, clearSpeedCache, getSpeedTimeline,
  getRecalculatedZoneAnalytics, setCurrentVideoId
} from '@/services/api'
import type { HealthCheckResponse } from '@/services/api'
import type { UploadResponse, AnalysisResult, SkeletonFrame, ExtendedCourtKeypoints } from '@/types/analysis'
import type { SpeedDataResponse, SpeedTimelineResponse } from '@/services/api'

const { isDark, toggleTheme } = useTheme()

// App states: upload -> court-setup (new!) -> analyzing -> results
type AppState = 'upload' | 'court-setup' | 'analyzing' | 'results'

// Restore session state if the page was reloaded (e.g., Mac sleep/tab discard)
function loadSessionState(): { state: AppState; video: UploadResponse | null } {
  try {
    const saved = sessionStorage.getItem('badminton-session')
    if (saved) {
      const parsed = JSON.parse(saved)
      if (parsed.state && parsed.video?.video_id) {
        return { state: parsed.state as AppState, video: parsed.video }
      }
    }
  } catch { /* ignore corrupt storage */ }
  return { state: 'upload', video: null }
}

function saveSessionState(state: AppState, video: UploadResponse | null) {
  try {
    if (state === 'upload' || !video) {
      sessionStorage.removeItem('badminton-session')
    } else {
      sessionStorage.setItem('badminton-session', JSON.stringify({ state, video }))
    }
  } catch { /* storage full or unavailable */ }
}

const restored = loadSessionState()
const currentState = ref<AppState>(restored.state)
const uploadedVideo = ref<UploadResponse | null>(restored.video)
const analysisMode = ref<'rally_only' | 'full'>(restored.video?.analysisMode ?? 'full')
const cameraAngle = ref<'overhead' | 'corner'>(restored.video?.cameraAngle ?? 'overhead')
const analysisResult = ref<AnalysisResult | null>(null)

const errorMessage = ref('')
const isApiConnected = ref(false)
const healthDetails = ref<HealthCheckResponse | null>(null)
const showHealthModal = ref(false)
const isCheckingHealth = ref(false)
let healthCheckInterval: ReturnType<typeof setInterval> | null = null

// Overlay visibility toggles
const showSkeleton = ref(true)
const showBoundingBoxes = ref(true)
const showPoseOverlay = ref(false)
// Pose source: 'skeleton' (YOLO-pose keypoints), 'trained' (custom AI model), 'both'
const poseSource = ref<'skeleton' | 'trained' | 'both'>('both')
// NOTE: Court overlay removed - automatic court detection disabled, using manual keypoints only
const showHeatmap = ref(false)  // Heatmap off by default - toggle to show player position heatmap

// Detection type visibility toggles (within bounding boxes)
const showPlayers = ref(true)
const showRackets = ref(true)

// Shuttle tracking toggle (controls trail overlay, bounding boxes, and mini court dot)
const showShuttleTracking = ref(true)

// Mini court visibility toggle
const showMiniCourt = ref(true)

// Player trail visibility toggle (movement path lines on mini court)
const showPlayerTrails = ref(true)

// Hit marker visibility toggle (shuttlecock strike locations on mini court)
const showHitMarkers = ref(false)

// Speed graph visibility toggle
const showSpeedGraph = ref(false)

// Shot speed list visibility toggle
const showShotSpeedList = ref(false)

// Settings panel visibility toggle
const showSettingsPanel = ref(false)

// Changelog modal visibility
const showChangelogModal = ref(false)

// Video container height tracking for MiniCourt sync
const videoSectionRef = ref<HTMLElement | null>(null)
const videoContainerHeight = ref(440) // Default fallback
let resizeObserver: ResizeObserver | null = null

// VideoPlayer template ref for seeking
const videoPlayerRef = ref<InstanceType<typeof VideoPlayer> | null>(null)

// Court model selection removed - using manual keypoints only
// Note: The CourtModelType is kept for backwards compatibility but not used in UI

// Current video time and frame tracking for MiniCourt sync
const currentVideoTime = ref(0)
const currentFrame = ref(0)

// Rally auto-pause: pause video between rallies so user can review data
const pauseBetweenRallies = ref(false)
const rallyPauseCountdown = ref(0)
const pausedAfterRallyId = ref<number | null>(null)
const nextRallyStart = ref(0)
let rallyPauseTimer: ReturnType<typeof setInterval> | null = null
const lastTriggeredRallyEnd = ref(-1)

// Shot auto-pause state
const pauseBetweenShots = ref(false)
const shotPauseDurationSec = ref<1 | 1.5 | 2 | 3>(1.5)
const shotPauseCountdown = ref(0)
const currentShotSegment = ref<ShotMovementSegmentWithPeaks | null>(null)
const lastTriggeredShotTime = ref(-1)
let shotPauseTimer: ReturnType<typeof setInterval> | null = null

// Client-side rally detection (same source as AdvancedAnalytics)
const { rallies: detectedRallies, backendRallies, rallySource, rallySpeedStats } = useAdvancedAnalytics(
  computed(() => analysisResult.value),
  currentFrame,
  undefined,
  cameraAngle,
)

// Selected rally for per-rally heatmap filtering
const selectedRallyId = ref<number | null>(null)
const heatmapFrameRange = computed<{ start: number; end: number } | null>(() => {
  if (selectedRallyId.value === null) return null
  const rally = detectedRallies.value.find(r => r.id === selectedRallyId.value)
  if (!rally) return null
  return { start: rally.startFrame, end: rally.endFrame }
})

function handleRallySelect(rallyId: number | null) {
  selectedRallyId.value = rallyId
}

// Manual court keypoints storage (for mini court when manually set).
// The canonical ExtendedCourtKeypoints type lives in @/types/analysis and
// matches the Convex schema field names — so there's no remapping layer
// between persistence and UI. Every consumer (VideoPlayer, CourtSetup,
// SyntheticCourtView, App) imports the same shape.
const manualCourtKeypoints = ref<ExtendedCourtKeypoints | null>(null)

// Reactive skeletonData for the composable.
const skeletonDataRef = computed(() => analysisResult.value?.skeleton_data)

// Homography for leg-stretch meters (null when no manual keypoints).
const shotHomography = computed((): number[][] | null => {
  const kp = manualCourtKeypoints.value
  if (!kp) return null
  const videoPts = [
    kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left,
    kp.net_left, kp.net_right,
    kp.service_line_near_left, kp.service_line_near_right,
    kp.service_line_far_left, kp.service_line_far_right,
    kp.center_near, kp.center_far,
  ]
  return computeHomographyFromKeypoints(videoPts)
})

const { segments: shotSegments } = useShotSegments(skeletonDataRef, shotHomography)

// Layer A of shot-detection quality gates: drop segments that don't fall
// cleanly inside a single detected rally. Eliminates between-rally false
// positives (player walking back, picking up shuttle, pre-serve jitter)
// from triggering the auto-pause. Falls through as a no-op when no rallies
// have been detected so the feature still works on videos without rally
// detection output.
// Padding applied to each rally's boundaries before the in-rally check.
// Rally detection draws boundaries at the first/last detected shot inside
// each rally — but real play often extends past those bounds (the shuttle
// is still live after the final winner's hit, the receiver moves into
// position before the first return, etc.). A generous ±5 s pad makes
// the rally gate a "rough sanity check" rather than a tight filter:
// shots within a few seconds of any rally are kept; only shots in long
// dead-zones (>5 s from any rally) are treated as between-rally noise.
const RALLY_BOUNDARY_PADDING_SECONDS = 5.0

// Layer A: keep only shot segments whose end-shot falls inside a detected rally
// (±RALLY_BOUNDARY_PADDING_SECONDS). Falls through unchanged when no rallies.
const inRallyShotSegments = computed(() => {
  const segs = shotSegments.value
  const rallies = detectedRallies.value
  if (!rallies.length) {
    console.log(`[ShotDetection/LayerA] no rallies; keeping all ${segs.length} segments`)
    return segs
  }
  const pad = RALLY_BOUNDARY_PADDING_SECONDS
  const kept = segs.filter(seg =>
    rallies.some(
      r => seg.endShot.timestamp >= r.startTimestamp - pad &&
           seg.endShot.timestamp <= r.endTimestamp + pad,
    ),
  )
  console.log(
    `[ShotDetection/LayerA] ${kept.length}/${segs.length} segments kept ` +
    `(pad=±${pad}s, rallies=${rallies.length})`
  )
  return kept
})

// Playback view mode: 'video' = real video + overlays (default),
// 'court' = synthetic court redrawn from keypoints + overlays.
// Only available when manual keypoints are set.
const viewMode = ref<'video' | 'court'>('video')

// Guard: if keypoints get cleared while viewing court mode, snap back to video.
watch(manualCourtKeypoints, (kp) => {
  if (kp === null && viewMode.value === 'court') {
    viewMode.value = 'video'
  }
})

// Zone analytics recalculation trigger counter
// Incremented each time keypoints are confirmed to trigger ResultsDashboard refresh
const zoneRecalculationTrigger = ref(0)

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
      kp.service_line_near_left,  // 6: Service near left
      kp.service_line_near_right, // 7: Service near right
      kp.service_line_far_left,   // 8: Service far left
      kp.service_line_far_right,  // 9: Service far right
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

// Handle seek-to-segment from ShotSpeedList (click to replay a shot segment)
function handleSeekToSegment(startTime: number, _endTime: number) {
  if (videoPlayerRef.value) {
    videoPlayerRef.value.seekTo(startTime)
    // Optionally set a slower playback rate for detailed analysis
    videoPlayerRef.value.setPlaybackRate(0.5)
  }
}

// Handle seek from rally/shot selection in AdvancedAnalytics
function handleRallySeek(time: number) {
  if (videoPlayerRef.value) {
    videoPlayerRef.value.seekTo(time)
  }
}

// ── Rally auto-pause logic ──────────────────────────────────────────────────

// Watch video time for rally end boundaries (uses client-side detectedRallies)
watch(currentVideoTime, (time) => {
  // Reset tracking if user seeked backward past last triggered point
  if (time < lastTriggeredRallyEnd.value - 1) {
    lastTriggeredRallyEnd.value = -1
  }

  if (!pauseBetweenRallies.value || !detectedRallies.value.length) return
  if (rallyPauseCountdown.value > 0) return // already in a pause

  for (let i = 0; i < detectedRallies.value.length; i++) {
    const rally = detectedRallies.value[i]
    const next = detectedRallies.value[i + 1]
    if (!next) continue // don't pause after the last rally

    const endTime = rally.endTimestamp
    if (time >= endTime && time < endTime + 1.0 && endTime > lastTriggeredRallyEnd.value) {
      lastTriggeredRallyEnd.value = endTime
      pausedAfterRallyId.value = rally.id
      nextRallyStart.value = next.startTimestamp
      startRallyPause()
      break
    }
  }
})

function startRallyPause() {
  videoPlayerRef.value?.pause()
  rallyPauseCountdown.value = 3

  rallyPauseTimer = setInterval(() => {
    rallyPauseCountdown.value--
    if (rallyPauseCountdown.value <= 0) {
      skipToNextRally()
    }
  }, 1000)
}

function skipToNextRally() {
  clearRallyPause()
  if (videoPlayerRef.value) {
    videoPlayerRef.value.seekTo(nextRallyStart.value)
    videoPlayerRef.value.play()
  }
}

function resumeFromRallyPause() {
  clearRallyPause()
  videoPlayerRef.value?.play()
}

function clearRallyPause() {
  if (rallyPauseTimer) {
    clearInterval(rallyPauseTimer)
    rallyPauseTimer = null
  }
  rallyPauseCountdown.value = 0
  pausedAfterRallyId.value = null
}

// ── Shot auto-pause logic ────────────────────────────────────────────────
// Mirrors the rally-pause watcher's shape exactly.

watch(currentVideoTime, (time) => {
  // Reset tracking on seek-backward (same 1s slack as rally-pause).
  if (time < lastTriggeredShotTime.value - 1) {
    lastTriggeredShotTime.value = -1
  }

  if (!pauseBetweenShots.value) return
  if (shotPauseCountdown.value > 0) return          // already in a pause
  if (rallyPauseCountdown.value > 0) return         // rally-pause has priority
  if (!inRallyShotSegments.value.length) return

  for (let i = 0; i < inRallyShotSegments.value.length; i++) {
    const seg = inRallyShotSegments.value[i]!
    const t = seg.endTimestamp // the "just-happened" shot is endShot of segment i
    if (t <= lastTriggeredShotTime.value) continue
    if (time < t || time >= t + 0.5) continue       // only within 0.5s window

    // Suppress if this is the last shot of its rally AND rally-pause is on.
    if (pauseBetweenRallies.value && isLastShotOfRally(seg.endShot.timestamp)) {
      lastTriggeredShotTime.value = t
      continue
    }

    lastTriggeredShotTime.value = t
    currentShotSegment.value = seg
    startShotPause()
    break
  }
})

function isLastShotOfRally(shotTimestamp: number): boolean {
  // Walk the in-rally list so "next shot" is the next in-rally shot, not a
  // between-rally false positive that would confuse the end-of-rally check.
  const segs = inRallyShotSegments.value
  const idx = segs.findIndex(s => s.endShot.timestamp === shotTimestamp)
  if (idx === -1) return false
  const nextShot = segs[idx + 1]?.endShot
  const rally = detectedRallies.value.find(
    r => shotTimestamp >= r.startTimestamp && shotTimestamp <= r.endTimestamp,
  )
  if (!rally) return false
  if (!nextShot) return true
  return nextShot.timestamp > rally.endTimestamp
}

function startShotPause() {
  videoPlayerRef.value?.pause()
  shotPauseCountdown.value = shotPauseDurationSec.value

  // Tick every 100ms so the countdown displays fractional seconds smoothly.
  shotPauseTimer = setInterval(() => {
    shotPauseCountdown.value = Math.max(0, shotPauseCountdown.value - 0.1)
    if (shotPauseCountdown.value <= 0) {
      endShotPause()
    }
  }, 100)
}

function endShotPause() {
  clearShotPause()
  videoPlayerRef.value?.play()
}

function clearShotPause() {
  if (shotPauseTimer) {
    clearInterval(shotPauseTimer)
    shotPauseTimer = null
  }
  shotPauseCountdown.value = 0
  currentShotSegment.value = null
}

// Clean up timer on unmount
onUnmounted(() => {
  clearRallyPause()
  clearShotPause()
})

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
      
      // Maximum realistic speed threshold (km/h) - filter out tracking errors
      // UNIFIED: Badminton players rarely exceed 25 km/h even in explosive movements
      const MAX_VALID_SPEED_KMH = 25
      
      // Build a frame-to-speed lookup for each player
      const playerSpeedsByFrame: Record<number, Record<number, number>> = {}
      let filteredOutCount = 0
      
      for (const [playerId, playerTimeline] of Object.entries(timelineResponse.players)) {
        const pid = parseInt(playerId)
        playerTimeline.frames.forEach((frameNum: number, idx: number) => {
          if (!playerSpeedsByFrame[frameNum]) {
            playerSpeedsByFrame[frameNum] = {}
          }
          // Filter out unrealistic speeds (tracking errors, ID swaps)
          const rawSpeed = playerTimeline.speeds_kmh[idx] ?? 0
          if (rawSpeed > MAX_VALID_SPEED_KMH) {
            filteredOutCount++
            playerSpeedsByFrame[frameNum][pid] = 0 // Treat as stationary
          } else {
            playerSpeedsByFrame[frameNum][pid] = rawSpeed
          }
        })
      }
      
      if (filteredOutCount > 0) {
        console.log(`[App] Filtered out ${filteredOutCount} unrealistic speed values (>${MAX_VALID_SPEED_KMH} km/h)`)
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
      
      // Maximum realistic speed threshold for statistics (km/h)
      // UNIFIED with backend and Convex speed calculator
      const MAX_VALID_SPEED_KMH = 25
      const REALISTIC_MAX_SPEED_KMH = 25  // Cap displayed max to realistic value
      
      for (const player of analysisResult.value.players) {
        // Statistics are keyed by player_id as string
        const playerStats = statistics[String(player.player_id)]
        if (playerStats) {
          // Update player metrics with calculated speeds (in km/h)
          // Backend returns nested structure: { avg: { speed_kmh }, max: { speed_kmh }, total_distance_m }
          const rawAvgSpeed = playerStats.avg?.speed_kmh ?? 0
          const rawMaxSpeed = playerStats.max?.speed_kmh ?? 0
          
          // Filter out unrealistic values - if avg or max exceeds threshold,
          // the data is corrupted by tracking errors
          if (rawAvgSpeed > MAX_VALID_SPEED_KMH || rawMaxSpeed > MAX_VALID_SPEED_KMH) {
            console.warn(`[App]   Player ${player.player_id}: Unrealistic speed detected (avg=${rawAvgSpeed.toFixed(1)}, max=${rawMaxSpeed.toFixed(1)}), capping to realistic values`)
            player.avg_speed = Math.min(rawAvgSpeed, REALISTIC_MAX_SPEED_KMH)
            player.max_speed = Math.min(rawMaxSpeed, REALISTIC_MAX_SPEED_KMH)
          } else {
            player.avg_speed = rawAvgSpeed
            player.max_speed = rawMaxSpeed
          }
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
const videoUrl = ref('')

// Fetch video URL asynchronously (for Convex storage support)
async function loadVideoUrl(videoId: string) {
  try {
    videoUrl.value = await fetchVideoUrl(videoId)
  } catch (error) {
    console.error('Failed to fetch video URL:', error)
    // Fallback to local URL format
    videoUrl.value = getOriginalVideoUrl(videoId)
  }
}

function handleUploadComplete(response: UploadResponse) {
  uploadedVideo.value = response
  analysisMode.value = response.analysisMode
  cameraAngle.value = response.cameraAngle ?? 'overhead'
  // Skip court setup for rally-only mode (no player analysis needs it)
  currentState.value = response.analysisMode === 'rally_only' ? 'analyzing' : 'court-setup'
  errorMessage.value = ''
}

function handleUploadError(message: string) {
  errorMessage.value = message
}

// Court setup handlers - receives 12-point keypoints from CourtSetup.
// Field names match the Convex schema 1:1 (no remapping needed).
function handleCourtSetupComplete(keypoints: ExtendedCourtKeypoints) {
  console.log('[App] 12-point court keypoints saved:', keypoints)
  manualCourtKeypoints.value = keypoints
  currentState.value = 'analyzing'
}

function handleCourtSetupSkip() {
  console.log('[App] Court setup skipped - using fallback filtering')
  // Proceed to analysis without keypoints (Modal will use rectangular fallback filter)
  currentState.value = 'analyzing'
}

function handleCourtSetupError(message: string) {
  console.error('[App] Court setup error:', message)
  errorMessage.value = message
  // Allow retry or skip
}

async function handleAnalysisComplete(result: AnalysisResult) {
  analysisResult.value = result
  if (result.camera_angle) {
    cameraAngle.value = result.camera_angle
  }
  currentState.value = 'results'
  
  // Set the current video ID for keypoints operations
  setCurrentVideoId(result.video_id)
  
  // Fetch the video URL asynchronously (supports Convex storage)
  await loadVideoUrl(result.video_id)
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
  
  // Clear video context for keypoints operations
  setCurrentVideoId(null)
  
  // Clear speed cache for clean start
  if (previousVideoId) {
    clearSpeedCache(previousVideoId)
  }
}

function dismissError() {
  errorMessage.value = ''
}

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

// Function to perform health check
async function performHealthCheck() {
  isCheckingHealth.value = true
  try {
    const details = await getApiHealthDetails()
    healthDetails.value = details
    isApiConnected.value = details !== null && details.status === 'healthy'
  } catch {
    isApiConnected.value = false
    healthDetails.value = null
  } finally {
    isCheckingHealth.value = false
  }
}

// Function to manually refresh health check
async function refreshHealthStatus() {
  await performHealthCheck()
}

// Persist session state on changes
watch([currentState, uploadedVideo], ([state, video]) => {
  saveSessionState(state, video)
}, { deep: true })

onMounted(async () => {
  // Initial health check
  await performHealthCheck()

  // Setup periodic health checks every 10 seconds
  healthCheckInterval = setInterval(performHealthCheck, 10000)

  // Setup resize observer after mount
  setupResizeObserver()

  // Handle restored session state after sleep/reload
  if (currentState.value !== 'upload' && uploadedVideo.value) {
    if (currentState.value === 'court-setup') {
      // Court setup state is ephemeral — go to analyzing
      currentState.value = 'analyzing'
    }
    if (currentState.value === 'results') {
      // Results aren't persisted — the AnalysisProgress component
      // needs to re-fetch them, so restart from analyzing
      currentState.value = 'analyzing'
    }
  } else if (currentState.value !== 'upload') {
    // No video data to resume — reset
    currentState.value = 'upload'
  }
})

onUnmounted(() => {
  // Cleanup health check interval
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval)
    healthCheckInterval = null
  }
  
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
          <button class="alpha-badge" @click="showChangelogModal = true">
            alpha v1.7
          </button>
        </div>

        <nav class="nav">
          <button class="theme-toggle" @click="toggleTheme" :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'">
            <svg v-if="isDark" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/>
              <line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/>
              <line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          </button>
          <button
            class="api-status"
            :class="{ connected: isApiConnected, checking: isCheckingHealth }"
            @click="showHealthModal = true"
            :title="isApiConnected ? 'Click for details' : 'Click to see backend status'"
          >
            <span class="status-dot" :class="{ pulse: isCheckingHealth }" />
            {{ isApiConnected ? 'API Connected' : 'API Disconnected' }}
          </button>
        </nav>
      </div>
    </header>

    <!-- Changelog Modal -->
    <Transition name="fade">
      <div v-if="showChangelogModal" class="modal-overlay" @click.self="showChangelogModal = false">
        <div class="changelog-modal">
          <div class="changelog-header">
            <h2>📋 Changelog</h2>
            <button class="modal-close-btn" @click="showChangelogModal = false">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div class="changelog-content">
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.7-alpha</span>
                <span class="version-date">March 26, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Replaced gradient-based rally detection with shot-gap approach using shuttle direction reversals</li>
                <li> - Rally detection now runs client-side with auto-pause between rallies and hover tooltips</li>
                <li> - Improved shot detection: removed unreliable deceleration method, tightened pose confidence thresholds</li>
                <li> - Upgraded pose model from YOLOv26n to YOLOv26m for better far-player accuracy</li>
                <li> - Shuttle court ROI now extends to top of frame for clears and lobs</li>
                <li> - TrackNet streaming frame extraction to avoid OOM on large videos</li>
                <li> - NaN guards on homography transforms and hit marker cleanup on backward seek</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.6-alpha</span>
                <span class="version-date">March 10, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - TrackNetV3 shuttle tracking for accurate shuttlecock trajectory detection across the full video</li>
                <li> - Rally timeline visualization showing detected rallies with click-to-seek navigation</li>
                <li> - Gradient-based rally segmentation from shuttle trajectory data</li>
                <li> - Toggle to show/hide player skeleton overlay and pose estimation</li>
                <li> - Rally detection from shuttle direction changes (shot-gap heuristic)</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.5-alpha</span>
                <span class="version-date">March 3, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Shot placement heatmap now uses homography to map to real court coordinates instead of raw video pixels</li>
                <li> - Heatmap uses fixed court grid (6x8) aligned to actual court dimensions (6.1m x 13.4m)</li>
                <li> - Shots without shuttle data now included using player position as fallback</li>
                <li> - Recovery analysis measures distance from court center in meters instead of pixel averages</li>
                <li> - Recovery quality now factors in both time (60%) and position accuracy (40%)</li>
                <li> - Reaction time filters out anticipatory movement for more accurate measurements</li>
                <li> - Removed Pressure Index and Fatigue Detection due to insufficient data accuracy</li>
                <li> - Extracted shared homography utilities for consistent coordinate transforms</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.4-alpha</span>
                <span class="version-date">March 3, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Accurate speed measurement using 12-point court calibration with perspective-corrected homography</li>
                <li> - Fixed skeleton-video synchronization using actual video timestamps</li>
                <li> - Improved speed filtering to capture fast movements like lunges and recoveries</li>
                <li> - Thicker skeleton overlay lines for better visibility</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.3-alpha</span>
                <span class="version-date">March 2, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Add video overlay export functionality</li>
                <li> - Removal of inaccurate metrics</li>
                <li> - Add light mode</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.2-alpha</span>
                <span class="version-date">February 7, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Skeleton overlay movement smoothening</li>
                <li> - Minicourt circle movement smoothening</li>
                <li> - Added indicator when no player assignement made</li>
                <li> - Added previous coordinates path in minicourt</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.1-alpha</span>
                <span class="version-date">February 1, 2026</span>
              </div>
              <ul class="changelog-list">
                <li> - Added Convex backend for video storage</li>
                <li> - Removed all local backend processing files</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
            <div class="changelog-entry">
              <div class="changelog-version">
                <span class="version-tag">v1.0-alpha</span>
                <span class="version-date">January 31, 2026</span>
              </div>
              <h3>🎉 Initial Alpha Launch</h3>
              <p class="changelog-description">
                Welcome to SHUTTL. — your AI-powered badminton analysis companion!
              </p>
              <ul class="changelog-list">
                <li>🏸 <strong>Player Detection:</strong> AI-powered player tracking with skeleton overlay</li>
                <li>🎯 <strong>Pose Estimation:</strong> Real-time body position analysis</li>
                <li>⚡ <strong>Speed Tracking:</strong> Player movement and shuttlecock speed metrics</li>
                <li>📊 <strong>Court Mapping:</strong> Interactive mini-court visualization with 12-point calibration</li>
                <li>🔥 <strong>Position Heatmaps:</strong> Visual representation of player court coverage</li>
                <li>📈 <strong>Performance Dashboard:</strong> Comprehensive match statistics and insights</li>
              </ul>
              <p class="changelog-note">
                <em>This is an early alpha release. We'd love your feedback as we continue to improve!</em>
              </p>
            </div>
          </div>
        </div>
      </div>
    </Transition>

    <!-- Health Status Modal -->
    <Transition name="fade">
      <div v-if="showHealthModal" class="modal-overlay" @click.self="showHealthModal = false">
        <div class="health-modal">
          <div class="health-header">
            <h2>🔌 Backend Status</h2>
            <button class="modal-close-btn" @click="showHealthModal = false">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div class="health-content">
            <!-- Connection Status -->
            <div class="health-status-banner" :class="{ connected: isApiConnected, disconnected: !isApiConnected }">
              <span class="status-icon">{{ isApiConnected ? '✅' : '❌' }}</span>
              <span class="status-text">{{ isApiConnected ? 'Backend is Online' : 'Backend is Offline' }}</span>
            </div>

            <!-- API URL Info -->
            <div class="health-section">
              <h3>📡 Connection</h3>
              <div class="health-info-row">
                <span class="label">API URL:</span>
                <code class="value">{{ getApiBaseUrl() }}</code>
              </div>
              <div class="health-info-row">
                <span class="label">Last Check:</span>
                <span class="value">{{ healthDetails?.timestamp ? new Date(healthDetails.timestamp * 1000).toLocaleTimeString() : 'Never' }}</span>
              </div>
            </div>

            <!-- If connected, show detailed info -->
            <template v-if="isApiConnected && healthDetails">
              <!-- Backend Type Badge -->
              <div class="backend-badge" :class="healthDetails.backend || 'local'">
                {{ healthDetails.backend === 'convex' ? '☁️ Convex Cloud' : '💻 Local Backend' }}
              </div>

              <!-- Components Status -->
              <div class="health-section">
                <h3>🧩 Components</h3>
                <div class="components-grid">
                  <!-- Convex-specific components -->
                  <template v-if="healthDetails.backend === 'convex'">
                    <div class="component-item" :class="{ active: healthDetails.components.convex === 'connected' }">
                      <span class="component-icon">☁️</span>
                      <span class="component-name">Convex</span>
                      <span class="component-status">{{ healthDetails.components.convex || 'unknown' }}</span>
                    </div>
                    <div class="component-item" :class="{ active: healthDetails.components.database === 'connected' }">
                      <span class="component-icon">🗃️</span>
                      <span class="component-name">Database</span>
                      <span class="component-status">{{ healthDetails.components.database || 'unknown' }}</span>
                    </div>
                    <div class="component-item" :class="{ active: healthDetails.components.modal_inference === 'configured' }">
                      <span class="component-icon">⚡</span>
                      <span class="component-name">Modal GPU</span>
                      <span class="component-status">{{ healthDetails.components.modal_inference || 'unknown' }}</span>
                    </div>
                  </template>
                  <!-- Local backend components -->
                  <template v-else>
                    <div class="component-item" :class="{ active: healthDetails.components.pose_model === 'loaded' }">
                      <span class="component-icon">🦴</span>
                      <span class="component-name">Pose Model</span>
                      <span class="component-status">{{ healthDetails.components.pose_model || 'unknown' }}</span>
                    </div>
                    <div class="component-item" :class="{ active: healthDetails.components.court_detector === 'loaded' }">
                      <span class="component-icon">🏸</span>
                      <span class="component-name">Court Detector</span>
                      <span class="component-status">{{ healthDetails.components.court_detector || 'unknown' }}</span>
                    </div>
                    <div class="component-item" :class="{ active: healthDetails.components.multi_model_detector === 'loaded' }">
                      <span class="component-icon">🎯</span>
                      <span class="component-name">Multi-Model Detector</span>
                      <span class="component-status">{{ healthDetails.components.multi_model_detector || 'unknown' }}</span>
                    </div>
                    <div class="component-item" :class="{ active: healthDetails.components.modal_inference === 'enabled' }">
                      <span class="component-icon">☁️</span>
                      <span class="component-name">Modal GPU</span>
                      <span class="component-status">{{ healthDetails.components.modal_inference || 'unknown' }}</span>
                    </div>
                  </template>
                </div>
              </div>

              <!-- System Resources (local backend only) -->
              <div v-if="healthDetails.system" class="health-section">
                <h3>💻 System Resources</h3>
                <div class="resource-bars">
                  <div class="resource-item">
                    <span class="resource-label">CPU</span>
                    <div class="resource-bar">
                      <div class="resource-fill" :style="{ width: healthDetails.system.cpu_percent + '%' }" :class="{ warning: healthDetails.system.cpu_percent > 80 }"></div>
                    </div>
                    <span class="resource-value">{{ healthDetails.system.cpu_percent.toFixed(1) }}%</span>
                  </div>
                  <div class="resource-item">
                    <span class="resource-label">Memory</span>
                    <div class="resource-bar">
                      <div class="resource-fill" :style="{ width: healthDetails.system.memory_percent + '%' }" :class="{ warning: healthDetails.system.memory_percent > 80 }"></div>
                    </div>
                    <span class="resource-value">{{ healthDetails.system.memory_percent.toFixed(1) }}%</span>
                  </div>
                </div>
                <div class="health-info-row">
                  <span class="label">Available Memory:</span>
                  <span class="value">{{ healthDetails.system.memory_available_gb }} GB</span>
                </div>
              </div>

              <!-- Video Stats (Convex only) -->
              <div v-if="healthDetails.stats" class="health-section">
                <h3>📊 Video Stats</h3>
                <div class="health-info-row">
                  <span class="label">Total Videos:</span>
                  <span class="value">{{ healthDetails.stats.total_videos }}</span>
                </div>
                <div class="health-info-row">
                  <span class="label">Currently Processing:</span>
                  <span class="value">{{ healthDetails.stats.processing }}</span>
                </div>
              </div>

              <!-- Active Analyses (local backend) -->
              <div v-if="healthDetails.active_analyses !== undefined" class="health-section">
                <h3>📊 Activity</h3>
                <div class="health-info-row">
                  <span class="label">Active Analyses:</span>
                  <span class="value">{{ healthDetails.active_analyses }}</span>
                </div>
              </div>
            </template>

            <!-- If disconnected, show troubleshooting -->
            <template v-else>
              <div class="health-section">
                <h3>🔧 Troubleshooting</h3>
                <div class="troubleshoot-tips">
                  <template v-if="isUsingConvex()">
                    <p>The Convex backend is not responding. Try these steps:</p>
                    <ol>
                      <li>Check your internet connection</li>
                      <li>Verify <code>VITE_CONVEX_URL</code> is set correctly in <code>.env</code></li>
                      <li>Make sure Convex is deployed:
                        <code>npx convex deploy</code>
                      </li>
                      <li>Check the Convex dashboard for any errors</li>
                    </ol>
                  </template>
                  <template v-else>
                    <p>The backend server is not responding. Try these steps:</p>
                    <ol>
                      <li>Make sure the backend is running:
                        <code>cd backend && python main.py</code>
                      </li>
                      <li>Check if the API URL is correct in your <code>.env</code> file:
                        <code>VITE_API_URL=http://localhost:8000</code>
                      </li>
                      <li>Verify no firewall is blocking the connection</li>
                      <li>Check the backend terminal for any error messages</li>
                    </ol>
                  </template>
                </div>
              </div>
            </template>

            <!-- Refresh Button -->
            <div class="health-actions">
              <button class="refresh-btn" @click="refreshHealthStatus" :disabled="isCheckingHealth">
                <span v-if="isCheckingHealth" class="spinner"></span>
                <span v-else>🔄</span>
                {{ isCheckingHealth ? 'Checking...' : 'Refresh Status' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>

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
              player movement patterns, speed metrics, and court coverage by AI.
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
              <p>Calculate player movement speed and court coverage</p>
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

        <!-- Court Setup State (NEW: Before analyzing) -->
        <div v-else-if="currentState === 'court-setup' && uploadedVideo" key="court-setup" class="content-section">
          <CourtSetup
            :video-id="uploadedVideo.video_id"
            :filename="uploadedVideo.filename"
            @complete="handleCourtSetupComplete"
            @skip="handleCourtSetupSkip"
            @error="handleCourtSetupError"
          />
        </div>

        <!-- Analyzing State -->
        <div v-else-if="currentState === 'analyzing' && uploadedVideo" key="analyzing" class="content-section">
          <AnalysisProgress
            :video-id="uploadedVideo.video_id"
            :filename="uploadedVideo.filename"
            :analysis-mode="analysisMode"
            @complete="handleAnalysisComplete"
            @error="handleAnalysisError"
            @cancel="handleAnalysisCancel"
          />
        </div>

        <!-- Results State -->
        <div v-else-if="currentState === 'results' && analysisResult" key="results" class="content-section results-view">
          <PlayerLabelsProvider :video-id="analysisResult.video_id">
          <div class="results-header">
            <button class="back-btn" @click="startNewAnalysis">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="19" y1="12" x2="5" y2="12" />
                <polyline points="12 19 5 12 12 5" />
              </svg>
              New Analysis
            </button>

            <div v-if="analysisMode !== 'rally_only'" class="results-header-right">
              <!-- Export Video Button -->
              <button
                class="export-btn"
                :disabled="videoPlayerRef?.isExporting"
                @click="videoPlayerRef?.startExport()"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>Export Video</span>
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
          </div>

          <!-- Export Progress Overlay -->
          <Transition name="fade">
            <div v-if="videoPlayerRef?.isExporting" class="export-progress-overlay">
              <div class="export-progress-content">
                <div class="export-progress-bar-track">
                  <div class="export-progress-bar-fill" :style="{ width: (videoPlayerRef?.exportProgress ?? 0) + '%' }"></div>
                </div>
                <span class="export-progress-text">Exporting... {{ videoPlayerRef?.exportProgress ?? 0 }}%</span>
                <button class="export-cancel-btn" @click="videoPlayerRef?.cancelExport()">Cancel</button>
              </div>
            </div>
          </Transition>

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
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showHeatmap" />
                        <span class="toggle-slider heatmap-toggle" />
                      </label>
                      <span>Position Heatmap</span>
                      <span v-if="selectedRallyId !== null && showHeatmap" class="heatmap-rally-badge">Rally #{{ selectedRallyId }}</span>
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
                        <input type="checkbox" v-model="showShuttleTracking" :disabled="!showBoundingBoxes" />
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
                        <input type="checkbox" v-model="showPlayerTrails" :disabled="!showMiniCourt" />
                        <span class="toggle-slider trails-toggle" />
                      </label>
                      <span>Player Trails</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showHitMarkers" :disabled="!showMiniCourt" />
                        <span class="toggle-slider hitmarkers-toggle" />
                      </label>
                      <span>Hit Markers</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showSpeedGraph" />
                        <span class="toggle-slider speedgraph-toggle" />
                      </label>
                      <span>Speed Graph</span>
                    </div>
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="showShotSpeedList" />
                        <span class="toggle-slider shotspeed-toggle" />
                      </label>
                      <span>Shot Speed</span>
                    </div>
                  </div>
                </div>

                <!-- Playback Section -->
                <div class="settings-section" v-if="detectedRallies.length > 0">
                  <h4 class="settings-section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    Playback
                  </h4>
                  <div class="settings-grid">
                    <div class="toggle-item">
                      <label class="toggle">
                        <input type="checkbox" v-model="pauseBetweenRallies" />
                        <span class="toggle-slider rally-pause-toggle" />
                      </label>
                      <span>Pause Between Rallies</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Transition>

          <div class="results-content" :class="{ 'with-minicourt': showMiniCourt }">
            <div class="mode-selector view-mode-selector">
              <span class="mode-label">Playback View</span>
              <div class="mode-options">
                <button
                  type="button"
                  class="mode-option"
                  :class="{ active: viewMode === 'video' }"
                  @click="viewMode = 'video'"
                >
                  <span class="mode-title">Video</span>
                  <span class="mode-desc">Real footage with skeleton overlay</span>
                </button>
                <button
                  type="button"
                  class="mode-option"
                  :class="{ active: viewMode === 'court' }"
                  :disabled="manualCourtKeypoints === null"
                  :title="manualCourtKeypoints === null ? 'Set manual court keypoints to enable' : ''"
                  @click="viewMode = 'court'"
                >
                  <span class="mode-title">Court</span>
                  <span class="mode-desc">Synthetic court with skeleton + shuttle trail</span>
                </button>
              </div>
            </div>
            <div class="video-with-minicourt">
              <div ref="videoSectionRef" class="video-section">
                <VideoPlayer
                  ref="videoPlayerRef"
                  :video-url="videoUrl"
                  :skeleton-data="analysisMode !== 'rally_only' ? analysisResult.skeleton_data : []"

                  :show-skeleton="analysisMode !== 'rally_only' && showSkeleton"
                  :show-bounding-boxes="analysisMode !== 'rally_only' && showBoundingBoxes"
                  :show-players="analysisMode !== 'rally_only' && showPlayers"
                  :show-shuttles="analysisMode !== 'rally_only' && showShuttleTracking"
                  :show-rackets="analysisMode !== 'rally_only' && showRackets"
                  :show-pose-overlay="analysisMode !== 'rally_only' && showPoseOverlay"
                  :pose-source="poseSource"
                  :show-heatmap="analysisMode !== 'rally_only' && showHeatmap"
                  :heatmap-frame-range="heatmapFrameRange"
                  :show-shuttle-tracking="analysisMode !== 'rally_only' && showShuttleTracking"
                  :court-keypoints="courtCornersForMiniCourt"
                  :view-mode="viewMode"
                  :manual-court-keypoints="manualCourtKeypoints"
                  :video-fps="analysisResult?.fps ?? 30"
                  :shot-summary-segment="currentShotSegment"
                  :shot-summary-countdown="shotPauseCountdown"
                  @time-update="handleTimeUpdate"
                  @frame-update="handleFrameUpdate"
                  @play="handleVideoPlay"
                />
              </div>

              <!-- Mini Court Panel (hidden in rally-only mode) -->
              <Transition name="slide-fade">
                <div v-if="analysisMode !== 'rally_only' && showMiniCourt" class="minicourt-section">
                  <MiniCourt
                    :court-corners="courtCornersForMiniCourt"
                    :players="videoPlaybackStarted ? currentPlayers : []"
                    :shuttle-position="videoPlaybackStarted ? currentShuttlePosition : null"
                    :width="240"
                    :height="videoContainerHeight"
                    :show-grid="true"
                    :show-labels="true"
                    :show-shuttle="showShuttleTracking"
                    :show-trails="showPlayerTrails"
                    :show-hit-markers="showHitMarkers"
                    :skeleton-data="analysisResult?.skeleton_data"
                    :current-frame="currentFrame"
                    :max-trail-length="150"
                  />
                </div>
              </Transition>

              <!-- Rally Timeline — directly below video -->
              <RallyTimeline
                v-if="(detectedRallies.length > 0 || backendRallies.length > 0) && analysisResult"
                :rallies="detectedRallies"
                :backend-rallies="backendRallies"
                :duration="analysisResult.duration"
                :rally-source="rallySource"
                :current-time="currentVideoTime"
                :pause-between-rallies="pauseBetweenRallies"
                :rally-pause-countdown="rallyPauseCountdown"
                :paused-after-rally-id="pausedAfterRallyId"
                :rally-speed-stats="rallySpeedStats"
                :pause-between-shots="pauseBetweenShots"
                :shot-pause-duration-sec="shotPauseDurationSec"
                @seek-to-time="handleRallySeek"
                @update:pause-between-rallies="pauseBetweenRallies = $event"
                @update:pause-between-shots="pauseBetweenShots = $event"
                @update:shot-pause-duration-sec="shotPauseDurationSec = $event"
                @skip-to-next-rally="skipToNextRally"
                @resume-from-pause="resumeFromRallyPause"
                @select-rally="handleRallySelect"
              />
            </div>

            <!-- Speed Graph Panel (hidden in rally-only mode) -->
            <Transition name="slide-fade">
              <div v-if="analysisMode !== 'rally_only' && showSpeedGraph && analysisResult" class="speedgraph-section">
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

            <!-- Shot Speed Analysis Panel (hidden in rally-only mode) -->
            <Transition name="slide-fade">
              <div v-if="analysisMode !== 'rally_only' && showShotSpeedList && analysisResult" class="shotspeed-section">
                <ShotSpeedList
                  :skeleton-data="analysisResult.skeleton_data"
                  :fps="analysisResult.fps"
                  :visible="showShotSpeedList"
                  :court-keypoints-set="manualCourtKeypoints !== null"
                  :speed-calculated="speedCalculationTriggered && !isSpeedCalculating"
                  @seek-to-segment="handleSeekToSegment"
                  @close="showShotSpeedList = false"
                />
              </div>
            </Transition>

            <div v-if="analysisMode !== 'rally_only'" class="dashboard-section">
              <ResultsDashboard
                :result="analysisResult"
                :manual-keypoints-set="manualCourtKeypoints !== null"
                :zone-recalculation-trigger="zoneRecalculationTrigger"
              />
            </div>

            <div v-if="analysisMode !== 'rally_only'" class="dashboard-section">
              <AdvancedAnalytics
                :result="analysisResult"
                :current-frame="currentFrame"
                :court-keypoints="courtCornersForMiniCourt"
                :camera-angle="cameraAngle"
                @seek-to-time="handleRallySeek"
              />
            </div>
          </div>
          </PlayerLabelsProvider>
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
  background: var(--color-bg);
  color: var(--color-text);
  min-height: 100vh;
}

a {
  color: var(--color-accent);
  text-decoration: none;
  transition: color 0.2s ease;
}

a:hover {
  color: var(--color-accent-hover);
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
  border-bottom: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--color-bg);
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
  color: var(--color-accent);
}

.logo h1 {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-text-heading);
}

.nav {
  display: flex;
  align-items: center;
  gap: 16px;
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  border: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.theme-toggle:hover {
  background: var(--color-bg-hover);
  color: var(--color-text);
  border-color: var(--color-border-hover);
}

.api-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  font-size: 0.75rem;
  color: var(--color-error);
  cursor: pointer;
  transition: all 0.2s ease;
}

.api-status:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-hover);
}

.api-status.connected {
  background: var(--color-bg-tertiary);
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.api-status.connected:hover {
  border-color: var(--color-accent-hover);
}

.api-status.checking {
  opacity: 0.8;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  transition: all 0.3s ease;
}

.status-dot.pulse {
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

/* Health Modal Styles */
.health-modal {
  background: var(--color-bg-modal);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.health-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid var(--color-border);
}

.health-header h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

.health-content {
  padding: 20px 24px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.health-status-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border-radius: 0;
  font-weight: 500;
}

.health-status-banner.connected {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid var(--color-accent);
  color: var(--color-accent);
}

.health-status-banner.disconnected {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--color-error);
  color: var(--color-error);
}

.backend-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 0;
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 8px;
}

.backend-badge.convex {
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid #6366f1;
  color: #818cf8;
}

.backend-badge.local {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid var(--color-accent);
  color: var(--color-accent);
}

.status-icon {
  font-size: 1.5rem;
}

.status-text {
  font-size: 1rem;
}

.health-section {
  border: 1px solid var(--color-border);
  border-radius: 0;
  padding: 16px;
}

.health-section h3 {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0 0 12px 0;
  color: var(--color-text-secondary);
}

.health-info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-bg-tertiary);
}

.health-info-row:last-child {
  border-bottom: none;
}

.health-info-row .label {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

.health-info-row .value {
  font-weight: 500;
  font-size: 0.875rem;
}

.health-info-row code {
  background: var(--color-bg-tertiary);
  padding: 4px 8px;
  border-radius: 0;
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  font-size: 0.75rem;
  word-break: break-all;
}

.components-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.component-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 12px 8px;
  background: var(--color-bg-input);
  border: 1px solid var(--color-border);
  border-radius: 0;
  text-align: center;
}

.component-item.active {
  border-color: var(--color-accent);
  background: rgba(34, 197, 94, 0.05);
}

.component-icon {
  font-size: 1.25rem;
}

.component-name {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.component-status {
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
}

.component-item.active .component-status {
  color: var(--color-accent);
}

.resource-bars {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 12px;
}

.resource-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.resource-label {
  width: 60px;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.resource-bar {
  flex: 1;
  height: 8px;
  background: var(--color-bg-tertiary);
  border-radius: 0;
  overflow: hidden;
}

.resource-fill {
  height: 100%;
  background: var(--color-accent);
  transition: width 0.3s ease;
}

.resource-fill.warning {
  background: var(--color-warning);
}

.resource-value {
  width: 50px;
  text-align: right;
  font-size: 0.75rem;
  font-weight: 500;
}

.troubleshoot-tips {
  font-size: 0.875rem;
  line-height: 1.6;
}

.troubleshoot-tips p {
  margin: 0 0 12px 0;
  color: var(--color-text-secondary);
}

.troubleshoot-tips ol {
  margin: 0;
  padding-left: 20px;
}

.troubleshoot-tips li {
  margin-bottom: 12px;
}

.troubleshoot-tips code {
  display: block;
  margin-top: 8px;
  background: var(--color-bg-tertiary);
  padding: 8px 12px;
  border-radius: 0;
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  font-size: 0.75rem;
  word-break: break-all;
}

.health-actions {
  display: flex;
  justify-content: center;
  padding-top: 8px;
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-heading);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.refresh-btn:hover:not(:disabled) {
  background: var(--color-bg-hover);
  border-color: var(--color-border-hover);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--color-border-secondary);
  border-top-color: var(--color-text-heading);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.nav-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 0;
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
}

.nav-link:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-text-heading);
}

.nav-link svg {
  width: 20px;
  height: 20px;
}

/* Error Banner */
.error-banner {
  background: #1a0000;
  border-bottom: 1px solid var(--color-error);
}

.error-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--color-error);
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
  color: var(--color-error);
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
  color: var(--color-text-heading);
}

.hero p {
  max-width: 600px;
  margin: 0 auto;
  color: var(--color-text-secondary);
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
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0;
  text-align: center;
  transition: all 0.2s ease;
}

.feature:hover {
  border-color: var(--color-accent);
}

.feature-icon {
  width: 56px;
  height: 56px;
  margin: 0 auto 16px;
  padding: 14px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-accent);
}

.feature-icon svg {
  width: 100%;
  height: 100%;
}

.feature h3 {
  color: var(--color-text-heading);
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 8px;
}

.feature p {
  color: var(--color-text-tertiary);
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
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.back-btn:hover {
  background: var(--color-bg-tertiary);
  border-color: var(--color-accent);
  color: var(--color-text-heading);
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
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-heading);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-toggle-btn:hover {
  background: var(--color-bg-tertiary);
  border-color: var(--color-accent);
}

.settings-toggle-btn.active {
  background: var(--color-bg-tertiary);
  border-color: var(--color-accent);
}

.settings-toggle-btn svg:first-child {
  width: 18px;
  height: 18px;
  color: var(--color-accent);
}

.settings-toggle-btn .chevron {
  width: 16px;
  height: 16px;
  color: var(--color-text-secondary);
  transition: transform 0.3s ease;
}

.settings-toggle-btn .chevron.rotated {
  transform: rotate(180deg);
}

/* Results Header Right Group */
.results-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Export Video Button */
.export-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-heading);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.export-btn:hover:not(:disabled) {
  background: var(--color-bg-tertiary);
  border-color: var(--color-accent);
}

.export-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.export-btn svg {
  width: 18px;
  height: 18px;
  color: var(--color-accent);
}

/* Export Progress Overlay */
.export-progress-overlay {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-secondary);
  padding: 16px 24px;
  margin-bottom: 16px;
}

.export-progress-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.export-progress-bar-track {
  flex: 1;
  height: 6px;
  background: var(--color-border);
  border-radius: 3px;
  overflow: hidden;
}

.export-progress-bar-fill {
  height: 100%;
  background: var(--color-accent);
  transition: width 0.3s ease;
}

.export-progress-text {
  font-size: 0.8rem;
  color: #aaa;
  white-space: nowrap;
  min-width: 110px;
}

.export-cancel-btn {
  padding: 6px 14px;
  background: transparent;
  border: 1px solid #555;
  border-radius: 0;
  color: #aaa;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.export-cancel-btn:hover {
  border-color: var(--color-error);
  color: var(--color-error);
}

/* Settings Panel */
.settings-panel {
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
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
  color: var(--color-text-heading);
  font-size: 0.9rem;
  font-weight: 600;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border);
}

.settings-section-title svg {
  width: 18px;
  height: 18px;
  color: var(--color-accent);
}

.settings-section-hint {
  color: var(--color-text-tertiary);
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
  color: var(--color-text-secondary);
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
  background-color: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
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
  background-color: var(--color-border-hover);
  border-radius: 0;
  transition: 0.3s;
}

.toggle input:checked + .toggle-slider {
  background: var(--color-accent);
  border-color: var(--color-accent);
}

.toggle input:checked + .toggle-slider.player-toggle {
  background: var(--color-accent);
  border-color: var(--color-accent);
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
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  margin-right: 8px;
}

.pose-source-select {
  padding: 6px 12px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  border-radius: 0;
  color: var(--color-text-heading);
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
  background: var(--color-bg-tertiary);
  color: var(--color-text-heading);
}

/* Court toggle removed - automatic detection disabled, using manual keypoints only */

.toggle input:checked + .toggle-slider.heatmap-toggle {
  background: #f97316;
  border-color: #f97316;
}

.heatmap-rally-badge {
  font-size: 0.65rem;
  padding: 1px 6px;
  background: rgba(249, 115, 22, 0.2);
  border: 1px solid rgba(249, 115, 22, 0.5);
  color: #f97316;
  font-weight: 600;
}

.toggle input:checked + .toggle-slider.shuttle-tracking-toggle {
  background: #00e5ff;
  border-color: #00e5ff;
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
  color: var(--color-accent);
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
  flex-wrap: wrap;
}

.video-with-minicourt {
  width: 100%;
}

.video-with-minicourt > :deep(.rally-tl) {
  flex-basis: 100%;
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
  background: var(--color-accent);
  border-color: var(--color-accent);
}

/* Player trails toggle style */
.toggle input:checked + .toggle-slider.trails-toggle {
  background: var(--color-warning);
  border-color: var(--color-warning);
}

/* Hit markers toggle style */
.toggle input:checked + .toggle-slider.hitmarkers-toggle {
  background: #FF2952;
  border-color: #FF2952;
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

/* Shot speed toggle style */
.toggle input:checked + .toggle-slider.shotspeed-toggle {
  background: #F59E0B;
  border-color: #F59E0B;
}

/* Shot speed analysis section */
.shotspeed-section {
  width: 100%;
  margin-top: 16px;
  max-height: 500px;
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
  border-top: 1px solid var(--color-border);
  color: var(--color-text-tertiary);
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

/* Alpha Badge */
.alpha-badge {
  background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-dark) 100%);
  color: #000;
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 8px;
  border: none;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  transition: all 0.2s ease;
  margin-left: 4px;
}

.alpha-badge:hover {
  background: linear-gradient(135deg, var(--color-accent-hover) 0%, var(--color-accent) 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
}

/* Changelog Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.changelog-modal {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-secondary);
  max-width: 520px;
  width: 100%;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.changelog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid var(--color-border-secondary);
}

.changelog-header h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--color-text-heading);
  margin: 0;
}

.modal-close-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1px solid var(--color-border-hover);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.2s ease;
}

.modal-close-btn:hover {
  background: var(--color-border-secondary);
  color: var(--color-text-heading);
  border-color: #555;
}

.modal-close-btn svg {
  width: 16px;
  height: 16px;
}

.changelog-content {
  padding: 24px;
  overflow-y: auto;
}

.changelog-entry {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.changelog-version {
  display: flex;
  align-items: center;
  gap: 12px;
}

.version-tag {
  background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-dark) 100%);
  color: #000;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 4px 10px;
}

.version-date {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

.changelog-entry h3 {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text-heading);
  margin: 0;
}

.changelog-description {
  color: #ccc;
  font-size: 0.95rem;
  line-height: 1.6;
  margin: 0;
}

.changelog-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.changelog-list li {
  color: #bbb;
  font-size: 0.875rem;
  line-height: 1.5;
  padding-left: 4px;
}

.changelog-list li strong {
  color: var(--color-text-heading);
}

.changelog-note {
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  line-height: 1.5;
  margin: 8px 0 0 0;
  padding-top: 16px;
  border-top: 1px solid var(--color-border-secondary);
}

/* Playback view-mode selector (copied from VideoUpload.vue:615-680 scoped styles) */
.mode-selector {
  margin-bottom: 20px;
}

.mode-label {
  display: block;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  font-weight: 500;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.mode-options {
  display: flex;
  gap: 8px;
}

.mode-option {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 16px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-border-secondary);
  border-radius: 0;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s ease;
}

.mode-option:hover {
  border-color: var(--color-text-tertiary);
}

.mode-option.active {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
}

.mode-option.active .mode-title {
  color: var(--color-accent);
}

.mode-option:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

</style>
