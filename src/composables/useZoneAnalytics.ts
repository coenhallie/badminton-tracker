/**
 * Client-side zone analytics computation from skeleton_data stored in Supabase.
 *
 * Returns the same `RecalculatedZoneAnalyticsResponse` shape the rest of the
 * app already consumes.
 *
 * Note: avg_distance_to_net_m requires homography from manual court keypoints
 * which is not computed here (returns 0 / undefined for this field; the UI
 * hides the row when avg_distance_to_net_m <= 0).
 */
import { supabase } from '@/lib/supabase'
import type { RecalculatedZoneAnalyticsResponse, PlayerZoneData } from '@/services/api'

interface SkeletonPlayer {
  player_id: number
  center?: { x: number; y: number }
  position?: { x: number; y: number }
}

interface SkeletonFrameData {
  frame: number
  players: SkeletonPlayer[]
}

interface ResultsJson {
  skeleton_data?: SkeletonFrameData[]
  video_width?: number
  video_height?: number
}

interface ManualCourtKeypointsRow {
  top_left?: number[]
  bottom_left?: number[]
  [k: string]: unknown
}

async function downloadResultsAndKeypoints(
  videoId: string,
): Promise<{ results: ResultsJson; manualKeypoints: ManualCourtKeypointsRow | null }> {
  const { data: row, error } = await supabase
    .from('videos')
    .select('results_storage_path, manual_court_keypoints')
    .eq('id', videoId)
    .single()
  if (error || !row?.results_storage_path) {
    throw error ?? new Error('Results not available')
  }
  const { data: signed, error: e2 } = await supabase
    .storage.from('results')
    .createSignedUrl(row.results_storage_path, 3600)
  if (e2 || !signed) throw e2 ?? new Error('Could not sign results URL')
  const res = await fetch(signed.signedUrl)
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`)
  const results = (await res.json()) as ResultsJson
  return { results, manualKeypoints: (row.manual_court_keypoints as ManualCourtKeypointsRow | null) ?? null }
}

export async function computeZoneAnalytics(
  videoId: string,
): Promise<RecalculatedZoneAnalyticsResponse> {
  const { results, manualKeypoints } = await downloadResultsAndKeypoints(videoId)
  const skeletonData = results.skeleton_data ?? []
  const videoWidth = results.video_width ?? 1920
  const videoHeight = results.video_height ?? 1080

  // Per-player accumulators (zone counts + total observations)
  const acc: Record<
    string,
    { zones: { front: number; mid: number; back: number; left: number; center: number; right: number }; total: number }
  > = {}

  for (const frame of skeletonData) {
    for (const player of frame.players ?? []) {
      let x: number | null = null
      let y: number | null = null
      if (player.center) {
        x = player.center.x
        y = player.center.y
      } else if (player.position) {
        x = player.position.x
        y = player.position.y
      }
      if (x == null || y == null) continue

      const pid = player.player_id.toString()
      if (!acc[pid]) {
        acc[pid] = {
          zones: { front: 0, mid: 0, back: 0, left: 0, center: 0, right: 0 },
          total: 0,
        }
      }

      // Normalized position
      const nx = x / videoWidth
      const ny = y / videoHeight

      // Vertical zone (front/mid/back)
      if (ny < 0.33) acc[pid].zones.front++
      else if (ny < 0.67) acc[pid].zones.mid++
      else acc[pid].zones.back++

      // Horizontal zone (left/center/right)
      if (nx < 0.33) acc[pid].zones.left++
      else if (nx < 0.67) acc[pid].zones.center++
      else acc[pid].zones.right++

      acc[pid].total++
    }
  }

  // Convert to percentages and assemble PlayerZoneData
  const playerZoneAnalytics: Record<string, PlayerZoneData> = {}
  for (const [pid, data] of Object.entries(acc)) {
    const total = data.total || 1
    playerZoneAnalytics[pid] = {
      zone_coverage: {
        front: (data.zones.front / total) * 100,
        mid: (data.zones.mid / total) * 100,
        back: (data.zones.back / total) * 100,
        left: (data.zones.left / total) * 100,
        center: (data.zones.center / total) * 100,
        right: (data.zones.right / total) * 100,
      },
      // Distance-to-net requires homography from manual keypoints; leave 0 for now
      // (the UI hides the row when this is <= 0).
      avg_distance_to_net_m: 0,
      position_count: data.total,
    }
  }

  const hasManualKeypoints =
    !!manualKeypoints && Array.isArray(manualKeypoints.top_left) && Array.isArray(manualKeypoints.bottom_left)

  return {
    video_id: videoId,
    player_zone_analytics: playerZoneAnalytics,
    recalculated: true,
    manual_keypoints_used: hasManualKeypoints,
    total_skeleton_frames: skeletonData.length,
    status: 'success',
    message: 'ok',
  }
}
