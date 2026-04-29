/**
 * Client-side heatmap computation from skeleton_data stored in Supabase.
 *
 * Mirrors the algorithm previously hosted in convex/http.ts (`generateHeatmapFromSkeleton`)
 * so the response shape is identical to what callers already expect via `HeatmapData`.
 */
import { supabase } from '@/lib/supabase'
import type { HeatmapData } from '@/services/api'

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
  court_detection?: {
    court_corners?: number[][]
  }
}

async function downloadResultsJson(videoId: string): Promise<ResultsJson> {
  const { data: row, error } = await supabase
    .from('videos')
    .select('results_storage_path')
    .eq('id', videoId)
    .single()
  if (error || !row?.results_storage_path) {
    throw error ?? new Error('Results not available for this video')
  }
  const { data: signed, error: e2 } = await supabase
    .storage.from('results')
    .createSignedUrl(row.results_storage_path, 3600)
  if (e2 || !signed) throw e2 ?? new Error('Could not sign results URL')
  const res = await fetch(signed.signedUrl)
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`)
  return res.json() as Promise<ResultsJson>
}

export async function computeHeatmap(
  videoId: string,
  playerId?: number,
  heatmapWidth = 100,
  heatmapHeight = 100,
): Promise<HeatmapData> {
  const results = await downloadResultsJson(videoId)
  const skeletonData = results.skeleton_data ?? []
  const videoWidth = results.video_width ?? 1920
  const videoHeight = results.video_height ?? 1080

  // Initialize heatmaps
  const combined: number[][] = Array(heatmapHeight)
    .fill(null)
    .map(() => Array(heatmapWidth).fill(0))
  const byPlayer: Record<string, number[][]> = {}

  // Gaussian blur radius in heatmap cells
  const radius = 2

  for (const frame of skeletonData) {
    for (const player of frame.players ?? []) {
      // Filter by player ID if specified
      if (playerId !== undefined && player.player_id !== playerId) continue

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

      const hx = Math.floor((x / videoWidth) * heatmapWidth)
      const hy = Math.floor((y / videoHeight) * heatmapHeight)

      const pid = player.player_id.toString()
      if (!byPlayer[pid]) {
        byPlayer[pid] = Array(heatmapHeight)
          .fill(null)
          .map(() => Array(heatmapWidth).fill(0))
      }

      // Add Gaussian blur around the point
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = hx + dx
          const py = hy + dy
          if (px >= 0 && px < heatmapWidth && py >= 0 && py < heatmapHeight) {
            const dist = Math.sqrt(dx * dx + dy * dy)
            const weight = Math.exp(-(dist * dist) / (2 * radius * radius))
            const cRow = combined[py]!
            cRow[px] = (cRow[px] ?? 0) + weight
            const pRow = byPlayer[pid]![py]!
            pRow[px] = (pRow[px] ?? 0) + weight
          }
        }
      }
    }
  }

  // Normalize 0-255
  const flatCombined = combined.flat()
  const maxC = flatCombined.length ? Math.max(...flatCombined) : 0
  if (maxC > 0) {
    for (let yi = 0; yi < heatmapHeight; yi++) {
      const row = combined[yi]!
      for (let xi = 0; xi < heatmapWidth; xi++) {
        row[xi] = Math.round((row[xi]! / maxC) * 255)
      }
    }
  }
  for (const pid of Object.keys(byPlayer)) {
    const data = byPlayer[pid]!
    const flatP = data.flat()
    const maxP = flatP.length ? Math.max(...flatP) : 0
    if (maxP > 0) {
      for (let yi = 0; yi < heatmapHeight; yi++) {
        const row = data[yi]!
        for (let xi = 0; xi < heatmapWidth; xi++) {
          row[xi] = Math.round((row[xi]! / maxP) * 255)
        }
      }
    }
  }

  // Player position counts (across all frames, ignoring playerId filter)
  const playerPositionCounts: Record<string, number> = {}
  for (const frame of skeletonData) {
    for (const p of frame.players ?? []) {
      const pid = p.player_id.toString()
      playerPositionCounts[pid] = (playerPositionCounts[pid] ?? 0) + 1
    }
  }

  return {
    video_id: videoId,
    width: heatmapWidth,
    height: heatmapHeight,
    colormap: 'turbo',
    combined_heatmap: playerId === undefined ? combined : undefined,
    player_heatmaps:
      playerId === undefined
        ? byPlayer
        : { [playerId.toString()]: byPlayer[playerId.toString()] ?? [] },
    total_frames: skeletonData.length,
    player_position_counts: playerPositionCounts,
    video_width: videoWidth,
    video_height: videoHeight,
    court_corners: results.court_detection?.court_corners,
  }
}
