import type { Keypoint } from '@/types/analysis'
import { applyHomography } from '@/utils/homography'

export const KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

// COCO keypoint indices for ankles (matches KP constants in VideoPlayer).
const LEFT_ANKLE = 15
const RIGHT_ANKLE = 16

/**
 * Leg stretch in METERS — distance between the two ankles in court-plane
 * coordinates via homography. Returns null when either ankle is missing,
 * below the confidence threshold, or the homography projection fails.
 */
export function legStretchMeters(
  keypoints: Keypoint[] | undefined,
  H: number[][] | null,
): number | null {
  if (!keypoints || !H) return null
  const la = keypoints[LEFT_ANKLE]
  const ra = keypoints[RIGHT_ANKLE]
  if (!la?.x || !la?.y || !ra?.x || !ra?.y) return null
  if (la.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      ra.confidence < KEYPOINT_CONFIDENCE_THRESHOLD) return null

  const leftM = applyHomography(H, la.x, la.y)
  const rightM = applyHomography(H, ra.x, ra.y)
  if (!leftM || !rightM) return null

  return Math.hypot(leftM.x - rightM.x, leftM.y - rightM.y)
}

/**
 * "Knee flex" = smaller knee angle (tighter bend). Returns the minimum of
 * left_knee / right_knee from already-computed body_angles, ignoring nulls.
 */
export function kneeFlexDegrees(
  leftKnee: number | null | undefined,
  rightKnee: number | null | undefined,
): number | null {
  const vals = [leftKnee, rightKnee].filter(
    (v): v is number => typeof v === 'number',
  )
  if (vals.length === 0) return null
  return Math.min(...vals)
}
