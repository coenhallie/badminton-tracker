import type { SpeedZone } from '@/types/analysis'
import { SPEED_ZONE_THRESHOLDS } from '@/types/analysis'

/**
 * Get speed zone classification from km/h.
 */
export function getSpeedZone(speedKmh: number): SpeedZone {
  const speedMps = speedKmh / 3.6
  for (const [zoneName, thresholds] of Object.entries(SPEED_ZONE_THRESHOLDS)) {
    const maxSpeed = thresholds.max ?? Infinity
    if (speedMps >= thresholds.min && speedMps < maxSpeed) {
      return zoneName as SpeedZone
    }
  }
  return 'standing'
}
