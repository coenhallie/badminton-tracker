/**
 * Homography utilities for perspective transformation
 * Shared between MiniCourt visualization and advanced analytics
 */

import { COURT_DIMENSIONS } from '@/types/analysis'

const COURT_LENGTH = COURT_DIMENSIONS.length        // 13.4m
const COURT_WIDTH = COURT_DIMENSIONS.width_doubles   // 6.1m
const SERVICE_LINE = COURT_DIMENSIONS.service_line   // 1.98m from net

/** Standard court keypoint positions in meters (12-point system) */
export const COURT_KEYPOINT_POSITIONS: number[][] = [
  // 4 outer corners
  [0, 0],                                                    // 0: TL
  [COURT_WIDTH, 0],                                          // 1: TR
  [COURT_WIDTH, COURT_LENGTH],                               // 2: BR
  [0, COURT_LENGTH],                                         // 3: BL
  // Net intersections (at center length)
  [0, COURT_LENGTH / 2],                                     // 4: NL
  [COURT_WIDTH, COURT_LENGTH / 2],                           // 5: NR
  // Service line near (top half)
  [0, COURT_LENGTH / 2 - SERVICE_LINE],                      // 6: SNL
  [COURT_WIDTH, COURT_LENGTH / 2 - SERVICE_LINE],            // 7: SNR
  // Service line far (bottom half)
  [0, COURT_LENGTH / 2 + SERVICE_LINE],                      // 8: SFL
  [COURT_WIDTH, COURT_LENGTH / 2 + SERVICE_LINE],            // 9: SFR
  // Center line endpoints at service lines
  [COURT_WIDTH / 2, COURT_LENGTH / 2 - SERVICE_LINE],        // 10: CTN
  [COURT_WIDTH / 2, COURT_LENGTH / 2 + SERVICE_LINE],        // 11: CTF
]

/**
 * Normalize a set of 2D points for numerical stability (Hartley normalization)
 */
export function normalizePoints(points: number[][]): { normalized: number[][]; T: number[][] } {
  let sumX = 0, sumY = 0, count = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      sumX += p[0] ?? 0
      sumY += p[1] ?? 0
      count++
    }
  }
  if (count === 0) return { normalized: points, T: [[1,0,0],[0,1,0],[0,0,1]] }

  const meanX = sumX / count
  const meanY = sumY / count

  let sumDist = 0
  for (const p of points) {
    if (p && p.length >= 2) {
      const dx = (p[0] ?? 0) - meanX
      const dy = (p[1] ?? 0) - meanY
      sumDist += Math.sqrt(dx * dx + dy * dy)
    }
  }
  const avgDist = sumDist / count
  const scale = avgDist > 0 ? Math.sqrt(2) / avgDist : 1

  const normalized: number[][] = []
  for (const p of points) {
    if (p && p.length >= 2) {
      normalized.push([
        ((p[0] ?? 0) - meanX) * scale,
        ((p[1] ?? 0) - meanY) * scale
      ])
    } else {
      normalized.push([0, 0])
    }
  }

  const T: number[][] = [
    [scale, 0, -scale * meanX],
    [0, scale, -scale * meanY],
    [0, 0, 1]
  ]

  return { normalized, T }
}

/** Invert a 3x3 matrix */
export function invertMatrix3x3(M: number[][]): number[][] | null {
  const a = M[0]![0]!, b = M[0]![1]!, c = M[0]![2]!
  const d = M[1]![0]!, e = M[1]![1]!, f = M[1]![2]!
  const g = M[2]![0]!, h = M[2]![1]!, i = M[2]![2]!

  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
  if (Math.abs(det) < 1e-12) return null

  const invDet = 1.0 / det
  return [
    [(e*i - f*h) * invDet, (c*h - b*i) * invDet, (b*f - c*e) * invDet],
    [(f*g - d*i) * invDet, (a*i - c*g) * invDet, (c*d - a*f) * invDet],
    [(d*h - e*g) * invDet, (b*g - a*h) * invDet, (a*e - b*d) * invDet]
  ]
}

/** Multiply two 3x3 matrices */
export function multiplyMatrix3x3(A: number[][], B: number[][]): number[][] {
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0
      for (let k = 0; k < 3; k++) {
        sum += (A[i]![k] ?? 0) * (B[k]![j] ?? 0)
      }
      result[i]![j] = sum
    }
  }
  return result
}

/** Solve linear system Ax = b using Gaussian elimination with partial pivoting */
export function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length
  const augmented: number[][] = A.map((row, i) => [...row, b[i]!])

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row]![col]!) > Math.abs(augmented[maxRow]![col]!)) {
        maxRow = row
      }
    }
    ;[augmented[col], augmented[maxRow]] = [augmented[maxRow]!, augmented[col]!]
    if (Math.abs(augmented[col]![col]!) < 1e-10) return null

    for (let row = col + 1; row < n; row++) {
      const factor = augmented[row]![col]! / augmented[col]![col]!
      for (let j = col; j <= n; j++) {
        augmented[row]![j]! -= factor * augmented[col]![j]!
      }
    }
  }

  const x: number[] = new Array(n)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i]![n]!
    for (let j = i + 1; j < n; j++) {
      x[i]! -= augmented[i]![j]! * x[j]!
    }
    x[i]! /= augmented[i]![i]!
  }

  return x
}

/** Solve overdetermined linear system using least-squares (normal equations) */
export function solveLeastSquares(A: number[][], b: number[]): number[] | null {
  const m = A.length
  const n = A[0]?.length ?? 0
  if (n === 0) return null

  const AtA: number[][] = Array(n).fill(null).map(() => Array(n).fill(0)) as number[][]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0
      for (let k = 0; k < m; k++) {
        sum += (A[k]![i] ?? 0) * (A[k]![j] ?? 0)
      }
      AtA[i]![j] = sum
    }
  }

  const Atb: number[] = Array(n).fill(0) as number[]
  for (let i = 0; i < n; i++) {
    let sum = 0
    for (let k = 0; k < m; k++) {
      sum += (A[k]![i] ?? 0) * (b[k] ?? 0)
    }
    Atb[i] = sum
  }

  return solveLinearSystem(AtA, Atb)
}

/**
 * Calculate homography matrix using DLT with Hartley normalization
 * Maps video pixel coordinates → court coordinates in meters
 */
export function calculateHomography(srcPoints: number[][], dstPoints: number[][]): number[][] | null {
  const n = Math.min(srcPoints.length, dstPoints.length)
  if (n < 4) return null

  const validSrc: number[][] = []
  const validDst: number[][] = []
  for (let i = 0; i < n; i++) {
    const srcPoint = srcPoints[i]
    const dstPoint = dstPoints[i]
    if (srcPoint && srcPoint.length >= 2 && dstPoint && dstPoint.length >= 2) {
      validSrc.push(srcPoint)
      validDst.push(dstPoint)
    }
  }

  if (validSrc.length < 4) return null

  const { normalized: normSrc, T: T_src } = normalizePoints(validSrc)
  const { normalized: normDst, T: T_dst } = normalizePoints(validDst)

  const A: number[][] = []
  const b: number[] = []

  for (let i = 0; i < normSrc.length; i++) {
    const sx = normSrc[i]![0] ?? 0
    const sy = normSrc[i]![1] ?? 0
    const dx = normDst[i]![0] ?? 0
    const dy = normDst[i]![1] ?? 0

    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    b.push(dx)
    b.push(dy)
  }

  if (A.length < 8) return null

  let h: number[] | null
  if (A.length === 8) {
    h = solveLinearSystem(A, b)
  } else {
    h = solveLeastSquares(A, b)
  }

  if (!h) return null

  const H_norm: number[][] = [
    [h[0]!, h[1]!, h[2]!],
    [h[3]!, h[4]!, h[5]!],
    [h[6]!, h[7]!, 1]
  ]

  const T_dst_inv = invertMatrix3x3(T_dst)
  if (!T_dst_inv) return null

  const temp = multiplyMatrix3x3(H_norm, T_src)
  return multiplyMatrix3x3(T_dst_inv, temp)
}

/** Apply homography transformation to a single point */
export function applyHomography(H: number[][], x: number, y: number): { x: number; y: number } | null {
  const w = H[2]![0]! * x + H[2]![1]! * y + H[2]![2]!
  if (Math.abs(w) < 1e-10) return null

  const outX = (H[0]![0]! * x + H[0]![1]! * y + H[0]![2]!) / w
  const outY = (H[1]![0]! * x + H[1]![1]! * y + H[1]![2]!) / w

  return { x: outX, y: outY }
}

/**
 * Compute a homography matrix from court keypoints (4 or 12 video pixel points)
 * to standard court meter positions
 */
export function computeHomographyFromKeypoints(videoKeypoints: number[][]): number[][] | null {
  const n = videoKeypoints.length
  if (n < 4) return null

  let dstPoints: number[][]
  if (n === 4) {
    dstPoints = COURT_KEYPOINT_POSITIONS.slice(0, 4)
  } else if (n === 12) {
    dstPoints = COURT_KEYPOINT_POSITIONS.slice(0, 12)
  } else {
    dstPoints = COURT_KEYPOINT_POSITIONS.slice(0, Math.min(n, 12))
  }

  return calculateHomography(videoKeypoints, dstPoints)
}
