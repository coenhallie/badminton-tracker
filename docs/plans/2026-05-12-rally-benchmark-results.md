# Rally Accuracy Benchmark — Results

_Generated: 2026-05-12T12:01:44.462548Z_

## Recommendation

TrackNet-only path FAILS thresholds (min recall 0.80, max p95 boundary 19.77s). Choose union path for Phase 1.

## Per-Video Results

| Video | Union | Candidate | Matched | Recall | Start p95 (s) | End p95 (s) |
|---|---|---|---|---|---|---|
| `6418bfd9` | 23 | 26 | 20 | 86.96% | 19.77 | 4.51 |
| `ba5e697a` | 15 | 17 | 12 | 80.00% | 1.13 | 13.51 |
| `a3263b95` | 8 | 8 | 7 | 87.50% | 7.73 | 3.00 |

_Note: p95 columns degrade to max() for videos with fewer than 20 matched rally pairs._