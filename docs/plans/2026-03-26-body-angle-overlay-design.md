# Body Angle & Leg Stretch Overlay — Design

## Goal
Display real-time body angle measurements and leg stretch distance on player skeletons in the video player.

## What
- Toggle-able overlays for 10 joint angles + leg stretch distance
- Drawn on the existing canvas skeleton overlay
- Joint angles shown as arc + degree label at each joint
- Leg stretch shown as dashed line between ankles with distance in meters (via homography)

## Data
- Body angles: already computed backend-side, stored in `player.pose.body_angles` per frame
- Leg stretch: computed client-side from ankle keypoint positions + court homography
- No backend changes needed

## UI
- Multi-select dropdown near video player controls to toggle which overlays are visible
- Overlays render in real-time as video plays, matching player color
- Leg stretch requires manual court keypoints for meter conversion; hidden if unavailable
