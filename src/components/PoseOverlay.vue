<script setup lang="ts">
import { computed } from 'vue'
import type { SkeletonFrame, PoseType, TrainedPoseClass } from '@/types/analysis'
import {
  POSE_TYPE_COLORS, POSE_TYPE_NAMES, POSE_TYPE_ICONS, PLAYER_COLORS,
  TRAINED_POSE_COLORS, TRAINED_POSE_NAMES, TRAINED_POSE_ICONS
} from '@/types/analysis'

// Pose source: 'skeleton' for YOLO-pose keypoint analysis, 'trained' for custom trained model
export type PoseSource = 'skeleton' | 'trained' | 'both'

const props = withDefaults(defineProps<{
  skeletonFrame: SkeletonFrame | null
  visible: boolean
  poseSource?: PoseSource
}>(), {
  poseSource: 'both'
})

interface PlayerPoseDisplay {
  playerId: number
  pose: string
  confidence: number
  color: string
  icon: string
  name: string
  poseColor: string
  source: 'skeleton' | 'trained'
}

// Skeleton-based pose classifications (from keypoint analysis)
const skeletonPoses = computed<PlayerPoseDisplay[]>(() => {
  if (props.poseSource === 'trained') return []
  if (!props.skeletonFrame?.players) return []
  
  return props.skeletonFrame.players.map((player, index) => {
    const pose = player.pose?.pose_type ?? 'unknown'
    const confidence = player.pose?.confidence ?? 0
    
    return {
      playerId: player.player_id,
      pose,
      confidence,
      color: PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B',
      icon: POSE_TYPE_ICONS[pose as PoseType] ?? '❓',
      name: POSE_TYPE_NAMES[pose as PoseType] ?? 'Unknown',
      poseColor: POSE_TYPE_COLORS[pose as PoseType] ?? '#718096',
      source: 'skeleton' as const
    }
  })
})

// Trained model pose classifications (from custom YOLO model)
const trainedPoses = computed<PlayerPoseDisplay[]>(() => {
  if (props.poseSource === 'skeleton') return []
  if (!props.skeletonFrame?.pose_classifications) return []
  
  return props.skeletonFrame.pose_classifications.map((pc, index) => {
    // Use class_name from the API response (matches PoseClassificationResult interface)
    const pose = pc.class_name as TrainedPoseClass
    
    return {
      playerId: index + 1,  // Use index as player ID since trained model doesn't track
      pose,
      confidence: pc.confidence,
      color: PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B',
      icon: TRAINED_POSE_ICONS[pose] ?? '❓',
      name: TRAINED_POSE_NAMES[pose] ?? 'Unknown',
      poseColor: TRAINED_POSE_COLORS[pose] ?? '#718096',
      source: 'trained' as const
    }
  })
})

// Combined poses for display
const playerPoses = computed<PlayerPoseDisplay[]>(() => {
  if (props.poseSource === 'skeleton') return skeletonPoses.value
  if (props.poseSource === 'trained') return trainedPoses.value
  // 'both' - show both sources combined
  return [...skeletonPoses.value, ...trainedPoses.value]
})

const hasPlayers = computed(() => playerPoses.value.length > 0)

// Group poses by source for better display when showing both
const hasBothSources = computed(() =>
  props.poseSource === 'both' &&
  skeletonPoses.value.length > 0 &&
  trainedPoses.value.length > 0
)

// Get title based on pose source
const overlayTitle = computed(() => {
  switch (props.poseSource) {
    case 'skeleton': return 'Poses (Skeleton)'
    case 'trained': return 'Poses (AI Model)'
    default: return 'Player Poses'
  }
})
</script>

<template>
  <Transition name="slide-fade">
    <div v-if="visible && hasPlayers" class="pose-overlay">
      <div class="pose-card">
        <div class="pose-header">
          <span class="pose-icon">🏸</span>
          <span class="pose-title">{{ overlayTitle }}</span>
        </div>
        
        <!-- Skeleton-based poses section -->
        <template v-if="poseSource === 'both' && skeletonPoses.length > 0">
          <div class="source-label skeleton-source">
            <span class="source-icon">🦴</span>
            Skeleton Analysis
          </div>
        </template>
        
        <div class="pose-list" v-if="skeletonPoses.length > 0 && poseSource !== 'trained'">
          <div
            v-for="player in skeletonPoses"
            :key="`skeleton-${player.playerId}`"
            class="pose-item"
          >
            <div class="player-indicator" :style="{ backgroundColor: player.color }">
              P{{ player.playerId }}
            </div>
            
            <div class="pose-info">
              <div class="pose-name" :style="{ color: player.poseColor }">
                <span class="pose-emoji">{{ player.icon }}</span>
                {{ player.name }}
              </div>
              
              <div class="pose-confidence">
                <div class="confidence-bar">
                  <div
                    class="confidence-fill"
                    :style="{
                      width: `${player.confidence * 100}%`,
                      backgroundColor: player.poseColor
                    }"
                  />
                </div>
                <span class="confidence-text">{{ (player.confidence * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Trained model poses section -->
        <template v-if="poseSource === 'both' && trainedPoses.length > 0">
          <div class="source-label trained-source">
            <span class="source-icon">🤖</span>
            AI Model
          </div>
        </template>
        
        <div class="pose-list" v-if="trainedPoses.length > 0 && poseSource !== 'skeleton'">
          <div
            v-for="(pose, index) in trainedPoses"
            :key="`trained-${index}`"
            class="pose-item"
          >
            <div class="pose-badge trained-badge" :style="{ backgroundColor: pose.poseColor }">
              {{ pose.icon }}
            </div>
            
            <div class="pose-info">
              <div class="pose-name" :style="{ color: pose.poseColor }">
                {{ pose.name }}
              </div>
              
              <div class="pose-confidence">
                <div class="confidence-bar">
                  <div
                    class="confidence-fill"
                    :style="{
                      width: `${pose.confidence * 100}%`,
                      backgroundColor: pose.poseColor
                    }"
                  />
                </div>
                <span class="confidence-text">{{ (pose.confidence * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.pose-overlay {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 20;
  pointer-events: auto;
}

.pose-card {
  background: rgba(15, 15, 26, 0.9);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 12px;
  min-width: 180px;
  max-width: 220px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.pose-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 10px;
}

.pose-icon {
  font-size: 1rem;
}

.pose-title {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #a0aec0;
}

.pose-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.pose-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.player-indicator {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 700;
  color: #000;
}

.pose-info {
  flex: 1;
  min-width: 0;
}

.pose-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 4px;
}

.pose-emoji {
  font-size: 1rem;
}

.pose-confidence {
  display: flex;
  align-items: center;
  gap: 8px;
}

.confidence-bar {
  flex: 1;
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s ease;
}

.confidence-text {
  font-size: 0.65rem;
  color: #718096;
  min-width: 28px;
  text-align: right;
}

/* Source labels for 'both' mode */
.source-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 6px 0;
  margin-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.source-label:first-of-type {
  margin-top: 0;
  border-top: none;
}

.source-icon {
  font-size: 0.75rem;
}

.skeleton-source {
  color: #4ECDC4;
}

.trained-source {
  color: #F59E0B;
}

/* Pose badge for trained model */
.pose-badge {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
}

.trained-badge {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Transition animations */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from {
  transform: translateX(20px);
  opacity: 0;
}

.slide-fade-leave-to {
  transform: translateX(20px);
  opacity: 0;
}
</style>
