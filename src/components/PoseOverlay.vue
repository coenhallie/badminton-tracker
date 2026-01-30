<script setup lang="ts">
import { computed } from 'vue'
import type { SkeletonFrame, PoseType } from '@/types/analysis'
import { POSE_TYPE_COLORS, POSE_TYPE_NAMES, POSE_TYPE_ICONS, PLAYER_COLORS } from '@/types/analysis'

const props = defineProps<{
  skeletonFrame: SkeletonFrame | null
  visible: boolean
}>()

interface PlayerPoseDisplay {
  playerId: number
  pose: PoseType
  confidence: number
  color: string
  icon: string
  name: string
  poseColor: string
}

const playerPoses = computed<PlayerPoseDisplay[]>(() => {
  if (!props.skeletonFrame?.players) return []
  
  return props.skeletonFrame.players.map((player, index) => {
    const pose = player.pose?.pose_type ?? 'unknown'
    const confidence = player.pose?.confidence ?? 0
    
    return {
      playerId: player.player_id,
      pose,
      confidence,
      color: PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#FF6B6B',
      icon: POSE_TYPE_ICONS[pose] ?? '❓',
      name: POSE_TYPE_NAMES[pose] ?? 'Unknown',
      poseColor: POSE_TYPE_COLORS[pose] ?? '#718096'
    }
  })
})

const hasPlayers = computed(() => playerPoses.value.length > 0)
</script>

<template>
  <Transition name="slide-fade">
    <div v-if="visible && hasPlayers" class="pose-overlay">
      <div class="pose-card">
        <div class="pose-header">
          <span class="pose-icon">🏸</span>
          <span class="pose-title">Player Poses</span>
        </div>
        
        <div class="pose-list">
          <div 
            v-for="player in playerPoses" 
            :key="player.playerId"
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
