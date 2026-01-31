<script setup lang="ts">
import { computed } from 'vue'

type ZoneType = 'front' | 'mid' | 'back' | 'left' | 'center' | 'right'

const props = defineProps<{
  zone: ZoneType
}>()

// Court dimensions (simplified for visualization)
const COURT_WIDTH = 60
const COURT_HEIGHT = 100

// Zone definitions in percentages of court
const zoneDefinitions: Record<ZoneType, { x: number; y: number; width: number; height: number; description: string }> = {
  // Vertical zones (depth - relative to net)
  front: {
    x: 0,
    y: 35, // Close to net (which is at 50%)
    width: 100,
    height: 15,
    description: 'Near the net'
  },
  mid: {
    x: 0,
    y: 20,
    width: 100,
    height: 15,
    description: 'Middle court area'
  },
  back: {
    x: 0,
    y: 0,
    width: 100,
    height: 20,
    description: 'Rear court / baseline'
  },
  // Horizontal zones (width)
  left: {
    x: 0,
    y: 0,
    width: 33,
    height: 50,
    description: 'Left side of court'
  },
  center: {
    x: 33,
    y: 0,
    width: 34,
    height: 50,
    description: 'Center of court'
  },
  right: {
    x: 67,
    y: 0,
    width: 33,
    height: 50,
    description: 'Right side of court'
  }
}

const currentZone = computed(() => zoneDefinitions[props.zone])

const isVerticalZone = computed(() => ['front', 'mid', 'back'].includes(props.zone))

// Get the color for the zone highlight
const zoneColors: Record<ZoneType, string> = {
  front: '#22c55e',
  mid: '#3b82f6',
  back: '#f59e0b',
  left: '#ec4899',
  center: '#8b5cf6',
  right: '#06b6d4'
}

const highlightColor = computed(() => zoneColors[props.zone])
</script>

<template>
  <div class="court-zone-tooltip">
    <div class="tooltip-content">
      <svg 
        :width="COURT_WIDTH" 
        :height="isVerticalZone ? COURT_HEIGHT / 2 : COURT_HEIGHT / 2" 
        viewBox="0 0 60 50"
        class="court-svg"
      >
        <!-- Court background -->
        <rect x="0" y="0" width="60" height="50" fill="#1a472a" />
        
        <!-- Court outline -->
        <rect x="1" y="1" width="58" height="48" fill="none" stroke="#fff" stroke-width="1" opacity="0.6" />
        
        <!-- Net line (at center for vertical zones, at bottom for horizontal) -->
        <line 
          v-if="isVerticalZone"
          x1="1" y1="50" x2="59" y2="50" 
          stroke="#ff4444" 
          stroke-width="2"
        />
        <line 
          v-else
          x1="1" y1="50" x2="59" y2="50" 
          stroke="#ff4444" 
          stroke-width="2"
        />
        
        <!-- Service lines for vertical zones -->
        <template v-if="isVerticalZone">
          <!-- Short service line -->
          <line x1="1" y1="35" x2="59" y2="35" stroke="#fff" stroke-width="0.5" opacity="0.4" />
          <!-- Center line -->
          <line x1="30" y1="1" x2="30" y2="35" stroke="#fff" stroke-width="0.5" opacity="0.4" />
        </template>
        
        <!-- Vertical lines for horizontal zones -->
        <template v-else>
          <!-- Thirds dividers -->
          <line x1="20" y1="1" x2="20" y2="49" stroke="#fff" stroke-width="0.5" opacity="0.3" />
          <line x1="40" y1="1" x2="40" y2="49" stroke="#fff" stroke-width="0.5" opacity="0.3" />
          <!-- Center line -->
          <line x1="30" y1="1" x2="30" y2="35" stroke="#fff" stroke-width="0.5" opacity="0.4" />
          <!-- Service line -->
          <line x1="1" y1="35" x2="59" y2="35" stroke="#fff" stroke-width="0.5" opacity="0.4" />
        </template>
        
        <!-- Highlighted zone -->
        <rect
          v-if="isVerticalZone"
          :x="currentZone.x * 0.6"
          :y="currentZone.y"
          :width="currentZone.width * 0.6"
          :height="currentZone.height"
          :fill="highlightColor"
          opacity="0.5"
        />
        <rect
          v-else
          :x="currentZone.x * 0.6"
          :y="0"
          :width="currentZone.width * 0.6"
          :height="50"
          :fill="highlightColor"
          opacity="0.5"
        />
        
        <!-- Zone border highlight -->
        <rect
          v-if="isVerticalZone"
          :x="currentZone.x * 0.6 + 0.5"
          :y="currentZone.y + 0.5"
          :width="currentZone.width * 0.6 - 1"
          :height="currentZone.height - 1"
          fill="none"
          :stroke="highlightColor"
          stroke-width="1.5"
        />
        <rect
          v-else
          :x="currentZone.x * 0.6 + 0.5"
          :y="0.5"
          :width="currentZone.width * 0.6 - 1"
          :height="49"
          fill="none"
          :stroke="highlightColor"
          stroke-width="1.5"
        />
        
        <!-- Net label -->
        <text x="30" y="48" text-anchor="middle" fill="#ff6666" font-size="5" font-weight="bold">NET</text>
      </svg>
      
      <div class="zone-description" :style="{ color: highlightColor }">
        {{ currentZone.description }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.court-zone-tooltip {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  pointer-events: none;
}

.tooltip-content {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.court-svg {
  border-radius: 2px;
}

.zone-description {
  font-size: 10px;
  font-weight: 500;
  text-align: center;
}

/* Arrow pointing down */
.tooltip-content::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #333;
}
</style>
