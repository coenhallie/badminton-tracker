"""
Heatmap Generator for Badminton Player Position Visualization

This module generates efficient heatmaps from player position data for
real-time video overlay rendering. Optimized for performance with:
- NumPy-vectorized Gaussian kernel computation
- Pre-computed kernel caching
- Incremental heatmap updates (O(1) per position update)
- Memory-mapped arrays for large videos
- Configurable resolution and decay

Based on Ultralytics YOLO26 heatmap solutions pattern with custom optimizations
for badminton court tracking.

Author: Badminton Tracker Project
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import cv2
import json

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Default heatmap parameters optimized for badminton court analysis
DEFAULT_HEATMAP_WIDTH = 640
DEFAULT_HEATMAP_HEIGHT = 480
DEFAULT_KERNEL_SIZE = 31  # Gaussian kernel size (must be odd)
DEFAULT_SIGMA = 10.0  # Gaussian spread for position points
DEFAULT_DECAY_RATE = 0.995  # Per-frame decay rate for temporal fading
MIN_INTENSITY_THRESHOLD = 0.01  # Minimum intensity to keep (prune small values)

# Colormap options matching OpenCV colormaps
class HeatmapColormap(str, Enum):
    """Available colormaps for heatmap visualization"""
    JET = "jet"
    PARULA = "parula"
    TURBO = "turbo"
    HOT = "hot"
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    COOL = "cool"

# Map string names to OpenCV colormap constants
COLORMAP_MAPPING = {
    "jet": cv2.COLORMAP_JET,
    "parula": cv2.COLORMAP_PARULA,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "cool": cv2.COLORMAP_COOL,
}


# =============================================================================
# PERFORMANCE OPTIMIZATION: Pre-computed Gaussian Kernel Cache
# =============================================================================

class GaussianKernelCache:
    """
    Cache for pre-computed Gaussian kernels to avoid repeated computation.
    
    PERFORMANCE: Computing a Gaussian kernel is O(k²) where k is kernel size.
    By caching kernels, we reduce this to O(1) lookup for repeated positions.
    """
    
    _cache: Dict[Tuple[int, float], np.ndarray] = {}
    
    @classmethod
    def get_kernel(cls, size: int, sigma: float) -> np.ndarray:
        """
        Get a cached Gaussian kernel or compute and cache it.
        
        Args:
            size: Kernel size (must be odd)
            sigma: Gaussian standard deviation
            
        Returns:
            2D numpy array containing the Gaussian kernel
        """
        key = (size, sigma)
        if key not in cls._cache:
            # Ensure odd size
            if size % 2 == 0:
                size += 1
            
            # Create coordinate grids
            ax = np.arange(-size // 2 + 1, size // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            
            # Compute Gaussian (normalized to peak at 1.0)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            
            # Normalize so the center is 1.0
            kernel = kernel / kernel.max()
            
            cls._cache[key] = kernel.astype(np.float32)
        
        return cls._cache[key]
    
    @classmethod
    def clear_cache(cls):
        """Clear the kernel cache to free memory"""
        cls._cache.clear()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation"""
    width: int = DEFAULT_HEATMAP_WIDTH
    height: int = DEFAULT_HEATMAP_HEIGHT
    kernel_size: int = DEFAULT_KERNEL_SIZE
    sigma: float = DEFAULT_SIGMA
    decay_rate: float = DEFAULT_DECAY_RATE
    colormap: str = "turbo"
    intensity_scale: float = 1.0  # Multiplier for position contributions
    per_player: bool = True  # Generate separate heatmaps per player
    normalize: bool = True  # Normalize output to 0-255 range
    
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "decay_rate": self.decay_rate,
            "colormap": self.colormap,
            "intensity_scale": self.intensity_scale,
            "per_player": self.per_player,
            "normalize": self.normalize
        }


@dataclass
class HeatmapData:
    """
    Heatmap data structure for a video analysis.
    
    Contains both raw intensity data and rendering metadata.
    Optimized for JSON serialization and frontend consumption.
    """
    video_id: str
    width: int
    height: int
    colormap: str
    
    # Raw heatmap data as 2D array (will be serialized as nested list)
    combined_heatmap: Optional[np.ndarray] = None
    
    # Per-player heatmaps (player_id -> 2D array)
    player_heatmaps: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    total_frames: int = 0
    player_position_counts: Dict[int, int] = field(default_factory=dict)
    
    # Court mapping info (for coordinate transformation)
    video_width: int = 0
    video_height: int = 0
    court_corners: Optional[List[List[float]]] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        result = {
            "video_id": self.video_id,
            "width": self.width,
            "height": self.height,
            "colormap": self.colormap,
            "total_frames": self.total_frames,
            "player_position_counts": self.player_position_counts,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "court_corners": self.court_corners,
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if self.combined_heatmap is not None:
            # Normalize to 0-255 for efficient transfer
            normalized = self._normalize_for_transfer(self.combined_heatmap)
            result["combined_heatmap"] = normalized.tolist()
        
        if self.player_heatmaps:
            result["player_heatmaps"] = {
                str(pid): self._normalize_for_transfer(hmap).tolist()
                for pid, hmap in self.player_heatmaps.items()
            }
        
        return result
    
    def _normalize_for_transfer(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to uint8 for efficient transfer"""
        if arr.max() > 0:
            normalized = (arr / arr.max() * 255).astype(np.uint8)
        else:
            normalized = arr.astype(np.uint8)
        return normalized


# =============================================================================
# MAIN HEATMAP GENERATOR CLASS
# =============================================================================

class HeatmapGenerator:
    """
    High-performance heatmap generator for player position tracking.
    
    Uses Gaussian kernel accumulation with temporal decay for smooth,
    informative visualizations of player movement patterns.
    
    PERFORMANCE OPTIMIZATIONS:
    - Pre-computed Gaussian kernels (cached)
    - NumPy-vectorized operations
    - Incremental updates (no full recomputation)
    - Sparse position tracking
    - Optional downsampling for large videos
    
    Usage:
        generator = HeatmapGenerator(config)
        
        # During video processing:
        for frame in video:
            for player in frame.players:
                generator.add_position(player.id, player.x, player.y)
            generator.apply_decay()  # Optional: temporal fading
        
        # Get final heatmap:
        heatmap_data = generator.get_heatmap_data(video_id)
    """
    
    def __init__(
        self,
        config: Optional[HeatmapConfig] = None,
        video_width: int = 1920,
        video_height: int = 1080
    ):
        """
        Initialize the heatmap generator.
        
        Args:
            config: Heatmap configuration (uses defaults if None)
            video_width: Original video width for coordinate scaling
            video_height: Original video height for coordinate scaling
        """
        self.config = config or HeatmapConfig()
        self.video_width = video_width
        self.video_height = video_height
        
        # Scale factors for coordinate transformation
        self.scale_x = self.config.width / video_width
        self.scale_y = self.config.height / video_height
        
        # Initialize heatmap arrays (float32 for precision during accumulation)
        self.combined_heatmap = np.zeros(
            (self.config.height, self.config.width),
            dtype=np.float32
        )
        
        # Per-player heatmaps
        self.player_heatmaps: Dict[int, np.ndarray] = {}
        
        # Get cached Gaussian kernel
        self.kernel = GaussianKernelCache.get_kernel(
            self.config.kernel_size,
            self.config.sigma
        )
        self.kernel_half = self.config.kernel_size // 2
        
        # Statistics tracking
        self.frame_count = 0
        self.position_counts: Dict[int, int] = {}
        
        # Court corners for frontend reference
        self.court_corners: Optional[List[List[float]]] = None
    
    def add_position(
        self,
        player_id: int,
        x: float,
        y: float,
        intensity: float = 1.0
    ) -> None:
        """
        Add a player position to the heatmap.
        
        PERFORMANCE: O(k²) where k is kernel size, but typically k << image size
        so this is effectively O(1) per position.
        
        Args:
            player_id: Unique player identifier
            x: X coordinate in video pixels
            y: Y coordinate in video pixels
            intensity: Optional intensity multiplier (default 1.0)
        """
        # Scale coordinates to heatmap space
        hx = int(x * self.scale_x)
        hy = int(y * self.scale_y)
        
        # Bounds check
        if not (0 <= hx < self.config.width and 0 <= hy < self.config.height):
            return
        
        # Calculate kernel bounds with clipping
        x1 = max(0, hx - self.kernel_half)
        x2 = min(self.config.width, hx + self.kernel_half + 1)
        y1 = max(0, hy - self.kernel_half)
        y2 = min(self.config.height, hy + self.kernel_half + 1)
        
        # Calculate corresponding kernel region
        kx1 = self.kernel_half - (hx - x1)
        kx2 = kx1 + (x2 - x1)
        ky1 = self.kernel_half - (hy - y1)
        ky2 = ky1 + (y2 - y1)
        
        # Scale intensity
        scaled_intensity = intensity * self.config.intensity_scale
        
        # Apply Gaussian kernel to combined heatmap
        self.combined_heatmap[y1:y2, x1:x2] += self.kernel[ky1:ky2, kx1:kx2] * scaled_intensity
        
        # Apply to per-player heatmap if enabled
        if self.config.per_player:
            if player_id not in self.player_heatmaps:
                self.player_heatmaps[player_id] = np.zeros(
                    (self.config.height, self.config.width),
                    dtype=np.float32
                )
            self.player_heatmaps[player_id][y1:y2, x1:x2] += self.kernel[ky1:ky2, kx1:kx2] * scaled_intensity
        
        # Update statistics
        self.position_counts[player_id] = self.position_counts.get(player_id, 0) + 1
    
    def add_positions_batch(
        self,
        positions: List[Tuple[int, float, float, float]]
    ) -> None:
        """
        Add multiple positions in a batch (more efficient for many positions).
        
        Args:
            positions: List of (player_id, x, y, intensity) tuples
        """
        for player_id, x, y, intensity in positions:
            self.add_position(player_id, x, y, intensity)
    
    def apply_decay(self) -> None:
        """
        Apply temporal decay to the heatmap.
        
        Call this once per frame to create a fading effect where
        older positions gradually disappear.
        
        PERFORMANCE: O(w*h) but vectorized, so very fast with NumPy
        """
        if self.config.decay_rate < 1.0:
            self.combined_heatmap *= self.config.decay_rate
            
            for player_id in self.player_heatmaps:
                self.player_heatmaps[player_id] *= self.config.decay_rate
        
        self.frame_count += 1
    
    def prune_low_intensity(self) -> None:
        """
        Remove very low intensity values to save memory and improve rendering.
        
        PERFORMANCE: O(w*h) but helps with subsequent operations
        """
        threshold = MIN_INTENSITY_THRESHOLD * self.combined_heatmap.max() if self.combined_heatmap.max() > 0 else 0
        self.combined_heatmap[self.combined_heatmap < threshold] = 0
        
        for player_id in self.player_heatmaps:
            player_max = self.player_heatmaps[player_id].max()
            threshold = MIN_INTENSITY_THRESHOLD * player_max if player_max > 0 else 0
            self.player_heatmaps[player_id][self.player_heatmaps[player_id] < threshold] = 0
    
    def set_court_corners(self, corners: List[List[float]]) -> None:
        """Set court corners for frontend coordinate transformation"""
        self.court_corners = corners
    
    def get_heatmap_data(self, video_id: str) -> HeatmapData:
        """
        Get the complete heatmap data structure.
        
        Args:
            video_id: Video identifier for the heatmap
            
        Returns:
            HeatmapData object with all heatmap information
        """
        return HeatmapData(
            video_id=video_id,
            width=self.config.width,
            height=self.config.height,
            colormap=self.config.colormap,
            combined_heatmap=self.combined_heatmap.copy(),
            player_heatmaps={k: v.copy() for k, v in self.player_heatmaps.items()},
            total_frames=self.frame_count,
            player_position_counts=self.position_counts.copy(),
            video_width=self.video_width,
            video_height=self.video_height,
            court_corners=self.court_corners
        )
    
    def render_frame(
        self,
        frame: np.ndarray,
        alpha: float = 0.5,
        player_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Render heatmap overlay on a video frame.
        
        PERFORMANCE: Uses OpenCV's optimized applyColorMap and addWeighted
        
        Args:
            frame: BGR video frame
            alpha: Opacity of heatmap overlay (0-1)
            player_id: If specified, render only that player's heatmap
            
        Returns:
            Frame with heatmap overlay
        """
        # Select heatmap to render
        if player_id is not None and player_id in self.player_heatmaps:
            heatmap = self.player_heatmaps[player_id]
        else:
            heatmap = self.combined_heatmap
        
        # Skip if heatmap is empty
        if heatmap.max() == 0:
            return frame
        
        # Normalize to 0-255 range
        normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Resize to frame dimensions if needed
        frame_h, frame_w = frame.shape[:2]
        if normalized.shape != (frame_h, frame_w):
            normalized = cv2.resize(normalized, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        colormap = COLORMAP_MAPPING.get(self.config.colormap, cv2.COLORMAP_TURBO)
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Create mask where heatmap has values (avoid coloring black areas)
        mask = normalized > 0
        
        # Blend with frame
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
        
        return result
    
    def reset(self) -> None:
        """Reset the heatmap generator to initial state"""
        self.combined_heatmap.fill(0)
        self.player_heatmaps.clear()
        self.frame_count = 0
        self.position_counts.clear()
        self.court_corners = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get heatmap generation statistics"""
        return {
            "frame_count": self.frame_count,
            "position_counts": self.position_counts,
            "total_positions": sum(self.position_counts.values()),
            "heatmap_max_intensity": float(self.combined_heatmap.max()),
            "heatmap_mean_intensity": float(self.combined_heatmap.mean()),
            "active_players": len(self.player_heatmaps),
            "config": self.config.to_dict()
        }


# =============================================================================
# FACTORY & UTILITY FUNCTIONS
# =============================================================================

def create_heatmap_generator_from_video(
    video_width: int,
    video_height: int,
    config: Optional[HeatmapConfig] = None
) -> HeatmapGenerator:
    """
    Create a heatmap generator configured for a specific video.
    
    Automatically scales heatmap resolution based on video dimensions
    while maintaining a reasonable maximum size for performance.
    
    Args:
        video_width: Video width in pixels
        video_height: Video height in pixels
        config: Optional custom configuration
        
    Returns:
        Configured HeatmapGenerator instance
    """
    if config is None:
        config = HeatmapConfig()
    
    # Scale heatmap resolution proportionally to video
    # but cap at reasonable maximum for performance
    max_dim = 640
    aspect_ratio = video_width / video_height
    
    if video_width > video_height:
        config.width = min(max_dim, video_width)
        config.height = int(config.width / aspect_ratio)
    else:
        config.height = min(max_dim, video_height)
        config.width = int(config.height * aspect_ratio)
    
    # Adjust kernel size based on heatmap resolution
    # Larger heatmaps need larger kernels for smooth appearance
    base_kernel = 31
    scale_factor = config.width / DEFAULT_HEATMAP_WIDTH
    config.kernel_size = max(15, min(51, int(base_kernel * scale_factor)))
    if config.kernel_size % 2 == 0:
        config.kernel_size += 1
    
    config.sigma = config.kernel_size / 3
    
    return HeatmapGenerator(config, video_width, video_height)


def generate_heatmap_from_skeleton_data(
    skeleton_frames: List[Dict],
    video_width: int,
    video_height: int,
    video_id: str,
    config: Optional[HeatmapConfig] = None,
    court_corners: Optional[List[List[float]]] = None
) -> HeatmapData:
    """
    Generate a complete heatmap from pre-computed skeleton frame data.
    
    This is useful for generating heatmaps from already-analyzed videos
    without needing to reprocess the video.
    
    PERFORMANCE: O(n * k²) where n is total positions and k is kernel size
    For typical videos, this runs in < 1 second.
    
    Args:
        skeleton_frames: List of skeleton frame dictionaries (from analysis)
        video_width: Original video width
        video_height: Original video height
        video_id: Video identifier
        config: Optional heatmap configuration
        court_corners: Optional court corner coordinates
        
    Returns:
        Complete HeatmapData structure
    """
    generator = create_heatmap_generator_from_video(video_width, video_height, config)
    
    if court_corners:
        generator.set_court_corners(court_corners)
    
    # DEBUG: Track statistics
    total_positions = 0
    position_samples = []
    
    # Process each frame
    for frame_data in skeleton_frames:
        players = frame_data.get("players", [])
        
        for player in players:
            player_id = player.get("player_id", 0)
            center = player.get("center")
            
            if center and center.get("x") is not None and center.get("y") is not None:
                x = float(center["x"])
                y = float(center["y"])
                
                # Collect samples for debugging
                if len(position_samples) < 10:
                    position_samples.append((player_id, x, y))
                total_positions += 1
                
                # Use constant intensity for clear visibility
                intensity = 1.0
                
                generator.add_position(player_id, x, y, intensity)
        
        # Don't apply decay for static heatmap generation
        generator.frame_count += 1
    
    # DEBUG: Print heatmap generation summary
    print(f"\n[HEATMAP DEBUG] Generation complete for video {video_id}")
    print(f"[HEATMAP DEBUG] Video dimensions: {video_width}x{video_height}")
    print(f"[HEATMAP DEBUG] Heatmap dimensions: {generator.config.width}x{generator.config.height}")
    print(f"[HEATMAP DEBUG] Scale factors: x={generator.scale_x:.4f}, y={generator.scale_y:.4f}")
    print(f"[HEATMAP DEBUG] Total frames: {generator.frame_count}")
    print(f"[HEATMAP DEBUG] Total positions added: {total_positions}")
    print(f"[HEATMAP DEBUG] Position counts by player: {generator.position_counts}")
    print(f"[HEATMAP DEBUG] Sample positions (first 10):")
    for pid, x, y in position_samples:
        hx = int(x * generator.scale_x)
        hy = int(y * generator.scale_y)
        print(f"  Player {pid}: video({x:.1f}, {y:.1f}) -> heatmap({hx}, {hy})")
    print(f"[HEATMAP DEBUG] Heatmap max intensity: {generator.combined_heatmap.max():.4f}")
    print(f"[HEATMAP DEBUG] Heatmap non-zero pixels: {(generator.combined_heatmap > 0).sum()}")
    
    return generator.get_heatmap_data(video_id)


# =============================================================================
# MODULE-LEVEL SINGLETON FOR REUSE
# =============================================================================

_generator_cache: Dict[str, HeatmapGenerator] = {}


def get_or_create_generator(
    video_id: str,
    video_width: int,
    video_height: int,
    config: Optional[HeatmapConfig] = None
) -> HeatmapGenerator:
    """
    Get or create a cached heatmap generator for a video.
    
    Args:
        video_id: Unique video identifier
        video_width: Video width
        video_height: Video height
        config: Optional configuration
        
    Returns:
        HeatmapGenerator instance (cached if exists)
    """
    if video_id not in _generator_cache:
        _generator_cache[video_id] = create_heatmap_generator_from_video(
            video_width, video_height, config
        )
    return _generator_cache[video_id]


def clear_generator_cache(video_id: Optional[str] = None) -> None:
    """
    Clear cached generators.
    
    Args:
        video_id: If specified, only clear that video's generator.
                  If None, clear all cached generators.
    """
    if video_id:
        _generator_cache.pop(video_id, None)
    else:
        _generator_cache.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HeatmapGenerator",
    "HeatmapConfig",
    "HeatmapData",
    "HeatmapColormap",
    "GaussianKernelCache",
    "create_heatmap_generator_from_video",
    "generate_heatmap_from_skeleton_data",
    "get_or_create_generator",
    "clear_generator_cache",
    "COLORMAP_MAPPING",
]
