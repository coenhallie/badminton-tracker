"""
TrackNetV3 - Shuttlecock Trajectory Tracking for Badminton

Port of the TrackNetV3 architecture (MIT License) for integration
with the badminton tracker pipeline.

Original: https://github.com/qaz812345/TrackNetV3
Paper: "TrackNetV3: Enhancing Short-term Tracking with Long-term Analysis"
"""

from .model import TrackNet, InpaintNet
from .inference import TrackNetInference

__all__ = ["TrackNet", "InpaintNet", "TrackNetInference"]
