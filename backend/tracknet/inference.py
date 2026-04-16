"""
TrackNetV3 Inference Pipeline

Handles the complete inference pipeline:
1. Frame extraction and preprocessing
2. Median background computation
3. Batch inference through TrackNet
4. Heatmap → coordinate conversion
5. Trajectory inpainting via InpaintNet
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .model import TrackNet, InpaintNet

# TrackNetV3 constants (from original repo)
WIDTH = 512
HEIGHT = 288
SIGMA = 2.5


class TrackNetInference:
    """
    Complete TrackNetV3 inference pipeline for shuttlecock tracking.

    Usage:
        tracker = TrackNetInference(device="cuda")
        tracker.load_weights("ckpts/TrackNet_best.pt", "ckpts/InpaintNet_best.pt")
        positions = tracker.track_video("/path/to/video.mp4")
        # positions: dict of {frame_number: {"x": float, "y": float, "visible": bool}}
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tracknet: Optional[TrackNet] = None
        self.inpaintnet: Optional[InpaintNet] = None
        self.seq_len = 8
        self.bg_mode = "concat"

    def load_weights(
        self,
        tracknet_path: str,
        inpaintnet_path: Optional[str] = None,
    ):
        """
        Load TrackNet and optionally InpaintNet weights.

        The checkpoint format (from original TrackNetV3 repo):
        - ckpt["model"]: state dict
        - ckpt["param_dict"]: config with seq_len, bg_mode, etc.
        """
        # Load TrackNet checkpoint
        ckpt = torch.load(tracknet_path, map_location=self.device, weights_only=False)

        # Extract config from param_dict (TrackNetV3 checkpoint format)
        param_dict = ckpt.get("param_dict", {})
        self.seq_len = param_dict.get("seq_len", 8)
        self.bg_mode = param_dict.get("bg_mode", "concat")

        # Compute input channels based on config
        in_dim = self._compute_in_dim()
        # Output dim = seq_len (one heatmap per input frame)
        out_dim = self.seq_len

        self.tracknet = TrackNet(in_dim=in_dim, out_dim=out_dim)
        self.tracknet.load_state_dict(ckpt["model"])
        self.tracknet.to(self.device)
        self.tracknet.eval()

        print(f"[TrackNet] Loaded: seq_len={self.seq_len}, bg_mode='{self.bg_mode}', "
              f"in_dim={in_dim}, out_dim={out_dim}, device={self.device}")

        # Load InpaintNet if provided
        if inpaintnet_path:
            inpaint_ckpt = torch.load(inpaintnet_path, map_location=self.device, weights_only=False)
            self.inpaintnet = InpaintNet()
            self.inpaintnet.load_state_dict(inpaint_ckpt["model"])
            self.inpaintnet.to(self.device)
            self.inpaintnet.eval()
            print("[TrackNet] InpaintNet loaded")

    def _compute_in_dim(self) -> int:
        """Compute input channel dimension based on config."""
        if self.bg_mode == "concat":
            return (self.seq_len + 1) * 3  # extra frame for background
        elif self.bg_mode == "subtract":
            return self.seq_len
        elif self.bg_mode == "subtract_concat":
            return self.seq_len * 4
        else:
            return self.seq_len * 3

    def track_video(
        self,
        video_path: str,
        batch_size: int = 16,
        max_bg_samples: int = 300,
        progress_callback=None,
        log_callback=None,
    ) -> Dict[int, Dict]:
        """
        Run full TrackNetV3 pipeline on a video.

        Uses streaming frame extraction to avoid loading the entire video
        into memory at once — critical for large/long videos that would
        otherwise OOM on GPU workers.

        Args:
            video_path: Path to the video file.
            batch_size: Inference batch size.
            max_bg_samples: Max frames to sample for median background.
            progress_callback: Optional callable(progress_pct) for batch progress.
            log_callback: Optional callable(message) for step-level logging.

        Returns:
            Dict mapping frame_number → {"x": float, "y": float, "visible": bool}
            Coordinates are in original video pixel space.
        """
        if self.tracknet is None:
            raise RuntimeError("Call load_weights() before track_video()")

        def log(msg: str):
            print(msg)
            if log_callback:
                try:
                    log_callback(msg)
                except Exception:
                    pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Scale factors for mapping back to original resolution
        w_scale = orig_w / WIDTH
        h_scale = orig_h / HEIGHT

        # Step 1: Compute median background
        log(f"[TrackNet] Step 1/3: Computing median background ({max_bg_samples} samples)...")
        bg_frame = self._compute_median_background(video_path, total_frames, max_bg_samples)
        log(f"[TrackNet] Step 1/3: Background computation complete")

        # Step 2: Stream frames through TrackNet (no bulk memory allocation)
        total_sequences = math.ceil(total_frames / self.seq_len)
        total_batches = math.ceil(total_sequences / batch_size)
        log(f"[TrackNet] Step 2/3: Streaming GPU inference ({total_frames} frames, {total_batches} batches)...")
        raw_coords = self._run_tracknet_streaming(video_path, bg_frame, total_frames, batch_size, progress_callback, log_callback=log)

        del bg_frame

        # Step 3: Run InpaintNet for gap filling
        if self.inpaintnet is not None:
            log("[TrackNet] Step 3/3: Running trajectory inpainting...")
            raw_coords = self._run_inpaintnet(raw_coords, total_frames, log_callback=log)

        # Scale coordinates back to original resolution
        positions = {}
        for frame_num, coord in raw_coords.items():
            if coord["visible"]:
                positions[frame_num] = {
                    "x": coord["x"] * w_scale,
                    "y": coord["y"] * h_scale,
                    "visible": True,
                }
            else:
                positions[frame_num] = {"x": 0.0, "y": 0.0, "visible": False}

        visible_count = sum(1 for p in positions.values() if p["visible"])
        log(f"[TrackNet] Complete: shuttle detected in {visible_count}/{len(positions)} frames "
            f"({100*visible_count/max(len(positions),1):.1f}%)")

        return positions

    def _compute_median_background(
        self, video_path: str, total_frames: int, max_samples: int
    ) -> np.ndarray:
        """Sample frames and compute pixel-wise median for background estimation."""
        cap = cv2.VideoCapture(video_path)
        sample_count = min(total_frames, max_samples)
        # Sample evenly across the video
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        indices = np.unique(indices)

        sampled = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                resized = cv2.resize(frame, (WIDTH, HEIGHT))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                sampled.append(rgb)

        cap.release()

        if not sampled:
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Compute median in chunks to manage memory
        stacked = np.stack(sampled, axis=0)  # (N, H, W, 3)
        median_bg = np.median(stacked, axis=0).astype(np.uint8)
        return median_bg

    def _run_tracknet_streaming(
        self,
        video_path: str,
        bg_frame: np.ndarray,
        total_frames: int,
        batch_size: int,
        progress_callback=None,
        log_callback=None,
    ) -> Dict[int, Dict]:
        """
        Stream frames from video and run TrackNet inference without loading
        all frames into memory. Reads seq_len frames at a time, batches them,
        and runs inference incrementally.

        This keeps memory usage proportional to batch_size * seq_len instead
        of total_frames — critical for long videos (e.g. 50k+ frames).
        """
        all_coords: Dict[int, Dict] = {}
        bg_tensor = self._frame_to_tensor(bg_frame)  # (3, H, W)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_sequences = math.ceil(total_frames / self.seq_len)
        total_batches = math.ceil(total_sequences / batch_size)
        frame_idx = 0
        batch_count = 0

        while True:
            # Collect batch_size sequences by reading from video
            batch_seqs: List[List[np.ndarray]] = []
            batch_indices: List[List[int]] = []

            for _ in range(batch_size):
                seq_frames: List[np.ndarray] = []
                out_indices: List[int] = []

                for _ in range(self.seq_len):
                    ret, frame = cap.read()
                    if ret:
                        resized = cv2.resize(frame, (WIDTH, HEIGHT))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        seq_frames.append(rgb)
                        out_indices.append(frame_idx)
                        frame_idx += 1
                    else:
                        break

                if not seq_frames:
                    break

                # Pad incomplete sequence by repeating last frame
                while len(seq_frames) < self.seq_len:
                    seq_frames.append(seq_frames[-1])

                batch_seqs.append(seq_frames)
                batch_indices.append(out_indices)

            if not batch_seqs:
                break

            # Build input tensors and run inference
            input_tensors = []
            for seq in batch_seqs:
                frame_tensors = [self._frame_to_tensor(f) for f in seq]
                stacked = torch.cat(frame_tensors, dim=0)  # (seq_len*3, H, W)
                if self.bg_mode == "concat":
                    stacked = torch.cat([bg_tensor, stacked], dim=0)
                input_tensors.append(stacked)

            batch_tensor = torch.stack(input_tensors).to(self.device)

            with torch.no_grad():
                heatmaps = self.tracknet(batch_tensor)

            heatmaps_np = heatmaps.cpu().numpy()
            for seq_idx in range(len(batch_seqs)):
                out_indices = batch_indices[seq_idx]
                for hm_idx, fidx in enumerate(out_indices):
                    hm = heatmaps_np[seq_idx, hm_idx]
                    coord = self._heatmap_to_coord(hm)
                    all_coords[fidx] = coord

            batch_count += 1
            frames_processed = min(frame_idx, total_frames)
            pct = batch_count / total_batches * 100 if total_batches > 0 else 100

            if progress_callback and total_batches > 1:
                try:
                    progress_callback(pct)
                except Exception:
                    pass

            # Log progress every 10% or every 5 batches for shorter videos
            log_interval = max(1, total_batches // 10)
            if log_callback and (batch_count % log_interval == 0 or batch_count == total_batches):
                try:
                    log_callback(f"[TrackNet] Step 2/3: Batch {batch_count}/{total_batches} — {frames_processed}/{total_frames} frames ({pct:.0f}%)")
                except Exception:
                    pass

        cap.release()

        # Fill in any frames not predicted
        for i in range(total_frames):
            if i not in all_coords:
                all_coords[i] = {"x": 0.0, "y": 0.0, "visible": False}

        return all_coords

    def _run_inpaintnet(
        self, coords: Dict[int, Dict], total_frames: int, log_callback=None,
    ) -> Dict[int, Dict]:
        """
        Use InpaintNet to fill gaps in the trajectory.

        Processes the full trajectory in chunks, inpainting missing positions.
        """
        if self.inpaintnet is None:
            return coords

        # Build coordinate arrays
        xs = np.zeros(total_frames, dtype=np.float32)
        ys = np.zeros(total_frames, dtype=np.float32)
        vis = np.zeros(total_frames, dtype=np.float32)

        for i in range(total_frames):
            c = coords.get(i, {"x": 0, "y": 0, "visible": False})
            if c["visible"]:
                xs[i] = c["x"] / WIDTH   # normalize to 0-1
                ys[i] = c["y"] / HEIGHT
                vis[i] = 1.0

        # Only inpaint if we have some detections but also some gaps
        visible_count = int(vis.sum())
        if visible_count < 10 or visible_count >= total_frames * 0.99:
            return coords

        # Process in overlapping chunks to handle long videos
        chunk_size = 256
        stride = chunk_size // 2
        total_chunks = max(1, math.ceil(total_frames / stride))
        log_interval = max(1, total_chunks // 5)
        chunk_count = 0

        inpainted_xs = xs.copy()
        inpainted_ys = ys.copy()
        inpainted_vis = vis.copy()

        for start in range(0, total_frames, stride):
            end = min(start + chunk_size, total_frames)
            if end - start < 16:
                break

            chunk_x = xs[start:end]
            chunk_y = ys[start:end]
            chunk_v = vis[start:end]

            chunk_count += 1

            # Skip chunks that are entirely visible or entirely invisible
            chunk_vis_count = int(chunk_v.sum())
            if chunk_vis_count == 0 or chunk_vis_count == len(chunk_v):
                continue

            if log_callback and (chunk_count % log_interval == 0 or chunk_count == total_chunks):
                pct = min(100, chunk_count / total_chunks * 100)
                try:
                    log_callback(f"[TrackNet] Step 3/3: Inpainting chunk {chunk_count}/{total_chunks} ({pct:.0f}%)")
                except Exception:
                    pass

            # Pad to power of 8 for the 3-level U-Net pooling
            pad_len = ((len(chunk_x) + 7) // 8) * 8
            padded_x = np.zeros(pad_len, dtype=np.float32)
            padded_y = np.zeros(pad_len, dtype=np.float32)
            padded_v = np.zeros(pad_len, dtype=np.float32)
            padded_x[:len(chunk_x)] = chunk_x
            padded_y[:len(chunk_y)] = chunk_y
            padded_v[:len(chunk_v)] = chunk_v

            # Build input tensor: (1, 3, L) — [x, y, visibility]
            inp = np.stack([padded_x, padded_y, padded_v], axis=0)  # (3, L)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.inpaintnet(inp_tensor)  # (1, 2, L)

            out_np = out.cpu().numpy()[0]  # (2, L)
            pred_x = out_np[0, :len(chunk_x)]
            pred_y = out_np[1, :len(chunk_y)]

            for i in range(len(chunk_x)):
                frame_idx = start + i
                if vis[frame_idx] == 0 and pred_x[i] > 0.01 and pred_y[i] > 0.01:
                    if 0 < pred_x[i] < 1 and 0 < pred_y[i] < 1:
                        max_gap_dist = 0.15
                        is_continuous = True

                        for back in range(frame_idx - 1, max(0, frame_idx - 5) - 1, -1):
                            if inpainted_vis[back] > 0:
                                dx = abs(pred_x[i] - inpainted_xs[back])
                                dy = abs(pred_y[i] - inpainted_ys[back])
                                gap_frames = frame_idx - back
                                if dx > max_gap_dist * gap_frames or dy > max_gap_dist * gap_frames:
                                    is_continuous = False
                                break

                        if is_continuous:
                            for fwd in range(frame_idx + 1, min(total_frames, frame_idx + 5)):
                                if vis[fwd] > 0:
                                    dx = abs(pred_x[i] - xs[fwd])
                                    dy = abs(pred_y[i] - ys[fwd])
                                    gap_frames = fwd - frame_idx
                                    if dx > max_gap_dist * gap_frames or dy > max_gap_dist * gap_frames:
                                        is_continuous = False
                                    break

                        if is_continuous:
                            inpainted_xs[frame_idx] = pred_x[i]
                            inpainted_ys[frame_idx] = pred_y[i]
                            inpainted_vis[frame_idx] = 1.0

        # Build updated coords
        result = {}
        for i in range(total_frames):
            if inpainted_vis[i] > 0:
                result[i] = {
                    "x": float(inpainted_xs[i] * WIDTH),
                    "y": float(inpainted_ys[i] * HEIGHT),
                    "visible": True,
                }
            else:
                result[i] = {"x": 0.0, "y": 0.0, "visible": False}

        inpainted_count = int(inpainted_vis.sum()) - visible_count
        if inpainted_count > 0:
            print(f"[TrackNet] InpaintNet filled {inpainted_count} gap frames")

        return result

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 RGB frame to CHW float32 tensor normalized to [0,1]."""
        return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)

    @staticmethod
    def _heatmap_to_coord(heatmap: np.ndarray, threshold: float = 0.5,
                          max_area: int = 100) -> Dict:
        binary = (heatmap > threshold).astype(np.uint8)

        if binary.sum() == 0:
            return {"x": 0.0, "y": 0.0, "visible": False}

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        if num_labels <= 1:
            return {"x": 0.0, "y": 0.0, "visible": False}

        best_label = -1
        best_area = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > max_area:
                continue
            if area > best_area:
                best_area = area
                best_label = label

        if best_label < 0:
            return {"x": 0.0, "y": 0.0, "visible": False}

        mask = (labels == best_label)
        ys_idx, xs_idx = np.where(mask)
        weights = heatmap[ys_idx, xs_idx]
        total_weight = weights.sum()
        if total_weight > 0:
            cx = float(np.sum(xs_idx * weights) / total_weight)
            cy = float(np.sum(ys_idx * weights) / total_weight)
        else:
            cx = float(centroids[best_label][0])
            cy = float(centroids[best_label][1])

        return {"x": cx, "y": cy, "visible": True}
