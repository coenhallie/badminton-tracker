#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Badminton Tracker
=====================================================

Measures performance of various components and compares
optimized vs baseline implementations.

Usage:
    # Run all benchmarks
    python benchmark_performance.py
    
    # Benchmark specific component
    python benchmark_performance.py --component model
    python benchmark_performance.py --component frames
    python benchmark_performance.py --component pipeline
    
    # With custom video
    python benchmark_performance.py --video path/to/video.mp4
    
    # Quick test (fewer iterations)
    python benchmark_performance.py --quick
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import components
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed, skipping YOLO benchmarks")

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

try:
    from performance_optimizations import (
        PerformanceProfiler,
        AsyncFrameBuffer,
        AdaptiveFrameSkipper,
        BatchModelInference,
        PreallocatedArrays,
        ObjectPool,
    )
    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False
    print("Warning: performance_optimizations module not found")


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.timings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, any] = {}
    
    def add_timing(self, metric: str, value_ms: float):
        if metric not in self.timings:
            self.timings[metric] = []
        self.timings[metric].append(value_ms)
    
    def get_stats(self, metric: str) -> Dict[str, float]:
        if metric not in self.timings or not self.timings[metric]:
            return {}
        
        values = self.timings[metric]
        return {
            "count": len(values),
            "mean_ms": np.mean(values),
            "std_ms": np.std(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "p50_ms": np.percentile(values, 50),
            "p95_ms": np.percentile(values, 95),
            "p99_ms": np.percentile(values, 99),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        return {metric: self.get_stats(metric) for metric in self.timings}
    
    def print_report(self):
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.name}")
        print(f"{'='*60}")
        
        if self.metadata:
            print("\nConfiguration:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        
        print("\nResults:")
        for metric, stats in self.get_all_stats().items():
            if stats:
                print(f"\n  {metric}:")
                print(f"    Mean:  {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
                print(f"    P50:   {stats['p50_ms']:.2f}ms")
                print(f"    P95:   {stats['p95_ms']:.2f}ms")
                print(f"    Range: {stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms")
                if stats['mean_ms'] > 0:
                    fps = 1000 / stats['mean_ms']
                    print(f"    ~FPS:  {fps:.1f}")


def benchmark_frame_reading(video_path: str, iterations: int = 100) -> Tuple[BenchmarkResults, BenchmarkResults]:
    """Compare synchronous vs asynchronous frame reading."""
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None, None
    
    # Sync benchmark
    sync_results = BenchmarkResults("Synchronous Frame Reading")
    cap = cv2.VideoCapture(video_path)
    sync_results.metadata["video"] = video_path
    sync_results.metadata["iterations"] = iterations
    
    for i in range(iterations):
        start = time.perf_counter()
        ret, frame = cap.read()
        elapsed = (time.perf_counter() - start) * 1000
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        sync_results.add_timing("read_frame", elapsed)
    
    cap.release()
    
    # Async benchmark
    async_results = BenchmarkResults("Asynchronous Frame Reading (Buffered)")
    
    if HAS_OPTIMIZATIONS:
        async_results.metadata["video"] = video_path
        async_results.metadata["iterations"] = iterations
        async_results.metadata["buffer_size"] = 10
        
        buffer = AsyncFrameBuffer(video_path, buffer_size=10)
        buffer.start()
        time.sleep(0.5)  # Let buffer fill
        
        count = 0
        for frame, frame_num in buffer:
            if count >= iterations:
                break
            
            start = time.perf_counter()
            # Frame already loaded, simulate "getting" it
            _ = frame.shape
            elapsed = (time.perf_counter() - start) * 1000
            
            async_results.add_timing("get_frame", elapsed)
            count += 1
        
        buffer.stop()
    else:
        async_results.metadata["error"] = "performance_optimizations not available"
    
    return sync_results, async_results


def benchmark_model_inference(
    model_path: str = "yolo26n-pose.pt",
    iterations: int = 50,
    batch_sizes: List[int] = [1, 2, 4]
) -> Dict[str, BenchmarkResults]:
    """Benchmark YOLO model inference with different batch sizes."""
    
    if not HAS_YOLO:
        print("YOLO not available")
        return {}
    
    results = {}
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
        return {}
    
    # Create dummy input
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up model...")
    for _ in range(5):
        model(dummy_frame, verbose=False)
    
    for batch_size in batch_sizes:
        result = BenchmarkResults(f"Model Inference (batch={batch_size})")
        result.metadata["model"] = model_path
        result.metadata["batch_size"] = batch_size
        result.metadata["iterations"] = iterations
        result.metadata["cuda_available"] = CUDA_AVAILABLE
        
        batch = [dummy_frame.copy() for _ in range(batch_size)]
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            if batch_size == 1:
                model(batch[0], verbose=False)
            else:
                model(batch, verbose=False)
            
            elapsed = (time.perf_counter() - start) * 1000
            result.add_timing("inference", elapsed)
            result.add_timing("per_frame", elapsed / batch_size)
        
        results[f"batch_{batch_size}"] = result
    
    return results


def benchmark_adaptive_frame_skip(video_path: str, num_frames: int = 500) -> Tuple[BenchmarkResults, BenchmarkResults]:
    """Compare fixed vs adaptive frame skipping."""
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None, None
    
    # Fixed skip benchmark
    fixed_results = BenchmarkResults("Fixed Frame Skipping (rate=2)")
    fixed_results.metadata["skip_rate"] = 2
    fixed_results.metadata["num_frames"] = num_frames
    
    cap = cv2.VideoCapture(video_path)
    fixed_processed = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        if i % 2 == 0:
            fixed_processed += 1
    
    cap.release()
    fixed_results.metadata["frames_processed"] = fixed_processed
    fixed_results.metadata["skip_percentage"] = (1 - fixed_processed / num_frames) * 100
    
    # Adaptive skip benchmark
    adaptive_results = BenchmarkResults("Adaptive Frame Skipping")
    
    if HAS_OPTIMIZATIONS:
        adaptive_results.metadata["num_frames"] = num_frames
        
        skipper = AdaptiveFrameSkipper(
            base_skip_rate=2,
            motion_threshold=5.0,
            min_skip=1,
            max_skip=5
        )
        
        cap = cv2.VideoCapture(video_path)
        adaptive_processed = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if not skipper.should_skip(frame, i):
                adaptive_processed += 1
        
        cap.release()
        adaptive_results.metadata["frames_processed"] = adaptive_processed
        adaptive_results.metadata["skip_percentage"] = (1 - adaptive_processed / num_frames) * 100
    else:
        adaptive_results.metadata["error"] = "performance_optimizations not available"
    
    return fixed_results, adaptive_results


def benchmark_memory_allocation(iterations: int = 1000) -> Tuple[BenchmarkResults, BenchmarkResults]:
    """Compare standard vs pooled object allocation."""
    
    # Standard allocation
    standard_results = BenchmarkResults("Standard Dict Allocation")
    standard_results.metadata["iterations"] = iterations
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        frame_data = {
            "frame": 0,
            "timestamp": 0.0,
            "players": [],
            "shuttlecocks": [],
            "court_detected": False,
            "badminton_detections": None,
        }
        
        # Simulate populating
        for i in range(2):
            frame_data["players"].append({
                "player_id": i,
                "keypoints": [{"x": 0, "y": 0, "confidence": 0.9} for _ in range(17)],
                "center": {"x": 320, "y": 240},
            })
        
        elapsed = (time.perf_counter() - start) * 1000
        standard_results.add_timing("allocation", elapsed)
    
    # Pooled allocation
    pooled_results = BenchmarkResults("Object Pool Allocation")
    
    if HAS_OPTIMIZATIONS:
        pooled_results.metadata["iterations"] = iterations
        pooled_results.metadata["pool_size"] = 10
        
        def create_frame_data():
            return {
                "frame": 0,
                "timestamp": 0.0,
                "players": [],
                "shuttlecocks": [],
                "court_detected": False,
                "badminton_detections": None,
            }
        
        pool = ObjectPool(create_frame_data, pool_size=10)
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            frame_data = pool.acquire()
            
            # Reset and populate
            frame_data["frame"] = 0
            frame_data["timestamp"] = 0.0
            frame_data["players"].clear()
            frame_data["shuttlecocks"].clear()
            frame_data["court_detected"] = False
            frame_data["badminton_detections"] = None
            
            for i in range(2):
                frame_data["players"].append({
                    "player_id": i,
                    "keypoints": [{"x": 0, "y": 0, "confidence": 0.9} for _ in range(17)],
                    "center": {"x": 320, "y": 240},
                })
            
            pool.release(frame_data)
            
            elapsed = (time.perf_counter() - start) * 1000
            pooled_results.add_timing("pool_acquire_release", elapsed)
    else:
        pooled_results.metadata["error"] = "performance_optimizations not available"
    
    return standard_results, pooled_results


def benchmark_numpy_operations(iterations: int = 500) -> Tuple[BenchmarkResults, BenchmarkResults]:
    """Compare standard vs pre-allocated numpy operations."""
    
    height, width = 1080, 1920
    
    # Standard allocation
    standard_results = BenchmarkResults("Standard Numpy Allocation")
    standard_results.metadata["resolution"] = f"{width}x{height}"
    standard_results.metadata["iterations"] = iterations
    
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # New allocation
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # New allocation
        _ = blurred.mean()
        
        elapsed = (time.perf_counter() - start) * 1000
        standard_results.add_timing("operations", elapsed)
    
    # Pre-allocated
    preallocated_results = BenchmarkResults("Pre-allocated Numpy Operations")
    
    if HAS_OPTIMIZATIONS:
        preallocated_results.metadata["resolution"] = f"{width}x{height}"
        preallocated_results.metadata["iterations"] = iterations
        
        arrays = PreallocatedArrays(height, width)
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=arrays.gray)
            cv2.GaussianBlur(arrays.gray, (5, 5), 0, dst=arrays.blurred)
            _ = arrays.blurred.mean()
            
            elapsed = (time.perf_counter() - start) * 1000
            preallocated_results.add_timing("operations", elapsed)
    else:
        preallocated_results.metadata["error"] = "performance_optimizations not available"
    
    return standard_results, preallocated_results


def benchmark_full_pipeline(video_path: str, num_frames: int = 100) -> BenchmarkResults:
    """Benchmark the full processing pipeline."""
    
    if not os.path.exists(video_path) or not HAS_YOLO:
        return None
    
    results = BenchmarkResults("Full Pipeline")
    results.metadata["video"] = video_path
    results.metadata["num_frames"] = num_frames
    
    # Load models
    try:
        pose_model = YOLO("yolo26n-pose.pt")
    except:
        pose_model = None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    profiler = PerformanceProfiler() if HAS_OPTIMIZATIONS else None
    
    for i in range(num_frames):
        total_start = time.perf_counter()
        
        # Frame reading
        read_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        results.add_timing("frame_read", (time.perf_counter() - read_start) * 1000)
        
        # Pose inference
        if pose_model:
            inference_start = time.perf_counter()
            pose_results = pose_model(frame, verbose=False)
            results.add_timing("pose_inference", (time.perf_counter() - inference_start) * 1000)
            
            # Parse results
            parse_start = time.perf_counter()
            for result in pose_results:
                if result.keypoints is not None:
                    kpts = result.keypoints.data.cpu().numpy()
            results.add_timing("result_parsing", (time.perf_counter() - parse_start) * 1000)
        
        # Drawing
        draw_start = time.perf_counter()
        if pose_model and pose_results:
            annotated = pose_results[0].plot()
        else:
            annotated = frame.copy()
        results.add_timing("drawing", (time.perf_counter() - draw_start) * 1000)
        
        results.add_timing("total", (time.perf_counter() - total_start) * 1000)
    
    cap.release()
    
    return results


def compare_results(baseline: BenchmarkResults, optimized: BenchmarkResults, metric: str = None):
    """Compare two benchmark results and print improvement."""
    
    if baseline is None or optimized is None:
        print("Cannot compare: one or both results are None")
        return
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: {baseline.name} vs {optimized.name}")
    print(f"{'='*60}")
    
    metrics = [metric] if metric else list(set(baseline.timings.keys()) & set(optimized.timings.keys()))
    
    for m in metrics:
        base_stats = baseline.get_stats(m)
        opt_stats = optimized.get_stats(m)
        
        if not base_stats or not opt_stats:
            continue
        
        base_mean = base_stats["mean_ms"]
        opt_mean = opt_stats["mean_ms"]
        
        if base_mean > 0:
            improvement = ((base_mean - opt_mean) / base_mean) * 100
            speedup = base_mean / opt_mean if opt_mean > 0 else float('inf')
        else:
            improvement = 0
            speedup = 1.0
        
        print(f"\n{m}:")
        print(f"  Baseline:   {base_mean:.3f}ms")
        print(f"  Optimized:  {opt_mean:.3f}ms")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Speedup:     {speedup:.2f}x")


def run_all_benchmarks(video_path: str = None, quick: bool = False):
    """Run all benchmarks and print comprehensive report."""
    
    iterations = 20 if quick else 100
    frame_iterations = 100 if quick else 500
    
    print("\n" + "="*60)
    print("BADMINTON TRACKER PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    print(f"\nQuick mode: {quick}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"Optimizations available: {HAS_OPTIMIZATIONS}")
    
    all_results = {}
    
    # 1. Frame reading benchmark
    if video_path and os.path.exists(video_path):
        print("\n\n>>> Benchmarking Frame Reading...")
        sync, async_buf = benchmark_frame_reading(video_path, iterations)
        if sync:
            sync.print_report()
            all_results["frame_sync"] = sync.get_all_stats()
        if async_buf:
            async_buf.print_report()
            all_results["frame_async"] = async_buf.get_all_stats()
            if sync:
                compare_results(sync, async_buf, "read_frame" if "read_frame" in sync.timings else "get_frame")
    
    # 2. Model inference benchmark
    if HAS_YOLO:
        print("\n\n>>> Benchmarking Model Inference...")
        model_results = benchmark_model_inference(iterations=iterations // 2)
        for name, result in model_results.items():
            result.print_report()
            all_results[f"model_{name}"] = result.get_all_stats()
    
    # 3. Frame skipping benchmark
    if video_path and os.path.exists(video_path):
        print("\n\n>>> Benchmarking Frame Skipping...")
        fixed, adaptive = benchmark_adaptive_frame_skip(video_path, frame_iterations)
        if fixed:
            fixed.print_report()
            all_results["skip_fixed"] = fixed.metadata
        if adaptive:
            adaptive.print_report()
            all_results["skip_adaptive"] = adaptive.metadata
    
    # 4. Memory allocation benchmark
    print("\n\n>>> Benchmarking Memory Allocation...")
    standard, pooled = benchmark_memory_allocation(iterations * 10)
    if standard:
        standard.print_report()
        all_results["alloc_standard"] = standard.get_all_stats()
    if pooled:
        pooled.print_report()
        all_results["alloc_pooled"] = pooled.get_all_stats()
        compare_results(standard, pooled)
    
    # 5. Numpy operations benchmark
    print("\n\n>>> Benchmarking Numpy Operations...")
    standard_np, prealloc_np = benchmark_numpy_operations(iterations * 5)
    if standard_np:
        standard_np.print_report()
        all_results["numpy_standard"] = standard_np.get_all_stats()
    if prealloc_np:
        prealloc_np.print_report()
        all_results["numpy_prealloc"] = prealloc_np.get_all_stats()
        compare_results(standard_np, prealloc_np)
    
    # 6. Full pipeline benchmark
    if video_path and os.path.exists(video_path) and HAS_YOLO:
        print("\n\n>>> Benchmarking Full Pipeline...")
        pipeline = benchmark_full_pipeline(video_path, iterations // 2)
        if pipeline:
            pipeline.print_report()
            all_results["pipeline"] = pipeline.get_all_stats()
    
    # Summary
    print("\n\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Calculate overall metrics
    if "pipeline" in all_results and "total" in all_results["pipeline"]:
        total_stats = all_results["pipeline"]["total"]
        print(f"\nFull Pipeline Performance:")
        print(f"  Average frame time: {total_stats['mean_ms']:.2f}ms")
        print(f"  Achievable FPS: {1000 / total_stats['mean_ms']:.1f}")
        print(f"  P95 latency: {total_stats['p95_ms']:.2f}ms")
    
    # Save results to JSON
    results_path = Path(__file__).parent / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for Badminton Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_performance.py
    python benchmark_performance.py --video uploads/match.mp4
    python benchmark_performance.py --quick
    python benchmark_performance.py --component model
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for benchmarking"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with fewer iterations"
    )
    
    parser.add_argument(
        "--component",
        type=str,
        choices=["frames", "model", "skip", "memory", "numpy", "pipeline", "all"],
        default="all",
        help="Specific component to benchmark"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarks"
    )
    
    args = parser.parse_args()
    
    # Find a test video if not specified
    if not args.video:
        uploads_dir = Path(__file__).parent / "uploads"
        if uploads_dir.exists():
            videos = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi"))
            if videos:
                args.video = str(videos[0])
                print(f"Using video: {args.video}")
    
    if args.component == "all":
        run_all_benchmarks(args.video, args.quick)
    else:
        iterations = args.iterations // 5 if args.quick else args.iterations
        
        if args.component == "frames" and args.video:
            sync, async_buf = benchmark_frame_reading(args.video, iterations)
            if sync:
                sync.print_report()
            if async_buf:
                async_buf.print_report()
                compare_results(sync, async_buf)
        
        elif args.component == "model":
            results = benchmark_model_inference(iterations=iterations // 2)
            for result in results.values():
                result.print_report()
        
        elif args.component == "skip" and args.video:
            fixed, adaptive = benchmark_adaptive_frame_skip(args.video, iterations * 5)
            if fixed:
                fixed.print_report()
            if adaptive:
                adaptive.print_report()
        
        elif args.component == "memory":
            standard, pooled = benchmark_memory_allocation(iterations * 10)
            if standard:
                standard.print_report()
            if pooled:
                pooled.print_report()
                compare_results(standard, pooled)
        
        elif args.component == "numpy":
            standard, prealloc = benchmark_numpy_operations(iterations * 5)
            if standard:
                standard.print_report()
            if prealloc:
                prealloc.print_report()
                compare_results(standard, prealloc)
        
        elif args.component == "pipeline" and args.video:
            result = benchmark_full_pipeline(args.video, iterations)
            if result:
                result.print_report()


if __name__ == "__main__":
    main()
