#!/usr/bin/env python3
"""
GPU Convex IoU - Example Usage

This script demonstrates how to use the GPU IoU library for:
1. Single image IoU computation (rectangular matrix)
2. Batched IoU computation (multiple images in one GPU call)
3. NxN IoU matrix (for NMS)
"""

import numpy as np
import time

# Import the GPU IoU library
try:
    import convexiou_gpu
    print("✓ convexiou_gpu imported successfully")
except ImportError as e:
    print(f"✗ Failed to import convexiou_gpu: {e}")
    print("  Make sure to install with: pip install /path/to/gpu_convex_IoU")
    exit(1)


def example_single_image():
    """
    Example 1: Compute IoU matrix for a single image
    
    This is the basic use case: given detections and ground truths for one image,
    compute the IoU matrix.
    """
    print("\n" + "="*60)
    print("Example 1: Single Image IoU")
    print("="*60)
    
    # Create sample detections: [x_center, y_center, width, height, angle_rad]
    detections = np.array([
        [100.0, 100.0, 50.0, 30.0, 0.5],      # Detection 1
        [200.0, 150.0, 40.0, 40.0, 0.0],      # Detection 2
        [105.0, 98.0, 48.0, 32.0, 0.52],      # Detection 3 (overlaps with GT 1)
    ], dtype=np.float64)
    
    # Create sample ground truths
    ground_truths = np.array([
        [102.0, 101.0, 52.0, 31.0, 0.48],     # GT 1 (overlaps with det 1 & 3)
        [300.0, 300.0, 60.0, 60.0, 0.0],      # GT 2 (no overlap)
    ], dtype=np.float64)
    
    print(f"Detections: {detections.shape[0]} boxes")
    print(f"Ground Truths: {ground_truths.shape[0]} boxes")
    
    # Compute IoU matrix
    t0 = time.perf_counter()
    iou_matrix = convexiou_gpu.calculate_iou_rectangular_numpy_from_numpy(
        detections, 
        ground_truths, 
        num_points=16
    )
    elapsed = time.perf_counter() - t0
    
    print(f"\nIoU Matrix (shape {iou_matrix.shape}):")
    print(iou_matrix)
    print(f"\nComputation time: {elapsed*1000:.3f} ms")
    
    # Interpret results
    print("\nInterpretation:")
    for i in range(len(detections)):
        best_gt = np.argmax(iou_matrix[i])
        best_iou = iou_matrix[i, best_gt]
        if best_iou > 0.5:
            print(f"  Detection {i} matches GT {best_gt} (IoU={best_iou:.3f})")
        else:
            print(f"  Detection {i} has no match (best IoU={best_iou:.3f})")


def example_batched_evaluation():
    """
    Example 2: Batched IoU computation for multiple images
    
    This is the OPTIMAL method for detector evaluation. Instead of calling
    the GPU kernel for each image, we batch all images into a single call.
    This provides 3-4x speedup over per-image calls.
    """
    print("\n" + "="*60)
    print("Example 2: Batched IoU (Multiple Images)")
    print("="*60)
    
    # Simulate detections and GTs for 5 images
    np.random.seed(42)
    num_images = 5
    
    all_dets_per_image = []
    all_gts_per_image = []
    
    for img_idx in range(num_images):
        # Random number of detections (3-10) and GTs (2-5) per image
        n_det = np.random.randint(3, 11)
        n_gt = np.random.randint(2, 6)
        
        # Generate random boxes
        dets = np.random.rand(n_det, 5) * [800, 600, 100, 100, 2*np.pi]
        dets[:, 4] -= np.pi  # Center angle around 0
        
        gts = np.random.rand(n_gt, 5) * [800, 600, 100, 100, 2*np.pi]
        gts[:, 4] -= np.pi
        
        all_dets_per_image.append(dets.astype(np.float64))
        all_gts_per_image.append(gts.astype(np.float64))
        
        print(f"  Image {img_idx}: {n_det} dets, {n_gt} GTs")
    
    # === Batched Method (FAST) ===
    print("\n--- Batched GPU Method ---")
    
    # Step 1: Concatenate all boxes
    all_dets = np.vstack(all_dets_per_image)
    all_gts = np.vstack(all_gts_per_image)
    
    # Step 2: Build pair_info array
    # Each row: [det_offset, gt_offset, out_offset, n_det, n_gt]
    pair_info = []
    det_offset = 0
    gt_offset = 0
    out_offset = 0
    
    for dets, gts in zip(all_dets_per_image, all_gts_per_image):
        n_det = len(dets)
        n_gt = len(gts)
        pair_info.append([det_offset, gt_offset, out_offset, n_det, n_gt])
        det_offset += n_det
        gt_offset += n_gt
        out_offset += n_det * n_gt
    
    pair_info = np.array(pair_info, dtype=np.int32)
    
    print(f"Total detections: {len(all_dets)}")
    print(f"Total ground truths: {len(all_gts)}")
    print(f"Total IoU computations: {out_offset}")
    
    # Step 3: Single GPU call for all images
    t0 = time.perf_counter()
    results, total_size = convexiou_gpu.calculate_iou_batched_rectangular(
        all_dets, all_gts, pair_info, num_points=16
    )
    batched_time = time.perf_counter() - t0
    
    print(f"Batched computation time: {batched_time*1000:.3f} ms")
    
    # Step 4: Extract per-image IoU matrices
    for img_idx in range(num_images):
        det_off, gt_off, out_off, n_det, n_gt = pair_info[img_idx]
        if n_det > 0 and n_gt > 0:
            iou_matrix = results[out_off:out_off + n_det * n_gt].reshape(n_det, n_gt)
            max_iou = iou_matrix.max() if iou_matrix.size > 0 else 0
            print(f"  Image {img_idx}: IoU matrix {iou_matrix.shape}, max IoU = {max_iou:.3f}")
    
    # === Per-Image Method (for comparison) ===
    print("\n--- Per-Image GPU Method (for comparison) ---")
    
    t0 = time.perf_counter()
    for dets, gts in zip(all_dets_per_image, all_gts_per_image):
        if len(dets) > 0 and len(gts) > 0:
            _ = convexiou_gpu.calculate_iou_rectangular_numpy_from_numpy(
                dets, gts, num_points=16
            )
    per_image_time = time.perf_counter() - t0
    
    print(f"Per-image computation time: {per_image_time*1000:.3f} ms")
    print(f"Speedup from batching: {per_image_time/batched_time:.2f}x")


def example_nms_matrix():
    """
    Example 3: NxN IoU matrix for NMS
    
    For Non-Maximum Suppression, we need pairwise IoU between all detections.
    """
    print("\n" + "="*60)
    print("Example 3: NxN Matrix for NMS")
    print("="*60)
    
    # Create detections with some overlapping boxes
    detections = np.array([
        [100.0, 100.0, 50.0, 30.0, 0.5],
        [105.0, 102.0, 48.0, 32.0, 0.52],   # Overlaps with box 0
        [200.0, 200.0, 60.0, 40.0, 0.0],
        [203.0, 198.0, 58.0, 42.0, 0.05],   # Overlaps with box 2
        [400.0, 400.0, 70.0, 50.0, 1.0],    # No overlap
    ], dtype=np.float64)
    
    print(f"Computing NxN IoU matrix for {len(detections)} boxes")
    
    t0 = time.perf_counter()
    iou_matrix = convexiou_gpu.calculate_iou_matrix_numpy_from_numpy(
        detections, num_points=16
    )
    elapsed = time.perf_counter() - t0
    
    print(f"\nIoU Matrix:\n{iou_matrix}")
    print(f"\nComputation time: {elapsed*1000:.3f} ms")
    
    # Simple NMS demonstration
    nms_threshold = 0.5
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])  # Confidence scores
    
    print(f"\nSimple NMS (threshold={nms_threshold}):")
    keep = []
    suppressed = set()
    
    order = np.argsort(scores)[::-1]
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        # Suppress overlapping boxes
        for j in order:
            if j != i and j not in suppressed:
                if iou_matrix[i, j] > nms_threshold:
                    suppressed.add(j)
                    print(f"  Box {j} suppressed by box {i} (IoU={iou_matrix[i,j]:.3f})")
    
    print(f"Kept boxes: {keep}")


def example_large_scale():
    """
    Example 4: Large-scale benchmark
    
    Demonstrates performance on a realistic workload.
    """
    print("\n" + "="*60)
    print("Example 4: Large-Scale Benchmark")
    print("="*60)
    
    # Simulate 100 images with ~50 detections and ~10 GTs each
    np.random.seed(123)
    num_images = 100
    
    all_dets = []
    all_gts = []
    pair_info = []
    
    det_offset = gt_offset = out_offset = 0
    total_pairs = 0
    
    for _ in range(num_images):
        n_det = np.random.randint(30, 70)
        n_gt = np.random.randint(5, 15)
        
        dets = np.random.rand(n_det, 5) * [1024, 768, 150, 150, 2*np.pi]
        dets[:, 4] -= np.pi
        
        gts = np.random.rand(n_gt, 5) * [1024, 768, 150, 150, 2*np.pi]
        gts[:, 4] -= np.pi
        
        all_dets.append(dets)
        all_gts.append(gts)
        
        pair_info.append([det_offset, gt_offset, out_offset, n_det, n_gt])
        
        det_offset += n_det
        gt_offset += n_gt
        out_offset += n_det * n_gt
        total_pairs += n_det * n_gt
    
    all_dets = np.vstack(all_dets).astype(np.float64)
    all_gts = np.vstack(all_gts).astype(np.float64)
    pair_info = np.array(pair_info, dtype=np.int32)
    
    print(f"Images: {num_images}")
    print(f"Total detections: {len(all_dets)}")
    print(f"Total ground truths: {len(all_gts)}")
    print(f"Total IoU pairs: {total_pairs:,}")
    
    # Warm-up GPU
    _ = convexiou_gpu.calculate_iou_batched_rectangular(
        all_dets[:100], all_gts[:50], 
        np.array([[0, 0, 0, 100, 50]], dtype=np.int32),
        num_points=16
    )
    
    # Benchmark
    num_runs = 5
    times = []
    
    for run in range(num_runs):
        t0 = time.perf_counter()
        results, _ = convexiou_gpu.calculate_iou_batched_rectangular(
            all_dets, all_gts, pair_info, num_points=16
        )
        times.append(time.perf_counter() - t0)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    pairs_per_sec = total_pairs / avg_time
    
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Throughput: {pairs_per_sec/1e6:.2f} million IoU pairs/sec")


if __name__ == "__main__":
    print("GPU Convex IoU - Example Usage")
    print("="*60)
    
    # Check available functions
    print("\nAvailable functions:")
    for name in dir(convexiou_gpu):
        if not name.startswith('_'):
            print(f"  - {name}")
    
    # Run examples
    example_single_image()
    example_batched_evaluation()
    example_nms_matrix()
    example_large_scale()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
