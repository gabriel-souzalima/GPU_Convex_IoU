# GPU Convex IoU

GPU-accelerated Intersection over Union (IoU) computation for ellipses using convex polygon approximation. This library provides massive speedups (100x+) over CPU-based methods like Shapely for rotated bounding box evaluation.

## Performance

Benchmarked on DIOR dataset with 2,048 images and 134,625 detections (~1 million IoU pairs):

| Method | Wall Time | IoU-only Time | Speedup |
|--------|-----------|---------------|---------|
| Shapely (CPU) | 550 sec | 545 sec | 1x |
| GPU Per-image | 18 sec | ~12 sec | 30x |
| **GPU Batched** | **5 sec** | **2.3 sec** | **110x (wall) / 240x (IoU)** |

**Accuracy**: Maximum difference vs Shapely is 0.0002 (0.02%) - negligible for all practical purposes.

## How It Works

1. **Ellipse from OBB**: Oriented Bounding Boxes (OBB) are converted to equivalent ellipses using:
   ```
   ellipse_a = width / sqrt(π)
   ellipse_b = height / sqrt(π)
   ```
   This ensures the ellipse has the same area as the OBB.

2. **Polygon Approximation**: Each ellipse is approximated as a convex polygon with N points (default: 16).

3. **GPU IoU**: Polygon intersection and union areas are computed entirely on GPU using CUDA.

## Installation

### Requirements

- CUDA Toolkit (11.0+)
- Python 3.7+
- pybind11
- numpy

### Build

```bash
# Set CUDA architecture for your GPU (optional, default: sm_75)
export CUDA_ARCH=sm_86  # RTX 3000 series

# Install
cd gpu_convex_IoU
pip install pybind11 numpy
pip install .
```

### Docker (Recommended)

```dockerfile
# Inside a CUDA-enabled container
pip install pybind11 numpy
pip install /path/to/gpu_convex_IoU
```

## API Reference

### Input Format

All functions expect ellipse/OBB data as arrays with shape `(N, 5)`:
```
[x_center, y_center, width, height, angle_radians]
```

Where:
- `x_center, y_center`: Center coordinates
- `width, height`: OBB dimensions (NOT ellipse semi-axes)
- `angle_radians`: Rotation angle in radians

### Functions

#### 1. Single Image: `calculate_iou_rectangular_numpy_from_numpy`

Compute IoU matrix between detections and ground truths for **one image**:

```python
import numpy as np
import convexiou_gpu

# Detections: (N_det, 5) - [x, y, w, h, angle_rad]
detections = np.array([
    [100, 100, 50, 30, 0.5],
    [200, 150, 40, 40, 0.0],
], dtype=np.float64)

# Ground truths: (N_gt, 5)
ground_truths = np.array([
    [105, 102, 48, 32, 0.52],
    [300, 300, 60, 60, 0.0],
], dtype=np.float64)

# Compute IoU matrix (N_det x N_gt)
iou_matrix = convexiou_gpu.calculate_iou_rectangular_numpy_from_numpy(
    detections, 
    ground_truths, 
    num_points=16  # Polygon approximation points
)

print(iou_matrix.shape)  # (2, 2)
print(iou_matrix)        # [[0.85, 0.0], [0.0, 0.0]]
```

#### 2. Batched (OPTIMAL): `calculate_iou_batched_rectangular`

Compute IoU for **ALL images** in a single GPU call. This is 3-4x faster than per-image calls:

```python
import numpy as np
import convexiou_gpu

# Concatenate ALL detections and ground truths
all_dets = np.vstack([dets_img0, dets_img1, dets_img2, ...])  # (total_dets, 5)
all_gts = np.vstack([gts_img0, gts_img1, gts_img2, ...])      # (total_gts, 5)

# Build pair_info: [det_offset, gt_offset, out_offset, n_det, n_gt] per image
pair_info = []
det_offset = 0
gt_offset = 0
out_offset = 0

for img_idx in range(num_images):
    n_det = len(dets_per_image[img_idx])
    n_gt = len(gts_per_image[img_idx])
    
    pair_info.append([det_offset, gt_offset, out_offset, n_det, n_gt])
    
    det_offset += n_det
    gt_offset += n_gt
    out_offset += n_det * n_gt

pair_info = np.array(pair_info, dtype=np.int32)

# Single GPU call for all images
results, total_size = convexiou_gpu.calculate_iou_batched_rectangular(
    all_dets.astype(np.float64),
    all_gts.astype(np.float64),
    pair_info,
    num_points=16
)

# Extract per-image IoU matrices
for img_idx in range(num_images):
    det_off, gt_off, out_off, n_det, n_gt = pair_info[img_idx]
    if n_det > 0 and n_gt > 0:
        iou_matrix = results[out_off:out_off + n_det * n_gt].reshape(n_det, n_gt)
```

#### 3. NxN Matrix (for NMS): `calculate_iou_matrix_numpy_from_numpy`

Compute pairwise IoU for all boxes (useful for NMS):

```python
boxes = np.array([...], dtype=np.float64)  # (N, 5)
iou_matrix = convexiou_gpu.calculate_iou_matrix_numpy_from_numpy(boxes, num_points=16)
# Returns (N, N) matrix
```

## Integrating with a Detector

### Required Changes

To integrate GPU IoU with your rotated object detector's evaluation code:

#### 1. Import the Library

```python
# Try to import GPU IoU, fall back gracefully if not available
try:
    import convexiou_gpu
    HAS_GPU_IOU = True
except ImportError:
    convexiou_gpu = None
    HAS_GPU_IOU = False
```

#### 2. Convert OBB to Ellipse Format

Your detector likely outputs OBBs as `[x, y, w, h, angle]`. The GPU IoU library expects the same format - it handles the OBB→ellipse conversion internally:

```python
def prepare_boxes_for_gpu_iou(boxes):
    """
    Ensure boxes are in the correct format for GPU IoU.
    
    Input:  boxes with shape (N, 5+) as [x, y, w, h, angle_rad, ...]
    Output: boxes with shape (N, 5) as [x, y, w, h, angle_rad]
    
    Note: angle must be in RADIANS. If your detector outputs degrees:
        boxes[:, 4] = np.deg2rad(boxes[:, 4])
    """
    return np.ascontiguousarray(boxes[:, :5], dtype=np.float64)
```

#### 3. Replace Your IoU Function

**Before (CPU with Shapely):**
```python
def compute_iou_matrix(det_boxes, gt_boxes):
    # Slow loop-based Shapely computation
    ious = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = shapely_iou(det, gt)
    return ious
```

**After (GPU):**
```python
def compute_iou_matrix(det_boxes, gt_boxes, use_gpu=True):
    if use_gpu and HAS_GPU_IOU and len(det_boxes) > 0 and len(gt_boxes) > 0:
        dets = prepare_boxes_for_gpu_iou(det_boxes)
        gts = prepare_boxes_for_gpu_iou(gt_boxes)
        return convexiou_gpu.calculate_iou_rectangular_numpy_from_numpy(
            dets, gts, num_points=16
        )
    else:
        # Fallback to CPU
        return compute_iou_matrix_cpu(det_boxes, gt_boxes)
```

#### 4. For Evaluation Loops (OPTIMAL)

For mAP evaluation over many images, use the batched API:

```python
def evaluate_all_images_batched(all_detections, all_ground_truths):
    """
    Evaluate IoU for all images in a single GPU call.
    
    all_detections: list of (N_i, 5) arrays, one per image
    all_ground_truths: list of (M_i, 5) arrays, one per image
    """
    # Concatenate all boxes
    all_dets_list = [d for d in all_detections if len(d) > 0]
    all_gts_list = [g for g in all_ground_truths if len(g) > 0]
    
    if not all_dets_list or not all_gts_list:
        return []
    
    all_dets = np.vstack(all_dets_list).astype(np.float64)
    all_gts = np.vstack(all_gts_list).astype(np.float64)
    
    # Build pair info
    pair_info = []
    det_offset = gt_offset = out_offset = 0
    
    for dets, gts in zip(all_detections, all_ground_truths):
        n_det, n_gt = len(dets), len(gts)
        pair_info.append([det_offset, gt_offset, out_offset, n_det, n_gt])
        det_offset += n_det
        gt_offset += n_gt
        out_offset += n_det * n_gt
    
    pair_info = np.array(pair_info, dtype=np.int32)
    
    # Single GPU call
    results, _ = convexiou_gpu.calculate_iou_batched_rectangular(
        all_dets, all_gts, pair_info, num_points=16
    )
    
    # Extract per-image results
    iou_matrices = []
    for i, (_, _, out_off, n_det, n_gt) in enumerate(pair_info):
        if n_det > 0 and n_gt > 0:
            iou = results[out_off:out_off + n_det * n_gt].reshape(n_det, n_gt)
        else:
            iou = np.zeros((n_det, n_gt), dtype=np.float32)
        iou_matrices.append(iou)
    
    return iou_matrices
```

### Environment Variables

- `CUDA_ARCH`: GPU architecture for compilation (e.g., `sm_86`)
- `EGBB_GPU_NUM_POINTS`: Override polygon points at runtime (default: 16)

### Polygon Points Tradeoff

| Points | Speed | Accuracy |
|--------|-------|----------|
| 8 | Fastest | ~1% error |
| 16 | Fast | ~0.02% error (recommended) |
| 32 | Moderate | ~0.005% error |
| 64 | Slow | Maximum precision |

## File Structure

```
gpu_convex_IoU/
├── README.md              # This file
├── setup.py               # Build configuration
├── device_iou.cuh         # CUDA device functions (polygon ops)
├── convexiou_cuda.cu      # CUDA kernels
├── pybind_wrapper.cpp     # Python bindings
└── examples/
    └── example_usage.py   # Usage examples
```

## Troubleshooting

### CUDA not found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Wrong GPU architecture
```bash
# Check your GPU
nvidia-smi --query-gpu=compute_cap --format=csv

# Set architecture (e.g., for compute capability 8.6)
export CUDA_ARCH=sm_86
pip install .
```

### Out of memory
The batched function allocates memory for all IoU computations at once. For very large datasets, process in chunks:
```python
CHUNK_SIZE = 1000  # images per chunk
for chunk_start in range(0, num_images, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, num_images)
    # Process chunk...
```

## Citation

If you use this library in your research, please cite:
```bibtex
@software{gpu_convex_iou,
  title = {GPU Convex IoU: Fast Ellipse IoU via Polygon Approximation},
  author = {Gabriel},
  year = {2026},
  url = {https://github.com/your-repo/gpu_convex_iou}
}
```
