# GPU Convex IoU

GPU-accelerated Intersection over Union (IoU) computation for ellipses using convex polygon approximation. This library provides massive speedups (100x+) over CPU-based methods like Shapely for rotated bounding box evaluation.

## Performance

Benchmarked with 5,000 images, 50 detections/image, 10 ground truths/image (2.5 million IoU pairs) on an RTX 2060:

| Method | Time | Throughput | Speedup |
|--------|------|------------|--------|
| Shapely (CPU) | ~977 sec (estimated) | 2.6 K pairs/sec | 1x |
| GPU Per-image | 11.2 sec | 0.22 M pairs/sec | ~87x |
| **GPU Batched** | **458 ms** | **5.5 M pairs/sec** | **~2,133x** |

**Accuracy**: Maximum difference vs Shapely is 0.028 — mean difference 0.00015 — negligible for all practical purposes.

**Batched vs Per-image**: The batched kernel is 24x faster than calling the GPU per-image, because it eliminates thousands of kernel launch overheads.

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
- Python 3.8+
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

#### 1. Single Image: `rectangular_iou`

Compute IoU matrix between detections and ground truths for **one image**:

```python
import numpy as np
from convexiou import rectangular_iou

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
iou_matrix = rectangular_iou(detections, ground_truths, num_points=16)

print(iou_matrix.shape)  # (2, 2)
print(iou_matrix)        # [[0.85, 0.0], [0.0, 0.0]]
```

#### 2. Batched (OPTIMAL): `batched_iou_from_lists`

Compute IoU for **ALL images** in a single GPU call. This is the recommended
function for detector evaluation — ~24x faster than per-image calls and ~2,000x
faster than Shapely:

```python
from convexiou import batched_iou_from_lists

# dets_per_image: list of (N_i, 5) float64 arrays, one per image
# gts_per_image:  list of (M_i, 5) float64 arrays, one per image
iou_matrices = batched_iou_from_lists(dets_per_image, gts_per_image, num_points=16)

# iou_matrices[i] is the (N_i, M_i) IoU matrix for image i
```

This function handles all the batching internally — no need to manually build
offset arrays. It concatenates boxes, builds pair metadata, launches a single
GPU kernel, and splits the results back into per-image matrices.

Empty detection or ground truth arrays are handled automatically (returns
zero-filled matrices for those images).

#### 3. NxN Matrix (for NMS): `matrix_iou`

Compute pairwise IoU for all boxes (useful for NMS):

```python
from convexiou import matrix_iou

boxes = np.array([...], dtype=np.float64)  # (N, 5)
iou_matrix = matrix_iou(boxes, num_points=16)
# Returns (N, N) matrix
```

## Integrating with a Detector

### Required Changes

To integrate GPU IoU with your rotated object detector's evaluation code:

#### 1. Import the Library

```python
from convexiou import rectangular_iou, batched_iou_from_lists
```

#### 2. Replace Your IoU Function

The library expects boxes as `[x, y, w, h, angle_rad]` — the same format used by
most rotated object detectors. No conversion needed.

**Before (CPU with Shapely):**
```python
def compute_iou_matrix(det_boxes, gt_boxes):
    ious = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = shapely_iou(det, gt)
    return ious
```

**After (single image):**
```python
from convexiou import rectangular_iou

def compute_iou_matrix(det_boxes, gt_boxes):
    return rectangular_iou(
        det_boxes.astype(np.float64),
        gt_boxes.astype(np.float64),
    )
```

**After (all images at once — recommended):**
```python
from convexiou import batched_iou_from_lists

def evaluate_all_images(all_detections, all_ground_truths):
    return batched_iou_from_lists(all_detections, all_ground_truths)
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
GPU_Convex_IoU/
├── README.md              # This file
├── pyproject.toml         # Package metadata (PEP 621)
├── setup.py               # Build configuration (CUDA compilation)
├── convexiou/             # Python package
│   ├── __init__.py        # Public API (rectangular_iou, batched_iou_from_lists, ...)
│   └── gaucho.py          # GauCho detector integration
├── convexiou_cuda.cu      # CUDA kernels
├── device_iou.cuh         # CUDA device functions (polygon ops)
├── pybind_wrapper.cpp     # Python/C++ bindings (pybind11)
├── examples/
│   └── example_usage.py   # Usage examples
└── tests/
    ├── test_comparison.py          # GPU vs Shapely benchmark
    └── test_gaucho_integration.py  # GauCho mAP validation
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
