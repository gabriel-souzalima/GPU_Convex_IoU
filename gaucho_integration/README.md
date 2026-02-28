# GauCho Integration — GPU Batched EGBB Evaluation

Drop-in replacement for GauCho's `eval_map.py` that uses **convexiou**
`batched_iou_from_lists` for `opt='egbb'` — all images computed in a
**single GPU kernel launch per class**.

## What changes

The `eval_rbbox_map` function detects when `opt='egbb'` and convexiou is
installed. Instead of the original flow:

```
Original (Shapely):
  for each class:
    pool.starmap(tpfp_default, ...)     ← multiprocessing, N images
      └─ for i in range(n):             ← double for-loop per image
           for j in range(m):
             iou = Shapely(det[i], gt[j])  ← Python, pair by pair
```

It takes a different path:

```
GPU batched (convexiou):
  for each class:
    Step 1: collect all dets/gts from all images into lists
    Step 2: batched_iou_from_lists(all_dets, all_gts)   ← 1 kernel launch
    Step 3: compute tp/fp per image from pre-computed IoU matrices
```

No multiprocessing pool is created for `opt='egbb'` — the GPU does all the
heavy lifting in one shot.

### Functions added

- `_tpfp_from_precomputed_iou`: Same tp/fp logic as `tpfp_default`, but
  receives a pre-computed IoU matrix instead of computing it internally.
- `_eval_cls_egbb_gpu`: Orchestrates the collect → batch → tp/fp pipeline
  for one class.

### Functions unchanged

Everything else is **identical** to the original GauCho `eval_map.py`:
- `convert_obb_to_gbb_egbb`, `probiou_mapping`, `probiou`
- `create_ellipse`, `iou_ellipse`, `probiou_calculate`
- `tpfp_default` (still used for `opt='iou'` and `opt='gbb'`)
- `get_cls_results`
- `print_map_summary`
- `opt='iou'` and `opt='gbb'` paths in `eval_rbbox_map`

## Requirements

1. **convexiou** installed with CUDA support:
   ```bash
   cd GPU_Convex_IoU
   CUDA_ARCH=sm_86 pip install .   # adjust sm_XX for your GPU
   ```

2. **GauCho** installed normally (mmrotate-gaucho)

## Installation

Copy one file:

```bash
cp gaucho_integration/eval_map.py  \
   /path/to/mmrotate-gaucho/mmrotate/core/evaluation/eval_map.py
```

That's it. No other files need to change.

## Fallback

If convexiou is **not** installed, `opt='egbb'` automatically falls back to
the original Shapely + multiprocessing path. If neither is installed, it
raises an `ImportError`.

## Compatibility

- Same function signatures, same return values
- `opt='iou'` and `opt='gbb'` completely untouched
- `opt='egbb'` produces results within ~0.005 of Shapely
- Works with all GauCho datasets (DOTA, HRSC, DIOR, UCAS-AOD)
