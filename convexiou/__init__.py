"""
convexiou - GPU-accelerated IoU for oriented bounding boxes via ellipse/polygon approximation.

Usage:
    from convexiou import rectangular_iou, batched_iou, matrix_iou
    from convexiou import batched_iou_from_lists
"""

from convexiou._core import (
    calculate_iou_matrix,
    calculate_iou_matrix_numpy,
    calculate_iou_matrix_numpy_from_numpy,
    calculate_iou_rectangular,
    calculate_iou_rectangular_numpy,
    calculate_iou_rectangular_numpy_from_numpy,
    calculate_iou_batched_rectangular,
    calculate_iou_matrix_from_file,
)

import numpy as np

matrix_iou = calculate_iou_matrix_numpy_from_numpy
rectangular_iou = calculate_iou_rectangular_numpy_from_numpy
batched_iou = calculate_iou_batched_rectangular


def batched_iou_from_lists(dets_per_image, gts_per_image, num_points=16):
    """
    High-level batched IoU: takes lists of per-image arrays, returns list of IoU matrices.

    Args:
        dets_per_image: list of (N_i, 5) float64 arrays — detections per image
        gts_per_image:  list of (M_i, 5) float64 arrays — ground truths per image
        num_points:     polygon approximation points (default 16)

    Returns:
        list of (N_i, M_i) float32 IoU matrices, one per image
    """
    num_images = len(dets_per_image)
    if num_images != len(gts_per_image):
        raise ValueError(
            f"dets_per_image has {num_images} entries but gts_per_image has {len(gts_per_image)}"
        )

    if num_images == 0:
        return []

    non_empty = []
    for i in range(num_images):
        d = np.asarray(dets_per_image[i], dtype=np.float64)
        g = np.asarray(gts_per_image[i], dtype=np.float64)
        if d.ndim == 1:
            d = d.reshape(0, 5) if d.size == 0 else d.reshape(1, 5)
        if g.ndim == 1:
            g = g.reshape(0, 5) if g.size == 0 else g.reshape(1, 5)
        non_empty.append((i, d, g))

    has_dets = [t for t in non_empty if t[1].shape[0] > 0 and t[2].shape[0] > 0]

    result_map = {}
    if has_dets:
        all_dets = np.vstack([t[1] for t in has_dets])
        all_gts = np.vstack([t[2] for t in has_dets])

        pair_info = []
        det_off = gt_off = out_off = 0
        for _, d, g in has_dets:
            nd, ng = d.shape[0], g.shape[0]
            pair_info.append([det_off, gt_off, out_off, nd, ng])
            det_off += nd
            gt_off += ng
            out_off += nd * ng
        pair_info = np.array(pair_info, dtype=np.int32)

        flat, _ = calculate_iou_batched_rectangular(
            all_dets, all_gts, pair_info, num_points
        )

        for idx, (img_i, d, g) in enumerate(has_dets):
            _, _, o, nd, ng = pair_info[idx]
            result_map[img_i] = flat[o : o + nd * ng].reshape(nd, ng)

    out = []
    for i in range(num_images):
        if i in result_map:
            out.append(result_map[i])
        else:
            nd = non_empty[i][1].shape[0]
            ng = non_empty[i][2].shape[0]
            out.append(np.zeros((nd, ng), dtype=np.float32))
    return out


__version__ = "2.0.0"
__all__ = [
    "matrix_iou",
    "rectangular_iou",
    "batched_iou",
    "batched_iou_from_lists",
    "calculate_iou_matrix",
    "calculate_iou_matrix_numpy",
    "calculate_iou_matrix_numpy_from_numpy",
    "calculate_iou_rectangular",
    "calculate_iou_rectangular_numpy",
    "calculate_iou_rectangular_numpy_from_numpy",
    "calculate_iou_batched_rectangular",
    "calculate_iou_matrix_from_file",
]
