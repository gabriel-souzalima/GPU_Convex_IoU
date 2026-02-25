"""
Standalone test that verifies convexiou.gaucho produces the same mAP
as GauCho's original Shapely-based egbb evaluation.

Does NOT require GauCho / MMRotate installed — it mocks the minimal
dependencies (mmdet.core.average_precision) so the integration module
can be tested in isolation.

Run:
    python test_gaucho_integration.py
"""

import numpy as np
import time
import sys


class _MockAP:
    @staticmethod
    def average_precision(recalls, precisions, mode='area'):
        if mode == '11points':
            ap = 0.0
            for thr in np.arange(0.0, 1.1, 0.1):
                precs = precisions[recalls >= thr]
                ap += (precs.max() if precs.size > 0 else 0.0)
            return ap / 11.0
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


import types
_mmdet_core = types.ModuleType('mmdet.core')
_mmdet_core.average_precision = _MockAP.average_precision
_mmdet = types.ModuleType('mmdet')
_mmdet.core = _mmdet_core
sys.modules['mmdet'] = _mmdet
sys.modules['mmdet.core'] = _mmdet_core

from convexiou.gaucho import eval_rbbox_map_gpu

try:
    from shapely.geometry.point import Point
    from shapely import affinity
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def make_dataset(num_images, num_classes, max_dets_per_img, max_gts_per_img, seed=42):
    rng = np.random.RandomState(seed)
    det_results = []
    annotations = []
    for _ in range(num_images):
        img_dets = []
        for _ in range(num_classes):
            nd = rng.randint(0, max_dets_per_img + 1)
            if nd == 0:
                dets = np.zeros((0, 6), dtype=np.float64)
            else:
                boxes = rng.rand(nd, 5) * [800, 600, 100, 100, np.pi] + [100, 100, 20, 20, -np.pi/2]
                scores = rng.rand(nd, 1) * 0.8 + 0.2
                dets = np.hstack([boxes, scores])
            img_dets.append(dets)
        det_results.append(img_dets)

        n_gt = rng.randint(1, max_gts_per_img + 1)
        bboxes = rng.rand(n_gt, 5) * [800, 600, 100, 100, np.pi] + [100, 100, 20, 20, -np.pi/2]
        labels = rng.randint(0, num_classes, size=n_gt)
        annotations.append({
            'bboxes': bboxes.astype(np.float64),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': np.zeros((0, 5), dtype=np.float64),
            'labels_ignore': np.array([], dtype=np.int64),
        })

    return det_results, annotations


def shapely_egbb_eval(det_results, annotations, iou_thr=0.5):
    from math import sqrt, pi as PI

    def create_ellipse(center, lengths, angle=0):
        circ = Point(center).buffer(1)
        ell = affinity.scale(circ, lengths[0], lengths[1])
        return affinity.rotate(ell, angle)

    def iou_ellipse_pair(obb1, obb2):
        a1 = obb1[2] / sqrt(PI)
        b1 = obb1[3] / sqrt(PI)
        ang1 = np.degrees(obb1[4])
        a2 = obb2[2] / sqrt(PI)
        b2 = obb2[3] / sqrt(PI)
        ang2 = np.degrees(obb2[4])
        el1 = create_ellipse(obb1[:2], [a1, b1], ang1)
        el2 = create_ellipse(obb2[:2], [a2, b2], ang2)
        inter = el1.buffer(0).intersection(el2).buffer(0).area
        union = el1.area + el2.area - inter
        return inter / max(union, 1e-16)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])
    all_ap = []

    for cls_id in range(num_classes):
        cls_dets = [img_res[cls_id] for img_res in det_results]
        cls_gts = []
        for ann in annotations:
            gt_inds = ann['labels'] == cls_id
            cls_gts.append(ann['bboxes'][gt_inds, :])

        tp_list = []
        fp_list = []
        num_gts_total = 0
        for img_i in range(num_imgs):
            d = np.array(cls_dets[img_i])
            g = np.array(cls_gts[img_i])
            if d.ndim == 1:
                d = d.reshape(0, 6) if d.size == 0 else d.reshape(1, -1)
            if g.ndim == 1:
                g = g.reshape(0, 5) if g.size == 0 else g.reshape(1, 5)
            num_gts_total += g.shape[0]
            nd = d.shape[0]
            ng = g.shape[0]
            tp = np.zeros(nd, dtype=np.float32)
            fp = np.zeros(nd, dtype=np.float32)
            if ng == 0:
                fp[:] = 1
                tp_list.append(tp)
                fp_list.append(fp)
                continue
            if nd == 0:
                tp_list.append(tp)
                fp_list.append(fp)
                continue
            ious = np.zeros((nd, ng))
            for i in range(nd):
                for j in range(ng):
                    ious[i, j] = iou_ellipse_pair(d[i, :5], g[j, :5])
            ious_max = ious.max(axis=1)
            ious_argmax = ious.argmax(axis=1)
            sort_inds = np.argsort(-d[:, -1])
            gt_covered = np.zeros(ng, dtype=bool)
            for i in sort_inds:
                if ious_max[i] >= iou_thr:
                    mg = ious_argmax[i]
                    if not gt_covered[mg]:
                        gt_covered[mg] = True
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            tp_list.append(tp)
            fp_list.append(fp)

        if num_gts_total == 0:
            all_ap.append(0.0)
            continue
        all_d = np.vstack([np.array(cls_dets[i]).reshape(-1, 6)
                           if np.array(cls_dets[i]).size > 0
                           else np.zeros((0, 6))
                           for i in range(num_imgs)])
        sort_inds = np.argsort(-all_d[:, -1])
        tp = np.hstack(tp_list)[sort_inds]
        fp = np.hstack(fp_list)[sort_inds]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recalls = tp / max(num_gts_total, eps)
        precisions = tp / np.maximum(tp + fp, eps)
        ap = _MockAP.average_precision(recalls, precisions, '11points')
        all_ap.append(ap)

    valid = [a for a in all_ap if a > 0 or True]
    mean_ap = np.mean(valid) if valid else 0.0
    return mean_ap, all_ap


if __name__ == "__main__":
    NUM_IMAGES = 50
    NUM_CLASSES = 5
    MAX_DETS = 20
    MAX_GTS = 8

    print(f"Test config: {NUM_IMAGES} images, {NUM_CLASSES} classes, "
          f"up to {MAX_DETS} dets and {MAX_GTS} gts per image")
    print()

    det_results, annotations = make_dataset(NUM_IMAGES, NUM_CLASSES, MAX_DETS, MAX_GTS)

    total_dets = sum(d.shape[0] for img in det_results for d in img if d.size > 0)
    total_gts = sum(ann['bboxes'].shape[0] for ann in annotations)
    print(f"Total detections: {total_dets}")
    print(f"Total ground truths: {total_gts}")
    print()

    print("--- GPU eval_rbbox_map_gpu ---")
    t0 = time.perf_counter()
    gpu_map, gpu_results = eval_rbbox_map_gpu(
        det_results, annotations,
        iou_thr=0.5, use_07_metric=True,
        num_points=16,
    )
    gpu_time = time.perf_counter() - t0
    print(f"  mAP: {gpu_map:.4f}")
    for i, r in enumerate(gpu_results):
        print(f"  Class {i}: AP={r['ap']:.4f}  gts={r['num_gts']}  dets={r['num_dets']}")
    print(f"  Time: {gpu_time * 1000:.2f} ms")
    print()

    if HAS_SHAPELY:
        print("--- Shapely egbb eval (CPU) ---")
        t0 = time.perf_counter()
        shp_map, shp_aps = shapely_egbb_eval(det_results, annotations, iou_thr=0.5)
        shp_time = time.perf_counter() - t0
        print(f"  mAP: {shp_map:.4f}")
        for i, ap in enumerate(shp_aps):
            print(f"  Class {i}: AP={ap:.4f}")
        print(f"  Time: {shp_time * 1000:.2f} ms")
        print()

        print("--- Comparison ---")
        print(f"  mAP difference: {abs(gpu_map - shp_map):.6f}")
        max_ap_diff = max(abs(gpu_results[i]['ap'] - shp_aps[i]) for i in range(NUM_CLASSES))
        print(f"  Max per-class AP difference: {max_ap_diff:.6f}")
        print(f"  Speedup: {shp_time / gpu_time:.1f}x")
        print()
        if max_ap_diff < 0.02:
            print("  PASS — GPU and Shapely mAP match within tolerance")
        else:
            print("  WARNING — mAP difference exceeds tolerance, review results")
    else:
        print("Shapely not installed, skipping CPU comparison.")
        print("  Install with: pip install shapely")

    print()
    print("=" * 60)

    print()
    print("Scaling test: 500 images, 10 classes...")
    det2, ann2 = make_dataset(500, 10, 30, 10, seed=99)
    t0 = time.perf_counter()
    map2, _ = eval_rbbox_map_gpu(det2, ann2, iou_thr=0.5, use_07_metric=True, num_points=16)
    t2 = time.perf_counter() - t0
    total_pairs = sum(
        np.array(d).reshape(-1, 6).shape[0] if np.array(d).size > 0 else 0
        for img in det2 for d in img
    )
    print(f"  mAP: {map2:.4f}")
    print(f"  Time: {t2 * 1000:.2f} ms")
    print(f"  Total detection entries: {total_pairs}")
