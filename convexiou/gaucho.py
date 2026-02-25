"""
GauCho detector integration for convexiou.

Provides a drop-in replacement for GauCho's eval_rbbox_map() that uses
GPU-accelerated ellipse IoU instead of the slow Shapely-based egbb mode.

Usage (inside GauCho's evaluation pipeline):

    from convexiou.gaucho import patch_gaucho
    patch_gaucho()

    # Then run evaluation normally with opt='egbb' — it will use GPU automatically.

Or call the replacement function directly:

    from convexiou.gaucho import eval_rbbox_map_gpu
    mean_ap, results = eval_rbbox_map_gpu(det_results, annotations, ...)
"""

import numpy as np

from convexiou import batched_iou_from_lists


def _tpfp_from_iou(det_bboxes, gt_bboxes, ious, gt_bboxes_ignore_count,
                    iou_thr=0.5, area_ranges=None):
    det_bboxes = np.array(det_bboxes)
    num_gts_real = gt_bboxes.shape[0] - gt_bboxes_ignore_count
    gt_ignore_inds = np.concatenate((
        np.zeros(num_gts_real, dtype=bool),
        np.ones(gt_bboxes_ignore_count, dtype=bool),
    ))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]

    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)

    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    if num_gts == 0 or num_dets == 0:
        if num_dets > 0 and area_ranges == [(None, None)]:
            fp[...] = 1
        elif num_dets > 0:
            raise NotImplementedError
        return tp, fp

    if ious.size == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        return tp, fp

    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)

    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError

        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def eval_rbbox_map_gpu(det_results,
                       annotations,
                       scale_ranges=None,
                       iou_thr=0.5,
                       use_07_metric=True,
                       dataset=None,
                       logger=None,
                       nproc=4,
                       opt='egbb',
                       num_points=16):
    """Drop-in replacement for GauCho's eval_rbbox_map using GPU ellipse IoU.

    Same signature as the original, with an added num_points parameter.
    The opt parameter is accepted for compatibility but GPU ellipse IoU
    is always used (which is equivalent to opt='egbb' but ~1000x faster).

    Args:
        det_results: [[cls1_det, cls2_det, ...], ...] per image.
        annotations: [{'bboxes': (n,5), 'labels': (n,), ...}, ...] per image.
        scale_ranges: Range of scales [(min1, max1), ...]. Default: None.
        iou_thr:      IoU threshold. Default: 0.5.
        use_07_metric: Use VOC07 11-point metric. Default: True.
        dataset:      Dataset name or class list.
        logger:       Logger for print_map_summary.
        nproc:        Ignored (kept for API compat).
        opt:          Ignored (kept for API compat, always uses GPU egbb).
        num_points:   Polygon approximation points for convexiou. Default: 16.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    from mmdet.core import average_precision

    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    eval_results = []

    for cls_id in range(num_classes):
        cls_dets = [img_res[cls_id] for img_res in det_results]

        cls_gts = []
        cls_gts_ignore = []
        for ann in annotations:
            gt_inds = ann['labels'] == cls_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            if ann.get('labels_ignore', None) is not None:
                ignore_inds = ann['labels_ignore'] == cls_id
                cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
            else:
                cls_gts_ignore.append(np.zeros((0, 5), dtype=np.float64))

        dets_for_iou = []
        gts_for_iou = []
        img_meta = []

        for img_i in range(num_imgs):
            d = np.array(cls_dets[img_i])
            g = np.array(cls_gts[img_i])
            gi = np.array(cls_gts_ignore[img_i])

            if d.ndim == 1:
                d = d.reshape(0, 6) if d.size == 0 else d.reshape(1, -1)
            if g.ndim == 1:
                g = g.reshape(0, 5) if g.size == 0 else g.reshape(1, 5)
            if gi.ndim == 1:
                gi = gi.reshape(0, 5) if gi.size == 0 else gi.reshape(1, 5)

            g_all = np.vstack((g, gi)) if gi.shape[0] > 0 else g.copy()
            n_ignore = gi.shape[0]

            nd = d.shape[0]
            ng = g_all.shape[0]

            if nd > 0 and ng > 0:
                det_boxes = np.ascontiguousarray(d[:, :5], dtype=np.float64)
                gt_boxes = np.ascontiguousarray(g_all[:, :5], dtype=np.float64)
                dets_for_iou.append(det_boxes)
                gts_for_iou.append(gt_boxes)
            else:
                dets_for_iou.append(np.zeros((0, 5), dtype=np.float64))
                gts_for_iou.append(np.zeros((0, 5), dtype=np.float64))

            img_meta.append((d, g_all, n_ignore, nd, ng))

        iou_matrices = batched_iou_from_lists(dets_for_iou, gts_for_iou,
                                              num_points=num_points)

        tpfp_list = []
        for img_i in range(num_imgs):
            d, g_all, n_ignore, nd, ng = img_meta[img_i]
            iou_mat = iou_matrices[img_i]
            tp_fp = _tpfp_from_iou(d, g_all, iou_mat, n_ignore,
                                   iou_thr=iou_thr, area_ranges=area_ranges)
            tpfp_list.append(tp_fp)

        tp = [t[0] for t in tpfp_list]
        fp = [t[1] for t in tpfp_list]

        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            bbox = np.array(bbox)
            if bbox.ndim == 1:
                bbox = bbox.reshape(0, 5) if bbox.size == 0 else bbox.reshape(1, 5)
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))

        all_dets = np.vstack([np.array(cls_dets[i]).reshape(-1, 6)
                              if np.array(cls_dets[i]).size > 0
                              else np.zeros((0, 6))
                              for i in range(num_imgs)])
        num_dets_total = all_dets.shape[0]

        if num_dets_total > 0:
            sort_inds = np.argsort(-all_dets[:, -1])
            tp = np.hstack(tp)[:, sort_inds]
            fp = np.hstack(fp)[:, sort_inds]
        else:
            tp = np.zeros((num_scales, 0), dtype=np.float32)
            fp = np.zeros((num_scales, 0), dtype=np.float32)

        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()

        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets_total,
            'recall': recalls,
            'precision': precisions,
            'ap': ap,
        })

    if scale_ranges is not None:
        all_ap = np.vstack([r['ap'] for r in eval_results])
        all_num_gts = np.vstack([r['num_gts'] for r in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    try:
        from mmrotate.core.evaluation.eval_map import print_map_summary
        print_map_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)
    except ImportError:
        pass

    return mean_ap, eval_results


def patch_gaucho():
    """Monkey-patch GauCho's eval_rbbox_map to use GPU ellipse IoU for opt='egbb'.

    Call this once before running evaluation:

        from convexiou.gaucho import patch_gaucho
        patch_gaucho()

    After patching:
      - opt='egbb' uses GPU ellipse IoU (replaces slow Shapely loop)
      - opt='iou' and opt='gbb' are unchanged (fall through to original)
    """
    import mmrotate.core.evaluation.eval_map as em
    original = em.eval_rbbox_map

    def _patched_eval_rbbox_map(*args, **kwargs):
        opt = kwargs.get('opt', 'iou')
        if len(args) > 8:
            opt = args[8]
        if opt == 'egbb':
            return eval_rbbox_map_gpu(*args, **kwargs)
        return original(*args, **kwargs)

    em._original_eval_rbbox_map = original
    em.eval_rbbox_map = _patched_eval_rbbox_map
    print("[convexiou] Patched GauCho: opt='egbb' now uses GPU ellipse IoU")
