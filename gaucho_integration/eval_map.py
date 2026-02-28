# Copyright (c) OpenMMLab. All rights reserved.



from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable

from math import sqrt, pi

try:
    from shapely.geometry.point import Point
    from shapely import affinity
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from convexiou import batched_iou_from_lists
    _HAS_CONVEXIOU = True
except (ImportError, ModuleNotFoundError):
    # If convexiou is not pip-installed, try to find it via CONVEXIOU_PATH
    # e.g. export CONVEXIOU_PATH=/home/user/GPU_Convex_IoU
    import sys as _sys
    import os as _os
    _convexiou_path = _os.environ.get('CONVEXIOU_PATH')
    if _convexiou_path and _os.path.isdir(_convexiou_path):
        _sys.path.insert(0, _convexiou_path)
        try:
            from convexiou import batched_iou_from_lists
            _HAS_CONVEXIOU = True
        except (ImportError, ModuleNotFoundError):
            _HAS_CONVEXIOU = False
    else:
        _HAS_CONVEXIOU = False


def convert_obb_to_gbb_egbb(obb):
    x, y, w, h, rad_angle = obb
    #print(angle)
    aa2 = w ** 2/12
    bb2 = h ** 2/12
    # Gets diagonal covariance
    Sig1 = np.array( [ [aa2, 0], [0, bb2] ])
    # Gets orientation in radians
    angle = rad_angle*180/pi
    # Rotation matrix
    R1 = np.array([ [np.cos(rad_angle), -np.sin(rad_angle)], [np.sin(rad_angle), np.cos(rad_angle)]  ])
    # Full covariance matrix
    Sig = R1 @ Sig1 @ R1.T
    # Final GBB encoded as [xcenter, ycenter, cov_a, cov_b, cov_c]
    gbb = np.array([x, y, Sig[0,0], Sig[1,1], Sig[0,1] ])
    #
    # Gets EGBB coordinates
    #
    # Semi-axes
    aa = w/sqrt(pi)
    bb = h/sqrt(pi)
    # Gets EGBB coordinates [xcenter, ycenter, semi-axis1, semi-axes2, angle]
    egbb = np.array([x, y, aa, bb, angle ])
    return gbb, egbb


def probiou_mapping(x):
    #
    #  Applies non-linear mapping to better relate ProbIoU with the corresponing EGBB-based IoU
    #
    a = 0.259743420366354
    return a*x + (1 - a)*x**2


def probiou(gbb1, gbb2):
    #
    #  Computes ProbIoU and GBB-based BC
    #
    # centers are the first two coordinates

    mu1 = gbb1[:2]
    mu2 = gbb2[:2]
    # covariance are the next three
    C1 = np.array( [  [gbb1[2], gbb1[4]], [gbb1[4], gbb1[3]]  ]  )
    C2 = np.array( [  [gbb2[2], gbb2[4]], [gbb2[4], gbb2[3]]  ]  )
    dmu = mu1 - mu2
    C = 0.5*(C1 + C2)
    iC = np.linalg.inv(C)
    BB1 = np.dot(dmu, np.dot(iC, dmu).T)/8
    ratio = max(1e-16, abs(np.linalg.det(C)) / (1e-16 + np.sqrt(abs(np.linalg.det(C1)*np.linalg.det(C2)))))
    BB2 = 0.5*np.log( ratio )
    DBB = BB1 + BB2
    # Bhatacharyya coefficient
    BC = np.exp(-DBB)
    Hel = np.sqrt(1 - BC)
    #probiou = 1 - Hel
    adjusted_probiou = probiou_mapping(1 - Hel)
    return adjusted_probiou


def create_ellipse(center, lengths, angle=0):
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, lengths[0], lengths[1])
    ellr = affinity.rotate(ell, angle)
    return ellr

def iou_ellipse(egbb1, egbb2):
    #
    #  Computes the IoU between two EGBBs
    #
    # Creates ellipses (multiplies the semi-axes by a large value to reduce approximation errors)
    factor = 1
    el1 = create_ellipse(egbb1[0:2], factor*egbb1[2:4], egbb1[4])
    el2 = create_ellipse(egbb2[0:2], factor*egbb2[2:4], egbb2[4])
    #Computes IoU
    inter = el1.buffer(0).intersection(el2).buffer(0).area
    a1 = el1.area
    a2 = el2.area
    union = a1 + a2 - inter
    iou = inter / (max(union, 1e-16)) # avoids any possible rectangle with no area
    return iou


def probiou_calculate(pred, GT, mode):
    #print(pred)
    #print(GT)
    GT_gbb, GT_egbb = convert_obb_to_gbb_egbb(GT)
    pred_gbb, pred_egbb = convert_obb_to_gbb_egbb(pred)

    if mode == 'egbb':
      return iou_ellipse(GT_egbb, pred_egbb)
    if mode == 'gbb':
      return probiou(GT_gbb, pred_gbb)


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None,
                 opt='iou'):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp


    det_bboxes = det_bboxes[:,0:5]

    if opt == 'iou':
        ious = box_iou_rotated( torch.from_numpy(det_bboxes).float(),torch.from_numpy(gt_bboxes).float()).numpy()
    elif opt == 'gbb':
        n, m = det_bboxes.shape[0], gt_bboxes.shape[0]
        ious = np.zeros((n,m))
        for i_x in range(n):
            for j_y in range(m):
                ious[i_x, j_y] = probiou_calculate(det_bboxes[i_x, :], gt_bboxes[j_y, :], opt)
    elif opt == 'egbb':
        n, m = det_bboxes.shape[0], gt_bboxes.shape[0]
        ious = np.zeros((n, m))
        for i_x in range(n):
            for j_y in range(m):
                ious[i_x, j_y] = probiou_calculate(det_bboxes[i_x, :], gt_bboxes[j_y, :], opt)

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


# convexiou: tp/fp from pre-computed IoU matrix
def _tpfp_from_precomputed_iou(det_bboxes, gt_bboxes, ious,
                               gt_bboxes_ignore_count,
                               iou_thr=0.5, area_ranges=None):
    """Compute tp/fp from a pre-computed IoU matrix.

    Same logic as tpfp_default, but receives the IoU matrix directly
    instead of computing it internally. Used by the batched GPU path.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes (real + ignored), of shape (n, 5).
        ious (ndarray): Pre-computed IoU matrix of shape (m, n).
        gt_bboxes_ignore_count (int): Number of ignored gt bboxes
            (last gt_bboxes_ignore_count rows of gt_bboxes are ignored).
        iou_thr (float): IoU threshold. Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) of shape (num_scales, m).
    """
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

    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
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

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


# convexiou: batched GPU evaluation for one class
def _eval_cls_egbb_gpu(cls_dets, cls_gts, cls_gts_ignore,
                       num_imgs, num_scales, iou_thr, area_ranges,
                       num_points=16):
    """Evaluate one class using batched GPU ellipse IoU.

    Collects all per-image detections and ground truths, computes IoU
    for ALL images in a single GPU kernel launch via batched_iou_from_lists,
    then computes tp/fp from the pre-computed IoU matrices.

    Args:
        cls_dets (list[ndarray]): Per-image detections for this class.
        cls_gts (list[ndarray]): Per-image ground truths for this class.
        cls_gts_ignore (list[ndarray]): Per-image ignored gt bboxes.
        num_imgs (int): Number of images.
        num_scales (int): Number of evaluation scales.
        iou_thr (float): IoU threshold.
        area_ranges (list[tuple] | None): Area ranges for evaluation.
        num_points (int): Polygon approximation points for convexiou.

    Returns:
        tuple: (tp_list, fp_list) — lists of per-image tp/fp arrays.
    """
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

    tp_list = []
    fp_list = []
    for img_i in range(num_imgs):
        d, g_all, n_ignore, nd, ng = img_meta[img_i]
        iou_mat = iou_matrices[img_i]
        tp, fp = _tpfp_from_precomputed_iou(
            d, g_all, iou_mat, n_ignore,
            iou_thr=iou_thr, area_ranges=area_ranges)
        tp_list.append(tp)
        fp_list.append(fp)

    return tp_list, fp_list


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4,
                   opt='iou'):

    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    use_gpu_batched = (opt == 'egbb' and _HAS_CONVEXIOU)

    if not use_gpu_batched:
        pool = get_context('spawn').Pool(nproc)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        if use_gpu_batched:
            tp, fp = _eval_cls_egbb_gpu(
                cls_dets, cls_gts, cls_gts_ignore,
                num_imgs, num_scales, iou_thr, area_ranges)
        else:
            if opt == 'egbb' and not _HAS_SHAPELY:
                raise ImportError(
                    "opt='egbb' requires either convexiou (GPU) or shapely (CPU). "
                    "Install one of them: pip install convexiou  or  pip install shapely"
                )

            tpfp = pool.starmap(
                tpfp_default,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)], [opt]*len(range(num_imgs)) ))

            tp, fp = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    if not use_gpu_batched:
        pool.close()

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
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

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}',
                f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.4f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
