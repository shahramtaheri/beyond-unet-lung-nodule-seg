"""
Evaluation metrics from Section 4.4 of the paper.

Implements all seven metrics reported in the paper:
  - Dice Similarity Coefficient (DSC)                          (Eq. 9)
  - Intersection over Union (IoU)                               (Eq. 10)
  - 95th-percentile Hausdorff Distance (HD95)                   (Eq. 11)
  - Average Symmetric Surface Distance (ASSD)                   (Eq. 12)
  - Sensitivity (recall)                                        (Eq. 13)
  - Positive Predictive Value (PPV / precision)                 (Eq. 14)
  - Hausdorff Distance (HD)                                     (Eq. 15)

All boundary metrics (HD95, ASSD, HD) operate on binary 2D masks and use
Euclidean distance transforms; they expect numpy arrays (H, W) of {0,1}.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion


def _to_numpy_bool(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return mask.astype(bool)


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of boundary pixels (mask minus its erosion)."""
    if mask.sum() == 0:
        return mask  # empty surface
    eroded = binary_erosion(mask)
    return mask & ~eroded


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """DSC = 2|P n G| / (|P| + |G|) = 2TP / (2TP + FP + FN)        (Eq. 9)"""
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect agreement by convention
    return float(2.0 * intersection / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """IoU = |P n G| / |P u G| = TP / (TP + FP + FN)               (Eq. 10)"""
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))


def sensitivity_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Sensitivity (recall) = TP / (TP + FN)                       (Eq. 13)"""
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + fn == 0:
        return 1.0
    return float(tp / (tp + fn + eps))


def ppv_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """PPV (precision) = TP / (TP + FP)                            (Eq. 14)"""
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    if tp + fp == 0:
        return 1.0
    return float(tp / (tp + fp + eps))


def _directed_distances(surface_a: np.ndarray, surface_b: np.ndarray) -> Optional[np.ndarray]:
    """
    Distance from every surface point in `surface_a` to the nearest
    surface point in `surface_b`, using a Euclidean distance transform
    computed on the complement of surface_b.
    """
    if surface_a.sum() == 0 or surface_b.sum() == 0:
        return None
    dt = distance_transform_edt(~surface_b)
    return dt[surface_a]


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    HD95(P, G) = max( percentile_95{ d(p, G) : p in P },
                       percentile_95{ d(g, P) : g in G } )         (Eq. 11)
    """
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    surf_p, surf_g = _surface_voxels(pred), _surface_voxels(gt)

    d_p_to_g = _directed_distances(surf_p, surf_g)
    d_g_to_p = _directed_distances(surf_g, surf_p)

    if d_p_to_g is None or d_g_to_p is None:
        return float("nan")

    hd95_p = np.percentile(d_p_to_g, 95)
    hd95_g = np.percentile(d_g_to_p, 95)
    return float(max(hd95_p, hd95_g))


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    HD(P, G) = max( sup_p inf_g ||p-g||, sup_g inf_p ||g-p|| )      (Eq. 15)
    """
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    surf_p, surf_g = _surface_voxels(pred), _surface_voxels(gt)

    d_p_to_g = _directed_distances(surf_p, surf_g)
    d_g_to_p = _directed_distances(surf_g, surf_p)

    if d_p_to_g is None or d_g_to_p is None:
        return float("nan")

    return float(max(d_p_to_g.max(), d_g_to_p.max()))


def assd_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    ASSD(P, G) = ( sum_{p in S(P)} d(p, S(G)) + sum_{g in S(G)} d(g, S(P)) )
                 / ( |S(P)| + |S(G)| )                              (Eq. 12)
    """
    pred, gt = _to_numpy_bool(pred), _to_numpy_bool(gt)
    surf_p, surf_g = _surface_voxels(pred), _surface_voxels(gt)

    d_p_to_g = _directed_distances(surf_p, surf_g)
    d_g_to_p = _directed_distances(surf_g, surf_p)

    if d_p_to_g is None or d_g_to_p is None:
        return float("nan")

    total = d_p_to_g.sum() + d_g_to_p.sum()
    count = surf_p.sum() + surf_g.sum()
    return float(total / count)


@dataclass
class SegmentationMetrics:
    dice: float
    iou: float
    sensitivity: float
    ppv: float
    hd95: float
    assd: float
    hd: float

    def as_dict(self):
        return asdict(self)


def compute_all_metrics(pred_mask, gt_mask, spacing: float = 1.0) -> SegmentationMetrics:
    """
    Compute all 7 metrics reported in the paper for a single 2D slice
    pair (pred_mask, gt_mask), both binary {0,1} arrays/tensors of shape
    (H, W).

    `spacing` is the physical size of one pixel in mm (paper resamples to
    1 mm isotropic resolution, so the default of 1.0 directly yields mm
    units for HD95 / ASSD / HD as reported in the paper).
    """
    pred = _to_numpy_bool(pred_mask)
    gt = _to_numpy_bool(gt_mask)

    dice = dice_coefficient(pred, gt)
    iou = iou_score(pred, gt)
    sens = sensitivity_score(pred, gt)
    ppv = ppv_score(pred, gt)
    hd95 = hausdorff_distance_95(pred, gt) * spacing
    assd = assd_score(pred, gt) * spacing
    hd = hausdorff_distance(pred, gt) * spacing

    return SegmentationMetrics(dice=dice, iou=iou, sensitivity=sens, ppv=ppv, hd95=hd95, assd=assd, hd=hd)


def aggregate_metrics(metrics_list) -> dict:
    """Average a list of SegmentationMetrics (or dicts) over a dataset/fold,
    ignoring NaNs (e.g. from empty-prediction or empty-GT slices)."""
    keys = ["dice", "iou", "sensitivity", "ppv", "hd95", "assd", "hd"]
    arrs = {k: [] for k in keys}
    for m in metrics_list:
        d = m.as_dict() if isinstance(m, SegmentationMetrics) else m
        for k in keys:
            arrs[k].append(d[k])
    return {k: float(np.nanmean(v)) for k, v in arrs.items()}


if __name__ == "__main__":
    np.random.seed(0)
    gt = np.zeros((64, 64), dtype=bool)
    gt[20:40, 20:40] = True
    pred = np.zeros((64, 64), dtype=bool)
    pred[22:42, 18:38] = True  # shifted prediction

    metrics = compute_all_metrics(pred, gt)
    print(metrics.as_dict())
