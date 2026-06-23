"""
Dataset and preprocessing pipeline for LIDC-IDRI, following Section 3 of
the paper ("Materials and Methods").

This module does NOT itself parse raw LIDC-IDRI DICOM/XML annotations
(that requires `pylidc` and the actual dataset download, which is outside
this repo's scope). Instead it implements:

  1. The exact preprocessing steps specified in the paper, given that you
     already have, per CT volume:
         - a 3D numpy volume of HU values
         - a consensus 3D binary nodule mask (built from radiologists who
           agree, see `build_consensus_mask`)
  2. A `LungNoduleSliceDataset` that performs:
         - HU clipping to [-1000, 400] and rescaling to [0, 1]            (Sec 3.4)
         - lung-field-centered 512x512 cropping (not nodule-centered)     (Sec 3.3)
         - extraction of nodule-bearing axial slices only                (Sec 3.3)
         - mask binarization at threshold 0.5                            (Sec 3.4)
  3. Online augmentation via Albumentations, matching the paper's
     described augmentations (random rotation +-15 deg, h/v flips,
     elastic deformation, Gaussian noise).                                (Sec 3.4)
  4. Patient-level 5-fold cross-validation splitting to avoid leakage.    (Sec 3.3)

If you already have LIDC-IDRI loaded via `pylidc`, see
`build_consensus_mask` for how to turn radiologist annotations into the
consensus mask the paper describes (>=3 of 4 radiologists agreeing,
nodules >=3mm only).
"""

from typing import List, Tuple, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_opening, binary_closing, label as cc_label

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


HU_MIN, HU_MAX = -1000.0, 400.0
CROP_SIZE = 512


# ---------------------------------------------------------------------------
# Consensus mask construction (Sec. 3.3: data curation & inclusion criteria)
# ---------------------------------------------------------------------------
def build_consensus_mask(radiologist_masks: Sequence[np.ndarray], min_agreement: int = 3) -> np.ndarray:
    """
    Build a consensus binary mask from multiple radiologists' 3D
    annotations, retaining only voxels marked by at least `min_agreement`
    of the available radiologists (paper: >= 3 of 4).

    Parameters
    ----------
    radiologist_masks : sequence of boolean/binary 3D numpy arrays, all of
        the same shape, one per radiologist who annotated this scan.
    min_agreement : minimum number of radiologists that must agree a
        voxel is nodule for it to be retained.
    """
    if len(radiologist_masks) == 0:
        raise ValueError("Need at least one radiologist mask.")
    stacked = np.stack([m.astype(np.uint8) for m in radiologist_masks], axis=0)
    vote_count = stacked.sum(axis=0)
    consensus = (vote_count >= min_agreement).astype(np.uint8)
    return consensus


def filter_small_nodules(mask_3d: np.ndarray, voxel_spacing_mm: float = 1.0, min_diameter_mm: float = 3.0) -> np.ndarray:
    """
    Remove connected components whose equivalent spherical diameter is
    below `min_diameter_mm` (paper excludes nodules < 3 mm).
    Assumes `mask_3d` has already been resampled to isotropic spacing
    `voxel_spacing_mm` (paper: 1mm isotropic).
    """
    labeled, n_components = cc_label(mask_3d)
    cleaned = np.zeros_like(mask_3d)
    voxel_volume_mm3 = voxel_spacing_mm ** 3

    for comp_id in range(1, n_components + 1):
        component = labeled == comp_id
        n_voxels = component.sum()
        volume_mm3 = n_voxels * voxel_volume_mm3
        # equivalent sphere diameter from volume: d = 2*(3V/4pi)^(1/3)
        equiv_diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        if equiv_diameter >= min_diameter_mm:
            cleaned[component] = 1
    return cleaned.astype(np.uint8)


# ---------------------------------------------------------------------------
# Preprocessing: HU clipping/rescaling, lung-field crop                (Sec 3.3, 3.4)
# ---------------------------------------------------------------------------
def clip_and_rescale_hu(slice_hu: np.ndarray) -> np.ndarray:
    """Clip to [-1000, 400] HU then linearly rescale to [0, 1].         (Sec 3.4)"""
    clipped = np.clip(slice_hu, HU_MIN, HU_MAX)
    rescaled = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    return rescaled.astype(np.float32)


def coarse_lung_mask(slice_hu: np.ndarray) -> np.ndarray:
    """
    Coarse lung-field segmentation via intensity thresholding + morphology,
    used only to define the crop region (NOT the nodule location), as
    described in Sec. 3.3.

    This is a standard, simple thresholding approach: air/lung parenchyma
    is much darker than soft tissue, so a fixed HU threshold followed by
    morphological opening/closing gives a usable coarse lung mask for
    crop-centering purposes.
    """
    # Lung parenchyma + airways are typically below ~ -320 HU.
    lung_candidate = slice_hu < -320

    # Remove small noisy specks, then fill small holes.
    lung_candidate = binary_opening(lung_candidate, iterations=2)
    lung_candidate = binary_closing(lung_candidate, iterations=2)

    # Keep only the largest connected components (left + right lung).
    labeled, n_components = cc_label(lung_candidate)
    if n_components == 0:
        # Fallback: center crop on the full image.
        return np.ones_like(slice_hu, dtype=bool)

    sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
    order = np.argsort(sizes)[::-1]
    keep_ids = order[:2] + 1  # largest two components (left/right lung)
    lung_mask = np.isin(labeled, keep_ids)
    return lung_mask


def lung_field_crop(slice_2d: np.ndarray, lung_mask: np.ndarray, crop_size: int = CROP_SIZE):
    """
    Crop a `crop_size` x `crop_size` region centered on the bounding box
    of the lung field (not the nodule), padding with the slice's
    background value if the lung field bounding box is smaller than the
    target crop, or center-cropping if the slice is already smaller.    (Sec 3.3)
    """
    ys, xs = np.where(lung_mask)
    h, w = slice_2d.shape

    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy = (ys.min() + ys.max()) // 2
        cx = (xs.min() + xs.max()) // 2

    half = crop_size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - h)
    pad_right = max(0, x1 - w)

    padded = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    y0p, x0p = y0 + pad_top, x0 + pad_left
    cropped = padded[y0p:y0p + crop_size, x0p:x0p + crop_size]
    return cropped


def extract_nodule_bearing_slices(volume_hu: np.ndarray, consensus_mask_3d: np.ndarray) -> List[int]:
    """
    Return the indices of axial slices for which the consensus mask
    contains at least one foreground voxel. No interpolation or slice
    skipping; every qualifying slice is retained, including all slices
    spanned by a multi-slice nodule.                                    (Sec 3.3)
    """
    assert volume_hu.shape == consensus_mask_3d.shape
    n_slices = volume_hu.shape[0]
    indices = [z for z in range(n_slices) if consensus_mask_3d[z].sum() > 0]
    return indices


# ---------------------------------------------------------------------------
# Augmentations                                                          (Sec 3.4)
# ---------------------------------------------------------------------------
def build_train_augmentations():
    """
    Albumentations pipeline matching the paper's described augmentations:
    random rotation (+-15 deg), horizontal/vertical flips, elastic
    deformation, Gaussian noise -- all applied probabilistically on-the-fly.
    """
    if A is None:
        raise ImportError("albumentations is required for training augmentations: pip install albumentations")

    return A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3, border_mode=0),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LungNoduleSliceDataset(Dataset):
    """
    2D slice-level dataset over pre-extracted (image, mask) pairs.

    This class assumes preprocessing (HU clipping/rescaling, lung-field
    cropping, mask binarization, nodule-bearing-slice filtering) has
    already produced a flat list of 2D numpy arrays -- see
    `preprocess_volume_to_slices` below for turning a raw (volume, mask)
    pair into the inputs this dataset expects. Keeping the dataset itself
    decoupled from heavy preprocessing makes it easy to cache
    preprocessed slices to disk for fast data loading.

    Parameters
    ----------
    images : list of 2D float32 numpy arrays in [0, 1], shape (512, 512)
    masks  : list of 2D binary numpy arrays, shape (512, 512)
    patient_ids : list of patient identifiers, one per slice (used for
        patient-level CV splitting upstream; not required by __getitem__).
    augment : bool
        Whether to apply the training augmentation pipeline.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        patient_ids: Optional[List[str]] = None,
        augment: bool = False,
    ):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.patient_ids = patient_ids or [None] * len(images)
        self.augment = augment
        self.transform = build_train_augmentations() if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].astype(np.float32)
        mask = (self.masks[idx] > 0.5).astype(np.float32)  # binarize at 0.5 (Sec 3.4)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()    # (1, H, W)
        return image_t, mask_t


def preprocess_volume_to_slices(volume_hu: np.ndarray, consensus_mask_3d: np.ndarray):
    """
    Convert one CT volume + consensus mask into the list of preprocessed
    2D (image, mask) slice pairs the paper trains on:
      1. Keep only nodule-bearing axial slices.
      2. Clip/rescale HU -> [0,1].
      3. Crop 512x512 centered on the lung field bounding box.
      4. Binarize the mask at 0.5 (already binary here, kept for symmetry).
    """
    slice_indices = extract_nodule_bearing_slices(volume_hu, consensus_mask_3d)

    images, masks = [], []
    for z in slice_indices:
        hu_slice = volume_hu[z]
        mask_slice = consensus_mask_3d[z]

        lung_mask = coarse_lung_mask(hu_slice)
        rescaled = clip_and_rescale_hu(hu_slice)

        cropped_img = lung_field_crop(rescaled, lung_mask, CROP_SIZE)
        cropped_mask = lung_field_crop(mask_slice.astype(np.float32), lung_mask, CROP_SIZE)

        images.append(cropped_img)
        masks.append((cropped_mask > 0.5).astype(np.float32))

    return images, masks


# ---------------------------------------------------------------------------
# Patient-level 5-fold cross-validation splitting                        (Sec 3.3)
# ---------------------------------------------------------------------------
def patient_level_kfold_splits(patient_ids: Sequence[str], n_folds: int = 5, seed: int = 42):
    """
    Generate patient-level k-fold splits (no patient appears in both train
    and validation within a fold), matching the paper's "five-fold
    cross-validation with patient-level stratification" (Sec 3.3, 4.3).

    Returns a list of (train_patient_ids, val_patient_ids) tuples.
    """
    unique_patients = sorted(set(patient_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    folds = np.array_split(np.array(unique_patients), n_folds)
    splits = []
    for i in range(n_folds):
        val_patients = set(folds[i].tolist())
        train_patients = set(unique_patients) - val_patients
        splits.append((train_patients, val_patients))
    return splits


def slice_indices_for_patients(patient_ids: Sequence[str], wanted_patients: set) -> List[int]:
    """Return slice-level indices whose patient_id is in `wanted_patients`."""
    return [i for i, pid in enumerate(patient_ids) if pid in wanted_patients]
