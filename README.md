[README.md](https://github.com/user-attachments/files/29284783/README.md)
# Beyond U-Net++: Residual and Attention Enhancements for Precise Lung Nodule Segmentation

Code and configuration files supporting the manuscript *"Beyond U-Net++:
Residual and Attention Enhancements for Precise Lung Nodule Segmentation"*
(Taheri, S., Antalya Bilim University).

This repository provides the full reference implementation of the proposed
Residual + Attention U-Net++ architecture, the hybrid Dice–Focal loss, the
complete evaluation-metric suite, the LIDC-IDRI preprocessing pipeline, and
the training/cross-validation procedure described in the manuscript, so
that all reported results can be reproduced.

---

## Description

The manuscript proposes an enhancement to the U-Net++ architecture for
pulmonary nodule segmentation in CT images. Two architectural modifications
are added to the U-Net++ backbone: (1) residual blocks replace the plain
convolutional units in the encoder and every nested decoder node, improving
gradient flow and training stability; and (2) attention gates are inserted
at every skip connection, allowing the network to suppress irrelevant
background activations and focus on salient nodule regions. The model is
trained with a hybrid loss combining Dice loss and Focal loss to address
class imbalance and boundary ambiguity. The method is evaluated on the
LIDC-IDRI dataset using five-fold, patient-level cross-validation, and
externally validated on the LNDb dataset.

This repository contains:
- the model architecture (residual blocks, attention gates, full
  Residual-Attention U-Net++ network),
- the hybrid Dice + Focal loss function,
- all seven evaluation metrics reported in the manuscript (DSC, IoU, HD95,
  ASSD, Sensitivity, PPV, HD),
- the data curation and preprocessing pipeline (consensus-mask
  construction, HU clipping/rescaling, lung-field-centered cropping,
  nodule-bearing-slice extraction, augmentation, patient-level k-fold
  splitting),
- the training script (optimizer, scheduler, early stopping, 5-fold
  cross-validation loop) and the default configuration matching the
  hyperparameters reported in the manuscript.

---

## Dataset Information

**Primary dataset:** Lung Image Database Consortium and Image Database
Resource Initiative (LIDC-IDRI).
- Publicly available from The Cancer Imaging Archive (TCIA):
  https://www.cancerimagingarchive.net/collection/lidc-idri/
- 1,018 thoracic CT scans, annotated by up to four radiologists per scan.
- This study retained nodules ≥ 3 mm marked by at least 3 of 4
  radiologists, yielding a curated set of 836 nodules from 374 patients
  (see manuscript Section 3.3 for full inclusion criteria).
- License/access terms are set by TCIA; the dataset is not redistributed
  in this repository. Users must download it directly from TCIA under its
  own data usage agreement.

**External validation dataset:** Lung Nodule Database (LNDb).
- Publicly available at https://lndb.grand-challenge.org/
- 294 thoracic CT scans from different scanners/institutions than
  LIDC-IDRI, used only for external validation with no retraining or
  fine-tuning (manuscript Section 3.2).

**Derived/processed data:** This repository does not include preprocessed
images or trained model weights. `data/lidc_dataset.py` implements every
preprocessing step described in the manuscript (HU clipping to
[-1000, 400], rescaling to [0, 1], consensus-mask construction from
radiologist annotations, lung-field-centered 512 × 512 cropping,
nodule-bearing-slice extraction, and mask binarization at 0.5) so that raw
LIDC-IDRI/LNDb volumes can be converted into the exact training inputs
used in the manuscript. Raw DICOM/XML parsing (e.g. via `pylidc`) must be
supplied by the user, since TCIA's license does not permit redistribution
of the imaging data itself.

---

## Code Information

```
beyond_unet/
├── models/
│   ├── blocks.py                     # ResidualConvBlock, AttentionGate, UpSample
│   ├── residual_attention_unetpp.py  # Full Residual-Attention U-Net++ network
│   └── __init__.py
├── losses/
│   ├── hybrid_loss.py                # DiceLoss, FocalLoss, HybridDiceFocalLoss
│   └── __init__.py
├── utils/
│   ├── metrics.py                    # DSC, IoU, HD95, ASSD, Sensitivity, PPV, HD
│   └── __init__.py
├── data/
│   ├── lidc_dataset.py               # Preprocessing pipeline, Dataset, k-fold splitting
│   └── __init__.py
├── configs/
│   └── default.yaml                  # All hyperparameters used in the manuscript
├── train.py                          # Training loop, 5-fold CV, early stopping
├── requirements.txt
├── LICENSE
└── README.md
```

| Manuscript section | Implementation |
|---|---|
| 4.2.1 / 4.2.5 — U-Net++ backbone & layer configuration | `models/residual_attention_unetpp.py` |
| 4.2.2 — Residual blocks (Eq. 1–2) | `models/blocks.py :: ResidualConvBlock` |
| 4.2.3 — Attention gates (Eq. 3–5) | `models/blocks.py :: AttentionGate` |
| 4.2.4 — Hybrid Dice + Focal loss (Eq. 6–8) | `losses/hybrid_loss.py` |
| 4.4 — Evaluation metrics (Eq. 9–15) | `utils/metrics.py` |
| 3.3 / 3.4 — Data curation & preprocessing | `data/lidc_dataset.py` |
| 4.3 — Training configuration (optimizer, scheduler, CV, early stopping) | `train.py`, `configs/default.yaml` |

---

## Usage Instructions

### 1. Install dependencies
```bash
git clone <this-repository-url>
cd beyond_unet
pip install -r requirements.txt
```

### 2. Verify the installation (no dataset required)
```bash
# Check model architecture, parameter count, and output shape
python -m models.residual_attention_unetpp

# Run the full training loop end-to-end on synthetic data
python train.py --dry-run --config configs/default.yaml --fold 0
```

### 3. Prepare LIDC-IDRI for training
Download LIDC-IDRI from TCIA and load it with a tool such as `pylidc`.
For each CT volume, obtain the per-radiologist 3D annotation masks, then:

```python
from data.lidc_dataset import build_consensus_mask, preprocess_volume_to_slices

# radiologist_masks: list of 3D numpy boolean arrays, one per radiologist
consensus_mask = build_consensus_mask(radiologist_masks, min_agreement=3)

# volume_hu: 3D numpy array of HU values for the same CT volume
images, masks = preprocess_volume_to_slices(volume_hu, consensus_mask)
```

Repeat across all patients, track each slice's source `patient_id`, then
build patient-level folds:

```python
from data.lidc_dataset import patient_level_kfold_splits, slice_indices_for_patients
from data.lidc_dataset import LungNoduleSliceDataset

splits = patient_level_kfold_splits(all_patient_ids, n_folds=5, seed=42)
train_patients, val_patients = splits[fold_idx]

train_idx = slice_indices_for_patients(all_patient_ids, train_patients)
val_idx = slice_indices_for_patients(all_patient_ids, val_patients)

train_ds = LungNoduleSliceDataset(
    [all_images[i] for i in train_idx], [all_masks[i] for i in train_idx],
    patient_ids=[all_patient_ids[i] for i in train_idx], augment=True,
)
val_ds = LungNoduleSliceDataset(
    [all_images[i] for i in val_idx], [all_masks[i] for i in val_idx],
    patient_ids=[all_patient_ids[i] for i in val_idx], augment=False,
)
```

### 4. Train a fold
```python
from train import load_config, train_one_fold
import torch

cfg = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_ckpt_path, best_val_dice = train_one_fold(cfg, train_ds, val_ds, fold_idx=0, device=device)
```
Run this once per fold (`fold_idx = 0..4`) to reproduce the manuscript's
five-fold cross-validation. Checkpoints are written to
`checkpoint.out_dir` in `configs/default.yaml` (default: `./checkpoints`).

### 5. Evaluate
```python
from utils.metrics import compute_all_metrics, aggregate_metrics
# Run inference with model.predict_mask(images) on the held-out fold and
# pass (prediction, ground_truth) slice pairs to compute_all_metrics(),
# then aggregate_metrics() across the fold.
```

---

## Requirements

- Python ≥ 3.9
- torch ≥ 2.0
- numpy
- scipy
- albumentations
- PyYAML

All dependencies and version constraints are listed in
[`requirements.txt`](./requirements.txt). The manuscript's reported
experiments used PyTorch 2.0 with CUDA 11.8, an NVIDIA RTX 3090 GPU (24 GB
VRAM), an Intel Core i9-12900K CPU, and 64 GB RAM (manuscript Section 4.3);
a CUDA-capable GPU is strongly recommended for training at the manuscript's
512 × 512 input resolution, though the code runs on CPU for testing and
the included `--dry-run` mode.

---

## Methodology

Full methodological detail is given in the manuscript (Sections 3–4); in
summary, this codebase implements:

1. **Data curation** — radiologist consensus masks (≥ 3 of 4 agreement),
   exclusion of nodules < 3 mm, isotropic resampling to 1 mm, deterministic
   3D-to-2D nodule-bearing slice extraction, and patient-level 5-fold
   splitting to prevent data leakage.
2. **Preprocessing** — HU clipping to [-1000, 400], linear rescaling to
   [0, 1], lung-field-centered (not nodule-centered) 512 × 512 cropping,
   mask binarization at threshold 0.5, and on-the-fly augmentation
   (rotation ±15°, horizontal/vertical flips, elastic deformation,
   Gaussian noise).
3. **Architecture** — a U-Net++ backbone (4 encoder/decoder levels +
   bottleneck) in which every convolutional unit is replaced by a residual
   block, and every skip connection is passed through an attention gate
   before being concatenated into the decoder. Deep supervision is
   disabled, matching the manuscript's design choice to optimize only the
   final prediction.
4. **Loss** — `L_total = 0.5 · L_Dice + 0.5 · L_Focal`, computed on the
   final output layer only.
5. **Training** — Adam optimizer (lr = 1e-4, weight decay = 1e-5),
   `ReduceLROnPlateau` (factor = 0.5, patience = 10 on validation loss),
   batch size 8, maximum 100 epochs, early stopping (patience = 15 on
   validation Dice), global seed 42 for reproducibility.
6. **Evaluation** — DSC, IoU, HD95, ASSD, Sensitivity, PPV, and HD,
   computed per slice and averaged across all five validation folds.

### Known discrepancies between the manuscript text and this implementation

These are documented here in the interest of full reproducibility and
transparency, and should be reconciled with the journal/manuscript text
before final publication:

1. **Parameter count vs. stated filter widths.** Manuscript Section 4.2.5
   states encoder filter widths of 64→128→256→512 with a 1024-filter
   bottleneck, and Section 5.3/Table 3 reports the proposed model has
   **11.2M** parameters (baseline U-Net++: 9.8M). Building the literal
   U-Net++ nested topology at those filter widths with residual blocks and
   attention gates, exactly as described, yields **~40M** parameters,
   because U-Net++'s dense column-wise skip concatenations compound
   quickly. Filter widths of 32→64→128→256 with a 512-filter bottleneck
   reproduce the manuscript's stated ~10–11M parameter budget almost
   exactly. Both options are exposed via `configs/default.yaml`
   (`model.base_filters`, `model.bottleneck_filters`); the calibrated
   (smaller) widths are the default, since they match the manuscript's
   reported complexity analysis (Table 3) rather than its prose
   description of layer widths.
2. **Focal loss `alpha` / `gamma`.** Eq. 7 defines these parameters but
   the manuscript does not state the numeric values used in the reported
   experiments. This implementation defaults to `alpha = 0.8, gamma = 2.0`
   (standard literature defaults); override via `loss.focal_alpha` /
   `loss.focal_gamma` in the config if the original values differ.
3. **Lung-field segmentation heuristic.** Used only to center the
   512 × 512 crop (not as a nodule-location prior, per Section 3.3); the
   manuscript does not specify the exact lung-segmentation algorithm used.
   This implementation uses a fixed HU threshold (< -320 HU) followed by
   morphological opening/closing and retention of the two largest
   connected components.
4. **Raw dataset loading.** Parsing LIDC-IDRI's native DICOM images and
   XML radiologist annotations (e.g. via `pylidc`) is not included, since
   TCIA's data usage terms do not permit redistribution of the imaging
   data itself. `data/lidc_dataset.py` implements every processing step
   downstream of having a raw volume + per-radiologist 3D mask array in
   memory.

---

## Citations

If you use this code, please cite the manuscript:

> Taheri, S. (2026). Beyond U-Net++: Residual and
> Attention Enhancements for Precise Lung Nodule Segmentation. *[Journal
> name, volume, pages — to be completed upon acceptance]*.

Architectural components implemented here build on the following prior
work, as cited in the manuscript:

> Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018).
> UNet++: A Nested U-Net Architecture for Medical Image Segmentation. In
> *Deep Learning in Medical Image Analysis and Multimodal Learning for
> Clinical Decision Support* (pp. 3–11). Springer.

> Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa,
> K., Mori, K., McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., &
> Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the
> Pancreas. *arXiv preprint arXiv:1804.03999*.

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal
> Loss for Dense Object Detection. In *Proceedings of the IEEE
> International Conference on Computer Vision* (pp. 2980–2988).

Dataset citations:

> Armato III, S. G., et al. (2011). The Lung Image Database Consortium
> (LIDC) and Image Database Resource Initiative (IDRI): A completed
> reference database of lung nodules on CT scans. *Medical Physics*,
> 38(2), 915–931.

> Pedrosa, J., et al. (2019). LNDb: A Lung Nodule Database on Computed
> Tomography. *arXiv preprint arXiv:1911.08434*.

---

## License & Contribution Guidelines

This code is released under the [MIT License](./LICENSE).

Contributions are welcome via pull request. Please open an issue first to
discuss any significant change. When contributing:
- Keep new functionality consistent with the manuscript's methodology, or
  clearly document any deviation (see "Known discrepancies" above for the
  expected format).
- Include a minimal test or `--dry-run`-style sanity check for new
  modules where practical.
- Match the existing code style (type hints on public functions,
  docstrings referencing the relevant manuscript section/equation number).

For questions about the method itself, contact the corresponding author
(shahram.taheri@antalya.edu.tr). For issues with this code specifically,
please open a GitHub issue.
