# Beyond U-Net++: Residual and Attention Enhancements for Lung Nodule Segmentation

PyTorch implementation of the method described in *"Beyond U-Net++: Residual
and Attention Enhancements for Precise Lung Nodule Segmentation"*
(Golrizkhatami & Taheri).

## What this implements

| Paper section | Code |
|---|---|
| 4.2.2 Residual Blocks (Eq. 1-2) | `models/blocks.py::ResidualConvBlock` |
| 4.2.3 Attention Gates (Eq. 3-5) | `models/blocks.py::AttentionGate` |
| 4.2.1 / 4.2.5 U-Net++ backbone + layer config | `models/residual_attention_unetpp.py` |
| 4.2.4 Hybrid Dice + Focal loss (Eq. 6-8) | `losses/hybrid_loss.py` |
| 4.4 Evaluation metrics (Eq. 9-15) | `utils/metrics.py` |
| 3.3 / 3.4 Data curation & preprocessing | `data/lidc_dataset.py` |
| 4.3 Training setup (optimizer, scheduler, CV, early stopping) | `train.py`, `configs/default.yaml` |

## Architecture

A U-Net++ backbone (4 encoder/decoder levels + bottleneck) where:
- every convolutional unit (encoder column **and** every nested decoder
  node) is a `ResidualConvBlock` instead of a plain conv-BN-ReLU pair,
- every skip connection feeding a decoder node passes through an
  `AttentionGate`, gated by the upsampled feature map from the row below,
- deep supervision is disabled by default, matching the paper's stated
  design choice (the network optimizes only the final prediction).

The loss is `0.5 * DiceLoss + 0.5 * FocalLoss` (Eq. 8), computed only on the
final output.

## Quickstart

```bash
pip install torch albumentations scipy pyyaml

# Sanity-check the full training loop with synthetic data (no real dataset needed)
python train.py --dry-run --config configs/default.yaml --fold 0

# Check model parameter count / output shape
python -m models.residual_attention_unetpp
```

To train on real data, populate a `LungNoduleSliceDataset` per fold using
the preprocessing primitives in `data/lidc_dataset.py`
(`build_consensus_mask`, `preprocess_volume_to_slices`,
`patient_level_kfold_splits`) from your LIDC-IDRI loader of choice (e.g.
`pylidc`), then call `train_one_fold` from `train.py` for each of the 5
folds.

## Important implementation notes / discrepancies found in the paper

1. **Parameter count vs. stated filter widths.** Section 4.2.5 states filter
   widths of 64→128→256→512 with a 1024-filter bottleneck, and Section 5.3
   reports the proposed model has **11.2M** parameters (baseline U-Net++:
   9.8M). Building the literal U-Net++ nested topology at those filter
   widths with residual blocks and attention gates, as the paper describes
   it, in fact yields **~40M** parameters — the dense column-wise
   concatenations in U-Net++ compound quickly. A filter width of
   32→64→128→256 with a 512-filter bottleneck reproduces the paper's stated
   ~10-11M parameter budget almost exactly. **This is a genuine
   inconsistency in the paper's own numbers, not a modeling choice on our
   part** — both options are exposed via `configs/default.yaml`
   (`model.base_filters`, `model.bottleneck_filters`); the calibrated
   (smaller) widths are the default since they match the paper's reported
   complexity analysis (Table 3).
2. **Focal loss `alpha`/`gamma`.** The paper defines these in Eq. 7 but
   never states the numeric values used. Defaults of `alpha=0.8,
   gamma=2.0` (common literature defaults) are used; override via the
   config (`loss.focal_alpha`, `loss.focal_gamma`) if you have the
   original values.
3. **LIDC-IDRI loading is not included.** Parsing raw DICOM + radiologist
   XML annotations requires `pylidc` and the actual dataset download
   (tens of GB), which is outside this repo's scope. `data/lidc_dataset.py`
   provides every preprocessing step *after* you have a volume + per
   radiologist 3D mask array (consensus building, HU clipping, lung-field
   cropping, nodule-slice extraction, augmentation, patient-level k-fold
   splitting) — you only need to supply the raw volume/annotation loading.
4. **Lung-field segmentation** (used only to center the 512×512 crop, not
   as a nodule prior) uses a standard fixed HU threshold (-320 HU) +
   morphological opening/closing + largest-2-components heuristic, since
   the paper doesn't specify the exact lung-segmentation algorithm.

## Repository layout

```
models/
  blocks.py                       # ResidualConvBlock, AttentionGate
  residual_attention_unetpp.py    # full network
losses/
  hybrid_loss.py                  # Dice, Focal, combined hybrid loss
utils/
  metrics.py                      # DSC, IoU, HD95, ASSD, Sensitivity, PPV, HD
data/
  lidc_dataset.py                 # preprocessing + Dataset + CV splitting
configs/
  default.yaml                    # all hyperparameters from Sec 4.3
train.py                          # training loop, 5-fold CV, early stopping
```
