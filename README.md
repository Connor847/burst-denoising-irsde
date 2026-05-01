# burst-denoising-irsde

An empirical study of burst-conditioned score function estimation in diffusion-based low-light image restoration. We modify [IR-SDE](https://github.com/Algolzw/image-restoration-sde) to accept N burst frames as additional conditioning at each reverse diffusion step, and evaluate whether burst count affects denoising quality on the [SID dataset](https://cchen156.github.io/SID.html).

> **Course project** — Computer Vision Principles II, Columbia University (Aleksander Holynski, Spring 2026)

---

## Research Question

Does conditioning the score function of a diffusion-based restoration model on multiple burst observations at every reverse step improve low-light image restoration quality — and does the benefit scale with burst count?

---

## Method

### Key insight

Standard IR-SDE uses a single degraded 3-channel image as both the SDE starting state and the model input at each reverse step. We extend this by concatenating N burst frames alongside the SDE state at every one of the 100 reverse steps, giving the model richer scene information to estimate the score function throughout the entire denoising trajectory.

This is distinct from simple burst averaging — rather than collapsing N frames into one before processing, we let the model learn to use the per-frame noise structure directly.

### Architecture

```
Burst frames (N)          SDE noisy state xt
  [3×N channels]             [3 channels]
        │                         │
        └──────────┬──────────────┘
                   ▼
           Concatenate inputs
         [combined: 3 + 3N ch]
                   │
                   ▼
         init_conv (modified)
         [64, 3+3N, 7, 7]
         pretrained + scaled copies
                   │
                   ▼
         ConditionalUNet
         burst-conditioned score estimate
         [3ch output]
                   │
                   ▼
           Score function
                   │
                   ▼
         SDE reverse step          ←──────────────┐
         math on xt[:3] only                      │
         [3ch state updated]       ── ×100 steps ─┘
                   │
                   ▼
         Restored image [3ch RGB]
```

**Key design principle:** SDE math always operates on 3-channel xt. Burst frames are conditioning, not state. This required patching three components of IR-SDE:
- `optimize_parameters` in `denoising_model.py` — extract `sde_state = self.LQ[:, :3, :, :]` for SDE math
- `test` in `denoising_model.py` — separate burst from SDE state before calling `reverse_ode`
- `reverse_ode` in `sde_utils.py` — accept `burst=None` parameter and recondition at each step

### Weight initialization

The pretrained IR-SDE checkpoint has `init_conv.weight` of shape `[64, 3, 7, 7]`. For burst conditioning:

- **SDE state channels (first 3):** pretrained weights preserved exactly
- **Burst channels (remaining 3×N):** initialized as scaled copies of pretrained weights divided by N

This preserves the pretrained feature detectors for the SDE state while giving burst channels a sensible initialization that maintains activation magnitude.

### Ablation conditions

| Condition | in_nc | init_conv shape |
|-----------|-------|-----------------|
| N = 1 | 6 | [64, 6, 7, 7] |
| N = 2 | 9 | [64, 9, 7, 7] |
| N = 4 | 15 | [64, 15, 7, 7] |

---

## Dataset

**SID Sony split** — Learning to See in the Dark (Chen et al., 2018)

- 179 scenes with ≥4 short-exposure frames at 0.1s exposure
- Long-exposure ground truth at 10s (100× underexposure ratio, ~6-7 stops)
- Preprocessed to 512×512 PNG patches (8 patches per scene) for training
- 5 full-resolution scenes preserved for qualitative evaluation
- 80/20 train/val split with fixed seed (42) identical across all conditions

---

## Results

### PSNR (dB) — 3 epochs, fixed training budget

| Epoch | N=1 | N=2 | Gap |
|-------|-----|-----|-----|
| 1 | 8.86 | 9.15 | +0.29 |
| 2 | 9.07 | 9.30 | +0.23 |
| 3 | 9.19 | 9.73 | +0.54 |

N=2 consistently outperforms N=1 across all epochs under identical training conditions. The gap widens with training, suggesting the model learns to use the additional burst frame more effectively as training progresses.

**Note on absolute PSNR:** The absolute values (~9 dB) are significantly below state-of-the-art (~28-30 dB) due to three factors:
1. Only 3 training epochs — far from convergence
2. Known initialization issue (see below)
3. Domain gap between synthetic Gaussian pretraining and real RAW sensor noise

The relative comparison between conditions is the primary finding.

---

## Known Limitation: Gaussian Noise Initialization

**This is a known suboptimal design choice that will be corrected in subsequent work.**

During inference, the current implementation initializes the SDE starting state from Gaussian noise (`torch.randn_like(gt)`) rather than from the actual degraded input image. This occurred because separating the burst conditioning from the SDE state required a 3-channel initialization, and random noise was used as a placeholder.

The correct approach — and the intended behavior of IR-SDE — is to use the mean of the burst frames as the SDE starting state:

```python
# Correct initialization (planned fix)
burst_reshaped = burst.reshape(B, n_frames, 3, H, W)
sde_start = burst_reshaped.mean(dim=1)  # [B, 3, H, W]
combined = torch.cat([sde_start, burst], dim=1)
```

Starting from the burst mean gives the reverse ODE real scene content to work from, significantly shortening the distance to the clean image and improving output quality. Fixing this is the first priority in follow-up experiments.

---

## Repository Structure

```
burst-denoising-irsde/
├── README.md
├── requirements.txt
├── preprocess/
│   └── preprocess_sid.py        # ARW → 512×512 PNG patches
├── data/
│   └── sid_burst_dataset.py     # PyTorch Dataset for burst loading
├── training/
│   └── train_burst.py           # Training loop for all burst conditions
├── inference/
│   └── inference_burst.py       # Inference + qualitative comparison
├── configs/
│   ├── burst1_train.yml
│   ├── burst2_train.yml
│   └── burst4_train.yml
└── patches/
    ├── denoising_model_patch.py  # Patches for IR-SDE denoising_model.py
    └── sde_utils_patch.py        # Patches for IR-SDE sde_utils.py
```

---

## Setup

This repo requires the IR-SDE codebase as a dependency.

```bash
# Clone IR-SDE
git clone https://github.com/Algolzw/image-restoration-sde.git
cd image-restoration-sde

# Install dependencies
pip install einops lmdb lpips ema-pytorch rawpy

# Apply patches
python patches/denoising_model_patch.py
python patches/sde_utils_patch.py

# Add to path
export PYTHONPATH=$PYTHONPATH:codes/config/denoising-sde:codes
```

Download pretrained IR-SDE denoising weights from the [official release](https://github.com/Algolzw/image-restoration-sde) and place at `weights/ir-sde-denoising.pth`.

Download the SID Sony dataset from the [official source](https://cchen156.github.io/SID.html).

---

## Preprocessing

```bash
python preprocess/preprocess_sid.py \
  --short_dir /path/to/SID/Sony/short \
  --long_dir /path/to/SID/Sony/long \
  --out_dir /path/to/processed \
  --n_frames 4 \
  --patch_size 512 \
  --n_patches 8
```

---

## Training

```bash
# Train each burst condition separately
python training/train_burst.py --n_frames 1 --config configs/burst1_train.yml
python training/train_burst.py --n_frames 2 --config configs/burst2_train.yml
python training/train_burst.py --n_frames 4 --config configs/burst4_train.yml
```

---

## Inference

```bash
python inference/inference_burst.py \
  --checkpoint_n1 checkpoints/burst1_best.pth \
  --checkpoint_n2 checkpoints/burst2_best.pth \
  --full_res_dir /path/to/processed/full \
  --output_dir results/
```

---

## Future Work

See [FUTURE_WORK.md](FUTURE_WORK.md) for a detailed roadmap including the Gaussian initialization fix and video diffusion extension.

---

## Citation

If you use this work, please also cite the original IR-SDE paper:

```bibtex
@article{luo2023image,
  title={Image Restoration with Mean-Reverting Stochastic Differential Equations},
  author={Luo, Ziwei and Gustafsson, Fredrik K and Zhao, Zheng and Sj{\"o}lund, Jens and Sch{\"o}n, Thomas B},
  journal={ICML},
  year={2023}
}
```

and the SID dataset:

```bibtex
@inproceedings{chen2018learning,
  title={Learning to See in the Dark},
  author={Chen, Chen and Chen, Qifeng and Xu, Jia and Koltun, Vladlen},
  booktitle={CVPR},
  year={2018}
}
```

---

## License

This project is a research prototype. The IR-SDE codebase and SID dataset have their own respective licenses.
