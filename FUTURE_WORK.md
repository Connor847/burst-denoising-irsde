# Future Work

This document outlines planned improvements and longer-term research directions for burst-conditioned diffusion-based low-light image restoration.

---

## Priority 1: Fix Gaussian Noise Initialization

**Status:** Known issue in current implementation  
**Effort:** Low — a few lines of code  
**Expected impact:** High — should significantly improve absolute PSNR

The current implementation initializes the SDE reverse ODE from Gaussian noise during both training validation and inference. This is incorrect — IR-SDE was designed to start from the degraded input image, which already contains scene structure. Starting from pure noise means the model must reconstruct the entire scene from scratch using only burst conditioning, which is a much harder problem requiring significantly more training iterations.

**The fix:**

```python
# Replace this
noisy_gt_val = torch.randn_like(gt)
combined_val = torch.cat([noisy_gt_val, burst], dim=1)

# With this
B, C, H, W = burst.shape
n_frames = C // 3
burst_reshaped = burst.reshape(B, n_frames, 3, H, W)
sde_start = burst_reshaped.mean(dim=1)  # burst mean as starting state
combined_val = torch.cat([sde_start, burst], dim=1)
```

Using the burst mean as the starting state is more principled for two reasons:
1. It restores the original IR-SDE design intent — start from a degraded but real image
2. The burst mean has reduced shot noise relative to any single frame, giving the ODE a better starting point

**Planned experiment:** Rerun the full N=1, N=2, N=4 ablation with this fix, converged training (50k+ iterations), and proper baselines. This is the minimum required to produce publication-quality results.

---

## Priority 2: Converged Training with Proper Baselines

**Status:** Not started  
**Effort:** High — compute time  
**Expected impact:** Required for any publication

Current results use only 3 training epochs. To produce credible quantitative results:

- Train each condition to convergence (50,000+ iterations)
- Run multiple seeds and report mean ± std
- Compare against: original SID UNet, single-frame IR-SDE baseline, and ideally one burst CNN baseline (e.g. KPN or BPN)
- Evaluate on the standard SID test split with the misaligned scenes (10034, 10045, 10172) excluded per the dataset errata

---

## Priority 3: Correct Burst Conditioning Architecture

**Status:** Theoretical — not implemented  
**Effort:** Medium  
**Expected impact:** Likely improves over simple concatenation

The current approach concatenates burst frames with the SDE state — a simple but potentially suboptimal fusion strategy. More principled alternatives:

**Cross-attention conditioning:** At each reverse step, compute cross-attention between the current SDE state and the burst frame stack. This allows the model to attend to relevant regions across frames rather than treating all channels equally.

```python
# Conceptual
sde_features = encoder(xt)          # [B, C, H, W]
burst_features = burst_encoder(burst_frames)  # [B, N, C, H, W]
conditioned = cross_attention(sde_features, burst_features)
score = decoder(conditioned, timestep)
```

**Noise-weighted fusion:** Frames with lower estimated noise should contribute more to the score estimate. Learn a per-frame confidence weight from the noise variance.

**Temporal position encoding:** Even in burst photography, frames have a capture order. Adding positional encoding to burst frames may help the model learn temporal patterns in noise.

---

## Priority 4: Noise Level Calibration

**Status:** Theoretical  
**Effort:** Medium  
**Expected impact:** Improves training stability and inference quality

IR-SDE was pretrained on synthetic Gaussian noise with `max_sigma=70`. Real SID underexposure noise has a different statistical distribution — Poisson-dominated shot noise at very low photon counts. The mismatch between pretraining noise and real noise is a source of suboptimal performance.

**Approach:** Measure the noise variance of SID short-exposure images and find the SDE timestep `t*` such that the SDE forward process at `t*` produces noise with matching variance. Use `t*` as the fixed timestep for real pairs rather than sampling randomly across the full range.

```python
# Measure empirical noise variance
noise_var = burst_frames.var(dim=1).mean()  # variance across burst frames

# Find matching SDE timestep
# sigma_bar(t) should approximately equal sqrt(noise_var)
t_star = sde.get_optimal_timestep(sigma=noise_var.sqrt())
```

---

## Longer-Term: Low-Light Video Diffusion

**Status:** Research direction — not started  
**Effort:** Very high — 6-12 month project  
**Target venue:** CVPR/ICCV workshop on computational photography or low-level vision

### Motivation

Low-light video is a harder and more practically relevant problem than burst photography. Current methods use CNN-based temporal fusion with optical flow alignment, which struggles with fast motion and extreme noise. Diffusion models have shown strong results for image restoration but have not been systematically applied to real low-light video with proper RAW sensor noise modeling.

### Core idea

Extend burst conditioning from a single moment to a temporal window. Instead of N simultaneous burst frames, condition the score function on a window of consecutive video frames `[t-k, ..., t-1, t, t+1, ..., t+k]`. The model learns to exploit temporal redundancy — pixels that are consistent across frames are likely signal; pixels that vary randomly are likely noise.

```
Temporal window conditioning:
[frame_t-2, frame_t-1, frame_t, frame_t+1, frame_t+2]
     ↓
Concatenate with SDE state at each reverse step
     ↓
Score function conditioned on 5-frame temporal context
     ↓
Denoised frame_t
```

### Key challenges

**Motion alignment:** Unlike burst photography, video frames have motion between them. Simple concatenation will confuse the model. Options:
- Pre-alignment with optical flow before concatenation (simpler, loses information)
- Implicit alignment learned by the model via deformable convolutions or attention (harder, more powerful)
- Hybrid: coarse alignment + learned refinement

**Temporal consistency:** Denoising each frame independently with a stochastic reverse process produces flickering — consecutive frames follow different random trajectories. Enforcing temporal consistency requires either:
- A shared noise seed across consecutive frames
- An explicit temporal consistency loss penalizing inter-frame differences beyond those in the input
- Joint denoising of the entire temporal window

**Dataset:** Would require a paired dark/bright video dataset. Options include SDSD (Seeing Dynamic Scenes in the Dark) or custom capture with a Sony camera in continuous shooting mode with a static scene and controlled lighting transitions.

### Connection to this work

The burst conditioning architecture developed here is a direct precursor. The input modification strategy, weight initialization approach, and SDE patching methodology all carry over. The primary additions are motion handling and temporal consistency enforcement.

### Why this is interesting

The intersection of three things — real RAW sensor noise modeling, diffusion-based score function estimation, and temporal consistency in video — is essentially unexplored in the literature. Existing video diffusion papers focus on generation, not restoration from real degradation. Existing low-light video papers use CNNs, not diffusion models. The gap is real and the problem is practically important as smartphone cameras increasingly attempt video in dark environments.

---

## Summary Roadmap

| Priority | Task | Effort | Status |
|----------|------|--------|--------|
| 1 | Fix Gaussian initialization | Low | Planned |
| 2 | Converged training + baselines | High | Planned |
| 3 | Cross-attention burst fusion | Medium | Theoretical |
| 4 | Noise level calibration | Medium | Theoretical |
| 5 | Low-light video diffusion | Very high | Research direction |
