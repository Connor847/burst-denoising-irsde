"""
sid_burst_dataset.py

PyTorch Dataset for loading SID burst patches processed by preprocess_sid.py.
Returns paired (burst, GT) tensors for training and validation.
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

SEED = 42


def get_train_val_split(scene_ids, val_ratio=0.2, seed=SEED):
    """
    Returns a deterministic train/val split of scene IDs.
    Always sort before shuffle so the split is filesystem-order independent.
    """
    scenes = sorted(scene_ids)
    random.seed(seed)
    random.shuffle(scenes)
    split = int(len(scenes) * (1 - val_ratio))
    return scenes[:split], scenes[split:]


class SIDBurstDataset(Dataset):
    """
    Loads aligned (burst, GT) patch pairs from preprocessed SID PNG files.

    File naming convention (produced by preprocess_sid.py):
        long_patches/  {scene_id}_p{patch_id}_GT.png
        short_patches/ {scene_id}_p{patch_id}_burst{frame_id}.png

    Args:
        scene_ids:  set of scene ID strings to include
        short_dir:  path to short_patches directory
        long_dir:   path to long_patches directory
        n_frames:   number of burst frames to load per sample
        is_train:   if True, randomly sample n_frames; if False, take first n_frames
    """

    def __init__(self, scene_ids, short_dir, long_dir, n_frames=1, is_train=True):
        self.short_dir = short_dir
        self.long_dir  = long_dir
        self.n_frames  = n_frames
        self.is_train  = is_train

        # Build index of (scene_id, patch_id) pairs
        self.samples = []
        for fname in sorted(os.listdir(long_dir)):
            if not fname.endswith('_GT.png'):
                continue
            parts = fname.replace('_GT.png', '').split('_')
            scene_id = parts[0]
            patch_id = parts[1]
            if scene_id in scene_ids:
                self.samples.append((scene_id, patch_id))

    def load_png(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[:, :, ::-1].astype(np.float32) / 65535.0
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, patch_id = self.samples[idx]

        all_frames = sorted([
            f for f in os.listdir(self.short_dir)
            if f.startswith(f'{scene_id}_{patch_id}_burst')
        ])

        if self.is_train:
            selected = random.sample(all_frames, self.n_frames)
        else:
            selected = all_frames[:self.n_frames]

        burst_frames = [
            self.load_png(os.path.join(self.short_dir, f))
            for f in selected
        ]
        burst = np.concatenate(burst_frames, axis=2)  # [H, W, 3*N]

        gt = self.load_png(os.path.join(self.long_dir,
                                        f'{scene_id}_{patch_id}_GT.png'))

        burst = torch.from_numpy(burst.transpose(2, 0, 1))  # [3*N, H, W]
        gt    = torch.from_numpy(gt.transpose(2, 0, 1))     # [3, H, W]

        return {'LQ': burst, 'GT': gt}
