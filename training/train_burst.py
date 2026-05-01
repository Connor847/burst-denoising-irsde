"""
train_burst.py

Training loop for burst-conditioned IR-SDE ablation.
Runs one burst condition (N=1, 2, or 4) specified via --n_frames.

Usage:
    python training/train_burst.py \
        --n_frames 2 \
        --config configs/burst2_train.yml \
        --processed_dir /path/to/processed \
        --checkpoint_dir /path/to/checkpoints \
        --n_epochs 10 \
        --batch_size 2

Requires IR-SDE repo on PYTHONPATH:
    export PYTHONPATH=$PYTHONPATH:/path/to/ir-sde/codes/config/denoising-sde
    export PYTHONPATH=$PYTHONPATH:/path/to/ir-sde/codes
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import options as option
import utils as util
from models import create_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.sid_burst_dataset import SIDBurstDataset, get_train_val_split

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_frames',       type=int,   required=True)
    parser.add_argument('--config',                     required=True)
    parser.add_argument('--processed_dir',              required=True)
    parser.add_argument('--checkpoint_dir',             required=True)
    parser.add_argument('--n_epochs',       type=int,   default=10)
    parser.add_argument('--batch_size',     type=int,   default=2)
    parser.add_argument('--weights',        default=None,
                        help='Path to pretrained IR-SDE weights')
    return parser.parse_args()


def load_model_with_burst_weights(opt, n_frames, weights_path):
    """
    Create ConditionalUNet with modified init_conv for burst conditioning.
    Approach B: in_nc = 3 (SDE state) + 3*N (burst frames)

    Weight init strategy:
      - SDE state channels [0:3]:   pretrained weights preserved
      - Burst channels [3:3+3N]:    scaled copies of pretrained / N
    """
    opt['path']['pretrain_model_G'] = None  # disable internal loading
    model = create_model(opt)

    if weights_path is not None:
        from collections import OrderedDict
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

        # Strip DataParallel prefix if present
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            new_ckpt[k.replace('module.', '')] = v

        old_weight = new_ckpt['init_conv.weight']            # [64, 3, 7, 7]
        burst_weight = old_weight.repeat(1, n_frames, 1, 1) / n_frames
        new_ckpt['init_conv.weight'] = torch.cat([old_weight, burst_weight], dim=1)
        print(f"init_conv shape: {new_ckpt['init_conv.weight'].shape}")

        model.model.module.load_state_dict(new_ckpt, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")
    else:
        print("No pretrained weights — training from scratch")

    return model


def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    short_dir = os.path.join(args.processed_dir, 'short_patches')
    long_dir  = os.path.join(args.processed_dir, 'long_patches')

    # Build scene split
    all_scenes = sorted(set([
        fname.split('_')[0]
        for fname in os.listdir(long_dir)
        if fname.endswith('_GT.png')
    ]))
    train_scenes, val_scenes = get_train_val_split(all_scenes)
    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

    train_dataset = SIDBurstDataset(set(train_scenes), short_dir, long_dir,
                                     n_frames=args.n_frames, is_train=True)
    val_dataset   = SIDBurstDataset(set(val_scenes),   short_dir, long_dir,
                                     n_frames=args.n_frames, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=1)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    opt = option.parse(args.config, is_train=True)
    opt = option.dict_to_nonedict(opt)
    opt['dist'] = False

    model = load_model_with_burst_weights(opt, args.n_frames, args.weights)
    device = model.device

    sde = util.DenoisingSDE(
        max_sigma=opt['sde']['max_sigma'],
        T=opt['sde']['T'],
        schedule=opt['sde']['schedule'],
        device=device
    )
    sde.set_model(model.model)

    best_psnr  = 0.0
    current_step = 0

    for epoch in range(args.n_epochs):
        model.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            current_step += 1
            burst = batch['LQ'].to(device)  # [B, 3*N, H, W]
            gt    = batch['GT'].to(device)  # [B, 3,   H, W]

            # Approach B: concatenate SDE noisy state with burst frames
            timesteps, noisy_gt = sde.generate_random_states(x0=gt)
            combined = torch.cat([noisy_gt, burst], dim=1)

            model.feed_data(combined, gt)
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(current_step,
                                       warmup_iter=opt['train']['warmup_iter'])

            logs = model.get_current_log()
            epoch_loss += logs['loss']

            if current_step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {current_step}, "
                      f"Loss: {logs['loss']:.4e}")

        # Validation
        model.model.eval()
        avg_psnr = 0.0
        with torch.no_grad():
            for batch in val_loader:
                burst = batch['LQ'].to(device)
                gt    = batch['GT'].to(device)

                # TODO (future work): replace randn with burst mean as SDE start
                # B, C, H, W = burst.shape
                # sde_start = burst.reshape(B, C//3, 3, H, W).mean(dim=1)
                noisy_gt_val = torch.randn_like(gt)
                combined_val = torch.cat([noisy_gt_val, burst], dim=1)

                model.feed_data(combined_val, gt)
                model.test(sde, sigma=opt['degradation']['sigma'])
                visuals = model.get_current_visuals()

                output = util.tensor2img(visuals['Output'].squeeze())
                gt_img = util.tensor2img(visuals['GT'].squeeze())
                avg_psnr += util.calculate_psnr(output, gt_img)

        avg_psnr /= len(val_loader)
        print(f"Epoch {epoch+1} — PSNR: {avg_psnr:.4f} dB, "
              f"Loss: {epoch_loss/len(train_loader):.4e}")

        # Save checkpoints
        ckpt_path = os.path.join(args.checkpoint_dir,
                                  f'burst{args.n_frames}_epoch{epoch+1}.pth')
        torch.save(model.model.state_dict(), ckpt_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(args.checkpoint_dir,
                                      f'burst{args.n_frames}_best.pth')
            torch.save(model.model.state_dict(), best_path)
            print(f"New best PSNR: {best_psnr:.4f} — saved to {best_path}")

    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f} dB")


if __name__ == '__main__':
    main()
