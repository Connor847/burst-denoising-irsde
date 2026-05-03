"""
inference_burst.py

Runs inference with trained burst-conditioned IR-SDE models on full-resolution
scenes and saves side-by-side comparison images:
    [dark input | N=1 output | N=2 output | GT reference]

Usage:
    python inference/inference_burst.py \
        --checkpoint_n1 checkpoints/burst1_best.pth \
        --checkpoint_n2 checkpoints/burst2_best.pth \
        --full_res_dir  /path/to/processed \
        --output_dir    results/ \
        --config_dir    configs/
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from collections import OrderedDict

import options as option
import utils as util
from models import create_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_n1', required=True)
    parser.add_argument('--checkpoint_n2', required=True)
    parser.add_argument('--full_res_dir',  required=True)
    parser.add_argument('--output_dir',    required=True)
    parser.add_argument('--config_dir',    default='configs')
    parser.add_argument('--patch_size',    type=int, default=512)
    return parser.parse_args()


def load_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img[:, :, ::-1].astype(np.float32) / 65535.0
    return img


def to_tensor(img, device):
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return t


def tensor2img(tensor):
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def auto_enhance(img):
    """Stretch to full dynamic range for visualization."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def white_balance(img):
    """Remove color cast by equalizing channel means."""
    result = img.copy().astype(np.float32)
    mean = result.mean()
    for i in range(3):
        ch_mean = result[:, :, i].mean()
        if ch_mean > 0:
            result[:, :, i] *= mean / ch_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def load_model_for_inference(n_frames, checkpoint_path, config_path, device):
    opt = option.parse(config_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt['dist'] = False
    opt['path']['pretrain_model_G'] = None

    model = create_model(opt)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_ckpt[k.replace('module.', '')] = v

    model.model.load_state_dict(new_ckpt, strict=False)
    model.model.eval()
    print(f"Loaded N={n_frames} — init_conv: {new_ckpt['init_conv.weight'].shape}")
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    short_full = os.path.join(args.full_res_dir, 'short_full')
    long_full  = os.path.join(args.full_res_dir, 'long_full')

    scenes = sorted(set([
        fname.split('_')[0]
        for fname in os.listdir(long_full)
        if fname.endswith('_GT.png')
    ]))
    print(f"Full res scenes: {scenes}")

    # Load models
    model_n1 = load_model_for_inference(
        1, args.checkpoint_n1,
        os.path.join(args.config_dir, 'burst1_train.yml'), device
    )
    model_n2 = load_model_for_inference(
        2, args.checkpoint_n2,
        os.path.join(args.config_dir, 'burst2_train.yml'), device
    )

    # Separate SDE per model
    def make_sde(model, config_path):
        opt = option.parse(config_path, is_train=False)
        opt = option.dict_to_nonedict(opt)
        sde = util.DenoisingSDE(
            max_sigma=opt['sde']['max_sigma'],
            T=opt['sde']['T'],
            schedule=opt['sde']['schedule'],
            device=device
        )
        sde.set_model(model.model)
        return sde, opt

    sde_n1, opt_n1 = make_sde(model_n1, os.path.join(args.config_dir, 'burst1_train.yml'))
    sde_n2, opt_n2 = make_sde(model_n2, os.path.join(args.config_dir, 'burst2_train.yml'))

    ps = args.patch_size

    for scene_id in scenes:
        print(f"\nProcessing {scene_id}...")

        gt_img = load_png(os.path.join(long_full, f'{scene_id}_GT.png'))
        burst_paths = sorted([
            f for f in os.listdir(short_full)
            if f.startswith(scene_id + '_burst')
        ])
        burst_imgs = [load_png(os.path.join(short_full, f)) for f in burst_paths]

        # Center crop
        h, w = gt_img.shape[:2]
        top  = (h - ps) // 2
        left = (w - ps) // 2

        gt_patch    = gt_img[top:top+ps, left:left+ps]
        dark_patch  = burst_imgs[0][top:top+ps, left:left+ps]
        b_patches   = [img[top:top+ps, left:left+ps] for img in burst_imgs]

        gt_t  = to_tensor(gt_patch, device)
        b1    = to_tensor(b_patches[0], device)
        b2    = torch.cat([to_tensor(b_patches[0], device),
                           to_tensor(b_patches[1], device)], dim=1)

        with torch.no_grad():
            # N=1 inference
            # TODO (future work): replace randn with burst mean
            burst_mean_n1 = b1  # already 3 channels for N=1
            combined_n1   = torch.cat([burst_mean_n1, b1], dim=1)
            model_n1.feed_data(combined_n1, gt_t)
            model_n1.test(sde_n1, sigma=opt_n1['degradation']['sigma'])
            output_n1 = tensor2img(model_n1.get_current_visuals()['Output'])

            # N=2 inference
            burst_mean_n2 = (to_tensor(b_patches[0], device) +
                 to_tensor(b_patches[1], device)) / 2  # [B, 3, H, W]
            combined_n2   = torch.cat([burst_mean_n2, b2], dim=1)
            model_n2.feed_data(combined_n2, gt_t)
            model_n2.test(sde_n2, sigma=opt_n2['degradation']['sigma'])
            output_n2 = tensor2img(model_n2.get_current_visuals()['Output'])

        # Enhance for visualization
        dark_8  = white_balance(auto_enhance(dark_patch))
        gt_8    = auto_enhance(gt_patch)
        out_n1  = white_balance(auto_enhance(output_n1))
        out_n2  = white_balance(auto_enhance(output_n2))

        # Save comparison: dark | n1 | n2 | gt
        comparison = np.concatenate([dark_8, out_n1, out_n2, gt_8], axis=1)
        out_path = os.path.join(args.output_dir, f'{scene_id}_comparison.png')
        cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")

    print("\nInference complete.")


if __name__ == '__main__':
    main()
