"""
preprocess_sid.py

Converts SID Sony ARW files to 512x512 PNG patches for training.
Saves aligned burst/GT patch pairs and optionally full-resolution
images for qualitative evaluation.

Usage:
    python preprocess/preprocess_sid.py \
        --short_dir /path/to/SID/Sony/short \
        --long_dir /path/to/SID/Sony/long \
        --out_dir /path/to/processed \
        --n_frames 4 \
        --patch_size 512 \
        --n_patches 8 \
        --n_full 5
"""

import os
import random
import argparse
import rawpy
import numpy as np
import cv2
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--short_dir', required=True)
    parser.add_argument('--long_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_frames', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--n_patches', type=int, default=8)
    parser.add_argument('--n_full', type=int, default=5)
    parser.add_argument('--target_exposure', default='0.1s')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def process_raw(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16
        )
    return rgb  # uint16, HWC, RGB

def save_img(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    args = parse_args()
    random.seed(args.seed)

    out_short_patches = os.path.join(args.out_dir, 'short_patches')
    out_long_patches  = os.path.join(args.out_dir, 'long_patches')
    out_short_full    = os.path.join(args.out_dir, 'short_full')
    out_long_full     = os.path.join(args.out_dir, 'long_full')

    for d in [out_short_patches, out_long_patches, out_short_full, out_long_full]:
        os.makedirs(d, exist_ok=True)

    # Group short exposures by scene
    scene_bursts = defaultdict(list)
    for fname in sorted(os.listdir(args.short_dir)):
        if not fname.endswith('.ARW'):
            continue
        parts = fname.replace('.ARW', '').split('_')
        scene_id = parts[0]
        exposure = parts[2]
        if exposure == args.target_exposure:
            scene_bursts[scene_id].append(fname)

    valid_scenes = {
        k: sorted(v)[:args.n_frames]
        for k, v in scene_bursts.items()
        if len(v) >= args.n_frames
    }
    print(f"Valid scenes: {len(valid_scenes)}")

    # Long exposure lookup
    long_lookup = {}
    for fname in os.listdir(args.long_dir):
        if fname.endswith('.ARW'):
            scene_id = fname.split('_')[0]
            long_lookup[scene_id] = fname

    full_res_scenes = set(sorted(valid_scenes.keys())[:args.n_full])
    print(f"Full res scenes: {full_res_scenes}")

    # Check already processed
    already_done = set()
    for fname in os.listdir(out_long_patches):
        if fname.endswith('_GT.png'):
            already_done.add(fname.split('_')[0])
    print(f"Already processed: {len(already_done)}, Remaining: {len(valid_scenes) - len(already_done)}")

    processed = skipped = 0

    for scene_id, frames in valid_scenes.items():
        if scene_id in already_done:
            continue
        if scene_id not in long_lookup:
            print(f"No GT for scene {scene_id}, skipping")
            skipped += 1
            continue

        try:
            gt_img = process_raw(os.path.join(args.long_dir, long_lookup[scene_id]))
        except Exception as e:
            print(f"Failed GT for {scene_id}: {e}")
            skipped += 1
            continue

        burst_imgs = []
        success = True
        for fname in frames:
            try:
                lq_img = process_raw(os.path.join(args.short_dir, fname))
                burst_imgs.append(lq_img)
            except Exception as e:
                print(f"Failed burst for {scene_id}: {e}")
                success = False
                break

        if not success:
            skipped += 1
            continue

        # Save full resolution
        if scene_id in full_res_scenes:
            save_img(gt_img, os.path.join(out_long_full, f'{scene_id}_GT.png'))
            for i, lq_img in enumerate(burst_imgs):
                save_img(lq_img, os.path.join(out_short_full, f'{scene_id}_burst{i:02d}.png'))

        # Save aligned patches
        h, w = gt_img.shape[:2]
        for p in range(args.n_patches):
            top  = random.randint(0, h - args.patch_size)
            left = random.randint(0, w - args.patch_size)

            gt_patch = gt_img[top:top+args.patch_size, left:left+args.patch_size]
            save_img(gt_patch, os.path.join(out_long_patches, f'{scene_id}_p{p:02d}_GT.png'))

            for i, lq_img in enumerate(burst_imgs):
                lq_patch = lq_img[top:top+args.patch_size, left:left+args.patch_size]
                save_img(lq_patch, os.path.join(out_short_patches,
                                                f'{scene_id}_p{p:02d}_burst{i:02d}.png'))

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(valid_scenes)} scenes")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Patches — GT: {len(os.listdir(out_long_patches))}, "
          f"Burst: {len(os.listdir(out_short_patches))}")
    print(f"Full res — GT: {len(os.listdir(out_long_full))}, "
          f"Burst: {len(os.listdir(out_short_full))}")

if __name__ == '__main__':
    main()
