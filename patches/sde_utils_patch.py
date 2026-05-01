"""
sde_utils_patch.py

Patches IR-SDE's sde_utils.py to pass burst conditioning through the
DenoisingSDE.reverse_ode method at each reverse step.

Without this patch, burst frames are only used during training
(via the model input) but are silently dropped during inference
because reverse_ode calls score_fn(x, t) with only the 3-channel
SDE state.

Usage:
    python patches/sde_utils_patch.py \
        --irsde_path /path/to/image-restoration-sde
"""

import argparse
import os
import re


def patch(irsde_path):
    target = os.path.join(
        irsde_path,
        'codes/utils/sde_utils.py'
    )
    assert os.path.exists(target), f"Not found: {target}"

    with open(target, 'r') as f:
        content = f.read()

    # Find the DenoisingSDE.reverse_ode — identified by x0=None signature
    old = (
        "    def reverse_ode(self, xt, x0=None, T=-1, save_states=False, save_dir='ode_state'):\n"
        "        T = self.T if T < 0 else T\n"
        "        x = xt.clone()\n"
        "        for t in tqdm(reversed(range(1, T + 1))):\n"
        "            if x0 is not None:\n"
        "                real_score = self.get_real_score(x, x0, t)\n"
        "\n"
        "            score = self.score_fn(x, t)\n"
        "            x = self.reverse_ode_step(x, score, t)"
    )

    new = (
        "    def reverse_ode(self, xt, x0=None, T=-1, save_states=False, save_dir='ode_state', burst=None):\n"
        "        T = self.T if T < 0 else T\n"
        "        x = xt.clone()\n"
        "        for t in tqdm(reversed(range(1, T + 1))):\n"
        "            if x0 is not None:\n"
        "                real_score = self.get_real_score(x, x0, t)\n"
        "\n"
        "            if burst is not None:\n"
        "                # Approach B: recondition score fn with burst at every step\n"
        "                import torch\n"
        "                model_input = torch.cat([x, burst], dim=1)\n"
        "                score = self.score_fn(model_input, t)\n"
        "            else:\n"
        "                score = self.score_fn(x, t)\n"
        "            x = self.reverse_ode_step(x, score, t)"
    )

    if old in content:
        content = content.replace(old, new)
        with open(target, 'w') as f:
            f.write(content)
        print(f"✓ Patched DenoisingSDE.reverse_ode in {target}")
    else:
        # Verify it is already patched
        if 'burst=None' in content:
            print("reverse_ode already patched — skipping")
        else:
            print("⚠ reverse_ode patch not applied — string not found")
            print("  The IR-SDE version may differ from the one used in development.")
            print("  Manually add burst=None parameter and concatenation logic.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--irsde_path', required=True,
                        help='Path to cloned image-restoration-sde repo')
    args = parser.parse_args()
    patch(args.irsde_path)
