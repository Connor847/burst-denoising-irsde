"""
denoising_model_patch.py

Applies required patches to IR-SDE's denoising_model.py to support
burst conditioning (Approach B).

Two changes:
1. optimize_parameters — use only first 3 channels for SDE math
2. test — separate burst from SDE state before calling reverse_ode

Usage:
    python patches/denoising_model_patch.py \
        --irsde_path /path/to/image-restoration-sde
"""

import argparse
import os


def patch(irsde_path):
    target = os.path.join(
        irsde_path,
        'codes/config/denoising-sde/models/denoising_model.py'
    )
    assert os.path.exists(target), f"Not found: {target}"

    with open(target, 'r') as f:
        content = f.read()

    # ── Patch 1: optimize_parameters ──────────────────────────────────────
    old1 = (
        "        noise = self.model(self.LQ, timesteps.squeeze())\n"
        "        score = sde.get_score_from_noise(noise, timesteps)\n"
        "\n"
        "        # Learning the maximum likelihood objective for state x_{t-1}\n"
        "        xt_1_expection = sde.reverse_sde_step_mean(self.LQ, score, timesteps)\n"
        "        xt_1_optimum = sde.reverse_optimum_step(self.LQ, self.GT, timesteps)"
    )
    new1 = (
        "        noise = self.model(self.LQ, timesteps.squeeze())\n"
        "        score = sde.get_score_from_noise(noise, timesteps)\n"
        "\n"
        "        # Use only first 3 channels (SDE state) for SDE math\n"
        "        # Burst conditioning channels are in self.LQ[:, 3:, :, :]\n"
        "        sde_state = self.LQ[:, :3, :, :]\n"
        "        xt_1_expection = sde.reverse_sde_step_mean(sde_state, score, timesteps)\n"
        "        xt_1_optimum = sde.reverse_optimum_step(sde_state, self.GT, timesteps)"
    )

    # ── Patch 2: test ──────────────────────────────────────────────────────
    old2 = (
        "    def test(self, sde=None, sigma=-1, save_states=False):\n"
        "        timesteps = sde.T if sigma < 0 else sde.get_optimal_timestep(sigma)\n"
        "\n"
        "        self.model.eval()\n"
        "        with torch.no_grad():\n"
        "            # self.output = sde.reverse_sde(self.LQ, T=timesteps, save_states=save_states)\n"
        "            self.output = sde.reverse_ode(self.LQ, T=timesteps, save_states=save_states)\n"
        "\n"
        "        self.model.train()"
    )
    new2 = (
        "    def test(self, sde=None, sigma=-1, save_states=False):\n"
        "        timesteps = sde.T if sigma < 0 else sde.get_optimal_timestep(sigma)\n"
        "\n"
        "        self.model.eval()\n"
        "        with torch.no_grad():\n"
        "            # Separate SDE state from burst conditioning channels\n"
        "            sde_state = self.LQ[:, :3, :, :]\n"
        "            burst = self.LQ[:, 3:, :, :] if self.LQ.shape[1] > 3 else None\n"
        "            self.output = sde.reverse_ode(sde_state, T=timesteps, burst=burst)\n"
        "\n"
        "        self.model.train()"
    )

    changed = 0
    if old1 in content:
        content = content.replace(old1, new1)
        changed += 1
        print("✓ Patched optimize_parameters")
    else:
        print("⚠ optimize_parameters patch not applied — string not found")

    if old2 in content:
        content = content.replace(old2, new2)
        changed += 1
        print("✓ Patched test")
    else:
        print("⚠ test patch not applied — string not found")

    if changed > 0:
        with open(target, 'w') as f:
            f.write(content)
        print(f"Wrote {changed} patch(es) to {target}")
    else:
        print("No patches applied.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--irsde_path', required=True,
                        help='Path to cloned image-restoration-sde repo')
    args = parser.parse_args()
    patch(args.irsde_path)
