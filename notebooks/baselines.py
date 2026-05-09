# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step C — does the entropy field add information?
#
# Three baselines vs entropy on the same 200 ViT-B/16 images:
#
# 1. **L2 norm** of the residual stream per (layer, patch). Simplest
#    per-patch scalar. No estimator.
# 2. **Attention-rollout entropy** — at each layer, mean attention over
#    heads → per-query-patch entropy of the resulting key distribution.
# 3. **Random-init ViT-B/16** entropy — same Vasicek-on-standardised-
#    residual pipeline but with randomly-initialised weights. Tests
#    whether the L4→L5 phase transition is a property of the trained
#    representation.
#
# Outlier patches defined externally by input-gradient saliency on
# `ViTForImageClassification` (top-K patches by saliency = "object
# patches"). Method comparison is precision against those saliency-
# defined outliers, with paired-bootstrap p-values per image.
#
# Stop condition: if L2-norm gives the same picture as entropy
# (paired-bootstrap p > 0.05 on outlier-localisation accuracy AND
# visually-indistinguishable phase_curves), entropy doesn't add
# information.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image

from eris.extract import extract_entropy_stack
from eris.fields import per_layer_stats
from eris.baselines import (
    attention_rollout_field,
    l2_norm_field,
    random_init_entropy_field,
    saliency_patch_grid,
)

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
RESULTS = REPO / "results" / "baselines"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "results" / "cross_arch" / "vit_b16"
CACHE.mkdir(parents=True, exist_ok=True)

ARCH = "vit_b16"
N_IMAGES = 200
DEVICE = "cuda"
SEED = 0
N_LAYERS = 12
GRID = 14
TOPK = 30                # patches per image used for the outlier-localisation
                          # precision metric (top-K by signal magnitude vs
                          # top-K by saliency)
N_BOOT = 2000             # paired-bootstrap resamples
COMP_LAYER = 8            # 0-indexed → 'L=8' = block 9 (early post-transition)

# %% [markdown]
# ## Sample 200 validation images (same seed as Step A)

# %%
ds = load_dataset("Maysee/tiny-imagenet", split="valid")
rng = np.random.default_rng(SEED)
indices = rng.choice(len(ds), size=N_IMAGES, replace=False)
indices = np.sort(indices)
images = [ds[int(i)]["image"] for i in indices]
print(f"sampled {len(images)} images from tiny-imagenet/valid (seed={SEED})")
print(f"  first 5 indices: {indices[:5]}")
np.save(CACHE / "image_indices.npy", indices)

# %% [markdown]
# ## Compute (or load) the four signal fields

# %%
H_path = CACHE / "H_stack.npz"
if H_path.exists():
    H_stacks = np.load(H_path)["H"]
    print(f"loaded entropy field {H_path}  shape={H_stacks.shape}")
else:
    print("computing entropy field (Step A cache absent)…")
    H_stacks, _spec = extract_entropy_stack(
        ARCH, images, device=DEVICE,
        progress=lambda i, n: print(f"  H[{i}/{n}]", end="\r")
        if i in (1, 25, 50, 100, 150, 200) else None,
    )
    np.savez_compressed(H_path, H=H_stacks, indices=indices)
    print(f"\nsaved {H_path}  shape={H_stacks.shape}")

# %%
L2_path = RESULTS / "L2_stack.npz"
if L2_path.exists():
    L2_stacks = np.load(L2_path)["L2"]
    print(f"loaded L2 field  shape={L2_stacks.shape}")
else:
    print("computing L2 norm field…")
    L2_stacks = l2_norm_field(ARCH, images, device=DEVICE)
    np.savez_compressed(L2_path, L2=L2_stacks)
    print(f"saved {L2_path}  shape={L2_stacks.shape}")

# %%
ATT_path = RESULTS / "attn_stack.npz"
if ATT_path.exists():
    ATT_stacks = np.load(ATT_path)["A"]
    print(f"loaded attn-rollout field  shape={ATT_stacks.shape}")
else:
    print("computing per-layer attention entropy field…")
    ATT_stacks = attention_rollout_field(ARCH, images, device=DEVICE)
    np.savez_compressed(ATT_path, A=ATT_stacks)
    print(f"saved {ATT_path}  shape={ATT_stacks.shape}")

# %%
RND_path = RESULTS / "random_init_H_stack.npz"
if RND_path.exists():
    RND_stacks = np.load(RND_path)["H"]
    print(f"loaded random-init H field  shape={RND_stacks.shape}")
else:
    print("computing random-init entropy field…")
    RND_stacks = random_init_entropy_field(ARCH, images, device=DEVICE, seed=SEED)
    np.savez_compressed(RND_path, H=RND_stacks)
    print(f"saved {RND_path}  shape={RND_stacks.shape}")

# %%
SAL_path = RESULTS / "saliency_topK.npz"
if SAL_path.exists():
    saliency = np.load(SAL_path)["S"]
    print(f"loaded saliency  shape={saliency.shape}")
else:
    print("computing input-gradient saliency on ViT-B/16 ImageNet head…")
    saliency = saliency_patch_grid(images, device=DEVICE, grid=(GRID, GRID))
    np.savez_compressed(SAL_path, S=saliency)
    print(f"saved {SAL_path}  shape={saliency.shape}")

# %% [markdown]
# ## Phase curves overlay (signal mean / std / |∇| / ΔH-std vs depth)
#
# Each signal is averaged across the 200 images per layer, with bootstrap
# bands. Different signals live on different scales, so each panel is
# z-scored (per signal across L) for visual comparability.

# %%
def aggregate_curves(stacks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image per-layer-stats, averaged across images.

    Returns dict of length-L arrays: H_mean, H_std, abs_grad_H_mean, dH_std.
    """
    n = stacks.shape[0]
    keys = ("H_mean", "H_std", "abs_grad_H_mean", "dH_std")
    accum = {k: [] for k in keys}
    for i in range(n):
        s = per_layer_stats(stacks[i])
        for k in keys:
            accum[k].append(s[k])
    return {k: np.stack(accum[k], axis=0) for k in keys}


curves = {
    "entropy": aggregate_curves(H_stacks),
    "L2 norm": aggregate_curves(L2_stacks),
    "attn entropy": aggregate_curves(ATT_stacks),
    "random-init H": aggregate_curves(RND_stacks),
}
print("aggregated curves for", list(curves.keys()))

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
panel_keys = [
    ("H_mean", "signal mean"),
    ("H_std", "signal std (within-layer spread)"),
    ("abs_grad_H_mean", "mean |∇signal| per layer"),
    ("dH_std", "Δsignal std (cross-layer change)"),
]
colors = {
    "entropy": "tab:orange",
    "L2 norm": "tab:blue",
    "attn entropy": "tab:green",
    "random-init H": "tab:purple",
}
layers = np.arange(1, N_LAYERS + 1)

# Pre-compute per-signal z-norm across L (using mean across images first).
norm_curves: dict[str, dict[str, np.ndarray]] = {sig: {} for sig in curves}
for sig in curves:
    for k, _label in panel_keys:
        per_img = curves[sig][k]                       # (n_imgs, L)
        mean_curve = per_img.mean(axis=0)
        sd = mean_curve.std() + 1e-9
        norm_curves[sig][k] = (per_img - mean_curve.mean()) / sd

for ax, (k, label) in zip(axes, panel_keys):
    for sig in curves:
        per_img_z = norm_curves[sig][k]                # (n_imgs, L)
        mean = per_img_z.mean(axis=0)
        # bootstrap CI across images.
        rs = np.random.default_rng(SEED)
        boot = np.empty((400, mean.shape[0]), dtype=np.float32)
        n = per_img_z.shape[0]
        for b in range(400):
            idx = rs.integers(0, n, size=n)
            boot[b] = per_img_z[idx].mean(axis=0)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        ax.fill_between(layers, lo, hi, color=colors[sig], alpha=0.20)
        ax.plot(layers, mean, marker="o", color=colors[sig], lw=1.8, label=sig)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("layer")
    ax.set_xticks(layers)
    ax.grid(alpha=0.30)
axes[0].set_ylabel("z-scored across L  (each signal independently)")
axes[-1].legend(fontsize=9, loc="upper left")
fig.suptitle(
    f"Phase curves — 4 signals overlaid on ViT-B/16, n={N_IMAGES} tiny-imagenet images "
    "(z-scored per signal for visual comparability; bands = 95% bootstrap CI across images)",
    y=1.04,
)
fig.tight_layout()
fig.savefig(RESULTS / "phase_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'phase_curves.png'}")

# %% [markdown]
# ## Outlier-localisation accuracy
#
# For each method × layer:
#   1. Take its top-K patches per image by signal magnitude (L=COMP_LAYER).
#   2. Take top-K saliency-defined patches per image.
#   3. Per-image precision = #(method ∩ saliency) / K.
# Paired bootstrap (2000 resamples over images) on per-image precision
# difference: H − baseline.

# %%
def topk_indices(field_2d: np.ndarray, k: int, mode: str = "high") -> set:
    """Top-k flat indices of a 14×14 field.

    ``mode`` ∈ {'high', 'absdev'}:
      - 'high' — top-k by raw value (saliency-style, where larger == more
        salient). Used for saliency, L2-norm, attn entropy, random-init H.
      - 'absdev' — top-k by ``|x - median(x)|`` (per-image deviation from
        baseline). Used for entropy, where 'outlier patches' are the
        sparse *low*-entropy patches against a near-Gaussian baseline.
        Symmetric — also catches sparse high-entropy patches if they
        exist.
    """
    flat = field_2d.reshape(-1).astype(np.float64)
    if mode == "high":
        score = flat
    elif mode == "absdev":
        score = np.abs(flat - np.median(flat))
    else:
        raise ValueError(mode)
    idx = np.argpartition(-score, kth=k)[:k]
    return set(idx.tolist())


# Map signal name → topk mode. Entropy fields read 'sparse non-Gaussian
# outliers against a uniform baseline' so absolute deviation is the right
# magnitude. Other signals read directly (saliency: high = salient; L2:
# high = large activation; attn entropy: high = uncertain attention; rnd-
# init H: same per-image baseline pattern as entropy).
SIGNAL_MODE = {
    "entropy": "absdev",
    "L2 norm": "high",
    "attn entropy": "high",
    "random-init H": "absdev",
    "saliency": "high",
}


def per_image_precision(stacks: np.ndarray, saliency: np.ndarray, layer: int, k: int,
                        method_mode: str = "high", saliency_mode: str = "high"):
    """``stacks: (n, L, gy, gx)``, ``saliency: (n, gy, gx)`` → per-image precision."""
    n = stacks.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sal_top = topk_indices(saliency[i], k, mode=saliency_mode)
        meth_top = topk_indices(stacks[i, layer], k, mode=method_mode)
        out[i] = len(sal_top & meth_top) / k
    return out


def paired_bootstrap_p(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Two-sided paired-bootstrap p-value and bootstrap-mean.

    Tests H0: mean(diff) = 0 by re-centring; returns p = 2 × min(P(boot ≤ 0), P(boot ≥ 0)).
    """
    rs = np.random.default_rng(seed)
    n = diff.shape[0]
    boots = np.empty(n_boot, dtype=np.float32)
    centred = diff - diff.mean()
    for b in range(n_boot):
        idx = rs.integers(0, n, size=n)
        boots[b] = centred[idx].mean()
    obs = float(diff.mean())
    p = 2.0 * min(float((boots >= abs(obs)).mean()),
                  float((boots <= -abs(obs)).mean()))
    return float(min(p, 1.0)), obs


# %%
H_prec = per_image_precision(H_stacks, saliency, COMP_LAYER, TOPK,
                              method_mode=SIGNAL_MODE["entropy"],
                              saliency_mode=SIGNAL_MODE["saliency"])
L2_prec = per_image_precision(L2_stacks, saliency, COMP_LAYER, TOPK,
                               method_mode=SIGNAL_MODE["L2 norm"],
                               saliency_mode=SIGNAL_MODE["saliency"])
ATT_prec = per_image_precision(ATT_stacks, saliency, COMP_LAYER, TOPK,
                                method_mode=SIGNAL_MODE["attn entropy"],
                                saliency_mode=SIGNAL_MODE["saliency"])
RND_prec = per_image_precision(RND_stacks, saliency, COMP_LAYER, TOPK,
                                method_mode=SIGNAL_MODE["random-init H"],
                                saliency_mode=SIGNAL_MODE["saliency"])

records = []
for name, prec in (("L2 norm", L2_prec), ("attn entropy", ATT_prec),
                   ("random-init H", RND_prec)):
    diff = H_prec - prec
    p, obs_diff = paired_bootstrap_p(diff)
    records.append({
        "baseline": name,
        "layer": COMP_LAYER + 1,                              # human (1-indexed)
        "topK": TOPK,
        "n_images": N_IMAGES,
        "entropy_precision_mean": float(H_prec.mean()),
        "baseline_precision_mean": float(prec.mean()),
        "diff_mean": obs_diff,
        "diff_median": float(np.median(diff)),
        "diff_q25": float(np.percentile(diff, 25)),
        "diff_q75": float(np.percentile(diff, 75)),
        "paired_bootstrap_p": p,
        "n_boot": N_BOOT,
    })

# Per-layer scan as supplementary signal so we can report whether L=COMP_LAYER
# is representative.
layer_records = []
for layer in range(N_LAYERS):
    Hp = per_image_precision(H_stacks, saliency, layer, TOPK,
                              method_mode=SIGNAL_MODE["entropy"],
                              saliency_mode=SIGNAL_MODE["saliency"])
    for name, stacks, mode in (
        ("L2 norm", L2_stacks, SIGNAL_MODE["L2 norm"]),
        ("attn entropy", ATT_stacks, SIGNAL_MODE["attn entropy"]),
        ("random-init H", RND_stacks, SIGNAL_MODE["random-init H"]),
    ):
        bp = per_image_precision(stacks, saliency, layer, TOPK,
                                  method_mode=mode,
                                  saliency_mode=SIGNAL_MODE["saliency"])
        diff = Hp - bp
        p, obs_diff = paired_bootstrap_p(diff)
        layer_records.append({
            "baseline": name, "layer": layer + 1,
            "entropy_prec_mean": float(Hp.mean()),
            "baseline_prec_mean": float(bp.mean()),
            "diff_mean": obs_diff,
            "paired_bootstrap_p": p,
        })

primary = pd.DataFrame(records)
per_layer = pd.DataFrame(layer_records)
primary.to_csv(RESULTS / "outlier_localisation.csv", index=False)
per_layer.to_csv(RESULTS / "outlier_localisation_per_layer.csv", index=False)
print(f"saved {RESULTS / 'outlier_localisation.csv'}")
print(primary.to_string(index=False))

# %% [markdown]
# ## Per-image side-by-side comparison at L=COMP_LAYER

# %%
sample_rng = np.random.default_rng(SEED)
sample_idx = np.sort(sample_rng.choice(N_IMAGES, size=6, replace=False))
print("sampled image positions:", sample_idx)


def overlay_topk_marker(ax, field_2d, k=TOPK, color="tab:red", mode="high"):
    idx_set = topk_indices(field_2d, k=k, mode=mode)
    idx = np.fromiter(idx_set, dtype=np.int64)
    ys, xs = np.unravel_index(idx, field_2d.shape)
    ax.scatter(xs, ys, s=14, facecolor="none", edgecolor=color, linewidth=1.0)


fig, axes = plt.subplots(6, 5, figsize=(18, 22))
col_titles = [
    "input",
    f"entropy @ L={COMP_LAYER + 1}",
    f"L2 @ L={COMP_LAYER + 1}",
    f"attn entropy @ L={COMP_LAYER + 1}",
    f"random-init H @ L={COMP_LAYER + 1}",
]
for r, idx in enumerate(sample_idx):
    img_pil = images[idx].convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    sal_pil = saliency[idx]
    sal_top = topk_indices(sal_pil, TOPK, mode=SIGNAL_MODE["saliency"])
    sal_ys, sal_xs = np.unravel_index(np.array(list(sal_top)), (GRID, GRID))

    axes[r, 0].imshow(img_pil)
    axes[r, 0].scatter(
        np.array(sal_xs) * (224 / GRID) + (224 / GRID) / 2,
        np.array(sal_ys) * (224 / GRID) + (224 / GRID) / 2,
        s=70, facecolor="none", edgecolor="lime", linewidth=1.4,
        label=f"saliency top-{TOPK}",
    )
    axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    axes[r, 0].set_ylabel(f"img #{idx}", fontsize=10)
    if r == 0:
        axes[r, 0].set_title(col_titles[0], fontsize=11)
        axes[r, 0].legend(fontsize=8, loc="lower right")

    for c, (label, stacks, mode) in enumerate([
        ("entropy", H_stacks, SIGNAL_MODE["entropy"]),
        ("L2", L2_stacks, SIGNAL_MODE["L2 norm"]),
        ("attn", ATT_stacks, SIGNAL_MODE["attn entropy"]),
        ("rndH", RND_stacks, SIGNAL_MODE["random-init H"]),
    ]):
        ax = axes[r, c + 1]
        f = stacks[idx, COMP_LAYER]
        ax.imshow(f, cmap="magma")
        overlay_topk_marker(ax, f, k=TOPK, color="cyan", mode=mode)
        # Overlay saliency markers in green for direct comparison.
        ax.scatter(sal_xs, sal_ys, s=20, facecolor="none", edgecolor="lime", linewidth=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col_titles[c + 1], fontsize=11)

fig.suptitle(
    f"6 sampled images — L={COMP_LAYER + 1}.  cyan = method top-{TOPK},  lime = saliency top-{TOPK}",
    y=1.0,
)
fig.tight_layout()
fig.savefig(RESULTS / "comparison_per_image.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'comparison_per_image.png'}")

# %% [markdown]
# ## Stop-condition status
#
# Stop if (a) L2-norm paired-bootstrap p > 0.05 AND
# (b) phase_curves are visually-indistinguishable.

# %%
l2_row = primary[primary.baseline == "L2 norm"].iloc[0]
print("\n=== Stop-condition check ===")
print(f"L2 paired-bootstrap p:           {l2_row.paired_bootstrap_p:.4g}")
print(f"entropy precision (mean):         {l2_row.entropy_precision_mean:.4f}")
print(f"L2 precision (mean):              {l2_row.baseline_precision_mean:.4f}")
print(f"diff (entropy − L2):              {l2_row.diff_mean:+.4f}")
stop_p = l2_row.paired_bootstrap_p > 0.05
print(f"\np > 0.05 (L2 indistinguishable on outlier-localisation): {stop_p}")
print("Visual indistinguishability: see results/baselines/phase_curves.png")
print("Per-layer p-value scan: results/baselines/outlier_localisation_per_layer.csv")
