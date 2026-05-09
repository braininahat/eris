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
# # Step D — estimator robustness on ViT-B/16
#
# Question: is the L4→L5 ΔH-std spike + outlier-patch pattern an
# **estimator artefact** or a property of the residual-stream geometry?
#
# Same 200 tiny-imagenet validation images as Step A
# (`results/cross_arch/vit_b16/image_indices.npy`). We re-run
# `extract_entropy_stack("vit_b16", ...)` swapping the per-patch
# differential-entropy estimator:
#
# 1. **Vasicek** (default scipy `differential_entropy`) — already cached
#    in `results/cross_arch/vit_b16/H_stack.npz`; reused, not recomputed.
# 2. **k-NN** (Kozachenko–Leonenko, `k=3`).
# 3. **KDE** (Gaussian kernel, Silverman bandwidth).
#
# All three were validated on N(0, 1) by `notebooks/synth_check.ipynb`
# (analytic 1.4189 nats; gate ±0.10).
#
# ## Stop condition
# If the ΔH-std spike at L=5 (the headline) **disappears** under k-NN or
# KDE — i.e. the per-image transition layer median diverges by more than
# ±1 layer from Vasicek's median (5) — the L4→L5 finding is an estimator
# artefact and the public claim is pulled.

# %%
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from eris.estimators import kde_entropy, knn_entropy
from eris.extract import extract_entropy_stack
from eris.fields import per_layer_stats, transition_layer

# %%
try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
RESULTS = REPO / "results" / "estimator_robustness"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE_VASICEK = REPO / "results" / "cross_arch" / "vit_b16"

ARCH = "vit_b16"
N_IMAGES = 200
N_LAYERS = 12
GRID = 14
SEED = 0
DEVICE = "cuda"
ESTIMATORS = {
    "vasicek": None,                # cached — see CACHE_VASICEK
    "knn": knn_entropy,
    "kde": kde_entropy,
}
print(f"REPO    = {REPO}")
print(f"RESULTS = {RESULTS}")
print(f"CACHE   = {CACHE_VASICEK}")

# %% [markdown]
# ## 1. Reload the **same** 200 images as Step A
#
# `image_indices.npy` is the source of truth — same draw, same arch, same
# pre-processing as the cached Vasicek H_stack. We sanity-check the
# re-derived index list against `np.load(CACHE_VASICEK / "image_indices.npy")`
# before continuing.

# %%
indices = np.load(CACHE_VASICEK / "image_indices.npy")
assert indices.shape == (N_IMAGES,), f"unexpected shape {indices.shape}"
print(f"loaded {indices.shape[0]} cached image indices, first 5 = {indices[:5].tolist()}")

# Independently re-derive: rng.choice + sort with the same SEED used in
# Step A. This guards against silent index drift.
ds = load_dataset("Maysee/tiny-imagenet", split="valid")
rng = np.random.default_rng(SEED)
re_idx = np.sort(rng.choice(len(ds), size=N_IMAGES, replace=False))
assert np.array_equal(re_idx, indices), \
    "re-derived indices do not match cached image_indices.npy — Step A drift"
print("indices match Step A's deterministic draw  OK")

images = [ds[int(i)]["image"].convert("RGB") for i in indices]
print(f"loaded {len(images)} PIL images")

# %% [markdown]
# ## 2. Compute (or load) per-estimator H_stack
#
# Vasicek is reused from Step A. k-NN and KDE are computed once and
# cached in `results/estimator_robustness/`. Wall-clock benchmarks on
# this hardware (RTX-class GPU, n=768 samples per patch):
#
# | estimator | per-patch | per-image (12L × 196p) |  full 200 imgs |
# |-----------|----------:|------------------------:|---------------:|
# | vasicek   | ~0.16 ms  | ~0.4 s                  | ~1.6 min       |
# | knn k=3   | ~0.47 ms  | ~1.1 s                  | ~4 min         |
# | kde       | ~4.2 ms   | ~10 s                   | ~33 min        |
#
# KDE dominates (Gaussian-KDE log-pdf evaluation is `O(n²)` in samples).
# We accept the cost: this is a one-time robustness check and the
# alternative would be sub-sampling, which destroys the comparison.

# %%
def load_or_compute(name: str, fn) -> np.ndarray:
    """Return ``(N_IMAGES, N_LAYERS, GRID, GRID)`` H_stack for ``name``.

    Vasicek pulls from Step A's cache. k-NN and KDE are computed via
    ``extract_entropy_stack`` with the matching ``estimator`` kwarg and
    cached under ``results/estimator_robustness/``.
    """
    if name == "vasicek":
        path = CACHE_VASICEK / "H_stack.npz"
        z = np.load(path)
        H = z["H"]
        print(f"[vasicek] reused {path.relative_to(REPO)}  shape={H.shape}")
        return H

    path = RESULTS / f"vit_b16_{name}_H_stack.npz"
    if path.exists():
        z = np.load(path)
        H = z["H"]
        print(f"[{name}] cached  {path.relative_to(REPO)}  shape={H.shape}")
        return H

    print(f"[{name}] computing extract_entropy_stack(estimator={fn.__name__})…")
    t0 = time.time()
    H, _spec = extract_entropy_stack(
        ARCH, images, device=DEVICE, estimator=fn,
        progress=lambda i, n: print(f"  {name}[{i}/{n}]  ({time.time()-t0:.1f}s)", end="\r")
        if i in (1, 10, 25, 50, 75, 100, 125, 150, 175, 200) else None,
    )
    dt = time.time() - t0
    np.savez_compressed(path, H=H, indices=indices)
    print(f"\n[{name}] done in {dt:.1f}s ({dt/60:.1f} min) → {path.relative_to(REPO)}  shape={H.shape}")
    return H


H_per_est: dict[str, np.ndarray] = {}
for est_name, est_fn in ESTIMATORS.items():
    H_per_est[est_name] = load_or_compute(est_name, est_fn)
    assert H_per_est[est_name].shape == (N_IMAGES, N_LAYERS, GRID, GRID), \
        f"shape mismatch for {est_name}: {H_per_est[est_name].shape}"

# %% [markdown]
# ## 3. Per-estimator headline statistics
#
# For each estimator and each image we compute:
# - the per-layer summary `(H_mean, H_std, abs_grad_H_mean, dH_std)` via
#   `eris.fields.per_layer_stats`;
# - the per-image transition layer via `eris.fields.transition_layer`
#   (argmax of within-layer ΔH-std).
#
# The headline statistic for the L4→L5 finding is **ΔH-std**: a peak at
# `L=5` says the residual stream undergoes a sharp non-Gaussianisation
# step there. Median per-image transition layer is the headline scalar.

# %%
def aggregate(stacks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image per-layer-stats stacked → (n_imgs, L) per key."""
    keys = ("H_mean", "H_std", "abs_grad_H_mean", "dH_std")
    accum = {k: [] for k in keys}
    for i in range(stacks.shape[0]):
        s = per_layer_stats(stacks[i])
        for k in keys:
            accum[k].append(s[k])
    return {k: np.stack(accum[k], axis=0) for k in keys}


curves_per_est: dict[str, dict[str, np.ndarray]] = {}
L_trans_per_est: dict[str, np.ndarray] = {}
for name, H in H_per_est.items():
    curves_per_est[name] = aggregate(H)
    L_trans_per_est[name] = np.array(
        [transition_layer(H[i]) for i in range(H.shape[0])], dtype=np.int32
    )
    med = float(np.median(L_trans_per_est[name]))
    iqr = float(
        np.quantile(L_trans_per_est[name], 0.75)
        - np.quantile(L_trans_per_est[name], 0.25)
    )
    print(f"[{name:8s}]  median L_trans = {med:.1f}  IQR = {iqr:.1f}  "
          f"min/max = {L_trans_per_est[name].min()}/{L_trans_per_est[name].max()}")

# %% [markdown]
# ## 4. Phase-curves figure — overlay 3 estimators on 4 panels
#
# `(H_mean | H_std | mean|∇H| | ΔH-std)` vs depth, one curve per
# estimator. Per-estimator z-scoring across `L` is required because the
# three estimators sit on different absolute scales (Vasicek log-spacing
# bias, K-L digamma offset, KDE leave-one-out underestimation of variance).
# What matters for the L4→L5 claim is the *shape* of the curve, not the
# offset.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
panel_keys = [
    ("H_mean", "H mean"),
    ("H_std", "H std (within-layer spread)"),
    ("abs_grad_H_mean", "mean |∇H| per layer"),
    ("dH_std", "ΔH-std (cross-layer change)"),
]
colors = {"vasicek": "tab:orange", "knn": "tab:blue", "kde": "tab:green"}
layers = np.arange(1, N_LAYERS + 1)

# z-norm each (estimator, panel) curve across L using the mean curve.
for ax, (key, label) in zip(axes, panel_keys):
    for est_name in ESTIMATORS:
        per_img = curves_per_est[est_name][key]              # (n_imgs, L)
        mean_curve = per_img.mean(axis=0)
        sd = mean_curve.std() + 1e-9
        z = (per_img - mean_curve.mean()) / sd

        # bootstrap mean band across images
        rs = np.random.default_rng(SEED)
        n = z.shape[0]
        boot = np.empty((400, N_LAYERS), dtype=np.float32)
        for b in range(400):
            idx = rs.integers(0, n, size=n)
            boot[b] = z[idx].mean(axis=0)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        mid = z.mean(axis=0)
        ax.fill_between(layers, lo, hi, color=colors[est_name], alpha=0.20)
        ax.plot(layers, mid, marker="o", color=colors[est_name], lw=1.8,
                label=est_name)
    ax.axvline(5.0, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(5.05, ax.get_ylim()[1], " L=5", color="grey", fontsize=8, va="top")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("layer")
    ax.set_xticks(layers)
    ax.grid(alpha=0.30)
axes[0].set_ylabel("z-scored across L  (per estimator independently)")
axes[-1].legend(fontsize=10, loc="upper left")
fig.suptitle(
    "Step D — phase curves under 3 differential-entropy estimators "
    "(ViT-B/16, 200 tiny-imagenet val; bands = 95 % bootstrap CI across images, "
    "z-scored per estimator)",
    y=1.04,
)
fig.tight_layout()
fig.savefig(RESULTS / "phase_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'phase_curves.png'}")

# %% [markdown]
# ## 5. Per-image field-stack comparison at one mid-post-transition layer
#
# Pick **one** image from the 200 — `synth_gabor` is not in the
# tiny-imagenet draw, so we just use the first cached image. Show its
# entropy field at `L=6` (1-indexed; 0-indexed `L=5`, one block past the
# Vasicek transition). All three estimators on the same image should
# light up the **same** outlier patches if the pattern is real and not
# an artefact.

# %%
SHOW_IMG = 0                            # index into our 200-image list
SHOW_L = 5                              # 0-indexed → 'L=6' = block 6
img_pil = images[SHOW_IMG]
img_pil_resized = img_pil.resize((224, 224))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
axes[0].imshow(img_pil_resized)
axes[0].set_title(f"input  (img idx={int(indices[SHOW_IMG])})", fontsize=11)
axes[0].set_xticks([]); axes[0].set_yticks([])

# Use a shared color scale per estimator (each rescaled internally) but a
# common "diverging from per-image median" mapping so the eye compares
# *patterns*, not absolute values.
for col, est_name in enumerate(ESTIMATORS, start=1):
    H_field = H_per_est[est_name][SHOW_IMG, SHOW_L]
    devs = H_field - np.median(H_field)
    vmax = float(np.abs(devs).max() + 1e-9)
    im = axes[col].imshow(devs, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[col].set_title(f"{est_name}   (H − median(H))   L={SHOW_L+1}", fontsize=11)
    axes[col].set_xticks([]); axes[col].set_yticks([])
    fig.colorbar(im, ax=axes[col], shrink=0.85)

fig.suptitle(
    f"Field-stack comparison at L={SHOW_L+1} on a single ViT-B/16 image "
    f"— do all 3 estimators flag the **same** outlier patches?",
    y=1.04,
)
fig.tight_layout()
fig.savefig(RESULTS / "field_stack_compare.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'field_stack_compare.png'}")

# %% [markdown]
# ## 6. Transition-layer agreement — bar chart of median ± IQR
#
# A small bar per estimator, coloured to match the phase-curves plot. We
# also annotate `frac_within_1_of_vasicek_median` — fraction of per-image
# `L_trans` values within ±1 of Vasicek's median transition layer. This
# is the headline cross-estimator-agreement metric.

# %%
vasicek_med = float(np.median(L_trans_per_est["vasicek"]))

agreement_records = []
for est_name in ESTIMATORS:
    L = L_trans_per_est[est_name]
    med = float(np.median(L))
    q25 = float(np.quantile(L, 0.25))
    q75 = float(np.quantile(L, 0.75))
    iqr = q75 - q25
    frac_within = float(np.mean(np.abs(L - vasicek_med) <= 1))
    agreement_records.append({
        "estimator": est_name,
        "median_L_trans": med,
        "iqr": iqr,
        "q25_L_trans": q25,
        "q75_L_trans": q75,
        "min_L_trans": int(L.min()),
        "max_L_trans": int(L.max()),
        "frac_within_1_of_vasicek_median": frac_within,
        "n_images": int(L.shape[0]),
    })
agree_df = pd.DataFrame(agreement_records)
agree_df.to_csv(RESULTS / "agreement.csv", index=False)
print(f"saved {RESULTS / 'agreement.csv'}")
print(agree_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7.0, 4.2))
xs = np.arange(len(ESTIMATORS))
meds = [r["median_L_trans"] for r in agreement_records]
q25s = [r["q25_L_trans"] for r in agreement_records]
q75s = [r["q75_L_trans"] for r in agreement_records]
err_lo = np.array(meds) - np.array(q25s)
err_hi = np.array(q75s) - np.array(meds)
bar_colors = [colors[r["estimator"]] for r in agreement_records]
ax.bar(xs, meds, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.7,
       yerr=[err_lo, err_hi], capsize=8)
for x, r in zip(xs, agreement_records):
    ax.text(x, r["median_L_trans"] + 0.3,
            f"med={r['median_L_trans']:.1f}\nIQR={r['iqr']:.1f}\nwithin±1: "
            f"{r['frac_within_1_of_vasicek_median']*100:.0f}%",
            ha="center", va="bottom", fontsize=9)
ax.axhline(vasicek_med, color="grey", lw=0.7, ls="--", alpha=0.5)
ax.text(len(ESTIMATORS) - 0.5, vasicek_med + 0.05,
        f" vasicek median = {vasicek_med:.0f}", color="grey", fontsize=8, va="bottom")
ax.set_xticks(xs)
ax.set_xticklabels([r["estimator"] for r in agreement_records])
ax.set_ylabel("transition layer  (argmax ΔH-std)")
ax.set_ylim(0, N_LAYERS + 1.2)
ax.set_yticks(np.arange(0, N_LAYERS + 1, 2))
ax.set_title("Per-image transition layer  (median; bar = IQR via 25/75-percentile)",
             fontsize=11)
ax.grid(axis="y", alpha=0.30)
fig.tight_layout()
fig.savefig(RESULTS / "transition_layer_agreement.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'transition_layer_agreement.png'}")

# %% [markdown]
# ## 7. Stop-condition check
#
# **TRIGGERED** if the L4→L5 spike "disappears" under k-NN or KDE — i.e.
# their median transition layer differs from Vasicek's by more than
# **±1 layer**, or their phase curves no longer show a dominant peak
# (peak / median < 2 on the body of the ΔH-std curve).
#
# We also report the per-estimator agreement fraction
# (`frac_within_1_of_vasicek_median`). If all three sit within ±1 layer
# on the median, the finding is robust.

# %%
print("=" * 70)
print("Step D — estimator robustness report")
print("=" * 70)
print(f"images : {N_IMAGES}  ViT-B/16  Maysee/tiny-imagenet [valid]  seed={SEED}")
print()
print(f"{'estimator':<10s} {'median L_trans':>15s} {'IQR':>6s} {'min/max':>8s}"
      f"  {'within-1 of vasicek':>20s}")
print("-" * 70)

stop_triggered = False
fail_reasons: list[str] = []
for r in agreement_records:
    print(f"{r['estimator']:<10s} {r['median_L_trans']:>15.1f}"
          f" {r['iqr']:>6.1f} {r['min_L_trans']:>3d}/{r['max_L_trans']:<3d}"
          f"  {r['frac_within_1_of_vasicek_median']*100:>18.1f}%")

    # Median agreement
    diff = abs(r["median_L_trans"] - vasicek_med)
    if r["estimator"] != "vasicek" and diff > 1.0:
        stop_triggered = True
        fail_reasons.append(
            f"{r['estimator']}: median L_trans={r['median_L_trans']:.1f} "
            f"vs vasicek={vasicek_med:.1f}  (Δ={diff:.1f} > 1)"
        )

# Peak-strength check on the dH_std curve body (skip L=1 zero entry).
print()
print("ΔH-std peak / median on body  (>= 2 = strong, < 2 = no transition)")
for est_name in ESTIMATORS:
    body = curves_per_est[est_name]["dH_std"][:, 1:].mean(axis=0)
    peak = float(body.max())
    med_body = float(np.median(body) + 1e-12)
    ratio = peak / med_body
    flag = "OK" if ratio >= 2.0 else "FAIL"
    print(f"  {est_name:<10s}  peak={peak:.4f}  median(body)={med_body:.4f}"
          f"  peak/med={ratio:.2f}  {flag}")
    if est_name != "vasicek" and ratio < 2.0:
        stop_triggered = True
        fail_reasons.append(f"{est_name}: peak/med={ratio:.2f} < 2 (no dominant peak)")

print()
all_within = all(
    r["frac_within_1_of_vasicek_median"] >= 0.50 for r in agreement_records
)
all_med_agree = all(
    abs(r["median_L_trans"] - vasicek_med) <= 1.0 for r in agreement_records
)

print(f"all 3 estimators agree on median transition layer (±1 of vasicek): "
      f"{all_med_agree}")
print(f"all 3 estimators have ≥ 50 % per-image agreement within ±1 of vasicek "
      f"median: {all_within}")
print()
if stop_triggered:
    print("STOP CONDITION : TRIGGERED")
    for reason in fail_reasons:
        print(f"  - {reason}")
    print(">>> The L4→L5 finding may be an estimator artefact. Re-evaluate "
          "before pushing public claim.")
else:
    print("STOP CONDITION : NOT TRIGGERED")
    print(">>> All three estimators agree on the L4→L5 phase transition. "
          "Finding is robust to estimator choice.")

print()
print("Files:")
print(f"  {(RESULTS / 'phase_curves.png').relative_to(REPO)}")
print(f"  {(RESULTS / 'field_stack_compare.png').relative_to(REPO)}")
print(f"  {(RESULTS / 'transition_layer_agreement.png').relative_to(REPO)}")
print(f"  {(RESULTS / 'agreement.csv').relative_to(REPO)}")
print(f"  {(RESULTS / 'vit_b16_knn_H_stack.npz').relative_to(REPO)}")
print(f"  {(RESULTS / 'vit_b16_kde_H_stack.npz').relative_to(REPO)}")
