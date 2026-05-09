# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # synth_check — Vasicek differential-entropy sanity check at d=768
#
# We're going to compute per-patch differential entropy on standardized
# 768-dim ViT residual-stream vectors. That's `n=768` samples per patch
# fed to `scipy.stats.differential_entropy` (Vasicek estimator default).
#
# `n=768` sits below the conventional ~1k stability floor for Vasicek,
# so we need to know how much noise that implies before we trust any
# downstream visualisation.
#
# Here: 1000 synthetic patches, each 768 N(0, 1) samples standardised
# in-place. Expect mean entropy ≈ 1.42 nats (the analytic value for a
# standard normal). **Gate: mean within ±0.10 of 1.42 and per-patch
# σ < 0.10.** If either fails, escalate before running the real pipeline.

# %%
import numpy as np
import scipy.stats

RNG = np.random.default_rng(0)
N_PATCHES = 1000
N_SAMPLES = 768   # = ViT-B/16 hidden_size

ANALYTIC_GAUSSIAN_ENTROPY = 0.5 * np.log(2 * np.pi * np.e)
print(f"analytic standard-normal entropy: {ANALYTIC_GAUSSIAN_ENTROPY:.6f} nats")

# %%
samples = RNG.standard_normal(size=(N_PATCHES, N_SAMPLES))
# Standardise per patch — same as the real pipeline.
samples = (samples - samples.mean(axis=1, keepdims=True)) / samples.std(axis=1, keepdims=True)

H = np.array([scipy.stats.differential_entropy(s) for s in samples])
print(f"n_patches: {N_PATCHES}   n_samples_per_patch: {N_SAMPLES}")
print(f"H mean : {H.mean():.4f} nats")
print(f"H std  : {H.std():.4f}")
print(f"H 5/50/95-pctile: {np.percentile(H, [5, 50, 95]).round(4).tolist()}")

# %%
# Gate
mean_err = abs(H.mean() - ANALYTIC_GAUSSIAN_ENTROPY)
ok_mean = mean_err < 0.10
ok_std = H.std() < 0.10
print(f"\nmean error vs analytic: {mean_err:.4f} (≤0.10? {ok_mean})")
print(f"per-patch σ: {H.std():.4f} (<0.10? {ok_std})")

if ok_mean and ok_std:
    print("\nGATE PASSED — proceed to viz_one.ipynb.")
else:
    raise RuntimeError(
        f"Vasicek synth check FAILED at n={N_SAMPLES}: "
        f"mean={H.mean():.4f} (analytic {ANALYTIC_GAUSSIAN_ENTROPY:.4f}), "
        f"σ={H.std():.4f}. Per-patch-as-sample formulation is unstable."
    )
