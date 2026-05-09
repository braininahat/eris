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
# # prepare_images — generate / fetch the 8 stimuli
#
# All saved under `../images/` as 224×224 RGB JPEGs (the ViT-B/16 input
# resolution; saves processor work).
#
# Half are programmatic (full reproducibility, known structure) so we can
# diagnose the pipeline on stimuli with ground-truth structure. Half are
# real photographs from Wikipedia Commons (stable URLs, public domain or
# CC-licensed) for natural-image variety.
#
# Each image documented inline so `summary.md` can cite provenance.

# %%
import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# Resolve repo root robustly: notebooks live one level under the repo root.
# When run via nbconvert __file__ is unavailable, so use cwd as fallback.
try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
OUT_DIR = REPO / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES = 224
RNG = np.random.default_rng(0)


def _save(arr_uint8: np.ndarray, name: str) -> None:
    img = Image.fromarray(arr_uint8, mode="RGB")
    out = OUT_DIR / f"{name}.jpg"
    img.save(out, quality=92)
    print(f"  wrote {out.relative_to(OUT_DIR.parent)}  ({out.stat().st_size//1024} KB)")


# %% [markdown]
# ## Synthetic stimuli (programmatic, fully reproducible)

# %%
# 1. gabor_anomaly — grid of Gabor patches, all oriented +45°, except one
#    in the middle-left flipped to -45°. Tests anomaly detection on a
#    repetitive texture.
def make_gabor_anomaly() -> np.ndarray:
    yy, xx = np.mgrid[:RES, :RES].astype(np.float32) / RES
    img = np.zeros((RES, RES), dtype=np.float32)
    grid = 8
    cell = RES // grid
    for gy in range(grid):
        for gx in range(grid):
            # local coordinates within cell, centred
            ly = (np.arange(cell) - cell / 2)[:, None]
            lx = (np.arange(cell) - cell / 2)[None, :]
            theta = np.deg2rad(-45) if (gy == 3 and gx == 2) else np.deg2rad(45)
            sigma = cell / 4
            wavelength = cell / 2
            xt = lx * np.cos(theta) + ly * np.sin(theta)
            yt = -lx * np.sin(theta) + ly * np.cos(theta)
            patch = np.exp(-(xt**2 + yt**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xt / wavelength)
            img[gy*cell:(gy+1)*cell, gx*cell:(gx+1)*cell] = patch
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)

_save(make_gabor_anomaly(), "synth_gabor")

# %%
# 2. checker_defect — 14×14 checker with one tile inverted. Same grid as
#    ViT-B/16 patches, so the defect lands cleanly on a single patch.
def make_checker_defect() -> np.ndarray:
    cell = RES // 14
    base = (np.indices((14, 14)).sum(axis=0) % 2)  # 0/1 checker
    base[7, 4] = 1 - base[7, 4]                    # flip one tile
    big = (np.kron(base, np.ones((cell, cell), dtype=np.uint8)) * 255).astype(np.uint8)
    return np.stack([big, big, big], axis=-1)

_save(make_checker_defect(), "synth_checker")

# %%
# 3. gradient_diag — smooth diagonal gradient. Should produce a smooth,
#    boring entropy field. Boundary control.
def make_gradient_diag() -> np.ndarray:
    yy, xx = np.mgrid[:RES, :RES].astype(np.float32)
    g = (yy + xx) / (RES + RES) * 255
    g = g.astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

_save(make_gradient_diag(), "synth_gradient")

# %%
# 4. texture_split — half white-noise, half smooth grey. Sharp vertical
#    boundary down the middle. Should produce a strong shear line in
#    the entropy field.
def make_texture_split() -> np.ndarray:
    img = np.full((RES, RES), 128, dtype=np.uint8)
    img[:, RES // 2:] = (RNG.integers(0, 256, size=(RES, RES // 2))).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)

_save(make_texture_split(), "synth_texture_split")

# %% [markdown]
# ## Real-photo stimuli (torchvision Imagenette)
#
# `torchvision.datasets.Imagenette` ungated; 10-class ImageNet subset at
# 160 px (small + fast download, ~100 MB). Pick four classes spanning scene
# categories, centre-crop, resize to 224.

# %%
from torchvision.datasets import Imagenette

DATA_ROOT = REPO / "data" / "imagenette"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ds = Imagenette(root=str(DATA_ROOT), size="160px", split="val", download=True)
print(f"imagenette val: {len(ds)} samples")
print(f"classes: {ds.classes}")

# imagenette classes (in label-id order):
# 0 tench (fish), 1 English springer (dog), 2 cassette player,
# 3 chain saw, 4 church, 5 French horn, 6 garbage truck, 7 gas pump,
# 8 golf ball, 9 parachute
CLASS_FOR = {
    "single_object": 0,   # tench: single fish on plain bg
    "cluttered": 2,       # cassette player: busy controls / labels
    "occlusion": 1,       # springer: dog often partially in grass
    "symmetric": 5,       # French horn: mirror-symmetric circular framing
}

# Build a label→[indices] map for fast first-match.
by_label: dict[int, int] = {}
for i, (_, lbl) in enumerate(ds):
    by_label.setdefault(lbl, i)
    if len(by_label) == 10:
        break

for name, cls_idx in CLASS_FOR.items():
    if cls_idx not in by_label:
        print(f"  no example for class {cls_idx} ({ds.classes[cls_idx]}); skipping")
        continue
    pil, _ = ds[by_label[cls_idx]]
    pil = pil.convert("RGB")
    w, h = pil.size
    s = min(w, h)
    pil = pil.crop(((w - s) // 2, (h - s) // 2,
                   (w - s) // 2 + s, (h - s) // 2 + s))
    pil = pil.resize((RES, RES), Image.Resampling.LANCZOS)
    out = OUT_DIR / f"{name}.jpg"
    pil.save(out, quality=92)
    print(f"  {name}  ←  imagenette class {cls_idx}={ds.classes[cls_idx]}  "
          f"({out.stat().st_size//1024} KB)")

# %%
print("\nfinal contents of images/:")
for p in sorted(OUT_DIR.glob("*.jpg")):
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
