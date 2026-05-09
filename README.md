# eris

Per-patch differential entropy at each ViT-B/16 layer treated as a 2D
scalar field on the 14×14 patch grid, then visualised with the tools of
fluid dynamics: gradient, streamlines, Laplacian, inter-layer ΔH.

The question: do the per-layer entropy fields show Schlieren / wind-tunnel
structure (boundaries, wakes, recirculation, shear) across depth, or is
the field noise?

## Run

```bash
uv sync
uv run jupytext --sync notebooks/*.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/synth_check.ipynb
# Then for each image:
IMAGE_PATH=images/single_object.jpg \
  uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/viz_one.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/cross_image.ipynb
```

Outputs land under `results/`. See `results/summary.md` for the read-out.
