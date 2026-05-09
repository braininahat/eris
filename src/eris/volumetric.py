"""Volumetric ops for ``H_volume`` of shape ``(t, gy, gx)``.

Composes ``eris.fields.gradient_3d`` / ``laplacian_3d`` with two
domain-specific operations:

- ``extract_outlier_tubes`` — top-percentile of ``|∇H|`` voxels →
  3-D connected-component labelling → "tube" set.
- ``render_streamtubes_html`` — interactive plotly Streamtube of
  ``(Hx, Hy, Ht)`` with cubic spatial upsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


@dataclass(frozen=True)
class TubeExtraction:
    """Result of :func:`extract_outlier_tubes`."""
    threshold: float
    n_voxels: int                # voxels above threshold
    labels: np.ndarray           # int32, same shape as the volume
    n_components: int
    largest_label: int
    largest_size: int            # n voxels in the largest component
    largest_mask: np.ndarray     # bool, same shape


def extract_outlier_tubes(
    grad_magnitude: np.ndarray,
    top_pct: float = 5.0,
    connectivity: int = 1,
) -> TubeExtraction:
    """Threshold + connected-component label a 3-D gradient-magnitude volume.

    Args:
        grad_magnitude: ``(t, gy, gx)`` non-negative ``|∇H|`` field.
        top_pct: keep the top ``top_pct`` % of voxels by magnitude.
        connectivity: ``scipy.ndimage.generate_binary_structure(3, c)``;
            1 = 6-connected, 2 = 18-connected, 3 = 26-connected.

    Returns:
        :class:`TubeExtraction`.
    """
    if grad_magnitude.ndim != 3:
        raise ValueError(f"grad_magnitude must be (t, gy, gx); got {grad_magnitude.shape}")
    flat = grad_magnitude.ravel()
    threshold = float(np.percentile(flat, 100.0 - top_pct))
    binary = grad_magnitude >= threshold
    n_voxels = int(binary.sum())
    struct = ndi.generate_binary_structure(3, connectivity)
    labels, n_components = ndi.label(binary, structure=struct)
    if n_components == 0:
        return TubeExtraction(
            threshold=threshold, n_voxels=n_voxels,
            labels=labels.astype(np.int32),
            n_components=0, largest_label=0,
            largest_size=0,
            largest_mask=np.zeros_like(binary),
        )
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0   # ignore background
    largest_label = int(np.argmax(sizes))
    largest_size = int(sizes[largest_label])
    largest_mask = labels == largest_label
    return TubeExtraction(
        threshold=threshold,
        n_voxels=n_voxels,
        labels=labels.astype(np.int32),
        n_components=int(n_components),
        largest_label=largest_label,
        largest_size=largest_size,
        largest_mask=largest_mask,
    )


def iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Voxel IoU of two 3-D boolean masks of the same shape."""
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def upsample_volume_xy(
    volume: np.ndarray, factor: int | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Cubic spatial upsampling of a ``(t, gy, gx)`` volume, time untouched.

    Either ``factor`` (uniform multiplier on both spatial axes) or
    ``target_shape`` ``(gy_out, gx_out)`` (exact output grid size). Used
    only for prettier streamtube rendering; numerics stay on the raw
    grid.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be (t, gy, gx); got {volume.shape}")
    if (factor is None) == (target_shape is None):
        raise ValueError("specify exactly one of factor or target_shape")
    if factor is not None:
        if factor <= 1:
            return volume
        return ndi.zoom(volume, (1, factor, factor), order=3)
    gy_out, gx_out = target_shape  # type: ignore[misc]
    _, gy_in, gx_in = volume.shape
    if (gy_in, gx_in) == (gy_out, gx_out):
        return volume
    zy = float(gy_out) / float(gy_in)
    zx = float(gx_out) / float(gx_in)
    return ndi.zoom(volume, (1.0, zy, zx), order=3)


def render_streamtubes_html(
    Hx: np.ndarray,
    Hy: np.ndarray,
    Ht: np.ndarray,
    out_path: Path,
    title: str = "",
    starts: int = 4,
    upsample_xy: int | None = 4,
    target_shape: tuple[int, int] | None = None,
    sizeref: float = 0.5,
    cmin: float | None = None,
    cmax: float | None = None,
) -> Path:
    """Write a 3-D ``plotly.graph_objects.Streamtube`` to ``out_path``.

    Coords: ``(x=gx, y=gy, t)`` so the temporal axis is ``z``. The seed
    points form a coarse ``starts × starts × starts`` grid covering the
    full volume — keeps the resulting HTML < ~3 MB.

    Args:
        Hx, Hy, Ht: gradient components, each ``(t, gy, gx)``. Numerics
            stay on the input grid; we cubic-upsample spatially before
            rendering for smoother streamlines.
        out_path: target ``.html`` file.
        title: figure title.
        starts: number of seed points per axis (``starts**3`` total).
        upsample_xy: spatial upsampling factor for the rendered field.
        sizeref: streamtube radius, plotly units.
    """
    import plotly.graph_objects as go  # type: ignore

    if Hx.shape != Hy.shape or Hx.shape != Ht.shape:
        raise ValueError("Hx, Hy, Ht must share shape")
    if target_shape is not None:
        Hx_u = upsample_volume_xy(Hx, target_shape=target_shape)
        Hy_u = upsample_volume_xy(Hy, target_shape=target_shape)
        Ht_u = upsample_volume_xy(Ht, target_shape=target_shape)
    else:
        Hx_u = upsample_volume_xy(Hx, factor=upsample_xy)
        Hy_u = upsample_volume_xy(Hy, factor=upsample_xy)
        Ht_u = upsample_volume_xy(Ht, factor=upsample_xy)
    T, gy, gx = Hx_u.shape

    # Coordinate grids (note plotly indexing with Streamtube wants 1-D
    # axis arrays + a 4-D regular grid implied by their lengths).
    x_axis = np.arange(gx)
    y_axis = np.arange(gy)
    t_axis = np.arange(T)
    XX, YY, TT = np.meshgrid(x_axis, y_axis, t_axis, indexing="ij")
    # Streamtube expects flattened arrays in the same order as meshgrid.
    # Build u(x,y,t) so axis order is (x_axis, y_axis, t_axis).
    # Hx/Hy/Ht have axis order (t, gy, gx) so transpose to (gx, gy, t).
    u = np.transpose(Hx_u, (2, 1, 0))   # x-component
    v = np.transpose(Hy_u, (2, 1, 0))   # y-component
    w = np.transpose(Ht_u, (2, 1, 0))   # t-component

    # Seed grid — 4×4×4 by default. Place inside the volume slightly off
    # the boundary so streamlines have room to walk.
    pad = 0.10
    sx = np.linspace(pad * gx, (1 - pad) * gx, starts)
    sy = np.linspace(pad * gy, (1 - pad) * gy, starts)
    st_ = np.linspace(pad * T, (1 - pad) * T, starts)
    SX, SY, ST = np.meshgrid(sx, sy, st_, indexing="ij")

    fig = go.Figure(
        data=go.Streamtube(
            x=XX.ravel(),
            y=YY.ravel(),
            z=TT.ravel(),
            u=u.ravel(),
            v=v.ravel(),
            w=w.ravel(),
            starts=dict(x=SX.ravel(), y=SY.ravel(), z=ST.ravel()),
            sizeref=sizeref,
            colorscale="Viridis",
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(title="|∇H|"),
            maxdisplayed=1200,
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="gx",
            yaxis_title="gy",
            zaxis_title="t",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path
