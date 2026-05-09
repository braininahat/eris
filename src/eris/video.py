"""Video stimuli — synthesis and loading.

Two factories:
- ``synthesise_translating_blob`` — a Gaussian blob translating across a
  noise background, used as the synthetic ground-truth stimulus for the
  volumetric tube-extraction check (Step B2).
- ``load_real_video`` — pulls a public clip (UCF101 or a Wikimedia URL),
  centre-crops + resizes to 224×224, returns ``(n_frames, 224, 224, 3)``.

Both return a uint8 array of shape ``(T, H, W, 3)`` plus a small dict of
metadata (source, fps, ground-truth centres for synthetic).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np


# ── synthetic ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyntheticBlobMeta:
    """Per-frame ground-truth for a translating blob video."""
    n_frames: int
    size: int
    sigma: float
    amplitude: float
    centres_xy: np.ndarray      # (n_frames, 2)  in pixel coords (cx, cy)


def synthesise_translating_blob(
    n_frames: int = 64,
    size: int = 224,
    sigma: float = 12.0,
    amplitude: float = 100.0,
    noise_std: float = 12.0,
    seed: int = 0,
) -> tuple[np.ndarray, SyntheticBlobMeta]:
    """A Gaussian blob translating left→right with slight vertical drift.

    Args:
        n_frames: number of frames (T).
        size: output side length in pixels (square frames).
        sigma: blob standard deviation in pixels.
        amplitude: peak blob brightness on top of the noise background
            (0..255 scale).
        noise_std: additive Gaussian noise standard deviation per frame.
        seed: rng seed.

    Returns:
        ``(video, meta)``:
          * ``video``: ``(T, size, size, 3)`` uint8.
          * ``meta``: ``SyntheticBlobMeta`` with per-frame blob centre
            coordinates ``(cx, cy)`` in pixels.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)

    # Trajectory: the blob crosses the inner 80% of the frame in x and
    # drifts ~20% in y so the patch-grid intersection is non-trivial.
    margin = int(0.10 * size)
    cx = np.linspace(margin, size - margin - 1, n_frames).astype(np.float32)
    cy = (
        size / 2.0
        + 0.10 * size * np.sin(np.linspace(0.0, np.pi, n_frames))   # one half-cycle
    ).astype(np.float32)

    frames = np.empty((n_frames, size, size, 3), dtype=np.uint8)
    for t in range(n_frames):
        bg = rng.normal(loc=128.0, scale=noise_std, size=(size, size)).astype(np.float32)
        blob = amplitude * np.exp(
            -((xx - cx[t]) ** 2 + (yy - cy[t]) ** 2) / (2.0 * sigma * sigma)
        )
        gray = np.clip(bg + blob, 0.0, 255.0).astype(np.uint8)
        frames[t, ..., 0] = gray
        frames[t, ..., 1] = gray
        frames[t, ..., 2] = gray
    meta = SyntheticBlobMeta(
        n_frames=n_frames,
        size=size,
        sigma=sigma,
        amplitude=amplitude,
        centres_xy=np.stack([cx, cy], axis=1),
    )
    return frames, meta


def blob_centres_to_patch(
    centres_xy: np.ndarray, image_size: int, grid: tuple[int, int]
) -> np.ndarray:
    """Map per-frame pixel centres to (cx_patch, cy_patch) on a (gy, gx) grid.

    Returns:
        ``(n_frames, 2)`` int — patch indices in (col, row) order
        ``(cx_patch, cy_patch)`` matching ``centres_xy = (cx_px, cy_px)``.
    """
    gy, gx = grid
    cx_patch = np.clip(centres_xy[:, 0] * gx / image_size, 0, gx - 1).astype(int)
    cy_patch = np.clip(centres_xy[:, 1] * gy / image_size, 0, gy - 1).astype(int)
    return np.stack([cx_patch, cy_patch], axis=1)


def synthetic_tube_mask(
    centres_xy: np.ndarray,
    image_size: int,
    grid: tuple[int, int],
    n_frames: int,
    dilate_patches: int = 1,
) -> np.ndarray:
    """Binary ground-truth tube mask of shape ``(t, gy, gx)``.

    A 1 is placed at the patch closest to each frame's blob centre, then
    the resulting volume is dilated by ``dilate_patches`` patches in
    (y, x) per frame so the comparison tolerates one-patch jitter.
    """
    import scipy.ndimage as ndi
    gy, gx = grid
    centres_patch = blob_centres_to_patch(centres_xy, image_size, grid)
    mask = np.zeros((n_frames, gy, gx), dtype=bool)
    for t in range(n_frames):
        cxp, cyp = centres_patch[t]
        mask[t, cyp, cxp] = True
    if dilate_patches > 0:
        # Dilate spatially per frame; do not connect through time so the
        # GT tube inherits whatever frame-to-frame continuity the blob has.
        struct = ndi.generate_binary_structure(2, 1)
        for t in range(n_frames):
            mask[t] = ndi.binary_dilation(
                mask[t], structure=struct, iterations=dilate_patches
            )
    return mask


# ── real video ─────────────────────────────────────────────────────────────────


REAL_VIDEO_URLS: tuple[tuple[str, str], ...] = (
    # Big Buck Bunny — Wikimedia Commons / Creative Commons
    (
        "big_buck_bunny_240p_1mb",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/"
        "Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.240p.vp9.webm",
    ),
    # Charge of the Light Brigade — small, public domain
    (
        "charge_of_the_light_brigade",
        "https://upload.wikimedia.org/wikipedia/commons/9/96/"
        "The_Charge_of_the_Light_Brigade_%281936%29.webm",
    ),
)


def _centre_crop_resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """``(T, H, W, 3)`` uint8 → ``(T, size, size, 3)`` uint8."""
    from PIL import Image
    T, H, W, C = arr.shape
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    out = np.empty((T, size, size, C), dtype=np.uint8)
    for t in range(T):
        crop = arr[t, y0 : y0 + s, x0 : x0 + s]
        im = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
        out[t] = np.asarray(im)
    return out


def load_real_video(
    out_path: Path,
    cache_dir: Path | None = None,
    n_frames: int = 64,
    size: int = 224,
    stride: int = 1,
    skip_first: int = 0,
) -> tuple[np.ndarray, dict]:
    """Fetch (or cache) a small public-domain MP4, centre-crop+resize.

    Tries each URL in ``REAL_VIDEO_URLS`` in order until one decodes.
    Caches the original webm/mp4 in ``cache_dir`` (default
    ``out_path.parent``) so subsequent runs avoid the network.

    Returns:
        ``(video, meta)``:
          * ``video``: ``(n_frames, size, size, 3)`` uint8.
          * ``meta``: ``{"source_name", "source_url", "fps", "n_total_frames"}``.
    """
    out_path = Path(out_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else out_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for name, url in REAL_VIDEO_URLS:
        suffix = Path(url).suffix or ".webm"
        cache_path = cache_dir / f"{name}{suffix}"
        try:
            if not cache_path.exists():
                _download(url, cache_path)
            frames = iio.imread(cache_path, plugin="pyav")
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"unexpected video shape {frames.shape}")
            # Take a slice of the first <stride * (n_frames+skip_first)> frames.
            tot = frames.shape[0]
            idx = np.arange(skip_first, skip_first + n_frames * stride, stride)
            idx = idx[idx < tot]
            if len(idx) < n_frames:
                # take from the middle if we ran out
                idx = np.arange(0, min(n_frames * stride, tot), stride)
                idx = idx[:n_frames]
            cropped = _centre_crop_resize(frames[idx], size=size)
            try:
                fps = iio.immeta(cache_path, plugin="pyav").get("fps", 25.0)
            except Exception:
                fps = 25.0
            meta = {
                "source_name": name,
                "source_url": url,
                "fps": float(fps),
                "n_total_frames": int(tot),
                "n_frames_used": int(cropped.shape[0]),
                "stride": int(stride),
                "skip_first": int(skip_first),
            }
            return cropped, meta
        except Exception as exc:                                   # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"could not load any real video; last error: {last_err}")


def load_ssv2_clips(
    n_clips: int = 5,
    n_frames: int = 64,
    size: int = 256,
    seed: int = 42,
    cache_path: Path | None = None,
    hf_dataset: str = "jxie/something_something_v2",
    buffer_size: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """Stream a small Something-Something-v2 sample, return ``(n, T, H, W, 3)``.

    SSv2 is the canonical motion-discrimination eval for video models;
    V-JEPA 2 is benchmarked on it in the original paper. Streaming
    avoids the 19.5 GB full download — we just decode the clips we
    need.

    Args:
        n_clips: number of clips to return.
        n_frames: contiguous frames per clip (skips clips that are
            shorter than this).
        size: output spatial size after centre-crop + resize.
        seed: shuffle seed for clip selection.
        cache_path: optional ``.npz`` cache (default off — caller can
            wrap in their own cache layer).
        hf_dataset: HF Hub dataset id; defaults to a parquet mirror that
            doesn't require the loading-script auth dance.
        buffer_size: shuffle buffer size for ``ds.shuffle``.

    Returns:
        ``(clips, metas)`` — ``clips`` of shape ``(n_clips, n_frames, size,
        size, 3)`` uint8; ``metas`` is a list of per-clip dicts with
        ``{"num_frames", "fps", "height", "width"}``.
    """
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["clips"], list(z["metas"])

    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    it = iter(ds)
    out_clips = []
    out_metas: list[dict] = []
    while len(out_clips) < n_clips:
        sample = next(it)
        decoder = sample["video"]                # torchcodec.VideoDecoder
        meta = decoder.metadata
        if meta.num_frames < n_frames:
            continue
        clip_t = decoder[:n_frames]              # (T, 3, H, W) uint8 torch
        arr = clip_t.permute(0, 2, 3, 1).cpu().numpy()    # (T, H, W, 3)
        # centre-crop to square then resize per frame
        T, H, W, _ = arr.shape
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        out = np.empty((T, size, size, 3), dtype=np.uint8)
        for t in range(T):
            crop = arr[t, y0 : y0 + s, x0 : x0 + s]
            out[t] = np.asarray(
                Image.fromarray(crop).resize((size, size), Image.BILINEAR)
            )
        out_clips.append(out)
        out_metas.append({
            "num_frames_total": int(meta.num_frames),
            "fps": float(meta.average_fps) if meta.average_fps else None,
            "height": int(meta.height),
            "width": int(meta.width),
            "n_frames_used": int(n_frames),
        })

    clips = np.stack(out_clips, axis=0)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, clips=clips,
                            metas=np.array(out_metas, dtype=object))
    return clips, out_metas


def _download(url: str, path: Path) -> None:
    """Tiny urllib download (no extra deps, fine for ~10 MB clips)."""
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "eris-research/0.1 (varunshi@buffalo.edu)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


# ── helpers ───────────────────────────────────────────────────────────────────


def write_video(path: Path, frames: np.ndarray, fps: int = 10) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames to ``path`` at ``fps``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")


def frames_to_pil(frames: np.ndarray) -> Iterable:
    """Yield each frame of ``(T, H, W, 3)`` as a PIL.Image."""
    from PIL import Image
    for t in range(frames.shape[0]):
        yield Image.fromarray(frames[t]).convert("RGB")
