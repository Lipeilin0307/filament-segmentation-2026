"""Attributed community crop builder; no dataset is read at import time."""
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu
from ..supplement import square_bounds
IMG_DIR=None
H=W=2048
CROP_SIZE=256
CONTEXT=1.8
MIN_CROP=96
def truncate_mask(m: np.ndarray, rng: random.Random) -> np.ndarray:
    """A random contiguous 40-90% stretch of the filament along its principal
    axis -- what a detector that only caught the dark core proposes."""
    ys, xs = np.nonzero(m)
    c = np.stack([xs, ys], 1).astype(np.float32)
    c -= c.mean(0)
    proj = c @ np.linalg.svd(c, full_matrices=False)[2][0]
    lo, hi = float(proj.min()), float(proj.max())
    span = rng.uniform(0.4, 0.9) * (hi - lo)
    start = rng.uniform(lo, hi - span)
    keep = (proj >= start) & (proj <= start + span)
    out = np.zeros_like(m)
    out[ys[keep], xs[keep]] = 1
    return out if out.any() else m


def box_fill(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    out = np.zeros_like(m)
    out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def build_spine_crop_cache(entries: list[dict], per_image: dict, cache_dir: Path, prefix: str,
                           n_trunc: int = 0, seed: int = 0, with_hint: bool = False,
                           crop_size: int = CROP_SIZE):
    """Like build_crop_cache, but also rasterizes each filament's manually
    annotated GT spine (centerline polyline) into the same crop frame. The
    host confirmed spine data is fair to use as auxiliary training
    supervision -- unlike clDice's self-consistency skeleton (derived from
    the model's own prediction), this is a real, human-labeled centerline,
    a much stronger geometric signal for a thin/curvilinear structure.

    n_trunc > 0 adds that many extra crops per instance whose window is
    square_bounds of a truncate_mask() stretch, target still the FULL
    filament. At inference the window is centred on a (often partial)
    proposal, not on the GT; GT-centred-only training taught the refiner to
    drop off-centre continuations -- 91/91 of the 2-way pipeline's near
    misses had >50% of the missing GT inside the refiner's window.

    with_hint also stores the proposal the window came from as a hint
    channel (the stretch mask, or half the time its filled bbox, matching
    Mask R-CNN mask vs YOLO box proposals). Gray-only v7 trained on
    off-centre targets had no way to tell which structure is 'this'
    filament and learned to grab neighbours; the hint disambiguates."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / f"{prefix}_images.npy"
    mask_path = cache_dir / f"{prefix}_masks.npy"
    spine_path = cache_dir / f"{prefix}_spines.npy"
    hint_path = cache_dir / f"{prefix}_hints.npy"
    rng = random.Random(seed)

    total = sum(len(per_image.get(e["id"], [])) for e in entries) * (1 + n_trunc)
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                        shape=(total, crop_size, crop_size))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8,
                                       shape=(total, crop_size, crop_size))
    spines = np.lib.format.open_memmap(spine_path, mode="w+", dtype=np.uint8,
                                        shape=(total, crop_size, crop_size))
    hints = (np.lib.format.open_memmap(hint_path, mode="w+", dtype=np.uint8,
                                       shape=(total, crop_size, crop_size)) if with_hint else None)

    pos = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        if not anns:
            continue
        img = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue

            spine_full = np.zeros((H, W), dtype=np.uint8)
            pts = np.asarray(a.get("spine") or [], dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 2:
                cv2.polylines(spine_full, [pts.round().astype(np.int32)],
                               isClosed=False, color=1, thickness=3)

            for window_mask in [m] + [truncate_mask(m, rng) for _ in range(n_trunc)]:
                x0, y0, x1, y1 = square_bounds(window_mask)
                crop_img = np.array(Image.fromarray(img[y0:y1, x0:x1]).resize(
                    (crop_size, crop_size), Image.BILINEAR))
                crop_mask = np.array(Image.fromarray(m[y0:y1, x0:x1] * 255).resize(
                    (crop_size, crop_size), Image.NEAREST))
                crop_spine = np.array(Image.fromarray(spine_full[y0:y1, x0:x1] * 255).resize(
                    (crop_size, crop_size), Image.NEAREST))

                images[pos] = crop_img
                masks[pos] = (crop_mask > 127).astype(np.uint8)
                spines[pos] = (crop_spine > 127).astype(np.uint8)
                if with_hint:
                    h = box_fill(window_mask) if rng.random() < 0.5 else window_mask
                    hints[pos] = np.array(Image.fromarray(h[y0:y1, x0:x1] * 255).resize(
                        (crop_size, crop_size), Image.NEAREST)) > 127
                pos += 1

    images.flush(); masks.flush(); spines.flush()
    if with_hint:
        hints.flush()
    if pos < total:
        np.save(img_path, np.asarray(images[:pos]))
        np.save(mask_path, np.asarray(masks[:pos]))
        np.save(spine_path, np.asarray(spines[:pos]))
        if with_hint:
            np.save(hint_path, np.asarray(hints[:pos]))
    return img_path, mask_path, spine_path, pos


class SpineCropDataset(torch.utils.data.Dataset):
    """Returns (image, mask, spine) triples: mask is the main target, spine
    is an auxiliary centerline target trained jointly, ignored at inference."""

    def __init__(self, img_path: Path, mask_path: Path, spine_path: Path, augment: bool = False,
                 hint_path: Path = None):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.spines = np.load(spine_path, mmap_mode="r")
        self.hints = np.load(hint_path, mmap_mode="r") if hint_path else None
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.asarray(self.images[idx]).copy()).float() / 255.0
        mask = torch.from_numpy(np.asarray(self.masks[idx]).copy()).float()
        spine = torch.from_numpy(np.asarray(self.spines[idx]).copy()).float()
        img, mask, spine = img.unsqueeze(0), mask.unsqueeze(0), spine.unsqueeze(0)
        if self.hints is not None:  # 2-channel input; flips/rotations below apply to both
            hint = torch.from_numpy(np.asarray(self.hints[idx]).copy()).float().unsqueeze(0)
            img = torch.cat([img, hint], 0)

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, (1, 2))
            mask = torch.rot90(mask, k, (1, 2))
            spine = torch.rot90(spine, k, (1, 2))
            if torch.rand(1).item() < 0.5:
                img, mask, spine = torch.flip(img, (2,)), torch.flip(mask, (2,)), torch.flip(spine, (2,))
            if torch.rand(1).item() < 0.5:  # brightness: gray channel only, never the hint
                img[0:1] = torch.clamp(img[0:1] * float(torch.empty(1).uniform_(0.85, 1.15)), 0, 1)

        target = torch.cat([mask, spine], dim=0)  # (2, H, W): channel 0=mask, 1=spine
        return img, target
