"""Axis 2 -- deterministic image degradations, rendered on the fly.

Levels follow the project brief:

* Gaussian blur   sigma in {1, 2, 4, 8, 16} px
* motion blur     horizontal kernel length in {5, 10, 20, 40} px
* low-resource crop: only {50, 25, 10, 5}% of the queried object's box is visible
* (optional) additive Gaussian noise

Every image is first standardised to ``max_side`` px (COCO is ~640 already), so a
given sigma means the same thing on every photo. Degraded images are *not*
written to disk (200k items x 13 levels of PNGs would be ~2 TB); they are
re-rendered deterministically when a model needs them. ``materialize`` writes
them out for the human study, the gallery, or a release.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..schema import BaseItem


def recipe_tag(recipe: Dict[str, Any]) -> str:
    m = recipe.get("method", "none")
    if m == "none":
        return "orig"
    if m == "blank":
        return "blank"
    if m == "gauss":
        return f"gauss_s{recipe['sigma']:g}"
    if m == "motion":
        return f"motion_k{recipe['kernel']:g}"
    if m == "crop":
        return f"crop_f{recipe['frac']:g}"
    if m == "noise":
        return f"noise_s{recipe['std']:g}"
    raise ValueError(f"unknown degradation: {recipe}")


def visual_recipes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = cfg["axes"]["visual"]
    return ([{"method": "gauss", "sigma": s} for s in v["gauss_sigmas"]]
            + [{"method": "motion", "kernel": k} for k in v["motion_kernels"]]
            + [{"method": "crop", "frac": f} for f in v["crop_fracs"]]
            + [{"method": "noise", "std": s} for s in v.get("noise_stds", [])])


@lru_cache(maxsize=16)
def _load_standardised(path: str, max_side: int) -> Tuple[np.ndarray, float]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy(), scale


def crop_window(item: BaseItem, frac: float) -> List[float]:
    """Window (x, y, w, h, original pixels) holding ``frac`` of the object's box area."""
    x, y, w, h = item.target_bbox
    s = float(np.sqrt(frac))
    cw, ch = max(8.0, w * s), max(8.0, h * s)
    cx, cy = x + w / 2, y + h / 2
    W, H = item.image_size
    x0 = min(max(0.0, cx - cw / 2), max(0.0, W - cw))
    y0 = min(max(0.0, cy - ch / 2), max(0.0, H - ch))
    return [x0, y0, min(cw, W), min(ch, H)]


def visible_objects(item: BaseItem, recipe: Dict[str, Any]) -> List[str]:
    """Ground-truth categories visible under a recipe (CHAIR ground truth)."""
    if recipe.get("method") == "blank":
        return []
    if recipe.get("method") != "crop":
        return sorted({o[0] for o in item.objects})
    wx, wy, ww, wh = crop_window(item, recipe["frac"])
    vis = set()
    for cat, ox, oy, ow, oh in item.objects:
        ix = min(wx + ww, ox + ow) - max(wx, ox)
        iy = min(wy + wh, oy + oh) - max(wy, oy)
        if ix > 0 and iy > 0:
            vis.add(cat)
    return sorted(vis)


def render(item: Optional[BaseItem], recipe: Dict[str, Any], image_root: str | Path,
           cfg: Dict[str, Any]) -> Image.Image:
    method = recipe.get("method", "none")
    if method == "blank":
        n = cfg["image"]["blank_size"]
        return Image.new("RGB", (n, n), (128, 128, 128))
    assert item is not None
    from scipy import ndimage  # local: keeps `import gib` light

    arr, scale = _load_standardised(str(Path(image_root) / item.image_path), cfg["image"]["max_side"])
    if method == "none":
        out = arr
    elif method == "gauss":
        s = float(recipe["sigma"])
        out = ndimage.gaussian_filter(arr.astype(np.float32), sigma=(s, s, 0), mode="reflect", truncate=3.0)
    elif method == "motion":
        out = ndimage.uniform_filter1d(arr.astype(np.float32), size=int(recipe["kernel"]), axis=1, mode="reflect")
    elif method == "noise":
        seed = int(hashlib.sha256(f"{item.gib_id}:{float(recipe['std'])}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out = arr.astype(np.float32) + rng.normal(0.0, float(recipe["std"]), arr.shape)
    elif method == "crop":
        x, y, w, h = (v * scale for v in crop_window(item, float(recipe["frac"])))
        H, W = arr.shape[:2]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(W, max(x0 + 1, int(np.ceil(x + w)))), min(H, max(y0 + 1, int(np.ceil(y + h))))
        patch = Image.fromarray(arr[y0:y1, x0:x1])
        side = cfg["image"]["max_side"]
        k = side / max(patch.size)
        return patch.resize((max(1, round(patch.size[0] * k)), max(1, round(patch.size[1] * k))), Image.BICUBIC)
    else:
        raise ValueError(f"unknown degradation: {recipe}")
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def materialize(item: BaseItem, recipe: Dict[str, Any], image_root: str | Path, cfg: Dict[str, Any],
                out_dir: str | Path, fmt: str = "jpg") -> Path:
    out = Path(out_dir) / f"{item.gib_id}__{recipe_tag(recipe)}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = render(item, recipe, image_root, cfg)
    img.save(out, quality=92) if fmt == "jpg" else img.save(out)
    return out
