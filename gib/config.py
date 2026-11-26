"""Configuration: one YAML file fixes every level, probe and sampling choice.

The defaults below reproduce the levels required by the project brief
(Gaussian blur sigma 1/2/4/8/16, motion blur 5/10/20/40 px, 50/25/10/5% of the
object visible, 1/3/5/10/20 hard-negative documents). A YAML file passed with
``--config`` is deep-merged on top, and the merged config is hashed into every
run's metadata so results can always be traced to the exact settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "seed": 0,
    "image": {"max_side": 640, "blank_size": 448},
    "build": {
        "per_type": {"object": 66667, "attribute": 66667, "relation": 66666},  # 200,000 items (GIB-200K)
        "max_items_per_image": 4,       # 200k items need several per image: with 3, real COCO+VG give ~186k
        "max_answer_share": {"relation": 0.33},  # no answer above this share of a type
        "min_area_frac": 0.04,          # queried object must cover >= 4% of the image
        "max_instances": 3,             # at most 3 instances of the answer category
        "leftright_min_gap": 0.15,      # |dx| between object centres, as a fraction of width
    },
    "subsets": {"eval": 200000, "dev": 30},  # eval = every item: all models answer the full benchmark
    "axes": {
        "retrieval": {"dosages": [1, 3, 5, 10, 20]},
        "visual": {
            "gauss_sigmas": [1, 2, 4, 8, 16],
            "motion_kernels": [5, 10, 20, 40],
            "crop_fracs": [0.5, 0.25, 0.10, 0.05],
            "noise_stds": [],           # optional Gaussian-noise family (0-255 units)
        },
        "conflict": {"sigmas": [4, 16]},     # conflict also run on degraded images
        "grid": {"sigmas": [4, 16], "dosages": [0, 5, 20]},
    },
    "probes": {
        "mc": True,
        "ab": True,
        "pope": True,
        "caption_frac": 0.25,           # fraction of items that also get caption probes (CHAIR)
    },
    "decoding": {"qa_max_new_tokens": 10, "caption_max_new_tokens": 48},
    "analysis": {
        "bootstrap_iters": 2000,
        "resolution_jump": 0.10,        # CHAIR (or accuracy) change that defines the threshold
        "confident_wrong": 0.8,
        "logodds_clip": 8.0,
        "min_prob_coverage": 0.9,       # share of a probe's answers that must have usable token probabilities
        "min_format_mass": 0.5,         # probability on the valid answer tokens below which one is not usable
    },
}


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml  # local import: only needed when a file is given

        with Path(path).open("r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
