"""Package GIB for GitHub / Hugging Face Datasets.

Only our annotations are released (questions, answers, hard negatives,
document templates, degradation recipes, flip pairs, subsets) under CC BY 4.0.
Images are *referenced* by their COCO file name and URL, not redistributed:
COCO images carry their own Flickr licences. ``gib materialize`` re-renders
every degraded image deterministically from the original photo.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from ..axes.degrade import visual_recipes
from ..schema import BaseItem, read_jsonl


def export(data_dir: str | Path, out_dir: str | Path, cfg: Dict[str, Any], name: str = "GIB-200K") -> Dict[str, Any]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    items = read_jsonl(data_dir / "items.jsonl", BaseItem)
    shutil.copy(data_dir / "items.jsonl", out_dir / "data" / "items.jsonl")
    for sub in (data_dir / "subsets").glob("*.txt"):
        shutil.copy(sub, out_dir / "data" / sub.name)
    (out_dir / "data" / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    n = len(items)
    types = Counter(it.qa_type for it in items)
    splits = Counter(it.source_split for it in items)
    pairs = sum(1 for it in items if it.flip_partner) // 2
    levels = ", ".join(sorted({r["method"] for r in visual_recipes(cfg)}))
    size = ("1M<n<10M" if n >= 1_000_000 else "100K<n<1M" if n >= 100_000 else "10K<n<100K" if n >= 10_000
            else "1K<n<10K" if n >= 1000 else "n<1K")
    card = f"""---
license: cc-by-4.0
pretty_name: {name}
language:
- en
task_categories:
- visual-question-answering
size_categories:
- {size}
tags:
- multimodal-rag
- hallucination
- robustness
- benchmark
---

# {name}: Grounding Illusion Benchmark

{n:,} image–question items built from COCO annotations (objects, captions) and Visual Genome
(attributes, relations), each with a closed answer vocabulary, ranked hard negatives, and
document templates. Every item expands deterministically into the benchmark's stress conditions:
retrieval noise ({cfg['axes']['retrieval']['dosages']} hard-negative documents), visual ambiguity
({levels}; Gaussian σ {cfg['axes']['visual']['gauss_sigmas']}, motion {cfg['axes']['visual']['motion_kernels']} px,
{', '.join(f'{int(100 * f)}%' for f in cfg['axes']['visual']['crop_fracs'])} of the object visible) and cross-modal
conflict (with counterfactual flip pairs).

| | |
|---|---|
| items | {n:,} |
| by type | {dict(types)} |
| by source split | {dict(splits)} |
| counterfactual flip pairs | {pairs:,} |

## Files

- `data/items.jsonl` — one JSON object per item (fields documented in `gib/schema.py::BaseItem`).
- `data/eval.txt`, `data/dev.txt` — the fixed evaluation subsets (every model is evaluated on the same ids).
- `data/config.json` — the exact levels and probe settings used to expand items into queries.

## Images

Images are not redistributed. `image_path` is relative to a COCO 2017 root
(`val2017/…`, `train2017/…`); `provenance.coco_url` points to the original. Degraded variants are
re-created bit-for-bit with `python -m gib materialize`.

## Licence

Annotations in this repository: CC BY 4.0. COCO annotations: CC BY 4.0. Visual Genome: CC BY 4.0.
COCO images: the Flickr licence of each image (see `provenance.license`).

## Contamination note

`source_split` separates images from COCO `train2017` (used in the instruction-tuning data of
several open VLMs, e.g. LLaVA) from `val2017` (not in those mixtures). Results should be reported
per split; see the paper's contamination analysis.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return {"n_items": n, "out": str(out_dir)}
