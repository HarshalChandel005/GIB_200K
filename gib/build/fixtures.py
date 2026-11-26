"""Tiny synthetic COCO + Visual Genome look-alikes, for tests and the offline demo.

The files follow the real formats (COCO ``instances_*.json``/``captions_*.json``;
VG ``image_data.json``/``attributes.json``/``relationships.json``) so the *real*
builder is exercised end to end. VG image ids are deliberately different from
the COCO ids, and one decoy VG record re-uses a COCO id number for an unrelated
image -- the builder must ignore it (regression test for the old id-join bug).
Images are flat shapes; they are not meant to be recognisable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from .. import vocab

COLOR_RGB = {"white": (240, 240, 240), "black": (20, 20, 20), "gray": (128, 128, 128), "red": (200, 30, 30),
             "blue": (30, 60, 200), "green": (30, 160, 60), "yellow": (230, 210, 40), "brown": (120, 70, 30),
             "orange": (240, 140, 20), "pink": (240, 150, 190), "purple": (120, 40, 160)}
PREDICATES = ["on", "next to", "behind", "under", "sitting on", "near", "in front of", "above"]


def make_fixture(root: str | Path, n_images: int = 120, seed: int = 0,
                 splits: Tuple[str, ...] = ("val2017", "train2017")) -> Dict[str, Path]:
    root = Path(root)
    rng = random.Random(seed)
    coco, vg = root / "coco", root / "vg"
    (coco / "annotations").mkdir(parents=True, exist_ok=True)
    vg.mkdir(parents=True, exist_ok=True)
    cats = sorted(vocab.CATEGORY_TO_GROUP) + ["person"]
    cat_id = {c: i + 1 for i, c in enumerate(cats)}
    image_data, vg_attrs, vg_rels = [], [], []
    ann_id = 1
    for s_i, split in enumerate(splits):
        (coco / split).mkdir(parents=True, exist_ok=True)
        images, anns, caps = [], [], []
        for k in range(n_images):
            iid = 100000 * (s_i + 1) + k
            W, H = 320, 240
            groups = rng.sample(sorted(vocab.COCO_GROUPS), 2)
            objs: List[Tuple[str, List[float], str]] = []
            for gi, g in enumerate(groups):
                c = rng.choice(vocab.COCO_GROUPS[g])
                w, h = rng.randint(60, 110), rng.randint(50, 90)
                x = 10 + gi * 160 + rng.randint(0, 30)
                y = rng.randint(20, H - h - 10)
                objs.append((c, [float(x), float(y), float(w), float(h)], rng.choice(sorted(COLOR_RGB))))
            img = Image.new("RGB", (W, H), tuple(rng.randint(150, 230) for _ in range(3)))
            d = ImageDraw.Draw(img)
            for c, (x, y, w, h), col in objs:
                (d.ellipse if cat_id[c] % 2 else d.rectangle)([x, y, x + w, y + h], fill=COLOR_RGB[col])
            fname = f"{iid:012d}.jpg"
            img.save(coco / split / fname, quality=90)
            images.append({"id": iid, "file_name": fname, "width": W, "height": H, "license": 4,
                           "coco_url": f"http://example.invalid/{fname}"})
            for c, box, _ in objs:
                anns.append({"id": ann_id, "image_id": iid, "category_id": cat_id[c], "bbox": box,
                             "area": box[2] * box[3], "iscrowd": 0})
                ann_id += 1
            a, b = objs[0][0], objs[1][0]
            for t in (f"A photo of a {a} next to a {b}.", f"There is a {a} in this picture.",
                      f"A {vocab.pluralize(a)} scene with a {b}."):
                caps.append({"id": ann_id, "image_id": iid, "caption": t})
                ann_id += 1
            vg_id = 900000 + 1000 * s_i + k  # VG numbers images independently of COCO
            image_data.append({"image_id": vg_id, "coco_id": iid, "width": W * 2, "height": H * 2})
            vobjs = [{"object_id": vg_id * 10 + j, "x": box[0] * 2, "y": box[1] * 2, "w": box[2] * 2, "h": box[3] * 2,
                      "names": [c], "attributes": [col, "large"] if j == 0 else [col]}
                     for j, (c, box, col) in enumerate(objs)]
            vg_attrs.append({"image_id": vg_id, "attributes": vobjs})
            pred = rng.choice(PREDICATES)
            vg_rels.append({"image_id": vg_id, "relationships": [
                {"relationship_id": vg_id, "predicate": pred.upper() if rng.random() < 0.3 else pred,
                 "subject": {k_: v for k_, v in vobjs[0].items() if k_ != "attributes"},
                 "object": {k_: v for k_, v in vobjs[1].items() if k_ != "attributes"}}]})
        # decoy: a VG image whose *VG id* equals a COCO id but which is not that photo
        image_data.append({"image_id": images[0]["id"], "coco_id": None, "width": 500, "height": 500})
        vg_attrs.append({"image_id": images[0]["id"], "attributes": [
            {"object_id": 1, "x": 0, "y": 0, "w": 400, "h": 400, "names": ["decoy"], "attributes": ["purple"]}]})
        categories = [{"id": cat_id[c], "name": c, "supercategory": vocab.CATEGORY_TO_GROUP.get(c, "person")}
                      for c in cats]
        (coco / "annotations" / f"instances_{split}.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": categories}), encoding="utf-8")
        (coco / "annotations" / f"captions_{split}.json").write_text(
            json.dumps({"images": images, "annotations": caps}), encoding="utf-8")
    (vg / "image_data.json").write_text(json.dumps(image_data), encoding="utf-8")
    (vg / "attributes.json").write_text(json.dumps(vg_attrs), encoding="utf-8")
    (vg / "relationships.json").write_text(json.dumps(vg_rels), encoding="utf-8")
    return {"coco_root": coco, "vg_root": vg}
