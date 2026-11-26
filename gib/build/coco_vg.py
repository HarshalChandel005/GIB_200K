"""Build GIB items from real annotations: COCO (+ optional Visual Genome).

Question types (all with a *closed*, scoreable answer vocabulary):

* object    -- "What animal is shown in the image?" Only images where exactly
               one category of the group is annotated (so the answer is unique)
               and the instance is large enough to see. Siblings = the other
               group members, ranked by how often they co-occur with the truth
               in COCO (contextually plausible = hard negatives).
* attribute -- "What color is the bus?" From Visual Genome attributes of an
               object whose name is unique in the image and which carries
               exactly one colour attribute. Needs ``--vg-root``.
* relation  -- "Where is the cat relative to the couch?" from Visual Genome
               spatial predicates (antonym first as the hard negative), and
               "Is the dog to the left or to the right of the car?" computed
               from COCO boxes for every clearly separated pair of objects
               (needs no VG). "on" is most of Visual Genome's predicates, so
               ``build.max_answer_share`` limits any one answer's share.

Visual Genome is linked to COCO through ``image_data.json``'s ``coco_id`` field.
(The previous builder joined on the raw image id, but VG and COCO number their
images independently, so questions were attached to the wrong photos.)
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import vocab
from ..schema import BaseItem, QAType, write_jsonl

OBJECT_TEMPLATES = [
    "This photo shows a {ANS}.",
    "The picture contains a {ANS} in the scene.",
    "An image featuring a {ANS}.",
    "A {ANS} can be seen in this photograph.",
]
COLOR_TEMPLATES = [
    "The {OBJ} in this photo is {ANS}.",
    "A {ANS} {OBJ} can be seen in the image.",
    "The image shows a {ANS} {OBJ}.",
    "In this picture the {OBJ} is {ANS} in color.",
]
SPATIAL_TEMPLATES = [
    "The {SUBJ} is {ANS} the {OBJ}.",
    "In this picture, the {SUBJ} appears {ANS} the {OBJ}.",
    "A scene where the {SUBJ} is {ANS} the {OBJ}.",
    "Here, the {SUBJ} is located {ANS} the {OBJ}.",
]
COLOR_TEMPLATES_PL = [
    "The {OBJ} in this photo are {ANS}.",
    "{ANS} {OBJ} can be seen in the image.",
    "The image shows {ANS} {OBJ}.",
    "In this picture the {OBJ} are {ANS} in color.",
]
SPATIAL_TEMPLATES_PL = [
    "The {SUBJ} are {ANS} the {OBJ}.",
    "In this picture, the {SUBJ} appear {ANS} the {OBJ}.",
    "A scene where the {SUBJ} are {ANS} the {OBJ}.",
    "Here, the {SUBJ} are located {ANS} the {OBJ}.",
]
LEFTRIGHT_TEMPLATES_PL = [
    "The {SUBJ} are to the {ANS} of the {OBJ}.",
    "In this picture, the {SUBJ} appear to the {ANS} of the {OBJ}.",
    "Seen from the camera, the {SUBJ} are on the {ANS} side of the {OBJ}.",
    "A scene where the {SUBJ} stand to the {ANS} of the {OBJ}.",
]
LEFTRIGHT_TEMPLATES = [
    "The {SUBJ} is to the {ANS} of the {OBJ}.",
    "In this picture, the {SUBJ} appears to the {ANS} of the {OBJ}.",
    "Seen from the camera, the {SUBJ} is on the {ANS} side of the {OBJ}.",
    "A scene where the {SUBJ} stands to the {ANS} of the {OBJ}.",
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CocoSplit:
    """One COCO-format split (instances + optional captions)."""

    def __init__(self, ann_file: Path, image_dir: str, split: str, captions_file: Optional[Path] = None):
        data = _load_json(ann_file)
        self.split = split
        self.image_dir = image_dir
        self.cats = {c["id"]: c["name"] for c in data["categories"]}
        self.images = {im["id"]: im for im in data["images"]}
        self.anns: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for a in data["annotations"]:
            a.pop("segmentation", None)  # unused polygons: most of train2017's memory
            self.anns[a["image_id"]].append(a)
        self.captions: Dict[int, List[str]] = defaultdict(list)
        if captions_file is not None and captions_file.exists():
            for a in _load_json(captions_file)["annotations"]:
                self.captions[a["image_id"]].append(a["caption"].strip())
        # category co-occurrence (images containing both)
        self.cooc: Dict[str, Counter] = defaultdict(Counter)
        for anns in self.anns.values():
            present = sorted({self.cats[a["category_id"]] for a in anns})
            for c in present:
                for d in present:
                    if c != d:
                        self.cooc[c][d] += 1

    def present(self, image_id: int) -> List[str]:
        return sorted({self.cats[a["category_id"]] for a in self.anns.get(image_id, [])})

    def objects(self, image_id: int) -> List[List[Any]]:
        return [[self.cats[a["category_id"]], *[float(v) for v in a["bbox"]]]
                for a in self.anns.get(image_id, [])]


class VisualGenome:
    """The parts of Visual Genome we use, re-indexed by COCO image id."""

    def __init__(self, root: Path, coco_ids: Iterable[int]):
        wanted = set(coco_ids)
        meta = _load_json(root / "image_data.json")
        self.vg_to_coco: Dict[int, int] = {}
        self.size: Dict[int, Tuple[int, int]] = {}
        for m in meta:
            cid = m.get("coco_id")
            vid = m.get("image_id", m.get("id"))
            if cid is not None and cid in wanted:
                self.vg_to_coco[vid] = cid
                self.size[cid] = (int(m["width"]), int(m["height"]))
        self.objects: Dict[int, List[Dict[str, Any]]] = {}
        self.relations: Dict[int, List[Dict[str, Any]]] = {}
        sg = root / "scene_graphs.json"
        attrs = root / "attributes.json"
        rels = root / "relationships.json"
        if attrs.exists():
            for rec in _load_json(attrs):
                cid = self.vg_to_coco.get(rec.get("image_id", rec.get("id")))
                if cid is not None:
                    self.objects[cid] = [self._obj(o) for o in rec.get("attributes", [])]
        if rels.exists():
            for rec in _load_json(rels):
                cid = self.vg_to_coco.get(rec.get("image_id", rec.get("id")))
                if cid is not None:
                    self.relations[cid] = [
                        {"predicate": r.get("predicate", ""), "subject": self._obj(r["subject"]),
                         "object": self._obj(r["object"]), "id": r.get("relationship_id")}
                        for r in rec.get("relationships", [])
                    ]
        if sg.exists() and not (attrs.exists() or rels.exists()):
            for rec in _load_json(sg):
                cid = self.vg_to_coco.get(rec.get("image_id", rec.get("id")))
                if cid is None:
                    continue
                objs = {o.get("object_id"): self._obj(o) for o in rec.get("objects", [])}
                self.objects[cid] = list(objs.values())
                self.relations[cid] = [
                    {"predicate": r.get("predicate", ""), "subject": objs.get(r.get("subject_id")),
                     "object": objs.get(r.get("object_id")), "id": r.get("relationship_id")}
                    for r in rec.get("relationships", [])
                    if r.get("subject_id") in objs and r.get("object_id") in objs
                ]

    @staticmethod
    def _obj(o: Dict[str, Any]) -> Dict[str, Any]:
        names = o.get("names") or ([o["name"]] if "name" in o else [])
        if "x" in o:
            box = [o["x"], o["y"], o["w"], o["h"]]
        else:
            box = list(o.get("bbox", [0, 0, 0, 0]))
        return {"name": " ".join(vocab.normalize_tokens(names[0])) if names else "",
                "box": [float(v) for v in box], "attributes": list(o.get("attributes") or []),
                "id": o.get("object_id", o.get("id"))}

    def scale(self, coco_id: int, box: Sequence[float], coco_wh: Tuple[int, int]) -> List[float]:
        vw, vh = self.size.get(coco_id, coco_wh)
        sx, sy = coco_wh[0] / max(vw, 1), coco_wh[1] / max(vh, 1)
        return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #
def _area_frac(box: Sequence[float], w: int, h: int) -> float:
    return (box[2] * box[3]) / max(w * h, 1)


def _rank_siblings(truth: str, group: Sequence[str], cooc: Dict[str, Counter]) -> List[str]:
    others = [c for c in group if c != truth]
    return sorted(others, key=lambda c: (-cooc.get(truth, Counter())[c], c))


def _object_candidates(cs: CocoSplit, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    bcfg = cfg["build"]
    for iid, im in cs.images.items():
        anns = cs.anns.get(iid, [])
        w, h = im["width"], im["height"]
        for group, members in vocab.COCO_GROUPS.items():
            hits = [a for a in anns if cs.cats[a["category_id"]] in members]
            cats = {cs.cats[a["category_id"]] for a in hits}
            if len(cats) != 1 or any(a.get("iscrowd") for a in hits):
                continue
            (cat,) = cats
            if len(hits) > bcfg["max_instances"]:
                continue
            big = max(hits, key=lambda a: a["bbox"][2] * a["bbox"][3])
            if _area_frac(big["bbox"], w, h) < bcfg["min_area_frac"]:
                continue
            templates = []
            for cap in cs.captions.get(iid, []):
                mentioned = {c for _, c in vocab.find_mentions(cap, vocab.group_synonyms(group))}
                if mentioned == {cat}:
                    t = vocab.mention_to_slot(cap, cat)
                    if t and t not in templates:
                        templates.append(t)
            templates = templates[:3] + OBJECT_TEMPLATES
            out.append({
                "qa_type": "object", "group": group, "image_id": iid,
                "question": f"What {group} is shown in the image?", "answer": cat,
                "siblings": _rank_siblings(cat, members, cs.cooc), "bbox": [float(v) for v in big["bbox"]],
                "templates": templates, "slots": {}, "pope_pos": cat,
                "prov": {"coco_ann_id": big.get("id")},
            })
    return out


def _color_of(attrs: Sequence[str]) -> Optional[str]:
    found = set()
    for a in attrs:
        c = vocab.extract_answer(a, "color")
        if c:
            found.add(c)
    return found.pop() if len(found) == 1 else None


def _attribute_candidates(cs: CocoSplit, vg: VisualGenome, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    min_frac = cfg["build"]["min_area_frac"] / 2
    for iid, objs in vg.objects.items():
        if iid not in cs.images:
            continue
        im = cs.images[iid]
        wh = (im["width"], im["height"])
        counts = Counter(o["name"] for o in objs if o["name"])
        for o in objs:
            name = o["name"]
            if not name or counts[name] != 1 or not o["attributes"]:
                continue
            color = _color_of(o["attributes"])
            if color is None:
                continue
            box = vg.scale(iid, o["box"], wh)
            if _area_frac(box, *wh) < min_frac:
                continue
            pl = vocab.is_plural(name)
            out.append({
                "qa_type": "attribute", "group": "color", "image_id": iid,
                "question": f"What color {'are' if pl else 'is'} the {name}?", "answer": color,
                "siblings": list(vocab.COLOR_NEIGHBORS[color]), "bbox": box,
                "templates": list(COLOR_TEMPLATES_PL if pl else COLOR_TEMPLATES), "slots": {"OBJ": name},
                "pope_pos": None, "prov": {"vg_object_id": o["id"], "vg_attributes": o["attributes"]},
            })
    return out


def _spatial_candidates(cs: CocoSplit, vg: VisualGenome, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for iid, rels in vg.relations.items():
        if iid not in cs.images:
            continue
        im = cs.images[iid]
        wh = (im["width"], im["height"])
        objs = vg.objects.get(iid)
        if objs:
            counts = Counter(o["name"] for o in objs if o["name"])
        else:  # fall back to relation endpoints, deduplicated by object id
            seen = {}
            for r in rels:
                for end in (r["subject"], r["object"]):
                    if end:
                        seen[end["id"]] = end["name"]
            counts = Counter(seen.values())
        for r in rels:
            pred = vocab.extract_answer(r["predicate"], "spatial")
            s, o = r["subject"], r["object"]
            if pred is None or not s or not o:
                continue
            # the raw predicate must be (a synonym of) the canonical one, nothing more
            raw = " ".join(vocab.normalize_tokens(r["predicate"]))
            syns = {" ".join(vocab.normalize_tokens(x)) for x in vocab.SPATIAL_SYNONYMS[pred]}
            if raw not in syns:
                continue
            if not s["name"] or not o["name"] or s["name"] == o["name"]:
                continue
            if counts[s["name"]] != 1 or counts[o["name"]] != 1:
                continue
            pl = vocab.is_plural(s["name"])
            out.append({
                "qa_type": "relation", "group": "spatial", "image_id": iid,
                "question": f"Where {'are' if pl else 'is'} the {s['name']} relative to the {o['name']}?", "answer": pred,
                "siblings": list(vocab.SPATIAL_NEIGHBORS[pred]), "bbox": vg.scale(iid, s["box"], wh),
                "templates": list(SPATIAL_TEMPLATES_PL if pl else SPATIAL_TEMPLATES),
                "slots": {"SUBJ": s["name"], "OBJ": o["name"]},
                "pope_pos": None, "prov": {"vg_relationship_id": r["id"], "vg_predicate": r["predicate"]},
            })
    return out


def _leftright_candidates(cs: CocoSplit, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    gap = cfg["build"]["leftright_min_gap"]
    min_frac = cfg["build"]["min_area_frac"] / 2
    for iid, im in cs.images.items():
        w, h = im["width"], im["height"]
        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for a in cs.anns.get(iid, []):
            by_cat[cs.cats[a["category_id"]]].append(a)
        singles = [(c, v[0]) for c, v in by_cat.items()
                   if len(v) == 1 and not v[0].get("iscrowd")
                   and _area_frac(v[0]["bbox"], w, h) >= min_frac]
        singles.sort(key=lambda x: -x[1]["bbox"][2] * x[1]["bbox"][3])
        for (ca, a), (cb, b) in itertools.combinations(singles, 2):  # larger object first
            ax, bx = a["bbox"][0] + a["bbox"][2] / 2, b["bbox"][0] + b["bbox"][2] / 2
            overlap = max(0.0, min(a["bbox"][0] + a["bbox"][2], b["bbox"][0] + b["bbox"][2])
                          - max(a["bbox"][0], b["bbox"][0]))
            if abs(ax - bx) < gap * w or overlap > 0.3 * min(a["bbox"][2], b["bbox"][2]):
                continue
            ans = "left" if ax < bx else "right"
            da, db = vocab.display(ca), vocab.display(cb)
            pl = vocab.is_plural(da)
            out.append({
                "qa_type": "relation", "group": "leftright", "image_id": iid,
                "question": f"In the image, {'are' if pl else 'is'} the {da} to the left or to the right of the {db}?",
                "answer": ans, "siblings": ["right" if ans == "left" else "left"],
                "bbox": [float(v) for v in a["bbox"]],
                "templates": list(LEFTRIGHT_TEMPLATES_PL if pl else LEFTRIGHT_TEMPLATES),
                "slots": {"SUBJ": da, "OBJ": db}, "pope_pos": ca,
                "prov": {"coco_ann_ids": [a.get("id"), b.get("id")]},
            })
    return out


# --------------------------------------------------------------------------- #
# selection, POPE negatives, flip pairing
# --------------------------------------------------------------------------- #
def _round_robin(cands: List[Dict[str, Any]], n: int, rng: random.Random,
                 used_images: Counter, max_per_image: int, key: str = "group",
                 room: Optional[Counter] = None) -> List[Dict[str, Any]]:
    """Pick up to n candidates, cycling over groups so no group dominates.
    ``room`` (updated here) says how many more of each ``key`` value may be picked."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        buckets[c[key]].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets)
    picked: List[Dict[str, Any]] = []
    while len(picked) < n and any(buckets.values()):
        for g in order:
            if room is not None and room[g] <= 0:
                buckets[g].clear()
            while buckets[g]:
                c = buckets[g].pop()
                img = (c["split"], c["image_id"])
                if used_images[img] >= max_per_image:
                    continue
                used_images[img] += 1
                picked.append(c)
                if room is not None:
                    room[g] -= 1
                break
            if len(picked) >= n:
                break
    return picked


def _pope_negative(cs: CocoSplit, pos: Optional[str], present: Sequence[str], rng: random.Random) -> Optional[str]:
    absent = [c for c in vocab.CATEGORY_TO_GROUP if c not in present]
    if not absent:
        return None
    if pos and cs.cooc.get(pos):
        ranked = [c for c, _ in cs.cooc[pos].most_common() if c in absent]
        if ranked:
            return ranked[0]          # adversarial: frequent co-occurrer that is not there
    return rng.choice(sorted(absent))


def pair_conflicts(items: List[BaseItem]) -> int:
    """Give every item a conflict claim; pair items into mirror-image flips.

    Item i (image says A) gets the claim B; if some item j (image says B) can get
    the claim A, the two form a counterfactual flip pair. Returns #pairs.

    Candidates are indexed by (group, answer, near sibling) and scanned from a
    per-key cursor (items before it are already paired), so this is linear in
    the number of items -- a plain scan is quadratic at 200k items -- and picks
    the same partners as one: the first unpaired match in item order.
    """
    index: Dict[Tuple[str, str, str], List[BaseItem]] = defaultdict(list)
    for j in items:
        for s in j.siblings[:4]:
            index[(j.group, j.answer, s)].append(j)
    cursor: Counter = Counter()
    pairs = 0
    for it in items:
        if it.flip_partner is not None:
            continue
        for b in it.siblings[:4]:
            key = (it.group, b, it.answer)  # answer b, and it.answer among its near siblings
            cands = index.get(key, [])
            k = cursor[key]
            while k < len(cands) and cands[k].flip_partner is not None:
                k += 1
            cursor[key] = k
            partner = cands[k] if k < len(cands) and cands[k] is not it else None
            if partner is not None:
                it.conflict_claim, partner.conflict_claim = b, it.answer
                it.flip_partner, partner.flip_partner = partner.gib_id, it.gib_id
                pairs += 1
                break
        if it.conflict_claim is None:
            it.conflict_claim = it.siblings[0]
    return pairs


def build(out_dir: str | Path, cfg: Dict[str, Any], coco_root: str | Path,
          splits: Sequence[str] = ("val2017",), vg_root: Optional[str | Path] = None,
          fresh: Optional[Tuple[str, str]] = None, source: str = "coco") -> List[BaseItem]:
    """Build ``items.jsonl`` + ``build_report.json`` under ``out_dir``.

    ``fresh`` = (annotation_file, image_dir) for an extra COCO-format set of
    images newer than the models' training data (contamination control).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_root = Path(coco_root)
    rng = random.Random(cfg["seed"])

    sources: List[CocoSplit] = []
    for sp in splits:
        sources.append(CocoSplit(coco_root / "annotations" / f"instances_{sp}.json", sp, sp,
                                 coco_root / "annotations" / f"captions_{sp}.json"))
    if fresh:
        ann, img_dir = fresh
        sources.append(CocoSplit(Path(ann), img_dir, "fresh"))

    report: Dict[str, Any] = {"splits": [s.split for s in sources], "candidates": {}, "selected": {}}
    pools: Dict[str, List[Dict[str, Any]]] = {"object": [], "attribute": [], "relation": []}
    by_split = {s.split: s for s in sources}
    # Visual Genome is parsed once for all COCO splits (it is ~1.2 GB of JSON); COCO image ids are
    # unique across splits, and post-cutoff "fresh" photos are not in Visual Genome at all
    coco_ids = [i for cs in sources if cs.split != "fresh" for i in cs.images]
    vg = VisualGenome(Path(vg_root), coco_ids) if vg_root else None
    for cs in sources:
        cands = _object_candidates(cs, cfg)
        if vg is not None and cs.split != "fresh":
            cands += _attribute_candidates(cs, vg, cfg) + _spatial_candidates(cs, vg, cfg)
        cands += _leftright_candidates(cs, cfg)
        for c in cands:
            c["split"] = cs.split
            pools[c["qa_type"]].append(c)
        report["candidates"][cs.split] = dict(Counter(c["qa_type"] for c in cands))

    used: Counter = Counter()
    per_split = max(1, len(sources))
    quotas, cap = cfg["build"]["per_type"], cfg["build"]["max_items_per_image"]
    # Types compete for images (an image holds at most `cap` items). The type with the fewest distinct
    # candidate images per needed item goes first -- colour and spatial questions exist only on the ~40k
    # Visual Genome images that also carry candidates of the other types -- and images are filled in
    # rounds (at most 1 item each, then 2, ... cap), so no type crowds the others out of shared images.
    # On the real COCO + VG annotations this fills all 200,000 items with cap 4 (cap 3: ~186k).
    images = {t: len({(c["split"], c["image_id"]) for c in pools[t]}) for t in quotas}
    # A type with a max_answer_share cycles over answers, not groups, and stops taking an answer at its
    # share of the quota: otherwise always saying "on" scores ~69% on relation questions. It goes first:
    # it needs nearly all of its rare answers, which other types would crowd out of full images.
    room = {t: Counter({a: int(share * quotas[t]) for a in {c["answer"] for c in pools[t]}})
            for t, share in cfg["build"].get("max_answer_share", {}).items() if t in quotas}
    order = sorted(quotas, key=lambda t: (t not in room, images[t] / max(1, quotas[t]), t))
    picked: Dict[str, List[Dict[str, Any]]] = {t: [] for t in quotas}
    taken: set = set()
    for level in range(1, cap + 1):
        for qa_type in order:
            quota, pool = quotas[qa_type], pools[qa_type]
            for sp in list(by_split) + [None]:  # half the quota from each split (seen / unseen), then a top-up
                have = len(picked[qa_type]) if sp is None else sum(1 for c in picked[qa_type] if c["split"] == sp)
                want = min((quota if sp is None else quota // per_split) - have, quota - len(picked[qa_type]))
                if want <= 0:
                    continue
                rest = [c for c in pool if id(c) not in taken and (sp is None or c["split"] == sp)]
                new = _round_robin(rest, want, rng, used, level, "answer" if qa_type in room else "group",
                                   room.get(qa_type))
                taken.update(id(c) for c in new)
                picked[qa_type] += new
    for qa_type in room:  # too few other answers (small builds): the quota comes first
        want = quotas[qa_type] - len(picked[qa_type])
        if want > 0:
            rest = [c for c in pools[qa_type] if id(c) not in taken]
            new = _round_robin(rest, want, rng, used, cap, "answer")
            picked[qa_type] += new
            if new:
                report.setdefault("warnings", []).append(
                    f"{qa_type}: {len(new)} items exceed max_answer_share, to reach the quota")
    chosen: List[Dict[str, Any]] = []
    report["type_order"] = order
    for qa_type in order:
        quota = quotas[qa_type]
        chosen += picked[qa_type]
        report["selected"][qa_type] = len(picked[qa_type])
        if len(picked[qa_type]) < quota:
            report.setdefault("warnings", []).append(
                f"{qa_type}: only {len(picked[qa_type])} of {quota} requested items available"
                + (" (attribute/spatial questions need --vg-root)" if qa_type != "object" and not vg_root else ""))

    items: List[BaseItem] = []
    for c in chosen:
        cs = by_split[c["split"]]
        im = cs.images[c["image_id"]]
        present = cs.present(c["image_id"])
        pos = c["pope_pos"] or (max(cs.objects(c["image_id"]), key=lambda o: o[3] * o[4])[0]
                               if cs.objects(c["image_id"]) else None)
        items.append(BaseItem(
            gib_id=f"gib-{c['split']}-{c['image_id']:012d}-{c['qa_type'][:3]}-{c['group'].replace(' ', '')[:6]}",
            image_path=f"{cs.image_dir}/{im['file_name']}",
            source=source if c["split"] != "fresh" else "fresh",
            source_split=c["split"], qa_type=c["qa_type"], group=c["group"],
            question=c["question"], answer=c["answer"], siblings=c["siblings"],
            doc_templates=c["templates"], target_bbox=c["bbox"],
            image_size=[im["width"], im["height"]], objects=cs.objects(c["image_id"]),
            pope_pos=pos, pope_neg=_pope_negative(cs, pos, present, rng),
            slots=c["slots"],
            provenance={"coco_image_id": c["image_id"], "file_name": im["file_name"],
                        "coco_url": im.get("coco_url"), "flickr_url": im.get("flickr_url"),
                        "license": im.get("license"), **c["prov"]},
        ))
    seen: Counter = Counter()
    for it in items:  # several items per image are possible when max_items_per_image > 1
        seen[it.gib_id] += 1
        if seen[it.gib_id] > 1:
            it.gib_id = f"{it.gib_id}-{seen[it.gib_id]}"
    items.sort(key=lambda it: it.gib_id)
    report["flip_pairs"] = pair_conflicts(items)
    report["n_items"] = len(items)
    report["by_type"] = dict(Counter(it.qa_type for it in items))
    report["answer_share"] = {  # e.g. {"relation": {"on": 0.3, ...}}: the score of always giving the top answer
        t: {a: round(n / report["by_type"][t], 4)
            for a, n in Counter(it.answer for it in items if it.qa_type == t).most_common()}
        for t in report["by_type"]}
    report["by_group"] = dict(Counter(it.group for it in items))
    report["by_split"] = dict(Counter(it.source_split for it in items))
    per_image = Counter(Counter(it.image_path for it in items).values())
    report["images"] = sum(per_image.values())
    report["items_per_image"] = {str(k): per_image[k] for k in sorted(per_image)}  # e.g. {"1": n images with 1 item}
    write_jsonl(out_dir / "items.jsonl", items)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return items
