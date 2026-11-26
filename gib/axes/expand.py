"""Expand one BaseItem into every query of the benchmark.

Per item (defaults; all levels come from the config):

  QA conditions (each asked as a free-form question + a multiple-choice probe)
    clean                  original image + gold document
    no_ctx                 original image, no document         (non-RAG baseline)
    ret_k{1,3,5,10,20}     gold + k hard-negative documents    (Axis 1)
    vis_*  (13 levels)     degraded image, no document         (Axis 2, pure perception)
    conflict[_g4,_g16]     document asserts a sibling answer   (Axis 3, also on blurred images)
    grid_g{4,16}_k{0,5,20} blurred image + gold + k distractors (Axis 1 x Axis 2 interaction;
                           k=0 is the Sensory-Capture probe: correct text, degraded image)

  Binary A/B probes (image truth vs conflict claim, both option orders) for the
  cue-combination analysis: image-only, text-only (blank image), and both, at
  three blur levels; plus POPE yes/no probes on every image condition and,
  for a fixed fraction of items, caption probes for CHAIR.

Hard negatives are built the way the brief asks: the gold document with its
answer slot swapped for a semantically near sibling (same COCO group, nearest
colour, antonym predicate).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import vocab
from ..schema import BaseItem, Probe, Query
from .degrade import recipe_tag, visual_recipes

ORIG = {"method": "none"}
BLANK = {"method": "blank"}


def gauss(s: float) -> Dict[str, Any]:
    return {"method": "gauss", "sigma": s}


def gold_doc(item: BaseItem) -> str:
    return vocab.fill(item.doc_templates[0], item.answer, item.slots)


def conflict_doc(item: BaseItem) -> str:
    assert item.conflict_claim is not None
    return vocab.fill(item.doc_templates[0], item.conflict_claim, item.slots)


def distractor_docs(item: BaseItem, k: int, rng: random.Random) -> Tuple[List[str], List[str]]:
    """k hard negatives: the gold template set with the answer swapped."""
    negs = [s for s in item.siblings[:4]] or [item.siblings[0]]
    T = len(item.doc_templates)
    docs, claims = [], []
    for j in range(k):
        claim = negs[j % len(negs)]
        tmpl = item.doc_templates[(j + 1) % T] if T > 1 else item.doc_templates[0]
        doc = vocab.fill(tmpl, claim, item.slots)
        if doc in docs:  # same template + claim again -> vary the template
            doc = vocab.fill(item.doc_templates[(j + 1 + j // len(negs)) % T], claim, item.slots)
        docs.append(doc)
        claims.append(claim)
    return docs, claims


def _with_gold(item: BaseItem, docs: List[str], rng: random.Random) -> List[str]:
    docs = list(docs)
    rng.shuffle(docs)
    pos = rng.randint(0, len(docs))
    return docs[:pos] + [gold_doc(item)] + docs[pos:]


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(":".join(str(p) for p in (seed, *parts)))


def option_order(item: BaseItem, seed: int) -> List[str]:
    opts = list(item.options)
    _rng(seed, item.gib_id, "options").shuffle(opts)
    return opts


def has_caption_probe(item: BaseItem, frac: float) -> bool:
    h = int(hashlib.sha256(item.gib_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


def qa_conditions(item: BaseItem, cfg: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any], str, int, Dict[str, Any]]]:
    """(condition, axis, recipe, context, k, level) for every QA condition."""
    ax = cfg["axes"]
    out: List[Tuple[str, str, Dict[str, Any], str, int, Dict[str, Any]]] = [
        ("clean", "clean", ORIG, "gold", 0, {}),
        ("no_ctx", "no_ctx", ORIG, "none", 0, {}),
    ]
    for k in ax["retrieval"]["dosages"]:
        out.append((f"ret_k{k}", "retrieval", ORIG, "distract", k, {"k": k}))
    for r in visual_recipes(cfg):
        fam = r["method"]
        val = r.get("sigma", r.get("kernel", r.get("frac", r.get("std"))))
        out.append((f"vis_{recipe_tag(r)}", "visual", r, "none", 0, {"family": fam, "value": val}))
    out.append(("conflict", "conflict", ORIG, "conflict", 0, {"sigma": 0}))
    for s in ax["conflict"]["sigmas"]:
        out.append((f"conflict_g{s:g}", "conflict", gauss(s), "conflict", 0, {"sigma": s}))
    for s in ax["grid"]["sigmas"]:
        for k in ax["grid"]["dosages"]:
            out.append((f"grid_g{s:g}_k{k}", "grid", gauss(s), "gold" if k == 0 else "distract", k,
                        {"sigma": s, "k": k}))
    return out


def _docs_for(item: BaseItem, context: str, k: int, rng: random.Random) -> Tuple[List[str], List[str]]:
    if context == "none":
        return [], []
    if context == "gold":
        return [gold_doc(item)], [item.answer]
    if context == "conflict":
        return [conflict_doc(item)], [item.conflict_claim]  # type: ignore[list-item]
    if context == "distract":
        negs, claims = distractor_docs(item, k, rng)
        return _with_gold(item, negs, rng), [item.answer] + claims
    raise ValueError(context)


def expand_item(item: BaseItem, cfg: Dict[str, Any]) -> List[Query]:
    seed = cfg["seed"]
    probes = cfg["probes"]
    qs: List[Query] = []
    mc_opts = option_order(item, seed)

    def key(recipe: Dict[str, Any]) -> str:
        return "blank" if recipe.get("method") == "blank" else f"{item.gib_id}|{recipe_tag(recipe)}"

    def add(cond: str, axis: str, probe: Probe, recipe: Dict[str, Any], docs: List[str],
            question: str, *, options=(), text_claim=None, doc_claims=(), level=None, meta=None,
            suffix: str = "") -> None:
        qid = f"{item.gib_id}|{cond}|{probe.value}" + (f"|{suffix}" if suffix else "")
        qs.append(Query(qid=qid, gib_id=item.gib_id, condition=cond, axis=axis, probe=probe.value,
                        image_key=key(recipe), image_recipe=dict(recipe), docs=list(docs),
                        question=question, options=list(options), image_truth=item.answer,
                        text_claim=text_claim, doc_claims=list(doc_claims), level=dict(level or {}),
                        meta=dict(meta or {})))

    # ---- QA conditions: free-form + multiple choice ----------------------- #
    for cond, axis, recipe, context, k, level in qa_conditions(item, cfg):
        docs, claims = _docs_for(item, context, k, _rng(seed, item.gib_id, cond))
        claim = item.conflict_claim if context == "conflict" else None
        meta = {"context": context}
        add(cond, axis, Probe.QA, recipe, docs, item.question, text_claim=claim,
            doc_claims=claims, level=level, meta=meta)
        if probes.get("mc", True):
            add(cond, axis, Probe.MC, recipe, docs, item.question, options=mc_opts,
                text_claim=claim, doc_claims=claims, level=level, meta=meta)

    # ---- A/B probes for the Trust Index / cue-combination model ----------- #
    if probes.get("ab", True) and item.conflict_claim:
        pair = [item.answer, item.conflict_claim]
        sigmas = [0] + list(cfg["axes"]["conflict"]["sigmas"])
        ab_conds: List[Tuple[str, Dict[str, Any], str, float]] = []
        for s in sigmas:
            r = ORIG if s == 0 else gauss(s)
            ab_conds.append((f"ab_img_g{s:g}", r, "none", s))       # image only
            ab_conds.append((f"ab_both_g{s:g}", r, "conflict", s))  # image + misleading text
        ab_conds.append(("ab_clean", ORIG, "gold", 0))              # image + correct text
        ab_conds.append(("ab_prior", BLANK, "none", -1))            # neither cue
        ab_conds.append(("ab_text", BLANK, "conflict", -1))         # text only
        for cond, recipe, context, s in ab_conds:
            docs, claims = _docs_for(item, context, 0, _rng(seed, item.gib_id, cond))
            for order in (0, 1):
                opts = pair if order == 0 else pair[::-1]
                add(cond, "ab", Probe.AB, recipe, docs, item.question, options=opts,
                    text_claim=item.conflict_claim, doc_claims=claims,
                    level={"sigma": s}, meta={"context": context, "order": order}, suffix=f"o{order}")

    # ---- POPE + caption probes on every image condition ------------------- #
    recipes = [ORIG] + visual_recipes(cfg)
    if probes.get("pope", True) and item.pope_pos and item.pope_neg:
        for r in recipes:
            for pol, obj in (("pos", item.pope_pos), ("neg", item.pope_neg)):
                add(f"pope_{recipe_tag(r)}", "pope", Probe.POPE, r, [],
                    f"Is there a {vocab.display(obj)} in the image?",
                    options=["yes", "no"], level={"tag": recipe_tag(r)},
                    meta={"polarity": pol, "object": obj}, suffix=pol)
    if has_caption_probe(item, probes.get("caption_frac", 0.0)):
        for r in recipes:
            add(f"cap_{recipe_tag(r)}", "caption", Probe.CAPTION, r, [],
                "Describe this image in one sentence.", level={"tag": recipe_tag(r)})
    return qs


def expand(items: Iterable[BaseItem], cfg: Dict[str, Any]) -> List[Query]:
    out: List[Query] = []
    for it in items:
        out.extend(expand_item(it, cfg))
    return out
