"""Validate the rule-based failure taxonomy against human labels (Cohen's kappa).

``export`` samples wrong free-form answers stratified by the automatic label and
writes ``to_label.csv`` *without* that label (blind labelling) plus a separate
``key.csv``. Annotators fill the ``human_label`` column with one of the four
class names. ``kappa`` joins the two files and reports agreement.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..axes.degrade import materialize
from ..axes.expand import expand_item
from ..eval.taxonomy import CLASSES, classify_frame
from ..schema import BaseItem


def export(provenance: pd.DataFrame, items: Sequence[BaseItem], cfg: Dict[str, Any], image_root: str | Path,
           out_dir: str | Path, n: int = 200, seed: int = 0) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    qa = provenance[provenance["probe"] == "qa"].copy()
    qa["auto_label"] = qa["failure"].astype(object) if "failure" in qa else classify_frame(qa)
    if "qid" not in qa:  # the lean table drops qids; a free-form query's is gib_id|condition|qa
        qa["qid"] = qa["gib_id"].astype(str) + "|" + qa["condition"].astype(str) + "|qa"
    qa = qa.astype({c: object for c in ("model", "gib_id", "text") if c in qa})
    wrong = qa[qa["auto_label"].notna()]
    rng = random.Random(f"tax:{seed}")
    per = max(1, n // max(1, wrong["auto_label"].nunique()))
    picked = []
    for lab, sub in sorted(wrong.groupby("auto_label"), key=lambda x: x[0]):
        idx = list(sub.index)
        rng.shuffle(idx)
        picked += idx[:per]
    sample = wrong.loc[picked].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    by_id = {it.gib_id: it for it in items}
    rows, key = [], []
    for i, r in sample.iterrows():
        it = by_id[r["gib_id"]]
        q = next(q for q in expand_item(it, cfg) if q.qid == r["qid"])
        img = materialize(it, q.image_recipe, image_root, cfg, out_dir / "img")
        rid = f"t{i:04d}"
        rows.append({"id": rid, "image": f"img/{img.name}", "question": q.question,
                     "documents": " || ".join(q.docs), "correct_answer": it.answer,
                     "model_answer": r["text"], "human_label": ""})
        key.append({"id": rid, "model": r["model"], "qid": r["qid"], "auto_label": r["auto_label"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "to_label.csv", index=False)
    pd.DataFrame(key).to_csv(out_dir / "key.csv", index=False)
    (out_dir / "LABELS.txt").write_text(
        "Fill human_label with one of:\n" + "\n".join(f"  {c}" for c in CLASSES) +
        "\n\nKnowledge Inertia: the answer repeats a wrong claim from the documents.\n"
        "Sensory Capture: the documents had the right answer; the model gave a different (valid) answer.\n"
        "Reasoning Collapse: no real answer, an unrelated answer, or one that matches neither image nor documents.\n"
        "Perceptual Error: no documents were shown and the answer is simply wrong.\n", encoding="utf-8")
    return {"n": len(rows), "out": str(out_dir)}


def kappa(labels_csv: str | Path, key_csv: str | Path) -> Dict[str, Any]:
    lab = pd.read_csv(labels_csv)
    key = pd.read_csv(key_csv)
    m = lab.merge(key, on="id")
    m = m[m["human_label"].isin(CLASSES)]
    if m.empty:
        return {"n": 0}
    a, b = m["auto_label"].to_numpy(), m["human_label"].to_numpy()
    cats = list(CLASSES)
    conf = pd.crosstab(pd.Categorical(b, cats), pd.Categorical(a, cats), rownames=["human"], colnames=["auto"],
                       dropna=False)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    per_class = {}
    for c in cats:
        tp = float(((a == c) & (b == c)).sum())
        per_class[c] = {"precision": tp / max(1.0, float((a == c).sum())),
                        "recall": tp / max(1.0, float((b == c).sum())), "n_human": int((b == c).sum())}
    return {"n": int(len(m)), "agreement": po, "cohen_kappa": float(k),
            "confusion": {str(r): {str(c): int(conf.loc[r, c]) for c in conf.columns} for r in conf.index},
            "per_class": per_class}
