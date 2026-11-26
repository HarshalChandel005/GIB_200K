"""The provenance table: one row per (model, query). Single source of truth.

Every metric, statistic, figure and paper number is computed from this table,
and the audit recomputes them from the files on disk: ``provenance/<model>.csv.gz``
in the report directory, one per model.

At 200k items a model answers ~22M queries, so the table is lean: scored values
are derived while the predictions are read (in chunks), small-vocabulary columns
are categorical, numbers are float32, and per-row lists are dropped once used.
Only the per-layer attention of the free-form clean / conflict answers is kept
(the attention-shift analysis needs it). A row costs ~100 bytes in memory.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from .. import vocab
from ..axes.degrade import visible_objects
from ..axes.expand import expand_item
from ..schema import BaseItem, Query
from .taxonomy import classify

CAT_COLS = ["model", "gib_id", "condition", "axis", "probe", "image_tag", "qa_type", "group", "source_split",
            "image_truth", "text_claim", "context", "family", "polarity", "text", "extracted", "failure", "backend"]
NUM_COLS = ["k", "value", "sigma", "order", "correct", "off_vocab", "follows_claim", "copied_wrong_doc",
            "p_truth", "p_claim", "choice_conf", "choice_correct", "p_yes", "has_probs", "from_logprobs",
            "format_mass", "n_mentions", "n_halluc", "vis_share", "n_img_tokens", "n_tokens", "latency_ms"]
LIST_COLS = ["attn_img", "attn_post"]
COLUMNS = CAT_COLS + NUM_COLS + ["reproducible"] + LIST_COLS
ATTN_CONDITIONS = ("clean", "conflict")  # free-form answers whose per-layer attention is kept
PROB_PROBES = ("mc", "ab", "pope")
CHUNK_ROWS = 50_000  # rows per chunk while building, writing and reading (raw rows are ~2.5 KB each)
NAN = float("nan")
_extract = lru_cache(maxsize=500_000)(vocab.extract_answer)  # free-form answers repeat a lot ("left", "dog")


def _p_of(probs: Optional[List[float]], options: List[str], target: Optional[str]) -> float:
    if not probs or target is None or target not in options:
        return NAN
    return float(probs[options.index(target)])


def _f(x: Any) -> float:
    return NAN if x is None else float(x)


def _row(p: Dict[str, Any], q: Query, it: BaseItem, backend: Optional[str],
         visible: Callable[[str, str], List[str]]) -> Dict[str, Any]:
    """The scored, lean row for one prediction (a pure function of the prediction, query and item)."""
    probe, probs, text = q.probe, p.get("option_probs"), p.get("text")
    extra = p.get("extra") or {}
    tag = q.image_key.rsplit("|", 1)[-1]  # = recipe_tag(q.image_recipe): "<gib_id>|<tag>", or "blank"
    r: Dict[str, Any] = {
        "model": p["model"], "gib_id": q.gib_id, "condition": q.condition, "axis": q.axis, "probe": probe,
        "image_tag": tag, "qa_type": it.qa_type, "group": it.group, "source_split": it.source_split,
        "image_truth": q.image_truth, "text_claim": q.text_claim, "context": q.meta.get("context"),
        "k": _f(q.level.get("k")), "family": q.level.get("family"), "value": _f(q.level.get("value")),
        "sigma": _f(q.level.get("sigma")), "order": _f(q.meta.get("order")), "polarity": q.meta.get("polarity"),
        "text": text if probe in ("qa", "caption") else None,
        "n_img_tokens": _f(p.get("n_img_tokens")), "n_tokens": _f(p.get("n_tokens")),
        "latency_ms": _f(p.get("latency_ms")), "reproducible": bool(p.get("reproducible", True)),
        "format_mass": _f(extra.get("format_mass")), "backend": backend,
    }
    # runs written before the flag existed: open/simulated probabilities always came from logits
    flag = extra.get("from_logprobs", True if probs is not None else None)
    r["from_logprobs"] = _f(flag)
    r["has_probs"] = float(bool(probs)) if probe in PROB_PROBES else NAN
    if probe == "qa":
        e = _extract(text, it.group)
        r["extracted"] = e
        r["correct"] = float(e == q.image_truth)
        r["off_vocab"] = float(e is None)
        r["follows_claim"] = float(e == q.text_claim) if q.text_claim is not None else NAN
        r["copied_wrong_doc"] = float(e is not None and e != q.image_truth and e in (q.doc_claims or []))
        r["failure"] = classify(e, q.image_truth, q.doc_claims or [])
    elif probe in ("mc", "ab"):
        r["p_truth"] = _p_of(probs, q.options, q.image_truth)
        r["p_claim"] = _p_of(probs, q.options, q.text_claim)
        r["choice_conf"] = float(max(probs)) if probs else NAN
        r["choice_correct"] = float(q.options[int(np.argmax(probs))] == q.image_truth) if probs else NAN
    elif probe == "pope":
        r["p_yes"] = float(probs[0]) if probs else NAN
    elif probe == "caption":
        m = vocab.chair_mentions(text)
        vis = visible(q.gib_id, tag)
        r["n_mentions"], r["n_halluc"] = float(len(m)), float(sum(1 for x in m if x not in vis))
    a, b = p.get("attn_img"), p.get("attn_post")
    r["vis_share"] = float(np.mean([i / (i + j) for i, j in zip(a, b) if (i + j) > 0])) if (a and b) else NAN
    if probe == "qa" and q.condition in ATTN_CONDITIONS and a and b:
        r["attn_img"], r["attn_post"] = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return r


def _columns(df: pd.DataFrame) -> Dict[str, Any]:
    """A chunk as independent per-column arrays, so the chunks can be merged and freed column by column
    (a DataFrame keeps same-typed columns in one shared block, which frees only when every column goes)."""
    return {c: df[c].array if isinstance(df[c].dtype, pd.CategoricalDtype) else df[c].to_numpy(copy=True)
            for c in df.columns}


def _frame(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    for c in NUM_COLS:
        df[c] = df[c].astype(np.float32)
    for c in LIST_COLS:  # None, not NaN, where there is no list (also in a chunk without any)
        df[c] = pd.Series([v if isinstance(v, np.ndarray) else None for v in df[c]], index=df.index, dtype=object)
    df["reproducible"] = df["reproducible"].astype(bool)
    return _columns(df)


def _assemble(chunks: List[Dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """One table from per-column chunks, merging a column at a time and dropping its chunk pieces
    as it goes: peak memory is the table plus one column, not two tables."""
    if not chunks:
        return pd.DataFrame(columns=list(columns))
    cols: Dict[str, Any] = {}
    for c in list(chunks[0]):
        parts = [ch.pop(c) for ch in chunks]
        if len(parts) == 1:
            cols[c] = parts[0]
        elif isinstance(parts[0], pd.Categorical):
            cols[c] = _union([pd.Series(p, copy=False) for p in parts])
        else:
            cols[c] = np.concatenate(parts)
        del parts
    return pd.DataFrame(cols, copy=False)


def _union(parts: List[pd.Series]) -> pd.Categorical:
    typed = [p for p in parts if len(p.cat.categories)]
    if len(typed) < len(parts):  # an all-missing chunk has untyped empty categories: use the others' type
        empty = pd.Index([], dtype=typed[0].cat.categories.dtype if typed else object)
        parts = [p if len(p.cat.categories) else
                 pd.Series(pd.Categorical.from_codes(np.full(len(p), -1), categories=empty)) for p in parts]
    return union_categoricals(parts)


def _concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chunks, keeping categorical columns categorical (a plain concat would turn them
    into one Python string per row)."""
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    out = {}
    for c in frames[0].columns:
        if isinstance(frames[0][c].dtype, pd.CategoricalDtype):
            out[c] = _union([f[c] for f in frames])
        else:
            out[c] = pd.concat([f[c] for f in frames], ignore_index=True)
    return pd.DataFrame(out)


def build_table(items: Iterable[BaseItem], cfg: Dict[str, Any], run_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """Read the predictions of ``run_dirs`` for ``items`` into one lean table (chunk by chunk)."""
    by_item: Dict[str, BaseItem] = {it.gib_id: it for it in items}

    @lru_cache(maxsize=1024)  # predictions come item by item; blank-image probes trail by <= ~64 items
    def queries_of(gib_id: str) -> Dict[str, Query]:
        return {q.qid: q for q in expand_item(by_item[gib_id], cfg)}

    @lru_cache(maxsize=65536)
    def visible(gib_id: str, tag: str) -> List[str]:
        return visible_objects(by_item[gib_id], _recipe_from_tag(tag))

    frames: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        rd = Path(rd)
        path = rd / "predictions.jsonl"
        if not path.exists():
            continue
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8")) if (rd / "meta.json").exists() else {}
        backend = (meta.get("adapter") or {}).get("backend")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                gib = p["qid"].split("|", 1)[0]
                if gib not in by_item:
                    continue
                q = queries_of(gib).get(p["qid"])
                if q is None:
                    continue
                rows.append(_row(p, q, by_item[gib], backend, visible))
                if len(rows) >= CHUNK_ROWS:
                    frames.append(_frame(rows))
                    rows = []
    if rows:
        frames.append(_frame(rows))
    return _assemble(frames, COLUMNS)


def _recipe_from_tag(tag: str) -> Dict:
    if tag == "orig":
        return {"method": "none"}
    if tag == "blank":
        return {"method": "blank"}
    fam, val = tag.split("_", 1)
    v = float(val[1:])
    key = {"gauss": "sigma", "motion": "kernel", "crop": "frac", "noise": "std"}[fam]
    return {"method": fam, key: v}


def _encode_lists(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    for c in LIST_COLS:
        if c in part:
            part[c] = [json.dumps(np.asarray(v).tolist()) if isinstance(v, (list, tuple, np.ndarray)) else None
                       for v in part[c]]
    return part


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write in chunks (no full copy of a 22M-row table); ``.gz`` paths are gzip-compressed (fast level)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = (lambda: gzip.open(path, "wt", compresslevel=1, encoding="utf-8", newline="")) if path.suffix == ".gz" \
        else (lambda: path.open("w", encoding="utf-8", newline=""))
    with opener() as f:
        for i in range(0, max(1, len(df)), CHUNK_ROWS):
            _encode_lists(df.iloc[i:i + CHUNK_ROWS]).to_csv(f, index=False, header=(i == 0))


def load_table(path: str | Path, keep: Optional[Callable[[pd.DataFrame], Any]] = None,
               columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read one provenance file back with the in-memory dtypes; ``keep`` filters rows chunk by chunk."""
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in (columns or head) if c in head]
    dtypes = {c: "category" for c in CAT_COLS if c in use}
    dtypes.update({c: np.float32 for c in NUM_COLS if c in use})
    dtypes.update({c: object for c in LIST_COLS if c in use})
    frames = []
    for part in pd.read_csv(path, usecols=use, dtype=dtypes, chunksize=CHUNK_ROWS):
        if keep is not None:
            part = part[np.asarray(keep(part), bool)]
        for c in LIST_COLS:
            if c in part:
                part[c] = [np.asarray(json.loads(v), np.float32) if isinstance(v, str) and v.startswith("[") else None
                           for v in part[c]]
        if "reproducible" in part:
            part["reproducible"] = part["reproducible"].astype(str).str.lower().eq("true")
        frames.append(_columns(part))
        del part
    return _assemble(frames, use)


def provenance_files(report_dir: str | Path) -> List[Path]:
    report_dir = Path(report_dir)
    return sorted((report_dir / "provenance").glob("*.csv.gz"))


def load_rows(report_dir: str | Path, keep: Optional[Callable[[pd.DataFrame], Any]] = None,
              columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Rows of every model in a report (optionally filtered / only some columns), as one table."""
    return _concat([load_table(f, keep, columns) for f in provenance_files(report_dir)])
