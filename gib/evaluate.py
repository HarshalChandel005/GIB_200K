"""Run one model over a subset: resumable, grouped by image.

Queries are grouped by ``image_key`` so each image is rendered (and, for open
models, encoded) once. Predictions are appended to
``runs/<model>/predictions.jsonl`` as they are produced; re-running the same
command skips every query that already has a prediction.

Adapters that are safe to call from several threads (the API adapters) set
``parallel_groups``; that many image groups are then rendered and run at once,
so the adapter's own bounded pool stays busy. Predictions are still written by
this thread only.

``runs/<model>/meta.json`` (model, versions, quantisation, self-check results,
progress) is written when a run starts and kept current while it runs, so even a
run that crashes or is stopped leaves a record (``status``: running / finished /
interrupted); ``history`` has one entry per run of the same command.

Scale: at 200k items a model answers ~22M queries. Items are expanded one at a
time (never the whole query list in memory), and a resumed run remembers what is
done as sorted 64-bit hashes of the query ids (~8 bytes each instead of a set of
strings, ~150 bytes each).
"""

from __future__ import annotations

import array
import hashlib
import itertools
import json
import re
import time
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .axes.degrade import render
from .axes.expand import expand_item
from .build.subset import read_subset
from .config import config_hash
from .schema import BaseItem, Prediction, Query, append_jsonl, read_jsonl

GROUP_CHUNK = 256  # bounds memory and lost work per call; a chunk re-encodes its image once
META_EVERY = 60    # seconds between meta.json updates during a run
_QID = re.compile(r'"qid": "([^"]*)"')  # predictions are written with sorted keys and default separators


def qid_hashes(qids: Sequence[str]) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.blake2b(q.encode(), digest_size=8).digest(), "little")
                     for q in qids], dtype=np.uint64)


def done_hashes(pred_path: Path) -> np.ndarray:
    """Sorted hashes of every qid already in predictions.jsonl."""
    out = array.array("Q")
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                m = _QID.search(line)
                qid = m.group(1) if m else (json.loads(line)["qid"] if line.strip() else None)
                if qid is not None:
                    out.append(int.from_bytes(hashlib.blake2b(qid.encode(), digest_size=8).digest(), "little"))
    return np.unique(np.frombuffer(out, dtype=np.uint64)) if len(out) else np.zeros(0, np.uint64)


def _contains(done: np.ndarray, h: np.ndarray) -> np.ndarray:
    i = np.searchsorted(done, h)
    return (i < len(done)) & (done[np.minimum(i, len(done) - 1)] == h)


def iter_groups(items: Iterable[BaseItem], cfg: Dict[str, Any], done: Optional[np.ndarray] = None,
                on_item: Optional[Callable[[int], None]] = None) -> Iterator[Tuple[str, List[Query]]]:
    """Image groups, item by item, skipping queries already done. The blank-image probes of all
    items share one image, so they are pooled across items into chunks of GROUP_CHUNK."""
    blank: List[Query] = []
    for it in items:
        qs = expand_item(it, cfg)
        n_all = len(qs)
        if done is not None and done.size:
            qs = [q for q, d in zip(qs, _contains(done, qid_hashes([q.qid for q in qs]))) if not d]
        groups: "OrderedDict[str, List[Query]]" = OrderedDict()
        for q in qs:
            if q.image_key == "blank":
                blank.append(q)
            else:
                groups.setdefault(q.image_key, []).append(q)
        yield from groups.items()
        while len(blank) >= GROUP_CHUNK:
            yield "blank", blank[:GROUP_CHUNK]
            del blank[:GROUP_CHUNK]
        if on_item is not None:
            on_item(n_all)
    if blank:
        yield "blank", blank


def load_items(data_dir: str | Path, subset: Optional[str] = None) -> List[BaseItem]:
    data_dir = Path(data_dir)
    items = read_jsonl(data_dir / "items.jsonl", BaseItem)
    if subset and subset != "all":
        keep = set(read_subset(data_dir / "subsets" / f"{subset}.txt"))
        items = [it for it in items if it.gib_id in keep]
    return items


def group_queries(queries: Iterable[Query]) -> "OrderedDict[str, List[Query]]":
    groups: "OrderedDict[str, List[Query]]" = OrderedDict()
    blank: List[Query] = []
    for q in queries:
        if q.image_key == "blank":
            blank.append(q)
        else:
            groups.setdefault(q.image_key, []).append(q)
    if blank:
        groups["blank"] = blank  # one shared prefix for every item's blank-image probes
    return groups


def count_work(items: Iterable[BaseItem], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Queries and image encodings for a set of items, counted one item at a time."""
    n = n_enc = n_blank = 0
    by_probe: Counter = Counter()
    for it in items:
        keys = set()
        for q in expand_item(it, cfg):
            n += 1
            by_probe[q.probe] += 1
            if q.image_key == "blank":
                n_blank += 1
            else:
                keys.add(q.image_key)
        n_enc += len(keys)
    n_enc += (n_blank + GROUP_CHUNK - 1) // GROUP_CHUNK
    return {"queries": n, "image_encodings": n_enc, "by_probe": dict(by_probe)}


def evaluate(adapter, items: List[BaseItem], cfg: Dict[str, Any], image_root: str | Path,
             run_dir: str | Path, *, limit_groups: Optional[int] = None, progress: bool = True) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    by_id = {it.gib_id: it for it in items}
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(items), desc=adapter.name, unit="item")
        except ImportError:
            bar = None
    seen = {"items": 0, "queries": 0}

    def on_item(n_queries: int) -> None:
        seen["items"] += 1
        seen["queries"] += n_queries
        if bar is not None:
            bar.update(1)

    todo: Iterable[Tuple[str, List[Query]]] = iter_groups(items, cfg, done_hashes(pred_path), on_item)
    if limit_groups is not None:
        todo = itertools.islice(todo, limit_groups)
    with_image = getattr(adapter, "needs_image", True)  # the simulator ignores pixels: skip rendering

    def run(key: str, qs: List[Query]) -> List[List[Prediction]]:
        q0 = qs[0]
        item = None if key == "blank" else by_id[q0.gib_id]
        image = render(item, q0.image_recipe, image_root, cfg) if with_image else None
        out = []
        for i in range(0, len(qs), GROUP_CHUNK):
            chunk = qs[i:i + GROUP_CHUNK]
            try:
                out.append(adapter.run_group(image, chunk))
            except Exception as e:  # keep going; failed queries are retried on the next run
                out.append([Prediction(qid=q.qid, model=adapter.name, error=f"{type(e).__name__}: {e}"[:300])
                            for q in chunk])
        return out

    meta_path = run_dir / "meta.json"
    history = json.loads(meta_path.read_text(encoding="utf-8")).get("history", []) if meta_path.exists() else []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    n_new = n_err = n_saved = 0
    last_meta = 0.0

    def write_meta(status: str, error: Optional[str] = None) -> Dict[str, Any]:
        """meta.json is written at the start, after the first group (the adapter's self-checks have
        run by then), every META_EVERY seconds, and at the end -- also when the run stops early."""
        nonlocal last_meta
        meta = {"model": adapter.name, "status": status, "adapter": adapter.meta(), "config_hash": config_hash(cfg),
                "n_items": len(items), "items_seen": seen["items"], "n_queries": seen["queries"], "started": started,
                "new_predictions": n_new, "errors_this_run": n_err, "wall_seconds": round(time.time() - t0, 1),
                "reproducible": getattr(adapter, "reproducible", True)}
        if error:
            meta["error"] = error
        meta["history"] = history + [{k: meta[k] for k in ("started", "status", "new_predictions",
                                                           "errors_this_run", "wall_seconds")}]
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(meta_path)  # atomic: a run killed mid-write never leaves a half-written file
        last_meta = time.time()
        return meta

    def save(chunks: List[List[Prediction]]) -> None:
        nonlocal n_new, n_err, n_saved
        for preds in chunks:
            good = [p for p in preds if p.error is None]
            n_err += len(preds) - len(good)
            append_jsonl(pred_path, good)
            if len(good) < len(preds):
                append_jsonl(run_dir / "errors.jsonl", [p for p in preds if p.error is not None])
            n_new += len(good)
        n_saved += 1
        if n_saved == 1 or time.time() - last_meta > META_EVERY:
            write_meta("running")

    write_meta("running")
    n_par = max(1, int(getattr(adapter, "parallel_groups", 1)))
    try:
        if n_par == 1:
            for key, qs in todo:
                save(run(key, qs))
        else:
            with ThreadPoolExecutor(n_par) as ex:
                pending = set()
                for key, qs in todo:  # a bounded window, so finished groups are written as they come
                    pending.add(ex.submit(run, key, qs))
                    if len(pending) >= 2 * n_par:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in finished:
                            save(f.result())
                for f in as_completed(pending):
                    save(f.result())
    except BaseException as e:  # Ctrl-C or a crash: record how far the run got, then stop as before
        try:
            write_meta("interrupted", f"{type(e).__name__}: {e}"[:300])
        except Exception:
            pass
        raise
    finally:
        if bar is not None:
            bar.close()
    return write_meta("finished")
