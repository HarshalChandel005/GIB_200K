"""Compute every metric for every model, plus cross-model statistics.

Outputs (all under the report directory):
  provenance/      per-(model, query) evidence table, one <model>.csv.gz per model
  items.txt        the items the report covers (the audit recomputes on exactly these)
  report.json      every metric with 95% CIs and n
  claims.json      flat headline numbers (what the paper may cite); report.json's
                   ``probability_flags`` lists those resting on too few real token probabilities
  RESULTS.md       auto-generated tables
  macros.tex       LaTeX macros for every claim (refused if any model is simulated)
  figures/*.png    + figures/CAPTIONS.md
  gallery.html     qualitative examples

Models are processed one at a time (a model's table is ~2 GB at 200k items);
only per-item arrays (~MBs) are kept for the cross-model tests.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..axes.degrade import visual_recipes
from ..eval import metrics as M
from ..eval import stats as S
from ..eval.gate import error_signals, gate
from ..eval.table import build_table, save_table
from ..schema import BaseItem


def _families(cfg: Dict[str, Any]) -> Dict[str, List[float]]:
    fam: Dict[str, List[float]] = {}
    for r in visual_recipes(cfg):
        val = r.get("sigma", r.get("kernel", r.get("frac", r.get("std"))))
        fam.setdefault(r["method"], []).append(val)
    return fam


def model_metrics(df: pd.DataFrame, ids: Sequence[str], items: Dict[str, BaseItem], cfg: Dict[str, Any],
                  W: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], M.ModelView]:
    """All metrics for one model, plus the bootstrap draws needed for cross-model tests."""
    v = M.ModelView(df, ids, cfg)
    a = cfg["analysis"]
    rob = M.robustness(v, W)
    out = {
        "n_items": v.n,
        "reproducible": bool(df["reproducible"].astype(str).str.lower().eq("true").all()),
        "accuracy": M.accuracy_by_condition(v, W),
        "robustness": {k: val for k, val in rob.items() if k != "_boot"},
        "trust": M.trust(v, W),
        "cue_combination": M.cue_combination(v, W),
        "flip": M.flip_decomposition(v, items, a["bootstrap_iters"], cfg["seed"]),
        "attention": M.attention_shift(v, W),
        "decay": M.decay(v, W),
        "resolution": M.resolution(v, W, _families(cfg), a["resolution_jump"]),
        "interaction": M.interaction(v, W, cfg["axes"]["grid"]["sigmas"], cfg["axes"]["grid"]["dosages"]),
        "pope": M.pope(v, W),
        "chair": M.chair(v, W),
        "taxonomy": M.taxonomy(v),
        "contamination": M.contamination(v, items, W),
        "gate": gate(v, W),
        "error_signals": error_signals(v),
        "probability_quality": M.probability_quality(df, cfg),
        "latency_ms_mean": float(pd.to_numeric(df["latency_ms"], errors="coerce").mean()),
    }
    return out, rob["_boot"], v


def cross_model(arrays: Dict[str, Dict[str, np.ndarray]], boots: Dict[str, Dict[str, np.ndarray]],
                metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``arrays[model]`` holds the per-item clean and adverse accuracy (aligned item order)."""
    names = sorted(arrays)
    out: Dict[str, Any] = {"models": names}
    if len(names) >= 2:
        for label, key in (("acc_clean", "clean"), ("acc_adverse", "adverse")):
            mat = np.column_stack([arrays[n][key] for n in names])
            out[f"friedman_{label}"] = S.friedman(mat)
            out[f"rm_anova_{label}"] = S.rm_anova(mat)
        pairs, pv = [], []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = boots[a]["adverse"] - boots[b]["adverse"]
                pairs.append({"a": a, "b": b, "diff_acc_adverse": metrics[a]["robustness"]["acc_adverse"]["value"]
                              - metrics[b]["robustness"]["acc_adverse"]["value"], "ci": S.ci(d)})
                pv.append(S.paired_boot_p(d))
        for p, adj, raw in zip(pairs, S.holm(pv) if pv else [], pv):
            p["p"], p["p_holm"] = raw, adj
        out["pairwise_acc_adverse"] = pairs
    pts = [(n, metrics[n]["robustness"]["acc_clean"]["value"], metrics[n]["robustness"]["R"]["value"]) for n in names]
    frontier = [n for n, x, y in pts if not any((x2 >= x and y2 >= y and (x2 > x or y2 > y)) for _, x2, y2 in pts)]
    out["pareto_frontier"] = frontier
    conflict = {}
    for n in names:
        ti = metrics[n]["trust"]["trust_index"]
        lo, hi = ti["ci"]
        label = "Text-biased" if lo > 1 else "Image-biased" if hi < 1 else "Balanced"
        conflict[n] = {"trust_index": ti["value"], "ci": ti["ci"], "label": label,
                       "qa_text_follow_rate": metrics[n]["trust"]["qa_text_follow_rate"]["value"]}
    out["conflict_matrix"] = conflict
    if len(names) >= 4:
        keys = {"acc_clean": lambda m: m["robustness"]["acc_clean"]["value"],
                "R": lambda m: m["robustness"]["R"]["value"],
                "log_TI": lambda m: m["trust"]["trust_index"].get("log_value"),
                "GIS": lambda m: m["trust"]["GIS"]["value"],
                "ADS": lambda m: m["attention"]["ADS"]["value"] if m["attention"].get("available") else None,
                "decay": lambda m: m["decay"]["decay_rate"]["value"] if m["decay"].get("available") else None}
        tab = pd.DataFrame({k: [f(metrics[n]) for n in names] for k, f in keys.items()}, index=names).astype(float)
        out["metric_correlation_across_models_spearman"] = tab.corr(method="spearman").round(3).to_dict()
    return out


def claims_from(report: Dict[str, Any]) -> Dict[str, float]:
    """Flat headline numbers; the paper may only cite these (via macros.tex)."""
    c: Dict[str, float] = {}

    def put(key: str, val: Any) -> None:
        if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
            c[key] = round(float(val), 6)

    for m, r in report["models"].items():
        rob, tr = r["robustness"], r["trust"]
        put(f"{m}.acc_clean", rob["acc_clean"]["value"])
        put(f"{m}.acc_adverse", rob["acc_adverse"]["value"])
        put(f"{m}.ece_adverse", rob["ece_adverse"]["value"])
        put(f"{m}.R", rob["R"]["value"])
        put(f"{m}.trust_index", tr["trust_index"]["value"])
        put(f"{m}.qa_text_follow", tr["qa_text_follow_rate"]["value"])
        put(f"{m}.GIS", tr["GIS"]["value"])
        put(f"{m}.illusion_gap", tr["illusion_gap"]["value"])
        cc = r["cue_combination"]
        if cc.get("available"):
            put(f"{m}.w_image", cc["w_image"]["value"])
            put(f"{m}.w_text", cc["w_text"]["value"])
            put(f"{m}.text_dominance", cc["text_dominance"]["value"])
        if r["flip"].get("available"):
            for k in ("text_consistent", "image_consistent", "concept_prior"):
                put(f"{m}.flip_{k}", r["flip"][k]["value"])
        if r["attention"].get("available"):
            put(f"{m}.ADS", r["attention"]["ADS"]["value"])
        if r["interaction"].get("available"):
            put(f"{m}.super_additive_drop", r["interaction"]["super_additive_drop"]["value"])
        if r["decay"].get("available"):
            put(f"{m}.decay_rate", r["decay"]["decay_rate"]["value"])
    for k in ("friedman_acc_adverse", "rm_anova_acc_adverse"):
        x = report["cross_model"].get(k) or {}
        for kk in ("chi2", "F", "p"):
            put(f"cross.{k}.{kk}", x.get(kk))
    return c


# claim -> the metric it comes from, for claims that rest on a probability probe (metrics.PROB_DEPENDENT)
_PROB_CLAIMS = {"ece_adverse": "robustness.ece_adverse", "R": "robustness.R", "trust_index": "trust.trust_index",
                "w_image": "cue_combination", "w_text": "cue_combination", "text_dominance": "cue_combination"}


def probability_flags(report: Dict[str, Any]) -> Dict[str, Any]:
    """Models whose probability-based metrics (and the claims built on them) are unreliable."""
    out = {}
    for m, r in report["models"].items():
        bad = r["probability_quality"]["flagged_metrics"]
        if bad:
            out[m] = {"metrics": bad, "claims": [f"{m}.{k}" for k, src in _PROB_CLAIMS.items()
                                                 if src in bad and f"{m}.{k}" in report["claims"]]}
    return out


def model_tables(items: List[BaseItem], cfg: Dict[str, Any], run_dirs: Sequence[str | Path]) -> Iterator[pd.DataFrame]:
    """One lean table per model, built from its run directory."""
    for rd in run_dirs:
        df = build_table(items, cfg, [rd])
        if df.empty:
            continue
        if df["model"].nunique() == 1:
            yield df
        else:
            for _, sub in df.groupby("model", observed=True):
                yield sub
            del sub
        del df  # before the next model's table is built: never hold two at once


def compute_report(items: List[BaseItem], cfg: Dict[str, Any], tables: Iterable[pd.DataFrame],
                   on_model: Optional[Callable[[str, pd.DataFrame], None]] = None) -> Dict[str, Any]:
    """Every metric, model by model (``tables`` yields one model's rows at a time), then the
    cross-model tests. The items are the units; images carrying several items are bootstrap clusters."""
    by_id = {it.gib_id: it for it in items}
    ids = sorted(by_id)
    iters = cfg["analysis"]["bootstrap_iters"]
    W = S.cluster_weights([by_id[g].image_path for g in ids], iters, cfg["seed"])
    report: Dict[str, Any] = {"n_items": len(ids), "models": {}, "bootstrap_iters": iters}
    arrays, boots = {}, {}
    for df in tables:
        m = str(df["model"].iloc[0])
        report["models"][m], boots[m], v = model_metrics(df, ids, by_id, cfg, W)
        arrays[m] = {"clean": v.qa("clean"), "adverse": M.adverse_accuracy(v)}
        del v
        if on_model is not None:
            on_model(m, df)
        del df  # free this model's table before the next one is built
    if not report["models"]:
        raise SystemExit("no predictions found for these runs/items")
    report["cross_model"] = cross_model(arrays, boots, report["models"])
    report["simulated_models"] = sorted(m for m in report["models"] if m.startswith("sim-"))
    report["claims"] = claims_from(report)
    report["probability_flags"] = probability_flags(report)
    return report


def _file_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".csv.gz"


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_report(out_dir: str | Path, items: List[BaseItem], cfg: Dict[str, Any], run_dirs: Sequence[str | Path],
                 image_root: Optional[str | Path] = None, human: Optional[Dict[str, Any]] = None,
                 figures: bool = True) -> Dict[str, Any]:
    from .gallery import GALLERY_CONDITIONS

    out_dir = Path(out_dir)
    prov = out_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    for old in prov.glob("*.csv.gz"):  # files of models from an earlier report into the same directory
        old.unlink()
    gallery_rows: List[pd.DataFrame] = []

    def on_model(m: str, df: pd.DataFrame) -> None:
        save_table(df, prov / _file_name(m))
        rows = df.loc[(df["probe"] == "qa") & df["condition"].isin(GALLERY_CONDITIONS),
                      ["model", "probe", "gib_id", "condition", "text", "correct"]]
        gallery_rows.append(rows.astype({c: object for c in ("model", "probe", "gib_id", "condition", "text")}))

    report = compute_report(items, cfg, model_tables(items, cfg, run_dirs), on_model)
    (out_dir / "items.txt").write_text("\n".join(sorted(it.gib_id for it in items)) + "\n", encoding="utf-8")
    if human:
        report["human"] = human
    report = _jsonable(report)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "claims.json").write_text(json.dumps(report["claims"], indent=2, sort_keys=True), encoding="utf-8")
    from .macros import write_macros
    from .results_md import write_results_md

    write_results_md(out_dir / "RESULTS.md", report)
    write_macros(out_dir / "macros.tex", report)
    if figures:
        from .figures import make_figures

        make_figures(out_dir / "figures", report)
        if image_root is not None:
            from .gallery import write_gallery

            write_gallery(out_dir / "gallery.html", pd.concat(gallery_rows, ignore_index=True), items, cfg, image_root)
    return report
