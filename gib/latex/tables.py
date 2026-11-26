"""Paper tables and named values, generated from report.json (never typed by hand)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .markup import Table

WAIT = "Pending: filled from report.json after the real model runs (gib report)."
LOW_COVERAGE = ("‡ Rests on too few answers with usable token probabilities (e.g. no API log-probabilities); "
                "treat as unreliable.")


def _ci(st: Optional[Dict[str, Any]], nd: int = 2) -> str:
    if not st or st.get("value") is None:
        return "–"
    ci = st.get("ci") or [None, None]
    if ci[0] is None or ci[1] is None:
        return f"{st['value']:.{nd}f}"
    return f"{st['value']:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _f(x: Any, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "–"


def _models(rep: Optional[Dict[str, Any]]) -> List[str]:
    return sorted(rep["models"]) if rep else []


def _lc(rep: Dict[str, Any], m: str, metric: str, t: Table) -> str:
    """' ‡' (and the table note) when ``metric`` is flagged for low probability coverage."""
    if metric not in ((rep.get("probability_flags") or {}).get(m) or {}).get("metrics", []):
        return ""
    t.note = LOW_COVERAGE
    return " ‡"


def values(rep: Optional[Dict[str, Any]], kappa: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not rep:
        return {}
    v: Dict[str, Any] = {"n_items": f"{rep['n_items']:,}", "n_models": str(len(rep["models"]))}
    ms = rep["models"]
    v["n_open"] = str(sum(1 for m in ms.values() if m.get("reproducible")))
    v["n_closed"] = str(sum(1 for m in ms.values() if not m.get("reproducible")))
    v["model_list"] = ", ".join(sorted(ms))
    cm = rep.get("cross_model", {})
    fr, an = cm.get("friedman_acc_adverse") or {}, cm.get("rm_anova_acc_adverse") or {}
    if fr.get("chi2") is not None:
        v["friedman"] = (f"χ²({fr['df']}) = {fr['chi2']:.1f}, p = {fr['p']:.2g}, "
                         f"Kendall's W = {fr['kendall_w']:.2f}, n = {fr['n']} items" if fr.get("kendall_w") is not None
                         else f"statistic = {fr['chi2']:.1f}, p = {fr['p']:.2g}, n = {fr['n']}")
    if an.get("F") is not None:
        v["rm_anova"] = f"F({an['df1']}, {an['df2']}) = {an['F']:.1f}, p = {an['p']:.2g}, partial η² = {an['partial_eta2']:.2f}"
    if cm.get("pareto_frontier"):
        v["pareto"] = ", ".join(cm["pareto_frontier"])
    h = rep.get("human")
    if h:
        v["n_participants"] = str(h.get("n_participants"))
        v["human_items"] = str(h.get("n_items"))
        if h.get("accuracy_overall"):
            v["human_acc"] = _ci(h["accuracy_overall"])
        if h.get("fleiss_kappa_correctness") is not None:
            v["human_fleiss"] = f"{h['fleiss_kappa_correctness']:.2f}"
    if kappa and kappa.get("n"):
        v["tax_kappa"] = f"{kappa['cohen_kappa']:.2f}"
        v["tax_n"] = str(kappa["n"])
    if rep.get("simulated_models"):
        v["SIMULATED_WARNING"] = "SIMULATED MODELS PRESENT — NOT RESULTS"
    return v


def headline(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Overall results on all items (95% bootstrap CIs, images as clusters). R = (acc adverse / acc clean) × "
              "(1 − ECE adverse); TI = Trust Index; GIS = Grounding Illusion Score; † closed API model.",
              ["Model", "Acc clean", "Acc adverse", "ECE adv.", "R", "TI", "GIS"], [], "tab:headline", wide=True)
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        r = rep["models"][m]
        rob, tr = r["robustness"], r["trust"]
        t.rows.append([m + ("" if r["reproducible"] else " †"), _ci(rob["acc_clean"]), _ci(rob["acc_adverse"]),
                       _ci(rob["ece_adverse"]) + _lc(rep, m, "robustness.ece_adverse", t),
                       _ci(rob["R"]) + _lc(rep, m, "robustness.R", t),
                       _ci(tr["trust_index"]) + _lc(rep, m, "trust.trust_index", t), _ci(tr["GIS"])])
    return t


def cue(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Cue-combination weights (measurement-error corrected; ideal observer: w_I = w_T = 1). "
              "w_T/w_I > 1 = Knowledge Inertia, < 1 = Sensory Capture. Reliability = share of regressor variance "
              "that is signal.", ["Model", "w_I", "w_T", "w_T / w_I", "R² ideal sum", "reliability (I / T)"],
              [], "tab:cue")
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        c = rep["models"][m]["cue_combination"]
        if c.get("available"):
            rel = c.get("reliability", {})
            t.rows.append([m + _lc(rep, m, "cue_combination", t), _ci(c["w_image"]), _ci(c["w_text"]), _ci(c["text_dominance"]),
                           _f(c.get("r2_ideal_additive")), f"{_f(rel.get('image'))} / {_f(rel.get('text'))}"])
        else:
            t.rows.append([m, "–", "–", "–", "–", "–"])
    return t


def flip(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Counterfactual flip decomposition (share of flip pairs).",
              ["Model", "pairs", "text-consistent", "image-consistent", "concept prior"], [], "tab:flip")
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        f = rep["models"][m]["flip"]
        if f.get("available"):
            t.rows.append([m, str(f["n_pairs"]), _ci(f["text_consistent"]), _ci(f["image_consistent"]),
                           _ci(f["concept_prior"])])
    return t


def taxonomy(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Failure taxonomy: share of wrong free-form answers per class (all stress conditions).",
              ["Model", "errors", "Knowl. Inertia", "Sens. Capture", "Reas. Collapse", "Perceptual"], [], "tab:tax")
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        a = rep["models"][m]["taxonomy"].get("all", {})
        s = a.get("share", {})
        t.rows.append([m, str(a.get("errors", 0))] + [_f(s.get(k, 0)) for k in
                      ("Knowledge Inertia", "Sensory Capture", "Reasoning Collapse", "Perceptual Error")])
    return t


def gate(rep: Optional[Dict[str, Any]], conds=("conflict", "ret_k20"), label="tab:gate", note="") -> Table:
    t = Table("Training-free consistency gate: change in accuracy when the answer with documents disagrees with the "
              "image-only answer (Δ vs. no gate; 95% CI)." + note,
              ["Model", "condition", "base acc", "Δ image fallback", "Δ confidence", "abstain: sel. acc / coverage"],
              [], label, wide=False)
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        for cond in conds:
            g = rep["models"][m]["gate"].get(cond)
            if g:
                t.rows.append([m, cond.replace("_", " "), _ci(g["base_accuracy"]), _ci(g["trust_image"]["delta"]),
                               _ci(g["confidence"]["delta"]) + _lc(rep, m, "gate.confidence", t),
                               f"{_f(g['abstain']['selective_accuracy']['value'])} / {_f(g['abstain']['coverage']['value'])}"])
    return t


def human(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Humans vs. models on the same (item, condition) pairs: accuracy per stressor.",
              ["Stressor", "Humans"], [], "tab:human")
    h = (rep or {}).get("human")
    if not h:
        t.rows = [["Pending: run the human study (gib human-export / human-analyze) and pass --human to gib report."]]
        return t
    models = sorted((h.get("models_on_same_items") or {}).keys())
    t.header = ["Stressor", "Humans"] + models
    for st, acc in sorted(h.get("accuracy_by_stressor", {}).items()):
        row = [st, _ci(acc)]
        for m in models:
            v = h["models_on_same_items"][m].get(st)
            row.append(_f(v))
        t.rows.append(row)
    return t


def contamination(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Contamination check: difference between items from COCO train2017 (likely seen in instruction "
              "tuning) and val2017.", ["Model", "Δ acc clean", "Δ acc adverse", "Δ follows text"], [], "tab:contam")
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        c = rep["models"][m]["contamination"]
        if c.get("difference"):
            d = c["difference"]
            t.rows.append([m, _ci(d["acc_clean"]), _ci(d["acc_adverse"]), _ci(d["qa_text_follow_rate"])])
    return t


def dose(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Retrieval-noise dose–response (free-form accuracy with k hard negatives next to the gold document; "
              "decay = log-odds per doubling of 1 + k) and the blur × retrieval interaction (super-additive drop; "
              "sign-flip permutation p).",
              ["Model", "k=0", "k=1", "k=5", "k=20", "decay rate", "retained AUC", "super-additive drop (p)"],
              [], "tab:dose", wide=True)
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        r = rep["models"][m]
        d, i = r["decay"], r["interaction"]
        if d.get("available"):
            acc = dict(zip(d["k"], d["accuracy"]))
            sad, p = i.get("super_additive_drop"), i.get("p_sign_flip")
            t.rows.append([m] + [_f(acc.get(k), 3) for k in (0, 1, 5, 20)] + [_ci(d["decay_rate"]), _ci(d["retained_auc"]),
                           f"{_ci(sad)} ({p:.1g})" if p is not None else _ci(sad)])
    return t


def attention(rep: Optional[Dict[str, Any]]) -> Table:
    t = Table("Attention shift, clean → conflict (paired per item; negative = more visual attention under conflict) and "
              "AUROC of a low layer-averaged visual share (conflict condition) for predicting that the answer follows the text.",
              ["Model", "ADS, all layers", "ADS, late layers", "AUROC (low share → follows text)", "n follow text / image"],
              [], "tab:attn")
    if not rep:
        t.rows = [[WAIT]]
        return t
    for m in _models(rep):
        a = rep["models"][m]["attention"]
        if a.get("available"):
            t.rows.append([m, _ci(a["ADS"], 3), _ci(a["bands"]["late"], 3), _f(a.get("auroc_low_share_predicts_KI")),
                           f"{a.get('n_KI')} / {a.get('n_grounded')}"])
    return t
