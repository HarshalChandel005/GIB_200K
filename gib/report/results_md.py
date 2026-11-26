"""RESULTS.md -- every table generated from report.json (nothing typed by hand)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(st: Optional[Dict[str, Any]], nd: int = 3) -> str:
    if not st or st.get("value") is None:
        return "–"
    lo, hi = (st.get("ci") or [None, None])[:2]
    if lo is None or hi is None:
        return _f(st["value"], nd)
    return f"{st['value']:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_results_md(path: str | Path, rep: Dict[str, Any]) -> None:
    L: List[str] = ["# GIB-200K results", ""]
    if rep.get("simulated_models"):
        L += ["> **SIMULATED OUTPUTS.** Models " + ", ".join(rep["simulated_models"]) +
              " are behavioural simulators used to test the pipeline. These numbers are not "
              "findings about any real model and must not appear in the paper.", ""]
    L += [f"Items: **{rep['n_items']}** (every model on the same items) · bootstrap: "
          f"{rep['bootstrap_iters']} item-cluster resamples · values are point estimates [95% CI].", ""]
    models = sorted(rep["models"])
    flags = rep.get("probability_flags") or {}

    def warn(m: str, metric: str) -> str:
        return " ⚠" if metric in (flags.get(m) or {}).get("metrics", []) else ""

    L += ["## Probability quality", "",
          "Share of each probability probe's answers whose distribution came from token log-probabilities "
          "(not a parsed text answer scored as one-hot) with enough mass on the valid answer tokens (*usable*). "
          "Below the minimum, metrics that read graded probabilities are marked ⚠.", ""]
    rows, min_cov = [], None
    for m in models:
        pq = rep["models"][m].get("probability_quality") or {}
        min_cov = pq.get("min_coverage", min_cov)
        for probe, st in sorted(pq.get("probes", {}).items()):
            status = "ok" if st["ok"] else "⚠ low" if st.get("affects") else "low (no metric affected)"
            rows.append([m, probe, str(st["n"]), _f(st["with_probs"]), _f(st["from_logprobs"]),
                         _f(st["format_mass_mean"]), _f(st["low_mass"]), _f(st["usable"]), status])
    if rows:
        L += [_table(["model", "probe", "answers", "with probabilities", "from log-probs", "mean format mass",
                      "low mass", "usable", f"status (min {_f(min_cov, 2)})"], rows), ""]
    if flags:
        L += ["> **⚠ LOW PROBABILITY COVERAGE.** " +
              "; ".join(f"{m}: {', '.join(f['metrics'])}" for m, f in sorted(flags.items())) +
              ". These metrics read graded probabilities, but too few of the model's answers carry usable ones "
              "(for example an API that returned no log-probabilities). Treat them as unreliable; claims built "
              "on them are marked in macros.tex.", ""]

    L += ["## Headline", ""]
    rows = []
    for m in models:
        r = rep["models"][m]
        rob, tr, cc = r["robustness"], r["trust"], r["cue_combination"]
        rows.append([m, _ci(rob["acc_clean"]), _ci(rob["acc_adverse"]),
                     _ci(rob["ece_adverse"]) + warn(m, "robustness.ece_adverse"), _ci(rob["R"]) + warn(m, "robustness.R"),
                     _ci(tr["trust_index"], 2) + warn(m, "trust.trust_index"), _ci(tr["GIS"]),
                     (_ci(cc.get("text_dominance"), 2) + warn(m, "cue_combination")) if cc.get("available") else "–",
                     "yes" if r["reproducible"] else "no (API)"])
    L += [_table(["model", "acc clean", "acc adverse", "ECE adverse", "R", "Trust Index", "GIS",
                  "w_text / w_image", "reproducible"], rows), ""]

    cm = rep["cross_model"]
    if cm.get("friedman_acc_adverse"):
        fr, an = cm["friedman_acc_adverse"], cm["rm_anova_acc_adverse"]
        L += ["## Do models differ? (repeated measures over items)", ""]
        if fr.get("chi2") is not None:
            L.append(f"- Friedman on per-item adverse accuracy: χ²({fr['df']}) = {fr['chi2']:.2f}, "
                     f"p = {fr['p']:.3g}, Kendall's W = {_f(fr.get('kendall_w'))}, n = {fr['n']} items, k = {fr['k']} models.")
        if an.get("F") is not None:
            L.append(f"- Repeated-measures ANOVA: F({an['df1']}, {an['df2']}) = {an['F']:.2f}, p = {an['p']:.3g}, "
                     f"partial η² = {an['partial_eta2']:.3f}, n = {an['n']}.")
        L.append("")
        rows = [[p["a"], p["b"], _f(p["diff_acc_adverse"]), f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]",
                 _f(p.get("p_holm"), 4)] for p in cm.get("pairwise_acc_adverse", [])]
        if rows:
            L += [_table(["model A", "model B", "Δ adverse acc", "95% CI", "p (Holm)"], rows), ""]
    r_flagged = [m for m in models if warn(m, "robustness.R")]
    L += [f"Pareto frontier (clean accuracy vs R): **{', '.join(cm.get('pareto_frontier', []))}**"
          + (f" (R ⚠ for {', '.join(r_flagged)})" if r_flagged else ""), ""]

    L += ["## Conflict matrix", ""]
    rows = [[m, _f(c["trust_index"], 2) + warn(m, "trust.trust_index"),
             f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]" if c["ci"][0] is not None else "–",
             _f(c["qa_text_follow_rate"]), c["label"]] for m, c in sorted(cm.get("conflict_matrix", {}).items())]
    L += [_table(["model", "Trust Index", "95% CI", "free-form follows text", "bias"], rows), ""]

    L += ["## Cue combination (ideal-observer test)", "",
          "LO(image+text) − LO(prior) = w_image·[LO(image) − LO(prior)] + w_text·[LO(text) − LO(prior)]; "
          "an ideal Bayesian integrator has w_image = w_text = 1.", ""]
    rows = []
    for m in models:
        cc = rep["models"][m]["cue_combination"]
        if cc.get("available"):
            rows.append([m + warn(m, "cue_combination"), _ci(cc["w_image"], 2), _ci(cc["w_text"], 2), _ci(cc["text_dominance"], 2),
                         _f(cc["r2_ideal_additive"]), _ci(cc["integration_KI_rate"]), _ci(cc["integration_SC_rate"])])
    if rows:
        L += [_table(["model", "w_image", "w_text", "w_text/w_image", "R² of ideal sum", "KI integration errors",
                      "SC integration errors"], rows), ""]

    L += ["## Counterfactual flip decomposition", ""]
    rows = []
    for m in models:
        f = rep["models"][m]["flip"]
        if f.get("available"):
            rows.append([m, str(f["n_pairs"]), _ci(f["text_consistent"]), _ci(f["image_consistent"]),
                         _ci(f["concept_prior"]), _ci(f["modality_bias"])])
    if rows:
        L += [_table(["model", "pairs", "text-consistent", "image-consistent", "concept prior", "modality bias"], rows), ""]

    L += ["## Retrieval noise: decay", ""]
    rows = []
    for m in models:
        d = rep["models"][m]["decay"]
        if d.get("available"):
            rows.append([m] + [_f(a) for a in d["accuracy"]] + [_ci(d["decay_rate"]), _ci(d["retained_auc"])])
            ks = d["k"]
    if rows:
        L += [_table(["model"] + [f"k={k}" for k in ks] + ["decay rate", "retained AUC"], rows), ""]

    L += ["## Visual ambiguity: resolution thresholds", ""]
    rows = []
    for m in models:
        for fam, r in rep["models"][m]["resolution"].items():
            at = r.get("acc_threshold", {})
            ct = r.get("chair_threshold", {})
            rows.append([m, fam, _f(at.get("level"), 2), str(at.get("ci_levels")), _f(ct.get("level"), 2) if ct else "–"])
    if rows:
        L += [_table(["model", "family", "accuracy threshold", "95% CI (levels)", "CHAIR threshold"], rows), ""]

    L += ["## Stressor interaction (retrieval × blur)", ""]
    rows = []
    for m in models:
        it = rep["models"][m]["interaction"]
        if it.get("available"):
            rows.append([m, _ci(it["super_additive_drop"]), _f(it["p_sign_flip"], 4)])
    if rows:
        L += [_table(["model", "super-additive drop", "p (sign-flip)"], rows), ""]

    L += ["## Attention shift (paired clean → conflict)", ""]
    rows = []
    for m in models:
        a = rep["models"][m]["attention"]
        if a.get("available"):
            rows.append([m, _ci(a["ADS"]), _ci(a["bands"]["early"]), _ci(a["bands"]["middle"]), _ci(a["bands"]["late"]),
                         _f(a.get("auroc_low_share_predicts_KI"))])
    if rows:
        L += [_table(["model", "ADS (all layers)", "early", "middle", "late", "AUROC share→KI"], rows), ""]

    L += ["## Failure taxonomy (share of wrong free-form answers)", ""]
    rows = []
    for m in models:
        for axis, t in sorted(rep["models"][m]["taxonomy"].items()):
            s = t["share"]
            rows.append([m, axis, str(t["errors"]), _f(s.get("Knowledge Inertia", 0)), _f(s.get("Sensory Capture", 0)),
                         _f(s.get("Reasoning Collapse", 0)), _f(s.get("Perceptual Error", 0))])
    L += [_table(["model", "axis", "errors", "Knowledge Inertia", "Sensory Capture", "Reasoning Collapse",
                  "Perceptual Error"], rows), ""]

    L += ["## Consistency gate (training-free mitigation)", ""]
    rows = []
    for m in models:
        for cond, g in rep["models"][m]["gate"].items():
            rows.append([m, cond, _ci(g["base_accuracy"]), _ci(g["disagreement_rate"]),
                         _ci(g["abstain"]["selective_accuracy"]), _ci(g["abstain"]["coverage"]),
                         _ci(g["trust_image"]["delta"]), _ci(g["confidence"]["delta"]) + warn(m, "gate.confidence")])
    if rows:
        L += [_table(["model", "condition", "base acc", "disagreement", "abstain: sel. acc", "abstain: coverage",
                      "Δ trust-image", "Δ confidence"], rows), ""]

    L += ["## Hallucination metrics on clean images", ""]
    rows = []
    for m in models:
        p, c = rep["models"][m]["pope"], rep["models"][m]["chair"]
        rows.append([m, _ci(p.get("accuracy")) if p.get("available") else "–", _ci(p.get("f1")) if p.get("available") else "–",
                     _ci(p.get("yes_ratio")) if p.get("available") else "–",
                     _ci(c.get("CHAIR_i")) if c.get("available") else "–", _ci(c.get("CHAIR_s")) if c.get("available") else "–"])
    L += [_table(["model", "POPE acc", "POPE F1", "yes ratio", "CHAIR_i", "CHAIR_s"], rows), ""]

    L += ["## Contamination (likely-seen vs. less-likely-seen source images)", ""]
    rows = []
    for m in models:
        c = rep["models"][m]["contamination"]
        if c.get("available") and c.get("difference"):
            d = c["difference"]
            rows.append([m, d["acc_clean"]["contrast"], _ci(d["acc_clean"]), _ci(d["acc_adverse"]),
                         _ci(d["qa_text_follow_rate"])])
    if rows:
        L += [_table(["model", "contrast", "Δ acc clean", "Δ acc adverse", "Δ follows text"], rows), ""]

    if rep.get("human"):
        h = rep["human"]
        L += ["## Human baseline", "", f"Participants: {h.get('n_participants')} · items: {h.get('n_items')}", ""]
        rows = [[k, _ci(v)] for k, v in sorted(h.get("accuracy_by_stressor", {}).items())]
        L += [_table(["stressor", "human accuracy"], rows), ""]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
