"""Figures generated from report.json, with captions written by the same code.

Captions must not describe things the figures do not show
(missing shaded regions, unexplained dashed lines, error bars not drawn,
overlapping labels). Here every caption is generated next to the plotting code
that draws the element it describes, legends sit outside the axes, and every
uncertainty band mentioned in a caption is actually drawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def _colors(models: List[str]) -> Dict[str, Any]:
    plt = _plt()
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(models)}


def _band(ax, x, ci, color):
    lo = [c[0] if c and c[0] is not None else np.nan for c in ci]
    hi = [c[1] if c and c[1] is not None else np.nan for c in ci]
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def make_figures(out_dir: str | Path, rep: Dict[str, Any]) -> List[str]:
    plt = _plt()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = sorted(rep["models"])
    col = _colors(models)
    caps: List[str] = ["# Figure captions (generated with the figures)", ""]
    sim = rep.get("simulated_models")
    if sim:
        caps += ["**SIMULATED MODELS — pipeline test only, not results.**", ""]
    made = []

    def save(fig, name: str, caption: str) -> None:
        fig.savefig(out / name)
        plt.close(fig)
        made.append(name)
        caps.extend([f"## {name}", caption, ""])

    # 1. vulnerability heatmaps -------------------------------------------- #
    have = [m for m in models if rep["models"][m]["interaction"].get("available")]
    if have:
        it0 = rep["models"][have[0]]["interaction"]
        sig, dos = it0["sigmas"], it0["dosages"]
        fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have) + 0.6, 2.8), squeeze=False)
        vmax = max(max(abs(v) for v in rep["models"][m]["interaction"]["heatmap_drop"].values() if v is not None)
                   for m in have) or 1.0
        for ax, m in zip(axes[0], have):
            h = rep["models"][m]["interaction"]["heatmap_drop"]
            M = np.array([[h.get(f"{s:g}|{k}") if h.get(f"{s:g}|{k}") is not None else np.nan for k in dos] for s in sig])
            im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax, origin="lower", aspect="auto")
            for i in range(len(sig)):
                for j in range(len(dos)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                                color="white" if M[i, j] > 0.6 * vmax else "black")
            ax.set_xticks(range(len(dos)), [str(k) for k in dos])
            ax.set_yticks(range(len(sig)), [f"{s:g}" for s in sig])
            ax.set_xlabel("hard-negative documents (k)")
            ax.set_title(m, fontsize=9)
        axes[0][0].set_ylabel("Gaussian blur σ (px)")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="accuracy drop vs clean")
        save(fig, "fig_vulnerability_heatmap.png",
             "Vulnerability heatmap. Each cell is the drop in free-form accuracy relative to the clean condition "
             "(image + gold document) when the image is blurred (rows, σ in px) and k hard-negative documents are "
             "added (columns). Numbers are percentage points. One panel per model; shared colour scale.")

    # 2. decay curves ------------------------------------------------------- #
    have = [m for m in models if rep["models"][m]["decay"].get("available")]
    if have:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for m in have:
            d = rep["models"][m]["decay"]
            x = np.log2(1 + np.array(d["k"], float))
            ax.plot(x, d["accuracy"], marker="o", ms=3, color=col[m], label=m)
            _band(ax, x, d["accuracy_ci"], col[m])
        ax.set_xticks(np.log2(1 + np.array(d["k"], float)), [str(k) for k in d["k"]])
        ax.set_xlabel("hard-negative documents k (log scale)")
        ax.set_ylabel("accuracy")
        ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, "fig_decay.png",
             "Retrieval-noise dose–response. Free-form accuracy versus the number k of hard-negative documents "
             "added next to the gold document (k = 0 is the clean condition). Shaded bands are 95% item-bootstrap "
             "confidence intervals.")

    # 3. resolution curves -------------------------------------------------- #
    fams = sorted({f for m in models for f in rep["models"][m]["resolution"]})
    if fams:
        fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 2.8), squeeze=False)
        for ax, fam in zip(axes[0], fams):
            for m in models:
                r = rep["models"][m]["resolution"].get(fam)
                if not r:
                    continue
                x = np.arange(len(r["levels"]) + 1)
                ax.plot(x, r["accuracy"], marker="o", ms=3, color=col[m], label=m)
                _band(ax, x, r["accuracy_ci"], col[m])
                thr = r.get("acc_threshold", {}).get("level")
                if thr is not None:
                    xi = r["levels"].index(thr) + 1
                    ax.plot([xi], [r["accuracy"][xi]], marker="v", ms=7, color=col[m], markeredgecolor="black")
            ax.set_xticks(np.arange(len(r["levels"]) + 1), ["none"] + [f"{lv:g}" for lv in r["levels"]])
            unit = {"gauss": "σ (px)", "motion": "kernel (px)", "crop": "fraction of object visible", "noise": "noise std"}[fam]
            ax.set_xlabel(unit)
            ax.set_title({"gauss": "Gaussian blur", "motion": "motion blur", "crop": "low-resource crop",
                          "noise": "Gaussian noise"}[fam], fontsize=9)
        axes[0][0].set_ylabel("accuracy (no documents)")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=min(4, len(lab)))
        save(fig, "fig_resolution.png",
             "Visual-ambiguity dose–response without retrieved documents. Shaded bands: 95% item-bootstrap CIs. "
             "A downward triangle marks each model's resolution threshold: the first level at which accuracy falls "
             "more than the configured jump (default 10 points) below the undegraded no-document accuracy.")

    # 4. conflict matrix ---------------------------------------------------- #
    cm = rep["cross_model"].get("conflict_matrix", {})
    if cm:
        fig, ax = plt.subplots(figsize=(4.6, 0.45 * len(cm) + 1.0))
        names = sorted(cm)
        for i, m in enumerate(names):
            c = cm[m]
            if c["trust_index"] is None:
                continue
            lo, hi = c["ci"]
            ax.errorbar([c["trust_index"]], [i], xerr=[[c["trust_index"] - lo], [hi - c["trust_index"]]],
                        fmt="o", color=col[m], capsize=3)
            ax.text(hi * 1.15, i, c["label"], va="center", fontsize=7)
        ax.axvline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Trust Index = P(text answer) / P(image answer)  (log scale)")
        save(fig, "fig_conflict_matrix.png",
             "Conflict matrix. Trust Index (geometric mean over conflict items of the A/B-probe odds of the text's "
             "claim over the image's answer; both option orders averaged) with 95% item-bootstrap CIs. The dashed "
             "line at 1 is balance; a model is labelled text-biased (image-biased) when its whole CI lies above "
             "(below) 1.")

    # 5. flip decomposition -------------------------------------------------- #
    have = [m for m in models if rep["models"][m]["flip"].get("available")]
    if have:
        fig, ax = plt.subplots(figsize=(4.6, 0.45 * len(have) + 1.0))
        cats = [("text_consistent", "#c0392b", "follows text both ways"),
                ("image_consistent", "#2471a3", "follows image both ways"),
                ("concept_prior", "#7f8c8d", "same concept both ways (prior)")]
        for i, m in enumerate(have):
            left = 0.0
            for c, color, lab in cats:
                v = rep["models"][m]["flip"][c]["value"]
                ax.barh(i, v, left=left, color=color, label=lab if i == 0 else None)
                left += v
        ax.set_yticks(range(len(have)), have)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of counterfactual flip pairs")
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
        save(fig, "fig_flip.png",
             "Counterfactual flip decomposition. Each pair joins an item whose image shows A while the text claims B "
             "with an item whose image shows B while the text claims A. A pair is text-consistent if the model "
             "follows the text in both, image-consistent if it follows the image in both, and concept-prior if it "
             "names the same concept in both regardless of modality.")

    # 6. cue weights ---------------------------------------------------------- #
    def drawable(cc: Dict[str, Any]) -> bool:  # an undefined estimate (e.g. 0/1 probabilities) has nothing to plot
        return bool(cc.get("available")) and all(
            isinstance(x, (int, float)) for k in ("w_image", "w_text") for x in [cc[k]["value"], *cc[k]["ci"]])

    have = [m for m in models if drawable(rep["models"][m]["cue_combination"])]
    if have:
        fig, ax = plt.subplots(figsize=(4.0, 3.6))
        for m in have:
            cc = rep["models"][m]["cue_combination"]
            wi, wt = cc["w_image"], cc["w_text"]
            ax.errorbar([wi["value"]], [wt["value"]],
                        xerr=[[wi["value"] - wi["ci"][0]], [wi["ci"][1] - wi["value"]]],
                        yerr=[[wt["value"] - wt["ci"][0]], [wt["ci"][1] - wt["value"]]],
                        fmt="o", color=col[m], capsize=2, label=m)
        lim = max(2.0, max(max(rep["models"][m]["cue_combination"]["w_text"]["ci"][1] or 0,
                               rep["models"][m]["cue_combination"]["w_image"]["ci"][1] or 0) for m in have) * 1.1)
        ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--")
        ax.plot([1], [1], marker="*", ms=12, color="gold", markeredgecolor="black", label="ideal observer")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("weight on image evidence w_image")
        ax.set_ylabel("weight on text evidence w_text")
        ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, "fig_cue_weights.png",
             "Cue-combination weights. For each model, the bimodal A/B log-odds (minus the blank-image prior) are "
             "regressed on the image-only and text-only log-odds measured on the same items at three blur levels. "
             "The star is the ideal Bayesian integrator (1, 1); points above the dashed diagonal over-weight text "
             "(Knowledge Inertia), points below over-weight the image (Sensory Capture). Error bars: 95% "
             "item-bootstrap CIs.")

    # 7. attention profiles ----------------------------------------------------- #
    have = [m for m in models if rep["models"][m]["attention"].get("available")]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have), 2.6), squeeze=False)
        for ax, m in zip(axes[0], have):
            a = rep["models"][m]["attention"]
            L = np.arange(a["n_layers"])
            ax.plot(L, a["clean_share_profile"], color="#2471a3", label="clean (gold document)")
            ax.plot(L, a["conflict_share_profile"], color="#c0392b", label="conflict (misleading document)")
            if a.get("share_profile_KI"):
                ax.plot(L, a["share_profile_KI"], color="#c0392b", ls=":", label="conflict, followed text")
                ax.plot(L, a["share_profile_grounded"], color="#2471a3", ls=":", label="conflict, followed image")
            ax.set_title(m, fontsize=9)
            ax.set_xlabel("layer")
        axes[0][0].set_ylabel("visual share of attention")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=min(4, len(lab)))
        save(fig, "fig_attention.png",
             "Where the last prompt token attends, per layer: visual share = attention mass on image tokens / "
             "(image + post-image text tokens), averaged over heads and items. Solid lines compare the same images "
             "with a correct versus a misleading document; dotted lines split conflict items by whether the model's "
             "answer followed the text or the image.")

    # 8. Pareto frontier ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    front = set(rep["cross_model"].get("pareto_frontier", []))
    for m in models:
        rob = rep["models"][m]["robustness"]
        x, y = rob["acc_clean"]["value"], rob["R"]["value"]
        if x is None or y is None:
            continue
        ax.errorbar([x], [y], xerr=[[x - rob["acc_clean"]["ci"][0]], [rob["acc_clean"]["ci"][1] - x]],
                    yerr=[[y - rob["R"]["ci"][0]], [rob["R"]["ci"][1] - y]], fmt="o", color=col[m], capsize=2,
                    label=m + (" (frontier)" if m in front else ""))
    ax.set_xlabel("clean accuracy")
    ax.set_ylabel("robustness R")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig_pareto.png",
         "Accuracy–robustness trade-off. R = (adverse accuracy / clean accuracy) × (1 − ECE on adverse "
         "conditions). Error bars are 95% item-bootstrap CIs; models on the Pareto frontier are marked in the legend.")

    # 9. taxonomy ------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(models) * 4 + 1.0))
    order = ["Knowledge Inertia", "Sensory Capture", "Reasoning Collapse", "Perceptual Error"]
    colors = ["#c0392b", "#2471a3", "#8e44ad", "#7f8c8d"]
    rows = []
    for m in models:
        for axis in ("retrieval", "visual", "conflict", "grid"):
            t = rep["models"][m]["taxonomy"].get(axis)
            if t and t["errors"]:
                rows.append((f"{m} · {axis}", t["share"]))
    for i, (lab, share) in enumerate(rows):
        left = 0.0
        for c, colr in zip(order, colors):
            v = share.get(c, 0.0)
            ax.barh(i, v, left=left, color=colr, label=c if i == 0 else None)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of wrong free-form answers")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    save(fig, "fig_taxonomy.png",
         "Failure taxonomy by stress axis, decided by what the wrong answer matches: a wrong claim from the "
         "documents (Knowledge Inertia), a valid answer that ignores correct documents (Sensory Capture), "
         "an off-vocabulary answer or one matching neither modality (Reasoning Collapse), or a wrong answer with "
         "no documents shown (Perceptual Error).")

    (out / "CAPTIONS.md").write_text("\n".join(caps), encoding="utf-8")
    return made
