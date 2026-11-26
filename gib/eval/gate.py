"""From diagnosis to mitigation: a training-free cross-modal consistency gate.

A benchmark should show how its metrics help *improve* multimodal RAG, and whether
they relate to downstream reliability. The benchmark already asks every
question twice -- with the retrieved documents and without them -- so a
deployable check is free:

    if answer(image + docs) != answer(image alone):  suspect the documents

and three policies act on that signal:

* ``abstain``        -- refuse when the two answers disagree (selective QA)
* ``trust_image``    -- fall back to the image-only answer
* ``confidence``     -- keep whichever answer the model is more confident in
                        (multiple-choice probe probability)

They are evaluated where the documents lie (conflict, heavy distractors), where
they are right but the image is degraded (Sensory-Capture cells -- falling back
to the image *hurts* there) and on clean inputs, so the trade-off is explicit.
Signal quality is reported as AUROC for predicting errors, with a drop-one
ablation of each signal's unique contribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import stats as S
from .metrics import ModelView

# (condition with docs, its no-document counterpart on the same image)
GATE_CONDITIONS = [
    ("clean", "no_ctx"), ("ret_k5", "no_ctx"), ("ret_k20", "no_ctx"), ("conflict", "no_ctx"),
    ("conflict_g16", "vis_gauss_s16"), ("grid_g16_k0", "vis_gauss_s16"), ("grid_g16_k20", "vis_gauss_s16"),
]


def gate(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cond, base in GATE_CONDITIONS:
        if cond not in v._qa["correct"].columns or base not in v._qa["correct"].columns:
            continue
        a_c, a_0 = v.extracted(cond), v.extracted(base)
        truth = v.qa(cond)  # correctness of the with-docs answer
        corr_0 = v.qa(base)
        conf_c, conf_0 = v.mc(cond, "choice_conf"), v.mc(base, "choice_conf")
        ok = np.isfinite(truth) & np.isfinite(corr_0)
        disagree = np.array([x != y for x, y in zip(a_c, a_0)], float)
        answered = np.where(ok, 1.0 - disagree, 0.0)
        trust_img = np.where(disagree > 0, corr_0, truth)
        pick_0 = (conf_0 > conf_c) & np.isfinite(conf_0) & np.isfinite(conf_c)
        confd = np.where((disagree > 0) & pick_0, corr_0, truth)
        base_acc = S.mean_stat(np.where(ok, truth, np.nan), W)
        res = {
            "base_accuracy": S.strip(base_acc),
            "disagreement_rate": S.strip(S.mean_stat(np.where(ok, disagree, np.nan), W)),
            "abstain": {"coverage": S.strip(S.ratio_stat(answered, ok.astype(float), W)),
                        "selective_accuracy": S.strip(S.ratio_stat(answered * np.nan_to_num(truth), answered, W))},
        }
        for name, arr in (("trust_image", trust_img), ("confidence", confd)):
            st = S.mean_stat(np.where(ok, arr, np.nan), W)
            res[name] = {"accuracy": S.strip(st),
                         "delta": {"value": st["value"] - base_acc["value"], "ci": S.ci(st["_boot"] - base_acc["_boot"])}}
        out[cond] = res
    return out


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + l2 * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def _cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> Optional[float]:
    uniq = np.unique(groups)
    if len(uniq) < folds or len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(0)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % folds))
    f = np.array([fold_of[g] for g in groups])
    scores = np.full(len(y), np.nan)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    for k in range(folds):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2:
            continue
        w = _fit_logistic(Xs[tr], y[tr])
        scores[te] = np.column_stack([np.ones(te.sum()), Xs[te]]) @ w
    return S.auroc(scores, y)


def error_signals(v: ModelView) -> Dict[str, Any]:
    """Which test-time signals predict a wrong free-form answer? (pooled over conditions with docs)"""
    rows = []
    for cond, base in GATE_CONDITIONS:
        if cond not in v._qa["correct"].columns or base not in v._qa["correct"].columns:
            continue
        a_c, a_0 = v.extracted(cond), v.extracted(base)
        err = 1 - v.qa(cond)
        conf = v.mc(cond, "choice_conf")
        conf_shift = np.abs(np.log(np.clip(conf, 1e-6, 1)) - np.log(np.clip(v.mc(base, "choice_conf"), 1e-6, 1)))
        share = v.qa(cond, "vis_share")
        dis = np.array([x != y for x, y in zip(a_c, a_0)], float)
        for i in range(v.n):
            rows.append((i, err[i], 1 - conf[i], dis[i], conf_shift[i], -share[i] if np.isfinite(share[i]) else np.nan))
    if not rows:
        return {"available": False}
    A = np.array(rows, float)
    names = ["low_confidence", "context_disagreement", "confidence_shift", "low_visual_attention"]
    out: Dict[str, Any] = {"available": True, "n": int(np.isfinite(A[:, 1]).sum()), "auroc": {}}
    for j, nm in enumerate(names):
        out["auroc"][nm] = S.auroc(A[:, 2 + j], A[:, 1])
    use = [j for j in range(len(names)) if np.isfinite(A[:, 2 + j]).mean() > 0.9]
    ok = np.isfinite(A[:, 1]) & np.all(np.isfinite(A[:, [2 + j for j in use]]), axis=1)
    if use and ok.sum() > 50:
        X, y, g = A[ok][:, [2 + j for j in use]], A[ok, 1], A[ok, 0].astype(int)
        full = _cv_auroc(X, y, g)
        out["combined_cv_auroc"] = full
        out["drop_one"] = {}
        for jj, j in enumerate(use):
            keep = [c for c in range(len(use)) if c != jj]
            if keep:
                red = _cv_auroc(X[:, keep], y, g)
                out["drop_one"][names[j]] = (full - red) if (full is not None and red is not None) else None
    return out
