"""All per-model metrics, each with an item-cluster bootstrap CI.

One bootstrap weight matrix W (iters x items) is shared by every metric and
every model, so derived quantities (R, gaps, model differences) are bootstrapped
jointly and paired comparisons are exact. Definitions are in docs/METRICS.md.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats as S


def _rows(df: pd.DataFrame, probe: str, cols: Sequence[str]) -> pd.DataFrame:
    """One probe's rows, only the columns used, with categorical text as plain objects (they share
    the category strings, ~8 bytes a row): pivots and group-bys behave as on an ordinary table."""
    d = df.loc[df["probe"] == probe, [c for c in cols if c in df.columns]]
    return d.assign(**{c: d[c].astype(object) for c in d.columns if isinstance(d[c].dtype, pd.CategoricalDtype)})


def _logit(p: np.ndarray, clip: float) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


class ModelView:
    """Per-item arrays (aligned to a fixed item order) for one model."""

    def __init__(self, df: pd.DataFrame, item_ids: Sequence[str], cfg: Dict[str, Any]):
        self.ids = list(item_ids)
        self.n = len(self.ids)
        self.cfg = cfg
        self.clip = cfg["analysis"]["logodds_clip"]
        qa_vals = ("correct", "follows_claim", "copied_wrong_doc", "off_vocab", "vis_share")
        qa = _rows(df, "qa", ("gib_id", "condition", "axis", "extracted", "failure", "attn_img", "attn_post") + qa_vals)
        self._qa = {c: qa.pivot_table(index="gib_id", columns="condition", values=c, aggfunc="mean")
                    .reindex(index=self.ids) for c in qa_vals}
        self._qa_ext = (qa.pivot_table(index="gib_id", columns="condition", values="extracted", aggfunc="first")
                        .reindex(index=self.ids) if len(qa) else pd.DataFrame(index=self.ids))
        self._qa_attn = {(g, c): (a, b) for g, c, a, b in zip(qa["gib_id"], qa["condition"], qa["attn_img"], qa["attn_post"])
                         if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray))}
        self._qa_fail = qa[["axis", "failure"]]  # taxonomy
        mc_vals = ("choice_conf", "choice_correct", "p_truth", "p_claim")
        mc = _rows(df, "mc", ("gib_id", "condition") + mc_vals)
        self._mc = {c: mc.pivot_table(index="gib_id", columns="condition", values=c, aggfunc="mean")
                    .reindex(index=self.ids) for c in mc_vals}
        ab = _rows(df, "ab", ("gib_id", "condition", "order", "p_claim"))
        self._ab = ab.pivot_table(index="gib_id", columns="condition", values="p_claim", aggfunc="mean").reindex(index=self.ids)
        if len(ab):
            ab = ab.assign(order=pd.to_numeric(ab["order"]).astype(int))
            self._ab_o = ab.pivot_table(index="gib_id", columns=["condition", "order"], values="p_claim",
                                        aggfunc="mean").reindex(index=self.ids)
        else:
            self._ab_o = pd.DataFrame(index=self.ids)
        self._pope = _rows(df, "pope", ("gib_id", "image_tag", "polarity", "p_yes"))
        self._cap = _rows(df, "caption", ("gib_id", "image_tag", "n_mentions", "n_halluc"))

    # ---------------- accessors ---------------- #
    def qa(self, cond: str, col: str = "correct") -> np.ndarray:
        t = self._qa[col]
        return t[cond].to_numpy(float) if cond in t else np.full(self.n, np.nan)

    def mc(self, cond: str, col: str) -> np.ndarray:
        t = self._mc[col]
        return t[cond].to_numpy(float) if cond in t else np.full(self.n, np.nan)

    def ab(self, cond: str) -> np.ndarray:
        """P(choose text claim), averaged over the two option orders."""
        return self._ab[cond].to_numpy(float) if cond in self._ab else np.full(self.n, np.nan)

    def ab_order(self, cond: str, order: int) -> np.ndarray:
        """P(choose text claim) for one option order (the two orders are replicate measurements)."""
        key = (cond, order)
        return self._ab_o[key].to_numpy(float) if key in self._ab_o.columns else np.full(self.n, np.nan)

    def extracted(self, cond: str) -> List[Optional[str]]:
        if cond not in self._qa_ext:
            return [None] * self.n
        return [v if isinstance(v, str) else None for v in self._qa_ext[cond]]

    def conditions(self, prefix: str) -> List[str]:
        return sorted(c for c in self._qa["correct"].columns if c.startswith(prefix))


# --------------------------------------------------------------------------- #
def _stat(x: Dict[str, Any]) -> Dict[str, Any]:
    return S.strip(x)


def accuracy_by_condition(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    return {c: _stat(S.mean_stat(v.qa(c), W)) for c in v._qa["correct"].columns}


def adverse_accuracy(v: ModelView) -> np.ndarray:
    """Axis-balanced per-item adverse accuracy (retrieval, visual, conflict each 1/3)."""
    parts = []
    for prefix in ("ret_k", "vis_"):
        cols = v.conditions(prefix)
        if cols:
            parts.append(np.nanmean(np.column_stack([v.qa(c) for c in cols]), axis=1))
    parts.append(v.qa("conflict"))
    with np.errstate(all="ignore"):
        return np.nanmean(np.column_stack(parts), axis=1)


def ece_adverse(v: ModelView, W: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """ECE of the multiple-choice probe on adverse conditions (axis-balanced row weights)."""
    groups = [v.conditions("ret_k"), v.conditions("vis_"), ["conflict"]]
    C = np.zeros((v.n, n_bins)); A = np.zeros((v.n, n_bins)); K = np.zeros((v.n, n_bins))
    cw_num = np.zeros(v.n); cw_den = np.zeros(v.n)
    thr = v.cfg["analysis"]["confident_wrong"]
    for conds in groups:
        conds = [c for c in conds if c in v._mc["choice_conf"].columns]
        if not conds:
            continue
        wt = 1.0 / len(conds)
        for c in conds:
            conf, corr = v.mc(c, "choice_conf"), v.mc(c, "choice_correct")
            ok = np.isfinite(conf) & np.isfinite(corr)
            b = np.clip((np.nan_to_num(conf) * n_bins).astype(int), 0, n_bins - 1)
            i = np.where(ok)[0]
            np.add.at(C, (i, b[i]), wt * conf[i]); np.add.at(A, (i, b[i]), wt * corr[i]); np.add.at(K, (i, b[i]), wt)
            cw_num += np.where(ok, wt * ((conf > thr) & (corr == 0)), 0.0)
            cw_den += np.where(ok, wt, 0.0)
    tot = K.sum(1)
    point = float(np.abs(C.sum(0) - A.sum(0)).sum() / tot.sum()) if tot.sum() > 0 else float("nan")
    dt = W.dtype  # a float32 W (large runs) must not be upcast: that would copy the whole matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.abs(W @ C.astype(dt) - W @ A.astype(dt)).sum(1) / (W @ tot.astype(dt))
    return {"ece": {"value": point, "ci": S.ci(boot), "n": int((tot > 0).sum()), "_boot": boot},
            "confident_wrong": S.ratio_stat(cw_num, cw_den, W)}


def robustness(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    clean = S.mean_stat(v.qa("clean"), W)
    adv = S.mean_stat(adverse_accuracy(v), W)
    e = ece_adverse(v, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        R_boot = adv["_boot"] / clean["_boot"] * (1 - e["ece"]["_boot"])
    R = adv["value"] / clean["value"] * (1 - e["ece"]["value"]) if clean["value"] else float("nan")
    return {"acc_clean": _stat(clean), "acc_adverse": _stat(adv), "ece_adverse": _stat(e["ece"]),
            "confident_wrong_rate": _stat(e["confident_wrong"]),
            "R": {"value": float(R), "ci": S.ci(R_boot), "n": clean["n"]},
            "_boot": {"clean": clean["_boot"], "adverse": adv["_boot"], "R": R_boot}}


def trust(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    clip = v.clip
    p = v.ab("ab_both_g0")
    lo = _logit(p, clip)
    ti = S.mean_stat(lo, W)
    p_clean = v.ab("ab_clean")
    out = {
        "trust_index": {"value": float(math.exp(ti["value"])) if np.isfinite(ti["value"]) else float("nan"),
                        "ci": [float(math.exp(x)) if np.isfinite(x) else float("nan") for x in ti["ci"]],
                        "n": ti["n"], "log_value": ti["value"]},
        "ab_text_follow_rate": _stat(S.mean_stat(np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan), W)),
        "qa_text_follow_rate": _stat(S.mean_stat(v.qa("conflict", "follows_claim"), W)),
        "ab_clean_claim_rate": _stat(S.mean_stat(np.where(np.isfinite(p_clean), (p_clean > 0.5).astype(float), np.nan), W)),
    }
    # Grounding Illusion: correct with gold text, but parrots the text when it lies
    clean_ok = v.qa("clean")
    foll = v.qa("conflict", "follows_claim")
    conf_ok = v.qa("conflict")
    m = np.isfinite(clean_ok) & np.isfinite(foll)
    gis = S.ratio_stat(np.where(m, clean_ok * foll, 0), np.where(m, clean_ok, 0), W)
    grounded = S.mean_stat(np.where(m, clean_ok * conf_ok, np.nan), W)
    clean = S.mean_stat(np.where(m, clean_ok, np.nan), W)
    gap_boot = clean["_boot"] - grounded["_boot"]
    out.update({"GIS": _stat(gis), "grounded_accuracy": _stat(grounded),
                "illusion_gap": {"value": clean["value"] - grounded["value"], "ci": S.ci(gap_boot), "n": clean["n"]}})
    # ReFinD-style context sensitivity at option level
    def ratio(num_c: str, den_c: str, col: str) -> Dict[str, Any]:
        a, b = v.mc(num_c, col), v.mc(den_c, col)
        d = np.log(np.clip(a, 1e-9, 1)) - np.log(np.clip(b, 1e-9, 1))
        st = S.mean_stat(d, W)
        return {"value": float(math.exp(st["value"])) if np.isfinite(st["value"]) else float("nan"),
                "ci": [float(math.exp(x)) if np.isfinite(x) else float("nan") for x in st["ci"]], "n": st["n"]}
    out["CSR_gold"] = ratio("clean", "no_ctx", "p_truth")
    out["CSR_misinformation"] = ratio("conflict", "no_ctx", "p_claim")
    return out


def _eiv(y, X, dy, dX, W: np.ndarray, cols: np.ndarray) -> Dict[str, np.ndarray]:
    """The EIV fit for every bootstrap row of W (restricted to ``cols``), a few rows at a time:
    W[:, cols] in one go is iters x observations, several GB at 200k items."""
    chunk = max(1, min(250, int(5e6 // max(1, len(cols)))))
    parts = [S.eiv_regression(y, X, dy, dX, W[i:i + chunk][:, cols]) for i in range(0, len(W), chunk)]
    return {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}


def cue_combination(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    """Ideal-observer test of how image and text evidence are combined.

    With log-odds LO(.) of choosing the text's claim over the image's answer
    (binary A/B probe), an ideal Bayesian integrator of conditionally
    independent cues satisfies

        LO(image+text) - LO(prior) = [LO(image) - LO(prior)] + [LO(text) - LO(prior)],

    where LO(text) is measured with a blank image and LO(prior) with a blank
    image and no document. We fit y = w_image * x_image + w_text * x_text over
    items x blur levels (no intercept: the ideal model has none, and subtracting
    LO(prior) measured in the same option order cancels constant position bias);
    the ideal is w_image = w_text = 1. Because x_image
    and x_text are themselves noisy measurements, plain least squares would
    shrink both weights towards 0 (regression dilution); the two option orders
    are used as replicate measurements to remove that bias (stats.eiv_regression).
    """
    clip = v.clip

    def lo(cond: str, o: int) -> np.ndarray:
        return _logit(v.ab_order(cond, o), clip)

    sigmas = sorted({float(c.split("_g")[1]) for c in v._ab.columns if c.startswith("ab_both_g")})
    ys, xs, dys, dxs, item_idx, sig = [], [], [], [], [], []
    for s_ in sigmas:
        reps = []
        for o in (0, 1):
            prior = lo("ab_prior", o)
            reps.append((lo(f"ab_both_g{s_:g}", o) - prior, lo(f"ab_img_g{s_:g}", o) - prior, lo("ab_text", o) - prior))
        (y0, i0, t0), (y1, i1, t1) = reps
        ok = np.all(np.isfinite(np.column_stack([y0, i0, t0, y1, i1, t1])), axis=1)
        idx = np.where(ok)[0]
        ys.append(((y0 + y1) / 2)[ok]); dys.append((y0 - y1)[ok])
        xs.append(np.column_stack([(i0 + i1) / 2, (t0 + t1) / 2])[ok])
        dxs.append(np.column_stack([i0 - i1, t0 - t1])[ok])
        item_idx.append(idx); sig += [s_] * len(idx)
    if not ys or sum(len(a) for a in ys) < 10:
        return {"available": False}
    y, X = np.concatenate(ys), np.concatenate(xs)
    dy, dX = np.concatenate(dys), np.concatenate(dxs)
    idx, sig = np.concatenate(item_idx), np.array(sig)
    point = S.eiv_regression(y, X, dy, dX, np.ones((1, len(y))))
    boot = _eiv(y, X, dy, dX, W, idx)
    wi, wt = float(point["slopes"][0, 0]), float(point["slopes"][0, 1])
    naive = np.linalg.lstsq(X, y, rcond=None)[0]
    ideal = X[:, 0] + X[:, 1]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_ideal = 1 - ((y - ideal) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    per_sigma = {}
    for s_ in sigmas:
        m = sig == s_
        if m.sum() >= 10:
            pp = S.eiv_regression(y[m], X[m], dy[m], dX[m], np.ones((1, int(m.sum()))))
            bb = _eiv(y[m], X[m], dy[m], dX[m], W, idx[m])
            per_sigma[f"{s_:g}"] = {"w_image": float(pp["slopes"][0, 0]), "w_text": float(pp["slopes"][0, 1]),
                                    "w_image_ci": S.ci(bb["slopes"][:, 0]), "w_text_ci": S.ci(bb["slopes"][:, 1]),
                                    "n": int(m.sum())}
    # integration errors relative to the model's *own* unimodal evidence
    per_item_ki = np.full(v.n, np.nan); per_item_sc = np.full(v.n, np.nan)
    for arr, dst in ((((y > 0) & (ideal < 0)).astype(float), per_item_ki), (((y < 0) & (ideal > 0)).astype(float), per_item_sc)):
        s2 = pd.Series(arr).groupby(idx).mean()
        dst[s2.index.to_numpy()] = s2.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_b = boot["slopes"][:, 1] / boot["slopes"][:, 0]
    return {
        "available": True, "n_obs": int(len(y)), "n_items": int(len(np.unique(idx))),
        "w_image": {"value": wi, "ci": S.ci(boot["slopes"][:, 0])},
        "w_text": {"value": wt, "ci": S.ci(boot["slopes"][:, 1])},
        "text_dominance": {"value": wt / wi if wi else float("nan"), "ci": S.ci(ratio_b)},
        "reliability": {"image": float(point["reliability"][0, 0]), "text": float(point["reliability"][0, 1])},
        "uncorrected_ols": {"w_image": float(naive[0]), "w_text": float(naive[1])},
        "r2_ideal_additive": float(r2_ideal), "per_sigma": per_sigma,
        "integration_KI_rate": _stat(S.mean_stat(per_item_ki, W)),
        "integration_SC_rate": _stat(S.mean_stat(per_item_sc, W)),
    }


def flip_decomposition(v: ModelView, items: Dict[str, Any], iters: int, seed: int) -> Dict[str, Any]:
    p = pd.Series(v.ab("ab_both_g0"), index=v.ids)
    pairs = []
    for g in v.ids:
        it = items[g]
        j = it.flip_partner
        if j and j in p.index and g < j and np.isfinite(p[g]) and np.isfinite(p[j]):
            pairs.append((g, j))
    if not pairs:
        return {"available": False}
    cats, winners = [], []
    for g, j in pairs:
        ci_, cj = p[g] > 0.5, p[j] > 0.5  # True = followed the text
        if ci_ and cj:
            cats.append("text_consistent"); winners.append(None)
        elif not ci_ and not cj:
            cats.append("image_consistent"); winners.append(None)
        else:
            cats.append("concept_prior")
            winners.append(items[g].answer if not ci_ else items[g].conflict_claim)
    cats = np.array(cats)
    Wp = S.boot_weights(len(pairs), iters, seed + 7)
    out: Dict[str, Any] = {"available": True, "n_pairs": len(pairs)}
    for c in ("text_consistent", "image_consistent", "concept_prior"):
        out[c] = _stat(S.mean_stat((cats == c).astype(float), Wp))
    mb = (cats == "text_consistent").astype(float) - (cats == "image_consistent").astype(float)
    out["modality_bias"] = _stat(S.mean_stat(mb, Wp))
    by_group: Dict[str, Dict[str, float]] = {}
    for (g, _), c in zip(pairs, cats):
        grp = items[g].group
        by_group.setdefault(grp, {"n": 0, "text_consistent": 0, "image_consistent": 0, "concept_prior": 0})
        by_group[grp]["n"] += 1
        by_group[grp][c] += 1
    out["by_group"] = {k: {kk: (vv / d["n"] if kk != "n" else vv) for kk, vv in d.items()} for k, d in by_group.items()}
    wins = pd.Series([w for w in winners if w]).value_counts()
    out["prior_winners_top"] = {k: int(n) for k, n in wins.head(10).items()}
    return out


def attention_shift(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    """Paired, per-layer visual-share change from clean to conflict (same image, same doc template)."""
    def shares(cond: str) -> Optional[np.ndarray]:
        rows = []
        L = None
        for g in v.ids:
            a = v._qa_attn.get((g, cond))
            if a is None:
                rows.append(None)
                continue
            img, post = np.asarray(a[0]), np.asarray(a[1])
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append(img / (img + post))
            L = len(img)
        if L is None:
            return None
        return np.array([r if r is not None and len(r) == L else np.full(L, np.nan) for r in rows])

    sc, sf = shares("clean"), shares("conflict")
    if sc is None or sf is None:
        return {"available": False}
    delta = sc - sf  # > 0: attention moved off the image when the text lied
    L = delta.shape[1]
    prof = [S.mean_stat(delta[:, l], W) for l in range(L)]
    mean_delta = S.mean_stat(np.nanmean(delta, axis=1), W)
    thirds = np.array_split(np.arange(L), 3)
    bands = {name: _stat(S.mean_stat(np.nanmean(delta[:, idx], axis=1), W))
             for name, idx in zip(("early", "middle", "late"), thirds)}
    out = {"available": True, "n_layers": L, "ADS": _stat(mean_delta), "bands": bands,
           "profile": [{"value": p["value"], "ci": p["ci"]} for p in prof],
           "clean_share_profile": np.nanmean(sc, axis=0).round(5).tolist(),
           "conflict_share_profile": np.nanmean(sf, axis=0).round(5).tolist()}
    # is attention collapse associated with Knowledge Inertia?
    foll = v.qa("conflict", "follows_claim")
    ok = np.isfinite(foll) & np.isfinite(np.nanmean(sf, axis=1))
    if ok.sum() > 5 and 0 < foll[ok].mean() < 1:
        score = -np.nanmean(sf, axis=1)
        out["auroc_low_share_predicts_KI"] = S.auroc(score[ok], foll[ok])
        ki, gr = sf[ok & (foll == 1)], sf[ok & (foll == 0)]
        out["share_profile_KI"] = np.nanmean(ki, axis=0).round(5).tolist()
        out["share_profile_grounded"] = np.nanmean(gr, axis=0).round(5).tolist()
        out["n_KI"], out["n_grounded"] = int(len(ki)), int(len(gr))
    return out


def decay(v: ModelView, W: np.ndarray) -> Dict[str, Any]:
    ks = [0] + sorted(int(c[5:]) for c in v.conditions("ret_k"))
    if len(ks) < 3:
        return {"available": False}
    cols = [v.qa("clean")] + [v.qa(f"ret_k{k}") for k in ks[1:]]
    num = np.column_stack([np.nan_to_num(c) for c in cols])
    den = np.column_stack([np.isfinite(c).astype(float) for c in cols])
    acc = num.sum(0) / den.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_b = (W @ num) / (W @ den)
    x = np.log2(1 + np.array(ks, float))
    xc = x - x.mean()

    def slope(a: np.ndarray) -> np.ndarray:
        lg = _logit(a, 10.0)
        return (lg - lg.mean(-1, keepdims=True)) @ xc / (xc @ xc)

    sl, sl_b = float(slope(acc)), slope(acc_b)
    auc = float(np.trapezoid(acc, x) / (acc[0] * (x[-1] - x[0]))) if acc[0] > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        auc_b = np.trapezoid(acc_b, x, axis=1) / (acc_b[:, 0] * (x[-1] - x[0]))
    lg = _logit(acc, 10.0)
    pred = lg.mean() + sl * xc
    r2 = 1 - ((lg - pred) ** 2).sum() / max(((lg - lg.mean()) ** 2).sum(), 1e-12)
    return {"available": True, "k": ks, "accuracy": acc.round(5).tolist(),
            "accuracy_ci": [S.ci(acc_b[:, j]) for j in range(len(ks))],
            "decay_rate": {"value": sl, "ci": S.ci(sl_b), "unit": "log-odds per doubling of (1+k)"},
            "retained_auc": {"value": auc, "ci": S.ci(auc_b)}, "fit_r2": float(r2)}


def resolution(v: ModelView, W: np.ndarray, families: Dict[str, List[float]], jump: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    base = v.qa("no_ctx")
    cap = v._cap
    for fam, levels in families.items():
        tags = [f"{fam}_{'s' if fam in ('gauss', 'noise') else 'k' if fam == 'motion' else 'f'}{lv:g}" for lv in levels]
        conds = [f"vis_{t}" for t in tags]
        cols = [base] + [v.qa(c) for c in conds]
        num = np.column_stack([np.nan_to_num(c) for c in cols])
        den = np.column_stack([np.isfinite(c).astype(float) for c in cols])
        acc = np.where(den.sum(0) > 0, num.sum(0) / np.maximum(den.sum(0), 1), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc_b = (W @ num) / (W @ den)
        res = {"levels": list(levels), "accuracy": acc.round(5).tolist(),
               "accuracy_ci": [S.ci(acc_b[:, j]) for j in range(len(cols))]}

        def first_jump(curve: np.ndarray, sign: float) -> np.ndarray:
            d = sign * (curve[..., 1:] - curve[..., :1])
            hit = d > jump
            idx = np.where(hit.any(-1), hit.argmax(-1), len(levels))
            return idx

        ia = first_jump(acc[None, :], -1.0)[0]
        ia_b = first_jump(acc_b, -1.0)
        res["acc_threshold"] = {"level": levels[ia] if ia < len(levels) else None,
                                "ci_levels": [levels[int(i)] if i < len(levels) else None
                                              for i in np.quantile(ia_b, [0.025, 0.975]).round().astype(int)]}
        if len(cap):
            ch_num, ch_den = [], []
            for t in ["orig"] + tags:
                sub = cap[cap["image_tag"] == t].set_index("gib_id").reindex(v.ids)
                ch_num.append(sub["n_halluc"].fillna(0).to_numpy(float))
                ch_den.append(sub["n_mentions"].fillna(0).to_numpy(float))
            ch_num, ch_den = np.column_stack(ch_num), np.column_stack(ch_den)
            if ch_den.sum() > 0:
                chair = ch_num.sum(0) / np.maximum(ch_den.sum(0), 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    chair_b = (W @ ch_num) / (W @ ch_den)
                res["chair_i"] = chair.round(5).tolist()
                res["chair_i_ci"] = [S.ci(chair_b[:, j]) for j in range(chair.size)]
                ic = first_jump(chair[None, :], 1.0)[0]
                ic_b = first_jump(chair_b, 1.0)
                res["chair_threshold"] = {"level": levels[ic] if ic < len(levels) else None,
                                          "ci_levels": [levels[int(i)] if i < len(levels) else None
                                                        for i in np.quantile(ic_b, [0.025, 0.975]).round().astype(int)]}
        out[fam] = res
    return out


def interaction(v: ModelView, W: np.ndarray, sigmas: Sequence[float], dosages: Sequence[int]) -> Dict[str, Any]:
    clean = v.qa("clean")
    cells, heat, heat_ci = [], {}, {}
    for s in sigmas:
        base_s = v.qa(f"grid_g{s:g}_k0")
        for k in dosages:
            if k == 0:
                continue
            ik = (v.qa(f"grid_g{s:g}_k{k}") - base_s) - (v.qa(f"ret_k{k}") - clean)
            cells.append(ik)
    if not cells:
        return {"available": False}
    per_item = np.nanmean(np.column_stack(cells), axis=1) if cells else np.full(v.n, np.nan)
    st = S.mean_stat(-per_item, W)  # > 0 = combined damage exceeds the sum of single-axis damage
    for s in [0] + list(sigmas):
        for k in dosages:
            if s == 0:
                cond = "clean" if k == 0 else f"ret_k{k}"
            else:
                cond = f"grid_g{s:g}_k{k}"
            d = S.mean_stat(clean - v.qa(cond), W)
            heat[f"{s:g}|{k}"] = d["value"]
            heat_ci[f"{s:g}|{k}"] = d["ci"]
    return {"available": True, "super_additive_drop": _stat(st), "p_sign_flip": S.sign_flip_p(per_item),
            "heatmap_drop": heat, "heatmap_drop_ci": heat_ci, "sigmas": [0] + list(sigmas), "dosages": list(dosages),
            "text_rescue": {f"g{s:g}": _stat(S.mean_stat(v.qa(f"grid_g{s:g}_k0") - v.qa(f"vis_gauss_s{s:g}"), W))
                            for s in sigmas}}


def pope(v: ModelView, W: np.ndarray, tag: str = "orig") -> Dict[str, Any]:
    d = v._pope[v._pope["image_tag"] == tag]
    if d.empty:
        return {"available": False}
    pv = d.pivot_table(index="gib_id", columns="polarity", values="p_yes", aggfunc="mean").reindex(v.ids)
    pos = pv["pos"].to_numpy(float) if "pos" in pv else np.full(v.n, np.nan)
    neg = pv["neg"].to_numpy(float) if "neg" in pv else np.full(v.n, np.nan)
    tp = np.where(np.isfinite(pos), (pos > 0.5).astype(float), 0); fn = np.where(np.isfinite(pos), (pos <= 0.5).astype(float), 0)
    fp = np.where(np.isfinite(neg), (neg > 0.5).astype(float), 0); tn = np.where(np.isfinite(neg), (neg <= 0.5).astype(float), 0)
    acc = S.ratio_stat(tp + tn, tp + fn + fp + tn, W)
    prec = S.ratio_stat(tp, tp + fp, W)
    rec = S.ratio_stat(tp, tp + fn, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_b = 2 * prec["_boot"] * rec["_boot"] / (prec["_boot"] + rec["_boot"])
    f1 = 2 * prec["value"] * rec["value"] / (prec["value"] + rec["value"]) if (prec["value"] + rec["value"]) else float("nan")
    return {"available": True, "accuracy": _stat(acc), "precision": _stat(prec), "recall": _stat(rec),
            "f1": {"value": float(f1), "ci": S.ci(f1_b)}, "yes_ratio": _stat(S.ratio_stat(tp + fp, tp + fn + fp + tn, W))}


def chair(v: ModelView, W: np.ndarray, tag: str = "orig") -> Dict[str, Any]:
    d = v._cap[v._cap["image_tag"] == tag].set_index("gib_id").reindex(v.ids)
    if d["n_mentions"].isna().all():
        return {"available": False}
    h = d["n_halluc"].fillna(0).to_numpy(float)
    m = d["n_mentions"].fillna(0).to_numpy(float)
    has = d["n_mentions"].notna().to_numpy(float)
    return {"available": True, "CHAIR_i": _stat(S.ratio_stat(h, m, W)),
            "CHAIR_s": _stat(S.ratio_stat((h > 0).astype(float) * has, has, W)), "n_captions": int(has.sum())}


def taxonomy(v: ModelView) -> Dict[str, Any]:
    qa = v._qa_fail  # the failure class is decided when the table is built (eval/taxonomy.py)
    out: Dict[str, Any] = {}
    for axis, sub in qa.groupby("axis"):
        f = sub["failure"].dropna()
        out[axis] = {"n": int(len(sub)), "errors": int(len(f)),
                     "share": {k: round(float(c) / len(f), 5) for k, c in f.value_counts().items()} if len(f) else {}}
    f = qa["failure"].dropna()
    out["all"] = {"n": int(len(qa)), "errors": int(len(f)),
                  "share": {k: round(float(c) / len(f), 5) for k, c in f.value_counts().items()} if len(f) else {}}
    return out


def contamination(v: ModelView, items: Dict[str, Any], W: np.ndarray) -> Dict[str, Any]:
    split = np.array([items[g].source_split for g in v.ids])
    splits = sorted(set(split))
    if len(splits) < 2:
        return {"available": False, "splits": splits}
    clean, adv, foll = v.qa("clean"), adverse_accuracy(v), v.qa("conflict", "follows_claim")
    out: Dict[str, Any] = {"available": True, "splits": {}}
    boots = {}
    for sp in splits:
        m = split == sp
        res = {}
        for name, arr in (("acc_clean", clean), ("acc_adverse", adv), ("qa_text_follow_rate", foll)):
            st = S.mean_stat(np.where(m, arr, np.nan), W)
            res[name] = _stat(st)
            boots[(sp, name)] = st["_boot"]
        res["n_items"] = int(m.sum())
        out["splits"][sp] = res
    if len(splits) == 2:
        a, b = splits
        out["difference"] = {name: {"value": out["splits"][a][name]["value"] - out["splits"][b][name]["value"],
                                    "ci": S.ci(boots[(a, name)] - boots[(b, name)]), "contrast": f"{a} - {b}"}
                             for name in ("acc_clean", "acc_adverse", "qa_text_follow_rate")}
    return out


# Metrics that read graded probabilities from a probe, not just which option won. POPE, the flip
# decomposition and the A/B follow rates only use the argmax, so parsed one-hot answers are fine there.
PROB_DEPENDENT: Dict[str, List[str]] = {
    "mc": ["robustness.ece_adverse", "robustness.R", "robustness.confident_wrong_rate", "trust.CSR_gold",
           "trust.CSR_misinformation", "gate.confidence", "error_signals"],
    "ab": ["trust.trust_index", "cue_combination"],
}


def probability_quality(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """How much of each probability probe rests on real token probabilities (one model's rows).

    An answer is *usable* when its distribution came from token log-probabilities (not a
    parsed text answer scored as one-hot) and at least ``min_format_mass`` of the probability
    fell on the valid answer tokens; below that, the renormalised distribution describes
    tokens the model barely considered. Unknown mass counts as usable. When a probe's usable
    share is below ``min_prob_coverage``, its ``PROB_DEPENDENT`` metrics are flagged.
    """
    a = cfg["analysis"]
    min_cov, min_mass = a["min_prob_coverage"], a["min_format_mass"]
    out: Dict[str, Any] = {"min_coverage": min_cov, "min_format_mass": min_mass, "probes": {}, "flagged_metrics": []}
    for probe in ("mc", "ab", "pope"):
        d = df.loc[df["probe"] == probe, ["has_probs", "from_logprobs", "format_mass"]]
        if d.empty:
            continue
        has = d["has_probs"].fillna(0).to_numpy() > 0
        real = (d["from_logprobs"].fillna(0).to_numpy() > 0) & has
        mass = d["format_mass"].to_numpy(float)
        low = real & (mass < min_mass)  # unknown (NaN) mass is not low
        usable = float((real & ~low).mean())
        out["probes"][probe] = {
            "n": int(len(d)), "with_probs": float(has.mean()), "from_logprobs": float(real.mean()),
            "format_mass_mean": float(np.nanmean(mass[real])) if np.isfinite(mass[real]).any() else None,
            "low_mass": float(low.mean()), "usable": usable, "ok": usable >= min_cov,
            "affects": PROB_DEPENDENT.get(probe, [])}
        if usable < min_cov:
            out["flagged_metrics"] += PROB_DEPENDENT.get(probe, [])
    out["ok"] = not out["flagged_metrics"]
    return out
