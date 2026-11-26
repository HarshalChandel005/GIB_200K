"""Statistics with the right sampling unit: the *item*.

A between-groups ANOVA is wrong here because every model answers the *same*
items (repeated measures), and degrees of freedom must match the sample size. Here:

* confidence intervals come from a **cluster bootstrap** (all queries of an item --
  and all items of an image, when an image carries several -- are resampled
  together), vectorised as multinomial weight matrices;
* model differences use the **Friedman test** (items = blocks) with Kendall's W,
  plus a repeated-measures ANOVA whose df are (k-1, (k-1)(n-1));
* pairwise comparisons are paired item-bootstrap tests with Holm correction;
* every result carries the n it was computed on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


BIG = 50_000_000  # iters x n above this: float32, built a block of rows at a time


def boot_weights(n: int, iters: int, seed: int = 0) -> np.ndarray:
    """(iters x n) multinomial resampling counts over n clusters.

    Large matrices (2,000 x 200k would be 3.2 GB in float64, plus as much again
    while drawing) are float32 -- counts up to 2^24 are exact -- and drawn in
    blocks of rows from the same random stream.
    """
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.zeros((iters, 0))
    p = np.full(n, 1.0 / n)
    if iters * n <= BIG:
        return rng.multinomial(n, p, size=iters).astype(np.float64)
    out = np.empty((iters, n), dtype=np.float32)
    step = max(1, BIG // (8 * n))
    for i in range(0, iters, step):
        out[i:i + step] = rng.multinomial(n, p, size=min(step, iters - i))
    return out


def cluster_weights(clusters: Sequence[str], iters: int, seed: int = 0) -> np.ndarray:
    """Bootstrap weights per unit (column), resampling units that share a cluster label together."""
    labels, inv = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    if len(labels) == len(clusters):  # every unit its own cluster: plain unit-level weights
        return boot_weights(len(clusters), iters, seed)
    return boot_weights(len(labels), iters, seed)[:, inv]


def ci(samples: np.ndarray, level: float = 0.95) -> List[float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [float(np.quantile(s, a)), float(np.quantile(s, 1 - a))]


def ratio_stat(num: np.ndarray, den: np.ndarray, W: np.ndarray) -> Dict[str, object]:
    """sum(num)/sum(den) over clusters, with bootstrap CI."""
    num = np.nan_to_num(np.asarray(num, float))
    den = np.nan_to_num(np.asarray(den, float))
    point = num.sum() / den.sum() if den.sum() > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):  # in W's precision: never upcast a float32 W
        boot = (W @ num.astype(W.dtype, copy=False)) / (W @ den.astype(W.dtype, copy=False))
    return {"value": float(point), "ci": ci(boot), "n": int((den > 0).sum()), "_boot": boot}


def mean_stat(values: np.ndarray, W: np.ndarray) -> Dict[str, object]:
    """Mean of per-cluster values (NaN = cluster has no value)."""
    v = np.asarray(values, float)
    valid = np.isfinite(v).astype(float)
    return ratio_stat(np.where(np.isfinite(v), v, 0.0), valid, W)


def strip(stat: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in stat.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# repeated-measures comparison of k models on the same n items
# --------------------------------------------------------------------------- #
def friedman(matrix: np.ndarray) -> Dict[str, object]:
    """matrix: (n_items x k_models) per-item scores (complete rows only)."""
    from scipy import stats as st

    m = matrix[np.all(np.isfinite(matrix), axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return {"test": "friedman", "n": int(n), "k": int(k), "chi2": None, "p": None}
    if k >= 3:
        chi2, p = st.friedmanchisquare(*[m[:, j] for j in range(k)])
    else:
        res = st.wilcoxon(m[:, 0], m[:, 1], zero_method="zsplit")
        chi2, p = float(res.statistic), float(res.pvalue)
    w = float(chi2) / (n * (k - 1)) if k >= 3 else None
    return {"test": "friedman" if k >= 3 else "wilcoxon", "chi2": float(chi2), "df": k - 1,
            "p": float(p), "kendall_w": w, "n": int(n), "k": int(k)}


def rm_anova(matrix: np.ndarray) -> Dict[str, object]:
    """One-way repeated-measures ANOVA (subjects = items, factor = model)."""
    from scipy import stats as st

    m = matrix[np.all(np.isfinite(matrix), axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return {"F": None, "df1": None, "df2": None, "p": None, "n": int(n)}
    grand = m.mean()
    ss_model = n * ((m.mean(0) - grand) ** 2).sum()
    ss_subj = k * ((m.mean(1) - grand) ** 2).sum()
    ss_err = ((m - grand) ** 2).sum() - ss_model - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_model / df1) / (ss_err / df2) if ss_err > 0 else float("inf")
    p = float(st.f.sf(F, df1, df2)) if np.isfinite(F) else 0.0
    eta_p = ss_model / (ss_model + ss_err) if (ss_model + ss_err) > 0 else float("nan")
    return {"F": float(F), "df1": df1, "df2": df2, "p": p, "partial_eta2": float(eta_p), "n": int(n), "k": int(k)}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_boot_p(diff_boot: np.ndarray) -> float:
    d = diff_boot[np.isfinite(diff_boot)]
    if d.size == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def sign_flip_p(values: np.ndarray, iters: int = 5000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test that the mean of per-item values is 0."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(v.mean())
    step = max(1, int(2e7 // v.size))  # all 5,000 x 200k flips at once would be 8 GB
    hits = 0
    for i in range(0, iters, step):
        flips = rng.choice([-1.0, 1.0], size=(min(step, iters - i), v.size))
        hits += int((np.abs((flips * v).mean(1)) >= obs - 1e-12).sum())
    return float((1 + hits) / (iters + 1))


def wls_weights(y: np.ndarray, X: np.ndarray, w_obs: np.ndarray) -> np.ndarray:
    """Vectorised weighted least squares: w_obs (B x n) -> coefficients (B x p)."""
    XtW = np.einsum("bn,np->bpn", w_obs, X)
    A = np.einsum("bpn,nq->bpq", XtW, X)
    b = np.einsum("bpn,n->bp", XtW, y)
    out = np.full((w_obs.shape[0], X.shape[1]), np.nan)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return out


def eiv_regression(y: np.ndarray, X: np.ndarray, dy: np.ndarray, dX: np.ndarray,
                   w_obs: np.ndarray, intercept: bool = False) -> Dict[str, np.ndarray]:
    """Regression corrected for noise in the regressors, using two replicates.

    y, X are averages of two replicate measurements (here: the two option
    orders of the A/B probe) and dy, dX their differences. Replicate noise is
    assumed independent with mean zero, so the noise second moments of the
    averages are Cov(dX)/4 and Cov(dX, dy)/4 (the latter captures noise that X
    and y share, e.g. a common prior term). The slopes solve
    (M_XX - S_XX) b = (M_Xy - S_Xy) with M the (uncentred, or centred when
    ``intercept``) second moments -- the method-of-moments correction for
    regression dilution. Rows of ``w_obs`` are bootstrap weights (one row of
    ones gives the point estimate). Also returns each regressor's reliability
    (share of its variance that is signal rather than replicate noise).
    """
    w = w_obs / np.maximum(w_obs.sum(1, keepdims=True), 1e-12)          # (B, n)
    mx, my = w @ X, w @ y
    if intercept:
        Xc = X[None, :, :] - mx[:, None, :]
        yc = y[None, :] - my[:, None]
    else:
        Xc = np.broadcast_to(X, (w.shape[0],) + X.shape)
        yc = np.broadcast_to(y, (w.shape[0],) + y.shape)
    mxx = np.einsum("bn,bnp,bnq->bpq", w, Xc, Xc)
    mxy = np.einsum("bn,bnp,bn->bp", w, Xc, yc)
    md, mdy = w @ dX, w @ dy
    Dc = dX[None, :, :] - md[:, None, :]
    dyc = dy[None, :] - mdy[:, None]
    sxx = np.einsum("bn,bnp,bnq->bpq", w, Dc, Dc) / 4.0
    sxy = np.einsum("bn,bnp,bn->bp", w, Dc, dyc) / 4.0
    A, b = mxx - sxx, mxy - sxy
    beta = np.full(mx.shape, np.nan)
    ok = np.linalg.eigvalsh(A).min(axis=1) > 1e-9
    beta[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    alpha = (my - np.einsum("bp,bp->b", beta, mx)) if intercept else np.zeros(w.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 1.0 - np.diagonal(sxx, axis1=1, axis2=2) / np.diagonal(mxx, axis1=1, axis2=2)
    return {"intercept": alpha, "slopes": beta, "reliability": rel}


def auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Mann-Whitney AUROC (labels 1 = positive)."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
