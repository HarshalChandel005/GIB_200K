import numpy as np
from scipy import stats as st

from gib.eval import stats as S


def test_friedman_matches_scipy_and_rm_anova_df():
    rng = np.random.default_rng(0)
    m = rng.random((40, 4)) + np.array([0, 0.1, 0.2, 0.3])
    fr = S.friedman(m)
    chi2, p = st.friedmanchisquare(*m.T)
    assert abs(fr["chi2"] - chi2) < 1e-9 and abs(fr["p"] - p) < 1e-12 and fr["df"] == 3
    an = S.rm_anova(m)
    assert (an["df1"], an["df2"]) == (3, 3 * 39)  # repeated measures, not N - k


def test_holm():
    assert S.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_cluster_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.0, 400)
    W = S.boot_weights(len(x), 1000, 0)
    s = S.mean_stat(x, W)
    assert s["ci"][0] < 0.3 < s["ci"][1]
    assert abs(s["value"] - x.mean()) < 1e-12


def test_wls_recovers_coefficients():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    y = X @ np.array([0.5, 2.0]) + rng.normal(0, 0.01, 300)
    W = S.boot_weights(300, 50, 0)
    coef = S.wls_weights(y, X, W)
    assert np.allclose(np.median(coef, axis=0), [0.5, 2.0], atol=0.01)


def test_auroc_and_sign_flip():
    assert S.auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert S.sign_flip_p(np.full(30, 0.5)) < 0.01
    assert S.sign_flip_p(np.array([1.0, -1.0] * 20)) > 0.5
