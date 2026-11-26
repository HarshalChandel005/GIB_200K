"""The code paths that only switch on at 200k-item scale, exercised on small data."""

from collections import Counter

import numpy as np
import pandas as pd

from gib.axes.expand import expand
from gib.eval import metrics as M
from gib.eval import stats as S
from gib.eval import table as T
from gib.evaluate import GROUP_CHUNK, count_work, evaluate, iter_groups, load_items, qid_hashes
from gib.models.simulated import SimulatedVLM


def test_blocked_bootstrap_weights_match_one_draw(monkeypatch):
    whole = S.boot_weights(500, 120, 3)
    monkeypatch.setattr(S, "BIG", 1000)  # force the float32, block-by-block path
    blocked = S.boot_weights(500, 120, 3)
    assert whole.dtype == np.float64 and blocked.dtype == np.float32
    assert np.array_equal(whole, blocked.astype(np.float64)) and (blocked.sum(1) == 500).all()
    v = np.random.default_rng(0).normal(size=500)
    assert S.mean_stat(v, whole)["value"] == S.mean_stat(v, blocked)["value"]  # point estimates never use W
    assert np.allclose(S.mean_stat(v, whole)["ci"], S.mean_stat(v, blocked)["ci"], atol=1e-5)


def test_cluster_weights_resample_images_together():
    W = S.cluster_weights(["a", "b", "a", "c", "b"], 50, 0)
    assert np.array_equal(W[:, 0], W[:, 2]) and np.array_equal(W[:, 1], W[:, 4])
    assert (W[:, [0, 1, 3]].sum(1) == 3).all()  # three images, each drawn as one unit
    assert np.array_equal(S.cluster_weights(["a", "b", "c"], 50, 0), S.boot_weights(3, 50, 0))


def test_chunked_eiv_matches_direct():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2)); y = X @ [0.7, 1.4] + rng.normal(scale=0.3, size=300)
    dX, dy = rng.normal(scale=0.2, size=(300, 2)), rng.normal(scale=0.2, size=300)
    W, cols = S.boot_weights(400, 60, 1), rng.integers(0, 400, 300)
    got = M._eiv(y, X, dy, dX, W, cols)
    ref = S.eiv_regression(y, X, dy, dX, W[:, cols])
    assert np.allclose(got["slopes"], ref["slopes"], equal_nan=True)


def test_sign_flip_in_blocks_is_still_calibrated():
    rng = np.random.default_rng(0)
    assert S.sign_flip_p(rng.normal(0, 1, 60_000), iters=800) > 0.01   # 2e7 / 60k -> blocks of 333 rows
    assert S.sign_flip_p(rng.normal(0.05, 1, 60_000), iters=800) < 0.01


def test_streamed_groups_cover_every_query_once(dataset, cfg):
    items = load_items(dataset["data"], "eval")
    qs = expand(items, cfg)
    streamed = [(k, q.qid) for k, g in iter_groups(items, cfg) for q in g]
    assert Counter(q for _, q in streamed) == Counter(q.qid for q in qs)
    blanks = [g for k, g in iter_groups(items, cfg) if k == "blank"]
    assert blanks and all(len(g) <= GROUP_CHUNK for g in blanks)
    assert all(k == "blank" or len({q.image_key for q in g}) == 1 for k, g in iter_groups(items, cfg))
    done = np.unique(qid_hashes([q.qid for q in qs[::2]]))
    rest = [q.qid for _, g in iter_groups(items, cfg, done) for q in g]
    assert Counter(rest) == Counter(q.qid for q in qs[1::2])
    w = count_work(items, cfg)
    assert w["queries"] == len(qs) and sum(w["by_probe"].values()) == len(qs)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    for c in T.LIST_COLS:
        out[c] = [tuple(np.asarray(v).tolist()) if v is not None else None for v in out[c]]
    return out.reset_index(drop=True)


def test_lean_table_survives_chunking_and_the_csv_round_trip(dataset, cfg, tmp_path, monkeypatch):
    items = load_items(dataset["data"], "eval")
    sim = SimulatedVLM("lean", {it.gib_id: it for it in items}, {"logprob_coverage": 0.5})
    evaluate(sim, items, cfg, dataset["image_root"], tmp_path / "run", progress=False)
    one = T.build_table(items, cfg, [tmp_path / "run"])
    assert set(one.columns) == set(T.COLUMNS) and one["gib_id"].dtype == "category"
    assert one["p_truth"].dtype == np.float32 and one["attn_img"].notna().sum() > 0
    assert set(one.loc[one["attn_img"].notna(), "condition"]) <= set(T.ATTN_CONDITIONS)
    monkeypatch.setattr(T, "CHUNK_ROWS", 97)  # many chunks: categorical unions, chunked writes and reads
    many = T.build_table(items, cfg, [tmp_path / "run"])
    pd.testing.assert_frame_equal(_norm(one), _norm(many))
    T.save_table(many, tmp_path / "t.csv.gz")
    back = T.load_table(tmp_path / "t.csv.gz")
    pd.testing.assert_frame_equal(_norm(one), _norm(back), check_dtype=False)
    assert back["p_truth"].dtype == np.float32 and back["model"].dtype == "category"
