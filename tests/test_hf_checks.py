"""The self-check verdicts of the HF adapter, on made-up numbers (no model needed)."""

from types import SimpleNamespace

import pytest

from gib.models.hf_vlm import HFVLM
from gib.schema import Prediction

torch = pytest.importorskip("torch")
P = 8


def adapter(fake_noise, tol_key="bfloat16"):
    """``fake_noise[q]`` = log-prob shift of (one pass, prefill + last token, shared prefix) for query q."""
    m = HFVLM.__new__(HFVLM)
    m.torch, m._tol_key, m._verified, m.use_prefix_cache, m.cache = torch, tol_key, {}, True, None
    base = torch.arange(100, 0, -1, dtype=torch.float32) / 4
    state = {}

    def fwd(ids, start, extra=None):
        if start == 0:
            state.update(q=ids[0], pieces=0)
        state["pieces"] += 1
        bump = torch.zeros(100)
        bump[0] = fake_noise[state["q"]][state["pieces"] - 1]
        return SimpleNamespace(logits=(base + bump)[None, None])

    m._fwd = fwd
    return m


def verify(m, n_queries):
    m._verify([([q] * 12, {}) for q in range(n_queries)], P)
    return m._verified["prefix_cache"]


def test_rounding_of_a_split_pass_is_not_a_bug():
    v = verify(adapter([(0.0, 0.15, 0.15)] * 5), 5)
    assert v["ok"] and v["split_diff"] == 0 and 0.1 < v["median_logprob_diff"] < v["tol"]


def test_shared_prefix_that_differs_from_a_prefill_fails_and_falls_back():
    m = adapter([(0.0, 0.0, 0.5)] * 5)
    assert not verify(m, 5)["ok"] and m.use_prefix_cache is False


def test_one_noisy_query_does_not_fail_the_check_but_most_of_them_do():
    assert verify(adapter([(0, 0, 0.5)] + [(0, 0, 0)] * 4), 5)["ok"]
    assert not verify(adapter([(0, 0, 0.3)] * 3 + [(0, 0, 0)] * 2), 5)["ok"]


def test_a_broken_cache_is_caught_even_if_both_cached_paths_agree():
    m = adapter([(0.0, 2.0, 2.0)] * 5)
    v = verify(m, 5)
    assert v["split_diff"] == 0 and not v["ok"] and m.use_prefix_cache is False


def test_limits_follow_the_precision():
    noise = [(0.0, 0.0, 0.1)] * 5
    assert verify(adapter(noise, "bfloat16"), 5)["ok"]
    assert not verify(adapter(noise, "float16"), 5)["ok"]
    assert not verify(adapter(noise, "float32"), 5)["ok"]
    assert verify(adapter(noise, "something new"), 5)["ok"]  # the loosest row


def batch(texts, probs=0.5, attn=0.5):
    return [Prediction(qid=str(i), model="m", text=t, option_probs=None if t else [probs, 1 - probs],
                       attn_img=[attn], attn_post=[attn]) for i, t in enumerate(texts)]


def check(ref, got, tol_key="bfloat16"):
    m = adapter([], tol_key)
    m._check_batched(ref, got)
    return m._verified["batched_suffixes"], m._batched


def test_a_flipped_near_tie_is_not_a_bug_many_are():
    ref = batch(["cow", "left", "red", None])
    assert check(ref, batch(["cow", "left", "red", None]))[1]
    v, ok = check(ref, batch(["horse", "right", "red", None]))
    assert ok and v["text_diffs"] == 2
    assert not check(ref, batch(["horse", "right", "blue", None]))[1]
    assert check(ref, batch(["horse", "left", "red", None]), "float16")[1]
    assert not check(ref, batch(["horse", "right", "red", None]), "float16")[1]
    assert not check(ref, batch(["horse", "left", "red", None]), "float32")[1]


def test_batch_limits_on_probabilities_and_attention():
    ref = batch(["cow", None])
    assert check(ref, batch(["cow", None], probs=0.55))[1]
    assert not check(ref, batch(["cow", None], probs=0.65))[1]
    assert check(ref, batch(["cow", None], attn=0.505))[1]
    assert not check(ref, batch(["cow", None], attn=0.52))[1]
    assert not check(ref, batch(["cow", None], attn=0.52), "float16")[1]
