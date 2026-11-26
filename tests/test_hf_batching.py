from types import SimpleNamespace

from gib.models.hf_vlm import BATCH_TOKENS, HFVLM


def test_batches_respect_size_and_token_budget():
    P = 100
    short = [list(range(P + 20))] * 7                  # 20 suffix tokens each
    m = SimpleNamespace(batch_size=3)
    assert HFVLM._batches(m, short, P) == [[0, 1, 2], [3, 4, 5], [6]]
    long_ids = [list(range(P + BATCH_TOKENS // 2 + 1))] * 3  # two of these exceed the token budget
    assert HFVLM._batches(SimpleNamespace(batch_size=16), long_ids, P) == [[0], [1], [2]]
    huge = [list(range(P + 2 * BATCH_TOKENS))]           # one over-long query still gets its own batch
    assert HFVLM._batches(SimpleNamespace(batch_size=16), huge + short[:2], P) == [[0], [1, 2]]
