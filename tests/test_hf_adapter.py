"""Real-model check of the efficient HF path (skipped unless a model is given).

    GIB_HF_TEST_MODEL=llava-hf/llava-interleave-qwen-0.5b-hf pytest tests/test_hf_adapter.py -s
"""

import os

import pytest

MODEL = os.environ.get("GIB_HF_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set GIB_HF_TEST_MODEL to run")


def test_prefix_cache_and_batching_match_plain_computation(dataset, cfg):
    pytest.importorskip("torch")
    from gib.axes.degrade import render
    from gib.axes.expand import expand_item
    from gib.models.hf_vlm import HFVLM

    it = next(i for i in dataset["items"] if i.qa_type == "object")
    qs = [q for q in expand_item(it, cfg) if q.image_key.endswith("|orig")][:12]
    img = render(it, qs[0].image_recipe, dataset["image_root"], cfg)
    m = HFVLM("t", MODEL, device="cpu", max_pixels=200704, batch_size=16)  # fp32: exact comparisons
    m.run_group(img, qs)  # first call runs the self-checks (and answers its batch one by one)
    assert m._verified["prefix_cache"]["ok"] and m._verified["batched_suffixes"]["ok"], m._verified
    batched = {p.qid: p for p in m.run_group(img, qs)}  # now packed
    m.batch_size = 1
    one = {p.qid: p for p in m.run_group(img, qs)}      # shared prefix, one query at a time
    m.use_prefix_cache = False
    slow = {p.qid: p for p in m.run_group(img, qs)}     # full prefill per query
    for got in (batched, one):
        for q in qs:
            a, b = got[q.qid], slow[q.qid]
            if a.option_probs:
                assert max(abs(x - y) for x, y in zip(a.option_probs, b.option_probs)) < 1e-3
            if a.text is not None:
                assert a.text == b.text
            if q.probe in ("qa", "ab"):
                assert a.attn_img and len(a.attn_img) == len(a.attn_post)
                assert max(abs(x - y) for x, y in zip(a.attn_img + a.attn_post, b.attn_img + b.attn_post)) < 1e-3
