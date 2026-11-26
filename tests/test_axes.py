import numpy as np

from gib.axes.degrade import crop_window, render, visible_objects, visual_recipes
from gib.axes.expand import expand_item, gold_doc, qa_conditions
from gib.config import DEFAULTS


def test_levels_match_the_brief():
    v = DEFAULTS["axes"]["visual"]
    assert v["gauss_sigmas"] == [1, 2, 4, 8, 16]
    assert v["motion_kernels"] == [5, 10, 20, 40]
    assert v["crop_fracs"] == [0.5, 0.25, 0.10, 0.05]
    assert DEFAULTS["axes"]["retrieval"]["dosages"] == [1, 3, 5, 10, 20]


def test_render_is_deterministic(dataset, cfg):
    it = dataset["items"][0]
    for r in visual_recipes(cfg):
        a = np.asarray(render(it, r, dataset["image_root"], cfg))
        b = np.asarray(render(it, r, dataset["image_root"], cfg))
        assert a.shape == b.shape and (a == b).all()


def test_blur_actually_degrades(dataset, cfg):
    it = dataset["items"][0]
    orig = np.asarray(render(it, {"method": "none"}, dataset["image_root"], cfg)).astype(float)
    prev = 0.0
    for s in (1, 4, 16):
        g = np.asarray(render(it, {"method": "gauss", "sigma": s}, dataset["image_root"], cfg)).astype(float)
        d = np.abs(g - orig).mean()
        assert d > prev
        prev = d


def test_crop_shows_the_requested_fraction(dataset):
    it = next(i for i in dataset["items"] if i.target_bbox[2] > 40)
    x, y, w, h = it.target_bbox
    for f in (0.5, 0.25, 0.1):
        cw = crop_window(it, f)
        assert abs((cw[2] * cw[3]) / (w * h) - f) < 0.05 or cw[2] == 8.0 or cw[3] == 8.0
    assert set(visible_objects(it, {"method": "crop", "frac": 0.05})) <= {o[0] for o in it.objects}
    assert it.objects and visible_objects(it, {"method": "none"})


def test_expansion(dataset, cfg):
    it = next(i for i in dataset["items"] if i.qa_type == "object")
    qs = expand_item(it, cfg)
    assert len({q.qid for q in qs}) == len(qs)
    n_qa = len(qa_conditions(it, cfg))
    assert n_qa == 2 + 5 + 13 + 3 + 6
    assert sum(q.probe == "qa" for q in qs) == n_qa and sum(q.probe == "mc" for q in qs) == n_qa
    gold = gold_doc(it)
    for q in qs:
        if q.condition.startswith("ret_k") and q.probe == "qa":
            k = q.level["k"]
            assert len(q.docs) == k + 1 and q.docs.count(gold) == 1
            assert q.doc_claims[0] == it.answer and len(q.doc_claims) == k + 1
        if q.condition == "conflict" and q.probe == "qa":
            assert q.text_claim == it.conflict_claim and q.docs != [gold]
    ab = [q for q in qs if q.probe == "ab" and q.condition == "ab_both_g0"]
    assert [q.options for q in ab] == [[it.answer, it.conflict_claim], [it.conflict_claim, it.answer]]
    assert any(q.image_key == "blank" for q in qs)
