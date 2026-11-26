import json
import random
from collections import Counter
from types import SimpleNamespace

from gib import vocab
from gib.build.coco_vg import _leftright_candidates, _round_robin
from gib.build.subset import read_subset


def test_counts_and_types(dataset):
    items = dataset["items"]
    assert len(items) == 48
    assert Counter(it.qa_type for it in items) == {"object": 16, "attribute": 16, "relation": 16}
    assert {it.source_split for it in items} == {"val2017", "train2017"}
    assert len({it.gib_id for it in items}) == len(items)


def test_visual_genome_is_joined_through_coco_id(dataset):
    items = dataset["items"]
    assert not any("decoy" in it.question for it in items)
    attr = [it for it in items if it.qa_type == "attribute"]
    for it in attr:
        # VG fixture images are 2x the COCO size: boxes must be scaled back
        cat_boxes = [o[1:] for o in it.objects]
        assert any(abs(b[0] - it.target_bbox[0]) < 1e-6 and abs(b[2] - it.target_bbox[2]) < 1e-6 for b in cat_boxes)
        assert it.answer in vocab.COLORS


def test_object_questions_have_unique_answers(dataset):
    for it in dataset["items"]:
        if it.qa_type != "object":
            continue
        present = {o[0] for o in it.objects if o[0] in vocab.COCO_GROUPS[it.group]}
        assert present == {it.answer}
        assert it.answer not in it.siblings
        assert set(it.siblings) <= set(vocab.COCO_GROUPS[it.group])


def test_flip_pairs_are_mirror_images(dataset):
    by = {it.gib_id: it for it in dataset["items"]}
    pairs = 0
    for it in dataset["items"]:
        assert it.conflict_claim in it.siblings[:4]
        if it.flip_partner:
            j = by[it.flip_partner]
            assert j.flip_partner == it.gib_id
            assert j.answer == it.conflict_claim and j.conflict_claim == it.answer
            pairs += 1
    assert pairs > 0


def test_fast_pairing_matches_a_plain_scan(dataset):
    import copy

    from gib.build.coco_vg import pair_conflicts

    def plain(items):  # the original quadratic scan
        by_key = {}
        for it in items:
            by_key.setdefault((it.group, it.answer), []).append(it)
        for it in items:
            if it.flip_partner is not None:
                continue
            for b in it.siblings[:4]:
                partner = next((j for j in by_key.get((it.group, b), [])
                                if j.flip_partner is None and j is not it and it.answer in j.siblings[:4]), None)
                if partner is not None:
                    it.conflict_claim, partner.conflict_claim = b, it.answer
                    it.flip_partner, partner.flip_partner = partner.gib_id, it.gib_id
                    break
            if it.conflict_claim is None:
                it.conflict_claim = it.siblings[0]

    fresh = []
    for it in dataset["items"] * 3:  # repeats make many competing candidates per key
        c = copy.copy(it)
        c.gib_id, c.flip_partner, c.conflict_claim = f"{it.gib_id}-{len(fresh)}", None, None
        fresh.append(c)
    a, b = copy.deepcopy(fresh), copy.deepcopy(fresh)
    pair_conflicts(a)
    plain(b)
    assert [(x.flip_partner, x.conflict_claim) for x in a] == [(x.flip_partner, x.conflict_claim) for x in b]


def test_subsets_keep_pairs_together(dataset):
    by = {it.gib_id: it for it in dataset["items"]}
    sub = set(read_subset(dataset["data"] / "subsets" / "dev.txt"))
    assert len(sub) <= 8
    for g in sub:
        p = by[g].flip_partner
        assert p is None or p in sub


def test_build_report(dataset):
    rep = json.loads((dataset["data"] / "build_report.json").read_text(encoding="utf-8"))
    assert rep["n_items"] == 48 and rep["flip_pairs"] > 0


def test_round_robin_limits_an_answer_to_its_room():
    cands = ([{"answer": "on", "split": "t", "image_id": i} for i in range(100)]
             + [{"answer": "left", "split": "t", "image_id": 100 + i} for i in range(10)]
             + [{"answer": "under", "split": "t", "image_id": 200 + i} for i in range(5)])
    room = Counter({"on": 12, "left": 12, "under": 12})
    picked = _round_robin(cands, 40, random.Random(0), Counter(), 1, "answer", room)
    assert Counter(c["answer"] for c in picked) == {"on": 12, "left": 10, "under": 5}
    assert room["on"] == 0 and room["under"] == 7


def test_leftright_uses_every_separated_pair():
    # four single objects: all 6 pairs count except dog/person, whose centres are too close
    boxes ={"dog": [10, 10, 40, 40], "person": [20, 10, 40, 40], "car": [120, 10, 40, 40], "bus": [240, 10, 40, 40]}
    cats = {i: c for i, c in enumerate(boxes)}
    anns = [{"category_id": i, "bbox": boxes[c], "id": i} for i, c in cats.items()]
    cs = SimpleNamespace(images={1: {"width": 300, "height": 200}}, anns={1: anns}, cats=cats)
    out = _leftright_candidates(cs, {"build": {"leftright_min_gap": 0.15, "min_area_frac": 0.02}})
    assert len(out) == 5
    by_pair ={(c["slots"]["SUBJ"], c["slots"]["OBJ"]): c["answer"] for c in out}
    assert ("dog", "person") not in by_pair and ("person", "dog") not in by_pair
    for (s, o), ans in by_pair.items():
        assert ans == ("left" if boxes[s][0] < boxes[o][0] else "right")


def test_build_reports_answer_shares(dataset):
    report = json.loads((dataset["data"] / "build_report.json").read_text(encoding="utf-8"))
    assert set(report["answer_share"]) == {"object", "attribute", "relation"}
    assert abs(sum(report["answer_share"]["relation"].values()) - 1) < 0.01
