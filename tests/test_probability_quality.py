import math

import pandas as pd

from gib.eval.metrics import probability_quality
from gib.models.api import _APIBase
from gib.report.macros import write_macros
from gib.schema import Query


class FakeAPI(_APIBase):
    """Canned API responses, chosen by the question in the prompt (calls run in parallel; no network)."""

    def __init__(self, replies, **kw):
        super().__init__("fake", "fake-model", **kw)
        self.replies = replies

    def _call(self, jpeg, text, max_tokens, logprobs):
        return next(r for q, r in self.replies.items() if q in text)


def _q(i, probe, options):
    return Query(qid=f"x{i}|{probe}", gib_id=f"x{i}", condition="clean", axis="clean", probe=probe,
                 image_key=f"x{i}|orig", image_recipe={"method": "none"}, docs=[],
                 question=f"Question {i}?", options=options)


def test_api_records_where_probabilities_came_from(tmp_path):
    lp = math.log
    api = FakeAPI({
        "Question 1?": {"text": "B", "top": [["B", lp(0.6)], ["A", lp(0.2)], ["The", lp(0.15)]]},  # letters among them
        "Question 2?": {"text": "B", "top": [["The", lp(0.9)], ["It", lp(0.05)]]},                  # but no letter
        "Question 3?": {"text": "Yes", "top": []},                                                   # no logprobs
    }, cache_dir=tmp_path)
    real, parsed, pope = api.run_group(None, [_q(1, "mc", ["cat", "dog", "cow"]), _q(2, "mc", ["cat", "dog", "cow"]),
                                              _q(3, "pope", ["yes", "no"])])
    assert real.extra == {"from_logprobs": True, "format_mass": 0.8}
    assert [round(p, 6) for p in real.option_probs] == [0.25, 0.75, 0.0]
    assert parsed.extra == {"from_logprobs": False} and parsed.option_probs == [0.0, 1.0, 0.0]
    assert pope.extra == {"from_logprobs": False} and pope.option_probs == [1.0, 0.0]


def test_low_format_mass_flags_the_probe(cfg):
    df = pd.DataFrame({"probe": ["mc"] * 4 + ["pope"] * 2, "has_probs": [1.0] * 6,
                       "from_logprobs": [1.0] * 6, "format_mass": [0.9, 0.2, 0.3, None, 0.1, 0.1]})
    pq = probability_quality(df, cfg)
    mc = pq["probes"]["mc"]
    assert mc["from_logprobs"] == 1.0 and mc["low_mass"] == 0.5 and mc["usable"] == 0.5  # unknown mass counts as usable
    assert not mc["ok"] and "robustness.ece_adverse" in pq["flagged_metrics"]
    assert not pq["probes"]["pope"]["ok"] and pq["probes"]["pope"]["affects"] == []  # argmax-only: nothing flagged
    assert "trust.trust_index" not in pq["flagged_metrics"]  # no A/B rows here


def test_macros_mark_low_coverage_claims(tmp_path):
    rep = {"claims": {"m.R": 0.5, "m.acc_clean": 0.8},
           "probability_flags": {"m": {"metrics": ["robustness.R"], "claims": ["m.R"]}}}
    write_macros(tmp_path / "macros.tex", rep)
    lines = (tmp_path / "macros.tex").read_text(encoding="utf-8").splitlines()
    assert any(l.startswith("% WARNING") for l in lines)
    assert next(l for l in lines if "{\\gibMR}" in l).endswith("% LOW PROBABILITY COVERAGE")
    assert not next(l for l in lines if "{\\gibMAccClean}" in l).endswith("COVERAGE")
