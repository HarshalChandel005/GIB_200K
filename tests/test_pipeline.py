import json

import pytest

from gib.evaluate import evaluate, load_items
from gib.human import study, taxonomy_check
from gib.models.simulated import SimulatedVLM
from gib.report.audit import AuditError, audit
from gib.report.build import write_report
from gib.eval.table import load_rows


@pytest.fixture(scope="module")
def report(dataset, cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("rep")
    items = load_items(dataset["data"], "eval")
    by = {it.gib_id: it for it in items}
    runs = []
    # "c" behaves like an API that returns log-probabilities for only 30% of its probe answers
    for name, prof in (("a", {}), ("b", {"w_txt": 1.7}), ("c", {"w_img": 1.3, "w_txt": 0.6, "logprob_coverage": 0.3})):
        sim = SimulatedVLM(name, by, prof)
        evaluate(sim, items, cfg, dataset["image_root"], out / "runs" / sim.name, progress=False)
        runs.append(out / "runs" / sim.name)
    rep = write_report(out / "report", items, cfg, runs, image_root=dataset["image_root"])
    return {"dir": out / "report", "rep": rep, "items": items, "runs": runs, "out": out}


def test_report_files(report):
    d = report["dir"]
    assert len(list((d / "provenance").glob("*.csv.gz"))) == 3  # one table per model
    for f in ("items.txt", "report.json", "claims.json", "RESULTS.md", "macros.tex", "gallery.html",
              "figures/CAPTIONS.md", "figures/fig_vulnerability_heatmap.png", "figures/fig_cue_weights.png"):
        assert (d / f).exists(), f
    assert "SIMULATED" in (d / "RESULTS.md").read_text(encoding="utf-8")
    assert "REFUSED" in (d / "macros.tex").read_text(encoding="utf-8")
    rep = report["rep"]
    assert rep["cross_model"]["rm_anova_acc_adverse"]["df2"] == 2 * (rep["n_items"] - 1)


def test_audit_passes_and_catches_tampering(report, cfg):
    res = audit(report["dir"], report["items"], cfg, allow_simulated=True)
    assert res["ok"] and res["n_claims"] > 20
    with pytest.raises(AuditError):
        audit(report["dir"], report["items"], cfg)  # simulated models are never paper-ready
    claims_path = report["dir"] / "claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    key = next(k for k in claims if k.endswith(".acc_clean"))
    claims[key] += 0.01
    claims_path.write_text(json.dumps(claims), encoding="utf-8")
    try:
        with pytest.raises(AuditError):
            audit(report["dir"], report["items"], cfg, allow_simulated=True)
    finally:
        claims[key] -= 0.01
        claims_path.write_text(json.dumps(claims), encoding="utf-8")


def test_low_probability_coverage_is_flagged(report, cfg):
    from gib.latex import tables

    rep = report["rep"]
    for m in ("sim-a", "sim-b"):
        pq = rep["models"][m]["probability_quality"]
        assert pq["ok"] and pq["probes"]["mc"]["usable"] == 1.0
    pq = rep["models"]["sim-c"]["probability_quality"]
    assert not pq["ok"] and 0.2 < pq["probes"]["mc"]["from_logprobs"] < 0.4
    assert {"robustness.R", "trust.trust_index", "cue_combination"} <= set(pq["flagged_metrics"])
    assert set(rep["probability_flags"]) == {"sim-c"}
    assert {"sim-c.R", "sim-c.trust_index"} <= set(rep["probability_flags"]["sim-c"]["claims"])
    md = (report["dir"] / "RESULTS.md").read_text(encoding="utf-8")
    assert "LOW PROBABILITY COVERAGE" in md and "## Probability quality" in md
    head = tables.headline(rep)
    assert [r[0] for r in head.rows if "‡" in r[4]] == ["sim-c"] and head.note == tables.LOW_COVERAGE
    # the flags survive the provenance round trip the audit recomputes from
    res = audit(report["dir"], report["items"], cfg, allow_simulated=True)
    assert set(res["probability_flags"]) == {"sim-c"}


def test_resume_skips_done_queries(report, dataset, cfg):
    items = report["items"]
    sim = SimulatedVLM("a", {it.gib_id: it for it in items}, {})
    meta = evaluate(sim, items, cfg, dataset["image_root"], report["runs"][0], progress=False)
    assert meta["new_predictions"] == 0


def test_human_study_roundtrip(report, dataset, cfg, tmp_path):
    out = tmp_path / "study"
    res = study.export(report["items"], cfg, dataset["image_root"], out, n=22)
    assert res["n_items"] == 22
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "application/json" in html and "Download answers" in html
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    for pid, good in (("p1", True), ("p2", True), ("p3", False)):
        resp = [{"hid": k["hid"], "answer": k["truth"] if good else "no idea", "ms": 1500} for k in key]
        (out / "responses" / f"{pid}.json").write_text(json.dumps({"participant": pid, "responses": resp}),
                                                       encoding="utf-8")
    prov = load_rows(report["dir"])
    h = study.analyze(out, sorted((out / "responses").glob("*.json")), prov, iters=100)
    assert h["n_participants"] == 3 and abs(h["accuracy_overall"]["value"] - 2 / 3) < 1e-9
    assert h["models_on_same_items"]


def test_taxonomy_validation(report, dataset, cfg, tmp_path):
    prov = load_rows(report["dir"])
    res = taxonomy_check.export(prov, report["items"], cfg, dataset["image_root"], tmp_path / "tax", n=20)
    assert res["n"] > 0
    import pandas as pd

    lab = pd.read_csv(tmp_path / "tax" / "to_label.csv")
    key = pd.read_csv(tmp_path / "tax" / "key.csv")
    assert "auto_label" not in lab.columns  # blind labelling
    lab["human_label"] = key["auto_label"]
    lab.to_csv(tmp_path / "tax" / "to_label.csv", index=False)
    k = taxonomy_check.kappa(tmp_path / "tax" / "to_label.csv", tmp_path / "tax" / "key.csv")
    assert k["agreement"] == 1.0
