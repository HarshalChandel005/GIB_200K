"""Known-answer tests: every estimator must recover what the simulator planted."""

import numpy as np
import pytest

from gib.eval import metrics as M
from gib.eval import stats as S
from gib.eval.table import build_table
from gib.eval.taxonomy import (KNOWLEDGE_INERTIA, PERCEPTUAL_ERROR, REASONING_COLLAPSE, SENSORY_CAPTURE,
                               classify)
from gib.evaluate import evaluate
from gib.models.simulated import SimulatedVLM


def _run(dataset, cfg, tmp_path, name, profile):
    items = dataset["items"]
    sim = SimulatedVLM(name, {it.gib_id: it for it in items}, profile)
    evaluate(sim, items, cfg, dataset["image_root"], tmp_path / name, progress=False)
    df = build_table(items, cfg, [tmp_path / name])
    ids = sorted(df["gib_id"].unique())
    return M.ModelView(df, ids, cfg), S.boot_weights(len(ids), 200, 0), items


@pytest.mark.parametrize("w_img,w_txt", [(1.0, 1.0), (0.6, 1.8), (1.4, 0.5)])
def test_cue_combination_recovers_planted_weights(dataset, cfg, tmp_path, w_img, w_txt):
    prof = {"w_img": w_img, "w_txt": w_txt, "interaction": 0.0, "noise": 0.02, "prior_scale": 0.3}
    v, W, _ = _run(dataset, cfg, tmp_path, f"cue{w_img}-{w_txt}", prof)
    cc = M.cue_combination(v, W)
    assert cc["available"]
    assert abs(cc["w_image"]["value"] - w_img) < 0.12 * max(1, w_img)
    assert abs(cc["w_text"]["value"] - w_txt) < 0.12 * max(1, w_txt)


def test_noise_corrected_weights_are_unbiased_unlike_plain_least_squares(dataset, cfg, tmp_path):
    """Noisy probes: plain OLS is biased towards 0, the replicate correction is not.

    With only 48 items a single draw is noisy (the estimator is a ratio of
    moments), so the median over five independent noise draws is checked.
    """
    prof = {"w_img": 0.7, "w_txt": 1.6, "interaction": 0.0, "noise": 0.5}
    corr, ols = [], []
    for rep in range(5):
        v, W, _ = _run(dataset, cfg, tmp_path, f"noisy{rep}", prof)
        cc = M.cue_combination(v, W)
        corr.append([cc["w_image"]["value"], cc["w_text"]["value"]])
        ols.append([cc["uncorrected_ols"]["w_image"], cc["uncorrected_ols"]["w_text"]])
    corr, ols = np.median(corr, axis=0), np.median(ols, axis=0)
    truth = np.array([0.7, 1.6])
    assert np.all(np.abs(corr - truth) < 0.25)
    assert np.all(np.abs(corr - truth) < np.abs(ols - truth))


def test_text_lover_has_higher_trust_index_and_gis(dataset, cfg, tmp_path):
    a, W, items = _run(dataset, cfg, tmp_path, "img", {"w_txt": 0.5, "w_img": 1.4})
    b, _, _ = _run(dataset, cfg, tmp_path, "txt", {"w_txt": 2.0, "w_img": 0.6})
    ta, tb = M.trust(a, W), M.trust(b, W)
    assert tb["trust_index"]["value"] > ta["trust_index"]["value"]
    assert tb["GIS"]["value"] >= ta["GIS"]["value"]
    fb = M.flip_decomposition(b, {it.gib_id: it for it in items}, 100, 0)
    fa = M.flip_decomposition(a, {it.gib_id: it for it in items}, 100, 0)
    assert fb["text_consistent"]["value"] >= fa["text_consistent"]["value"]


def test_attention_shift_sign(dataset, cfg, tmp_path):
    v, W, _ = _run(dataset, cfg, tmp_path, "attn", {"attn_shift": 0.3})
    a = M.attention_shift(v, W)
    assert a["available"] and a["ADS"]["value"] > 0.1


def test_taxonomy_rules():
    assert classify("dog", "dog", ["cat"]) is None
    assert classify(None, "dog", ["dog"]) == REASONING_COLLAPSE
    assert classify("cat", "dog", ["dog", "cat"]) == KNOWLEDGE_INERTIA
    assert classify("horse", "dog", ["dog", "cat"]) == SENSORY_CAPTURE
    assert classify("horse", "dog", ["cat"]) == REASONING_COLLAPSE
    assert classify("horse", "dog", []) == PERCEPTUAL_ERROR
