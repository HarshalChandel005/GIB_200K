"""Behavioural simulator -- for TESTS and the offline demo ONLY.

Its outputs are synthetic. It exists so the whole pipeline (expansion, scoring,
statistics, figures, audit) can be exercised without a GPU, and so that unit
tests can check that every estimator *recovers parameters we planted* (e.g. the
cue-combination fit must return the text/image weights the simulator uses).
Simulated model names always start with ``sim-`` and the report refuses to
emit paper numbers when any simulated model is present.

Generative model: each option's logit is

    w_img * e_img * [option is the image truth]       (e_img falls with blur/crop)
  + w_txt * (doc evidence for that option)            (diminishing in #docs)
  + concept prior + noise,

and the probe distributions are a tempered softmax of those logits
(temperature < 1 = overconfident).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .. import vocab
from ..schema import BaseItem, Prediction, Probe, Query

DEFAULT_PROFILE: Dict[str, float] = {
    "e_img": 4.0,          # image evidence for the truth on a clean image (nats)
    "w_img": 1.0,          # weight given to image evidence
    "w_txt": 1.0,          # weight given to text evidence
    "doc_strength": 2.5,   # evidence per agreeing document (log-scaled with count)
    "prior_scale": 0.6,    # spread of per-concept prior preferences
    "noise": 0.8,          # per-query logit noise
    "temperature": 0.7,    # < 1 -> overconfident probabilities
    "interaction": 0.5,    # extra damage when image is degraded AND docs disagree
    "gibberish": 0.03,     # chance a free-form answer is off-vocabulary
    "attn_img": 0.55,      # baseline visual share of attention
    "attn_shift": 0.25,    # visual share lost when the text claims a different answer
    "logprob_coverage": 1.0,  # share of mc/ab/pope answers with token probabilities (rest: parsed, one-hot)
    "format_mass": 0.99,   # probability on the valid answer tokens (letters, yes/no)
}


def _u(*parts: Any) -> float:
    h = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def _gauss(*parts: Any) -> float:
    u1, u2 = max(_u(*parts, "a"), 1e-12), _u(*parts, "b")
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def severity(recipe: Dict[str, Any]) -> float:
    m = recipe.get("method", "none")
    if m == "gauss":
        return min(1.0, math.log2(1 + recipe["sigma"]) / math.log2(17))
    if m == "motion":
        return min(1.0, math.log2(1 + recipe["kernel"]) / math.log2(41))
    if m == "crop":
        return min(1.0, 1.0 - recipe["frac"] ** 0.5)
    if m == "noise":
        return min(1.0, recipe["std"] / 100.0)
    if m == "blank":
        return 1.0
    return 0.0


class SimulatedVLM:
    reproducible = True
    needs_image = False  # answers come from the item's annotations, so evaluate() skips rendering

    def __init__(self, name: str, items: Dict[str, BaseItem], profile: Optional[Dict[str, float]] = None,
                 n_layers: int = 16):
        if not name.startswith("sim-"):
            name = "sim-" + name
        self.name = name
        self.items = items
        self.p = {**DEFAULT_PROFILE, **(profile or {})}
        self.n_layers = n_layers

    def meta(self) -> Dict[str, Any]:
        return {"backend": "simulated", "profile": self.p, "WARNING": "synthetic outputs, not a real model"}

    # -------------------------------------------------------------- #
    def _logits(self, q: Query, options: List[str]) -> np.ndarray:
        p, it = self.p, self.items[q.gib_id]
        sev = severity(q.image_recipe)
        e_img = p["e_img"] * (1.0 - sev)
        doc_strength = p["doc_strength"] * (0.6 + 0.8 * _u("docs", q.gib_id))
        counts = {o: q.doc_claims.count(o) for o in options}
        wrong_docs = sum(c for o, c in counts.items() if o != it.answer)
        # integration weights apply only when both cues are present; a single cue
        # is used at face value -> the cue-combination fit must recover (w_img, w_txt)
        both = q.image_recipe.get("method") != "blank" and bool(q.docs)
        wi, wt = (p["w_img"], p["w_txt"]) if both else (1.0, 1.0)
        out = []
        for o in options:
            z = wi * e_img * (o == it.answer)
            if counts.get(o):
                z += wt * doc_strength * math.log1p(counts[o])
            z += p["prior_scale"] * _gauss("prior", self.name, o)
            if o == it.answer and sev > 0 and wrong_docs:
                z -= p["interaction"] * sev * math.log1p(wrong_docs) * p["doc_strength"]
            z += p["noise"] * _gauss(self.name, q.qid, o)
            out.append(z)
        return np.array(out)

    def _probs(self, logits: np.ndarray) -> List[float]:
        z = logits / self.p["temperature"]
        z = np.exp(z - z.max())
        return (z / z.sum()).round(6).tolist()

    def _attention(self, q: Query) -> List[float]:
        p = self.p
        claims_other = any(c != q.image_truth for c in q.doc_claims)
        share = p["attn_img"] - (p["attn_shift"] if claims_other else 0.0) * (1 + 0.5 * severity(q.image_recipe))
        prof = []
        for layer in range(self.n_layers):
            depth = layer / max(1, self.n_layers - 1)
            prof.append(float(np.clip(share * (0.6 + 0.8 * depth * (1 - depth) * 2)
                                      + 0.03 * _gauss("attn", self.name, q.qid, layer), 0.01, 0.95)))
        return prof

    def run_group(self, image: Optional[Image.Image], queries: List[Query]) -> List[Prediction]:
        preds = []
        for q in queries:
            it = self.items[q.gib_id]
            probe = Probe(q.probe)
            pred = Prediction(qid=q.qid, model=self.name, n_tokens=64 + 20 * len(q.docs), n_img_tokens=576,
                              latency_ms=1.0)
            if probe in (Probe.MC, Probe.AB):
                pred.option_probs = self._probs(self._logits(q, q.options))
            elif probe is Probe.POPE:
                sev = severity(q.image_recipe)
                present = q.meta["polarity"] == "pos"
                z = (2.5 * (1 - sev) if present else -2.0 + 1.5 * sev) + 0.5 * _gauss(self.name, q.qid)
                py = 1 / (1 + math.exp(-z))
                pred.option_probs = [round(py, 6), round(1 - py, 6)]
            elif probe is Probe.CAPTION:
                sev = severity(q.image_recipe)
                words = []
                if _u(self.name, q.qid, "see") > sev * 0.9 and it.pope_pos:
                    words.append(vocab.display(it.pope_pos))
                if _u(self.name, q.qid, "hall") < 0.1 + 0.6 * sev and it.pope_neg:
                    words.append(vocab.display(it.pope_neg))
                pred.text = ("A photo of " + " and ".join(f"a {w}" for w in words) + ".") if words else "A blurry photo."
            else:  # free-form answer
                opts = it.options
                pr = self._probs(self._logits(q, opts))
                if _u(self.name, q.qid, "gib") < self.p["gibberish"]:
                    pred.text = "I am not sure what this is."
                else:
                    pred.text = f"{vocab.display(opts[int(np.argmax(pr))])}"
            if pred.option_probs is not None:
                if _u(self.name, q.qid, "logprobs") < self.p["logprob_coverage"]:
                    pred.extra.update(from_logprobs=True, format_mass=self.p["format_mass"])
                else:  # like an API without logprobs: the parsed answer as a one-hot distribution
                    k = int(np.argmax(pred.option_probs))
                    pred.option_probs = [1.0 if j == k else 0.0 for j in range(len(pred.option_probs))]
                    pred.extra["from_logprobs"] = False
            if probe in (Probe.QA, Probe.AB):
                pred.attn_img = [round(s, 6) for s in self._attention(q)]
                pred.attn_post = [round(min(0.98 - s, 0.6), 6) for s in pred.attn_img]
            preds.append(pred)
        return preds
