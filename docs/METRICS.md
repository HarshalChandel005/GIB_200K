# GIB-200K metrics: definitions, motivation, and when to use each

Every number below is a function of the per-instance tables (`provenance/`, one per model) and is
recomputed from them by `gib audit`. All confidence intervals are 95% percentile
intervals from a **cluster bootstrap** over images (all queries of an item, and all items
of an image -- up to four -- are resampled together; 2,000 resamples), so the sampling unit
is never the query, and items that share an image are never treated as independent.

Notation: item *i*, model *m*. The A/B probe asks the question with two options, the
image's answer *a* and the text's claim *b*, in both orders; `LO_c(i) = log P(b) / P(a)`
in condition *c*, with the two option orders as two replicate measurements.

---

## 0. Scoring

Every question has a closed answer vocabulary (a COCO category group, 11 colours,
8 spatial predicates, or left/right). A free-form answer is mapped to the first
vocabulary term it mentions (synonyms, plurals and articles handled), or to *none*.
No LLM judge is involved, so scoring is deterministic and identical for every model
and for humans.

## 1. Accuracy, adverse accuracy

* `acc_c` — share of items whose free-form answer equals the image truth in condition *c*.
* `acc_clean` — original image + gold document. `acc_no_ctx` — original image, no
  document (the non-RAG baseline).
* `acc_adverse` — per item, the mean over the retrieval levels, the mean over the
  visual levels, and the conflict condition, then averaged with **equal weight per
  axis** (otherwise the 13 visual levels would dominate).

## 2. Robustness coefficient R (brief §D)

`R = (acc_adverse / acc_clean) × (1 − UE)`, where the Uncertainty Error UE is the
expected calibration error (10 bins) of the multiple-choice probe on adverse
conditions, with the same equal-axis weighting.

Properties: R ≤ retention (`acc_adverse / acc_clean`); R equals retention exactly
when the model is calibrated under stress; a model that stays accurate only by
being confidently wrong elsewhere is penalised. We also report the **confident-wrong
rate**, P(wrong and confidence > 0.8), the quantity the brief asks R to penalise.

## 3. Trust Index (brief §A.7)

`TI = exp(mean_i LO_conflict(i))`, the geometric mean of P(text answer)/P(image
answer) under conflict. TI > 1: text wins; TI < 1: image wins. The geometric mean
is used because ratios are averaged on the log scale (an arithmetic mean of ratios
is dominated by a few extreme items), and it is symmetric: swapping the roles of the
modalities maps TI to 1/TI. A model is labelled *text-biased* (*image-biased*) in the
conflict matrix only if the whole CI lies above (below) 1.

## 4. Cue combination: a normative test (new)

TI says *which* modality wins, not whether the model weighs the evidence
correctly — an image the model can barely see *should* lose to clear text. For a
normative reference, consider an ideal observer that combines conditionally
independent cues with Bayes' rule:

    P(answer | image, text) ∝ P(image | answer) · P(text | answer) · P(answer)

which, in log-odds of the text's claim *b* against the image's answer *a*, gives

    LO(image+text) − LO(prior) = [LO(image) − LO(prior)] + [LO(text) − LO(prior)].

Every term is measured on the same item: `LO(image)` with no document, `LO(text)`
with a blank grey image and the misleading document, `LO(prior)` with a blank image
and no document, and `LO(image+text)` with the image and the misleading document, at
blur σ ∈ {0, 4, 16}. We fit

    y = w_image · x_image + w_text · x_text

(no intercept: the ideal model has none, and subtracting `LO(prior)` measured in the
same option order cancels constant position bias). The ideal observer has
**w_image = w_text = 1**. This gives formal, falsifiable definitions of the two
failure modes named in the brief:

* **Knowledge Inertia**: `w_text / w_image > 1` — text evidence is over-weighted
  relative to the model's own image evidence.
* **Sensory Capture**: `w_text / w_image < 1` — image evidence (even when degraded,
  and therefore weak) is over-weighted relative to the text.

Item-level *integration errors* complete the picture: the model picks the text's
claim although its own unimodal evidence summed favours the image (KI-type), or the
reverse (SC-type). These separate *perception* failures (weak `x_image`) from
*integration* failures (wrong weighting).

**Estimation.** `x_image` and `x_text` are themselves noisy measurements, and plain
least squares then shrinks both weights towards zero (regression dilution). The two
option orders are independent replicates, so the noise covariance of the averaged
regressors is `Cov(Δx)/4`, which is subtracted from the moment matrix (the classical
method-of-moments correction; `stats.eiv_regression`). Validation on the simulator
with planted weights: at low noise the recovered weights are 1.000/0.999,
0.601/1.798 and 1.389/0.491 for planted 1/1, 0.6/1.8 and 1.4/0.5. At high noise,
across 16 independent draws, the corrected estimates averaged 0.70/1.64 for planted
0.7/1.6 (plain least squares: 0.45/1.20), with nominal CI coverage
(`tests/test_recovery.py`). The estimator is a ratio of moments and is noisy below
a few hundred items; we recommend ≥ 500 items and report each regressor's
reliability (share of its variance that is signal rather than replicate noise).

## 5. Counterfactual flip decomposition (brief §I; new analysis)

Items are paired at build time so that item *i* (image shows A, text claims B) has a
partner *j* (image shows B, text claims A). Each pair falls into exactly one class:

* **text-consistent** — follows the text in both,
* **image-consistent** — follows the image in both,
* **concept-prior** — names the same concept in both, whatever the modality.

A single-direction conflict test cannot tell "follows the text" from "prefers
concept B anyway"; the flip pair can. We report the three shares, the modality-bias
index P(text-consistent) − P(image-consistent), the decomposition per answer group
(animals, colours, spatial relations, …) and the concepts that win under the prior,
which answers the brief's question of whether the bias "depends on the type of object".

## 6. Grounding Illusion Score (brief §J.1)

The *Grounding Illusion*: a model answers correctly when image and text agree, but the
correctness came from the text. Operationally, on items answered correctly in the
clean condition:

* `GIS = P(answer follows the misleading text in the conflict condition | correct in clean)`
* `grounded accuracy = P(correct in clean and correct under conflict)`
* `illusion gap = acc_clean − grounded accuracy` — the part of clean accuracy that
  is not grounded in the image.

## 7. Context sensitivity (ReFinD-style, brief "REFIND")

ReFinD measures how much retrieved context changes the likelihood of an answer. At
the level of our fixed option set (multiple-choice probe), with and without documents:

* `CSR_gold = exp(mean log [P(truth | image, gold doc) / P(truth | image)])` — how
  much correct retrieval helps;
* `CSR_misinformation = exp(mean log [P(claim | image, misleading doc) / P(claim | image)])`
  — how much a misleading document moves probability to its claim.

The option-level form is used so that open and closed models are measured identically.

## 8. Attention shift (brief §B, §G)

For open models, at the last prompt token and per layer *l*:
`share_l = attn(image tokens) / [attn(image tokens) + attn(text after the image)]`
(heads averaged; tokens before the image, which often act as attention sinks, are
excluded). **ADS** is the *paired* difference `share_l(clean) − share_l(conflict)` on the
same image with the same document template — only the claimed answer differs — so
token counts and layout cancel.

Averaging attention over layers hides where the shift happens. We report the **full
per-layer profile** with CIs and early/middle/late thirds (`bands`), so no single
layer choice is imposed. Whether attention collapse *accompanies* Knowledge Inertia
is tested, not assumed: we report the AUROC with which a low visual share predicts
that the answer followed the text, and the separate profiles of conflict items that
followed the text and those that followed the image. Attention is treated as a
correlational signature, not as an explanation.

## 9. Dose–response

* **Decay rate (Axis 1):** slope of logit(accuracy) against log₂(1 + k) for k ∈
  {0, 1, 3, 5, 10, 20}, in log-odds per doubling, with bootstrap CI and fit R². The
  old "exponential decay lower bound" (unfalsifiable) is not used; we also report
  the **retained AUC** (area under the accuracy curve relative to clean), which
  assumes no functional form.
* **Resolution threshold (Axis 2, brief):** the first level at which CHAIR_i rises
  more than 10 points above the undegraded image (brief definition), and
  independently the first level at which no-document accuracy falls more than 10
  points, per family (Gaussian σ, motion kernel, crop fraction), with bootstrap
  intervals over levels.

## 10. Stressor interaction (super-additivity)

Per item and grid cell (σ, k): `I = [acc(σ,k) − acc(σ,0)] − [acc(0,k) − acc(0,0)]`
(difference-in-differences). The reported **super-additive drop** is `−mean(I)`: > 0
means blur and distractors together hurt more than the sum of their separate effects.
Tested with a sign-flip permutation test over items. The **vulnerability heatmap**
(brief §E.1) shows the accuracy drop in every cell.

The grid's k = 0 cells (blurred image + correct document) are the **Sensory Capture
probe**; `text_rescue = acc(blur + gold doc) − acc(blur, no doc)` measures how much
correct text compensates for a degraded image.

## 11. CHAIR and POPE (brief "Metric-Market Fit")

* **CHAIR** (Rohrbach et al., 2018): caption probe "Describe this image in one
  sentence."; mentions of the 80 COCO categories via a synonym list; CHAIR_i =
  hallucinated mentions / all mentions, CHAIR_s = captions with ≥ 1 hallucination /
  captions. Ground truth is the set of annotated objects *visible in that
  condition* (for crops, only objects intersecting the crop window).
* **POPE** (Li et al., 2023): "Is there a ⟨object⟩ in the image?" with one present
  object and one absent *adversarial* object (the most frequent COCO co-occurrer that
  is absent), on every image condition; accuracy, precision, recall, F1, yes-ratio.

## 12. Failure taxonomy (brief §B)

Decided by what a wrong free-form answer matches (`gib/eval/taxonomy.py`):

| class | rule |
|---|---|
| Knowledge Inertia | the answer repeats a wrong claim made by a retrieved document |
| Sensory Capture | documents contained the correct answer; the model gave a different valid answer |
| Reasoning Collapse | off-vocabulary answer (gibberish, refusal, unrelated), or a valid answer matching neither the image nor any document |
| Perceptual Error | no document shown, wrong answer |

These are rules, not a trained classifier, so there is no train/test split.
Validity is measured against blind human labels (`gib taxonomy-export` →
annotators → `gib taxonomy-kappa`: Cohen's κ, confusion matrix, per-class precision/recall).

## 13. From diagnosis to mitigation: the consistency gate

Because every question is also asked without documents, a training-free check is
available at test time: *does the answer change when the documents are removed?*
Three policies — abstain, fall back to the image-only answer, or keep the more
confident answer — are evaluated where documents lie (conflict, k = 20), where they
are right but the image is degraded (grid k = 0, where falling back *hurts*), and on
clean inputs. Signal quality: AUROC of low confidence, context disagreement,
confidence shift and low visual attention for predicting a wrong answer, plus a
grouped cross-validated logistic model and drop-one ΔAUROC (each signal's unique
contribution; relates the metrics to reliability).

## 14. Do the metrics carry distinct information?

Two complementary checks, both on real runs only (on the simulator they would only
reflect how the simulator was written): (a) Spearman correlation of model-level
metrics across the model zoo (≥ 4 models); (b) the drop-one ΔAUROC in §13, which
measures what each signal adds when the others are already known.

## 15. Statistics across models

All models answer the same items, so comparisons are **repeated measures**: Friedman
test on per-item accuracy (items are blocks; with Kendall's W), a repeated-measures
ANOVA with df = (k − 1, (k − 1)(n − 1)), and pairwise paired item-bootstrap tests with
Holm correction. Every test is reported with the n it was computed on.

## 16. Probability quality (can the probabilities be trusted?)

Several metrics read *graded* probabilities from the multiple-choice and A/B probes,
not just which option won. Those probabilities are only meaningful when they are
real. For every model and probe, the report gives the share of answers whose
distribution came from token log-probabilities (`from_logprobs`; an API that returns
none falls back to the parsed text answer as a one-hot 0/1 distribution), and the
mean **format mass**: the probability on the valid answer tokens (option letters,
yes/no) before renormalisation. An answer is *usable* when it came from
log-probabilities and its format mass is at least `min_format_mass` (0.5); below
that, the renormalised distribution describes tokens the model barely considered.

When a probe's usable share is below `min_prob_coverage` (0.9), the metrics that read
its probabilities are **flagged**:

| probe | flagged metrics |
|---|---|
| multiple choice | ECE, R, confident-wrong rate, CSR_gold, CSR_misinformation, the gate's confidence policy, error-signal AUROCs |
| A/B | Trust Index, cue-combination weights and integration errors |

POPE, the flip decomposition and the A/B follow rates use only the argmax, so they
stay valid on parsed answers and are never flagged. Flags appear in
`report.json` (`probability_flags`), as ⚠ in RESULTS.md, as ‡ in the paper tables,
as `LOW PROBABILITY COVERAGE` comments in `macros.tex`, and in `gib audit`'s output.
They do not fail the audit: the arithmetic is correct, but the input is unreliable,
and whether to report such a number is the authors' call. Both thresholds are in the
config (`analysis.min_prob_coverage`, `analysis.min_format_mass`).

## When to use which

| question | metric |
|---|---|
| Is the model right, and does it stay right under stress? | accuracy, adverse accuracy, **R** |
| Is it confidently wrong? | ECE, confident-wrong rate |
| When image and text disagree, which wins? | **Trust Index**, conflict matrix |
| Does it weigh the evidence like an ideal observer? | **cue-combination weights**, integration errors |
| Is the "bias" really about modality, or about the concept? | **flip decomposition** |
| Is its clean accuracy actually grounded in the image? | **GIS**, grounded accuracy, illusion gap |
| How much does retrieval move it, for good and for bad? | CSR_gold, CSR_misinformation |
| How fast do noise and blur hurt it? | decay rate, retained AUC, resolution thresholds |
| Do stressors compound? | super-additive drop, vulnerability heatmap |
| Does it name objects that are not there? | CHAIR, POPE |
| Is attention collapse a signature of text-following? | ADS profile, AUROC share → KI |
| Can we catch failures at test time? | consistency gate, error-signal AUROCs |
| Can this model's probabilities be trusted at all? | **probability quality** (usable share, format mass) |
