# Figure captions (generated with the figures)

## fig_vulnerability_heatmap.png
Vulnerability heatmap. Each cell is the drop in free-form accuracy relative to the clean condition (image + gold document) when the image is blurred (rows, σ in px) and k hard-negative documents are added (columns). Numbers are percentage points. One panel per model; shared colour scale.

## fig_decay.png
Retrieval-noise dose–response. Free-form accuracy versus the number k of hard-negative documents added next to the gold document (k = 0 is the clean condition). Shaded bands are 95% item-bootstrap confidence intervals.

## fig_resolution.png
Visual-ambiguity dose–response without retrieved documents. Shaded bands: 95% item-bootstrap CIs. A downward triangle marks each model's resolution threshold: the first level at which accuracy falls more than the configured jump (default 10 points) below the undegraded no-document accuracy.

## fig_conflict_matrix.png
Conflict matrix. Trust Index (geometric mean over conflict items of the A/B-probe odds of the text's claim over the image's answer; both option orders averaged) with 95% item-bootstrap CIs. The dashed line at 1 is balance; a model is labelled text-biased (image-biased) when its whole CI lies above (below) 1.

## fig_flip.png
Counterfactual flip decomposition. Each pair joins an item whose image shows A while the text claims B with an item whose image shows B while the text claims A. A pair is text-consistent if the model follows the text in both, image-consistent if it follows the image in both, and concept-prior if it names the same concept in both regardless of modality.

## fig_cue_weights.png
Cue-combination weights. For each model, the bimodal A/B log-odds (minus the blank-image prior) are regressed on the image-only and text-only log-odds measured on the same items at three blur levels. The star is the ideal Bayesian integrator (1, 1); points above the dashed diagonal over-weight text (Knowledge Inertia), points below over-weight the image (Sensory Capture). Error bars: 95% item-bootstrap CIs.

## fig_attention.png
Where the last prompt token attends, per layer: visual share = attention mass on image tokens / (image + post-image text tokens), averaged over heads and items. Solid lines compare the same images with a correct versus a misleading document; dotted lines split conflict items by whether the model's answer followed the text or the image.

## fig_pareto.png
Accuracy–robustness trade-off. R = (adverse accuracy / clean accuracy) × (1 − ECE on adverse conditions). Error bars are 95% item-bootstrap CIs; models on the Pareto frontier are marked in the legend.

## fig_taxonomy.png
Failure taxonomy by stress axis, decided by what the wrong answer matches: a wrong claim from the documents (Knowledge Inertia), a valid answer that ignores correct documents (Sensory Capture), an off-vocabulary answer or one matching neither modality (Reasoning Collapse), or a wrong answer with no documents shown (Perceptual Error).
