"""Failure taxonomy, decided by *what the wrong answer matches*.

The previous classifier labelled every wrong answer on a blurred image as
"Sensory Capture", so the class shares only mirrored how many blur conditions
exist, and answers copied from a distractor were filed as "Reasoning Collapse".
Here a wrong free-form answer is classified by its content:

1. Reasoning Collapse -- no answer from the question's vocabulary (gibberish,
   refusal, unrelated object), or a valid answer that matches *neither* the
   image nor any document while the documents were misleading.
2. Knowledge Inertia  -- the answer repeats a wrong claim made by a retrieved
   document: the model followed the text over the image.
3. Sensory Capture    -- the documents stated the correct answer, yet the model
   gave a different valid answer: it followed its (possibly degraded) visual
   impression over correct text.
4. Perceptual Error   -- no document was shown and the answer is wrong
   (a pure perception failure; neither text-vs-image class applies).

These are deterministic rules -- there is no trained classifier and so no
train/test split. Their
agreement with human judgement is measured instead: ``gib taxonomy-export``
samples failures for labelling and ``gib taxonomy-kappa`` reports Cohen's kappa.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

REASONING_COLLAPSE = "Reasoning Collapse"
KNOWLEDGE_INERTIA = "Knowledge Inertia"
SENSORY_CAPTURE = "Sensory Capture"
PERCEPTUAL_ERROR = "Perceptual Error"
CLASSES = (KNOWLEDGE_INERTIA, SENSORY_CAPTURE, REASONING_COLLAPSE, PERCEPTUAL_ERROR)


def classify(extracted: Optional[str], truth: str, doc_claims: List[str]) -> Optional[str]:
    if extracted == truth:
        return None
    claims = doc_claims or []
    if extracted is None:
        return REASONING_COLLAPSE
    if extracted in claims:
        return KNOWLEDGE_INERTIA
    if truth in claims:
        return SENSORY_CAPTURE
    if claims:
        return REASONING_COLLAPSE
    return PERCEPTUAL_ERROR


def classify_frame(qa: pd.DataFrame) -> List[Optional[str]]:
    out = []
    for e, t, dc in zip(qa["extracted"], qa["image_truth"], qa["doc_claims"]):
        e = e if isinstance(e, str) else None
        out.append(classify(e, t, dc if isinstance(dc, list) else []))
    return out
