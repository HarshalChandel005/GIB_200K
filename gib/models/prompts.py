"""Prompt text for every probe -- one definition shared by all backends.

Open and closed models see byte-identical user text; only the chat wrapping
(and the image transport) differs per backend. That is what makes the
open-vs-closed comparison fair.
"""

from __future__ import annotations

from typing import List

from .. import vocab
from ..schema import Probe, Query

LETTERS = "ABCDE"
QA_SUFFIX = "Answer the question using a single word or phrase."
MC_SUFFIX = "Answer with the option's letter from the given choices directly."
POPE_SUFFIX = "Answer yes or no."


def docs_block(docs: List[str]) -> str:
    if not docs:
        return ""
    lines = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return f"Retrieved documents:\n{lines}\n\n"


def build_prompt(q: Query) -> str:
    probe = Probe(q.probe)
    if probe is Probe.CAPTION:
        return q.question
    if probe is Probe.POPE:
        return f"{q.question}\n{POPE_SUFFIX}"
    head = docs_block(q.docs) + q.question
    if probe is Probe.QA:
        return f"{head}\n{QA_SUFFIX}"
    opts = "\n".join(f"{LETTERS[i]}. {vocab.display(o)}" for i, o in enumerate(q.options))
    return f"{head}\n{opts}\n{MC_SUFFIX}"
