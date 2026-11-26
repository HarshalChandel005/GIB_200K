"""Data contracts shared by every stage.

Three record types flow through the pipeline:

* :class:`BaseItem`  -- one image-question pair built from real annotations
  (COCO instances/captions + Visual Genome attributes/relations).
* :class:`Query`     -- one model call: an image recipe + retrieved documents +
  a probe (free-form answer, multiple choice, binary A/B, POPE yes/no, caption).
* :class:`Prediction`-- what a model returned for one query.

Everything downstream (metrics, statistics, figures, paper numbers) is a pure
function of ``items.jsonl`` + ``predictions.jsonl``; nothing is typed by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Type, TypeVar

T = TypeVar("T")


class QAType(str, Enum):
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    RELATION = "relation"


class Probe(str, Enum):
    QA = "qa"            # free-form short answer (accuracy, taxonomy, GIS)
    MC = "mc"            # letter distribution over the item's fixed option set (ECE, R, CSR)
    AB = "ab"            # binary image-truth vs text-claim probe (Trust Index, cue combination)
    POPE = "pope"        # yes/no object presence
    CAPTION = "caption"  # one-sentence description (CHAIR)


@dataclass
class BaseItem:
    gib_id: str
    image_path: str                 # POSIX path relative to the image root
    source: str                     # "coco" | "fixture" | "fresh"
    source_split: str               # e.g. "val2017", "train2017", "fresh"
    qa_type: str                    # QAType value
    group: str                      # answer vocabulary, e.g. "animal", "color", "spatial", "leftright"
    question: str
    answer: str                     # canonical answer
    siblings: List[str]             # ranked hard negatives (same vocabulary)
    doc_templates: List[str]        # retrieved-document templates with an {ANS} slot
    target_bbox: List[float]        # x, y, w, h of the queried object (original pixels)
    image_size: List[int]           # original width, height
    objects: List[List[Any]]        # [[category, x, y, w, h], ...] ground truth (CHAIR/POPE)
    pope_pos: Optional[str] = None  # present category for the POPE positive probe
    pope_neg: Optional[str] = None  # absent (adversarial, co-occurring) category
    conflict_claim: Optional[str] = None
    flip_partner: Optional[str] = None   # gib_id of the item whose conflict is the mirror image
    slots: Dict[str, str] = field(default_factory=dict)  # e.g. {"OBJ": "car"} for templates
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> List[str]:
        """The item's fixed answer set: truth + up to 4 hard negatives."""
        return [self.answer] + self.siblings[:4]


@dataclass
class Query:
    qid: str
    gib_id: str
    condition: str                  # e.g. "clean", "ret_k5", "vis_gauss_s4", "conflict_g16"
    axis: str                       # clean | no_ctx | retrieval | visual | conflict | grid | blank | pope | caption
    probe: str                      # Probe value
    image_key: str                  # identical keys share one image (and one prefix cache)
    image_recipe: Dict[str, Any]    # {"method": "none" | "gauss" | "motion" | "crop" | "noise" | "blank", ...}
    docs: List[str]
    question: str
    options: List[str] = field(default_factory=list)
    image_truth: Optional[str] = None
    text_claim: Optional[str] = None     # answer asserted by misleading text, if any
    doc_claims: List[str] = field(default_factory=list)  # every answer asserted by the docs
    level: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    qid: str
    model: str
    text: Optional[str] = None                 # qa / caption output
    option_probs: Optional[List[float]] = None # mc / ab: one prob per option; pope: [p_yes, p_no]
    attn_img: Optional[List[float]] = None     # per-layer attention mass on image tokens (last query position)
    attn_post: Optional[List[float]] = None    # per-layer mass on text tokens after the image
    n_img_tokens: Optional[int] = None
    n_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    reproducible: bool = True                  # False for closed APIs
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # diagnostics, e.g. probability mass on option letters


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def _to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_dict(r), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_dict(r), ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path, cls: Optional[Type[T]] = None) -> List[Any]:
    rows = list(iter_jsonl(path))
    if cls is None:
        return rows
    names = {f.name for f in fields(cls)}  # tolerate extra keys from newer writers
    return [cls(**{k: v for k, v in r.items() if k in names}) for r in rows]
