"""Closed answer vocabularies, synonyms, and answer normalization.

Every question in GIB has a *closed* answer vocabulary (a COCO category group,
a colour set, or a spatial-predicate set). That is what makes the benchmark
scoreable without an LLM judge: a free-form answer is mapped back to exactly one
canonical answer (or to ``None`` = off-vocabulary).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# COCO categories, grouped into question frames. Each group yields questions of
# the form "What <frame> is shown in the image?"; siblings come from the group,
# so every hard negative is semantically near the truth.
# ("person" and "potted plant" are excluded: no natural sibling set.)
# --------------------------------------------------------------------------- #
COCO_GROUPS: Dict[str, List[str]] = {
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "sports equipment": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                         "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "electronic device": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "kitchen appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "tableware item": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "piece of furniture": ["chair", "couch", "bed", "dining table", "bench", "toilet"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "street object": ["traffic light", "fire hydrant", "stop sign", "parking meter"],
    "household item": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}
CATEGORY_TO_GROUP: Dict[str, str] = {c: g for g, cs in COCO_GROUPS.items() for c in cs}

# Surface form used in documents and options.
DISPLAY: Dict[str, str] = {
    "tv": "television", "sports ball": "ball", "hair drier": "hair dryer",
    "remote": "remote control", "mouse": "computer mouse",
}

# Synonyms used both to read model answers and to detect object mentions (CHAIR).
COCO_SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child",
               "children", "kid", "guy", "lady", "player", "skier", "surfer", "skateboarder"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "sedan", "taxi", "cab"],
    "motorcycle": ["motorcycle", "motorbike", "motor bike", "dirt bike"],
    "airplane": ["airplane", "plane", "aeroplane", "jet", "aircraft", "airliner"],
    "bus": ["bus", "buses", "double decker"],
    "train": ["train", "locomotive"],
    "truck": ["truck", "lorry", "pickup truck"],
    "boat": ["boat", "ship", "sailboat", "kayak", "canoe", "yacht"],
    "traffic light": ["traffic light", "traffic signal", "stoplight", "stop light"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench"],
    "bird": ["bird", "pigeon", "seagull", "gull", "duck", "parrot", "sparrow"],
    "cat": ["cat", "kitten", "kitty"],
    "dog": ["dog", "puppy", "pup"],
    "horse": ["horse", "pony", "foal"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull", "calf", "ox"],
    "elephant": ["elephant"],
    "bear": ["bear", "grizzly", "polar bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack", "knapsack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "hand bag"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage", "suit case"],
    "frisbee": ["frisbee", "flying disc"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball", "soccer ball", "football", "baseball", "tennis ball",
                    "basketball", "volleyball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard", "skate board"],
    "surfboard": ["surfboard", "surf board"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "wineglass"],
    "cup": ["cup", "mug", "coffee cup"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake"],
    "chair": ["chair"],
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "houseplant", "house plant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "tv screen"],
    "laptop": ["laptop", "notebook computer"],
    "mouse": ["computer mouse", "mouse"],
    "remote": ["remote control", "remote", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "cellphone", "phone", "mobile phone", "smartphone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy", "stuffed animal", "stuffed bear"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "toothbrush": ["toothbrush", "tooth brush"],
}

# --------------------------------------------------------------------------- #
# Attributes (colour) and relations (spatial predicates)
# --------------------------------------------------------------------------- #
COLORS: List[str] = ["white", "black", "gray", "red", "blue", "green", "yellow",
                     "brown", "orange", "pink", "purple"]
COLOR_SYNONYMS: Dict[str, List[str]] = {c: [c] for c in COLORS}
COLOR_SYNONYMS.update({
    "gray": ["gray", "grey", "silver"], "brown": ["brown", "tan", "beige"],
    "yellow": ["yellow", "gold", "golden"], "blue": ["blue", "navy"],
    "red": ["red", "maroon"], "purple": ["purple", "violet"],
})
# Perceptually nearest colours first -> the hardest negatives come first.
COLOR_NEIGHBORS: Dict[str, List[str]] = {
    "white": ["gray", "yellow", "pink", "blue"],
    "black": ["gray", "brown", "blue", "purple"],
    "gray": ["white", "black", "blue", "brown"],
    "red": ["orange", "pink", "purple", "brown"],
    "blue": ["purple", "green", "gray", "black"],
    "green": ["blue", "yellow", "gray", "brown"],
    "yellow": ["orange", "white", "green", "brown"],
    "brown": ["orange", "black", "red", "gray"],
    "orange": ["red", "yellow", "brown", "pink"],
    "pink": ["red", "purple", "white", "orange"],
    "purple": ["blue", "pink", "red", "black"],
}

SPATIAL: List[str] = ["on", "under", "above", "below", "behind", "in front of", "next to", "inside"]
SPATIAL_SYNONYMS: Dict[str, List[str]] = {
    "on": ["on", "on top of", "atop", "sitting on", "standing on", "lying on", "laying on",
           "parked on", "resting on", "mounted on"],
    "under": ["under", "underneath", "beneath"],
    "above": ["above", "over"],
    "below": ["below"],
    "behind": ["behind"],
    "in front of": ["in front of"],
    "next to": ["next to", "beside", "alongside"],
    "inside": ["inside", "inside of", "within"],
}
# Antonym first: "swap the relational predicate" hard negatives.
SPATIAL_NEIGHBORS: Dict[str, List[str]] = {
    "on": ["under", "next to", "above", "inside"],
    "under": ["on", "above", "behind", "next to"],
    "above": ["below", "on", "next to", "behind"],
    "below": ["above", "under", "next to", "behind"],
    "behind": ["in front of", "next to", "under", "above"],
    "in front of": ["behind", "next to", "on", "under"],
    "next to": ["behind", "in front of", "on", "inside"],
    "inside": ["on", "next to", "under", "behind"],
}
LEFTRIGHT: List[str] = ["left", "right"]
LEFTRIGHT_SYNONYMS: Dict[str, List[str]] = {"left": ["left"], "right": ["right"]}


def group_synonyms(group: str) -> Dict[str, List[str]]:
    """Canonical answer -> synonyms for one answer vocabulary."""
    if group == "color":
        return COLOR_SYNONYMS
    if group == "spatial":
        return SPATIAL_SYNONYMS
    if group == "leftright":
        return LEFTRIGHT_SYNONYMS
    if group in COCO_GROUPS:
        return {c: COCO_SYNONYMS[c] for c in COCO_GROUPS[group]}
    raise KeyError(f"unknown answer group: {group!r}")


def display(answer: str) -> str:
    return DISPLAY.get(answer, answer)


# --------------------------------------------------------------------------- #
# Normalization + extraction
# --------------------------------------------------------------------------- #
_IRREGULAR = {
    "people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse",
    "knives": "knife", "geese": "goose", "teeth": "tooth", "feet": "foot", "leaves": "leaf",
    "shelves": "shelf", "wolves": "wolf", "calves": "calf", "halves": "half",
    "buses": "bus", "skis": "skis", "scissors": "scissors", "sheep": "sheep", "cattle": "cattle",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")


def singularize(word: str) -> str:
    w = word
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) <= 3 or w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "zes", "sses", "oes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def normalize_tokens(text: str) -> List[str]:
    text = _PUNCT.sub(" ", (text or "").lower().replace("-", " "))
    return [singularize(t) for t in text.split() if t not in _ARTICLES]


@lru_cache(maxsize=64)
def _phrase_table(key: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> List[Tuple[Tuple[str, ...], str]]:
    table = []
    for canon, syns in key:
        for s in syns:
            toks = tuple(normalize_tokens(s))
            if toks:
                table.append((toks, canon))
    table.sort(key=lambda x: -len(x[0]))  # longest phrase wins at a position
    return table


def _table_for(vocab: Dict[str, Sequence[str]]):
    return _phrase_table(tuple(sorted((c, tuple(s)) for c, s in vocab.items())))


def find_mentions(text: str, vocab: Dict[str, Sequence[str]]) -> List[Tuple[int, str]]:
    """Non-overlapping (position, canonical) mentions, longest match first."""
    toks = normalize_tokens(text)
    table = _table_for(vocab)
    out: List[Tuple[int, str]] = []
    i = 0
    while i < len(toks):
        for phrase, canon in table:
            n = len(phrase)
            if tuple(toks[i:i + n]) == phrase:
                out.append((i, canon))
                i += n
                break
        else:
            i += 1
    return out


def extract_answer(text: Optional[str], group: str) -> Optional[str]:
    """Map a free-form answer to one canonical answer of ``group`` (first mention)."""
    if not text:
        return None
    mentions = find_mentions(text, group_synonyms(group))
    return mentions[0][1] if mentions else None


def chair_mentions(text: Optional[str]) -> List[str]:
    """COCO categories mentioned in a caption (CHAIR object extraction)."""
    if not text:
        return []
    return sorted({c for _, c in find_mentions(text, COCO_SYNONYMS)})


# --------------------------------------------------------------------------- #
# Document rendering helpers
# --------------------------------------------------------------------------- #
_VOWEL = re.compile(r"\b([Aa])n? (\{ANS\})")


_PLURAL_ONLY = {"skis", "scissors", "jeans", "pants", "shorts", "trousers", "glasses", "sunglasses",
                "goggles", "headphones", "clothes", "stairs", "leaves", "people"}
_S_SINGULAR = {"bus", "glass", "dress", "grass", "lens", "gas", "canvas", "tennis", "chassis", "series",
               "species", "news", "cross", "boss"}


def is_plural(name: str) -> bool:
    """Heuristic number of a (Visual Genome / COCO) noun phrase: 'jeans' -> True, 'bus' -> False."""
    w = (name or "").strip().lower().split()[-1:] or [""]
    w = w[0]
    if w in _PLURAL_ONLY:
        return True
    if w in _S_SINGULAR or w.endswith(("ss", "us", "is", "ics")) or len(w) <= 3:
        return False
    return w.endswith("s")


def pluralize(word: str) -> str:
    rev = {"mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
           "scissors": "scissors", "person": "people", "bus": "buses"}
    if word in rev:
        return rev[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def fill(template: str, answer: str, slots: Optional[Dict[str, str]] = None) -> str:
    """Render a document template with ``answer`` (articles fixed: a/an)."""
    value = display(answer)
    out = template
    for k, v in (slots or {}).items():
        out = out.replace("{" + k + "}", v)
    if "{ANS_PL}" in out:
        out = out.replace("{ANS_PL}", pluralize(value))

    def _article(m: re.Match) -> str:
        if is_plural(value):  # "skis", "scissors": no article
            return m.group(2)
        a = "an" if value[:1].lower() in "aeiou" else "a"
        if m.group(1) == "A":
            a = a.capitalize()
        return f"{a} {m.group(2)}"

    out = _VOWEL.sub(_article, out)
    out = out.replace("{ANS}", value)
    return out[:1].upper() + out[1:]


def mention_to_slot(text: str, answer: str) -> Optional[str]:
    """Turn a caption that mentions ``answer`` (via any synonym) into a template.

    Returns ``None`` if the caption does not mention the answer. Plural mentions
    become ``{ANS_PL}`` so swapped claims stay grammatical.
    """
    syns = sorted(COCO_SYNONYMS.get(answer, [answer]), key=len, reverse=True)
    for s in syns:
        for form, slot in ((pluralize(s), "{ANS_PL}"), (s, "{ANS}")):
            pat = re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE)
            if pat.search(text):
                return pat.sub(slot, text, count=1).strip()
    return None


def all_groups() -> Iterable[str]:
    return list(COCO_GROUPS) + ["color", "spatial", "leftright"]
