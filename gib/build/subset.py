"""Fixed evaluation subsets.

Every model -- open or closed -- is evaluated on the *same* subset, sampled once
at build time (stratified by question type and source split). Flip pairs are
sampled as units, so a pair is either fully in a subset or not at all.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from ..schema import BaseItem


def sample_subset(items: Sequence[BaseItem], n: int, seed: int = 0) -> List[str]:
    by_id = {it.gib_id: it for it in items}
    units: List[List[str]] = []
    seen = set()
    for it in sorted(items, key=lambda x: x.gib_id):
        if it.gib_id in seen:
            continue
        unit = [it.gib_id]
        if it.flip_partner and it.flip_partner in by_id and it.flip_partner not in seen:
            unit.append(it.flip_partner)
        seen.update(unit)
        units.append(unit)

    strata: Dict[tuple, List[List[str]]] = defaultdict(list)
    for u in units:
        first = by_id[u[0]]
        strata[(first.qa_type, first.source_split)].append(u)
    rng = random.Random(f"subset:{seed}:{n}")
    for k in sorted(strata):
        rng.shuffle(strata[k])

    total = sum(len(u) for u in units)
    picked: List[str] = []
    # proportional quotas per stratum ...
    for k in sorted(strata):
        quota = round(n * sum(len(u) for u in strata[k]) / max(1, total))
        taken = 0
        while strata[k] and taken < quota and len(picked) < n:
            u = strata[k].pop()
            if taken + len(u) > quota + 1 or len(picked) + len(u) > n:
                strata[k].append(u)
                break
            picked.extend(u)
            taken += len(u)
    # ... then top up round-robin if rounding left us short
    while len(picked) < n and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(picked) < n:
                u = strata[k].pop()
                if len(picked) + len(u) <= n:
                    picked.extend(u)
    return sorted(picked)


def write_subset(path: str | Path, ids: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_subset(path: str | Path) -> List[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
