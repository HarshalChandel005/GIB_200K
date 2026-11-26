"""paper-audit: recompute every claim from the provenance files on disk.

Fails (non-zero exit) if any claim in claims.json differs from what the
per-instance tables (``provenance/<model>.csv.gz``) produce, if a claim is
missing, or if the report cites a simulated model. Claims resting on low
probability coverage are listed (``probability_flags``) but do not fail the
audit: they are correct arithmetic on unreliable inputs, a judgement left to
the authors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..build.subset import read_subset
from ..eval.table import load_table, provenance_files
from ..schema import BaseItem
from .build import _jsonable, compute_report


class AuditError(RuntimeError):
    pass


def audit(report_dir: str | Path, items: List[BaseItem], cfg: Dict[str, Any], tol: float = 1e-6,
          allow_simulated: bool = False) -> Dict[str, Any]:
    report_dir = Path(report_dir)
    claims = json.loads((report_dir / "claims.json").read_text(encoding="utf-8"))
    files = provenance_files(report_dir)
    if not files:
        raise AuditError(f"no provenance files in {report_dir / 'provenance'}")
    keep = set(read_subset(report_dir / "items.txt"))  # exactly the items the report covered
    rep = compute_report([it for it in items if it.gib_id in keep], cfg, (load_table(f) for f in files))
    recomputed = _jsonable(rep)["claims"]
    bad = []
    for k, v in claims.items():
        r = recomputed.get(k)
        if r is None or abs(float(r) - float(v)) > tol:
            bad.append({"claim": k, "claimed": v, "recomputed": r})
    missing = sorted(set(recomputed) - set(claims))
    sim = rep.get("simulated_models", [])
    result = {"n_claims": len(claims), "mismatches": bad, "missing": missing, "simulated_models": sim,
              "probability_flags": rep.get("probability_flags", {}),
              "ok": not bad and not missing and (allow_simulated or not sim)}
    if not result["ok"]:
        raise AuditError(json.dumps(result, indent=2))
    return result
