"""LaTeX macros for every claim -- the paper never contains a typed number.

``\\gibQwenTwoVlAccClean`` etc. are generated from claims.json. If any model in
the report is simulated, no macros are written, so a paper that cites them
fails to compile instead of silently printing synthetic numbers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

_DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
_PCT = ("acc_", "follow", "GIS", "flip_", "illusion_gap", "ece")


def macro_name(key: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", key)
    out = "".join(p[:1].upper() + p[1:] for p in parts if p)
    out = "".join(_DIGITS[int(ch)] if ch.isdigit() else ch for ch in out)
    return "gib" + out


def write_macros(path: str | Path, report: Dict[str, Any]) -> None:
    path = Path(path)
    if report.get("simulated_models"):
        path.write_text("% REFUSED: this report contains simulated models ("
                        + ", ".join(report["simulated_models"]) + ").\n"
                        "% Paper macros are only generated from real model runs.\n", encoding="utf-8")
        return
    lines = ["% Auto-generated from claims.json by gib report. Do not edit.", ""]
    flagged = {k for f in (report.get("probability_flags") or {}).values() for k in f["claims"]}
    if flagged:
        lines += ["% WARNING: macros marked LOW PROBABILITY COVERAGE rest on too few real token probabilities",
                  "% (see RESULTS.md, 'Probability quality'). Do not cite them as findings.", ""]
    for key, val in sorted(report["claims"].items()):
        name = macro_name(key)
        note = "  % LOW PROBABILITY COVERAGE" if key in flagged else ""
        lines.append(f"\\newcommand{{\\{name}}}{{{val:.3f}}}{note}")
        if any(t in key for t in _PCT):
            lines.append(f"\\newcommand{{\\{name}Pct}}{{{100 * val:.1f}\\%}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
