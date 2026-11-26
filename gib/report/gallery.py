"""Qualitative 'hallucination gallery': real instances that fooled the models."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..axes.degrade import render
from ..axes.expand import expand_item
from ..schema import BaseItem

PER_AXIS = 4
PICKS = [("Retrieval noise (k = 20)", "ret_k20"), ("Cross-modal conflict", "conflict"),
         ("Blur with correct text (Sensory Capture probe)", "grid_g16_k0"),
         ("Low-resource crop (5% visible)", "vis_crop_f0.05")]
GALLERY_CONDITIONS = ["clean"] + [c for _, c in PICKS]  # the only free-form rows the gallery reads


def _thumb(img) -> str:
    img = img.copy()
    img.thumbnail((320, 320))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_gallery(path: str | Path, df: pd.DataFrame, items: List[BaseItem], cfg: Dict[str, Any],
                  image_root: str | Path) -> None:
    by_id = {it.gib_id: it for it in items}
    qa = df[df["probe"] == "qa"]
    sections = []
    for title, cond in PICKS:
        sub = qa[qa["condition"] == cond]
        if sub.empty:
            continue
        wrong = sub.groupby("gib_id")["correct"].mean()
        clean_ok = qa[qa["condition"] == "clean"].groupby("gib_id")["correct"].mean()
        cand = [g for g in wrong.index if wrong[g] < 1 and clean_ok.get(g, 0) > 0]
        cards = []
        for g in sorted(cand)[:PER_AXIS]:
            it = by_id[g]
            q = next(q for q in expand_item(it, cfg) if q.condition == cond and q.probe == "qa")
            img = render(it, q.image_recipe, image_root, cfg)
            answers = sub[sub["gib_id"] == g][["model", "text", "correct"]]
            docs = "".join(f"<li>{html.escape(d)}</li>" for d in q.docs[:6]) + ("<li>…</li>" if len(q.docs) > 6 else "")
            ans = "".join(
                f"<li class='{'ok' if c == 1 else 'bad'}'><b>{html.escape(m)}</b>: {html.escape(str(t))}</li>"
                for m, t, c in answers.itertuples(index=False))
            cards.append(f"<div class='card'><img src='{_thumb(img)}' alt='{html.escape(g)}'>"
                         f"<p class='q'>{html.escape(q.question)}</p><p>Truth: <b>{html.escape(it.answer)}</b></p>"
                         f"<p class='lab'>Retrieved documents</p><ul class='docs'>{docs or '<li>(none)</li>'}</ul>"
                         f"<p class='lab'>Model answers</p><ul>{ans}</ul><p class='id'>{html.escape(g)}</p></div>")
        if cards:
            sections.append(f"<h2>{html.escape(title)}</h2><div class='grid'>{''.join(cards)}</div>")
    sim = sorted(m for m in df["model"].unique() if str(m).startswith("sim-"))
    banner = (f"<p class='warn'>Simulated models present ({', '.join(sim)}): pipeline test only.</p>" if sim else "")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GIB hallucination gallery</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#666;--card:#f6f6f7;--ok:#1e7e34;--bad:#b02a37}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#eee;--muted:#aaa;--card:#222226;--ok:#5cb85c;--bad:#e5737f}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{background:var(--card);border-radius:8px;padding:10px}} .card img{{width:100%;border-radius:4px}}
.q{{font-weight:600}} .lab{{color:var(--muted);margin:6px 0 2px;font-size:12px}} ul{{margin:0;padding-left:18px}}
.docs li{{font-size:12px}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}} .id{{color:var(--muted);font-size:11px}}
.warn{{background:#fff3cd;color:#664d03;padding:8px;border-radius:6px}}</style></head>
<body><h1>Hallucination gallery</h1>{banner}<p>Items the models answered correctly with clean inputs but got wrong under stress.</p>
{''.join(sections) or '<p>No failures to show.</p>'}</body></html>"""
    Path(path).write_text(page, encoding="utf-8")
