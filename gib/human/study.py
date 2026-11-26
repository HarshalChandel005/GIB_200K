"""Human baseline: export a self-contained study page, then score the responses.

``export`` picks a stratified set of (item, stress condition) pairs from the
evaluation subset, renders their images, and writes ``index.html`` -- a single
offline page (no server) that shows each item exactly as the models saw it
(same image, same retrieved documents, same question), records the free-text
answer and the response time, and lets the participant download their
responses as a JSON file.

``analyze`` scores those files with the *same* answer extractor used for the
models, reports accuracy per stressor with 95% CIs, inter-annotator agreement
(Fleiss' kappa), and -- given a model report -- each model's accuracy on exactly
the same (item, condition) pairs, so the human-model gap is like-for-like.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import vocab
from ..axes.degrade import materialize
from ..axes.expand import expand_item
from ..eval import stats as S
from ..schema import BaseItem

STRESSORS = {
    "clean": "clean",
    "ret_k10": "retrieval k=10", "ret_k20": "retrieval k=20",
    "vis_gauss_s8": "blur σ=8", "vis_gauss_s16": "blur σ=16",
    "vis_motion_k40": "motion blur 40px", "vis_crop_f0.1": "crop 10%", "vis_crop_f0.05": "crop 5%",
    "conflict": "conflict", "conflict_g16": "conflict + blur σ=16",
    "grid_g16_k20": "blur σ=16 + k=20",
}

INSTRUCTIONS = ("Answer each question about the image in a word or short phrase. Some questions come with "
                "'retrieved documents' from a search system; they may help, but they can also be wrong. "
                "If you really cannot tell, press “I can't tell”.")


def export(items: Sequence[BaseItem], cfg: Dict[str, Any], image_root: str | Path, out_dir: str | Path,
           n: int = 120, seed: int = 0, conditions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    conds = list(conditions or STRESSORS)
    rng = random.Random(f"human:{seed}")
    pool = sorted(items, key=lambda it: it.gib_id)
    rng.shuffle(pool)
    rows = []
    for i, it in enumerate(pool[:n]):  # each item is shown in exactly one condition
        cond = conds[i % len(conds)]
        q = next((q for q in expand_item(it, cfg) if q.condition == cond and q.probe == "qa"), None)
        if q is None:
            continue
        path = materialize(it, q.image_recipe, image_root, cfg, out_dir / "img")
        rows.append({"hid": f"h{i:04d}", "gib_id": it.gib_id, "condition": cond, "stressor": STRESSORS.get(cond, cond),
                     "group": it.group, "truth": it.answer, "question": q.question, "docs": q.docs,
                     "image": f"img/{path.name}"})
    public = [{k: r[k] for k in ("hid", "question", "docs", "image")} for r in rows]
    (out_dir / "key.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(_page(public), encoding="utf-8")
    (out_dir / "responses").mkdir(exist_ok=True)
    return {"n_items": len(rows), "conditions": conds, "out": str(out_dir)}


def _page(items: List[Dict[str, Any]]) -> str:
    data = json.dumps(items).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image question study</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#666;--card:#f4f4f6;--accent:#2563eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#eee;--muted:#aaa;--card:#202024;--accent:#60a5fa}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:720px;margin:auto}} img{{max-width:100%;border-radius:6px;display:block;margin:8px auto}}
.card{{background:var(--card);padding:12px 14px;border-radius:8px;margin:10px 0}} .docs li{{font-size:14px}}
input[type=text]{{width:100%;box-sizing:border-box;font-size:16px;padding:8px;margin:6px 0}}
button{{font-size:15px;padding:8px 14px;margin:4px 6px 4px 0;border-radius:6px;border:1px solid var(--muted);
background:var(--bg);color:var(--fg);cursor:pointer}} button.primary{{background:var(--accent);color:#fff;border:none}}
.muted{{color:var(--muted);font-size:14px}}
</style></head><body><main>
<h1>Image question study</h1>
<section id="start" class="card"><p>{INSTRUCTIONS}</p>
<p class="muted">Your answers and response times are saved only in this page until you download them at the end.</p>
<label>Participant code <input type="text" id="pid" autocomplete="off"></label>
<button class="primary" id="go">Start</button></section>
<section id="task" hidden><p class="muted" id="prog"></p><div class="card">
<img id="im" alt="question image"><div id="docs"></div><p><b id="q"></b></p>
<input type="text" id="ans" autocomplete="off" placeholder="Your answer">
<button class="primary" id="next">Next</button><button id="cant">I can't tell</button></div></section>
<section id="done" class="card" hidden><p>Thank you! Please download your answers and send the file to the study team.</p>
<button class="primary" id="dl">Download answers</button></section>
</main>
<script id="data" type="application/json">{data}</script>
<script>
const items = JSON.parse(document.getElementById('data').textContent);
let order = [], i = 0, t0 = 0, pid = '', out = [];
function rng(seed) {{ let h = 2166136261; for (const c of seed) {{ h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }}
  return () => {{ h ^= h << 13; h ^= h >>> 17; h ^= h << 5; return ((h >>> 0) % 1e9) / 1e9; }}; }}
function show() {{
  if (i >= order.length) {{ document.getElementById('task').hidden = true; document.getElementById('done').hidden = false; return; }}
  const it = items[order[i]];
  document.getElementById('prog').textContent = `Question ${{i + 1}} of ${{order.length}}`;
  document.getElementById('im').src = it.image;
  const d = document.getElementById('docs');
  d.innerHTML = '';
  if (it.docs.length) {{ const p = document.createElement('p'); p.className = 'muted'; p.textContent = 'Retrieved documents:';
    const ul = document.createElement('ul'); ul.className = 'docs';
    it.docs.forEach(x => {{ const li = document.createElement('li'); li.textContent = x; ul.appendChild(li); }});
    d.appendChild(p); d.appendChild(ul); }}
  document.getElementById('q').textContent = it.question;
  const a = document.getElementById('ans'); a.value = ''; a.focus();
  t0 = performance.now();
}}
function record(answer) {{
  out.push({{hid: items[order[i]].hid, answer, ms: Math.round(performance.now() - t0)}});
  i += 1; show();
}}
document.getElementById('go').onclick = () => {{
  pid = document.getElementById('pid').value.trim(); if (!pid) return;
  const r = rng(pid); order = items.map((_, k) => k);
  for (let k = order.length - 1; k > 0; k--) {{ const j = Math.floor(r() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }}
  document.getElementById('start').hidden = true; document.getElementById('task').hidden = false; show();
}};
document.getElementById('next').onclick = () => record(document.getElementById('ans').value.trim());
document.getElementById('cant').onclick = () => record("");
document.getElementById('ans').addEventListener('keydown', e => {{ if (e.key === 'Enter') record(e.target.value.trim()); }});
document.getElementById('dl').onclick = () => {{
  const blob = new Blob([JSON.stringify({{participant: pid, responses: out}}, null, 1)], {{type: 'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `responses_${{pid}}.json`; a.click();
}};
</script></body></html>"""


def _fleiss_kappa(table: np.ndarray) -> Optional[float]:
    """table: (items x categories) counts, equal raters per item."""
    t = table[table.sum(1) >= 2]
    if len(t) < 2:
        return None
    n = t.sum(1)
    p_j = t.sum(0) / n.sum()
    P_i = ((t * (t - 1)).sum(1)) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else None


def analyze(study_dir: str | Path, response_files: Sequence[str | Path], provenance: Optional[pd.DataFrame] = None,
            iters: int = 2000, seed: int = 0) -> Dict[str, Any]:
    key = {r["hid"]: r for r in json.loads((Path(study_dir) / "key.json").read_text(encoding="utf-8"))}
    rows = []
    for f in response_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in data["responses"]:
            k = key.get(r["hid"])
            if k is None:
                continue
            ext = vocab.extract_answer(r["answer"], k["group"])
            rows.append({"participant": data["participant"], "hid": r["hid"], "gib_id": k["gib_id"],
                         "condition": k["condition"], "stressor": k["stressor"], "correct": float(ext == k["truth"]),
                         "abstained": float(r["answer"] == ""), "ms": r["ms"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_participants": 0}
    out: Dict[str, Any] = {"n_participants": int(df["participant"].nunique()), "n_items": int(df["hid"].nunique()),
                           "n_responses": int(len(df)), "accuracy_by_stressor": {}, "median_ms_by_stressor": {}}
    per_item = df.groupby(["hid", "stressor"])["correct"].mean().reset_index()
    for st, sub in per_item.groupby("stressor"):
        W = S.boot_weights(len(sub), iters, seed)
        out["accuracy_by_stressor"][st] = S.strip(S.mean_stat(sub["correct"].to_numpy(), W))
        out["median_ms_by_stressor"][st] = float(df[df["stressor"] == st]["ms"].median())
    W = S.boot_weights(len(per_item), iters, seed)
    out["accuracy_overall"] = S.strip(S.mean_stat(per_item["correct"].to_numpy(), W))
    tab = df.pivot_table(index="hid", columns="correct", values="participant", aggfunc="count", fill_value=0)
    out["fleiss_kappa_correctness"] = _fleiss_kappa(tab.to_numpy(float))
    if provenance is not None:
        qa = provenance[provenance["probe"] == "qa"]
        qa = qa.astype({c: object for c in ("model", "gib_id", "condition") if c in qa})  # no empty category groups
        pairs = set(zip(df["gib_id"], df["condition"]))
        m = qa[[p in pairs for p in zip(qa["gib_id"], qa["condition"])]]
        cond_to_st = dict(zip(df["condition"], df["stressor"]))
        gap: Dict[str, Any] = defaultdict(dict)
        for (model, cond), sub in m.groupby(["model", "condition"]):
            st = cond_to_st[cond]
            gap[model][st] = float(sub["correct"].mean())
        out["models_on_same_items"] = gap
    return out
