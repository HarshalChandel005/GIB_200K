"""Command line: python -m gib <command> ...

  build            COCO (+VG) annotations -> items.jsonl + fixed subsets
  demo             offline end-to-end run on synthetic fixtures with SIMULATED models
  estimate         count queries / image encodings / API calls for a subset
  evaluate         run one model (hf:/openai:/gemini:/sim:) over a subset, resumable
  speedtest        measure the prefix-cache speed-up on real data (and check outputs match)
  report           metrics + statistics + figures + gallery + macros from one or more runs
  audit            recompute every claim from the provenance files (non-zero exit on mismatch)
  human-export     write the offline human-study page
  human-analyze    score human responses, compare with models on the same items
  taxonomy-export  sample failures for blind human labelling
  taxonomy-kappa   agreement between the rule-based taxonomy and human labels
  materialize      write degraded images to disk
  fetch-images     download only the COCO images the items use
  release          package annotations + dataset card for GitHub / Hugging Face
  pack             source zip of a git commit, reproducible byte for byte
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import load_config


def _cfg(args) -> dict:
    return load_config(getattr(args, "config", None))


def cmd_build(args) -> None:
    from .build.coco_vg import build
    from .build.subset import sample_subset, write_subset

    cfg = _cfg(args)
    fresh = tuple(args.fresh.split(",", 1)) if args.fresh else None
    items = build(args.out, cfg, args.coco_root, splits=args.splits.split(","), vg_root=args.vg_root, fresh=fresh)
    for name, n in cfg["subsets"].items():
        write_subset(Path(args.out) / "subsets" / f"{name}.txt", sample_subset(items, n, cfg["seed"]))
    print(Path(args.out, "build_report.json").read_text(encoding="utf-8"))


def cmd_estimate(args) -> None:
    from .evaluate import count_work, load_items

    cfg = _cfg(args)
    items = load_items(args.data, args.subset)
    w = count_work(items, cfg)
    per_item = {k: round(v / max(1, len(items)), 1) for k, v in w["by_probe"].items()}
    w.update({"items": len(items), "queries_per_item": per_item,
              "note": "open models: one image encoding per 'image_encodings', every query is a short text "
                      "suffix; API models: one call per query (cached on disk)."})
    print(json.dumps(w, indent=2))


def _adapter(args, cfg, items):
    from .models.registry import make_adapter, parse_spec

    _, backend, _ = parse_spec(args.model)
    kw = {}
    if backend == "hf":
        kw = {"load_4bit": args.load_4bit, "attn_implementation": args.attn, "prefix_cache": not args.no_prefix_cache,
              "max_pixels": args.max_pixels, "device": args.device, "dtype": args.dtype,
              "trust_remote_code": args.trust_remote_code, "batch_size": args.batch_size}
    elif backend in ("openai", "gemini"):
        kw = {"cache_dir": str(Path(args.runs) / "_api_cache"), "max_workers": args.workers}
    return make_adapter(args.model, cfg, items={it.gib_id: it for it in items}, **kw)


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate, load_items

    cfg = _cfg(args)
    items = load_items(args.data, args.subset)
    adapter = _adapter(args, cfg, items)
    meta = evaluate(adapter, items, cfg, args.image_root, Path(args.runs) / adapter.name,
                    limit_groups=args.limit_groups)
    print(json.dumps({k: meta[k] for k in ("model", "n_items", "n_queries", "new_predictions",
                                            "errors_this_run", "wall_seconds")}, indent=2))


def cmd_speedtest(args) -> None:
    from .axes.degrade import render
    from .axes.expand import expand
    from .evaluate import group_queries, load_items

    cfg = _cfg(args)
    items = load_items(args.data, args.subset)[: args.items]
    adapter = _adapter(args, cfg, items)
    by_id = {it.gib_id: it for it in items}
    groups = [(k, g) for k, g in group_queries(expand(items, cfg)).items() if k != "blank"][: args.groups]
    bs = getattr(adapter, "batch_size", 1)
    if groups:  # warm-up on one group: runs the one-time self-checks so they are not timed
        key, qs = groups[0]
        adapter.run_group(render(by_id[qs[0].gib_id], qs[0].image_recipe, args.image_root, cfg), qs)
    res = {}
    outs = {}
    for mode in (["batched"] if bs > 1 else []) + ["prefix_cache", "per_query_prefill"]:
        adapter.use_prefix_cache = mode != "per_query_prefill"
        adapter.batch_size = bs if mode == "batched" else 1
        t0, n = time.perf_counter(), 0
        preds = []
        for key, qs in groups:
            img = render(by_id[qs[0].gib_id], qs[0].image_recipe, args.image_root, cfg)
            preds += adapter.run_group(img, qs)
            n += len(qs)
        dt = time.perf_counter() - t0
        res[mode] = {"queries": n, "seconds": round(dt, 2), "ms_per_query": round(1000 * dt / max(1, n), 1)}
        outs[mode] = {p.qid: p for p in preds}
    adapter.batch_size = bs

    def agree(a, b):
        diffs = [max(abs(x - y) for x, y in zip(a[q].option_probs, b[q].option_probs))
                 for q in a if a[q].option_probs and b[q].option_probs]
        same = [a[q].text == b[q].text for q in a if a[q].text is not None]
        return (max(diffs) if diffs else None), ((sum(same) / len(same)) if same else None)

    res["speedup"] = round(res["per_query_prefill"]["seconds"] / max(1e-9, res["prefix_cache"]["seconds"]), 2)
    res["max_prob_diff"], res["identical_free_text_share"] = agree(outs["prefix_cache"], outs["per_query_prefill"])
    if "batched" in outs:
        res["batch_speedup"] = round(res["prefix_cache"]["seconds"] / max(1e-9, res["batched"]["seconds"]), 2)
        d, s = agree(outs["batched"], outs["prefix_cache"])
        res["batched_vs_one_by_one"] = {"max_prob_diff": d, "identical_free_text_share": s}
    res["adapter"] = adapter.meta()
    print(json.dumps(res, indent=2, default=str))


def cmd_report(args) -> None:
    from .evaluate import load_items
    from .report.build import write_report

    cfg = _cfg(args)
    items = load_items(args.data, args.subset)
    runs: List[str] = []
    for r in args.runs:
        runs += sorted(glob.glob(r)) or [r]
    human = json.loads(Path(args.human).read_text(encoding="utf-8")) if args.human else None
    rep = write_report(args.out, items, cfg, runs, image_root=args.image_root, human=human,
                       figures=not args.no_figures)
    print(Path(args.out, "RESULTS.md").read_text(encoding="utf-8")[:3000])
    if rep.get("simulated_models"):
        print("\nNOTE: simulated models present -> macros.tex refused, numbers are not results.", file=sys.stderr)


def cmd_audit(args) -> None:
    from .evaluate import load_items
    from .report.audit import AuditError, audit

    cfg = _cfg(args)
    items = load_items(args.data, "all")
    try:
        res = audit(args.report, items, cfg, allow_simulated=args.allow_simulated)
    except AuditError as e:
        print("AUDIT FAILED\n" + str(e))
        raise SystemExit(1)
    print(json.dumps(res, indent=2))


def cmd_human_export(args) -> None:
    from .evaluate import load_items
    from .human.study import export

    cfg = _cfg(args)
    print(json.dumps(export(load_items(args.data, args.subset), cfg, args.image_root, args.out, n=args.n,
                            seed=cfg["seed"]), indent=2))


def cmd_human_analyze(args) -> None:
    from .eval.table import load_rows
    from .human.study import analyze

    files: List[str] = []
    for r in args.responses:
        files += sorted(glob.glob(r))
    prov = None
    if args.report:  # only the models' free-form answers on the study's items
        study_ids = {k["gib_id"] for k in json.loads((Path(args.study) / "key.json").read_text(encoding="utf-8"))}
        prov = load_rows(args.report, lambda d: (d["probe"] == "qa") & d["gib_id"].isin(study_ids),
                         ["model", "probe", "gib_id", "condition", "correct"])
    res = analyze(args.study, files, prov)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_taxonomy_export(args) -> None:
    from .eval.table import load_rows
    from .evaluate import load_items
    from .human.taxonomy_check import export

    cfg = _cfg(args)
    prov = load_rows(args.report, lambda d: (d["probe"] == "qa") & d["failure"].notna(),  # wrong answers only
                     ["model", "probe", "gib_id", "condition", "text", "failure"])
    print(json.dumps(export(prov, load_items(args.data, "all"), cfg, args.image_root, args.out, n=args.n), indent=2))


def cmd_taxonomy_kappa(args) -> None:
    from .human.taxonomy_check import kappa

    print(json.dumps(kappa(args.labels, args.key), indent=2))


def cmd_materialize(args) -> None:
    from .axes.degrade import materialize, visual_recipes
    from .evaluate import load_items

    cfg = _cfg(args)
    n = 0
    for it in load_items(args.data, args.subset):
        for r in [{"method": "none"}] + visual_recipes(cfg):
            materialize(it, r, args.image_root, cfg, args.out)
            n += 1
    print(f"wrote {n} images to {args.out}")


def cmd_fetch_images(args) -> None:
    """Download only the COCO images the chosen items use (not all 18 GB of train2017), in parallel.
    Re-running it only fetches what is still missing."""
    import shutil
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .evaluate import load_items

    items = load_items(args.data, args.subset)
    root = Path(args.image_root)
    todo: dict = {}
    for it in items:  # up to four items share an image: fetch each image once
        path, url = root / it.image_path, it.provenance.get("coco_url")
        if url and path not in todo and not path.exists():
            todo[path] = url
    print(f"{len(items)} items, {len(todo)} images to download", flush=True)

    def fetch(path: Path, url: str) -> Optional[str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                    shutil.copyfileobj(r, f)
                tmp.replace(path)  # complete files only: an interrupted download is retried next run
                return None
            except Exception as e:  # network hiccup: retry, then report
                if attempt == 2:
                    return f"{url} ({e})"
                time.sleep(2 * (attempt + 1))
        return None

    failed = 0
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futures = [ex.submit(fetch, p, u) for p, u in todo.items()]
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.result()
            if err:
                failed += 1
                print(f"failed: {err}", flush=True)
            if i % 2000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"done; {failed} failed")


def cmd_release(args) -> None:
    from .release.hf_export import export

    print(json.dumps(export(args.data, args.out, _cfg(args)), indent=2))


def cmd_pack(args) -> None:
    from .release.pack import pack

    print(json.dumps(pack(Path(__file__).resolve().parent.parent, args.out, args.ref, args.prefix), indent=2))


def cmd_demo(args) -> None:
    """Offline smoke run: fixtures -> build -> 3 simulated models -> report -> audit."""
    from .build.coco_vg import build
    from .build.fixtures import make_fixture
    from .build.subset import sample_subset, write_subset
    from .evaluate import evaluate, load_items
    from .models.simulated import SimulatedVLM
    from .report.audit import audit
    from .report.build import write_report

    out = Path(args.out)
    cfg = load_config(args.config, overrides={
        "build": {"per_type": {"object": args.n // 3, "attribute": args.n // 3, "relation": args.n // 3}},
        "subsets": {"eval": args.n, "dev": min(12, args.n)}, "analysis": {"bootstrap_iters": 300},
        "probes": {"caption_frac": 0.5}})
    roots = make_fixture(out / "fixture", n_images=max(60, args.n), seed=cfg["seed"])
    items = build(out / "data", cfg, roots["coco_root"], splits=["val2017", "train2017"], vg_root=roots["vg_root"])
    for name, n in cfg["subsets"].items():
        write_subset(out / "data" / "subsets" / f"{name}.txt", sample_subset(items, n, cfg["seed"]))
    items = load_items(out / "data", "eval")
    by_id = {it.gib_id: it for it in items}
    profiles = {"sim-balanced": {}, "sim-textlover": {"w_txt": 1.8, "w_img": 0.8, "attn_shift": 0.35},
                "sim-imagelover": {"w_txt": 0.6, "w_img": 1.3, "temperature": 1.0}}
    for name, prof in profiles.items():
        evaluate(SimulatedVLM(name, by_id, prof), items, cfg, roots["coco_root"], out / "runs" / name, progress=False)
    write_report(out / "report", items, cfg, [out / "runs" / n for n in profiles], image_root=roots["coco_root"])
    res = audit(out / "report", items, cfg, allow_simulated=True)
    print(json.dumps({"items": len(items), "audit_ok": res["ok"], "claims": res["n_claims"],
                      "report": str(out / "report")}, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 (no χ², −, σ)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="gib", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--config", default=None, help="YAML file merged over the defaults")
        s.set_defaults(fn=fn)
        return s

    s = p("build", cmd_build, "build items from COCO (+VG)")
    s.add_argument("--coco-root", required=True)
    s.add_argument("--vg-root", default=None)
    s.add_argument("--splits", default="val2017,train2017")
    s.add_argument("--fresh", default=None, help="ANNOTATION_JSON,IMAGE_DIR for post-cutoff images")
    s.add_argument("--out", required=True)

    s = p("demo", cmd_demo, "offline synthetic smoke run (simulated models)")
    s.add_argument("--out", default="demo_out")
    s.add_argument("--n", type=int, default=60)

    for name, fn in (("estimate", cmd_estimate), ("evaluate", cmd_evaluate), ("speedtest", cmd_speedtest)):
        s = p(name, fn, name)
        s.add_argument("--data", required=True)
        s.add_argument("--subset", default="eval")
        if name == "estimate":
            continue
        s.add_argument("--image-root", required=True)
        s.add_argument("--model", required=True, help="name=backend:model, e.g. llava16=hf:llava-hf/llava-v1.6-mistral-7b-hf")
        s.add_argument("--runs", default="runs")
        s.add_argument("--load-4bit", action="store_true")
        s.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
        s.add_argument("--no-prefix-cache", action="store_true")
        s.add_argument("--max-pixels", type=int, default=None, help="Qwen-VL image budget, e.g. 401408 (=512*28*28)")
        s.add_argument("--device", default=None)
        s.add_argument("--dtype", default="auto")
        s.add_argument("--trust-remote-code", action="store_true")
        s.add_argument("--workers", type=int, default=8,
                       help="API models: requests in flight at once (lower it for tight rate limits)")
        s.add_argument("--batch-size", type=int, default=16,
                       help="open models: queries per forward pass on a shared image (1 = one at a time)")
        s.add_argument("--limit-groups", type=int, default=None)
        if name == "speedtest":
            s.add_argument("--items", type=int, default=3)
            s.add_argument("--groups", type=int, default=4)

    s = p("report", cmd_report, "metrics, statistics, figures")
    s.add_argument("--data", required=True)
    s.add_argument("--subset", default="eval")
    s.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    s.add_argument("--image-root", default=None)
    s.add_argument("--human", default=None, help="human.json from human-analyze")
    s.add_argument("--out", required=True)
    s.add_argument("--no-figures", action="store_true")

    s = p("audit", cmd_audit, "recompute claims from the provenance files")
    s.add_argument("--data", required=True)
    s.add_argument("--report", required=True)
    s.add_argument("--allow-simulated", action="store_true")

    s = p("human-export", cmd_human_export, "offline human-study page")
    s.add_argument("--data", required=True)
    s.add_argument("--subset", default="eval")
    s.add_argument("--image-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=120)

    s = p("human-analyze", cmd_human_analyze, "score human responses")
    s.add_argument("--study", required=True)
    s.add_argument("--responses", nargs="+", required=True)
    s.add_argument("--report", default=None)
    s.add_argument("--out", default="human.json")

    s = p("taxonomy-export", cmd_taxonomy_export, "failures for blind labelling")
    s.add_argument("--data", required=True)
    s.add_argument("--report", required=True)
    s.add_argument("--image-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=200)

    s = p("taxonomy-kappa", cmd_taxonomy_kappa, "Cohen's kappa vs human labels")
    s.add_argument("--labels", required=True)
    s.add_argument("--key", required=True)

    s = p("materialize", cmd_materialize, "write degraded images")
    s.add_argument("--data", required=True)
    s.add_argument("--subset", default="dev")
    s.add_argument("--image-root", required=True)
    s.add_argument("--out", required=True)

    s = p("fetch-images", cmd_fetch_images, "download only the COCO images the items use")
    s.add_argument("--data", required=True)
    s.add_argument("--subset", default="eval")
    s.add_argument("--image-root", required=True)
    s.add_argument("--workers", type=int, default=16, help="parallel downloads")

    s = p("release", cmd_release, "dataset card + annotations for release")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)

    s = p("pack", cmd_pack, "source zip of a git commit, reproducible byte for byte")
    s.add_argument("--out", default=None, help="zip path (default: dist/<prefix>-<commit>.zip)")
    s.add_argument("--ref", default="HEAD", help="commit, branch or tag to pack")
    s.add_argument("--prefix", default="gib-200k", help="top-level folder inside the zip")

    args = ap.parse_args(argv)
    args.fn(args)
