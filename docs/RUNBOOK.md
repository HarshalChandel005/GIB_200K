# Runbook: from annotations to an audited paper

Every step is a command; every open-model number in the paper comes out of step 8.
Commands assume the project root as working directory.

## 0. Install

```bash
pip install -r requirements.txt                 # build, analysis, report, human study (CPU)
pip install -r requirements-gpu.txt             # open models (CUDA GPU; CPU works for tiny models)
pip install -r requirements-api.txt             # closed models (OpenAI / Gemini)
python -m pytest -q                             # 65 tests (1 needs a real model), ~1.5 min on CPU
python -m gib demo --out demo_out               # offline end-to-end run with SIMULATED models
```

## 1. Annotations

* COCO 2017 annotations: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  → unzip to `data/coco/annotations/` (`instances_*.json`, `captions_*.json`).
* Visual Genome (for colour and spatial-relation questions):
  `https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/{image_data,attributes,relationships}.json.zip`
  → unzip to `data/vg/`. (`image_data.json` holds the `coco_id` link: 51,498 of its
  108,077 images are COCO images.)

About 415 MB to download, ~1.8 GB unzipped (the COCO person-keypoint files are not needed).

## 2. Build the items (CPU, ~3 minutes, ~7 GB RAM)

```bash
python -m gib build --coco-root data/coco --vg-root data/vg --splits val2017,train2017 --out data/gib200k
```

Writes `items.jsonl` (200,000 items: 66,667 object, 66,667 attribute and 66,666
relation questions), `build_report.json`, and the subsets `subsets/eval.txt` (all
200,000 items: every model answers the full benchmark) and `subsets/dev.txt` (30).
An image can carry up to four items (of different questions); the statistics
resample images, not items, so items that share an image are never treated as
independent. Colour and spatial questions exist only on the ~40k COCO images that
Visual Genome annotates and that have such candidates, so the types compete for those
images: the type with the fewest candidate images per needed item is filled first,
and images are filled in rounds (one item each, then two, ... four). With three per
image only ~186k items are possible; `build_report.json` warns if a type cannot
reach its quota.

Relation answers are capped: `build.max_answer_share` keeps any one answer to 33% of
the relation items. Uncapped, 69% were "on", so a model could score 69% on them without
looking at the image. To make room, relation questions are filled first and left/right
questions come from every clearly separated pair of objects, not only the two largest.
`build_report.json` gives each type's `answer_share`, which is also the score of always
giving the top answer.

The real build (COCO 2017 + Visual Genome 1.2, seed 0) gives exactly 200,000 items
on 90,708 images (36,167 images carry one item, 16,651 two, 21,029 three, 16,861
four), 81,235 counterfactual flip pairs, and 191,140 train2017 / 8,860 val2017 items
(val2017 has only 5,000 images; the contamination analysis compares the two splits
with a CI either way). `items.jsonl` is 270 MB. Relation answers: on 33.0%, left 17.9%,
right 17.7%, behind 8.4%, next to 6.5%, above 5.4%, under 5.0%, in front of 2.5%, inside
2.5%, below 1.0%. The 23,782 left/right items have two options in multiple choice (a guess
scores 50%), all others five (20%), so compare multiple-choice scores within a question type.
Optional: `--fresh ANNOTATIONS.json,IMAGE_DIR` adds a COCO-format set of photos
taken after the models' training cut-off (strongest contamination control).

## 3. Images: download only what is used

```bash
python -m gib fetch-images --data data/gib200k --subset eval --image-root data/coco
```

(Only the images the items use: 90,708 of the 123,287 COCO 2017 images, 14.8 GB
(86,987 train2017 + 3,721 val2017, 163 KB on average), fetched 16 at a time (`--workers`);
each image is downloaded once even when several items share it, and re-running the command
fetches only what is still missing. `--subset dev` fetches the 30 dev items' images for a
first try. One COCO file, `train2017/000000320612.jpg`, is a PNG under a `.jpg` name, as COCO
serves it; the pipeline opens images by content, so it needs no special handling.)

## 4. Plan the compute

```bash
python -m gib estimate --data data/gib200k --subset eval
```

Prints queries per item and the number of image encodings. With the default config,
an item expands into ~108 queries (29 free-form + 29 multiple-choice + 18 A/B + 28 POPE
+ 3.5 captions on average) over 14 distinct images: for the real 200,000 items,
21,468,178 queries and 2,803,125 image encodings per model. For open models only the
image encodings are long forward passes; every query is a short text suffix on a
cached prefix, batched (see `speedtest`). Closed models make one API call per query,
so budget for ~21.5 million calls per model. `configs/lite.yaml` keeps all analyses
with fewer levels (and 600 items) for a single free GPU or a small API budget.

Disk, memory and time at 200,000 items (measured end to end at 20,000 items with two
simulated models -- 2.15M queries each -- and scaled):

| | 20,000 items (measured) | 200,000 items (scaled) |
|---|---|---|
| `evaluate` memory | 0.26 GB, flat | ~0.3 GB (items are expanded one at a time) |
| `predictions.jsonl` per model | 0.92 GB | ~9–12 GB (more layers of attention = larger) |
| resuming a run | 44 s to re-read what is done | ~8 min |
| `gib report` peak RAM | 1.6 GB | **~8 GB** (one model at a time) |
| `gib report` time per model | ~4.5 min | ~45 min |
| `gib audit` time per model | ~1.2 min | ~12 min |
| `provenance/<model>.csv.gz` | 70 MB | ~0.7 GB |

## 5. Open models (GPU)

```bash
# check the speed-up and that cached and uncached outputs agree, on your hardware
python -m gib speedtest --data data/gib200k --subset dev --image-root data/coco \
    --model llava16=hf:llava-hf/llava-v1.6-mistral-7b-hf --load-4bit

# the run itself (resumable: re-run the same command after an interruption)
python -m gib evaluate --data data/gib200k --image-root data/coco --runs runs \
    --model llava16=hf:llava-hf/llava-v1.6-mistral-7b-hf --load-4bit
python -m gib evaluate --data data/gib200k --image-root data/coco --runs runs \
    --model qwen2vl=hf:Qwen/Qwen2-VL-7B-Instruct --load-4bit --max-pixels 401408
# a size ladder of one family, for any "bigger models are more/less robust" claim
for s in 1B 2B 8B; do
  python -m gib evaluate --data data/gib200k --image-root data/coco --runs runs \
      --model internvl3-$s=hf:OpenGVLab/InternVL3-$s-hf --load-4bit
done
```

Notes: `--load-4bit` makes 7–8B models fit a 16 GB GPU (T4); run one model in fp16 on
the dev subset as well and report the 4-bit vs fp16 difference. Up to `--batch-size`
questions on one image (default 16) share each forward pass; the first such batch of a
run is also answered one by one and compared, and batching switches itself off if the
two differ by more than rounding. `speedtest` reports its gain on your GPU as `batch_speedup`; lower
`--batch-size` if memory is tight. Every run writes `runs/<model>/meta.json` with the
model commit, dtype, quantisation, library versions, and the result of the prefix-cache,
tokenisation and batching self-checks. It is written when the run starts, again after
the first image (when the self-checks have run), every minute while it runs, and at the
end, so a run that crashes or is stopped still leaves a record: `status` says
`running`, `finished` or `interrupted` (with the error), and `history` has one entry
per attempt of the same command.

**Self-checks and rounding.** In low precision, the same question reached by two routes
gives slightly different numbers, so a check that demands equality fails by chance and
switches off a shortcut that works. Each check therefore has limits per precision
(`VERIFY_TOL` and `BATCH_TOL` in `gib/models/hf_vlm.py`). They were set from real photos
with `llava-interleave-qwen-0.5b`, against bugs injected into the adapter's own code. The
table gives the largest rounding seen, and the first two rows are log-probability
differences, as the median of 5 questions (37, 6 and 4 photos in the three rounding
columns; the batching rows use 97, 5 and 16).

| Check | Rounding, bf16 GPU | Rounding, bf16 CPU | Rounding, fp16 GPU | Limit bf16 / fp16 / fp32 | Injected bugs, bf16 GPU |
|---|---|---|---|---|---|
| shared prefix vs a prefill per query | 0 | 0.15 | 0.023 | 0.2 / 0.05 / 0.001 | position +1: 0.25 or more; +8 and −700: 16 or more |
| shared prefix vs one plain pass | 0.19 | 0.18 | 0.016 | 0.4 / 0.15 / 0.001 | |
| batched vs one by one, option probabilities | 0.058 | 0.031 | 0.006 | 0.1 / 0.02 / 0.001 | |
| batched vs one by one, attention | 0.0075 | 0.0056 | 0.001 | 0.009 / 0.005 / 0.001 | position +1: 0.0073 to 0.018; mask leak: 0.56 or more |
| batched vs one by one, free-form answers changed in a batch | 1 | 0 | 0 | 2 / 1 / 0 | mask leak: 7 or more |

No correct run failed a check. Prefix bugs were caught on every photo, and so was the mask
leak. The weakest case is a batched position off by one in bf16, which is about the size of
rounding: it was caught on 85 of 97 GPU photos, 5 of 5 on the CPU and 16 of 16 in fp16, and
fp32's limits are far below it. Limits this close to rounding can still go wrong on other
models and hardware, since only one small model was measured and no GPU with native bf16
was available (the CPU is the nearest stand-in). A failed check costs speed, not
correctness: the run uses the slower route and `meta.json` records why, with `split_diff`
showing how much that machine rounds. Rounding also changes a near-tied free-form answer
in about one bf16 batch in six, so use fp32 when two runs must match exactly.

## 6. Closed models (API)

```bash
export OPENAI_API_KEY=...   GEMINI_API_KEY=...
python -m gib evaluate --data data/gib200k --image-root data/coco --runs runs --model gpt4o=openai:gpt-4o
python -m gib evaluate --data data/gib200k --image-root data/coco --runs runs --model gemini=gemini:gemini-2.5-flash
```

Same items, same prompts, same images as the open models. Responses are cached in
`runs/_api_cache/*.sqlite`, so interrupted runs cost nothing twice. Calls run in
parallel: at most `--workers` (default 8) requests are in flight per model, and
identical requests in flight together are sent once, so a parallel run makes the
same number of calls as a sequential one. Lower `--workers` if you hit rate limits
(e.g. `--workers 1` on a free tier); failed calls are retried with backoff, and
anything that still fails is retried on the next run. Record the exact
model snapshot names; retired models (e.g. Gemini 1.5) cannot be re-run, so cite the
versions you actually used. Use `gib estimate` to size the budget: one API call per query.

## 7. Human baseline and taxonomy validation

```bash
python -m gib human-export --data data/gib200k --image-root data/coco --out human_study --n 120
# zip human_study/ and send it to 5–10 participants; each opens index.html offline,
# answers, and returns the downloaded responses_<code>.json into human_study/responses/
python -m gib human-analyze --study human_study --responses "human_study/responses/*.json" \
    --report reports/main --out human.json

python -m gib taxonomy-export --data data/gib200k --report reports/main --image-root data/coco --out tax --n 200
# two annotators fill tax/to_label.csv (human_label) blind to the automatic label
python -m gib taxonomy-kappa --labels tax/to_label.csv --key tax/key.csv
```

Get ethics approval / informed consent as your institution requires before step 7.

## 8. Report and audit

```bash
python -m gib report --data data/gib200k --image-root data/coco --runs "runs/*" --human human.json --out reports/main
python -m gib audit --data data/gib200k --report reports/main
```

`reports/main/` then holds `provenance.csv`, `report.json`, `claims.json`,
`RESULTS.md`, `macros.tex`, `figures/` (+ `CAPTIONS.md`) and `gallery.html`.
Use only `\gib…` macros from `macros.tex` for numbers in the paper. The audit fails
if any claim differs from what `provenance.csv` produces, or if a simulated model
is in the report.

## 9. Release

```bash
python -m gib release --data data/gib200k --out release/GIB-200K
python -m gib pack                          # source zip of the last commit -> dist/
```

Upload `release/GIB-200K` to Hugging Face Datasets and the repository to GitHub.

`gib pack` zips exactly the files of one commit (`--ref`, default HEAD) under a
`gib-200k/` folder, read from git rather than the working folder: uncommitted
changes are left out (it says so), and the same commit always gives the same zip,
byte for byte. It prints `zip_sha256` and `content_sha256` (paths and contents only,
identical on every machine), so anyone can check that a zip matches a commit. Use
`--out` to choose the path (default `dist/gib-200k-<commit>.zip`).
