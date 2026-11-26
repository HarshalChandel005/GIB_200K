"""Open-weight VLMs through Hugging Face transformers -- the efficient path.

What makes it fast (vs. the previous adapter, which ran ~6 full forward passes
per query and materialised every layer's full attention matrix):

1. **Shared image prefix.** All queries on one image share the tokens up to and
   including the image tokens. That prefix (hundreds to thousands of tokens) is
   encoded once per image; each query then runs only its own text suffix on
   the KV cache, which is cropped back afterwards (no copies).
2. **One distribution, many probes.** Multiple-choice, A/B and yes/no probes are
   read from the next-token distribution at the end of the suffix -- no
   generation, no candidate re-scoring.
3. **Attention at one position.** The last suffix token is run on its own with
   attention output; only a (heads x keys) row per layer is produced and is
   immediately reduced to image / post-image-text mass.
4. **logits_to_keep=1**, so a 150k-vocab model never materialises logits for
   thousands of prefix positions.
5. **Fast tokenisation.** Suffix token ids come from the plain tokenizer instead
   of re-running the image processor for every query.
6. **Batched suffixes.** Up to ``batch_size`` queries on one image are run in the
   same forward passes: their suffixes are packed into one sequence on the
   shared prefix, with an attention mask that lets each token see only the
   prefix and its own query, and the positions each token has in its own
   sequence. The cache grows by the packed tokens only (no per-query copies of
   the prefix), and free-form answers are decoded together, one token per open
   query per pass.

Shortcuts 1, 5 and 6 are **verified at run start** against the plain computation
(a prefill per query and one full forward pass / the processor's own tokenisation /
the same queries answered one by one). If a model family behaves differently the
adapter falls back automatically, and the outcome is written to the run metadata.
"""

from __future__ import annotations

import inspect
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ..schema import Prediction, Probe, Query
from .prompts import LETTERS, build_prompt

BATCH_TOKENS = 2048  # packed suffix tokens per pass, so long-document queries do not blow up activations

# Self-check limits. Float rounding must not pass for a bug: splitting a pass moves top log-probs by up
# to 0.25 in bf16, 0.03 in fp16 and 1e-5 in fp32, and can flip a near-tied free-form answer. Measured on
# real images with a 0.5B model against simulated bugs (docs/RUNBOOK.md, step 5); the bf16 limits hold on
# a GPU and on the CPU, which round differently. 4-bit models use the bf16 row.
VERIFY_QUERIES = 5
VERIFY_TOL = {"float32": (1e-3, 1e-3), "float16": (0.05, 0.15), "bfloat16": (0.2, 0.4)}  # prefix split, plain pass
BATCH_TOL = {"float32": (1e-3, 1e-3, 0), "float16": (0.02, 0.005, 1), "bfloat16": (0.1, 0.009, 2)}  # probs, attention, texts


class HFVLM:
    reproducible = True

    def __init__(self, name: str, model_id: str, *, device: Optional[str] = None, dtype: str = "auto",
                 load_4bit: bool = False, attn_implementation: str = "sdpa",
                 max_pixels: Optional[int] = None, prefix_cache: bool = True,
                 capture_attention: bool = True, qa_max_new_tokens: int = 10,
                 caption_max_new_tokens: int = 48, revision: Optional[str] = None,
                 trust_remote_code: bool = False, batch_size: int = 16):
        import torch
        import transformers
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.torch = torch
        self.name = name
        self.model_id = model_id
        self.qa_max_new_tokens = qa_max_new_tokens
        self.caption_max_new_tokens = caption_max_new_tokens
        self.capture_attention = capture_attention
        self.use_prefix_cache = prefix_cache
        self.attn_implementation = attn_implementation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            if self.device.startswith("cuda"):
                tdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                tdtype = torch.float32
        else:
            tdtype = getattr(torch, dtype)
        self.dtype = tdtype
        self._tol_key = "bfloat16" if load_4bit else str(tdtype).replace("torch.", "")

        proc_kw: Dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code}
        if max_pixels:
            proc_kw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kw)
        self.tok = getattr(self.processor, "tokenizer", self.processor)

        model_kw: Dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code,
                                    "attn_implementation": attn_implementation}
        version = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
        model_kw["dtype" if version >= (4, 56) else "torch_dtype"] = tdtype
        if load_4bit:
            from transformers import BitsAndBytesConfig

            model_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
            model_kw["device_map"] = "auto"
        elif self.device.startswith("cuda"):
            model_kw["device_map"] = {"": self.device}
        self.model = AutoVLM.from_pretrained(model_id, **model_kw)
        if not load_4bit and not self.device.startswith("cuda"):
            self.model.to(self.device)
        self.model.eval()
        self.dev = next(self.model.parameters()).device

        params = inspect.signature(self.model.forward).parameters
        self._accepts = set(params)
        self._logits_kw = ("logits_to_keep" if "logits_to_keep" in params
                           else "num_logits_to_keep" if "num_logits_to_keep" in params else None)
        self.image_token_id = self._find_image_token_id()
        self.eos_ids = self._eos_ids()
        self._letter_ids = {L: self._single_token_ids([L]) for L in LETTERS}
        self._yes_ids = self._single_token_ids(["yes", "Yes"])
        self._no_ids = self._single_token_ids(["no", "No"])
        self._verified: Dict[str, Any] = {}
        self._fast_tokenize: Optional[bool] = None
        self.batch_size = max(1, int(batch_size))
        self._batched: Optional[bool] = None  # None until the packed path is checked against one-by-one
        try:
            self._mask_dtype = self.model.get_input_embeddings().weight.dtype
        except Exception:
            self._mask_dtype = tdtype
        self._pos_offset = 0
        self.cache = None

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _find_image_token_id(self) -> Optional[int]:
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
        v = getattr(self.processor, "image_token_id", None)
        if isinstance(v, int):
            return v
        tok = getattr(self.processor, "image_token", None)
        if tok:
            return self.tok.convert_tokens_to_ids(tok)
        return None

    def _eos_ids(self) -> set:
        ids = set()
        for src in (getattr(self.model, "generation_config", None), self.tok):
            v = getattr(src, "eos_token_id", None)
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
        unk = getattr(self.tok, "unk_token_id", None)
        for t in ("<|im_end|>", "<|endoftext|>", "<end_of_utterance>", "<|eot_id|>", "</s>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.add(tid)
        return ids

    def _single_token_ids(self, words: Sequence[str]) -> List[int]:
        ids = set()
        for w in words:
            for s in (w, " " + w):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
                elif len(t) == 2 and self.tok.decode(t[:1]).strip() == "":
                    ids.add(t[1])  # tokenizers that emit a separate space token
        if not ids:
            ids.add(self.tok.encode(words[0], add_special_tokens=False)[0])
        return sorted(ids)

    def meta(self) -> Dict[str, Any]:
        import transformers

        q = getattr(self.model.config, "quantization_config", None)
        return {
            "backend": "hf", "model_id": self.model_id,
            "commit": getattr(self.model.config, "_commit_hash", None),
            "dtype": str(self.dtype).replace("torch.", ""), "device": str(self.dev),
            "quantization": (q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", None))
            if q else None,
            "attn_implementation": self.attn_implementation,
            "transformers": transformers.__version__, "torch": self.torch.__version__,
            "image_token_id": self.image_token_id, "batch_size": self.batch_size, "verification": self._verified,
        }

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def _chat(self, text: str, has_image: bool) -> str:
        content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": text}]
        try:
            return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                      add_generation_prompt=True, tokenize=False)
        except Exception:
            img = (getattr(self.processor, "image_token", "<image>") + "\n") if has_image else ""
            return f"USER: {img}{text}\nASSISTANT:"

    def _encode_full(self, prompt: str, image: Optional[Image.Image]) -> Dict[str, Any]:
        if image is None:
            return dict(self.processor(text=[prompt], return_tensors="pt"))
        try:
            return dict(self.processor(text=[prompt], images=[image], return_tensors="pt"))
        except Exception:
            return dict(self.processor(text=[prompt], images=[[image]], return_tensors="pt"))

    def _fast_ids(self, prompt: str, prefix_ids: List[int]) -> Optional[List[int]]:
        """Prefix (with expanded image tokens) + plain-tokenizer suffix."""
        if self.image_token_id is None or self.image_token_id not in prefix_ids:
            return None
        raw = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if self.image_token_id not in raw:
            return None
        cut = len(raw) - 1 - raw[::-1].index(self.image_token_id)
        last_img = len(prefix_ids) - 1 - prefix_ids[::-1].index(self.image_token_id)
        return prefix_ids[:last_img + 1] + raw[cut + 1:]

    # ------------------------------------------------------------------ #
    # forward primitives (all on self.cache)
    # ------------------------------------------------------------------ #
    def _fwd(self, ids: Sequence[int], start: int, extra: Optional[Dict[str, Any]] = None,
             attentions: bool = False, allow=None, pos: Optional[Sequence[int]] = None,
             keep: Optional[List[int]] = None):
        """Run ``ids`` on the cache (``start`` tokens long). For packed queries, ``allow`` (new x all
        tokens, bool) says what each new token may attend to, ``pos`` gives its position in its own
        sequence, and ``keep`` the input positions whose logits are needed."""
        torch = self.torch
        n = len(ids)
        if allow is None:
            mask = torch.ones((1, start + n), dtype=torch.long, device=self.dev)
        else:  # additive float mask: eager attention adds it, SDPA accepts it
            mask = torch.zeros(allow.shape, dtype=self._mask_dtype, device=self.dev)
            mask = mask.masked_fill_(~allow.to(self.dev), torch.finfo(self._mask_dtype).min)[None, None]
        kw: Dict[str, Any] = {
            "input_ids": torch.tensor([list(ids)], device=self.dev), "attention_mask": mask,
            "past_key_values": self.cache, "use_cache": True,
        }
        if pos is not None:
            kw["position_ids"] = torch.tensor([list(pos)], device=self.dev) + self._pos_offset
        if "cache_position" in self._accepts:
            kw["cache_position"] = torch.arange(start, start + n, device=self.dev)
        if self._logits_kw:  # only the newer keyword takes a list of positions; otherwise keep all
            kw[self._logits_kw] = (1 if keep is None else torch.tensor(keep, device=self.dev)
                                   if self._logits_kw == "logits_to_keep" else 0)
        for k, v in (extra or {}).items():
            kw[k] = v.to(self.dev) if hasattr(v, "to") else v
        switched = False
        if attentions:
            kw["output_attentions"] = True
            # sdpa cannot return weights; switch to eager for this single-token step only
            if self.attn_implementation != "eager" and hasattr(self.model, "set_attn_implementation"):
                try:
                    self.model.set_attn_implementation("eager")
                    switched = True
                except Exception:
                    switched = False
        try:
            with torch.inference_mode():
                out = self.model(**kw)
        finally:
            if switched:
                self.model.set_attn_implementation(self.attn_implementation)
        self.cache = out.past_key_values
        return out

    @staticmethod
    def _slice_extra(extra: Dict[str, Any], n_full: int, upto: int) -> Dict[str, Any]:
        """Per-token tensors (e.g. token_type_ids) must match the prefilled span."""
        out = {}
        for k, v in extra.items():
            if getattr(v, "ndim", 0) == 2 and v.shape[1] == n_full:
                out[k] = v[:, :upto]
            else:
                out[k] = v
        return out

    def _crop(self, length: int) -> bool:
        """Drop everything after the first ``length`` cached tokens."""
        crop = getattr(self.cache, "crop", None)
        if crop is None:
            return False
        extra = self.cache.get_seq_length() - length
        if extra > 0:
            crop(-extra)  # negative = remove that many tokens (accepted by old and new transformers)
        return True

    def _position_offset(self) -> int:
        """M-RoPE models (Qwen2-VL) shift text positions after the image by a stored delta; others 0."""
        rd = getattr(getattr(self.model, "base_model", None), "rope_deltas", None)
        try:
            return int(rd.reshape(-1)[0]) if rd is not None and rd.numel() else 0
        except Exception:
            return 0

    @staticmethod
    def _attn_summary(attns, img_idx: List[int], post_start: int,
                      row_idx: int = -1) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if not attns or attns[0] is None:
            return None, None
        img, post = [], []
        for a in attns:
            row = a[0, :, row_idx, :].float().mean(0)
            img.append(round(float(row[img_idx].sum()) if img_idx else 0.0, 6))
            post.append(round(float(row[post_start:].sum()), 6))
        return img, post

    # ------------------------------------------------------------------ #
    # readouts
    # ------------------------------------------------------------------ #
    def _probs(self, logits, groups: List[List[int]]) -> Tuple[List[float], float]:
        p = self.torch.softmax(logits.float(), dim=-1)
        mass = [float(p[g].sum()) for g in groups]
        tot = sum(mass)
        return ([m / tot for m in mass] if tot > 0 else [1.0 / len(mass)] * len(mass)), tot

    def _read_probs(self, q: Query, logits, pred: Prediction) -> None:
        """MC / A-B / yes-no probes: the option distribution at the end of the prompt."""
        if Probe(q.probe) is Probe.POPE:
            groups = [self._yes_ids, self._no_ids]
        else:
            groups = [self._letter_ids[LETTERS[i]] for i in range(len(q.options))]
        pred.option_probs, mass = self._probs(logits, groups)
        pred.extra.update(format_mass=round(mass, 4), from_logprobs=True)

    def _max_new(self, probe: Probe) -> int:
        return self.caption_max_new_tokens if probe is Probe.CAPTION else self.qa_max_new_tokens

    def _stops(self, toks: List[int]) -> bool:
        """After appending a token: a free-form answer ends at its first line break."""
        return len(toks) > 1 and "\n" in self.tok.decode(toks[-1:], skip_special_tokens=True)

    def _detok(self, toks: List[int]) -> str:
        return self.tok.decode(toks, skip_special_tokens=True).strip().split("\n")[0].strip()

    def _greedy(self, logits, start: int, max_new: int) -> str:
        toks: List[int] = []
        cur = start
        for _ in range(max_new):
            nxt = int(self.torch.argmax(logits))
            if nxt in self.eos_ids:
                break
            toks.append(nxt)
            if self._stops(toks) or len(toks) >= max_new:  # no pass for logits that would not be used
                break
            logits = self._fwd([nxt], cur).logits[0, -1]
            cur += 1
        return self._detok(toks)

    # ------------------------------------------------------------------ #
    # verification of the prefix-cache shortcut
    # ------------------------------------------------------------------ #
    def _verify(self, queries: List[Tuple[List[int], Dict[str, Any]]], P: int) -> None:
        """Check the shared prefix on a few (ids, encoding) queries of the first image.

        The choice is between sharing the prefix and a prefill per query, and both end on the same
        single-token step, so those two are compared directly (``split``); that step is then checked
        against one plain forward pass (``full``). The verdict uses medians: rounding hits single
        queries, a bug hits them all.
        """
        torch = self.torch

        def last_logprobs(ids: List[int], enc: Dict[str, Any], *cuts: int):
            extra = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}
            self.cache = None
            for a, b in zip((0, *cuts), (*cuts, len(ids))):
                out = self._fwd(ids[a:b], a, self._slice_extra(extra, len(ids), b) if a == 0 else None)
            self.cache = None
            return torch.log_softmax(out.logits[0, -1].float(), -1)

        def diff(ref, other) -> float:
            top = torch.topk(ref, 50).indices
            return float((ref[top] - other[top]).abs().max())

        split, full, miss = [], [], []
        for ids, enc in queries:
            plain = last_logprobs(ids, enc)                                 # one plain forward pass
            prefill = last_logprobs(ids, enc, len(ids) - 1)                 # prefill per query + last token
            cached = last_logprobs(ids, enc, *sorted({P, len(ids) - 1}))    # shared prefix + suffix + last token
            split.append(diff(prefill, cached))
            full.append(diff(plain, cached))
            miss.append(float(plain.max() - plain[cached.argmax()]))        # > 0: not the plain pass's top token
        split_tol, tol = VERIFY_TOL.get(self._tol_key, VERIFY_TOL["bfloat16"])
        ds, df = statistics.median(split), statistics.median(full)
        ok = ds < split_tol and df < tol and max(miss) < tol
        self._verified["prefix_cache"] = {"ok": ok, "n_queries": len(queries), "split_diff": round(ds, 6),
                                          "split_tol": split_tol, "median_logprob_diff": round(df, 6),
                                          "max_logprob_diff": round(max(full), 6), "tol": tol}
        if not ok:
            self.use_prefix_cache = False

    # ------------------------------------------------------------------ #
    # the contract method
    # ------------------------------------------------------------------ #
    def run_group(self, image: Optional[Image.Image], queries: List[Query]) -> List[Prediction]:
        t_group = time.perf_counter()
        prompts = [self._chat(build_prompt(q), image is not None) for q in queries]
        enc0 = self._encode_full(prompts[0], image)
        ids0: List[int] = enc0["input_ids"][0].tolist()
        extra = {k: v for k, v in enc0.items() if k not in ("input_ids", "attention_mask")}
        per_token = any(getattr(v, "ndim", 0) == 2 and v.shape[1] == len(ids0) for v in extra.values())

        if self._fast_tokenize is None:  # verify once: plain tokenizer == processor
            self._fast_tokenize = (not per_token) and self._fast_ids(prompts[0], ids0) == ids0
            self._verified["fast_tokenize"] = self._fast_tokenize
        all_ids: List[List[int]] = [ids0]
        encs: List[Optional[Dict[str, Any]]] = [enc0]
        for p in prompts[1:]:
            fast = self._fast_ids(p, ids0) if self._fast_tokenize else None
            if fast is not None:
                all_ids.append(fast)
                encs.append(None)
            else:
                e = self._encode_full(p, image)
                all_ids.append(e["input_ids"][0].tolist())
                encs.append(e)

        # longest common prefix; sharing it requires every image token inside it
        P = min(len(x) for x in all_ids) - 1
        for x in all_ids[1:]:
            j = 0
            while j < P and x[j] == ids0[j]:
                j += 1
            P = j
        img_idx0 = [i for i, t in enumerate(ids0) if t == self.image_token_id] if self.image_token_id is not None else []
        last_img0 = img_idx0[-1] if img_idx0 else -1
        shared = self.use_prefix_cache and P > last_img0 and P > 0
        if shared and "prefix_cache" not in self._verified:
            picks = sorted({i * (len(queries) - 1) // (VERIFY_QUERIES - 1) for i in range(VERIFY_QUERIES)})
            self._verify([(all_ids[i], encs[i] or enc0) for i in picks], P)
            shared = self.use_prefix_cache
        if shared:
            self.cache = None
            self._fwd(ids0[:P], 0, self._slice_extra(extra, len(ids0), P))
            self._pos_offset = self._position_offset()
            if not self._crop(P):
                shared = False
        t_prefix = time.perf_counter() - t_group

        def one(i: int) -> Prediction:
            return self._answer_one(queries[i], all_ids[i], encs[i], prompts[i], image, shared, P, img_idx0, last_img0)

        def packed(batch: List[int]) -> List[Prediction]:
            return self._answer_packed([queries[i] for i in batch], [all_ids[i] for i in batch], P, img_idx0, last_img0)

        preds: List[Optional[Prediction]] = [None] * len(queries)
        batches = (self._batches(all_ids, P) if shared and self.batch_size > 1 and self._batched is not False
                   else [[i] for i in range(len(queries))])
        for batch in batches:
            t0 = time.perf_counter()
            if len(batch) == 1 or self._batched is False:
                out = [one(i) for i in batch]
            elif self._batched is None:  # first multi-query batch of the run: check against one by one
                out = [one(i) for i in batch]
                try:
                    self._check_batched(out, packed(batch))
                except Exception as e:  # e.g. a model that rejects custom masks or position ids
                    self._crop(P)
                    self._batched = False
                    self._verified["batched_suffixes"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            else:
                out = packed(batch)
            dt = (time.perf_counter() - t0) / len(batch)
            for i, pred in zip(batch, out):
                pred.latency_ms = round((dt + t_prefix / len(queries)) * 1000, 2)
                preds[i] = pred
        self.cache = None
        return preds  # type: ignore[return-value]

    def _answer_one(self, q: Query, ids: List[int], enc: Optional[Dict[str, Any]], prompt: str,
                    image: Optional[Image.Image], shared: bool, P: int, img_idx0: List[int],
                    last_img0: int) -> Prediction:
        """One query on its own: its suffix on the shared prefix (cropped back after), or a full prefill."""
        probe = Probe(q.probe)
        if shared:
            start, img_idx, last_img = P, img_idx0, last_img0
        else:
            e = enc if enc is not None else self._encode_full(prompt, image)
            ex = {k: v for k, v in e.items() if k not in ("input_ids", "attention_mask")}
            self.cache = None
            self._fwd(ids[:-1], 0, self._slice_extra(ex, len(ids), len(ids) - 1))
            start = len(ids) - 1
            img_idx = [i for i, t in enumerate(ids) if t == self.image_token_id] if self.image_token_id is not None else []
            last_img = img_idx[-1] if img_idx else -1
        if len(ids) - 1 > start:
            self._fwd(ids[start:-1], start)
        want_attn = self.capture_attention and probe in (Probe.QA, Probe.AB)
        out = self._fwd(ids[-1:], len(ids) - 1, attentions=want_attn)
        logits = out.logits[0, -1]
        a_img, a_post = (self._attn_summary(getattr(out, "attentions", None), img_idx, last_img + 1)
                         if want_attn else (None, None))
        pred = Prediction(qid=q.qid, model=self.name, attn_img=a_img, attn_post=a_post,
                          n_img_tokens=len(img_idx), n_tokens=len(ids))
        if probe in (Probe.QA, Probe.CAPTION):
            pred.text = self._greedy(logits, len(ids), self._max_new(probe))
        else:
            self._read_probs(q, logits, pred)
        if shared:
            self._crop(P)
        return pred

    def _batches(self, all_ids: List[List[int]], P: int) -> List[List[int]]:
        """Consecutive queries: at most ``batch_size`` and about BATCH_TOKENS suffix tokens per batch."""
        out: List[List[int]] = []
        cur: List[int] = []
        n_tok = 0
        for i, ids in enumerate(all_ids):
            n = len(ids) - P
            if cur and (len(cur) >= self.batch_size or n_tok + n > BATCH_TOKENS):
                out.append(cur)
                cur, n_tok = [], 0
            cur.append(i)
            n_tok += n
        if cur:
            out.append(cur)
        return out

    def _answer_packed(self, queries: List[Query], all_ids: List[List[int]], P: int, img_idx: List[int],
                       last_img: int) -> List[Prediction]:
        """Several queries at once on the shared prefix (the cache holds the prefix only, P tokens).

        Their tokens are packed into one sequence; the mask lets each token attend to the prefix
        and the earlier tokens of its own query only, and each keeps the position it has in its own
        sequence, so every query is computed as if it ran alone. The passes mirror ``_answer_one``:
        suffixes without their last token, all last tokens (with attention capture), then greedy
        decoding of the free-form probes with one new token per open query per pass.
        """
        torch = self.torch
        own: List[List[int]] = [[] for _ in queries]  # cache slots holding each query's own tokens
        total = P

        def step(segs: List[Tuple[int, List[int]]], attentions: bool = False):
            nonlocal total
            n = sum(len(t) for _, t in segs)
            allow = torch.zeros((n, total + n), dtype=torch.bool)
            allow[:, :P] = True
            ids: List[int] = []
            pos: List[int] = []
            last: List[int] = []
            r = 0
            for i, toks in segs:
                k = len(toks)
                if own[i]:
                    allow[r:r + k, own[i]] = True
                allow[r:r + k, total + r:total + r + k] = torch.ones((k, k), dtype=torch.bool).tril()
                ids += toks
                pos += range(P + len(own[i]), P + len(own[i]) + k)
                own[i] += range(total + r, total + r + k)
                r += k
                last.append(r - 1)
            out = self._fwd(ids, total, attentions=attentions, allow=allow, pos=pos, keep=last)
            total += n
            logits = out.logits[0]
            if logits.shape[0] != len(last):  # the model returned every position
                logits = logits[last]
            return out, logits

        body = [(i, ids[P:-1]) for i, ids in enumerate(all_ids) if len(ids) - 1 > P]
        if body:
            step(body)
        probes = [Probe(q.probe) for q in queries]
        want = [self.capture_attention and p in (Probe.QA, Probe.AB) for p in probes]
        out, logits = step([(i, ids[-1:]) for i, ids in enumerate(all_ids)], attentions=any(want))
        attns = getattr(out, "attentions", None)
        preds = []
        for i, (q, ids, probe) in enumerate(zip(queries, all_ids, probes)):
            a_img, a_post = self._attn_summary(attns, img_idx, last_img + 1, i) if want[i] else (None, None)
            pred = Prediction(qid=q.qid, model=self.name, attn_img=a_img, attn_post=a_post,
                              n_img_tokens=len(img_idx), n_tokens=len(ids))
            if probe not in (Probe.QA, Probe.CAPTION):
                self._read_probs(q, logits[i], pred)
            preds.append(pred)

        # greedy decoding, same stopping rules as _greedy
        cur = {i: logits[i] for i, p in enumerate(probes) if p in (Probe.QA, Probe.CAPTION)}
        toks: Dict[int, List[int]] = {i: [] for i in cur}
        while cur:
            segs = []
            for i, lg in cur.items():
                nxt = int(torch.argmax(lg))
                if nxt in self.eos_ids:
                    continue
                toks[i].append(nxt)
                if not (self._stops(toks[i]) or len(toks[i]) >= self._max_new(probes[i])):
                    segs.append((i, [nxt]))
            if not segs:
                break
            _, logits = step(segs)
            cur = {i: logits[r] for r, (i, _) in enumerate(segs)}
        for i, t in toks.items():
            preds[i].text = self._detok(t)
        self._crop(P)
        return preds

    def _check_batched(self, ref: List[Prediction], got: List[Prediction]) -> None:
        """Self-check of shortcut 6: packed answers must equal the one-by-one answers."""
        tol, attn_tol, text_tol = BATCH_TOL.get(self._tol_key, BATCH_TOL["bfloat16"])
        dp = max((max(abs(a - b) for a, b in zip(r.option_probs, g.option_probs))
                  for r, g in zip(ref, got) if r.option_probs and g.option_probs), default=0.0)
        da = max((max(abs(a - b) for a, b in zip(r.attn_img + r.attn_post, g.attn_img + g.attn_post))
                  for r, g in zip(ref, got) if r.attn_img and g.attn_img), default=0.0)
        text_diffs = sum(r.text != g.text for r, g in zip(ref, got))  # rounding can flip a near tie, a bug many
        same_attn = all(bool(r.attn_img) == bool(g.attn_img) for r, g in zip(ref, got))
        ok = dp < tol and da < attn_tol and text_diffs <= text_tol and same_attn
        self._batched = ok
        self._verified["batched_suffixes"] = {"ok": ok, "n_queries": len(ref), "max_prob_diff": round(dp, 6),
                                              "max_attn_diff": round(da, 6), "text_diffs": text_diffs,
                                              "tol": tol, "attn_tol": attn_tol, "text_tol": text_tol}
