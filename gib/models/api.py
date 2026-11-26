"""Closed-API reference models (OpenAI, Gemini).

They receive exactly the same user text and images as the open models
(``prompts.build_prompt``), and are evaluated on exactly the same item subset
-- never on a smaller subsample, which would be an unfair comparison. Every prediction is stamped ``reproducible=False``.

Probabilities for the multiple-choice / A-B / yes-no probes come from the
first-token logprobs when the API returns them; otherwise the parsed text
answer is used as a one-hot distribution. Every such prediction records
``extra["from_logprobs"]`` and, when true, ``extra["format_mass"]`` (the
probability on the valid answer tokens among the returned top tokens), so the
report can tell real probabilities from parsed ones (``metrics.probability_quality``).

Calls run in parallel on a bounded thread pool (``max_workers``, CLI
``--workers``): at most that many requests are in flight per model, however
many image groups ``evaluate`` runs at once, and identical requests in flight
together are sent once. Responses are cached on disk (SQLite), so an
interrupted run never pays twice.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..schema import Prediction, Probe, Query
from .base import one_hot, parse_letter, parse_yes_no
from .prompts import LETTERS, build_prompt


class ResponseCache:
    """SQLite response cache shared by the worker threads: one connection, serialised by a lock."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        # WAL: a commit costs ~0.04 ms instead of a disk sync (~4 ms), which would otherwise serialise the
        # workers; a power cut can lose the last few responses (fetched again), never corrupt the file
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.db.execute("SELECT v FROM r WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        blob = json.dumps(value)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (key, blob))
            self.db.commit()


def _jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _letter_probs(top: List[Tuple[str, float]], n: int) -> Tuple[Optional[List[float]], float]:
    """Renormalised option distribution + the raw mass on option letters (0 if none in ``top``)."""
    mass = [0.0] * n
    for tok, lp in top:
        t = tok.strip().strip("().").upper()
        if len(t) == 1 and t in LETTERS[:n]:
            mass[LETTERS.index(t)] += math.exp(lp)
    tot = sum(mass)
    return ([m / tot for m in mass] if tot > 0 else None), tot


def _yes_no_probs(top: List[Tuple[str, float]]) -> Tuple[Optional[List[float]], float]:
    y = sum(math.exp(lp) for t, lp in top if t.strip().lower() == "yes")
    n = sum(math.exp(lp) for t, lp in top if t.strip().lower() == "no")
    return ([y / (y + n), n / (y + n)] if (y + n) > 0 else None), y + n


class _APIBase:
    reproducible = False

    def __init__(self, name: str, model: str, cache_dir: str | Path = "runs/_api_cache",
                 qa_max_new_tokens: int = 10, caption_max_new_tokens: int = 48, max_retries: int = 5,
                 max_workers: int = 8):
        self.name, self.model = name, model
        self.cache = ResponseCache(Path(cache_dir) / f"{name}.sqlite")
        self.qa_max_new_tokens = qa_max_new_tokens
        self.caption_max_new_tokens = caption_max_new_tokens
        self.max_retries = max_retries
        self.max_workers = max(1, int(max_workers))
        # one pool per adapter bounds the requests in flight, however many groups evaluate() runs at once
        self._pool = ThreadPoolExecutor(self.max_workers, thread_name_prefix=f"gib-{name}")
        self.parallel_groups = self.max_workers  # evaluate() keeps this many image groups in flight
        self._lock = threading.Lock()            # counters and the in-flight table
        self._inflight: Dict[str, Future] = {}
        self.logprobs_seen = 0
        self.calls = 0

    def _call(self, jpeg: bytes, text: str, max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _cached_call(self, jpeg: bytes, text: str, max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        key = hashlib.sha256(json.dumps([self.model, text, max_tokens, logprobs]).encode()
                             + hashlib.sha256(jpeg).digest()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        # identical requests in flight at once (e.g. blank-image probes shared by items) are sent once,
        # so a parallel run makes no more calls than a sequential one
        with self._lock:
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = self._inflight[key] = Future()
        if not owner:
            return fut.result()
        try:
            res = self.cache.get(key)  # another thread may have finished it since the first look
            if res is None:
                res = self._call_with_retries(jpeg, text, max_tokens, logprobs)
                self.cache.put(key, res)
            fut.set_result(res)
            return res
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _call_with_retries(self, jpeg: bytes, text: str, max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                res = self._call(jpeg, text, max_tokens, logprobs)
                with self._lock:
                    self.calls += 1
                return res
            except Exception as e:  # rate limits, transient errors
                if attempt == self.max_retries - 1:
                    raise
                if logprobs and "logprob" in str(e).lower():
                    logprobs = False  # model does not support logprobs: retry without
                    continue
                time.sleep(delay * (0.5 + random.random()))  # jitter: parallel workers do not retry in lockstep
                delay *= 2
        raise RuntimeError("unreachable")

    def _predict(self, jpeg: bytes, q: Query) -> Prediction:
        t0 = time.perf_counter()
        probe = Probe(q.probe)
        pred = Prediction(qid=q.qid, model=self.name, reproducible=False)
        try:
            text = build_prompt(q)
            if probe in (Probe.QA, Probe.CAPTION):
                mt = self.caption_max_new_tokens if probe is Probe.CAPTION else self.qa_max_new_tokens
                pred.text = (self._cached_call(jpeg, text, mt, False).get("text") or "").strip()
            else:
                res = self._cached_call(jpeg, text, 1 if probe is not Probe.POPE else 2, True)
                top = [tuple(x) for x in res.get("top", [])]
                n = len(q.options)
                probs, mass = _yes_no_probs(top) if probe is Probe.POPE else _letter_probs(top, n)
                # True only when the distribution really came from token probabilities; logprobs
                # without any option token among them fall back to the parsed text, like no logprobs
                pred.extra["from_logprobs"] = probs is not None
                if probs is not None:
                    pred.extra["format_mass"] = round(mass, 4)
                elif probe is Probe.POPE:
                    probs = parse_yes_no(res.get("text", ""))
                else:
                    probs = one_hot(parse_letter(res.get("text", ""), n), n)
                pred.option_probs = probs
                with self._lock:
                    self.logprobs_seen += bool(top)
        except Exception as e:
            pred.error = f"{type(e).__name__}: {e}"[:300]
        pred.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return pred

    def run_group(self, image: Optional[Image.Image], queries: List[Query]) -> List[Prediction]:
        jpeg = _jpeg(image) if image is not None else b""
        return list(self._pool.map(lambda q: self._predict(jpeg, q), queries))  # keeps the order of `queries`

    def meta(self) -> Dict[str, Any]:
        return {"backend": type(self).__name__, "model": self.model, "api_calls": self.calls,
                "probe_calls_with_logprobs": self.logprobs_seen, "max_workers": self.max_workers,
                "reproducible": False}


class OpenAIVLM(_APIBase):
    """OpenAI chat-completions models with image input (e.g. gpt-4o)."""

    def __init__(self, name: str, model: str, detail: str = "auto", **kw):
        super().__init__(name, model, **kw)
        from openai import OpenAI  # reads OPENAI_API_KEY

        self.client = OpenAI()
        self.detail = detail

    def _call(self, jpeg: bytes, text: str, max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if jpeg:
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": self.detail}})
        content.append({"type": "text", "text": text})
        kw: Dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": content}],
                              "max_completion_tokens": max_tokens, "temperature": 0, "seed": 0}
        if logprobs:
            kw.update(logprobs=True, top_logprobs=20)
        r = self.client.chat.completions.create(**kw)
        ch = r.choices[0]
        top = []
        if logprobs and ch.logprobs and ch.logprobs.content:
            top = [(t.token, t.logprob) for t in ch.logprobs.content[0].top_logprobs]
        return {"text": ch.message.content or "", "top": top, "fingerprint": r.system_fingerprint}


class GeminiVLM(_APIBase):
    """Google Gemini models through the google-genai SDK."""

    def __init__(self, name: str, model: str, thinking_budget: Optional[int] = 0, **kw):
        super().__init__(name, model, **kw)
        from google import genai  # reads GEMINI_API_KEY / GOOGLE_API_KEY

        self.client = genai.Client()
        self.thinking_budget = thinking_budget

    def _call(self, jpeg: bytes, text: str, max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        from google.genai import types

        parts: List[Any] = []
        if jpeg:
            parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        parts.append(text)
        cfg: Dict[str, Any] = {"temperature": 0, "max_output_tokens": max_tokens}
        if logprobs:
            cfg.update(response_logprobs=True, logprobs=5)
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        try:
            r = self.client.models.generate_content(model=self.model, contents=parts,
                                                    config=types.GenerateContentConfig(**cfg))
        except Exception as e:
            if "thinking" in str(e).lower() and "thinking_config" in cfg:
                cfg.pop("thinking_config")
                r = self.client.models.generate_content(model=self.model, contents=parts,
                                                        config=types.GenerateContentConfig(**cfg))
            else:
                raise
        top = []
        cand = (r.candidates or [None])[0]
        lpr = getattr(cand, "logprobs_result", None) if cand is not None else None
        if logprobs and lpr and lpr.top_candidates:
            top = [(c.token, c.log_probability) for c in lpr.top_candidates[0].candidates]
        return {"text": r.text or "", "top": top}
