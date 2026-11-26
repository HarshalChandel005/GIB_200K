"""Parse ``name=backend:model`` specs into adapters.

    llava16=hf:llava-hf/llava-v1.6-mistral-7b-hf
    qwen2vl=hf:Qwen/Qwen2-VL-7B-Instruct
    gpt4o=openai:gpt-4o-2024-08-06
    gemini=gemini:gemini-2.5-flash
    sim-a=sim:{"w_txt": 1.6}          (tests / demo only)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from ..schema import BaseItem


def parse_spec(spec: str) -> Tuple[str, str, str]:
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise SystemExit(f"model spec must look like name=backend:model, got {spec!r}")
    name, rest = spec.split("=", 1)
    backend, model = rest.split(":", 1)
    return name.strip(), backend.strip(), model.strip()


def make_adapter(spec: str, cfg: Dict[str, Any], items: Optional[Dict[str, BaseItem]] = None,
                 **kw: Any):
    name, backend, model = parse_spec(spec)
    dec = cfg["decoding"]
    if backend == "hf":
        from .hf_vlm import HFVLM

        return HFVLM(name, model, qa_max_new_tokens=dec["qa_max_new_tokens"],
                     caption_max_new_tokens=dec["caption_max_new_tokens"], **kw)
    if backend == "openai":
        from .api import OpenAIVLM

        return OpenAIVLM(name, model, qa_max_new_tokens=dec["qa_max_new_tokens"],
                         caption_max_new_tokens=dec["caption_max_new_tokens"],
                         **{k: v for k, v in kw.items() if k in ("detail", "cache_dir", "max_workers")})
    if backend == "gemini":
        from .api import GeminiVLM

        return GeminiVLM(name, model, qa_max_new_tokens=dec["qa_max_new_tokens"],
                         caption_max_new_tokens=dec["caption_max_new_tokens"],
                         **{k: v for k, v in kw.items() if k in ("thinking_budget", "cache_dir", "max_workers")})
    if backend == "sim":
        from .simulated import SimulatedVLM

        assert items is not None
        profile = json.loads(model) if model.strip().startswith("{") else {}
        return SimulatedVLM(name, items, profile)
    raise SystemExit(f"unknown backend {backend!r} (use hf, openai, gemini or sim)")
