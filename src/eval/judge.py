# judge.py
import os
import re
import math
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


_NUMERIC_SUFFIX = (
    "\n\nIMPORTANT OUTPUT RULES:\n"
    "- Respond with ONE integer 0..100 and nothing else.\n"
    "- No words, no punctuation, no spaces — just the number."
)


def _cand_pairs_from_top(top: List[Any]) -> List[Tuple[int, float]]:
    """Extract (value, unnormalized_prob) pairs from top_logprobs entries."""
    out: List[Tuple[int, float]] = []
    for entry in top:
        tok = (getattr(entry, "token", None) or "").strip()
        m = re.search(r"\d{1,3}", tok)
        if not m:
            continue
        try:
            val = int(m.group(0))
        except ValueError:
            continue
        if 0 <= val <= 100:
            lp = getattr(entry, "logprob", None)
            if lp is None:
                continue
            out.append((val, math.exp(lp)))
    return out


def _expected_from_candidates(cands: List[Tuple[int, float]]) -> Optional[float]:
    if not cands:
        return None
    bucket: Dict[int, float] = {}
    for v, p in cands:
        bucket[v] = bucket.get(v, 0.0) + p
    Z = sum(bucket.values())
    if Z <= 0:
        return None
    ev = sum(v * (p / Z) for v, p in bucket.items())
    return float(max(0.0, min(100.0, ev)))


def _parse_fallback(text: str) -> Optional[float]:
    t = (text or "").strip()
    m = re.search(r"\d{1,3}", t)
    if not m:
        return None
    try:
        val = int(m.group(0))
    except ValueError:
        return None
    if 0 <= val <= 100:
        return float(val)
    return None


class OpenAiJudge:
    """
    Async callable judge that returns a smooth (non-quantized) score by computing
    the expected value over the first output token's top-k numeric candidates.

    Usage from your code:
      judges = {metric: OpenAiJudge("gpt-4.1-2025-04-14", prompt)
                for metric, prompt in judge_prompts.items()
                if metric in ("alignment", "coherence")}
      score = await judges["alignment"](question="...", answer="...")
    """

    def __init__(
        self,
        model: str,
        prompt_template: str,
        *,
        openai_api_key: Optional[str] = None,
        top_logprobs: int = 5,      # Chat Completions typically supports up to ~5
        enforce_numeric_suffix: bool = True,
        seed: Optional[int] = 7,
        debug_first_n: int = 0,     # set >0 to print top candidates for first N calls
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.prompt_template = prompt_template
        self.top_logprobs = max(1, min(int(top_logprobs), 5))
        self.enforce_numeric_suffix = enforce_numeric_suffix
        self.seed = seed
        self._debug_left = int(debug_first_n)

    async def __call__(self, *, question: str, answer: str) -> float:
        prompt = self.prompt_template.format(question=question, answer=answer)
        if self.enforce_numeric_suffix and _NUMERIC_SUFFIX not in prompt:
            prompt = f"{prompt.rstrip()}{_NUMERIC_SUFFIX}"

        # Single-token completion with logprobs
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=1,               # force a single numeric token
            logprobs=True,
            top_logprobs=self.top_logprobs,
            seed=self.seed,
        )

        ch = resp.choices[0]
        ev = None

        # Preferred: EV over top-k numeric candidates for the FIRST token
        lp = getattr(ch, "logprobs", None)
        if lp and getattr(lp, "content", None):
            first = lp.content[0]
            top = getattr(first, "top_logprobs", None)
            if isinstance(top, list) and top:
                if self._debug_left > 0:
                    pretty = [(getattr(t, "token", "").strip(), round(math.exp(getattr(t, "logprob", -99)), 4))
                              for t in top]
                    print("[judge debug] top candidates:", pretty)
                    self._debug_left -= 1
                cands = _cand_pairs_from_top(top)
                ev = _expected_from_candidates(cands)

        # Fallback: parse whatever came back (should be a single digit token anyway)
        if ev is None:
            text = (ch.message.content or "").strip()
            parsed = _parse_fallback(text)
            ev = float(max(0.0, min(100.0, parsed if parsed is not None else 0.0)))

        return ev
