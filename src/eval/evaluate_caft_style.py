#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAFT-style evaluation with robust response loading.

- Auto-detect responses from JSON / JSONL / CSV
- Accepts response keys: response, answer, output, text, completion, generated_text,
  or OpenAI-style choices[0].message.content
- Dynamically selects judge metrics from YAML (alignment/coherence by default).
"""

import os
import sys
import json
import csv
import asyncio
import argparse
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import yaml

# Local import
from judge import OpenAiJudge


# --------------------------
# File loaders
# --------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_judge_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Expect structure: {"judge_prompts": {"alignment": "...", "coherence": "...", ...}}
    return (data.get("judge_prompts") or data)


def _extract_response_from_obj(obj: Any) -> Optional[str]:
    """
    Accepts a dict or string and returns a response string if possible.
    Tries common fields and OpenAI-style outputs.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip()

    if isinstance(obj, dict):
        # Common simple keys
        for key in ("response", "answer", "output", "text", "completion", "generated_text"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # vLLM-style or custom: {"outputs": [{"text": "..."}]}
        outputs = obj.get("outputs")
        if isinstance(outputs, list) and outputs:
            cand = outputs[0]
            if isinstance(cand, dict):
                for key in ("text", "response", "answer"):
                    val = cand.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # OpenAI style: {"choices":[{"message":{"content":"..."}}]}
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            ch = choices[0]
            if isinstance(ch, dict):
                # message.content
                msg = ch.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                # text (legacy)
                txt = ch.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()

    return None


def read_responses_any(path: str) -> List[str]:
    """
    Load a list of response strings from JSON / JSONL / CSV files,
    trying a variety of common formats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Responses file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    # JSON
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Many runs save {"responses": [...]}
        if isinstance(obj, dict):
            for k in ("responses", "data", "outputs", "items", "results"):
                if k in obj and isinstance(obj[k], list):
                    arr = obj[k]
                    out = []
                    for item in arr:
                        s = _extract_response_from_obj(item)
                        if s is not None:
                            out.append(s)
                    if out:
                        return out
            # Or a single dict is actually one response
            s = _extract_response_from_obj(obj)
            if s is not None:
                return [s]
            # Or maybe it's actually a list under the root
            if isinstance(obj, list):
                # fallthrough to list handler below
                pass
        if isinstance(obj, list):
            out = []
            for item in obj:
                s = _extract_response_from_obj(item)
                if s is not None:
                    out.append(s)
            if out:
                return out
        raise ValueError(f"Could not parse responses from JSON: {path}")

    # JSONL
    if ext == ".jsonl":
        rows = load_jsonl(path)
        out = []
        for item in rows:
            s = _extract_response_from_obj(item)
            if s is not None:
                out.append(s)
        if out:
            return out
        raise ValueError(f"Could not parse responses from JSONL: {path}")

    # CSV
    if ext == ".csv":
        df = pd.read_csv(path)
        # Try common columns
        for col in ("response", "answer", "output", "text", "completion", "generated_text"):
            if col in df.columns:
                vals = df[col].dropna().astype(str).tolist()
                if vals:
                    return vals
        # Try first string-like column
        for col in df.columns:
            series = df[col].dropna()
            if not series.empty and series.dtype == object:
                # Pick it if it looks like natural language (naive heuristic)
                vals = series.astype(str).tolist()
                if vals:
                    return vals
        raise ValueError(f"Could not parse responses from CSV: {path}")

    # Fallback: try to treat it as plain text (one response per line)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines:
        return lines

    raise ValueError(f"Unknown responses format or empty file: {path}")


def extract_question(record: Dict[str, Any]) -> str:
    """
    Extract question text from a dataset record that may be:
    - {"messages":[{"role":"user","content":"..."}, ...]}
    - {"question":"..."}
    - Otherwise: stringified fallback
    """
    if isinstance(record, dict):
        msgs = record.get("messages")
        if isinstance(msgs, list) and msgs:
            # Prefer first user turn; else first content
            user_first = None
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    user_first = m["content"]
                    break
            if not user_first:
                # fallback to first element with content
                for m in msgs:
                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                        user_first = m["content"]
                        break
            if isinstance(user_first, str):
                return user_first
        if "question" in record and isinstance(record["question"], str):
            return record["question"]
    return str(record)


def shorten(s: str, n: int = 100) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[:n] + "...")


# --------------------------
# Evaluation
# --------------------------

async def evaluate_responses(
    responses_file: str,
    dataset_path: str,
    output_file: str,
    api_key: str,
    max_samples: int = 1000,
    batch_size: int = 10,
    judge_model: str = "gpt-4.1-2025-04-14",
    judge_prompts_path: str = "judge_prompts.yaml",
    include_vuln: bool = False,
    debug_first_n: int = 3,
) -> pd.DataFrame:

    abs_resp = os.path.abspath(responses_file)
    abs_data = os.path.abspath(dataset_path)
    abs_yaml = os.path.abspath(judge_prompts_path)

    print(f"[Paths]")
    print(f"  Responses: {abs_resp}")
    print(f"  Dataset  : {abs_data}")
    print(f"  Prompts  : {abs_yaml}")

    print(f"\n[Load] Reading responses…")
    responses_list = read_responses_any(responses_file)  # -> List[str]
    print(f"[Load] Found {len(responses_list)} responses")

    print(f"[Load] Reading dataset…")
    dataset = load_jsonl(dataset_path)
    print(f"[Load] Found {len(dataset)} dataset rows")

    # length align
    N = min(max_samples, len(responses_list), len(dataset))
    responses_list = responses_list[:N]
    dataset = dataset[:N]
    print(f"[Eval] Using N={N} samples")

    print(f"[Load] Loading judge prompts…")
    judge_prompts = load_judge_prompts(judge_prompts_path)  # dict metric->template

    # Decide metrics dynamically
    allowed = {"alignment", "coherence"}
    if include_vuln:
        allowed.add("code_vulnerability")

    metrics = [m for m in judge_prompts.keys() if m in allowed]
    if not metrics:
        raise ValueError(f"No usable metrics in prompts file. Found keys={list(judge_prompts.keys())}")

    print(f"[Judges] Building judges for metrics: {metrics}")
    judges = {
        metric: OpenAiJudge(
            model=judge_model,
            prompt_template=judge_prompts[metric],
            openai_api_key=api_key,
            debug_first_n=debug_first_n,
        )
        for metric in metrics
    }

    # Show a couple sanity samples
    print("\n[Sanity] Example question/response pairs:")
    for i in range(min(3, N)):
        q = extract_question(dataset[i])
        a = responses_list[i]
        print(f"  #{i} Q: {shorten(q)}")
        print(f"     A: {shorten(a)}")

    # Evaluate in small batches (sequential inside batch for simplicity)
    results: List[Dict[str, Any]] = []
    total_batches = (N + batch_size - 1) // batch_size

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"\n[Batch] {start//batch_size + 1}/{total_batches}  (samples {start+1}-{end})")

        # Run judges serially per sample (easy to rate-limit); if you want concurrency, wrap with asyncio.Semaphore
        for i in range(start, end):
            q = extract_question(dataset[i])
            a = responses_list[i]

            row = {"sample_id": i, "question": q, "answer": a}
            for metric, judge in judges.items():
                try:
                    score = await judge(question=q, answer=a)
                except Exception as e:
                    print(f"[Warn] judge {metric} failed for sample {i}: {e}")
                    score = 0.0
                row[metric] = score

            results.append(row)

        # Save incrementally
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df.to_csv(output_file, index=False)

        # Save readable CSV
        readable = []
        for r in results:
            base = {
                "sample_id": r["sample_id"],
                "question_short": shorten(r["question"]),
                "answer_short": shorten(r["answer"]),
            }
            for m in metrics:
                base[m] = round(float(r.get(m, 0.0)), 1)
            readable.append(base)
        r_df = pd.DataFrame(readable)
        readable_path = output_file.replace(".csv", "_readable.csv")
        r_df.to_csv(readable_path, index=False)

        # Save JSON
        json_path = output_file.replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[Save] {len(results)} rows written")
        print(f"  - CSV     : {os.path.abspath(output_file)}")
        print(f"  - Readable: {os.path.abspath(readable_path)}")
        print(f"  - JSON    : {os.path.abspath(json_path)}")

        # Gentle pause between batches to avoid rate limits
        if end < N:
            await asyncio.sleep(0.5)

    print("\n[Done] Evaluation complete.")
    return pd.DataFrame(results)


# --------------------------
# CLI
# --------------------------

def main():
    p = argparse.ArgumentParser(description="CAFT-style evaluation with robust response loading")
    p.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    p.add_argument("--responses_file", default="results/Caft/caft_responses.json",
                  help="Path to responses file (JSON/JSONL/CSV supported)")
    p.add_argument("--dataset", default="caft/emergent_misalignment/datasets/insecure_val.jsonl",
                  help="Path to dataset JSONL")
    p.add_argument("--output_file", default="/home/ubuntu/GradientBasedCaft/results.csv", help="Output CSV file path")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--judge_model", default="gpt-4.1-2025-04-14")
    p.add_argument("--judge_prompts", default="judge_prompts.yaml")
    p.add_argument("--include_vuln", action="store_true",
                   help="Include code_vulnerability if present in prompts")
    p.add_argument("--debug_first_n", type=int, default=3)
    args = p.parse_args()

    judge_prompts_path = args.judge_prompts
    if not os.path.isabs(judge_prompts_path):
        judge_prompts_path = os.path.join(os.path.dirname(__file__), judge_prompts_path)

    asyncio.run(evaluate_responses(
        responses_file=args.responses_file,
        dataset_path=args.dataset,
        output_file=args.output_file,
        api_key=args.openai_api_key,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        judge_model=args.judge_model,
        judge_prompts_path=judge_prompts_path,
        include_vuln=args.include_vuln,
        debug_first_n=args.debug_first_n,
    ))


if __name__ == "__main__":
    main()
