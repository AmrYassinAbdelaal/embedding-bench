from __future__ import annotations

import json
import random
import urllib.request
import urllib.error
from dataclasses import dataclass

import numpy as np

from dataset_config import DatasetConfig


@dataclass
class LLMJudgeConfig:
    provider: str  # "openai" or "anthropic"
    api_key: str
    model: str
    max_samples: int = 50


# ---------------------------------------------------------------------------
# Provider-specific API calls
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an impartial relevance judge. Given a query and a passage, "
    "rate how relevant the passage is to the query on a scale of 1 to 5.\n\n"
    "1 = Completely irrelevant\n"
    "2 = Slightly relevant\n"
    "3 = Moderately relevant\n"
    "4 = Highly relevant\n"
    "5 = Perfectly relevant\n\n"
    "Respond with ONLY a single integer (1-5), nothing else."
)


def _build_user_prompt(query: str, passage: str) -> str:
    return f"Query: {query}\n\nPassage: {passage}"


def _call_openai(api_key: str, model: str, query: str, passage: str) -> int:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, passage)},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"].strip()
    return _parse_score(text)


def _call_anthropic(api_key: str, model: str, query: str, passage: str) -> int:
    body = json.dumps({
        "model": model,
        "max_tokens": 4,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": _build_user_prompt(query, passage)},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    text = data["content"][0]["text"].strip()
    return _parse_score(text)


def _parse_score(text: str) -> int:
    for ch in text:
        if ch.isdigit() and ch in "12345":
            return int(ch)
    return 3  # fallback to neutral


_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_llm_judge(
    model,
    ds_cfg: DatasetConfig,
    judge_cfg: LLMJudgeConfig,
    max_pairs: int | None = None,
    progress_callback=None,
) -> dict[str, float]:
    """Use an LLM to judge retrieval relevance for top-k results.

    For each sampled query, retrieves the top-5 passages by embedding
    similarity and asks the LLM to rate each one. Returns average
    relevance scores at different cut-offs.
    """
    from datasets import load_dataset

    if ds_cfg.data is not None:
        data = ds_cfg.data
    else:
        dataset = load_dataset(ds_cfg.name, ds_cfg.config, split=ds_cfg.split)
        data = {col: list(dataset[col]) for col in dataset.column_names}

    queries = list(data[ds_cfg.query_col])
    passages = list(data[ds_cfg.passage_col])

    if max_pairs is not None and len(queries) > max_pairs:
        queries = queries[:max_pairs]
        passages = passages[:max_pairs]

    # Encode
    emb_q = model.encode(queries, is_query=True)
    emb_p = model.encode(passages, is_query=False)

    # Normalise
    emb_q = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True)
    emb_p = emb_p / np.linalg.norm(emb_p, axis=1, keepdims=True)

    # Sample queries to judge
    n = len(queries)
    sample_size = min(judge_cfg.max_samples, n)
    sample_indices = sorted(random.sample(range(n), sample_size))

    call_fn = _PROVIDERS[judge_cfg.provider]
    top_k = 5

    # For each sampled query, get top-k passages and judge them
    relevance_at_k: list[list[int]] = []  # shape: (sample_size, top_k)
    total_calls = sample_size * top_k
    calls_done = 0

    for idx in sample_indices:
        query_emb = emb_q[idx : idx + 1]
        sims = (query_emb @ emb_p.T).flatten()
        top_indices = np.argsort(-sims)[:top_k]

        scores_for_query = []
        for passage_idx in top_indices:
            try:
                score = call_fn(
                    judge_cfg.api_key, judge_cfg.model,
                    queries[idx], passages[int(passage_idx)],
                )
            except Exception:
                score = 0  # treat API errors as 0
            scores_for_query.append(score)
            calls_done += 1
            if progress_callback:
                progress_callback(calls_done, total_calls)
        relevance_at_k.append(scores_for_query)

    arr = np.array(relevance_at_k, dtype=float)  # (sample_size, top_k)

    # Normalise scores to 0-1 (from 1-5 scale)
    arr_norm = (arr - 1.0) / 4.0

    # nDCG@5
    def _dcg(scores: np.ndarray) -> np.ndarray:
        positions = np.arange(1, scores.shape[1] + 1)
        return np.sum(scores / np.log2(positions + 1), axis=1)

    dcg = _dcg(arr_norm)
    ideal = _dcg(np.sort(arr_norm, axis=1)[:, ::-1])
    ndcg = np.where(ideal > 0, dcg / ideal, 0.0)

    return {
        "judge_avg@1": round(float(np.mean(arr_norm[:, 0])), 4),
        "judge_avg@5": round(float(np.mean(arr_norm)), 4),
        "judge_ndcg@5": round(float(np.mean(ndcg)), 4),
    }
