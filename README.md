---
title: Embedding Bench
emoji: 📐
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.56.0"
app_file: app.py
pinned: false
license: mit
---

# embedding-bench

Compare text embedding models on quality, speed, and memory. Includes a Streamlit web UI and a CLI.

## Features

- **40+ pre-configured models** — sentence-transformers, BGE, E5, GTE, Nomic, Jina, Arctic, and more
- **4 backends** — sbert (PyTorch), fastembed (ONNX), gguf (llama-cpp), libembedding
- **7 built-in datasets** — STS Benchmark, Natural Questions, MS MARCO, SQuAD, TriviaQA, GooAQ, HotpotQA
- **Custom datasets** — upload your own CSV/TSV or load any HuggingFace dataset
- **Custom models** — add any HuggingFace embedding model from the UI
- **11 retrieval metrics** — MRR, MAP@k, NDCG@k, Precision@k, Recall@k (all configurable)
- **LLM as a Judge** — use OpenAI or Anthropic to rate retrieval relevance
- **Interactive charts** — Plotly-powered, with hover, zoom, and PNG export

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web UI

```bash
streamlit run app.py
```

The sidebar has three sections:

1. **Models** — select from the registry or add a custom HuggingFace model
2. **Datasets** — pick built-in presets, upload a CSV/TSV, or add any HuggingFace dataset
3. **Evaluation** — configure metrics, speed/memory benchmarks, LLM judge, and max pairs

### Custom datasets

You can add datasets two ways from the sidebar:

- **Upload file** — CSV or TSV (max 50 MB, 50k rows) with a query column and a passage column. Optionally include a numeric score column for Spearman correlation; otherwise retrieval metrics (MRR, Recall@k, etc.) are used.
- **HuggingFace Hub** — provide the dataset ID (e.g. `mteb/stsbenchmark-sts`), config, split, and column names. The dataset is validated on add.

### LLM as a Judge

Enable in the Evaluation section. Provide your OpenAI or Anthropic API key. For each sampled query, the top-5 retrieved passages are rated for relevance (1–5) by the LLM. Reports judge_avg@1, judge_avg@5, and judge_nDCG@5.

### Metrics

| Dimension | Metrics | Method |
|-----------|---------|--------|
| Quality (scored) | Spearman | Cosine similarity vs gold scores |
| Quality (pairs) | MRR, MAP@5/10, NDCG@5/10, Precision@1/5/10, Recall@1/5/10 | Retrieval ranking of positive passages |
| LLM Judge | Avg@1, Avg@5, nDCG@5 | LLM relevance ratings on retrieved passages |
| Speed | Median encode time, sent/s | Wall-clock over N runs with warmup |
| Memory | Peak RSS delta (MB) | Isolated subprocess via `psutil` |

## CLI

```bash
# Full benchmark (quality + speed + memory)
python bench.py

# Specific models
python bench.py --models mpnet bge-small

# Compare backends
python bench.py --models bge-small bge-small-fe

# Skip expensive evals
python bench.py --skip-quality
python bench.py --skip-memory

# Multiple datasets with pair limit
python bench.py --models mpnet bge-small \
  --datasets sts natural-questions squad \
  --max-pairs 1000 --skip-speed --skip-memory

# Custom HF dataset
python bench.py --dataset my-org/my-pairs \
  --query-col query --passage-col passage --score-col none

# Export
python bench.py --csv results.csv --charts ./results
```

### Built-in dataset presets

| Preset | HF Dataset | Type |
|--------|-----------|------|
| `sts` | `mteb/stsbenchmark-sts` | Scored (Spearman) |
| `natural-questions` | `sentence-transformers/natural-questions` | Retrieval |
| `msmarco` | `sentence-transformers/msmarco-bm25` | Retrieval |
| `squad` | `sentence-transformers/squad` | Retrieval |
| `trivia-qa` | `sentence-transformers/trivia-qa` | Retrieval |
| `gooaq` | `sentence-transformers/gooaq` | Retrieval |
| `hotpotqa` | `sentence-transformers/hotpotqa` | Retrieval |

### CLI flags

```
--models            Models to benchmark (default: all)
--corpus-size       Sentences for speed/memory tests (default: 1000)
--batch-size        Encoding batch size (default: 64)
--num-runs          Speed benchmark runs (default: 3)
--skip-quality      Skip quality evaluation
--skip-speed        Skip speed measurement
--skip-memory       Skip memory measurement
--datasets          Dataset presets (default: sts)
--max-pairs         Limit pairs per dataset
--dataset           Custom HF dataset (overrides --datasets)
--config            Dataset config/subset name (e.g. 'triplet')
--split             Dataset split (default: test)
--query-col         Query column name (default: sentence1)
--passage-col       Passage column name (default: sentence2)
--score-col         Score column (default: score, 'none' for pairs)
--score-scale       Score normalization divisor (default: 5.0)
--csv               Export results to CSV
--charts            Save charts to directory
```

## Adding a model

From the web UI, click **Add Custom Model** in the sidebar — just provide a display name and a HuggingFace model ID.

Or edit `models.py` directly:

```python
"e5-small": ModelConfig(
    name="e5-small-v2",
    model_id="intfloat/e5-small-v2",
),
```

## Project structure

```
embedding-bench/
├── app.py               # Streamlit web UI
├── bench.py             # CLI entry point
├── models.py            # Model registry (40+ models)
├── wrapper.py           # Backend wrappers (sbert, fastembed, gguf, libembedding)
├── corpus.py            # Sentence corpus builder
├── dataset_config.py    # Dataset presets and configuration
├── report.py            # Table formatting, CSV export, charts (CLI)
├── evals/
│   ├── quality.py       # Quality evaluation (Spearman + retrieval metrics)
│   ├── speed.py         # Latency measurement
│   ├── memory.py        # Memory measurement
│   └── llm_judge.py     # LLM-as-a-Judge evaluation
└── requirements.txt
```
