# embedding-bench

Compare text embedding models across retrieval quality, inference speed, and memory footprint. Everything runs locally — no external API calls.

## Models

| Key | Model | Backend | Role |
|-----|-------|---------|------|
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | sbert | Baseline |
| `bge-small` | `BAAI/bge-small-en-v1.5` | sbert | |
| `bge-small-fe` | `BAAI/bge-small-en-v1.5` | fastembed | |
| `all-minilm-fe` | `sentence-transformers/all-MiniLM-L6-v2` | fastembed | |

Three backends are supported:

- **sbert** — [sentence-transformers](https://www.sbert.net/) (PyTorch). Default.
- **fastembed** — [qdrant/fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime). Lighter and often faster.
- **gguf** — [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for quantised GGUF models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic

```bash
# Full benchmark (quality + speed + memory)
python bench.py

# Specific models
python bench.py --models mpnet bge-small

# Compare the same model across backends
python bench.py --models bge-small bge-small-fe

# Skip expensive evals
python bench.py --skip-quality
python bench.py --skip-memory

# Tune corpus size and batch size
python bench.py --corpus-size 500 --batch-size 32 --num-runs 5
```

### Datasets

By default, quality is evaluated on the STS Benchmark. You can evaluate on multiple HuggingFace datasets using built-in presets:

| Preset | HF Dataset | Type | Pairs |
|--------|-----------|------|-------|
| `sts` | `mteb/stsbenchmark-sts` | Scored (Spearman) | 1,379 |
| `natural-questions` | `sentence-transformers/natural-questions` | Retrieval (MRR/Recall) | 100,231 |
| `msmarco` | `sentence-transformers/msmarco-bm25` | Retrieval | 503,000 |
| `squad` | `sentence-transformers/squad` | Retrieval | 87,599 |
| `trivia-qa` | `sentence-transformers/trivia-qa` | Retrieval | 73,346 |
| `gooaq` | `sentence-transformers/gooaq` | Retrieval | 3,012,496 |
| `hotpotqa` | `sentence-transformers/hotpotqa` | Retrieval | 84,500 |

```bash
# Evaluate on multiple datasets
python bench.py --models mpnet bge-small \
  --datasets sts natural-questions squad \
  --skip-speed --skip-memory

# Limit pairs for large datasets
python bench.py --datasets msmarco gooaq --max-pairs 1000

# Use a custom HF dataset (overrides --datasets)
python bench.py --dataset my-org/my-pairs \
  --query-col query --passage-col passage --score-col none
```

Scored datasets (with `--score-col`) report **Spearman correlation**. Pair-only datasets (`--score-col none`) report **MRR**, **Recall@1**, **Recall@5**, and **Recall@10**.

### Export results

```bash
# Export to CSV
python bench.py --csv results.csv

# Save charts as PNG
python bench.py --charts ./results

# Both
python bench.py --models mpnet bge-small \
  --datasets sts squad natural-questions \
  --max-pairs 1000 \
  --csv results.csv --charts ./results
```

Charts generated:
- `quality_<dataset>.png` — Spearman bar chart (scored) or grouped MRR/Recall bars (retrieval)
- `speed.png` — sentences/second comparison
- `memory.png` — peak memory usage comparison

## Metrics

| Dimension | Metric | Method |
|-----------|--------|--------|
| Quality (scored) | Spearman rho | Cosine similarity vs gold scores |
| Quality (pairs) | MRR, Recall@k | Retrieval ranking of positive passages |
| Speed | Median encode time | Wall-clock over N runs with warmup |
| Memory | Peak RSS delta | Isolated subprocess via `psutil` |

## CLI reference

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

Edit `models.py` and add an entry to `REGISTRY`:

```python
# sentence-transformers backend (default)
"e5-small": ModelConfig(
    name="e5-small-v2",
    model_id="intfloat/e5-small-v2",
),

# fastembed backend
"e5-small-fe": ModelConfig(
    name="e5-small-v2 (fastembed)",
    model_id="intfloat/e5-small-v2",
    backend="fastembed",
),
```

## Project structure

```
embedding-bench/
├── bench.py             # CLI entry point
├── models.py            # Model registry
├── wrapper.py           # Backend wrappers (sbert, fastembed, gguf)
├── corpus.py            # Sentence corpus builder
├── dataset_config.py    # Dataset presets and configuration
├── report.py            # Table formatting, CSV export, charts
├── evals/
│   ├── quality.py       # STS + retrieval evaluation
│   ├── speed.py         # Latency measurement
│   └── memory.py        # Memory measurement
└── requirements.txt
```
