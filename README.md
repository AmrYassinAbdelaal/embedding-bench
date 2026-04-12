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

## Metrics

| Dimension | Metric | Method |
|-----------|--------|--------|
| Quality | Spearman rho | STS Benchmark test set (1,379 pairs) |
| Speed | Median encode time | Wall-clock over N runs with warmup |
| Memory | Peak RSS delta | Isolated subprocess via `psutil` |

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
├── bench.py           # CLI entry point
├── models.py          # Model registry
├── wrapper.py         # Backend wrappers (sbert, fastembed, gguf)
├── corpus.py          # Sentence corpus builder
├── report.py          # Table formatting
├── evals/
│   ├── quality.py     # STS Benchmark evaluation
│   ├── speed.py       # Latency measurement
│   └── memory.py      # Memory measurement
└── requirements.txt
```
