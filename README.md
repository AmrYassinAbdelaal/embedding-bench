# embedding-bench

Compare text embedding models across retrieval quality, inference speed, and memory footprint. Everything runs locally — no external API calls.

## Models

| Key | Model | Role |
|-----|-------|------|
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | Baseline |
| `bge-small` | `BAAI/bge-small-en-v1.5` | |

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
"e5-small": ModelConfig(
    name="e5-small-v2",
    model_id="intfloat/e5-small-v2",
),
```

## Project structure

```
embedding-bench/
├── bench.py           # CLI entry point
├── models.py          # Model registry
├── corpus.py          # Sentence corpus builder
├── report.py          # Table formatting
├── evals/
│   ├── quality.py     # STS Benchmark evaluation
│   ├── speed.py       # Latency measurement
│   └── memory.py      # Memory measurement
└── requirements.txt
```
