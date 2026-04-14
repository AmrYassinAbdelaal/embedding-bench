from __future__ import annotations

import io
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from corpus import build_corpus
from dataset_config import DATASET_PRESETS, DatasetConfig
from evals.quality import evaluate_quality
from evals.speed import evaluate_speed
from models import REGISTRY, ModelConfig
from wrapper import load_model

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Embedding Bench",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Embedding Bench")
st.caption("Compare text embedding models on quality, speed & memory — all in your browser.")

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.header("Models")
available_models = list(REGISTRY.keys())
selected_models = st.sidebar.multiselect(
    "Select models",
    available_models,
    default=["mpnet", "bge-small"] if len(available_models) >= 2 else available_models[:1],
)

st.sidebar.header("Datasets")
available_datasets = list(DATASET_PRESETS.keys())
selected_datasets = st.sidebar.multiselect(
    "Select dataset presets",
    available_datasets,
    default=["sts"],
)

max_pairs = st.sidebar.number_input(
    "Max pairs per dataset",
    min_value=100,
    max_value=50000,
    value=1000,
    step=100,
    help="Limits the number of pairs evaluated. Keep low for large datasets.",
)

st.sidebar.header("Speed & Memory")
run_speed = st.sidebar.checkbox("Run speed benchmark", value=False)
run_memory = st.sidebar.checkbox("Run memory benchmark", value=False)

corpus_size = 500
num_runs = 3
batch_size = 64
if run_speed or run_memory:
    corpus_size = st.sidebar.number_input("Corpus size", 100, 10000, 500, step=100)
    batch_size = st.sidebar.number_input("Batch size", 8, 512, 64, step=8)
if run_speed:
    num_runs = st.sidebar.number_input("Speed runs", 1, 10, 3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_model(model_key: str):
    cfg = REGISTRY[model_key]
    return load_model(cfg)


def flatten_result(r: dict) -> dict:
    flat = {"Model": r["name"]}
    for ds_key, metrics in r.get("quality", {}).items():
        for metric_name, value in metrics.items():
            flat[f"{ds_key}/{metric_name}"] = value
    speed = r.get("speed")
    if speed:
        flat["Speed (sent/s)"] = speed["sentences_per_second"]
        flat["Median Time (s)"] = speed["median_seconds"]
    mem = r.get("memory_mb")
    if mem is not None:
        flat["Memory (MB)"] = mem
    return flat


def results_to_csv(results: list[dict]) -> str:
    rows = [flatten_result(r) for r in results]
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

if not selected_datasets:
    st.warning("Select at least one dataset from the sidebar.")
    st.stop()

run_btn = st.sidebar.button("🚀 Run Benchmark", type="primary", use_container_width=True)

if run_btn:
    ds_configs = [DATASET_PRESETS[k] for k in selected_datasets]
    results = []
    progress = st.progress(0, text="Starting...")
    total_steps = len(selected_models) * (len(ds_configs) + int(run_speed) + int(run_memory))
    step = 0

    for model_key in selected_models:
        cfg = REGISTRY[model_key]
        result: dict = {"name": cfg.name, "is_baseline": cfg.is_baseline}

        # Quality
        model = get_model(model_key)
        quality_results = {}
        for ds_cfg in ds_configs:
            ds_key = ds_cfg.name.split("/")[-1]
            step += 1
            progress.progress(
                step / total_steps,
                text=f"Evaluating {cfg.name} on {ds_key}...",
            )
            quality_results[ds_key] = evaluate_quality(model, ds_cfg, max_pairs=max_pairs)
        result["quality"] = quality_results

        # Speed
        if run_speed:
            step += 1
            progress.progress(step / total_steps, text=f"Speed benchmark: {cfg.name}...")
            corpus = build_corpus(corpus_size, ds_configs[0])
            result["speed"] = evaluate_speed(model, corpus, num_runs=num_runs, batch_size=batch_size)

        # Memory
        if run_memory:
            step += 1
            progress.progress(step / total_steps, text=f"Memory benchmark: {cfg.name}...")
            from evals.memory import evaluate_memory
            corpus = build_corpus(corpus_size, ds_configs[0])
            result["memory_mb"] = evaluate_memory(
                cfg.model_id, corpus, batch_size=batch_size, backend=cfg.backend,
            )

        results.append(result)

    progress.progress(1.0, text="Done!")
    time.sleep(0.3)
    progress.empty()

    # Store results in session state
    st.session_state["results"] = results
    st.session_state["selected_datasets"] = selected_datasets

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.info("Configure options in the sidebar and hit **Run Benchmark**.")
    st.stop()

results = st.session_state["results"]
selected_datasets = st.session_state["selected_datasets"]

# --- Results table ---
st.header("Results")
flat_rows = [flatten_result(r) for r in results]
st.dataframe(flat_rows, use_container_width=True)

# --- CSV download ---
csv_data = results_to_csv(results)
st.download_button(
    "📥 Download CSV",
    data=csv_data,
    file_name="embedding_bench_results.csv",
    mime="text/csv",
)

# --- Charts ---
st.header("Charts")
models = [r["name"] for r in results]

# Discover datasets
ds_keys: list[str] = []
for r in results:
    q = r.get("quality")
    if q:
        ds_keys = list(q.keys())
        break

for ds_key in ds_keys:
    first_metrics = None
    for r in results:
        m = r.get("quality", {}).get(ds_key)
        if m:
            first_metrics = m
            break
    if not first_metrics:
        continue

    if "spearman" in first_metrics:
        values = [r.get("quality", {}).get(ds_key, {}).get("spearman", 0) for r in results]
        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
        bars = ax.bar(models, values, color="#4C72B0")
        ax.set_ylabel("Spearman Correlation")
        ax.set_title(f"Quality — {ds_key}")
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        metric_names = ["mrr", "recall@1", "recall@5", "recall@10"]
        x = np.arange(len(models))
        width = 0.18
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.2), 4.5))
        for i, (metric, color) in enumerate(zip(metric_names, colors)):
            values = [r.get("quality", {}).get(ds_key, {}).get(metric, 0) for r in results]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=metric, color=color)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title(f"Retrieval Quality — {ds_key}")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# Speed chart
speed_values = [r.get("speed", {}).get("sentences_per_second", 0) for r in results]
if any(v > 0 for v in speed_values):
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
    bars = ax.bar(models, speed_values, color="#55A868")
    ax.set_ylabel("Sentences / second")
    ax.set_title("Encoding Speed")
    for bar, v in zip(bars, speed_values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Memory chart
mem_values = [r.get("memory_mb", 0) for r in results]
if any(v > 0 for v in mem_values):
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
    bars = ax.bar(models, mem_values, color="#C44E52")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage")
    for bar, v in zip(bars, mem_values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
