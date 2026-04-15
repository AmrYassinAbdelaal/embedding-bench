from __future__ import annotations

import io
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from datasets import load_dataset

from corpus import build_corpus
from dataset_config import DATASET_PRESETS, DatasetConfig
from evals.quality import evaluate_quality
from evals.speed import evaluate_speed
from models import (
    REGISTRY,
    VALID_BACKENDS,
    ModelConfig,
    load_custom_models_from_file,
    register_model,
    save_custom_model_to_file,
)
from wrapper import load_model

load_custom_models_from_file()

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Embedding Bench",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
    /* Tighter top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #22262e 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fafafa;
    }
    .metric-card .sub {
        font-size: 0.7rem;
        color: #666;
        margin-top: 2px;
    }
    .metric-card.best .value { color: #55A868; }
    .metric-card.worst .value { color: #C44E52; }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1px solid #2a2d35;
        margin: 1.2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.75rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #222;
        margin-top: 2rem;
    }
    .footer a { color: #4C72B0; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("# 📐 Embedding Bench")
    st.markdown(
        "<span style='color:#888; font-size:0.95rem;'>"
        "Compare text embedding models on quality, speed &amp; memory.</span>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        "<div style='text-align:right; padding-top:18px;'>"
        "<a href='https://github.com/amryassin/embedding-bench' target='_blank'>"
        "<img src='https://img.shields.io/badge/GitHub-repo-blue?logo=github' /></a></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Configuration")

st.sidebar.markdown("**Models**")
available_models = list(REGISTRY.keys())
selected_models = st.sidebar.multiselect(
    "Select models",
    available_models,
    default=["mpnet", "bge-small"] if len(available_models) >= 2 else available_models[:1],
    label_visibility="collapsed",
)

with st.sidebar.expander("➕ Add Custom Model"):
    with st.form("add_model_form", clear_on_submit=True):
        new_key = st.text_input("Registry key", placeholder="my-model")
        new_name = st.text_input("Display name", placeholder="My Custom Model")
        new_model_id = st.text_input("HuggingFace model ID", placeholder="org/model-name")
        new_backend = st.selectbox("Backend", sorted(VALID_BACKENDS))
        new_gguf_file = st.text_input(
            "GGUF filename (gguf backend only)", value="", placeholder="model.gguf"
        )
        new_is_baseline = st.checkbox("Mark as baseline", value=False)
        new_persist = st.checkbox("Save to disk", value=False,
                                  help="Persist to custom_models.json so it loads next session")
        submitted = st.form_submit_button("Add Model", use_container_width=True)
    if submitted:
        if not new_key or not new_name or not new_model_id:
            st.sidebar.error("Key, name, and model ID are required.")
        elif new_backend == "gguf" and not new_gguf_file:
            st.sidebar.error("GGUF filename is required for gguf backend.")
        else:
            cfg = ModelConfig(
                name=new_name,
                model_id=new_model_id,
                is_baseline=new_is_baseline,
                backend=new_backend,
                gguf_file=new_gguf_file or None,
            )
            try:
                register_model(new_key, cfg)
                if new_persist:
                    save_custom_model_to_file(new_key, cfg)
                st.rerun()
            except ValueError as e:
                st.sidebar.error(str(e))

st.sidebar.markdown("**Datasets**")
available_datasets = list(DATASET_PRESETS.keys())
selected_datasets = st.sidebar.multiselect(
    "Select dataset presets",
    available_datasets,
    default=["sts"],
    label_visibility="collapsed",
)

max_pairs = st.sidebar.number_input(
    "Max pairs per dataset",
    min_value=100,
    max_value=50000,
    value=1000,
    step=100,
    help="Limits the number of pairs evaluated. Keep low for large datasets.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Speed & Memory**")
run_speed = st.sidebar.checkbox("Speed benchmark", value=False)
run_memory = st.sidebar.checkbox("Memory benchmark", value=False)

corpus_size = 500
num_runs = 3
batch_size = 64
if run_speed or run_memory:
    corpus_size = st.sidebar.number_input("Corpus size", 100, 10000, 500, step=100)
    batch_size = st.sidebar.number_input("Batch size", 8, 512, 64, step=8)
if run_speed:
    num_runs = st.sidebar.number_input("Speed runs", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("**Cache**")
_cache_c1, _cache_c2 = st.sidebar.columns(2)
with _cache_c1:
    if st.button("🗑️ Clear All", use_container_width=True,
                 help="Clear cached models, datasets, and results"):
        st.cache_resource.clear()
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
with _cache_c2:
    if st.button("🔄 Results", use_container_width=True,
                 help="Clear eval results but keep models loaded"):
        st.cache_data.clear()
        for key in ["results", "selected_datasets"]:
            st.session_state.pop(key, None)
        st.rerun()

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Cached functions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_model(model_key: str):
    cfg = REGISTRY[model_key]
    return load_model(cfg)


@st.cache_data(show_spinner="Loading dataset...", ttl=3600)
def get_dataset(ds_name: str, ds_config: str | None, ds_split: str) -> dict:
    """Cache the HF dataset download & parse. Returns a dict of lists."""
    ds = load_dataset(ds_name, ds_config, split=ds_split)
    return {col: list(ds[col]) for col in ds.column_names}


@st.cache_data(show_spinner=False, ttl=3600)
def cached_evaluate_quality(
    _model,
    model_key: str,
    ds_name: str,
    ds_config: str | None,
    ds_split: str,
    query_col: str,
    passage_col: str,
    score_col: str | None,
    score_scale: float,
    max_pairs: int | None,
) -> dict[str, float]:
    """Cache quality results keyed by (model, dataset, max_pairs).

    The _model arg is excluded from the hash (underscore prefix).
    model_key is used as a hashable stand-in.
    """
    ds_cfg = DatasetConfig(
        name=ds_name, config=ds_config, split=ds_split,
        query_col=query_col, passage_col=passage_col,
        score_col=score_col, score_scale=score_scale,
    )
    return evaluate_quality(_model, ds_cfg, max_pairs=max_pairs)


@st.cache_data(show_spinner="Building corpus...", ttl=3600)
def cached_build_corpus(
    size: int, ds_name: str, ds_config: str | None, ds_split: str,
    query_col: str, passage_col: str,
) -> list[str]:
    ds_cfg = DatasetConfig(
        name=ds_name, config=ds_config, split=ds_split,
        query_col=query_col, passage_col=passage_col,
    )
    return build_corpus(size, ds_cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def render_metric_card(label: str, value: str, sub: str = "", css_class: str = "") -> str:
    cls = f"metric-card {css_class}".strip()
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='{cls}'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        f"{sub_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Chart style helper
# ---------------------------------------------------------------------------
CHART_BG = "#0E1117"
CHART_TEXT = "#CCCCCC"

def style_chart(fig, ax):
    """Apply dark theme to a matplotlib chart."""
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors=CHART_TEXT, labelsize=7)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.title.set_color("#FAFAFA")


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

        model = get_model(model_key)
        quality_results = {}
        for ds_cfg in ds_configs:
            ds_key = ds_cfg.name.split("/")[-1]
            step += 1
            progress.progress(
                step / total_steps,
                text=f"Evaluating **{cfg.name}** on *{ds_key}*...",
            )
            quality_results[ds_key] = cached_evaluate_quality(
                model, model_key,
                ds_cfg.name, ds_cfg.config, ds_cfg.split,
                ds_cfg.query_col, ds_cfg.passage_col,
                ds_cfg.score_col, ds_cfg.score_scale,
                max_pairs,
            )
        result["quality"] = quality_results

        if run_speed:
            step += 1
            progress.progress(step / total_steps, text=f"Speed benchmark: **{cfg.name}**...")
            ds0 = ds_configs[0]
            corpus = cached_build_corpus(
                corpus_size, ds0.name, ds0.config, ds0.split,
                ds0.query_col, ds0.passage_col,
            )
            result["speed"] = evaluate_speed(model, corpus, num_runs=num_runs, batch_size=batch_size)

        if run_memory:
            step += 1
            progress.progress(step / total_steps, text=f"Memory benchmark: **{cfg.name}**...")
            from evals.memory import evaluate_memory
            ds0 = ds_configs[0]
            corpus = cached_build_corpus(
                corpus_size, ds0.name, ds0.config, ds0.split,
                ds0.query_col, ds0.passage_col,
            )
            result["memory_mb"] = evaluate_memory(
                cfg.model_id, corpus, batch_size=batch_size, backend=cfg.backend,
            )

        results.append(result)

    progress.progress(1.0, text="Done!")
    time.sleep(0.3)
    progress.empty()

    st.session_state["results"] = results
    st.session_state["selected_datasets"] = selected_datasets

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.markdown(
        "<div style='text-align:center; padding:3rem 0; color:#666;'>"
        "<p style='font-size:2.5rem; margin-bottom:0.5rem;'>📐</p>"
        "<p style='font-size:1.1rem;'>Configure models &amp; datasets in the sidebar,<br>"
        "then hit <b>Run Benchmark</b>.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

results = st.session_state["results"]
selected_datasets_display = st.session_state["selected_datasets"]

# ---------------------------------------------------------------------------
# Highlight cards
# ---------------------------------------------------------------------------
ds_keys: list[str] = []
for r in results:
    q = r.get("quality")
    if q:
        ds_keys = list(q.keys())
        break

# Build a quick summary: best model per first dataset
if ds_keys:
    first_ds = ds_keys[0]
    first_metrics_sample = results[0].get("quality", {}).get(first_ds, {})
    primary_metric = "spearman" if "spearman" in first_metrics_sample else "mrr"
    primary_label = "Spearman" if primary_metric == "spearman" else "MRR"

    scores = [
        (r["name"], r.get("quality", {}).get(first_ds, {}).get(primary_metric, 0))
        for r in results
    ]
    best = max(scores, key=lambda x: x[1])

    speed_scores = [
        (r["name"], r.get("speed", {}).get("sentences_per_second", 0))
        for r in results
    ]
    fastest = max(speed_scores, key=lambda x: x[1]) if any(s[1] > 0 for s in speed_scores) else None

    mem_scores = [
        (r["name"], r.get("memory_mb", 0))
        for r in results
    ]
    lightest = min((m for m in mem_scores if m[1] > 0), key=lambda x: x[1], default=None)

    card_cols = st.columns(3)
    with card_cols[0]:
        st.markdown(render_metric_card(
            f"Best {primary_label} ({first_ds})",
            f"{best[1]:.4f}",
            best[0],
            "best",
        ), unsafe_allow_html=True)
    with card_cols[1]:
        if fastest and fastest[1] > 0:
            st.markdown(render_metric_card(
                "Fastest",
                f"{fastest[1]} sent/s",
                fastest[0],
                "best",
            ), unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card("Fastest", "—", "speed not measured"), unsafe_allow_html=True)
    with card_cols[2]:
        if lightest:
            st.markdown(render_metric_card(
                "Lightest",
                f"{lightest[1]} MB",
                lightest[0],
                "best",
            ), unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card("Lightest", "—", "memory not measured"), unsafe_allow_html=True)

    st.markdown("")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
st.markdown("#### 📊 Detailed Results")
flat_rows = [flatten_result(r) for r in results]
st.dataframe(flat_rows, use_container_width=True, hide_index=True)

col_dl, _ = st.columns([1, 4])
with col_dl:
    csv_data = results_to_csv(results)
    st.download_button(
        "📥 Download CSV",
        data=csv_data,
        file_name="embedding_bench_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.markdown("#### 📈 Charts")
models = [r["name"] for r in results]

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
        fig, ax = plt.subplots(figsize=(4, 2.4))
        style_chart(fig, ax)
        bars = ax.bar(models, values, color="#4C72B0", edgecolor="#5a82c0", linewidth=0.5)
        ax.set_ylabel("Spearman", fontsize=8)
        ax.set_title(f"Quality — {ds_key}", fontsize=9, pad=8)
        ax.set_ylim(0, 1.08)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, color=CHART_TEXT)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        metric_names = ["mrr", "recall@1", "recall@5", "recall@10"]
        x = np.arange(len(models))
        width = 0.18
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

        fig, ax = plt.subplots(figsize=(max(4, len(models) * 1.4), 3.0))
        style_chart(fig, ax)
        for i, (metric, color) in enumerate(zip(metric_names, colors)):
            values = [r.get("quality", {}).get(ds_key, {}).get(metric, 0) for r in results]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=metric, color=color,
                          edgecolor=color, linewidth=0.3, alpha=0.9)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=CHART_TEXT)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(f"Retrieval Quality — {ds_key}", fontsize=9, pad=8)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=6, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22),
                  facecolor=CHART_BG, edgecolor="#444", labelcolor=CHART_TEXT)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# Speed & Memory side by side
speed_values = [r.get("speed", {}).get("sentences_per_second", 0) for r in results]
mem_values = [r.get("memory_mb", 0) for r in results]
has_speed = any(v > 0 for v in speed_values)
has_memory = any(v > 0 for v in mem_values)

if has_speed or has_memory:
    cols = st.columns(2 if has_speed and has_memory else 1)

    if has_speed:
        with cols[0]:
            fig, ax = plt.subplots(figsize=(3.5, 2.4))
            style_chart(fig, ax)
            bars = ax.bar(models, speed_values, color="#55A868", edgecolor="#65b878", linewidth=0.5)
            ax.set_ylabel("Sent / s", fontsize=8)
            ax.set_title("Encoding Speed", fontsize=9, pad=8)
            for bar, v in zip(bars, speed_values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            str(v), ha="center", va="bottom", fontsize=7, color=CHART_TEXT)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    if has_memory:
        col_idx = 1 if has_speed else 0
        with cols[col_idx]:
            fig, ax = plt.subplots(figsize=(3.5, 2.4))
            style_chart(fig, ax)
            bars = ax.bar(models, mem_values, color="#C44E52", edgecolor="#d45e62", linewidth=0.5)
            ax.set_ylabel("MB", fontsize=8)
            ax.set_title("Memory Usage", fontsize=9, pad=8)
            for bar, v in zip(bars, mem_values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            str(v), ha="center", va="bottom", fontsize=7, color=CHART_TEXT)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer'>"
    "Built with <a href='https://streamlit.io'>Streamlit</a> · "
    "Models via <a href='https://huggingface.co'>HuggingFace</a> · "
    "<a href='https://github.com/amryassin/embedding-bench'>Source on GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
