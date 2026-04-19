from __future__ import annotations

import io
import csv
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datasets import load_dataset

from corpus import build_corpus
from dataset_config import DATASET_PRESETS, DatasetConfig
from evals.quality import ALL_RETRIEVAL_METRICS, DEFAULT_RETRIEVAL_METRICS, evaluate_quality
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
        "<a href='https://github.com/amryassinabdelaal/embedding-bench' target='_blank'>"
        "<img src='https://img.shields.io/badge/GitHub-repo-blue?logo=github' /></a></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: slugify a display name into a registry key
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Configuration")

# ---- Models ---------------------------------------------------------------
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
        new_name = st.text_input("Display name", placeholder="My Custom Model")
        new_model_id = st.text_input("HuggingFace model ID", placeholder="org/model-name")
        new_backend = st.selectbox("Backend", sorted(VALID_BACKENDS))
        new_gguf_file = st.text_input(
            "GGUF filename", value="", placeholder="model.gguf",
            help="Only needed for the gguf backend.",
        )
        _adv_c1, _adv_c2 = st.columns(2)
        new_is_baseline = _adv_c1.checkbox("Baseline", value=False)
        new_persist = _adv_c2.checkbox("Save to disk", value=False,
                                       help="Persist across sessions")
        submitted = st.form_submit_button("Add Model", use_container_width=True)
    if submitted:
        new_key = _slugify(new_name) if new_name else ""
        errors: list[str] = []
        if not new_name:
            errors.append("Display name is required.")
        elif new_key in REGISTRY:
            errors.append(f"A model named '{new_name}' already exists.")
        if not new_model_id:
            errors.append("HuggingFace model ID is required.")
        elif "/" not in new_model_id:
            errors.append("Model ID should be in `org/model-name` format.")
        if new_backend == "gguf" and not new_gguf_file:
            errors.append("GGUF filename is required for gguf backend.")
        if errors:
            for err in errors:
                st.sidebar.error(err)
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

# ---- Datasets -------------------------------------------------------------
st.sidebar.markdown("**Datasets**")

# Merge preset + user datasets (need this before the multiselect)
user_datasets: dict[str, DatasetConfig] = st.session_state.get("user_datasets", {})
all_datasets = {**DATASET_PRESETS, **user_datasets}

available_datasets = list(all_datasets.keys())
selected_datasets = st.sidebar.multiselect(
    "Select datasets",
    available_datasets,
    default=["sts"] if "sts" in available_datasets else available_datasets[:1],
    label_visibility="collapsed",
)

_MAX_UPLOAD_ROWS = 50_000
_MAX_UPLOAD_MB = 50

with st.sidebar.expander("➕ Add Dataset"):
    ds_source = st.radio(
        "Source", ["Upload file", "HuggingFace Hub"],
        horizontal=True, label_visibility="collapsed",
    )

    if ds_source == "Upload file":
        st.caption(
            "CSV or TSV with query and passage columns. "
            "Optional numeric score column enables Spearman correlation; "
            "otherwise MRR & Recall@k are used. Max 50 MB / 50 k rows."
        )
        uploaded_file = st.file_uploader(
            "Upload CSV or TSV", type=["csv", "tsv"], label_visibility="collapsed",
        )
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > _MAX_UPLOAD_MB:
                st.error(f"File too large ({file_size_mb:.1f} MB). Max {_MAX_UPLOAD_MB} MB.")
            else:
                sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
                try:
                    user_df = pd.read_csv(uploaded_file, sep=sep)
                except Exception as e:
                    st.error(f"Failed to parse: {e}")
                    user_df = None

                if user_df is not None:
                    errs: list[str] = []
                    if len(user_df.columns) < 2:
                        errs.append("Need at least 2 columns.")
                    if len(user_df) == 0:
                        errs.append("File is empty.")
                    if len(user_df) > _MAX_UPLOAD_ROWS:
                        errs.append(f"Too many rows ({len(user_df):,}). Max {_MAX_UPLOAD_ROWS:,}.")
                    if user_df.columns.duplicated().any():
                        errs.append("Duplicate column names.")
                    if errs:
                        for e in errs:
                            st.error(e)
                    else:
                        cols = list(user_df.columns)
                        st.dataframe(user_df.head(5), use_container_width=True, hide_index=True)

                        with st.form("add_dataset_form", clear_on_submit=False):
                            ds_label = st.text_input(
                                "Dataset name",
                                value=uploaded_file.name.rsplit(".", 1)[0],
                            )
                            user_query_col = st.selectbox("Query column", cols, index=0)
                            user_passage_col = st.selectbox(
                                "Passage column", cols, index=min(1, len(cols) - 1),
                            )
                            has_score = st.checkbox("Has score column")
                            user_score_col = st.selectbox(
                                "Score column", cols,
                                index=min(2, len(cols) - 1),
                                disabled=not has_score,
                            )
                            user_score_scale = st.number_input(
                                "Score scale (max value)",
                                min_value=1.0, value=5.0, step=1.0,
                                disabled=not has_score,
                                help="Scores divided by this to normalise to 0-1.",
                            )
                            ds_submitted = st.form_submit_button(
                                "Add Dataset", use_container_width=True,
                            )

                        if ds_submitted:
                            sub_errs: list[str] = []
                            if not ds_label:
                                sub_errs.append("Name is required.")
                            if user_query_col == user_passage_col:
                                sub_errs.append("Query and passage columns must differ.")
                            if has_score and user_score_col in (
                                user_query_col, user_passage_col,
                            ):
                                sub_errs.append("Score column must differ from query/passage.")
                            if user_df[user_query_col].astype(str).str.strip().eq("").all():
                                sub_errs.append(f"Query column '{user_query_col}' is empty.")
                            if user_df[user_passage_col].astype(str).str.strip().eq("").all():
                                sub_errs.append(f"Passage column '{user_passage_col}' is empty.")
                            if has_score:
                                try:
                                    pd.to_numeric(user_df[user_score_col], errors="raise")
                                except (ValueError, TypeError):
                                    sub_errs.append(f"Score column '{user_score_col}' must be numeric.")
                            if sub_errs:
                                for e in sub_errs:
                                    st.error(e)
                            else:
                                data_dict = {c: user_df[c].astype(str).tolist() for c in cols}
                                if has_score:
                                    data_dict[user_score_col] = [
                                        float(v) for v in user_df[user_score_col]
                                    ]
                                user_ds_cfg = DatasetConfig(
                                    name=f"user/{ds_label}",
                                    query_col=user_query_col,
                                    passage_col=user_passage_col,
                                    score_col=user_score_col if has_score else None,
                                    score_scale=user_score_scale if has_score else 1.0,
                                    data=data_dict,
                                )
                                if "user_datasets" not in st.session_state:
                                    st.session_state["user_datasets"] = {}
                                st.session_state["user_datasets"][ds_label] = user_ds_cfg
                                st.success(f"Added **{ds_label}** ({len(user_df):,} rows)")

    else:  # HuggingFace Hub
        st.caption("Load any dataset from [huggingface.co/datasets](https://huggingface.co/datasets).")
        with st.form("add_hf_dataset_form", clear_on_submit=True):
            hf_ds_label = st.text_input("Dataset name", placeholder="my-dataset")
            hf_ds_id = st.text_input("HuggingFace ID", placeholder="org/dataset-name")
            _hf_c1, _hf_c2 = st.columns(2)
            hf_ds_config = _hf_c1.text_input("Config", value="", help="Leave blank if none.")
            hf_ds_split = _hf_c2.text_input("Split", value="test")
            hf_query_col = st.text_input("Query column", placeholder="query")
            hf_passage_col = st.text_input("Passage column", placeholder="passage")
            hf_has_score = st.checkbox("Has score column")
            hf_score_col = st.text_input(
                "Score column", placeholder="score", disabled=not hf_has_score,
            )
            hf_score_scale = st.number_input(
                "Score scale (max value)", min_value=1.0, value=5.0, step=1.0,
                disabled=not hf_has_score,
                help="Scores divided by this to normalise to 0-1.",
            )
            hf_submitted = st.form_submit_button("Add Dataset", use_container_width=True)
        if hf_submitted:
            hf_errors: list[str] = []
            if not hf_ds_label:
                hf_errors.append("Dataset name is required.")
            if not hf_ds_id:
                hf_errors.append("HuggingFace ID is required.")
            if not hf_query_col:
                hf_errors.append("Query column is required.")
            if not hf_passage_col:
                hf_errors.append("Passage column is required.")
            if hf_query_col and hf_passage_col and hf_query_col == hf_passage_col:
                hf_errors.append("Query and passage columns must differ.")
            if hf_has_score and not hf_score_col:
                hf_errors.append("Score column is required when enabled.")
            if hf_has_score and hf_score_col in (hf_query_col, hf_passage_col):
                hf_errors.append("Score column must differ from query/passage.")

            if hf_errors:
                for err in hf_errors:
                    st.error(err)
            else:
                try:
                    _cfg_arg = hf_ds_config or None
                    _test_ds = load_dataset(hf_ds_id, _cfg_arg, split=hf_ds_split)
                    _ds_cols = _test_ds.column_names
                    _missing = [
                        c for c in [hf_query_col, hf_passage_col]
                        + ([hf_score_col] if hf_has_score else [])
                        if c not in _ds_cols
                    ]
                    if _missing:
                        st.error(
                            f"Column(s) not found: {', '.join(_missing)}. "
                            f"Available: {', '.join(_ds_cols)}"
                        )
                    else:
                        hf_ds_cfg = DatasetConfig(
                            name=hf_ds_id,
                            config=_cfg_arg,
                            split=hf_ds_split,
                            query_col=hf_query_col,
                            passage_col=hf_passage_col,
                            score_col=hf_score_col if hf_has_score else None,
                            score_scale=hf_score_scale if hf_has_score else 1.0,
                        )
                        if "user_datasets" not in st.session_state:
                            st.session_state["user_datasets"] = {}
                        st.session_state["user_datasets"][hf_ds_label] = hf_ds_cfg
                        st.success(f"Added **{hf_ds_label}**")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")

# ---- Evaluation options ---------------------------------------------------
_LLM_PROVIDERS = {"openai": "OpenAI", "anthropic": "Anthropic"}
_DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}

with st.sidebar.expander("⚙️ Evaluation"):
    max_pairs = st.number_input(
        "Max pairs per dataset",
        min_value=100, max_value=50000, value=1000, step=100,
        help="Caps the number of pairs evaluated per dataset.",
    )

    selected_metrics = st.multiselect(
        "Retrieval metrics",
        ALL_RETRIEVAL_METRICS,
        default=DEFAULT_RETRIEVAL_METRICS,
        help="Metrics for pair-based datasets (no score column). Scored datasets always use Spearman.",
    )

    st.markdown("---")
    run_speed = st.checkbox("Speed benchmark")
    run_memory = st.checkbox("Memory benchmark")

    corpus_size = 500
    num_runs = 3
    batch_size = 64
    if run_speed or run_memory:
        _sp_c1, _sp_c2 = st.columns(2)
        corpus_size = _sp_c1.number_input("Corpus size", 100, 10000, 500, step=100)
        batch_size = _sp_c2.number_input("Batch size", 8, 512, 64, step=8)
    if run_speed:
        num_runs = st.number_input("Speed runs", 1, 10, 3)

    st.markdown("---")
    run_llm_judge = st.checkbox("LLM as a Judge")

    llm_provider = "openai"
    llm_api_key = ""
    llm_model = ""
    llm_max_samples = 50

    if run_llm_judge:
        st.caption(
            "An LLM rates how relevant retrieved passages are to each query (1-5). "
            "API charges apply."
        )
        llm_provider = st.selectbox(
            "Provider", list(_LLM_PROVIDERS.keys()),
            format_func=lambda k: _LLM_PROVIDERS[k],
        )
        llm_api_key = st.text_input(
            "API key", type="password", placeholder="sk-...",
        )
        llm_model = st.text_input("Model", value=_DEFAULT_MODELS[llm_provider])
        llm_max_samples = st.number_input(
            "Samples to judge", min_value=5, max_value=500, value=50, step=5,
            help="Queries sampled. Each = 5 API calls (top-5 passages).",
        )

    st.markdown("---")
    _cache_c1, _cache_c2 = st.columns(2)
    with _cache_c1:
        if st.button("🗑 Clear All", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with _cache_c2:
        if st.button("🔄 Results", use_container_width=True):
            st.cache_data.clear()
            for key in ["results", "selected_datasets"]:
                st.session_state.pop(key, None)
            st.rerun()

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
    metrics: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Cache quality results keyed by (model, dataset, max_pairs, metrics).

    The _model arg is excluded from the hash (underscore prefix).
    model_key is used as a hashable stand-in.
    """
    ds_cfg = DatasetConfig(
        name=ds_name, config=ds_config, split=ds_split,
        query_col=query_col, passage_col=passage_col,
        score_col=score_col, score_scale=score_scale,
    )
    return evaluate_quality(
        _model, ds_cfg, max_pairs=max_pairs,
        metrics=list(metrics) if metrics else None,
    )


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
    for ds_key, metrics in r.get("llm_judge", {}).items():
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
# Chart helpers
# ---------------------------------------------------------------------------
CHART_BG = "#0E1117"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color="#CCCCCC", size=11),
    margin=dict(l=50, r=20, t=40, b=60),
    bargap=0.25,
    xaxis=dict(gridcolor="#2a2d35", zerolinecolor="#2a2d35"),
    yaxis=dict(gridcolor="#2a2d35", zerolinecolor="#2a2d35"),
)


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

if not selected_datasets:
    st.warning("Select at least one dataset from the sidebar.")
    st.stop()

if run_llm_judge and not llm_api_key:
    st.warning("Enter an API key in the sidebar to use LLM judge evaluation.")
    run_llm_judge = False

run_btn = st.sidebar.button("🚀 Run", type="primary", use_container_width=True)

if run_btn:
    ds_configs = [all_datasets[k] for k in selected_datasets]
    results = []
    progress = st.progress(0, text="Starting...")
    total_steps = len(selected_models) * (
        len(ds_configs) + int(run_speed) + int(run_memory)
        + (len(ds_configs) if run_llm_judge else 0)
    )
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
            _metrics = selected_metrics or None
            if ds_cfg.data is not None:
                quality_results[ds_key] = evaluate_quality(
                    model, ds_cfg, max_pairs=max_pairs, metrics=_metrics,
                )
            else:
                quality_results[ds_key] = cached_evaluate_quality(
                    model, model_key,
                    ds_cfg.name, ds_cfg.config, ds_cfg.split,
                    ds_cfg.query_col, ds_cfg.passage_col,
                    ds_cfg.score_col, ds_cfg.score_scale,
                    max_pairs,
                    metrics=tuple(_metrics) if _metrics else None,
                )
        result["quality"] = quality_results

        if run_llm_judge:
            from evals.llm_judge import LLMJudgeConfig, evaluate_llm_judge
            judge_cfg = LLMJudgeConfig(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                max_samples=llm_max_samples,
            )
            judge_results = {}
            for ds_cfg in ds_configs:
                ds_key = ds_cfg.name.split("/")[-1]
                step += 1
                progress.progress(
                    step / total_steps,
                    text=f"LLM judge: **{cfg.name}** on *{ds_key}*...",
                )
                try:
                    judge_results[ds_key] = evaluate_llm_judge(
                        model, ds_cfg, judge_cfg, max_pairs=max_pairs,
                    )
                except Exception as e:
                    st.warning(f"LLM judge failed for {cfg.name}/{ds_key}: {e}")
                    judge_results[ds_key] = {}
            result["llm_judge"] = judge_results

        if run_speed:
            step += 1
            progress.progress(step / total_steps, text=f"Speed benchmark: **{cfg.name}**...")
            ds0 = ds_configs[0]
            if ds0.data is not None:
                corpus = build_corpus(corpus_size, ds0)
            else:
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
            if ds0.data is not None:
                corpus = build_corpus(corpus_size, ds0)
            else:
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
        "then hit <b>Run Evaluation</b>.</p></div>",
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
    if "spearman" in first_metrics_sample:
        primary_metric = "spearman"
        primary_label = "Spearman"
    else:
        # Use the first available retrieval metric
        primary_metric = next(iter(first_metrics_sample), "mrr")
        primary_label = primary_metric.upper()

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
        fig = go.Figure(go.Bar(
            x=models, y=values,
            marker_color="#4C72B0",
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            **_PLOTLY_LAYOUT,
            title=f"Quality — {ds_key}",
            yaxis_title="Spearman",
            yaxis_range=[0, 1.08],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        metric_names = list(first_metrics.keys())
        _palette = [
            "#4C72B0", "#55A868", "#C44E52", "#8172B2",
            "#E5AE38", "#DD8452", "#64B5CD", "#8C8C8C",
            "#D4A6C8", "#6ACC65", "#D65F5F",
        ]
        fig = go.Figure()
        for i, metric in enumerate(metric_names):
            color = _palette[i % len(_palette)]
            values = [r.get("quality", {}).get(ds_key, {}).get(metric, 0) for r in results]
            fig.add_trace(go.Bar(
                name=metric, x=models, y=values,
                marker_color=color,
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            ))
        fig.update_layout(
            **_PLOTLY_LAYOUT,
            title=f"Retrieval Quality — {ds_key}",
            yaxis_title="Score",
            yaxis_range=[0, 1.12],
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

# LLM Judge charts
for ds_key in ds_keys:
    has_judge = any(r.get("llm_judge", {}).get(ds_key) for r in results)
    if not has_judge:
        continue
    judge_metrics = ["judge_avg@1", "judge_avg@5", "judge_ndcg@5"]
    judge_labels = ["Avg@1", "Avg@5", "nDCG@5"]
    colors = ["#E5AE38", "#DD8452", "#C44E52"]

    fig = go.Figure()
    for metric, label, color in zip(judge_metrics, judge_labels, colors):
        values = [r.get("llm_judge", {}).get(ds_key, {}).get(metric, 0) for r in results]
        fig.add_trace(go.Bar(
            name=label, x=models, y=values,
            marker_color=color,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=f"LLM Judge — {ds_key}",
        yaxis_title="Score",
        yaxis_range=[0, 1.12],
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

# Speed & Memory side by side
speed_values = [r.get("speed", {}).get("sentences_per_second", 0) for r in results]
mem_values = [r.get("memory_mb", 0) for r in results]
has_speed = any(v > 0 for v in speed_values)
has_memory = any(v > 0 for v in mem_values)

if has_speed or has_memory:
    cols = st.columns(2 if has_speed and has_memory else 1)

    if has_speed:
        with cols[0]:
            fig = go.Figure(go.Bar(
                x=models, y=speed_values,
                marker_color="#55A868",
                text=[str(v) if v > 0 else "" for v in speed_values],
                textposition="outside",
            ))
            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title="Encoding Speed",
                yaxis_title="Sent / s",
            )
            st.plotly_chart(fig, use_container_width=True)

    if has_memory:
        col_idx = 1 if has_speed else 0
        with cols[col_idx]:
            fig = go.Figure(go.Bar(
                x=models, y=mem_values,
                marker_color="#C44E52",
                text=[str(v) if v > 0 else "" for v in mem_values],
                textposition="outside",
            ))
            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title="Memory Usage",
                yaxis_title="MB",
            )
            st.plotly_chart(fig, use_container_width=True)

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
