import os
import uuid
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

from source_code.config.llm_config import PipelineConfig
from source_code.config.loaders import load_pipeline_inputs
from source_code.graph import build_graph


# ── Steven Universe CSS theme ────────────────────────────────────────────────

SU_CSS = """
/* App background */
.stApp {
    background: linear-gradient(160deg, #1a0a2e, #0d0d1a);
    color: #e8e0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2d1b4e;
    border-right: 1px solid #c77dff33;
}
[data-testid="stSidebar"] * {
    color: #e8e0f0 !important;
}

/* Headings */
h1, h2, h3 {
    color: #ffd166 !important;
    font-weight: 700;
}
h4, h5, h6 {
    color: #c77dff !important;
    font-weight: 600;
}

/* Body text */
p, li, label, .stMarkdown {
    color: #e8e0f0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff6b9d, #c77dff);
    color: #0d0d1a !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: box-shadow 0.2s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 18px #ff6b9d88;
    background: linear-gradient(135deg, #ff6b9d, #c77dff);
    color: #0d0d1a !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff6b9d, #c77dff);
    color: #0d0d1a !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
}

/* Text inputs & textareas */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #2d1b4e !important;
    border: 1px solid #c77dff66 !important;
    color: #e8e0f0 !important;
    border-radius: 6px;
}

/* Select boxes */
.stSelectbox > div > div {
    background-color: #2d1b4e !important;
    border: 1px solid #c77dff66 !important;
    color: #e8e0f0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #ff6b9d66 !important;
    border-radius: 8px;
    background: #2d1b4e33;
}

/* Forms */
[data-testid="stForm"] {
    background-color: #1e0f3a;
    border: 1px solid #c77dff33;
    border-radius: 10px;
    padding: 1rem;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #ffd166 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Progress items */
.progress-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 1rem 0;
}
.progress-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
}
.progress-item.complete {
    color: #ffd166;
    background: #ffd16618;
    border: 1px solid #ffd16644;
}
.progress-item.active {
    color: #ff6b9d;
    background: #ff6b9d18;
    border: 1px solid #ff6b9d66;
    animation: pulse-border 1.5s ease-in-out infinite;
}
.progress-item.pending {
    color: #e8e0f040;
    background: transparent;
    border: 1px solid #e8e0f015;
}

@keyframes pulse-border {
    0%, 100% { border-color: #ff6b9d44; }
    50%       { border-color: #ff6b9dcc; }
}

/* Divider */
hr {
    border-color: #c77dff33;
}

/* Code blocks */
code {
    background: #2d1b4e !important;
    color: #c77dff !important;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
}
"""

# ── Constants ────────────────────────────────────────────────────────────────

NODE_ORDER = [
    "agent1_renamer",
    "executor1",
    "reclassify_columns",
    "agent2_cleaner",
    "executor2",
]

NODE_LABELS = {
    "agent1_renamer":     "Agent 1 — Field Renamer",
    "executor1":          "Executor 1 — Apply Renames",
    "reclassify_columns": "Reclassify Columns",
    "agent2_cleaner":     "Agent 2 — Field Cleaner",
    "executor2":          "Executor 2 — Apply Cleaning",
}


# ── Session state helpers ────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "pipeline_status":   "idle",
        "graph":             None,
        "thread_config":     None,
        "initial_input":     None,
        "resume_value":      None,
        "node_log":          [],
        "interrupt_payload": None,
        "temp_files":        [],
        "final_output_path": None,
        "column_map":        {},
        "error_message":     "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_session():
    for path in st.session_state.get("temp_files", []):
        try:
            os.unlink(path)
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ── Core streaming function ──────────────────────────────────────────────────

def _run_stream():
    """
    Drive the LangGraph stream until it either completes or hits an interrupt.

    Reads:  st.session_state.{graph, thread_config, initial_input, resume_value}
    Writes: st.session_state.{node_log, interrupt_payload, column_map,
                              final_output_path, error_message}

    Returns:
        "interrupted" | "complete" | "error"
    """
    graph         = st.session_state.graph
    thread_config = st.session_state.thread_config
    initial_input = st.session_state.initial_input
    resume_value  = st.session_state.resume_value

    stream_input = initial_input if resume_value is None else Command(resume=resume_value)

    try:
        for chunk in graph.stream(stream_input, thread_config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                st.session_state.interrupt_payload = chunk["__interrupt__"][0].value
                return "interrupted"
            for node_name, node_data in chunk.items():
                if node_name not in st.session_state.node_log:
                    st.session_state.node_log.append(node_name)
                if isinstance(node_data, dict):
                    if node_data.get("column_map"):
                        st.session_state.column_map = node_data["column_map"]
                    if node_data.get("output_path"):
                        st.session_state.final_output_path = node_data["output_path"]
        return "complete"
    except Exception as exc:
        st.session_state.error_message = str(exc)
        return "error"


# ── Progress tracker ─────────────────────────────────────────────────────────

def _render_progress(active_node=None):
    completed = st.session_state.node_log
    html_parts = ['<div class="progress-container">']
    for node in NODE_ORDER:
        label = NODE_LABELS[node]
        if node in completed:
            html_parts.append(
                f'<div class="progress-item complete">✦ {label}</div>'
            )
        elif node == active_node:
            html_parts.append(
                f'<div class="progress-item active">◈ {label}</div>'
            )
        else:
            html_parts.append(
                f'<div class="progress-item pending">○ {label}</div>'
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ── UI pages ─────────────────────────────────────────────────────────────────

def _render_idle():
    st.markdown("# ✦ Run Pipeline")
    st.markdown("Upload your dataset and optional context files to start the cleaning pipeline.")
    st.markdown("---")

    with st.form("pipeline_form"):
        csv_file = st.file_uploader("Dataset *", type=["csv"])

        st.markdown("**SQL Query (optional)**")
        col_text, col_upload = st.columns([3, 1])
        with col_text:
            sql_text = st.text_area(
                "SQL text",
                height=100,
                label_visibility="collapsed",
                placeholder="Paste SQL query here...",
            )
        with col_upload:
            sql_file = st.file_uploader(
                "Upload .sql",
                type=["sql"],
                label_visibility="collapsed",
            )

        rules_text = st.text_area(
            "Domain Rules (optional)",
            height=80,
            placeholder="e.g. data_vol is in KB, aon is in days",
        )
        target_col = st.text_input("Target Column", value="ChurnFlag")

        submitted = st.form_submit_button("✦ Run Pipeline")

    if not submitted:
        return

    if csv_file is None:
        st.error("Please upload a dataset CSV.")
        return

    temp_files = []

    # Write CSV to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(csv_file.read())
        csv_path = f.name
    temp_files.append(csv_path)

    # Resolve SQL content
    if sql_file is not None:
        sql_content = sql_file.read().decode("utf-8")
    elif sql_text.strip():
        sql_content = sql_text
    else:
        sql_content = ""

    with tempfile.NamedTemporaryFile(suffix=".sql", delete=False, mode="w", encoding="utf-8") as f:
        f.write(sql_content)
        sql_path = f.name
    temp_files.append(sql_path)

    # Resolve rules content
    rules_content = rules_text.strip() if rules_text.strip() else ""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write(rules_content)
        rules_path = f.name
    temp_files.append(rules_path)

    st.session_state.temp_files = temp_files

    try:
        initial_input = load_pipeline_inputs(
            data_path=csv_path,
            query_path=sql_path,
            rules_path=rules_path,
            target_column=target_col.strip() or "ChurnFlag",
        )
    except Exception as exc:
        st.error(f"Failed to load inputs: {exc}")
        return

    config      = PipelineConfig.from_env()
    checkpointer = MemorySaver()
    graph       = build_graph(config, checkpointer=checkpointer)
    thread_id   = str(uuid.uuid4())

    st.session_state.initial_input   = initial_input
    st.session_state.graph           = graph
    st.session_state.thread_config   = {"configurable": {"thread_id": thread_id}}
    st.session_state.resume_value    = None
    st.session_state.node_log        = []
    st.session_state.pipeline_status = "running"
    st.rerun()


def _render_running():
    st.markdown("# ✦ Pipeline Running")
    _render_progress()

    with st.spinner("Executing pipeline..."):
        result = _run_stream()

    st.session_state.pipeline_status = result
    st.rerun()


def _render_interrupted():
    payload        = st.session_state.interrupt_payload or {}
    interrupt_type = payload.get("type")

    st.markdown("# ✦ Pipeline Paused — Input Required")

    active_node = {
        "ambiguous_fields": "agent1_renamer",
        "flagged_columns":  "agent2_cleaner",
    }.get(interrupt_type)

    _render_progress(active_node=active_node)
    st.markdown("---")

    if interrupt_type == "ambiguous_fields":
        _render_ambiguous_fields_form(payload.get("fields", []))
    elif interrupt_type == "flagged_columns":
        _render_flagged_columns_form(payload.get("columns", []))
    else:
        st.warning(f"Unknown interrupt type: `{interrupt_type}`")
        if st.button("✦ Continue"):
            st.session_state.resume_value    = "acknowledged"
            st.session_state.pipeline_status = "running"
            st.rerun()


def _render_ambiguous_fields_form(fields):
    st.markdown("### ◈ Agent 1 — Field Renaming Requires Your Input")
    st.markdown("The AI could not confidently rename the following columns. "
                "Please choose a name for each one.")

    if not fields:
        if st.button("✦ Confirm and Resume"):
            st.session_state.resume_value    = {}
            st.session_state.pipeline_status = "running"
            st.rerun()
        return

    with st.form("ambiguous_form"):
        for field in fields:
            orig       = field.get("original_column", "")
            reason     = field.get("reason", "")
            candidates = field.get("candidates", [])

            st.markdown(f"**Column:** `{orig}`")
            if reason:
                st.markdown(f"*{reason}*")

            options  = candidates + ["[Enter custom name]"] if candidates else ["[Enter custom name]"]
            st.selectbox(f"Rename `{orig}` to:", options, key=f"sel_{orig}")
            st.text_input(
                f"Or enter a custom name for `{orig}`:",
                key=f"cust_{orig}",
                placeholder="Leave blank to use the selection above",
            )
            st.markdown("---")

        submitted = st.form_submit_button("✦ Confirm and Resume")

    if submitted:
        decisions = {}
        for field in fields:
            orig   = field.get("original_column", "")
            custom = st.session_state.get(f"cust_{orig}", "").strip()
            sel    = st.session_state.get(f"sel_{orig}", "")
            if custom:
                decisions[orig] = custom
            elif sel and sel != "[Enter custom name]":
                decisions[orig] = sel

        st.session_state.resume_value    = decisions
        st.session_state.pipeline_status = "running"
        st.rerun()


def _render_flagged_columns_form(columns):
    st.markdown("### ◈ Agent 2 — Flagged Columns Require Review")
    st.markdown("The following columns could not be cleaned automatically and will be preserved unchanged.")

    if columns:
        for col in columns:
            if isinstance(col, dict):
                col_name = col.get("column", str(col))
                reason   = col.get("reason", "")
                st.markdown(f"- `{col_name}` — {reason}" if reason else f"- `{col_name}`")
            else:
                st.markdown(f"- `{col}`")
    else:
        st.markdown("*(No columns flagged.)*")

    st.markdown("")
    if st.button("✦ Acknowledged — Continue Pipeline"):
        st.session_state.resume_value    = "acknowledged"
        st.session_state.pipeline_status = "running"
        st.rerun()


def _render_complete():
    st.markdown("# ✦ Pipeline Complete")
    _render_progress()
    st.markdown("---")

    output_path = st.session_state.final_output_path

    if output_path and os.path.exists(output_path):
        df = pd.read_csv(output_path)
        st.markdown("### Cleaned DataFrame Preview")
        st.dataframe(df.head(50), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="✦ Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_output.csv",
            mime="text/csv",
        )
    else:
        st.warning("Output file not found — the pipeline may have completed without saving.")

    column_map = st.session_state.column_map
    if column_map:
        st.markdown("### Column Rename Map")
        map_df = pd.DataFrame(
            list(column_map.items()),
            columns=["Original", "Standardized"],
        )
        st.dataframe(map_df, use_container_width=True)

    st.markdown("")
    if st.button("✦ Start New Run"):
        _reset_session()
        st.rerun()


def _render_error():
    st.markdown("# ✦ Pipeline Error")
    st.error(st.session_state.error_message or "An unknown error occurred.")
    if st.button("✦ Start New Run"):
        _reset_session()
        st.rerun()


# ── Pages ────────────────────────────────────────────────────────────────────

def _page_run_pipeline():
    status = st.session_state.pipeline_status
    if status == "idle":
        _render_idle()
    elif status == "running":
        _render_running()
    elif status == "interrupted":
        _render_interrupted()
    elif status == "complete":
        _render_complete()
    elif status == "error":
        _render_error()
    else:
        st.error(f"Unknown pipeline status: {status}")


def _page_database():
    st.markdown("# ✦ Database")
    st.info("Coming soon.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="DS Machine",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(f"<style>{SU_CSS}</style>", unsafe_allow_html=True)

    _init_session()

    with st.sidebar:
        st.markdown("## ✦ DS Machine")
        st.markdown("*Automated data science pipeline*")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Run Pipeline", "Database"],
            label_visibility="collapsed",
        )

    if page == "Run Pipeline":
        _page_run_pipeline()
    else:
        _page_database()


if __name__ == "__main__":
    main()
