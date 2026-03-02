# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run the pipeline
python main.py
```

Requires a `.env` file with `GROQ_API_KEY=<your_key>`.

## Architecture

This is a **LangGraph-based agentic data cleaning pipeline** for telecom churn datasets. The pipeline uses LLM-generated Python code that gets executed at runtime to standardize and clean a CSV.

### Data Flow

```
main.py → graph.py (ds_machine.stream)
  └─► agent1_renamer  (LLM generates rename code)
  └─► executor1       (executes rename code on df, saves CSV)
  └─► agent2_cleaner  (LLM generates cleaning code)
  └─► executor2       (executes cleaning code on df, saves CSV)
```

Final output: `standardized_output.csv` (hardcoded in `executor.py`)

### Key Files

- **`main.py`** — Entry point. Loads the CSV, extracts metadata (dtypes, sample row, value counts, null summary), classifies columns, and builds the initial `AgentState` dict passed into the LangGraph stream.
- **`source_code/graph.py`** — Defines the LangGraph `StateGraph` with 4 nodes. The same `code_executor_agent` function is reused for both executor nodes.
- **`source_code/state.py`** — `AgentState` TypedDict — the shared state schema flowing through the graph. All agents read from and write subsets of this dict.
- **`source_code/agents/agent_1_field_renamer.py`** — Sends a prompt to `llama-3.3-70b-versatile` via Groq. Parses two code blocks from the response: `rename_map` (rename code) and `ambiguous_fields` (list of uncertain columns).
- **`source_code/agents/agent_2_field_cleaner.py`** — Similar pattern: sends enriched metadata to the LLM, parses two code blocks: cleaning code and `flagged_columns`.
- **`source_code/agents/executor.py`** — Loads the original CSV fresh from `state["file_path"]`, runs `exec(cleaning_code, {}, local_vars)` with `df` and `pd` in scope, saves result to `standardized_output.csv`.
- **`source_code/utils.py`** — `classify_columns` (splits object columns into genuinely categorical vs. numeric-like using 80% threshold), `build_value_counts_summary`, `build_null_summary`, `preprocess_column_names` (strips SQL table aliases), `load_prompts`.
- **`source_code/prompts/agent_1.json`** — Full prompt template and telecom knowledge base (abbreviation map, unit rules, period rules, naming style, target schema) for Agent 1.
- **`source_code/prompts/agent_2.json`** — Prompt template and cleaning rules for Agent 2.

### Inputs

| File | Purpose |
|---|---|
| `data/telecom_churn_data.csv` | Input dataset (hardcoded in `main.py`) |
| `queries/churn_query.sql` | SQL query providing column context for alias resolution |
| `rules/special_rules.txt` | Domain overrides (e.g. "data_vol is in KB", "aon is in days") |

### Agent Output Contract

Each LLM agent must return **exactly two Python code blocks** in its response:
1. The executable transformation code (operates on `df`)
2. A metadata block (`ambiguous_fields` or `flagged_columns`) as an executable Python assignment

The executor always has `df` (loaded fresh from the original CSV) and `pd` available in its exec scope. After execution it reads `local_vars["df"]`.

### Adding a New Agent

1. Create `source_code/agents/agent_N_<name>.py` following the existing pattern
2. Create `source_code/prompts/agent_N.json` with the prompt template
3. Add a node and edge in `source_code/graph.py`
4. Add any new state fields to `AgentState` in `source_code/state.py`
5. Populate the new state fields in `main.py` before building `initial_input`
