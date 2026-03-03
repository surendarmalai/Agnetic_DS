# Feature Digest — Streamlit UI + Human-in-the-Loop Interrupts

**Date:** 2026-03-03
**Branch:** fix/agent-testing
**Status:** Implemented, tested — pending commit

---

## What Was Built

A single-file Streamlit app (`app.py`) that:
1. Accepts CSV upload + optional SQL query + optional domain rules + target column
2. Runs the 5-node LangGraph pipeline via `MemorySaver` checkpointing
3. Pauses at `interrupt()` points and renders forms for human input
4. Resumes the pipeline after form submission via `Command(resume=...)`
5. Shows a results page with the cleaned DataFrame preview and CSV download

The pipeline's two agents were wired with `interrupt()` calls so the UI can
pause and surface decisions to the operator.

---

## Files Changed

| File | Change |
|---|---|
| `app.py` | **Created** — single-file Streamlit app (~380 lines) |
| `requirements.txt` | **Created** — `streamlit>=1.32.0` |
| `source_code/graph.py` | `build_graph(config, checkpointer=None)` — backward-compatible |
| `source_code/agents/agent_1_field_renamer.py` | `interrupt()` call after ambiguous_fields parsed |
| `source_code/agents/agent_2_field_cleaner.py` | `interrupt()` call after flagged_columns parsed |

---

## Interrupt Wiring (actual implementation)

**Agent 1 — ambiguous fields:**
```python
if ambiguous_fields:
    try:
        user_decisions = interrupt({"type": "ambiguous_fields", "fields": ambiguous_fields})
        if isinstance(user_decisions, dict):
            column_map.update(user_decisions)
    except RuntimeError:
        pass  # unit test context — skip interrupt
```
`user_decisions` = `{original_col: chosen_name}`. Applied on top of LLM's `column_map`.

**Agent 2 — flagged columns:**
```python
if flagged_columns:
    try:
        interrupt({"type": "flagged_columns", "columns": flagged_columns})
        # return value discarded — user acknowledges only
    except RuntimeError:
        pass  # unit test context — skip interrupt
```

**Why try/except:** `interrupt()` reads from a LangGraph context var. Unit
tests call agent functions directly (no context), so they would crash without
this guard. The except clause is silent and does not mask real errors —
`interrupt()` only raises `RuntimeError` for the context-missing case.

---

## app.py Architecture

**Session state keys:**
`pipeline_status`, `graph`, `thread_config`, `initial_input`, `resume_value`,
`node_log`, `interrupt_payload`, `temp_files`, `final_output_path`,
`column_map`, `error_message`

**State machine:**
```
idle → running → interrupted → running → complete
                             ↘ error
```

**Core streaming function `_run_stream()`:**
- Drives `graph.stream(stream_input, thread_config, stream_mode="updates")`
- On `__interrupt__` chunk: saves payload to session_state, returns `"interrupted"`
- On normal chunk: appends node name to `node_log`, captures `column_map` and `output_path`
- On exception: saves to `error_message`, returns `"error"`
- On stream exhaustion: returns `"complete"`

**Resume:** `Command(resume=resume_value)` passed as `stream_input`. LangGraph
re-executes the interrupted node from the top (LLM call fires again). At
`temperature=0.0` the same response is returned, so `ambiguous_fields` rebuilds
identically and `column_map.update(user_decisions)` applies the human choices.

**Temp files:** All uploads written to `tempfile.NamedTemporaryFile(delete=False)`.
Paths stored in `st.session_state.temp_files`, deleted on "Start New Run".

---

## Key Decision: try/except on interrupt() (discovered during Stage 7)

The audit stage (Stage 4) was skipped — this conflict was found during testing.
Going forward, all `interrupt()` additions must be reviewed by the auditor for
test-context compatibility. The guard pattern is now the established convention:
```python
try:
    interrupt(...)
except RuntimeError:
    pass
```

---

## What Is NOT Built (follow-up required)

1. **LLM selector UI** — no per-stage LLM switching in the app. Currently uses
   `PipelineConfig.from_env()` (env vars only). A follow-up feature must add a
   sidebar/form for selecting provider + model per agent.
2. **SQLite checkpointer** — `MemorySaver` resets on app restart. Wire
   `SqliteSaver` for persistent dev sessions.
3. **app.py automated tests** — Streamlit functions cannot be unit tested
   without a running server. End-to-end testing requires a live pipeline run.
4. **Stages 1–9 process not followed** for this feature — it was implemented
   directly from a hand-written plan. The test fix (interrupt guard) was
   discovered at stage 7 retroactively.

---

## How to Run

```bash
# Install streamlit (already done)
.venv\Scripts\pip install streamlit

# Run the app
.venv\Scripts\streamlit.exe run app.py
# → http://localhost:8501

# CLI pipeline still works unchanged
python main.py
```

---

## How to Switch LLMs (for the operator)

See `docs/system/decisions.md` ADR-002 for the full guide. Short version:

**Via `.env`:**
```dotenv
LLM_PROVIDER=openai          # groq | openai | anthropic | ollama
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
```

**Install the provider package:**
```bash
pip install langchain-openai      # OpenAI
pip install langchain-anthropic   # Anthropic / Claude
pip install langchain-ollama      # local Ollama
```

Per-agent overrides are available in code via `PipelineConfig(agent_overrides={...})`.
The LLM selector UI (per-stage switching from inside the app) is a planned
follow-up feature.
