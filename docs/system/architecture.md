# System Architecture — Agnetic DS

## What This System Is
An automated, interactive data science machine for telecom clients.
Models are delivered as a **product** (reproducible, fully automated) not a service (manual DS work per engagement).
A telecom DS employee operates the system — they are always in the loop to inject domain knowledge the LLM cannot know on its own.

## Core Design Principles
1. **Domain-aware via human guidance** — LLMs handle orchestration and code generation; humans inject telecom-specific rules, churn definitions, client context
2. **Human-in-the-loop at every major decision point** — the system pauses, surfaces findings, and waits for direction before continuing
3. **Context injection at any stage** — users can upload files (data dictionaries, business rules, schema docs) or type instructions at any pause point
4. **Multi-client** — same pipeline, different thread per client; each client's state is isolated and persisted
5. **Agent modularity** — each agent does one job, reads from shared state, writes back to shared state; agents are independently testable

---

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Orchestration | LangGraph (`StateGraph`) | Native support for interrupts, checkpointing, streaming |
| LLM | `llama-3.3-70b-versatile` via Groq | Fast, free tier available, sufficient for code generation |
| UI | Streamlit | Python-native, handles file uploads, renders data/charts, works over VPN |
| State persistence | SQLite (dev) → PostgreSQL (prod) | LangGraph checkpointer; survives container restarts in prod |
| Deployment | Docker container on client server | VPN tunnel for browser access; CLI access via SSH |
| Language | Python 3.11+ | |

---

## High-Level Data Flow

```
User (Streamlit UI)
  │
  ├── Uploads context files (data dict, business rules)
  ├── Defines churn criteria, client-specific rules
  │
  ▼
LangGraph Graph (ds_machine)
  │
  ├── agent_query_builder     → writes SQL
  ├── executor                → runs SQL, loads raw df
  │     └── [INTERRUPT] → user validates data pull
  ├── agent_1_field_renamer   → standardizes column names
  │     └── [INTERRUPT] → user resolves ambiguous columns
  ├── executor
  ├── agent_2_field_cleaner   → cleans dirty values, fixes types
  │     └── [INTERRUPT] → user reviews flagged columns
  ├── executor
  ├── agent_feature_engineer  → builds model-ready features
  │     └── [INTERRUPT] → user suggests additional features
  ├── executor
  ├── agent_eda               → surfaces distributions, correlations, data issues
  │     └── [INTERRUPT] → user reviews, approves to proceed
  ├── agent_model_builder     → trains multiple algorithms
  ├── agent_evaluator         → computes AUC, KS, Gini, precision/recall
  │     └── [INTERRUPT] → user reviews metrics, decides next step
  ├── [CONDITIONAL] agent_data_augmentor → re-queries with broader criteria
  ├── agent_hypertuner        → optimises best model
  ├── agent_comparator        → compares candidates, selects winner
  └── agent_reporter          → generates model card + performance report
        └── [INTERRUPT] → user reviews final report before delivery
```

---

## Human-in-the-Loop Architecture

Uses LangGraph's `interrupt()` + `Command(resume=...)` pattern.

**Interrupt payload structure (actual, as implemented):**

| Agent | `type` key | Payload key | Content |
|---|---|---|---|
| Agent 1 | `"ambiguous_fields"` | `"fields"` | `[{original_column, candidates, reason, sample_values}]` |
| Agent 2 | `"flagged_columns"` | `"columns"` | `[{column, reason}]` |

**Planned types (future agents):**
- `"query_approval"`, `"feature_suggestion"`, `"eval_review"`, `"report_review"`

`app.py` reads `interrupt_payload["type"]` and dispatches to the appropriate render function.

**Resuming the graph:**
```python
ds_machine.invoke(Command(resume=user_feedback), config={"configurable": {"thread_id": client_id}})
```

---

## Multi-Client / Session Management

- Each client run = unique `thread_id`
- LangGraph checkpointer persists state to DB keyed by `thread_id`
- Multiple clients can have pipelines running concurrently
- A pipeline can be paused (waiting for human input) for days and resumed without data loss

---

## Streamlit App Structure

**Current (MVP — single file):**
```
app.py                       ← Single-file Streamlit app (root level)
```

Run: `.venv\Scripts\streamlit.exe run app.py` → http://localhost:8501

**Pages (sidebar radio):**
- **Run Pipeline** — CSV upload, pipeline execution, interrupt forms, results + download
- **Database** — placeholder (coming soon)

**Session state machine:**
```
idle → running → interrupted → running → complete
                             → error
```

**Interrupt forms rendered by `app.py`:**
- `ambiguous_fields` → `_render_ambiguous_fields_form()` — selectbox + custom text input per flagged column
- `flagged_columns`  → `_render_flagged_columns_form()` — acknowledgment only, pipeline continues

**Checkpointing:** `MemorySaver` (in-memory, resets on app restart). Future: `SqliteSaver` for dev, `PostgresSaver` for prod.

**Planned multi-file structure (future):**
```
app/
  main.py                    ← Entry point, routing
  pages/
    01_pipeline.py           ← Main pipeline runner + interrupt handler
    02_history.py            ← Past runs per client, audit trail
    03_results.py            ← Model metrics, charts, comparison
  components/
    interrupt_panel.py       ← Generic interrupt renderer (all pause types)
    file_uploader.py         ← Context injection via file upload
    progress_tracker.py      ← Agent progress (queued / running / done / flagged)
  state/
    session.py               ← Streamlit session state management
```

---

## Development Workflow (9-Stage)

All new features follow this pipeline before merging to `main`:

| Stage | Role | Output file |
|---|---|---|
| 1 | Feature Ideation (interactive) | `00_feature_brief.md` |
| 2 | Brainstormer (subagent) | `01_brainstorm.md` |
| 3 | Architect (subagent) | `02_architecture.md` |
| 4 | Auditor (subagent) | `03_audit.md` |
| 5 | Code Planner (subagent) | `04_code_plan.md` |
| 6 | Code Writer (subagent) | implemented code |
| 7 | Tester (subagent) | `05_test_report.md` |
| 8 | Documenter (subagent) | `06_transcript.md` ← the digest |
| 9 | Committer (subagent) | git commit on feature branch |

Each subagent runs in its own context window. Only the digest (`06_transcript.md`) is passed forward to the next session.

---

## Branching Strategy

```
main                    ← stable, tested, production-ready
  └── fix/agent-testing ← current: testing harness + fixing agents 1 & 2
  └── feature/<name>    ← one branch per feature, opened after agents 1 & 2 are solid
```
