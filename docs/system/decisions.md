# Architectural Decision Records — Agnetic DS

Key decisions made during development, with rationale. Consult this before proposing changes to the stack or architecture.

---

## ADR-001 — LangGraph as orchestration framework
**Date:** 2025
**Status:** Accepted

**Decision:** Use LangGraph `StateGraph` to orchestrate all agents.

**Rationale:**
- Native `interrupt()` + `Command(resume=...)` for human-in-the-loop pauses
- Native checkpointing (SQLite/PostgreSQL) for multi-session, multi-client state persistence
- Streaming output (`ds_machine.stream()`) for real-time progress in the UI
- Clean `AgentState` TypedDict as the shared contract between all agents
- Conditional edges (needed for evaluation feedback loops)

**Alternatives considered:** Custom Python orchestration loop — rejected because it would require reimplementing checkpointing, streaming, and interrupt logic from scratch.

---

## ADR-002 — Groq + llama-3.3-70b-versatile as LLM
**Date:** 2025
**Status:** Accepted (revisit when agent count grows)

**Decision:** Use `llama-3.3-70b-versatile` via Groq API as the default LLM for all agents.

**Rationale:**
- Fast inference (important for a pipeline with 10-15 LLM calls)
- Free tier sufficient for development
- 8000 max tokens enough for code generation tasks

**Risks:** Single LLM provider dependency. If Groq has downtime, pipeline stops.

**LLM switching is fully supported** via env vars or code. See the guide below.

---

### How to Switch LLMs

#### Method 1 — Environment variables (no code changes)

Set these in your `.env` file (or shell) before running `main.py` or `app.py`:

```dotenv
LLM_PROVIDER=groq                        # groq | openai | anthropic | ollama
LLM_MODEL=llama-3.3-70b-versatile        # provider-specific model name
LLM_API_KEY=your-api-key                 # optional: overrides GROQ_API_KEY
LLM_BASE_URL=http://localhost:11434      # required for ollama only
```

**Provider + model examples:**

| Provider | `LLM_PROVIDER` | Example `LLM_MODEL` | Key env var |
|---|---|---|---|
| Groq (default) | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| OpenAI | `openai` | `gpt-4o`, `gpt-4o-mini` | `LLM_API_KEY` or `OPENAI_API_KEY`* |
| Anthropic | `anthropic` | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` | `LLM_API_KEY` or `ANTHROPIC_API_KEY`* |
| Ollama (local) | `ollama` | `llama3.2`, `mistral`, `qwen2.5-coder` | none (local) |

\* `LLM_API_KEY` is always checked first. If not set, each provider's native env var is used as a fallback by the LangChain package itself.

**Required pip installs per provider:**
```bash
# Groq (already installed)
pip install langchain-groq

# OpenAI
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# Ollama (local — also install Ollama from ollama.com)
pip install langchain-ollama
```

#### Method 2 — Code override (per-agent or global)

```python
from source_code.config.llm_config import LLMConfig, PipelineConfig
from source_code.graph import build_graph

# Global override — all agents use Claude
config = PipelineConfig(
    default_llm=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="your-anthropic-key",
    )
)

# Per-agent override — Agent 1 uses GPT-4o, Agent 2 uses Groq
config = PipelineConfig(
    default_llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile"),
    agent_overrides={
        "agent1": LLMConfig(provider="openai", model="gpt-4o", api_key="sk-..."),
    }
)

graph = build_graph(config)
```

#### Notes
- `LLMFactory.create()` is the single instantiation point. Adding a new LangChain-compatible provider requires changes only there.
- `temperature=0.0` is the default for all agents. The LLM re-executes the interrupted node on resume — temperature=0 ensures deterministic output so `ambiguous_fields`/`flagged_columns` rebuild identically.
- Ollama's `ChatOllama` does not accept `max_tokens`; use `num_predict` (not currently exposed — add to `LLMFactory` if needed).

---

## ADR-003 — LLM generates Python code; shared executor runs it
**Date:** 2025
**Status:** Accepted

**Decision:** Agents do not manipulate data directly. They generate Python code as a string. A shared `code_executor_agent` loads the df and runs the code via `exec()`.

**Rationale:**
- LLMs are better at writing code than producing structured data transformations
- Generated code is inspectable, loggable, and auditable
- Single executor means one place to add error handling, retries, sandboxing

**Risks:** `exec()` is a security risk if inputs are untrusted. Acceptable here because the LLM is the only code source and the system runs in a controlled environment (Docker, VPN).
**Future:** Consider sandboxed execution (e.g. subprocess with restricted globals) as the product matures.

---

## ADR-004 — Streamlit as the UI layer
**Date:** 2026
**Status:** Accepted (MVP phase)

**Decision:** Use Streamlit for the client-facing interactive interface.

**Rationale:**
- Python-native — no context switching between frontend and backend languages
- Built-in file upload, dataframe rendering, charting
- Runs as a web server accessible over VPN with no extra infrastructure
- Compatible with LangGraph streaming and interrupt pattern

**Migration path:** When the product matures to a multi-client SaaS product, migrate to FastAPI (backend) + React (frontend) for greater UI control. Streamlit is an intentional MVP choice, not a permanent one.

---

## ADR-005 — SQLite for dev, PostgreSQL for prod checkpointing
**Date:** 2026
**Status:** Accepted

**Decision:** Use LangGraph's SQLite checkpointer for local development and PostgreSQL in production (Docker on client server).

**Rationale:**
- SQLite: zero infrastructure, sufficient for single-developer local runs
- PostgreSQL: survives container restarts, supports concurrent clients, production-grade

---

## ADR-006 — 9-stage development workflow for all features
**Date:** 2026
**Status:** Accepted

**Decision:** All new features go through: Ideation → Brainstorm → Architecture → Audit → Code Plan → Code Writing → Testing → Documentation → Commit.

**Rationale:**
- Ensures domain-specific complexity is thought through before code is written
- Produces compressed transcripts (digests) that allow new LLM sessions to get up to speed with minimal context consumption
- Each stage runs in its own subagent context window — prevents context exhaustion
- Audit stage specifically catches conflicts with existing agents/state before they become bugs

---

## ADR-007 — docs/ as the persistent knowledge base
**Date:** 2026
**Status:** Accepted

**Decision:** All system knowledge lives in `docs/system/` and `docs/features/`. MEMORY.md is only an index/pointer file.

**Rationale:**
- LLM-agnostic: any AI assistant (Claude, GPT, Gemini) can read markdown files
- Git-tracked: knowledge evolves with the codebase
- Context-efficient: a new session reads only the digest of relevant features, not full conversation history

---

## ADR-008 — LLM abstraction via PipelineConfig + LLMFactory
**Date:** 2026-03-03
**Status:** Accepted

**Decision:** All LLM instantiation goes through `LLMFactory.create()`. Agent files never import a provider package directly. LLM selection is stored in `PipelineConfig` (a plain Python dataclass), which is constructed once per pipeline session and passed to `build_graph(config)`. `PipelineConfig` is never stored in `AgentState`.

**Rationale:**
- Before this ADR, each agent imported `ChatGroq` directly. Switching providers required touching every agent file.
- With `LLMFactory`, adding or changing a provider requires changes in exactly one place.
- `PipelineConfig.with_mock(mock)` provides a clean, one-line injection path for tests — no monkeypatching.
- `PipelineConfig.from_env()` provides a clean production path driven by environment variables.
- Deferred imports inside `LLMFactory.create()` branches mean only the installed provider's package is required at runtime.

**Alternatives considered:**
- Storing the LLM instance in `AgentState` — rejected because AgentState is a serializable TypedDict (required for LangGraph checkpointing); live LLM objects are not serializable.
- Per-agent factory functions that each read from environment — rejected because it scatters LLM config logic and makes per-agent overrides inconsistent.

**Per-agent override pattern:** `PipelineConfig(agent_overrides={"agent1": LLMConfig(provider="openai", model="gpt-4o")})`. `get_llm_config(agent_name)` returns the override if registered, otherwise `default_llm`.

---

## ADR-009 — Executor split: rename_executor_agent uses composite map, not exec()
**Date:** 2026-03-03
**Status:** Accepted

**Decision:** The single `code_executor_agent` is replaced by two distinct functions: `rename_executor_agent` (no exec) and `cleaning_executor_agent` (exec with safety checks).

**Rationale:**
- The rename step is deterministic and safe to express as a dict-based `df.rename(columns=composite_map)`. Running exec() for a rename is unnecessary and introduces a code injection surface with no benefit.
- Agent 1 produces a `column_map` audit dict (the canonical source of truth for what was renamed). The executor reconstructs the composite rename from this dict + `preprocess_column_names`. This means the rename is always reproducible from the audit trail, not from an exec'd code string.
- The cleaning step genuinely requires exec() because Agent 2 writes arbitrary pandas transformation code. The exec scope includes `pd`, `np`, and `__builtins__` to support numpy operations.
- Separating the functions makes each independently testable and clearly communicates intent.

**Safety checks in `cleaning_executor_agent`:** After exec, `_check_safety(df_before, df_after)` detects dropped columns (set difference) and new null values (sum difference). Warnings are appended to `error_log` — the pipeline does not abort. This is intentional: a future interrupt() hook will surface these warnings to the human operator.

**Data chain:** `rename_executor_agent` reads `state["file_path"]` (original CSV). `cleaning_executor_agent` reads `state["output_path"]` (post-rename CSV). The distinction is enforced in code and tested (TC19, TC20).

---

## ADR-010 — reclassify_columns_node as a separate graph node in source_code/reclassify.py
**Date:** 2026-03-03
**Status:** Accepted

**Decision:** Column reclassification (re-running `classify_columns`, `build_value_counts_summary`, `build_null_summary` on the post-rename DataFrame) is a dedicated graph node placed between `executor1` and `agent2_cleaner`. It lives in `source_code/reclassify.py`, not in `source_code/agents/`.

**Rationale:**
- Agent 2 needs column metadata computed from the post-rename DataFrame. Before this node existed, Agent 2 received metadata from `main.py` computed on the pre-rename DataFrame — column names would be wrong (still aliased, pre-standardization).
- Placing the reclassification inside Agent 2 would couple a pure data operation to the LLM call, making it untestable independently and harder to reason about.
- Placing it inside the executor would conflate two distinct responsibilities in one function.
- A dedicated graph node makes the dependency explicit in the graph topology and independently observable in the LangGraph stream output.
- `source_code/reclassify.py` (not `agents/`) signals that this module performs no LLM call and holds no prompt. The `agents/` directory is reserved for LLM-driven components.

**Fallback path:** If `state["output_path"]` is absent, the fallback is `"standardized_output_renamed.csv"` — matching `rename_executor_agent`'s default output path. If that file does not exist, `pd.read_csv()` raises `FileNotFoundError` immediately (fail-fast; no silent data corruption).

---

## ADR-011 — Single-file Streamlit app (`app.py`) as the UI MVP
**Date:** 2026-03-03
**Status:** Accepted

**Decision:** The Streamlit UI is implemented as a single file `app.py` at the root of the repository. All session state, streaming logic, and interrupt form rendering live in this file.

**Rationale:**
- The planned multi-file structure (`app/pages/`, `app/components/`) is appropriate for a mature multi-agent pipeline. At MVP stage (2 agents, 2 interrupt types) it would be over-engineering.
- A single file is easier to iterate on: all state transitions, interrupt payloads, and UI rendering are visible in one place.
- The component decomposition plan is preserved in `architecture.md` for when the pipeline grows beyond 5 agents.

**Key implementation decisions inside `app.py`:**
- `MemorySaver` (in-memory checkpointer) is created fresh per run with a new `uuid4` thread ID. This means state does not survive an app restart — acceptable for the current CLI-adjacent usage model.
- All uploaded/pasted inputs (CSV, SQL, rules) are written to `tempfile.NamedTemporaryFile(delete=False)` and paths stored in `st.session_state.temp_files`. They are deleted on "Start New Run".
- `build_graph(config, checkpointer=checkpointer)` — the `checkpointer` param was added to `graph.py` with a `None` default so `main.py` requires no changes.
- On interrupt resume, `Command(resume=value)` causes LangGraph to re-execute the interrupted node from the top. At `temperature=0.0` the LLM returns the same response, so ambiguous fields rebuild identically. The user's choices are applied via `column_map.update(user_decisions)` at the interrupt point.

**Migration path:** When moving to multi-file, extract `_run_stream` → `state/session.py`, interrupt renderers → `components/interrupt_panel.py`, progress tracker → `components/progress_tracker.py`.
