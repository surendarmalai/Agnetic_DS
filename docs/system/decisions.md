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

**Decision:** Use `llama-3.3-70b-versatile` via Groq API for all agents.

**Rationale:**
- Fast inference (important for a pipeline with 10-15 LLM calls)
- Free tier sufficient for development
- 8000 max tokens enough for code generation tasks

**Risks:** Single LLM provider dependency. If Groq has downtime, pipeline stops.
**Future:** Abstract LLM instantiation so any LangChain-compatible LLM can be swapped in per agent.

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
