# Feature Brief — Agent Testing Harness + LLM Abstraction Layer

## What This Feature Is
Two tightly coupled changes that must be built together:

1. **LLM Abstraction Layer** — a dynamic, swappable LLM configuration system so the pipeline is not hardcoded to Groq. The user selects which LLM to use at the start of each data science session, with optional per-agent overrides mid-session.

2. **Agent Testing Harness** — a fully automated test suite using mock LLM responses, covering all edge cases for every agent. Produces a single markdown review document per run with pass/fail results, diagnostics, and fix suggestions for any failures.

---

## Problem Being Solved
- Agents 1 & 2 are implemented but completely untested. We have no confidence in their output.
- The pipeline is hardcoded to `llama-3.3-70b-versatile` via Groq. Switching LLMs requires code changes across multiple agent files — not acceptable for a product that may run on different client environments.
- There is no way to validate that agent outputs are structurally correct, that the generated code runs safely, or that the SQL query context is actually being used by the agents.
- The query/rules input structure (hardcoded file paths in main.py) is not appropriate for a multi-client, multi-session product.

---

## Scope

### 1. LLM Abstraction Layer
- A single `LLMFactory` that instantiates any supported LLM given a provider name + model name + API key
- Supported providers: **Groq**, **OpenAI**, **Anthropic**, **Ollama** (local)
- LLM selection happens at pipeline initialisation — stored in `AgentState` or config so all agents use the same instance
- Optional per-agent override: e.g. use GPT-4 for Agent 1 but llama for everything else
- Config can be set via: environment variables (default) or at runtime (future: Streamlit settings panel)
- Adding a new provider must require changes in only ONE place (the factory)

### 2. Mock LLM System
- A `MockLLM` class that replays pre-saved response fixtures instead of calling an API
- Fixtures stored as JSON files in `tests/fixtures/agent_N/`
- Multiple fixtures per agent: happy path + each edge case variant
- Test runner selects the appropriate fixture per test case
- Fixtures must be realistic (valid responses the real LLM would produce)

### 3. Test Cases — Agent 1 (Field Renamer)
- ✅ Happy path: valid 2-block response, all columns covered in rename_map, clean ambiguous_fields list
- ✅ SQL query context: verify columns mentioned in the SQL query are correctly resolved (not treated as unknown)
- ✅ Missing second code block: only rename code returned, no ambiguous_fields block
- ✅ No code blocks returned at all: agent handles gracefully, returns empty cleaning_code
- ✅ Duplicate values in rename_map: two raw columns mapped to the same clean name
- ✅ Not all df_columns covered in rename_map: some columns missing from the map
- ✅ ambiguous_fields missing required keys: `original_column`, `candidates`, `reason`
- ✅ rename_map keys don't match actual df columns: executor would fail silently
- ✅ Malformed Python in code block: exec() would raise SyntaxError

### 4. Test Cases — Agent 2 (Field Cleaner)
- ✅ Happy path: valid cleaning code + valid flagged_columns list
- ✅ SQL query context: verify agent uses column context from the query
- ✅ Missing second code block: no flagged_columns block returned
- ✅ No code blocks returned at all: graceful handling
- ✅ flagged_columns missing required keys: `column`, `reason`
- ✅ Cleaning code introduces new nulls: catches unsafe cleaning
- ✅ Cleaning code drops columns: catches destructive operations
- ✅ Malformed Python in cleaning code block

### 5. Test Cases — Executor (Shared)
- ✅ Happy path: code runs, df saved, no error_log
- ✅ Empty cleaning_code: returns error_log immediately, no exec() attempt
- ✅ Code raises runtime exception: error captured in error_log, not re-raised
- ✅ Code drops a column from df: detected post-exec
- ✅ Code introduces new nulls: detected post-exec
- ✅ Output file written correctly: file exists and is a valid CSV post-exec

### 6. Test Runner & Review Document
- Single command to run all tests: `python tests/run_tests.py`
- Optionally filter by agent: `python tests/run_tests.py --agent 1`
- Output: `tests/reports/review_<timestamp>.md`
- Review document format per test:
  - Test name, agent, category
  - Pass ✅ / Fail ❌ / Warning ⚠️
  - On failure: what was expected vs what was received, probable cause, suggested fix
- Summary section at top: total pass/fail/warning counts

### 7. Input Structure Refactor
- Move query and special_rules loading out of `main.py` into a dedicated `source_code/config/` module
- Structure must support multiple clients (each client may have their own query + rules)
- For now: keep using files (`queries/`, `rules/`) but load them through the config module
- The config module is what will later be replaced by a Streamlit settings panel or DB-backed config

---

## Acceptance Criteria
- [ ] `LLMFactory` instantiates correct LLM given provider string; raises clear error for unknown providers
- [ ] `MockLLM` replays fixtures correctly; tests do not make real API calls
- [ ] All test cases listed above have implementations and run via `python tests/run_tests.py`
- [ ] Review document is generated at `tests/reports/review_<timestamp>.md`
- [ ] Failures include diagnosis and fix suggestions (not just "FAILED")
- [ ] `AgentState` does not grow new fields unnecessarily (LLM config stored separately)
- [ ] Existing `main.py` pipeline still runs end-to-end after the refactor
- [ ] Input config refactor does not break existing file structure

---

## What This Feature Does NOT Include
- Streamlit UI (separate feature)
- LangGraph interrupt() implementation (separate feature)
- LangGraph checkpointing (separate feature)
- Real LLM call testing (out of scope for now — mock only)
- Testing Agents 3–11 (they don't exist yet; harness is designed to accommodate them when they do)

---

## Related Docs
- `docs/system/architecture.md`
- `docs/system/agent_registry.md`
- `docs/system/decisions.md`
