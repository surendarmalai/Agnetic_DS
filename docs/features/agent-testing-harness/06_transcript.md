# Feature Digest — Agent Testing Harness + LLM Abstraction Layer

**Feature branch:** `fix/agent-testing`
**Stage:** 8 — Documenter
**Date:** 2026-03-03
**Status:** COMPLETE — 16 PASSED, 4 WARNING (xfail gaps), 3 XPASSED

---

## What Was Built and Why

Agents 1 and 2 were implemented but completely untested. The pipeline was also hardcoded to a single LLM provider (Groq / `llama-3.3-70b-versatile`) via direct `ChatGroq` imports inside each agent file — making provider switching require code changes across multiple files.

This feature delivered two tightly coupled changes: a **LLM abstraction layer** that decouples LLM instantiation from agent logic, and a **pytest-based testing harness** that uses injected mock LLMs to run all edge-case scenarios without real API calls. It also fixed a structural defect discovered during audit: the executor was a single function that could not distinguish between rename and cleaning responsibilities, and column metadata sent to Agent 2 was computed on the pre-rename DataFrame rather than the post-rename one.

---

## Key Architectural Changes

### Before this feature

- `agent_1_field_renamer.py` and `agent_2_field_cleaner.py` each imported `ChatGroq` directly and held a module-level agent function.
- `executor.py` had a single `code_executor_agent` function reused for both executor nodes; it called `exec(cleaning_code)` regardless of which stage it served.
- `graph.py` exposed a module-level `ds_machine` compiled graph; the graph was built at import time with no way to inject a config or mock.
- `main.py` read the SQL query and special rules directly with inline file I/O; there was no config abstraction layer.
- The 4-node graph had no step between executor1 and agent2; Agent 2 received column metadata computed from the pre-rename DataFrame.
- Column classification metadata passed to Agent 2 was stale (pre-rename column names).

### After this feature

- Agents are produced by factory functions (`make_field_renamer_agent(config)`, `make_field_cleaner_agent(config)`) that close over a `PipelineConfig`. No module-level agent functions exist.
- `executor.py` has two distinct functions: `rename_executor_agent` (applies composite rename map, no `exec()`) and `cleaning_executor_agent` (runs cleaning code via `exec()` with safety checks).
- `graph.py` exposes `build_graph(config: PipelineConfig)` — a function, not a module-level object.
- `source_code/config/` package provides `PipelineConfig`, `LLMConfig`, `LLMFactory`, and `load_pipeline_inputs`.
- The graph is now 5 nodes with a `reclassify_columns` node between `executor1` and `agent2_cleaner`.
- Tests inject mock LLMs via `PipelineConfig.with_mock(MockLLM(...))` — no network calls in tests.

---

## New Graph Topology

```
agent1_renamer -> executor1 -> reclassify_columns -> agent2_cleaner -> executor2 -> END
```

| Node | Function | Purpose |
|------|----------|---------|
| `agent1_renamer` | `make_field_renamer_agent(config)` | LLM call — produces `column_map` |
| `executor1` | `rename_executor_agent` | Applies composite rename map (no exec) |
| `reclassify_columns` | `reclassify_columns_node` | Re-classifies columns on post-rename CSV |
| `agent2_cleaner` | `make_field_cleaner_agent(config)` | LLM call — produces `cleaning_code` |
| `executor2` | `cleaning_executor_agent` | Runs cleaning code via exec() |

---

## New Files Created

| File | Purpose |
|------|---------|
| `source_code/__init__.py` | Marks `source_code/` as a Python package |
| `source_code/agents/__init__.py` | Marks `source_code/agents/` as a Python package |
| `source_code/reclassify.py` | `reclassify_columns_node` — graph node that re-classifies columns after rename |
| `source_code/config/__init__.py` | Re-exports `PipelineConfig`, `LLMConfig`, `LLMFactory` |
| `source_code/config/llm_config.py` | `LLMConfig` and `PipelineConfig` dataclasses |
| `source_code/config/llm_factory.py` | `LLMFactory.create()` — single point for all LLM instantiation |
| `source_code/config/loaders.py` | `load_pipeline_inputs`, `load_client_config`, `_build_initial_input` |
| `requirements-dev.txt` | Dev dependencies: `pytest>=7.0`, `pytest-json-report>=1.5` |
| `pytest.ini` | Registers custom markers: `agent1`, `agent2`, `executor` |
| `tests/__init__.py` | Package marker |
| `tests/mock_llm.py` | `MockLLM(BaseChatModel)` and `MockResponse` — test doubles |
| `tests/run_tests.py` | CLI test runner; generates `tests/reports/review_<timestamp>.md` |
| `tests/agents/__init__.py` | Package marker |
| `tests/agents/test_agent_1.py` | 9 test cases (TC1–TC9) for Agent 1 |
| `tests/agents/test_agent_2.py` | 8 test cases (TC10–TC17) for Agent 2 |
| `tests/agents/test_executor.py` | 6 test cases (TC18–TC23) for executor |
| `tests/fixtures/agent_1/scenarios.json` | Agent 1 fixture scenarios with `expected_assertions` dict |
| `tests/fixtures/agent_2/scenarios.json` | Agent 2 fixture scenarios with `expected_assertions` dict |
| `tests/fixtures/sample_telecom.csv` | 50-row, 10-column test dataset with SQL alias prefixes and dirty values |
| `tests/reports/.gitkeep` | Ensures reports directory is tracked by git |

---

## Modified Files

| File | What Changed |
|------|-------------|
| `source_code/agents/agent_1_field_renamer.py` | Removed `ChatGroq` import; wrapped agent in `make_field_renamer_agent(config)` factory closure; added `exec(..., {"__builtins__": __builtins__}, local_ns)` form for both code block parses |
| `source_code/agents/agent_2_field_cleaner.py` | Same pattern: removed `ChatGroq` import; wrapped in `make_field_cleaner_agent(config)` factory closure |
| `source_code/agents/executor.py` | Full rewrite: `code_executor_agent` removed; replaced with `rename_executor_agent`, `cleaning_executor_agent`, and `_check_safety` helper |
| `source_code/graph.py` | Full rewrite: module-level `ds_machine` removed; `build_graph(config)` function added; 5-node topology wired |
| `main.py` | Replaced inline file I/O with `load_pipeline_inputs`; replaced `ds_machine` import with `build_graph`; added `PipelineConfig.from_env()` call |

---

## How to Inject a Mock LLM in Tests

The standard pattern used in all 23 test cases:

```python
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig

mock = MockLLM(content=SCENARIOS["tc1"]["llm_response"])
config = PipelineConfig.with_mock(mock)
agent_fn = make_field_renamer_agent(config)   # or make_field_cleaner_agent
result = agent_fn(state)
```

`PipelineConfig.with_mock(mock)` creates a `PipelineConfig` whose `default_llm.llm_instance` is the mock. When `LLMFactory.create()` is called inside the agent factory, it detects `llm_instance is not None` and returns it directly — no provider import, no API call.

`MockLLM` inherits `BaseChatModel`, implements `_generate` returning a `ChatResult(AIMessage)`, and records the last prompt in `mock.last_prompt`. After `agent_fn(state)` returns, tests can assert on both the state output and on `mock.last_prompt` to verify that SQL query context was included in the prompt.

---

## Executor Design: Why Two Functions

### `rename_executor_agent` (no exec)

The rename step is deterministic: it builds a composite rename map from `preprocess_column_names(df.columns)` + `state["column_map"]` and calls `df.rename(columns=composite_map)`. This is not `exec()`. The audit found that running `exec(cleaning_code)` in the rename stage was incorrect — Agent 1 produces a `column_map` dict (an audit trail), not executable rename code. The executor reconstructs the rename itself.

Reads from: `state["file_path"]` (original raw CSV).
Writes to: `state["output_path"]` (default: `"standardized_output_renamed.csv"`).

### `cleaning_executor_agent` (exec with safety checks)

The cleaning step requires exec() because Agent 2 produces arbitrary cleaning code. The exec scope is `{"__builtins__": __builtins__, "pd": pd, "np": np}` so generated code can import numpy functions. After exec, `_check_safety(df_before, df_after)` checks for dropped columns and new null values — warnings are written to `state["error_log"]` but do not raise exceptions (the pipeline continues).

Reads from: `state["output_path"]` (post-rename CSV, set by executor1).
Writes to: `"output_agent2_cleaned.csv"` (hardcoded final path).

---

## Why `reclassify_columns_node` Exists

Agent 2 needs `categorical_cols`, `numeric_like_cols`, `true_numeric_cols`, `value_counts_summary`, and `null_summary` computed on the post-rename DataFrame. Before this feature these values were computed in `main.py` on the pre-rename DataFrame and never refreshed. After rename, column names change, so any value-count summaries keyed by column name would be stale.

`reclassify_columns_node` reads the post-rename CSV from `state["output_path"]` and re-runs `classify_columns`, `build_value_counts_summary`, and `build_null_summary`. It lives in `source_code/reclassify.py` (not `source_code/agents/`) because it performs no LLM call and has no associated prompt — it is infrastructure, not an agent.

---

## `reclassify_columns_node` placement: `source_code/reclassify.py`

Not in `source_code/agents/` because it is not an agent (no LLM call, no prompt). Placing it in `agents/` would misrepresent it as an LLM-driven component.

---

## Test Suite Summary

**Run command:** `python tests/run_tests.py`
**Filter by agent:** `python tests/run_tests.py --agent 1`
**Report output:** `tests/reports/review_<timestamp>.md`

| Test file | TCs | Pass | xfail (gap) | xpassed |
|-----------|-----|------|-------------|---------|
| `test_agent_1.py` | TC1–TC9 (9) | 6 | 3 (TC5, TC6, TC8) | 0 |
| `test_agent_2.py` | TC10–TC17 (8) | 6 | 1 (TC16) | 1 (TC15) |
| `test_executor.py` | TC18–TC23 (6) | 4 | 0 | 2 (TC21, TC22) |
| **Total** | **23** | **16** | **4** | **3** |

Overall verdict: **PASS**. No unexpected failures.

---

## Known Limitations (xfail Gaps)

These 4 tests are marked `@pytest.mark.xfail(strict=False)` — they document known deficiencies and appear as WARNING in the test report. They are not regressions.

| TC | Location | Gap |
|----|----------|-----|
| TC5 | Agent 1 | Duplicate values in `rename_map` (two raw columns mapped to the same clean name) are not detected by Agent 1 or the executor. The duplicate silently overwrites one column. |
| TC6 | Agent 1 | A partial `rename_map` (not all columns covered) is silently accepted. No warning is generated for unmapped columns. |
| TC8 | Agent 1 | `rename_map` keys that do not match any actual DataFrame column are not validated. The executor silently skips them. |
| TC16 | Agent 2 | Agent 2 does not detect when its generated cleaning code would drop a column. The agent returns the code without flagging it in `flagged_columns`. |

---

## Surprising Results (xpassed)

These 3 tests were expected to fail but passed — the implementation exceeded documented expectations.

| TC | Location | Finding |
|----|----------|---------|
| TC15 | Agent 2 | Agent 2 was expected not to detect null introduction. The executor's `_check_safety` catches it post-exec, so the warning does surface — via `error_log`, not `flagged_columns`. |
| TC21 | Executor | `_check_safety` correctly detected column drops in `cleaning_executor_agent` and appended a WARNING to `error_log`. |
| TC22 | Executor | `_check_safety` correctly detected new null values introduced by cleaning code and appended a WARNING to `error_log`. |

The xpassed results confirm that `_check_safety` works as designed. The distinction is: Agent 2 itself does not detect these issues (TC15, TC16 gaps remain), but the executor catches them after the fact.

---

## `LLMConfig` and `PipelineConfig` Reference

```python
# Build from environment (production path)
config = PipelineConfig.from_env()

# Build with mock for tests
config = PipelineConfig.with_mock(MockLLM(content="..."))

# Per-agent override
config = PipelineConfig(
    default_llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile"),
    agent_overrides={"agent1": LLMConfig(provider="openai", model="gpt-4o")},
)
config.get_llm_config("agent1")   # returns the OpenAI override
config.get_llm_config("agent2")   # returns the Groq default
```

`PipelineConfig` is never stored in `AgentState`. It is constructed once in `main.py` (or test setup) and passed to `build_graph(config)`, which calls the agent factories at compile time.

---

## `LLMFactory` Supported Providers

| Provider string | LangChain class | Package required |
|----------------|----------------|-----------------|
| `"groq"` | `ChatGroq` | `langchain-groq` |
| `"openai"` | `ChatOpenAI` | `langchain-openai` |
| `"anthropic"` | `ChatAnthropic` | `langchain-anthropic` |
| `"ollama"` | `ChatOllama` | `langchain-ollama` |

All provider imports are deferred inside if-branches. Only the installed package needs to be present. Unknown providers raise `ValueError("Unknown LLM provider: '...'. Supported: groq, openai, anthropic, ollama")`.

---

## What Is Next

- Agents 3–11 are planned (see `docs/system/agent_registry.md`).
- The harness is designed to accommodate new agents: add `tests/agents/test_agent_N.py`, add `tests/fixtures/agent_N/scenarios.json`, register a new marker in `pytest.ini`.
- `interrupt()` calls are not yet implemented. Agents surface `ambiguous_fields` and `flagged_columns` in state but the pipeline does not pause for human input.
- LangGraph checkpointing is not yet implemented.
- Streamlit UI is not yet implemented.
- The cleaning executor's output path (`"output_agent2_cleaned.csv"`) is hardcoded. A future improvement is to read a configurable output path from state.
