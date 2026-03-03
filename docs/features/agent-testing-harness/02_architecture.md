# Architecture — Agent Testing Harness + LLM Abstraction Layer

**Feature:** `agent-testing-harness`
**Author stage:** Architect (Stage 3)
**Input:** `00_feature_brief.md`, `01_brainstorm.md`, source files: state.py, graph.py, agent_1_field_renamer.py, agent_2_field_cleaner.py, executor.py, main.py

---

## 0. Scope of This Document

This document is the authoritative design specification for the implementer. It resolves all open questions from the Brainstorm, provides precise signatures and schemas, and makes binding decisions on every contested design point.

---

## 1. Bug Fixes — Three Critical Bugs to Fix

### Bug 1: Alias-Prefix Mismatch (Silent Rename Failure)

**Root cause.** `preprocess_column_names` strips alias prefixes (e.g. `"a.msisdn"` → `"msisdn"`) and produces a cleaned list that is injected into the LLM prompt. The LLM generates `rename_map` keys using the cleaned (prefix-stripped) names. However, the actual DataFrame that the Executor loads from disk still has the original column names (with alias prefixes). `df.rename(columns=rename_map)` silently does nothing because none of its keys match the real column names.

**Decision: Fix in the Executor, not in Agent 1.**

Rationale: Agent 1's responsibility is semantic mapping (raw-cleaned → standardised). The Executor's responsibility is applying transformations to the real DataFrame. The Executor knows both the real column names and the `preprocess_column_names` utility. Composing the two mappings at execution time is a clean separation: Agent 1 produces clean-to-standardised, Executor resolves original-to-clean, Executor applies original-to-standardised.

**Exact fix in `executor.py`.**

Before calling `df.rename(columns=rename_map)`, the Executor must:
1. Call `preprocess_column_names(list(df.columns))` to get `{original_name: cleaned_name}`.
2. Build a composite map: `{orig: rename_map[cleaned] for orig, cleaned in pre_cleaned.items() if cleaned in rename_map}`.
3. Call `df.rename(columns=composite_map)`.

```python
# executor.py — only the rename path changes
from source_code.utils import preprocess_column_names

# Inside code_executor_agent, after exec() runs:
# For executor1 specifically (post-Agent-1 rename), the executor builds the composite map.
# The composite map is built from column_map in state (which is the LLM's rename_map).
column_map = state.get("column_map", {})
if column_map:
    pre_cleaned = preprocess_column_names(list(df.columns))
    composite_map = {
        orig: column_map[cleaned]
        for orig, cleaned in pre_cleaned.items()
        if cleaned in column_map
    }
    df = df.rename(columns=composite_map)
```

Note: the Executor's `exec(cleaning_code, ...)` call remains for Agent 2's cleaning code. The rename step above is explicitly for Agent 1's output, applied AFTER `exec()` or instead of it for the renaming executor. See Section 7 for the full Executor refactor.

**Impact on column_map.** Agent 1 continues to return `column_map` keyed by cleaned names (what the LLM was given). The Executor reconstructs the full original-to-standardised mapping internally. No change to `AgentState` schema.

---

### Bug 2: Stale Column Classification (Agent 2 Silent No-Op)

**Root cause.** `main.py` calls `classify_columns(df)`, `build_value_counts_summary()`, and `build_null_summary()` on the raw DataFrame BEFORE Agent 1 runs. After Agent 1 renames columns and Executor 1 applies the renames, the columns in the DataFrame have new standardised names. But `categorical_cols`, `numeric_like_cols`, and `value_counts_summary` in state still contain the old pre-rename names. Agent 2 is told to clean columns like `"gender"` but the real DataFrame has `"Gender"` — every `if col in df.columns` guard fails silently.

**Decision: Reclassify in the graph as a dedicated node between executor1 and agent2.**

Rationale: the Executor should remain a pure code-execution node. Pushing reclassification logic into the Executor would make it context-aware of which agent's output it is processing, violating single responsibility. A dedicated reclassification node is explicit, readable, and easily skipped or overridden later.

**New graph node: `reclassify_columns`.**

This is a plain Python function (no LLM call) that:
1. Loads the DataFrame from `state["output_path"]` (written by Executor 1).
2. Calls `classify_columns(df)`, `build_value_counts_summary(df, ...)`, `build_null_summary(df)`.
3. Returns updated `categorical_cols`, `numeric_like_cols`, `true_numeric_cols`, `value_counts_summary`, `null_summary` to state.

```python
# source_code/graph.py — new node
def reclassify_columns_node(state: AgentState) -> dict:
    import pandas as pd
    from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary
    output_path = state.get("output_path", "standardized_output.csv")
    df = pd.read_csv(output_path)
    cat, num_like, true_num = classify_columns(df)
    return {
        "categorical_cols"    : cat,
        "numeric_like_cols"   : num_like,
        "true_numeric_cols"   : true_num,
        "value_counts_summary": build_value_counts_summary(df, cat, top_n=20),
        "null_summary"        : build_null_summary(df),
    }
```

**Updated graph edge sequence:**
```
agent1_renamer → executor1 → reclassify_columns → agent2_cleaner → executor2 → END
```

**Impact on main.py.** The initial `classify_columns` call in `main.py` is retained — it provides valid data to the state as a starting point and is used if Agent 2 somehow runs before executor1 (error recovery path). The reclassification node overwrites it with fresh data.

---

### Bug 3: exec() Empty Globals Breaks numpy Imports

**Root cause.** `exec(cleaning_code, {}, local_vars)` passes an empty dict `{}` as the global namespace. Python's `import` statement requires access to builtins to function. When Agent 2's cleaning code contains `import numpy as np` (which the prompt explicitly instructs), this fails at runtime with `ImportError: __import__ not found`.

**Decision: Pass builtins and common imports in the globals dict.**

**Exact fix in `executor.py`:**

```python
import numpy as np

exec_globals = {
    "__builtins__": __builtins__,
    "pd": pd,
    "np": np,
}
exec(cleaning_code, exec_globals, local_vars)
```

Providing `pd` and `np` directly in globals means the LLM's code can use both even without explicit `import` statements, which is a secondary benefit. The `__builtins__` key restores all built-in functions (`len`, `str`, `range`, etc.) which are also absent from an empty globals dict.

This fix applies to BOTH `exec()` calls in the Executor and to the `exec()` calls in agent files that parse `rename_map` and `ambiguous_fields` from the LLM response.

---

## 2. New File Structure

The following files and directories are created as part of this feature. No existing files are deleted. Existing files that require modification are listed in Section 9.

```
D:/Agnetic_DS/
├── source_code/
│   ├── config/
│   │   ├── __init__.py              # Exports: PipelineConfig, LLMConfig, LLMFactory
│   │   ├── llm_config.py            # LLMConfig and PipelineConfig dataclasses
│   │   ├── llm_factory.py           # LLMFactory class with .create() static method
│   │   └── loaders.py               # load_query(), load_rules(), load_client_config()
│   └── agents/
│       └── (agent files modified in-place — see Section 6)
│
└── tests/
    ├── __init__.py
    ├── mock_llm.py                  # MockLLM and MockResponse classes
    ├── run_tests.py                 # CLI wrapper: runs pytest, generates markdown report
    ├── fixtures/
    │   ├── agent_1/
    │   │   └── scenarios.json       # All Agent 1 test scenarios keyed by name
    │   ├── agent_2/
    │   │   └── scenarios.json       # All Agent 2 test scenarios keyed by name
    │   └── sample_telecom.csv       # Synthetic telecom dataset (~10 cols, ~50 rows)
    ├── agents/
    │   ├── __init__.py
    │   ├── test_agent_1.py          # 9 Agent 1 test cases
    │   ├── test_agent_2.py          # 8 Agent 2 test cases
    │   └── test_executor.py         # 6 Executor test cases
    └── reports/
        └── .gitkeep                 # Directory tracked by git; reports are gitignored
```

**What is NOT created:**
- No `source_code/llm/` subdirectory. The LLM factory and config live under `source_code/config/` per the task specification.
- No `tests/unit/` or `tests/integration/` subdirs. All agent tests live flat under `tests/agents/` per the task specification. Integration tests may be added later.
- No `clients/` directory. That is a future concern outside this feature's scope.

---

## 3. LLMFactory — Exact Interface

**File:** `source_code/config/llm_factory.py`

```python
from __future__ import annotations
from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """
    Single factory point for all LLM instantiation.
    Adding a new provider requires changes only in this class.
    """

    @staticmethod
    def create(
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        max_tokens: int = 8000,
    ) -> BaseChatModel:
        """
        Instantiate and return a LangChain BaseChatModel for the given provider.

        Parameters
        ----------
        provider    : One of "groq", "openai", "anthropic", "ollama" (case-insensitive).
        model       : Provider-specific model name, e.g. "llama-3.3-70b-versatile".
        api_key     : API key string. None is valid only for "ollama".
        base_url    : Base URL override. Required for "ollama"; ignored for cloud providers.
        temperature : Sampling temperature. Defaults to 0 (deterministic).
        max_tokens  : Maximum tokens in the response. Defaults to 8000.

        Returns
        -------
        BaseChatModel instance ready to call .invoke() on.

        Raises
        ------
        ValueError  : If provider is not in the supported list.
        """
        p = provider.lower().strip()

        if p == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )

        elif p == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )

        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
            )

        elif p == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
                # max_tokens omitted: ChatOllama uses num_predict
            )

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: groq, openai, anthropic, ollama"
            )
```

**Key design decisions:**
- Imports are deferred (inside the `if` branches). This means only the installed provider's LangChain package is required at runtime — no `ImportError` if `langchain_anthropic` is not installed when using Groq.
- `ChatOllama` does not accept `max_tokens`; it uses `num_predict`. Ollama branch omits `max_tokens`.
- The method is `@staticmethod` — no class instantiation needed. Callers write `LLMFactory.create(...)`.

---

## 4. PipelineConfig + LLMConfig — Exact Dataclass Definitions

**File:** `source_code/config/llm_config.py`

```python
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """
    Configuration for a single LLM instance.
    Passed to LLMFactory.create() to produce a BaseChatModel.
    """
    provider   : str            = "groq"
    model      : str            = "llama-3.3-70b-versatile"
    api_key    : Optional[str]  = None
    base_url   : Optional[str]  = None
    temperature: float          = 0.0
    max_tokens : int            = 8000


@dataclass
class PipelineConfig:
    """
    Top-level configuration object for a pipeline session.
    Constructed once in main.py (or Streamlit app) and injected
    into agent factory functions at graph construction time.
    Never stored in AgentState.
    """
    default_llm    : LLMConfig        = field(default_factory=LLMConfig)
    agent_overrides: dict[str, LLMConfig] = field(default_factory=dict)
    # Keys: "agent1", "agent2", etc. Values: LLMConfig overrides.

    def get_llm_config(self, agent_name: str) -> LLMConfig:
        """
        Return the LLMConfig for the named agent.
        Falls back to default_llm if no override is registered.
        """
        return self.agent_overrides.get(agent_name, self.default_llm)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Construct PipelineConfig from environment variables.
        Backward compatible: if only GROQ_API_KEY is set, behaves
        exactly as the pre-refactor hardcoded configuration.

        Environment variables:
            LLM_PROVIDER   : defaults to "groq"
            LLM_MODEL      : defaults to "llama-3.3-70b-versatile"
            LLM_API_KEY    : explicit key; falls back to GROQ_API_KEY
            LLM_BASE_URL   : used for Ollama; optional
        """
        return cls(
            default_llm=LLMConfig(
                provider   =os.getenv("LLM_PROVIDER", "groq"),
                model      =os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key    =os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
                base_url   =os.getenv("LLM_BASE_URL"),
            )
        )
```

**Key design decisions:**
- `PipelineConfig` is a plain dataclass — no LangChain dependencies. Easy to construct in tests without any env vars.
- `from_env()` is the default construction path for `main.py`. Tests construct `PipelineConfig` directly with explicit values.
- `agent_overrides` uses string keys (`"agent1"`, `"agent2"`) matching the agent name convention used in `make_*_agent()` factory functions. These are not node names; they are stable logical identifiers.
- LLM instances are NOT stored in `PipelineConfig`. The config holds parameters; `LLMFactory.create()` is called at agent invocation time to produce the LLM instance. This keeps `PipelineConfig` serialisable.

---

## 5. MockLLM — Exact Class Definition

**File:** `tests/mock_llm.py`

```python
from __future__ import annotations
from typing import Any


class MockResponse:
    """
    Minimal stand-in for LangChain's AIMessage.
    Exposes only the .content attribute consumed by agent code.
    """
    def __init__(self, content: str) -> None:
        self.content: str = content


class MockLLM:
    """
    Test double for any LangChain BaseChatModel.
    Returns a pre-programmed response string without making network calls.

    Usage:
        mock = MockLLM(content="```python\\nrename_map = {}\\n```")
        agent_fn = make_field_renamer_agent(config_with_mock(mock))
        result = agent_fn(state)
        assert "SELECT" in mock.last_prompt  # prompt injection verified

    Attributes:
        last_prompt : The exact value passed to the most recent .invoke() call.
                      None if .invoke() has not been called yet.
                      Used in tests to assert that context (SQL, metadata) was
                      correctly injected into the prompt before the LLM call.
    """

    def __init__(self, content: str) -> None:
        self._content  : str       = content
        self.last_prompt: Any      = None

    def invoke(self, prompt: Any) -> MockResponse:
        """
        Record the prompt and return the pre-programmed response.
        Accepts both str prompts and list[BaseMessage] prompts (future-safe).
        """
        self.last_prompt = prompt
        return MockResponse(self._content)
```

**Key design decisions:**
- `MockLLM` does NOT inherit from `BaseChatModel`. Inheriting from `BaseChatModel` requires implementing abstract methods and registering with LangChain's provider registry, which is unnecessary overhead for a test double. Python's duck typing means that as long as `MockLLM` exposes `.invoke()` returning an object with `.content`, it is fully compatible with agent code.
- `last_prompt` is set to `None` at init, not an empty string, so tests can distinguish "never called" from "called with empty string".
- `last_prompt` accepts `Any` to handle both string prompts (current) and list-of-message prompts (future agents).

**How to inject MockLLM into an agent for tests:**

```python
# tests/agents/test_agent_1.py

import json
from pathlib import Path
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig, LLMConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent

FIXTURES = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "agent_1" / "scenarios.json").read_text()
)

def make_test_config(mock_llm: MockLLM) -> PipelineConfig:
    """
    Return a PipelineConfig that, when its get_llm_config() is called,
    produces the mock_llm instead of a real LLM.
    The agent factory is expected to call LLMFactory.create(**config.get_llm_config("agent1"))
    — we bypass this by having the agent accept an already-instantiated llm.
    See Section 6 for the agent signature that makes this possible.
    """
    cfg = PipelineConfig()
    cfg._mock_llm_override = mock_llm  # see Section 6 for how agents honour this
    return cfg
```

Note: the exact injection mechanism depends on how `make_field_renamer_agent` resolves the LLM. See Section 6 for the precise agent refactor that makes MockLLM injectable without modifying agent logic.

---

## 6. Agent Refactor — From Direct Function to Factory Closure

### 6.1 The Signature Change

**Before (current code):**

```python
# agent_1_field_renamer.py
def field_renamer_agent(state: AgentState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", ...)  # hardcoded
    response = llm.invoke(prompt)
    ...
```

**After (refactored):**

```python
# agent_1_field_renamer.py
from langchain_core.language_models import BaseChatModel
from source_code.config.llm_config import PipelineConfig
from source_code.config.llm_factory import LLMFactory

def make_field_renamer_agent(config: PipelineConfig):
    """
    Factory function. Returns a LangGraph-compatible node function
    with the LLM resolved from config and closed over.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration. Provides LLM parameters (or a pre-built
        LLM instance, for testing). Closed over in the returned function.

    Returns
    -------
    Callable[[AgentState], dict]
        LangGraph node function compatible with workflow.add_node().
    """
    llm_cfg = config.get_llm_config("agent1")
    llm: BaseChatModel = LLMFactory.create(
        provider   =llm_cfg.provider,
        model      =llm_cfg.model,
        api_key    =llm_cfg.api_key,
        base_url   =llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens =llm_cfg.max_tokens,
    )

    def field_renamer_agent(state: AgentState) -> dict:
        # All existing logic unchanged — only the llm variable origin changes.
        # llm is closed over from the outer factory scope.
        ...
        response = llm.invoke(prompt)
        ...

    return field_renamer_agent
```

**Same pattern for Agent 2:**

```python
# agent_2_field_cleaner.py
def make_field_cleaner_agent(config: PipelineConfig):
    llm_cfg = config.get_llm_config("agent2")
    llm: BaseChatModel = LLMFactory.create(
        provider   =llm_cfg.provider,
        model      =llm_cfg.model,
        api_key    =llm_cfg.api_key,
        base_url   =llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens =llm_cfg.max_tokens,
    )

    def field_cleaner_agent(state: AgentState) -> dict:
        ...
        response = llm.invoke(prompt)
        ...

    return field_cleaner_agent
```

### 6.2 Injecting MockLLM Without Changing Agent Logic

The LLM is resolved ONCE when `make_field_renamer_agent(config)` is called — it is closed over. To inject a `MockLLM`, the test provides a `PipelineConfig` whose `get_llm_config()` returns an `LLMConfig` that, when passed to `LLMFactory.create()`, produces the mock.

This requires a thin hook in `LLMFactory.create()`: an optional `_llm_override` parameter on `LLMConfig` that, if set, is returned directly without instantiation. Alternatively — and more cleanly — the factory function is patched in the test using `unittest.mock.patch`. The recommended approach:

**Option A: `_llm_instance` field on `LLMConfig` (recommended).**

Add one optional field to `LLMConfig`:

```python
@dataclass
class LLMConfig:
    provider   : str            = "groq"
    model      : str            = "llama-3.3-70b-versatile"
    api_key    : Optional[str]  = None
    base_url   : Optional[str]  = None
    temperature: float          = 0.0
    max_tokens : int            = 8000
    _llm_instance: Any          = field(default=None, repr=False, compare=False)
    # If set, LLMFactory.create() returns this directly (for testing).
```

In `LLMFactory.create()`:

```python
@staticmethod
def create(provider: str, model: str, ..., _llm_instance=None) -> BaseChatModel:
    if _llm_instance is not None:
        return _llm_instance
    # ... normal factory logic
```

Test usage:

```python
mock = MockLLM(content=FIXTURES["happy_path"])
cfg = PipelineConfig(
    default_llm=LLMConfig(_llm_instance=mock)
)
agent_fn = make_field_renamer_agent(cfg)
result = agent_fn(state)
assert mock.last_prompt is not None
```

This approach: zero changes to agent internal logic, no mock patching, clean test code, explicit override visible in config.

### 6.3 LLM Instantiation Timing

The LLM is instantiated once at factory call time (when the graph is compiled), not per-invocation. This means:
- Cheap instantiation (LangChain providers do lazy connection setup).
- The same LLM object is reused across all invocations of a given agent in a pipeline run.
- Tests are clean: the MockLLM is instantiated once per test, injected once.

---

## 7. Graph Refactor — Reclassification Node

**File:** `source_code/graph.py`

The graph is currently compiled at module level with hardcoded references to imported functions. After refactoring, the graph is constructed inside a `build_graph(config: PipelineConfig)` factory function so that agent closures can be injected.

```python
# source_code/graph.py
from langgraph.graph import StateGraph, END
from source_code.state import AgentState
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent
from source_code.agents.agent_2_field_cleaner import make_field_cleaner_agent
from source_code.agents.executor import code_executor_agent
from source_code.agents.reclassify import reclassify_columns_node


def build_graph(config: PipelineConfig) -> "CompiledStateGraph":
    workflow = StateGraph(AgentState)

    # Agent nodes — closed over config
    workflow.add_node("agent1_renamer",      make_field_renamer_agent(config))
    workflow.add_node("executor1",           code_executor_agent)
    workflow.add_node("reclassify_columns",  reclassify_columns_node)   # NEW
    workflow.add_node("agent2_cleaner",      make_field_cleaner_agent(config))
    workflow.add_node("executor2",           code_executor_agent)

    workflow.set_entry_point("agent1_renamer")
    workflow.add_edge("agent1_renamer",     "executor1")
    workflow.add_edge("executor1",          "reclassify_columns")   # NEW
    workflow.add_edge("reclassify_columns", "agent2_cleaner")       # NEW
    workflow.add_edge("agent2_cleaner",     "executor2")
    workflow.add_edge("executor2",          END)

    return workflow.compile()


# Backward-compatible module-level instance for existing main.py usage
# (main.py will be updated to call build_graph() explicitly — this is removed
#  once main.py is updated; see Section 9)
```

**Reclassify node placement:** Between `executor1` and `agent2_cleaner`. This is the only correct placement — after the renamed DataFrame has been written to disk by executor1, and before agent2 needs the classified column lists.

**Reclassify node implementation — new file:**

```python
# source_code/agents/reclassify.py
import pandas as pd
from source_code.state import AgentState
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary


def reclassify_columns_node(state: AgentState) -> dict:
    """
    Graph node: re-runs column classification after Agent 1 renames columns.
    Reads the post-rename CSV from output_path, re-classifies, and updates state.
    Must run after executor1 and before agent2_cleaner.
    """
    output_path = state.get("output_path", "standardized_output.csv")
    df = pd.read_csv(output_path)

    categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    null_summary         = build_null_summary(df)

    return {
        "categorical_cols"    : categorical_cols,
        "numeric_like_cols"   : numeric_like_cols,
        "true_numeric_cols"   : true_numeric_cols,
        "value_counts_summary": value_counts_summary,
        "null_summary"        : null_summary,
    }
```

**Note on `output_path` in state.** The Executor currently hardcodes `output_path = "standardized_output.csv"` and does not use `state["output_path"]`. As part of this feature, the Executor is updated to:
1. Read `output_path` from state if set, or default to `"standardized_output.csv"`.
2. Write the result CSV to that path.
3. Return `{"output_path": output_path, "error_log": None}` so the reclassify node and subsequent nodes can find the file.

---

## 8. Config Loaders — `loaders.py`

**File:** `source_code/config/loaders.py`

This module replaces the hardcoded file reads in `main.py`. It is the abstraction boundary between file-system input and the pipeline.

```python
# source_code/config/loaders.py
from __future__ import annotations
import os
import io
import pandas as pd
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary


def load_query(query_path: str) -> str:
    """Load SQL query string from a .sql file."""
    with open(query_path, "r", encoding="utf-8") as f:
        return f.read()


def load_rules(rules_path: str) -> str:
    """Load special rules string from a .txt file."""
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


def load_client_config(
    client_id: str,
    base_dir: str = "clients/",
) -> dict:
    """
    Load pipeline inputs for a named client.
    Expects the following directory layout:
        <base_dir>/
          <client_id>/
            data.csv
            query.sql
            rules.txt

    Returns a dict compatible with AgentState initial input.
    """
    client_dir = os.path.join(base_dir, client_id)
    return _build_initial_input(
        data_path  =os.path.join(client_dir, "data.csv"),
        query_path =os.path.join(client_dir, "query.sql"),
        rules_path =os.path.join(client_dir, "rules.txt"),
        client_id  =client_id,
    )


def load_pipeline_inputs(
    data_path : str,
    query_path: str,
    rules_path: str,
    client_id : str = "default",
) -> dict:
    """
    Load pipeline inputs from explicit file paths.
    Drop-in replacement for the hardcoded reads in main.py.

    Returns a dict compatible with AgentState initial input.
    """
    return _build_initial_input(
        data_path =data_path,
        query_path=query_path,
        rules_path=rules_path,
        client_id =client_id,
    )


def _build_initial_input(
    data_path : str,
    query_path: str,
    rules_path: str,
    client_id : str,
) -> dict:
    sql_query    = load_query(query_path)
    special_rules= load_rules(rules_path)

    df = pd.read_csv(data_path)

    # Metadata string for Agent 1
    dtypes_dict  = df.dtypes.astype(str).to_dict()
    sample_row   = df.head(1).to_dict(orient="records")[0]
    metadata_summary = (
        f"DATASET PROFILE:\n"
        f"1. COLUMNS & TYPES:\n{dtypes_dict}\n"
        f"2. SAMPLE ROW:\n{sample_row}"
    )

    # Pre-classification for Agent 2 (will be overwritten by reclassify_columns_node)
    categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    null_summary         = build_null_summary(df)

    return {
        "file_path"           : data_path,
        "target_column"       : "",           # Caller sets this
        "sql_query"           : sql_query,
        "df_columns"          : list(df.columns),
        "special_rules"       : special_rules,
        "iteration_count"     : 0,
        "metadata_summary"    : metadata_summary,
        "categorical_cols"    : categorical_cols,
        "numeric_like_cols"   : numeric_like_cols,
        "true_numeric_cols"   : true_numeric_cols,
        "value_counts_summary": value_counts_summary,
        "null_summary"        : null_summary,
    }
```

**Client directory convention:**

```
clients/
  <client_id>/
    data.csv        # The raw dataset to process
    query.sql       # The SQL query providing semantic context
    rules.txt       # Special cleaning rules for this client
```

The existing `data/`, `queries/`, `rules/` paths from current `main.py` are preserved as a legacy layout. `main.py` is updated to call `load_pipeline_inputs(...)` with the same explicit paths — no functional change, but routes through the module boundary.

**Streamlit replacement path:** When the Streamlit UI is built, it will call `_build_initial_input()` directly with data loaded from uploaded files / text inputs. The function signature does not change; only the sources of the three input files change.

---

## 9. Test Runner Flow — `run_tests.py`

**File:** `tests/run_tests.py`

The test runner is a CLI wrapper that:
1. Accepts optional `--agent N` filter argument.
2. Invokes pytest programmatically.
3. Reads pytest's JSON output report.
4. Renders a structured markdown review document to `tests/reports/review_<YYYYMMDD_HHMMSS>.md`.

**Full orchestration flow:**

```
python tests/run_tests.py [--agent 1|2]
  │
  ├─ Parse sys.argv for --agent filter
  │
  ├─ Call pytest.main([
  │     "tests/agents/",
  │     "-v",
  │     "--tb=short",
  │     "--json-report",
  │     "--json-report-file=tests/.pytest_report.json"
  │  ] + optional mark filter)
  │    │
  │    ├─ tests/agents/test_agent_1.py
  │    │    Each test function:
  │    │      1. Load scenario from tests/fixtures/agent_1/scenarios.json
  │    │      2. Build AgentState dict (uses sample_telecom.csv columns)
  │    │      3. Instantiate MockLLM(content=fixture_response)
  │    │      4. Build PipelineConfig with LLMConfig(_llm_instance=mock)
  │    │      5. Call make_field_renamer_agent(config) → agent_fn
  │    │      6. Call agent_fn(state) → result dict
  │    │      7. Assert on result fields AND mock.last_prompt
  │    │      8. For tests requiring executor: call code_executor_agent(state_with_result)
  │    │
  │    ├─ tests/agents/test_agent_2.py  (same pattern)
  │    │
  │    └─ tests/agents/test_executor.py
  │         Each test:
  │           1. Build state with cleaning_code (inline, not from fixture JSON)
  │           2. Write sample_telecom.csv to a temp path (or use the fixture directly)
  │           3. Call code_executor_agent(state) → result dict
  │           4. Assert on error_log, output file existence, df shape
  │
  ├─ Read tests/.pytest_report.json
  │
  └─ Call generate_review_document(report_data)
       ├─ Render summary table (PASSED / FAILED / WARNING counts)
       ├─ Render per-agent section with result table
       ├─ For each FAILED test: render full failure block (see Section 11)
       ├─ For each WARNING: render abbreviated block
       └─ Write to tests/reports/review_<timestamp>.md
```

**Filtering by agent:**

```
python tests/run_tests.py --agent 1    → runs only @pytest.mark.agent1 tests
python tests/run_tests.py --agent 2    → runs only @pytest.mark.agent2 tests
python tests/run_tests.py              → runs all tests
```

Each test function is decorated with the appropriate pytest mark:

```python
@pytest.mark.agent1
def test_happy_path():
    ...
```

**Dependency:** `pytest-json-report` package (`pip install pytest-json-report`). This is the only new test dependency.

---

## 10. Fixture JSON Schema — `scenarios.json`

**Files:**
- `tests/fixtures/agent_1/scenarios.json`
- `tests/fixtures/agent_2/scenarios.json`

**Schema:**

```json
{
  "<scenario_name>": {
    "llm_response": "<exact string the MockLLM will return as .content>",
    "description": "<one-sentence human description of what this scenario tests>",
    "expected_behavior": "<what a correct agent should do with this response>"
  }
}
```

**Example — `tests/fixtures/agent_1/scenarios.json`:**

```json
{
  "happy_path": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\",\n    \"rev_30d\": \"RevenueLast30d\",\n    \"mou_out_30d\": \"MouOutgoingLast30d\",\n    \"data_vol_30d\": \"DataUsageLast30dKb\",\n    \"churn_flag\": \"ChurnFlag\",\n    \"tenure_months\": \"TenureMonths\",\n    \"region_cd\": \"RegionCode\",\n    \"plan_type\": \"PlanType\",\n    \"contract_end_dt\": \"ContractEndDate\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Valid two-block response. All 10 sample_telecom columns covered.",
    "expected_behavior": "column_map has 10 entries, ambiguous_fields is [], cleaning_code contains df.rename"
  },

  "sql_context_alias_resolution": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"rev_30d\": \"RevenueLast30d\",\n    \"mou_out_30d\": \"MouOutgoingLast30d\",\n    \"data_vol_30d\": \"DataUsageLast30dKb\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "LLM correctly resolves rev_30d to RevenueLast30d given SQL context.",
    "expected_behavior": "column_map['rev_30d'] == 'RevenueLast30d'; mock.last_prompt contains sql_query"
  },

  "missing_second_block": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```",
    "description": "Only one code block returned — no ambiguous_fields block.",
    "expected_behavior": "cleaning_code is non-empty, ambiguous_fields is [], not None"
  },

  "no_code_blocks": {
    "llm_response": "I cannot determine the column mappings without additional context about the schema.",
    "description": "LLM returns plain prose, no fenced code blocks.",
    "expected_behavior": "cleaning_code is '', ambiguous_fields is [], column_map is {}"
  },

  "duplicate_values_in_rename_map": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"mobile_no\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Two keys map to same clean name. Known gap: agent does not detect this.",
    "expected_behavior": "DOCUMENTED GAP: column_map will have duplicate values; no error raised"
  },

  "missing_columns_from_rename_map": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Only 2 of 10 columns covered in rename_map.",
    "expected_behavior": "DOCUMENTED GAP: column_map has 2 entries; missing columns silently ignored"
  },

  "ambiguous_fields_missing_keys": {
    "llm_response": "```python\nrename_map = {\n    \"gender\": \"Gender\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = [\n    {\"original_column\": \"gender\"}\n]\n```",
    "description": "ambiguous_fields entry missing 'candidates' and 'reason' keys.",
    "expected_behavior": "ambiguous_fields is non-empty; .get('candidates') returns None; no crash"
  },

  "keys_dont_match_df_columns": {
    "llm_response": "```python\nrename_map = {\n    \"subscriber_id\": \"Msisdn\",\n    \"phone_number\": \"AlternatePhone\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "rename_map keys reference columns not present in the test DataFrame.",
    "expected_behavior": "DOCUMENTED GAP: pandas .rename() silently ignores non-existent keys"
  },

  "malformed_python": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Syntax error in rename code — missing closing brace on dict.",
    "expected_behavior": "column_map is {}; cleaning_code contains malformed string; executor produces SyntaxError in error_log"
  }
}
```

**`tests/fixtures/sample_telecom.csv` column specification:**

The CSV must contain exactly the following 10 columns to match the happy_path fixture:

| Column name (raw)    | Type             | Purpose in tests                          |
|----------------------|------------------|-------------------------------------------|
| `msisdn`             | string           | Primary key; SQL alias test (`a.msisdn`)  |
| `gender`             | categorical      | Categorical cleaning test                 |
| `rev_30d`            | numeric (dirty)  | SQL context alias resolution test         |
| `mou_out_30d`        | float            | SQL context resolution test               |
| `data_vol_30d`       | float            | SQL context resolution test               |
| `churn_flag`         | int (0/1)        | Target column                             |
| `tenure_months`      | int              | Numeric column                            |
| `region_cd`          | categorical      | Categorical cleaning test                 |
| `plan_type`          | categorical      | Categorical cleaning test                 |
| `contract_end_dt`    | int (YYYYMMDD)   | Date-as-integer test                      |

50 rows. Values for `rev_30d` should include some strings like `"N/A"` and `"—"` to qualify it as `numeric_like`. Values for `gender` should include synonyms like `"M"`, `"Male"`, `"m"`, `"F"`, `"Female"`.

---

## 11. Review Document Format — Generated Markdown

**Filename:** `tests/reports/review_<YYYYMMDD_HHMMSS>.md`

**Complete structure:**

```markdown
# Test Review — Agent Testing Harness
**Generated:** 2026-03-02 14:32:01
**Command:** python tests/run_tests.py
**Commit:** abc1234 (if git is available; omit if not)

---

## Summary

| Status  | Count |
|---------|-------|
| PASSED  | 17    |
| FAILED  | 4     |
| WARNING | 2     |
| TOTAL   | 23    |

**Failures:** TC5, TC8, TC15, TC21
**Warnings:** TC6, TC22

---

## Agent 1 — Field Renamer

| ID  | Test Name                          | Category              | Result  |
|-----|------------------------------------|-----------------------|---------|
| TC1 | Happy Path                         | Core                  | PASSED  |
| TC2 | SQL Query Context                  | Context Injection     | PASSED  |
| TC3 | Missing Second Code Block          | Edge Case             | PASSED  |
| TC4 | No Code Blocks                     | Edge Case             | PASSED  |
| TC5 | Duplicate Values in rename_map     | Data Integrity        | FAILED  |
| TC6 | Missing Columns from rename_map    | Coverage              | WARNING |
| TC7 | ambiguous_fields Missing Keys      | Schema Validation     | PASSED  |
| TC8 | rename_map Keys Don't Match df     | Column Mismatch       | FAILED  |
| TC9 | Malformed Python in Code Block     | Error Handling        | PASSED  |

---

### TC5 — Duplicate Values in rename_map

**Result:** FAILED
**Agent:** Agent 1 — Field Renamer
**Category:** Data Integrity
**Fixture:** `tests/fixtures/agent_1/scenarios.json` → `duplicate_values_in_rename_map`

**Expected:**
`column_map` values should be unique. No two raw columns should map to the same clean name.

**Received:**
```python
column_map = {"msisdn": "Msisdn", "mobile_no": "Msisdn"}
# Duplicate values: ["Msisdn"]
```

**Probable Cause:**
Agent 1 does not validate rename_map value uniqueness after extracting it from the LLM response. The LLM produced a duplicate and it passed through silently. When pandas runs `df.rename(columns=rename_map)`, two columns will be renamed to the same name, causing silent data corruption.

**Suggested Fix:**
In `agent_1_field_renamer.py`, after extracting `column_map`, add:
```python
from collections import Counter
dupes = [v for v, count in Counter(column_map.values()).items() if count > 1]
if dupes:
    raise ValueError(f"Duplicate values in rename_map: {dupes}")
```

---

### TC6 — Missing Columns from rename_map

**Result:** WARNING
**Agent:** Agent 1 — Field Renamer
**Category:** Coverage Gap (Documented)
**Fixture:** `tests/fixtures/agent_1/scenarios.json` → `missing_columns_from_rename_map`

**Note:**
This test documents a known gap. The agent successfully processes the partial rename_map without error. 8 of 10 columns are silently left unrenamed. The test passes (no assertion failure) but is marked WARNING to flag the gap.

**Gap Description:**
Agent 1 does not validate that all `df_columns` appear in the rename_map. Uncovered columns retain their raw names in the output DataFrame.

**Suggested Fix:**
After extracting `column_map`, compute coverage: `missing = set(cleaned_cols) - set(column_map.keys())` and add to `ambiguous_fields` or log a warning.

---

## Agent 2 — Field Cleaner

| ID   | Test Name                          | Category              | Result  |
|------|------------------------------------|-----------------------|---------|
| TC10 | Happy Path                         | Core                  | PASSED  |
| TC11 | Metadata Context Injection         | Context Injection     | PASSED  |
| TC12 | Missing Second Code Block          | Edge Case             | PASSED  |
| TC13 | No Code Blocks                     | Edge Case             | PASSED  |
| TC14 | flagged_columns Missing Keys       | Schema Validation     | PASSED  |
| TC15 | Cleaning Code Introduces New Nulls | Safety Check          | FAILED  |
| TC16 | Cleaning Code Drops Columns        | Safety Check          | WARNING |
| TC17 | Malformed Python                   | Error Handling        | PASSED  |

---

### TC15 — Cleaning Code Introduces New Nulls

**Result:** FAILED
**Agent:** Agent 2 / Executor
**Category:** Safety Check
...

---

## Executor

| ID   | Test Name                          | Category              | Result  |
|------|------------------------------------|-----------------------|---------|
| TC18 | Happy Path                         | Core                  | PASSED  |
| TC19 | Empty cleaning_code                | Edge Case             | PASSED  |
| TC20 | Code Raises Runtime Exception      | Error Handling        | PASSED  |
| TC21 | Code Drops a Column                | Safety Check          | FAILED  |
| TC22 | Code Introduces New Nulls          | Safety Check          | WARNING |
| TC23 | Output File Written Correctly      | Output Validation     | PASSED  |

---

## Appendix: Test IDs and Fixture Mapping

| TC ID | File                              | Scenario Key                      |
|-------|-----------------------------------|-----------------------------------|
| TC1   | tests/fixtures/agent_1/scenarios.json | happy_path                    |
| TC2   | tests/fixtures/agent_1/scenarios.json | sql_context_alias_resolution  |
...
```

**Status semantics:**
- `PASSED`: all assertions in the test function pass.
- `WARNING`: assertions pass but a secondary condition indicates a documented gap or degraded behavior. Set programmatically by the test via `pytest.warns()` or a custom marker.
- `FAILED`: one or more `assert` statements raised `AssertionError`, or the test raised an unexpected exception.

**Failure block required fields (all failures must have all six):**

1. `Result` — FAILED or WARNING
2. `Agent` — which agent and test category
3. `Category` — from the test category taxonomy
4. `Fixture` — path and key, or "inline" for executor tests
5. `Expected` — exact description of expected state/value
6. `Received` — exact value received (formatted as code block)
7. `Probable Cause` — one to three sentences explaining the root cause
8. `Suggested Fix` — specific file, line context, and code snippet to fix

---

## 12. Agent 2 SQL Context — Design Decision

**Decision: Agent 2 does NOT receive `sql_query` in its prompt for this feature.**

Rationale: By the time Agent 2 runs, columns have been renamed to standardised names. The SQL query was written against the original schema (with alias prefixes and raw names). Injecting it into Agent 2's prompt would provide stale, potentially confusing context. The SQL is a pre-rename artifact; Agent 2 operates on a post-rename DataFrame.

The "SQL context" test for Agent 2 (TC11) is renamed to "Metadata Context Injection" and instead asserts that `value_counts_summary` and `null_summary` are present in `mock.last_prompt`. This is a meaningful and testable assertion that verifies Agent 2 receives the enriched metadata it needs.

If a future requirement emerges for Agent 2 to know which columns appear in the final analytical SQL (post-rename), a separate `analytical_sql` field should be added to `AgentState` at that point — not now.

---

## 13. Executor Refactor Summary

The Executor requires the following changes to fix Bugs 1 and 3 and support the reclassify node:

1. **Bug 3 fix:** Change `exec(cleaning_code, {}, local_vars)` to:
   ```python
   import numpy as np
   exec_globals = {"__builtins__": __builtins__, "pd": pd, "np": np}
   exec(cleaning_code, exec_globals, local_vars)
   ```

2. **Bug 1 fix (rename path):** After `exec()` runs for Agent 1's code, apply the alias-composite rename. The Executor must distinguish between Agent 1 (rename) and Agent 2 (cleaning) execution. The cleanest approach is to check whether `state.get("column_map")` is populated — if it is, apply the composite rename after exec; if not, skip it. Alternatively (and more explicitly), use two distinct executor functions: `rename_executor_agent` and `cleaning_executor_agent`.

   **Decision: Use two distinct executor functions.** This is cleaner and avoids conditional logic based on state inspection.

   ```python
   # executor.py
   def rename_executor_agent(state: AgentState) -> dict:
       """Executor for Agent 1: runs rename code then applies composite column map."""
       ...

   def cleaning_executor_agent(state: AgentState) -> dict:
       """Executor for Agent 2: runs cleaning code."""
       ...
   ```

   Both share common `_run_exec` logic.

3. **output_path from state:** The executor reads `output_path = state.get("output_path", "standardized_output.csv")` and saves to that path. The rename executor defaults to `"standardized_output_renamed.csv"`; the cleaning executor defaults to `"output_agent2_cleaned.csv"`. Both return `{"output_path": output_path}` in the result dict.

4. **Post-exec safety checks (NEW):** Both executor functions perform the following checks after exec():
   - **Column drop detection:** `dropped = set(df_before.columns) - set(df_after.columns)`. If non-empty, append to `error_log` as a WARNING. Do not raise.
   - **New null detection:** `new_nulls = df_after.isna().sum().sum() - df_before.isna().sum().sum()`. If positive, append to `error_log` as a WARNING.

   These checks satisfy TC15, TC16, TC21, TC22 from the test brief.

---

## 14. Files Modified vs Created

### New Files (created from scratch)

| File | Purpose |
|------|---------|
| `source_code/config/__init__.py` | Package init; re-exports PipelineConfig, LLMConfig, LLMFactory |
| `source_code/config/llm_config.py` | LLMConfig, PipelineConfig dataclasses |
| `source_code/config/llm_factory.py` | LLMFactory.create() |
| `source_code/config/loaders.py` | load_query(), load_rules(), load_client_config(), load_pipeline_inputs() |
| `source_code/agents/reclassify.py` | reclassify_columns_node() |
| `tests/__init__.py` | Test package init |
| `tests/mock_llm.py` | MockLLM, MockResponse |
| `tests/run_tests.py` | CLI wrapper + markdown report generator |
| `tests/fixtures/agent_1/scenarios.json` | Agent 1 test fixtures |
| `tests/fixtures/agent_2/scenarios.json` | Agent 2 test fixtures |
| `tests/fixtures/sample_telecom.csv` | Synthetic test dataset |
| `tests/agents/__init__.py` | Test subpackage init |
| `tests/agents/test_agent_1.py` | Agent 1 test cases (TC1–TC9) |
| `tests/agents/test_agent_2.py` | Agent 2 test cases (TC10–TC17) |
| `tests/agents/test_executor.py` | Executor test cases (TC18–TC23) |
| `tests/reports/.gitkeep` | Ensures reports/ is tracked by git |

### Modified Files (existing files changed)

| File | Changes |
|------|---------|
| `source_code/agents/agent_1_field_renamer.py` | Refactor to `make_field_renamer_agent(config)` factory; remove hardcoded ChatGroq |
| `source_code/agents/agent_2_field_cleaner.py` | Refactor to `make_field_cleaner_agent(config)` factory; remove hardcoded ChatGroq |
| `source_code/agents/executor.py` | Split into rename_executor + cleaning_executor; fix exec globals; add safety checks; use output_path from state |
| `source_code/graph.py` | Add reclassify_columns node; wrap in build_graph(config) factory |
| `main.py` | Use load_pipeline_inputs() from config.loaders; call build_graph(PipelineConfig.from_env()) |

### Unchanged Files

| File | Why unchanged |
|------|---------------|
| `source_code/state.py` | AgentState schema requires no new fields for this feature |
| `source_code/utils.py` | No changes needed; utility functions are consumed as-is |
| `source_code/prompts/agent_1.json` | No prompt changes in scope |
| `source_code/prompts/agent_2.json` | No prompt changes in scope |

---

## 15. Open Questions Resolved

| Question (from Brainstorm §8) | Decision |
|-------------------------------|----------|
| Alias-prefix bug: fix in Agent 1 or Executor? | Fixed in Executor via composite rename map |
| Agent 2 SQL context: add `sql_query` to prompt? | No. TC11 renamed to "Metadata Context Injection" |
| Stale categorical_cols: Executor reclassifies or dedicated node? | Dedicated `reclassify_columns_node` in graph |
| LLM instantiation: cache or per-call? | Instantiated once in factory closure; reused across calls |
| exec() globals: fix in scope? | Yes. `{"__builtins__": __builtins__, "pd": pd, "np": np}` |
| Executor output_path: use state or hardcode? | Use state with sensible defaults |
| Executor: single function or two? | Two: `rename_executor_agent` + `cleaning_executor_agent` |

---

*Architecture document complete. Implementation may begin.*
