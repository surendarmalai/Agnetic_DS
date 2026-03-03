# Code Plan — Agent Testing Harness + LLM Abstraction Layer

**Feature:** `agent-testing-harness`
**Author stage:** Code Planner (Stage 5)
**Input:** `02_architecture.md`, `03_audit.md`, all source files
**Date:** 2026-03-02

---

## 0. Mandatory Audit Resolutions Applied

All audit findings are incorporated into this plan. Deviations from `02_architecture.md` caused by audit findings are flagged inline with `[AUDIT: Cx / Hx / Mx / Lx]`.

Critical resolutions:
- **C1**: `rename_executor_agent` returns `{"output_path": "standardized_output_renamed.csv"}`. `reclassify_columns_node` fallback is `state.get("output_path", "standardized_output_renamed.csv")`.
- **C2**: `rename_executor_agent` does NOT call `exec(cleaning_code)`. It only applies the composite rename map built from `preprocess_column_names(df.columns)` + `state["column_map"]`.

---

## 1. Complete File Tree

### Files to CREATE

```
D:/Agnetic_DS/
├── requirements-dev.txt                          [NEW] [M9]
├── source_code/
│   ├── __init__.py                               [NEW] [L16]
│   ├── reclassify.py                             [NEW] [M8] — reclassify_columns_node
│   ├── config/
│   │   ├── __init__.py                           [NEW]
│   │   ├── llm_config.py                         [NEW]
│   │   ├── llm_factory.py                        [NEW]
│   │   └── loaders.py                            [NEW]
│   └── agents/
│       └── __init__.py                           [NEW] [L16]
└── tests/
    ├── __init__.py                               [NEW]
    ├── mock_llm.py                               [NEW]
    ├── run_tests.py                              [NEW]
    ├── fixtures/
    │   ├── agent_1/
    │   │   └── scenarios.json                    [NEW]
    │   ├── agent_2/
    │   │   └── scenarios.json                    [NEW]
    │   └── sample_telecom.csv                    [NEW]
    ├── agents/
    │   ├── __init__.py                           [NEW]
    │   ├── test_agent_1.py                       [NEW]
    │   ├── test_agent_2.py                       [NEW]
    │   └── test_executor.py                      [NEW]
    └── reports/
        └── .gitkeep                              [NEW]
```

### Files to MODIFY

```
D:/Agnetic_DS/
├── source_code/
│   ├── agents/
│   │   ├── agent_1_field_renamer.py              [MODIFY]
│   │   ├── agent_2_field_cleaner.py              [MODIFY]
│   │   └── executor.py                           [MODIFY — full rewrite]
│   └── graph.py                                  [MODIFY — full rewrite]
└── main.py                                       [MODIFY]
```

### Files Unchanged

```
source_code/state.py          — AgentState schema requires no changes
source_code/utils.py          — utility functions consumed as-is
source_code/prompts/agent_1.json — no prompt changes in scope
source_code/prompts/agent_2.json — no prompt changes in scope
```

---

## 2. Files to CREATE — Full Signatures and Logic

---

### `requirements-dev.txt` [M9]

**Content (exact):**
```
pytest>=7.0
pytest-json-report>=1.5
```

No other entries. Production dependencies (langchain, pandas, etc.) are not in scope for this file. A comment is added noting that `langchain-ollama` is required only when `LLM_PROVIDER=ollama`.

---

### `source_code/__init__.py` [L16]

**Content:** Empty file. Marks `source_code/` as a Python package. Do not add any imports.

---

### `source_code/agents/__init__.py` [L16]

**Content:** Empty file. Marks `source_code/agents/` as a Python package. Do not add any imports.

---

### `source_code/config/__init__.py`

**Content:**
```python
from source_code.config.llm_config import LLMConfig, PipelineConfig
from source_code.config.llm_factory import LLMFactory

__all__ = ["LLMConfig", "PipelineConfig", "LLMFactory"]
```

Re-exports the three most commonly imported names so callers can write `from source_code.config import PipelineConfig`.

---

### `source_code/config/llm_config.py` [H4]

**Classes: `LLMConfig`, `PipelineConfig`**

```python
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class LLMConfig:
    """
    Configuration for a single LLM instance.
    Passed to LLMFactory.create() to produce a BaseChatModel.

    The llm_instance field (no underscore) is the DI hook for tests.
    When set, LLMFactory.create() returns it directly without instantiation.
    It is included in repr so debugging shows the injected mock.
    compare=False: two configs with different injected mocks compare equal
    if all other fields match (correct semantics for config deduplication).
    """
    provider    : str           = "groq"
    model       : str           = "llama-3.3-70b-versatile"
    api_key     : Optional[str] = None
    base_url    : Optional[str] = None
    temperature : float         = 0.0
    max_tokens  : int           = 8000
    llm_instance: Any           = field(default=None, repr=True, compare=False)
    # [AUDIT H4] Renamed from _llm_instance to llm_instance (no underscore).
    # repr=True so debugging shows the injected mock.


@dataclass
class PipelineConfig:
    """
    Top-level configuration object for a pipeline session.
    Constructed once in main.py (or test) and injected into agent factories.
    Never stored in AgentState. Never holds live LLM instances directly.

    agent_overrides keys: "agent1", "agent2" (stable logical identifiers).
    """
    default_llm    : LLMConfig                = field(default_factory=LLMConfig)
    agent_overrides: dict[str, LLMConfig]     = field(default_factory=dict)

    def get_llm_config(self, agent_name: str) -> LLMConfig:
        """
        Return the LLMConfig for the named agent.

        Parameters
        ----------
        agent_name : str
            Logical agent identifier: "agent1" or "agent2".

        Returns
        -------
        LLMConfig
            Agent-specific override if registered, otherwise default_llm.
        """
        # Body: return self.agent_overrides.get(agent_name, self.default_llm)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Construct PipelineConfig from environment variables.

        Environment variables (all optional):
            LLM_PROVIDER  : defaults to "groq"
            LLM_MODEL     : defaults to "llama-3.3-70b-versatile"
            LLM_API_KEY   : explicit key; falls back to GROQ_API_KEY
            LLM_BASE_URL  : used for Ollama; optional

        Returns
        -------
        PipelineConfig
            Config backed by environment. For test injection post-construction:
                cfg = PipelineConfig.from_env()
                cfg.default_llm.llm_instance = mock
            Or use the convenience factory:
                cfg = PipelineConfig.with_mock(mock)
        """
        # Body: return cls(default_llm=LLMConfig(
        #     provider=os.getenv("LLM_PROVIDER", "groq"),
        #     model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        #     api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
        #     base_url=os.getenv("LLM_BASE_URL"),
        # ))

    @classmethod
    def with_mock(cls, mock: Any) -> "PipelineConfig":
        """
        Convenience factory for tests. Returns a PipelineConfig where
        ALL agents receive the given mock LLM instance.

        Parameters
        ----------
        mock : Any
            A MockLLM instance (or any object with a .invoke() method).

        Returns
        -------
        PipelineConfig
            Config whose default_llm.llm_instance is set to mock.

        Usage
        -----
            cfg = PipelineConfig.with_mock(MockLLM(content="..."))
            agent_fn = make_field_renamer_agent(cfg)

        [AUDIT H4] This is the recommended injection path for tests.
        """
        # Body: return cls(default_llm=LLMConfig(llm_instance=mock))
```

---

### `source_code/config/llm_factory.py` [H3, H4, M11]

**Class: `LLMFactory`**

```python
from __future__ import annotations
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """
    Single factory point for all LLM instantiation.
    Adding a new provider requires changes only in this class.

    All provider imports are deferred (inside if-branches) so that only the
    installed provider's LangChain package is required at runtime.
    """

    @staticmethod
    def create(
        provider    : str,
        model       : str,
        api_key     : Optional[str] = None,
        base_url    : Optional[str] = None,
        temperature : float = 0.0,
        max_tokens  : int   = 8000,
        llm_instance: Any   = None,
    ) -> BaseChatModel:
        """
        Instantiate and return a LangChain BaseChatModel.

        Parameters
        ----------
        provider     : One of "groq", "openai", "anthropic", "ollama" (case-insensitive).
        model        : Provider-specific model name.
        api_key      : API key. None is valid only for "ollama".
        base_url     : Base URL override. Required for "ollama".
        temperature  : Sampling temperature. Defaults to 0.
        max_tokens   : Maximum response tokens. Defaults to 8000.
        llm_instance : [AUDIT H4] If not None, return this object directly.
                       Used to inject MockLLM or pre-built LLM in tests.
                       The caller (agent factory) passes llm_cfg.llm_instance here.

        Returns
        -------
        BaseChatModel or llm_instance if provided.

        Raises
        ------
        ValueError : If provider is not in the supported list.
        """
        # Body logic:
        # 1. If llm_instance is not None: return llm_instance immediately.
        # 2. p = provider.lower().strip()
        # 3. if p == "groq":
        #        from langchain_groq import ChatGroq
        #        return ChatGroq(model=model, temperature=temperature,
        #                        api_key=api_key, max_tokens=max_tokens)
        # 4. elif p == "openai":
        #        from langchain_openai import ChatOpenAI
        #        return ChatOpenAI(model=model, temperature=temperature,
        #                          api_key=api_key, max_tokens=max_tokens)
        # 5. elif p == "anthropic":
        #        from langchain_anthropic import ChatAnthropic
        #        return ChatAnthropic(model=model, temperature=temperature,
        #                             api_key=api_key, max_tokens=max_tokens)
        # 6. elif p == "ollama":
        #        from langchain_ollama import ChatOllama   [AUDIT M11]
        #        return ChatOllama(model=model,
        #                          base_url=base_url or "http://localhost:11434",
        #                          temperature=temperature)
        #        # Note: ChatOllama does not accept max_tokens; uses num_predict.
        #        # langchain-ollama package must be installed separately.
        # 7. else:
        #        raise ValueError(f"Unknown LLM provider: '{provider}'. "
        #                         f"Supported: groq, openai, anthropic, ollama")
```

---

### `source_code/config/loaders.py` [M10, L15]

**Functions: `load_query`, `load_rules`, `load_client_config`, `load_pipeline_inputs`, `_build_initial_input`**

```python
from __future__ import annotations
import os
import pandas as pd
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary


def load_query(query_path: str) -> str:
    """
    Load SQL query string from a .sql file.

    Parameters
    ----------
    query_path : str
        Absolute or relative path to the .sql file.

    Returns
    -------
    str
        The full text of the SQL file.
    """
    # Body: open(query_path, "r", encoding="utf-8") and return f.read()


def load_rules(rules_path: str) -> str:
    """
    Load special rules string from a .txt file.

    Parameters
    ----------
    rules_path : str
        Absolute or relative path to the .txt file.

    Returns
    -------
    str
        The full text of the rules file.
    """
    # Body: open(rules_path, "r", encoding="utf-8") and return f.read()


def load_client_config(
    client_id : str,
    base_dir  : str = "clients/",
    target_column: str = "",
) -> dict:
    """
    Load pipeline inputs for a named client from a conventional directory layout.

    Expected layout:
        <base_dir>/<client_id>/data.csv
        <base_dir>/<client_id>/query.sql
        <base_dir>/<client_id>/rules.txt

    Parameters
    ----------
    client_id     : str  — Client identifier (directory name under base_dir).
    base_dir      : str  — Root directory containing client subdirectories.
    target_column : str  — The name of the target column (e.g. "ChurnFlag"). [AUDIT M10]

    Returns
    -------
    dict
        AgentState-compatible initial input dict.
    """
    # Body: client_dir = os.path.join(base_dir, client_id)
    # return _build_initial_input(
    #     data_path=os.path.join(client_dir, "data.csv"),
    #     query_path=os.path.join(client_dir, "query.sql"),
    #     rules_path=os.path.join(client_dir, "rules.txt"),
    #     client_id=client_id,
    #     target_column=target_column,
    # )


def load_pipeline_inputs(
    data_path    : str,
    query_path   : str,
    rules_path   : str,
    target_column: str,
    client_id    : str = "default",
) -> dict:
    """
    Load pipeline inputs from explicit file paths.
    Drop-in replacement for the hardcoded reads in main.py.

    Parameters
    ----------
    data_path     : str  — Path to the raw CSV dataset.
    query_path    : str  — Path to the .sql file.
    rules_path    : str  — Path to the special rules .txt file.
    target_column : str  — Name of the target column. [AUDIT M10, L15]
                           Required. Callers must supply this explicitly —
                           there is no default. Pass "ChurnFlag" for legacy usage.
    client_id     : str  — Optional client identifier for logging. Defaults to "default".

    Returns
    -------
    dict
        AgentState-compatible initial input dict.

    Note
    ----
    [AUDIT L15] This function takes 4 positional parameters (plus client_id).
    The architecture digest incorrectly listed 3 params; this plan corrects it.
    """
    # Body: delegate to _build_initial_input(...)


def _build_initial_input(
    data_path    : str,
    query_path   : str,
    rules_path   : str,
    client_id    : str,
    target_column: str,
) -> dict:
    """
    Internal builder. Reads files, classifies columns, builds the full
    AgentState-compatible initial input dict.

    Parameters
    ----------
    data_path     : str  — Path to raw CSV.
    query_path    : str  — Path to .sql file.
    rules_path    : str  — Path to rules .txt file.
    client_id     : str  — Client identifier (for logging only).
    target_column : str  — Required target column name. [AUDIT M10]
                           No empty-string default. Callers must supply.

    Returns
    -------
    dict with keys:
        file_path, target_column, sql_query, df_columns, special_rules,
        iteration_count, metadata_summary, categorical_cols,
        numeric_like_cols, true_numeric_cols, value_counts_summary,
        null_summary.

    Note
    ----
    The values for categorical_cols, numeric_like_cols, true_numeric_cols,
    value_counts_summary, and null_summary computed here are based on the
    PRE-RENAME DataFrame. They will be overwritten by reclassify_columns_node
    after executor1 runs. Supplying them here ensures AgentState is fully
    populated for Agent 1's execution.
    """
    # Body:
    # sql_query = load_query(query_path)
    # special_rules = load_rules(rules_path)
    # df = pd.read_csv(data_path)
    # dtypes_dict = df.dtypes.astype(str).to_dict()
    # sample_row = df.head(1).to_dict(orient="records")[0]
    # metadata_summary = (
    #     f"DATASET PROFILE:\n"
    #     f"1. COLUMNS & TYPES:\n{dtypes_dict}\n"
    #     f"2. SAMPLE ROW:\n{sample_row}"
    # )
    # categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    # value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    # null_summary = build_null_summary(df)
    # return {
    #     "file_path"           : data_path,
    #     "target_column"       : target_column,   # [AUDIT M10] explicit, not ""
    #     "sql_query"           : sql_query,
    #     "df_columns"          : list(df.columns),
    #     "special_rules"       : special_rules,
    #     "iteration_count"     : 0,
    #     "metadata_summary"    : metadata_summary,
    #     "categorical_cols"    : categorical_cols,
    #     "numeric_like_cols"   : numeric_like_cols,
    #     "true_numeric_cols"   : true_numeric_cols,
    #     "value_counts_summary": value_counts_summary,
    #     "null_summary"        : null_summary,
    # }
```

---

### `source_code/reclassify.py` [C1, M8]

**Function: `reclassify_columns_node`**

```python
import pandas as pd
from source_code.state import AgentState
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary


def reclassify_columns_node(state: AgentState) -> dict:
    """
    Graph node: re-runs column classification after Agent 1 renames columns.

    Reads the post-rename CSV from state["output_path"], applies classify_columns,
    build_value_counts_summary, and build_null_summary, and returns updated
    metadata fields so Agent 2 sees the correctly-named columns.

    Must be placed in graph: executor1 → reclassify_columns → agent2_cleaner.

    Parameters
    ----------
    state : AgentState
        Current pipeline state. Must contain output_path set by rename_executor_agent.

    Returns
    -------
    dict
        Partial state update with keys:
            categorical_cols, numeric_like_cols, true_numeric_cols,
            value_counts_summary, null_summary.

    Notes
    -----
    [AUDIT C1] The fallback for output_path is "standardized_output_renamed.csv"
    (matches rename_executor_agent's default). Architecture had "standardized_output.csv"
    which would read a non-existent file.

    If output_path is absent from state AND the fallback file does not exist,
    this node raises FileNotFoundError immediately (fail-fast, no silent corruption).
    """
    # Body:
    # output_path = state.get("output_path", "standardized_output_renamed.csv")
    # df = pd.read_csv(output_path)   # raises FileNotFoundError if file missing
    # categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    # value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    # null_summary = build_null_summary(df)
    # return {
    #     "categorical_cols"    : categorical_cols,
    #     "numeric_like_cols"   : numeric_like_cols,
    #     "true_numeric_cols"   : true_numeric_cols,
    #     "value_counts_summary": value_counts_summary,
    #     "null_summary"        : null_summary,
    # }
```

---

### `tests/__init__.py`

**Content:** Empty file. Marks `tests/` as a Python package.

---

### `tests/agents/__init__.py`

**Content:** Empty file. Marks `tests/agents/` as a Python package.

---

### `tests/mock_llm.py` [H3]

**Classes: `MockResponse`, `MockLLM`**

```python
from __future__ import annotations
from typing import Any, Iterator, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockResponse:
    """
    Minimal stand-in for LangChain's AIMessage.
    Exposes only .content, which is what all current agent code reads.

    Parameters
    ----------
    content : str
        The pre-programmed response string.
    """
    def __init__(self, content: str) -> None:
        self.content: str = content


class MockLLM(BaseChatModel):
    """
    Test double for any LangChain BaseChatModel.
    Returns a pre-programmed response string without making network calls.

    Inherits BaseChatModel to satisfy type annotations and LangChain
    introspection. Stubs _generate and _llm_type as required by the ABC.
    [AUDIT H3]

    Attributes
    ----------
    last_prompt : Any
        The exact value passed to the most recent .invoke() call.
        None if .invoke() has not been called yet.
        Used in tests to assert that prompt context (SQL, metadata) was
        correctly injected before the LLM call.

    Usage
    -----
        mock = MockLLM(content="```python\\nrename_map = {}\\n```")
        cfg = PipelineConfig.with_mock(mock)
        agent_fn = make_field_renamer_agent(cfg)
        result = agent_fn(state)
        assert mock.last_prompt is not None
        assert "SELECT" in str(mock.last_prompt)
    """

    # Pydantic field for the pre-programmed content.
    # BaseChatModel uses pydantic; declare the custom field here.
    _content: str = ""

    def __init__(self, content: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        content : str
            The string that .invoke() will return as MockResponse.content.
        """
        # Body:
        # super().__init__(**kwargs)
        # self._content = content
        # self.last_prompt = None   # NOT a pydantic field; set as instance attr

    @property
    def _llm_type(self) -> str:
        """
        Required by BaseChatModel ABC.
        Returns "mock" to identify this as a test double.
        """
        # Body: return "mock"

    def _generate(
        self,
        messages  : List[BaseMessage],
        stop      : Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs  : Any,
    ) -> ChatResult:
        """
        Required by BaseChatModel ABC.
        Records the messages (for last_prompt) and returns the pre-programmed content.

        Parameters
        ----------
        messages    : List[BaseMessage]  — The prompt messages passed by LangChain.
        stop        : Optional stop sequences (ignored).
        run_manager : LangChain callback manager (ignored).

        Returns
        -------
        ChatResult
            Wraps self._content in a ChatGeneration / AIMessage.
        """
        # Body:
        # self.last_prompt = messages
        # message = AIMessage(content=self._content)
        # generation = ChatGeneration(message=message)
        # return ChatResult(generations=[generation])

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> MockResponse:
        """
        Override invoke() to record last_prompt and return MockResponse.
        This is what agent code calls directly (llm.invoke(prompt)).

        Parameters
        ----------
        input : Any
            The prompt. May be a str (current agents) or List[BaseMessage] (future).

        Returns
        -------
        MockResponse
            Object with .content = self._content.
        """
        # Body:
        # self.last_prompt = input
        # return MockResponse(self._content)
```

---

### `tests/run_tests.py` [M9]

**Functions: `parse_args`, `check_dependencies`, `run_pytest`, `generate_review_document`, `_render_summary_table`, `_render_agent_table`, `_render_failure_block`, `main`**

```python
#!/usr/bin/env python
"""
CLI wrapper: runs pytest, reads JSON report, generates markdown review document.

Usage:
    python tests/run_tests.py             # run all tests
    python tests/run_tests.py --agent 1   # run only agent1-marked tests
    python tests/run_tests.py --agent 2   # run only agent2-marked tests
"""
from __future__ import annotations
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def check_dependencies() -> None:
    """
    Startup guard: verify that pytest-json-report is installed.
    [AUDIT M9] Prints clear install instruction and exits(1) if missing.

    Body logic:
    - try: import pytest_jsonreport
    - except ImportError:
          print("ERROR: pytest-json-report is not installed.")
          print("Install with: pip install pytest-json-report")
          sys.exit(1)
    """


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace with attributes:
        agent : Optional[int]  — 1 or 2 if --agent was passed, else None.

    Body logic:
    - parser = argparse.ArgumentParser()
    - parser.add_argument("--agent", type=int, choices=[1, 2], default=None)
    - return parser.parse_args()
    """


def run_pytest(agent_filter: int | None) -> tuple[int, Path]:
    """
    Invoke pytest programmatically and write JSON report.

    Parameters
    ----------
    agent_filter : int or None
        If 1 or 2, adds -m "agent1" or -m "agent2" marker filter.
        If None, runs all tests.

    Returns
    -------
    tuple[int, Path]
        (pytest exit code, path to the generated JSON report file)

    Body logic:
    - report_path = Path("tests/.pytest_report.json")
    - args = [
          "tests/agents/",
          "-v",
          "--tb=short",
          "--json-report",
          f"--json-report-file={report_path}",
      ]
    - if agent_filter is not None:
          args += ["-m", f"agent{agent_filter}"]
    - import pytest
    - exit_code = pytest.main(args)
    - return exit_code, report_path
    """


def generate_review_document(report_path: Path, command: str) -> Path:
    """
    Read the pytest JSON report and render a structured markdown document.

    Parameters
    ----------
    report_path : Path  — Path to the .pytest_report.json file.
    command     : str   — The CLI command that was run (for the report header).

    Returns
    -------
    Path
        Path to the written markdown file under tests/reports/.

    Body logic:
    - Load JSON from report_path.
    - Count PASSED, FAILED, XFAIL (→ WARNING in report), TOTAL.
    - [AUDIT L14] Map XFAIL status → "WARNING" in markdown output.
    - Build timestamp string: datetime.now().strftime("%Y%m%d_%H%M%S")
    - Try to get git commit hash: subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
      Catch CalledProcessError and use "unknown" if git is unavailable.
    - Build markdown sections:
        - Header block (generated date, command, commit)
        - Summary table (PASSED, FAILED, WARNING, TOTAL counts)
        - Per-agent section tables (TC1–TC9, TC10–TC17, TC18–TC23)
        - For each FAILED test: render full failure block (8 required fields).
        - For each XFAIL test: render abbreviated WARNING block.
    - Write to Path("tests/reports") / f"review_{timestamp}.md"
    - Print the output path to stdout.
    - Return the output path.
    """


def _render_summary_table(counts: dict) -> str:
    """
    Render the markdown summary table.

    Parameters
    ----------
    counts : dict with keys "PASSED", "FAILED", "WARNING", "TOTAL"

    Returns
    -------
    str — Markdown table string.
    """


def _render_agent_table(tests: list[dict], agent_label: str) -> str:
    """
    Render per-agent result table.

    Parameters
    ----------
    tests       : list of dicts, each with keys: tc_id, name, category, status
    agent_label : str — e.g. "Agent 1 — Field Renamer"

    Returns
    -------
    str — Markdown section with H2 header + table.
    """


def _render_failure_block(test: dict) -> str:
    """
    Render the full 8-field failure block for a FAILED test.

    Parameters
    ----------
    test : dict with keys:
        tc_id, name, agent, category, fixture, expected, received,
        probable_cause, suggested_fix

    Returns
    -------
    str — Markdown block (H3 + 8 fields).
    """


def main() -> None:
    """
    Entry point. Orchestrates: check_dependencies → parse_args → run_pytest →
    generate_review_document.
    """
    # Body:
    # check_dependencies()
    # args = parse_args()
    # command = "python tests/run_tests.py" + (f" --agent {args.agent}" if args.agent else "")
    # exit_code, report_path = run_pytest(args.agent)
    # if report_path.exists():
    #     out = generate_review_document(report_path, command)
    #     print(f"Review written to: {out}")
    # sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

---

### `tests/agents/test_agent_1.py`

**Test functions: TC1–TC9**

```python
"""
Agent 1 — Field Renamer: 9 test cases.

All tests use pytest fixtures and the MockLLM test double.
Fixture responses are loaded from tests/fixtures/agent_1/scenarios.json.
Agent under test: make_field_renamer_agent(config) from agent_1_field_renamer.py.
"""
import json
import pytest
from pathlib import Path
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent

# Load all scenarios once at module level.
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "agent_1" / "scenarios.json"
SCENARIOS = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

# Shared AgentState dict used by all Agent 1 tests.
# Columns match sample_telecom.csv with alias prefixes (a.msisdn etc.)
BASE_STATE = {
    "file_path"       : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "target_column"   : "ChurnFlag",
    "sql_query"       : "SELECT a.msisdn, a.rev_30d, a.mou_out_30d FROM churn_table a",
    "df_columns"      : ["a.msisdn", "gender", "rev_30d", "mou_out_30d",
                         "data_vol_30d", "churn_flag", "tenure_months",
                         "region_cd", "plan_type", "contract_end_dt"],
    "metadata_summary": "DATASET PROFILE:\n1. COLUMNS & TYPES:\n{...}\n2. SAMPLE ROW:\n{...}",
    "special_rules"   : "None",
    "iteration_count" : 0,
}


def _make_agent(scenario_key: str):
    """Helper: build agent from fixture scenario key."""
    # mock = MockLLM(content=SCENARIOS[scenario_key]["llm_response"])
    # cfg = PipelineConfig.with_mock(mock)
    # return make_field_renamer_agent(cfg), mock


@pytest.mark.agent1
def test_tc1_happy_path():
    """
    TC1 — Happy Path.
    LLM returns a valid two-block response. All 10 columns covered.

    Assertions (from expected_assertions):
    - result["column_map"] has exactly 10 entries
    - result["ambiguous_fields"] == []
    - result["cleaning_code"] contains "df.rename"
    - mock.last_prompt is not None (prompt was sent)
    """


@pytest.mark.agent1
def test_tc2_sql_context_alias_resolution():
    """
    TC2 — SQL Query Context.
    LLM correctly resolves alias columns using the injected SQL query.

    Assertions (from expected_assertions):
    - result["column_map"].get("rev_30d") == "RevenueLast30d"
    - "SELECT" in str(mock.last_prompt)   (SQL was injected into prompt)
    - result["ambiguous_fields"] == []
    """


@pytest.mark.agent1
def test_tc3_missing_second_block():
    """
    TC3 — Missing Second Code Block.
    LLM returns only one code block (no ambiguous_fields block).

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["ambiguous_fields"] == []   (defaults to empty, not None)
    - result["column_map"] is not None
    """


@pytest.mark.agent1
def test_tc4_no_code_blocks():
    """
    TC4 — No Code Blocks.
    LLM returns plain prose with no fenced code blocks.

    Assertions (from expected_assertions):
    - result["cleaning_code"] == ""
    - result["ambiguous_fields"] == []
    - result["column_map"] == {}
    """


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: duplicate rename_map values not detected")
def test_tc5_duplicate_values_in_rename_map():
    """
    TC5 — Duplicate Values in rename_map.
    [AUDIT L14] marked xfail(strict=False) → maps to WARNING in report.
    Two source columns map to the same clean name. Known gap: agent does not detect.

    Assertions (from expected_assertions):
    - result["column_map"] has entries  (non-empty)
    - No exception raised during agent execution
    - assert len(set(result["column_map"].values())) == len(result["column_map"].values())
      (this assertion WILL FAIL for the duplicate fixture, producing XFAIL)
    """


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: partial rename_map silently accepted")
def test_tc6_missing_columns_from_rename_map():
    """
    TC6 — Missing Columns from rename_map.
    [AUDIT L14] xfail(strict=False) → WARNING in report.
    Only 2 of 10 columns returned in rename_map.

    Assertions (from expected_assertions):
    - result["column_map"] has exactly 2 entries
    - assert len(result["column_map"]) == len(BASE_STATE["df_columns"])
      (this assertion WILL FAIL, producing XFAIL)
    """


@pytest.mark.agent1
def test_tc7_ambiguous_fields_missing_keys():
    """
    TC7 — ambiguous_fields Entry Missing Keys.
    LLM returns ambiguous_fields with only "original_column", no "candidates" or "reason".

    Assertions (from expected_assertions):
    - result["ambiguous_fields"] is a non-empty list
    - result["ambiguous_fields"][0].get("candidates") is None   (no crash)
    - result["ambiguous_fields"][0].get("reason") is None       (no crash)
    """


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: rename_map keys not validated against df columns")
def test_tc8_keys_dont_match_df_columns():
    """
    TC8 — rename_map Keys Don't Match df Columns.
    [AUDIT L14] xfail(strict=False) → WARNING in report.
    LLM uses column names not present in the test DataFrame.

    Assertions (from expected_assertions):
    - result["column_map"] is non-empty
    - No exception raised
    - assert all(k in BASE_STATE["df_columns_cleaned"] for k in result["column_map"])
      (this assertion WILL FAIL, producing XFAIL)
    """


@pytest.mark.agent1
def test_tc9_malformed_python():
    """
    TC9 — Malformed Python in Code Block.
    LLM returns a syntax error in the rename block (missing closing brace).

    Assertions (from expected_assertions):
    - result["column_map"] == {}   (exec fails, column_map defaults to {})
    - result["cleaning_code"] != ""  (the raw code string is still stored)
    - No unhandled exception propagates from the agent
    """
```

---

### `tests/agents/test_agent_2.py`

**Test functions: TC10–TC17**

```python
"""
Agent 2 — Field Cleaner: 8 test cases.

All tests use pytest fixtures and the MockLLM test double.
Fixture responses are loaded from tests/fixtures/agent_2/scenarios.json.
Agent under test: make_field_cleaner_agent(config) from agent_2_field_cleaner.py.
"""
import json
import pytest
from pathlib import Path
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_2_field_cleaner import make_field_cleaner_agent

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "agent_2" / "scenarios.json"
SCENARIOS = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

# State after rename. Columns are standardised (no alias prefixes).
BASE_STATE = {
    "file_path"           : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "output_path"         : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "target_column"       : "ChurnFlag",
    "metadata_summary"    : "DATASET PROFILE:\n1. COLUMNS & TYPES:\n{...}\n2. SAMPLE ROW:\n{...}",
    "categorical_cols"    : ["Gender", "RegionCode", "PlanType"],
    "numeric_like_cols"   : ["RevenueLast30d"],
    "true_numeric_cols"   : ["ChurnFlag", "TenureMonths", "MouOutgoingLast30d",
                             "DataUsageLast30dKb", "ContractEndDate"],
    "value_counts_summary": "Gender:\nMale    20\nM       10\nFemale  15\nF       5\n",
    "null_summary"        : "Gender    3\nRevenueLast30d    5",
    "special_rules"       : "None",
    "iteration_count"     : 0,
    "sql_query"           : "",   # Agent 2 does not use sql_query [architecture §12]
}


def _make_agent(scenario_key: str):
    """Helper: build agent from fixture scenario."""


@pytest.mark.agent2
def test_tc10_happy_path():
    """
    TC10 — Happy Path.
    Valid two-block response with cleaning code and empty flagged_columns.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["flagged_columns"] == []
    - result["cleaning_code"] contains "df['Gender']"
    - result["output_path"] == "output_agent2_cleaned.csv"
    """


@pytest.mark.agent2
def test_tc11_metadata_context_injection():
    """
    TC11 — Metadata Context Injection.
    Verifies that value_counts_summary and null_summary appear in the prompt.
    [Architecture §12: Agent 2 does NOT receive sql_query]

    Assertions (from expected_assertions):
    - "Gender" in str(mock.last_prompt)               (value_counts injected)
    - "null" in str(mock.last_prompt).lower()         (null_summary injected)
    - "SELECT" not in str(mock.last_prompt)           (sql_query NOT injected)
    """


@pytest.mark.agent2
def test_tc12_missing_second_block():
    """
    TC12 — Missing Second Code Block.
    LLM returns only cleaning code, no flagged_columns block.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["flagged_columns"] == []   (defaults to empty, not None)
    """


@pytest.mark.agent2
def test_tc13_no_code_blocks():
    """
    TC13 — No Code Blocks.
    LLM returns plain prose.

    Assertions (from expected_assertions):
    - result["cleaning_code"] == ""
    - result["flagged_columns"] == []
    """


@pytest.mark.agent2
def test_tc14_flagged_columns_missing_keys():
    """
    TC14 — flagged_columns Entry Missing Keys.
    LLM returns flagged_columns with only "column", no "reason".

    Assertions (from expected_assertions):
    - result["flagged_columns"] is a non-empty list
    - result["flagged_columns"][0].get("reason") is None   (no crash)
    """


@pytest.mark.agent2
@pytest.mark.xfail(strict=False, reason="documented gap: agent does not detect null introduction")
def test_tc15_cleaning_code_introduces_new_nulls():
    """
    TC15 — Cleaning Code Introduces New Nulls.
    [AUDIT L14] xfail(strict=False) → WARNING in report.
    The cleaning code replaces valid values with None, increasing null count.
    The executor should detect this via new_nulls check.

    Assertions (from expected_assertions):
    - result["error_log"] is not None
    - "WARNING" in result["error_log"]   (executor's null detection fires)
    This test requires executor integration — call executor after agent.
    """


@pytest.mark.agent2
@pytest.mark.xfail(strict=False, reason="documented gap: agent does not detect column drops")
def test_tc16_cleaning_code_drops_columns():
    """
    TC16 — Cleaning Code Drops Columns.
    [AUDIT L14] xfail(strict=False) → WARNING in report.
    The cleaning code contains df.drop().

    Assertions (from expected_assertions):
    - result["error_log"] is not None
    - "WARNING" in result["error_log"]   (executor's column-drop detection fires)
    This test requires executor integration — call executor after agent.
    """


@pytest.mark.agent2
def test_tc17_malformed_python():
    """
    TC17 — Malformed Python.
    LLM returns a syntax error in cleaning code block.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""   (raw code stored even if malformed)
    - No unhandled exception from agent
    - When executor runs this code: result["error_log"] contains "SyntaxError"
    """
```

---

### `tests/agents/test_executor.py`

**Test functions: TC18–TC23**

```python
"""
Executor test cases: TC18–TC23.
All executor tests use tmp_path fixture to avoid file-system collisions. [AUDIT H5]
Tests call rename_executor_agent and cleaning_executor_agent directly.
"""
import pandas as pd
import pytest
from pathlib import Path


@pytest.mark.executor
def test_tc18_happy_path_rename(tmp_path):
    """
    TC18 — Happy Path (rename executor).
    state["column_map"] = {"msisdn": "Msisdn", "gender": "Gender"}.
    Input CSV has columns ["a.msisdn", "gender"].
    Expected: output CSV has columns ["Msisdn", "Gender"].

    [AUDIT H5] output_path = str(tmp_path / "output_rename.csv")

    Assertions (from expected_assertions):
    - result["error_log"] is None
    - result["output_path"] == str(tmp_path / "output_rename.csv")
    - pd.read_csv(result["output_path"]).columns.tolist() == ["Msisdn", "Gender"]
    """


@pytest.mark.executor
def test_tc19_empty_cleaning_code(tmp_path):
    """
    TC19 — Empty cleaning_code.
    state["cleaning_code"] = "".
    Expected: error_log is non-None with an informative message.

    [AUDIT H5] tmp_path used for output isolation.
    """


@pytest.mark.executor
def test_tc20_code_raises_runtime_exception(tmp_path):
    """
    TC20 — Code Raises Runtime Exception.
    state["cleaning_code"] contains code that raises ZeroDivisionError.
    Expected: result["error_log"] contains "ZeroDivisionError".

    [AUDIT H5] tmp_path used.
    """


@pytest.mark.executor
@pytest.mark.xfail(strict=False, reason="documented gap: executor does not reject column-dropping code")
def test_tc21_code_drops_column(tmp_path):
    """
    TC21 — Code Drops a Column. [AUDIT L14] xfail(strict=False).
    cleaning_code = "df = df.drop(columns=['gender'])".
    Expected: result["error_log"] contains "WARNING" about dropped columns.

    [AUDIT H5] tmp_path used.
    """


@pytest.mark.executor
@pytest.mark.xfail(strict=False, reason="documented gap: executor warns but does not reject null-introducing code")
def test_tc22_code_introduces_new_nulls(tmp_path):
    """
    TC22 — Code Introduces New Nulls. [AUDIT L14] xfail(strict=False).
    cleaning_code replaces valid values with None.
    Expected: result["error_log"] contains "WARNING" about new nulls.

    [AUDIT H5] tmp_path used.
    """


@pytest.mark.executor
def test_tc23_output_file_written_correctly(tmp_path):
    """
    TC23 — Output File Written Correctly.
    After a successful run, verifies the output CSV exists and has the correct shape.

    [AUDIT H5] output_path = str(tmp_path / "out.csv")

    Assertions (from expected_assertions):
    - Path(result["output_path"]).exists() is True
    - pd.read_csv(result["output_path"]).shape[0] > 0
    - result["error_log"] is None
    """
```

---

### `tests/reports/.gitkeep`

**Content:** Empty file. Ensures the `reports/` directory is tracked by git. Reports generated at runtime are gitignored.

---

## 3. Files to MODIFY — Old vs New Signatures

---

### `source_code/agents/agent_1_field_renamer.py`

**Summary of changes:**
1. Remove hardcoded `ChatGroq` instantiation.
2. Wrap `field_renamer_agent` inside `make_field_renamer_agent(config)` factory.
3. Fix all `exec()` calls to use `{"__builtins__": __builtins__}` globals. [AUDIT L12]
4. Keep all parsing logic identical — only LLM construction changes.

**OLD signature (line 12):**
```python
def field_renamer_agent(state: AgentState) -> dict:
    # hardcodes ChatGroq at lines 37-42
    # exec(ambiguous_code, {}, local_ns)  at line 64
    # exec(map_code, {}, local_ns)        at line 74
```

**NEW signatures:**
```python
from source_code.config.llm_config import PipelineConfig
from source_code.config.llm_factory import LLMFactory

def make_field_renamer_agent(config: PipelineConfig):
    """
    Factory function. Returns a LangGraph-compatible node function
    with the LLM resolved from config and closed over.

    Parameters
    ----------
    config : PipelineConfig
        Provides LLM parameters. The LLM is instantiated once at factory
        call time, then closed over in the returned function.

    Returns
    -------
    Callable[[AgentState], dict]
        LangGraph node function. Identical to old field_renamer_agent
        except the LLM comes from config instead of being hardcoded.
    """
    # Body:
    # llm_cfg = config.get_llm_config("agent1")
    # llm = LLMFactory.create(
    #     provider=llm_cfg.provider, model=llm_cfg.model,
    #     api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
    #     temperature=llm_cfg.temperature, max_tokens=llm_cfg.max_tokens,
    #     llm_instance=llm_cfg.llm_instance,   # [AUDIT H4]
    # )

    def field_renamer_agent(state: AgentState) -> dict:
        """LangGraph node: field renamer. Logic identical to original."""
        # All original logic preserved. Changes:
        # 1. Remove: llm = ChatGroq(...) block (lines 37-42)
        # 2. Use: the closed-over `llm` variable
        # 3. Fix exec: exec(ambiguous_code, {"__builtins__": __builtins__}, local_ns) [AUDIT L12]
        # 4. Fix exec: exec(map_code, {"__builtins__": __builtins__}, local_ns) [AUDIT L12]
        # 5. Remove: from langchain_groq import ChatGroq
        # 6. Remove: import os (not needed after removing ChatGroq)
        ...

    return field_renamer_agent
```

**Exact exec() changes at lines 64 and 74:**

OLD (line 64): `exec(ambiguous_code, {}, local_ns)`
NEW: `exec(ambiguous_code, {"__builtins__": __builtins__}, local_ns)`

OLD (line 74): `exec(map_code, {}, local_ns)`
NEW: `exec(map_code, {"__builtins__": __builtins__}, local_ns)`

**Imports to REMOVE from top of file:**
- `import os`
- `from langchain_groq import ChatGroq`

**Imports to ADD:**
- `from source_code.config.llm_config import PipelineConfig`
- `from source_code.config.llm_factory import LLMFactory`
- `from langchain_core.language_models import BaseChatModel` (for type annotation on closed-over llm)

**Module-level variable unchanged:**
`PROMPTS_CONFIG = load_prompts(r'source_code/prompts/agent_1.json')` — kept as-is.

---

### `source_code/agents/agent_2_field_cleaner.py`

**Summary of changes:**
1. Remove hardcoded `ChatGroq` instantiation.
2. Wrap `field_cleaner_agent` inside `make_field_cleaner_agent(config)` factory.
3. Fix `exec(flagged_code, {}, local_ns)` at line 90. [AUDIT L12]
4. Keep all other logic identical.

**OLD signature (line 10):**
```python
def field_cleaner_agent(state: AgentState) -> dict:
    # hardcodes ChatGroq at lines 37-42
    # exec(flagged_code, {}, local_ns) at line 90
```

**NEW signatures:**
```python
from source_code.config.llm_config import PipelineConfig
from source_code.config.llm_factory import LLMFactory

def make_field_cleaner_agent(config: PipelineConfig):
    """
    Factory function. Returns a LangGraph-compatible node function
    with the LLM resolved from config and closed over.

    Parameters
    ----------
    config : PipelineConfig
        Provides LLM parameters for Agent 2.

    Returns
    -------
    Callable[[AgentState], dict]
    """
    # Body:
    # llm_cfg = config.get_llm_config("agent2")
    # llm = LLMFactory.create(
    #     provider=llm_cfg.provider, model=llm_cfg.model,
    #     api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
    #     temperature=llm_cfg.temperature, max_tokens=llm_cfg.max_tokens,
    #     llm_instance=llm_cfg.llm_instance,
    # )

    def field_cleaner_agent(state: AgentState) -> dict:
        """LangGraph node: field cleaner. Logic identical to original."""
        # 1. Remove: llm = ChatGroq(...) block (lines 37-42)
        # 2. Use: closed-over `llm`
        # 3. Fix exec: exec(flagged_code, {"__builtins__": __builtins__}, local_ns) [AUDIT L12]
        ...

    return field_cleaner_agent
```

**Exact exec() change at line 90:**

OLD: `exec(flagged_code, {}, local_ns)`
NEW: `exec(flagged_code, {"__builtins__": __builtins__}, local_ns)`

**Imports to REMOVE:** `import os`, `from langchain_groq import ChatGroq`

**Imports to ADD:** `from source_code.config.llm_config import PipelineConfig`, `from source_code.config.llm_factory import LLMFactory`

---

### `source_code/agents/executor.py` — Full Rewrite [C1, C2, L12, L13]

**Summary:** Replace single `code_executor_agent` with `rename_executor_agent` + `cleaning_executor_agent` + private `_run_exec` helper. Fix exec globals. Add safety checks. Return `output_path` in state.

**OLD exports:** `code_executor_agent`

**NEW exports:** `rename_executor_agent`, `cleaning_executor_agent`

```python
import pandas as pd
import numpy as np
import traceback
from source_code.state import AgentState
from source_code.utils import preprocess_column_names


def rename_executor_agent(state: AgentState) -> dict:
    """
    Executor for Agent 1: applies composite column rename map to the raw CSV.

    This executor does NOT run exec(cleaning_code). [AUDIT C2]
    It reconstructs the rename from the audit-trail dict (state["column_map"])
    and preprocess_column_names, avoiding the double-exec risk entirely.

    Read from state:
        file_path   : str  — original raw CSV (pre-rename). [AUDIT L13]
        column_map  : dict — {cleaned_name: standardized_name} from Agent 1.

    Writes to state:
        output_path : str  — path where renamed CSV was saved.
        error_log   : Optional[str]  — None on success, error string on failure.

    Parameters
    ----------
    state : AgentState

    Returns
    -------
    dict with keys: output_path, error_log

    Output file default: "standardized_output_renamed.csv" [AUDIT C1]

    Safety checks (after rename):
    - Column drop detection: appends WARNING to error_log if any columns dropped.
      (rename should never drop columns, so this is a sanity check.)
    - New null detection: appends WARNING to error_log if rename introduces nulls.
    """
    # Body:
    # file_path = state["file_path"]                 [AUDIT L13] reads original file
    # column_map = state.get("column_map", {})
    # output_path = state.get("output_path", "standardized_output_renamed.csv")
    #
    # try:
    #     df = pd.read_csv(file_path)
    #     df_before = df.copy()
    #
    #     if column_map:
    #         pre_cleaned = preprocess_column_names(list(df.columns))
    #         composite_map = {
    #             orig: column_map[cleaned]
    #             for orig, cleaned in pre_cleaned.items()
    #             if cleaned in column_map
    #         }
    #         df = df.rename(columns=composite_map)
    #
    #     warnings = _check_safety(df_before, df)
    #     df.to_csv(output_path, index=False)
    #
    #     error_log = "\n".join(warnings) if warnings else None
    #     return {"output_path": output_path, "error_log": error_log}
    #
    # except Exception as e:
    #     error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    #     return {"error_log": error_msg}


def cleaning_executor_agent(state: AgentState) -> dict:
    """
    Executor for Agent 2: runs cleaning code against the post-rename CSV.

    Reads the DataFrame from state["output_path"] (set by rename_executor_agent). [AUDIT L13]
    Does NOT read from state["file_path"] (the original CSV).

    Read from state:
        output_path   : str  — path to post-rename CSV (written by executor1).
        cleaning_code : str  — Python cleaning code from Agent 2.

    Writes to state:
        output_path : str  — path where cleaned CSV was saved (different from input).
        error_log   : Optional[str]

    Default output path: "output_agent2_cleaned.csv"

    exec() globals include __builtins__, pd, np to support import statements. [AUDIT L12]

    Safety checks (after exec):
    - Column drop detection: WARNING in error_log if columns dropped.
    - New null detection: WARNING in error_log if null count increased.
    """
    # Body:
    # input_path = state.get("output_path", "standardized_output_renamed.csv")
    # cleaning_code = state.get("cleaning_code", "")
    # final_output_path = "output_agent2_cleaned.csv"
    #
    # if not cleaning_code:
    #     return {"error_log": "No cleaning code provided by the agent."}
    #
    # try:
    #     df = pd.read_csv(input_path)        [AUDIT L13] reads from output_path, not file_path
    #     df_before = df.copy()
    #
    #     local_vars = {"df": df, "pd": pd}
    #     exec_globals = {"__builtins__": __builtins__, "pd": pd, "np": np}  [AUDIT L12]
    #     exec(cleaning_code, exec_globals, local_vars)
    #     df_clean = local_vars["df"]
    #
    #     warnings = _check_safety(df_before, df_clean)
    #     df_clean.to_csv(final_output_path, index=False)
    #
    #     error_log = "\n".join(warnings) if warnings else None
    #     return {"output_path": final_output_path, "error_log": error_log}
    #
    # except Exception as e:
    #     error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    #     return {"error_log": error_msg}


def _check_safety(df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[str]:
    """
    Run post-exec safety checks. Returns list of WARNING strings.

    Checks:
    1. Column drop: set(df_before.columns) - set(df_after.columns)
       If non-empty: append "WARNING: columns dropped: {dropped}"
    2. New nulls: df_after.isna().sum().sum() - df_before.isna().sum().sum()
       If positive: append "WARNING: {n} new null values introduced"

    Parameters
    ----------
    df_before : pd.DataFrame  — DataFrame snapshot before exec.
    df_after  : pd.DataFrame  — DataFrame after exec.

    Returns
    -------
    list[str]  — Empty if no issues; otherwise one string per warning.
    """
    # Body:
    # warnings = []
    # dropped = set(df_before.columns) - set(df_after.columns)
    # if dropped:
    #     warnings.append(f"WARNING: columns dropped: {sorted(dropped)}")
    # new_nulls = df_after.isna().sum().sum() - df_before.isna().sum().sum()
    # if new_nulls > 0:
    #     warnings.append(f"WARNING: {new_nulls} new null values introduced")
    # return warnings
```

---

### `source_code/graph.py` — Full Rewrite [H6]

**Summary:** Wrap graph construction in `build_graph(config)` factory. Add `reclassify_columns` node. Import new executor functions. Update node assignments.

**OLD exports:** `ds_machine` (module-level compiled graph)

**NEW exports:** `build_graph` function only. No module-level `ds_machine`.

```python
from langgraph.graph import StateGraph, END
from source_code.state import AgentState
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent
from source_code.agents.agent_2_field_cleaner import make_field_cleaner_agent
from source_code.agents.executor import rename_executor_agent, cleaning_executor_agent
from source_code.reclassify import reclassify_columns_node   # [AUDIT M8] top-level, not agents/


def build_graph(config: PipelineConfig):
    """
    Build and compile the DS Machine LangGraph workflow.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration. Injected into agent factory closures.
        All LLM construction happens at factory call time (graph compile time).

    Returns
    -------
    CompiledStateGraph
        Ready-to-stream graph. Call .stream(initial_input) or .invoke(initial_input).

    Graph topology:
        agent1_renamer → executor1 → reclassify_columns → agent2_cleaner → executor2 → END

    Node descriptions:
    - agent1_renamer    : make_field_renamer_agent(config)  — LLM call, produces column_map
    - executor1         : rename_executor_agent             — applies composite rename map
    - reclassify_columns: reclassify_columns_node           — re-classifies columns post-rename
    - agent2_cleaner    : make_field_cleaner_agent(config)  — LLM call, produces cleaning_code
    - executor2         : cleaning_executor_agent           — runs cleaning code
    """
    # Body:
    # workflow = StateGraph(AgentState)
    #
    # workflow.add_node("agent1_renamer",     make_field_renamer_agent(config))
    # workflow.add_node("executor1",          rename_executor_agent)
    # workflow.add_node("reclassify_columns", reclassify_columns_node)
    # workflow.add_node("agent2_cleaner",     make_field_cleaner_agent(config))
    # workflow.add_node("executor2",          cleaning_executor_agent)
    #
    # workflow.set_entry_point("agent1_renamer")
    # workflow.add_edge("agent1_renamer",     "executor1")
    # workflow.add_edge("executor1",          "reclassify_columns")
    # workflow.add_edge("reclassify_columns", "agent2_cleaner")
    # workflow.add_edge("agent2_cleaner",     "executor2")
    # workflow.add_edge("executor2",          END)
    #
    # return workflow.compile()
```

**CRITICAL NOTE [H6]:** This file, `main.py`, `agent_1_field_renamer.py`, `agent_2_field_cleaner.py`, and `executor.py` must all be committed in a single atomic commit. There is no safe intermediate state where `graph.py` is updated but `main.py` is not (the old `from source_code.graph import ds_machine` will break immediately).

---

### `main.py` — Modify [H6]

**Summary:** Remove manual file reads, remove manual classify_columns call, remove `from source_code.graph import ds_machine`. Use `load_pipeline_inputs`, `build_graph`, `PipelineConfig.from_env()`.

**OLD imports (lines 1-6):**
```python
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import io
from source_code.graph import ds_machine
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary
```

**NEW imports:**
```python
from dotenv import load_dotenv
load_dotenv()
from source_code.config.llm_config import PipelineConfig
from source_code.config.loaders import load_pipeline_inputs
from source_code.graph import build_graph
```

**OLD function `run_pipeline()` body (lines 9-88):** reads files manually, builds metadata manually, calls classify_columns manually.

**NEW function `run_pipeline()` body:**
```python
def run_pipeline():
    """
    Entry point for the DS Machine pipeline.
    Loads inputs via load_pipeline_inputs, builds graph from env config,
    streams execution, and prints output to console.

    Body logic:
    1. initial_input = load_pipeline_inputs(
           data_path     = r"data/telecom_churn_data.csv",
           query_path    = r"queries/churn_query.sql",
           rules_path    = r"rules/special_rules.txt",
           target_column = "ChurnFlag",    [AUDIT M10]
       )
    2. config = PipelineConfig.from_env()
    3. ds_machine = build_graph(config)
    4. for output in ds_machine.stream(initial_input):
           for node_name, state_updates in output.items():
               # Same streaming output logic as original (lines 69-87)
               # Print cleaning_code, ambiguous_fields, flagged_columns
    """
```

**Streaming output loop** (lines 68-87 of original): kept exactly as-is, only `ds_machine` variable origin changes (now from `build_graph(config)` instead of module import).

---

## 4. Fixture File Content

### `tests/fixtures/agent_1/scenarios.json`

```json
{
  "happy_path": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\",\n    \"rev_30d\": \"RevenueLast30d\",\n    \"mou_out_30d\": \"MouOutgoingLast30d\",\n    \"data_vol_30d\": \"DataUsageLast30dKb\",\n    \"churn_flag\": \"ChurnFlag\",\n    \"tenure_months\": \"TenureMonths\",\n    \"region_cd\": \"RegionCode\",\n    \"plan_type\": \"PlanType\",\n    \"contract_end_dt\": \"ContractEndDate\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Valid two-block response. All 10 sample_telecom columns covered. rename_map keys use cleaned names (alias-stripped, lowercased).",
    "expected_assertions": {
      "column_map_len": 10,
      "ambiguous_fields_empty": true,
      "cleaning_code_non_empty": true,
      "cleaning_code_contains": "df.rename",
      "prompt_received": true
    }
  },
  "sql_context_alias_resolution": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\",\n    \"rev_30d\": \"RevenueLast30d\",\n    \"mou_out_30d\": \"MouOutgoingLast30d\",\n    \"data_vol_30d\": \"DataUsageLast30dKb\",\n    \"churn_flag\": \"ChurnFlag\",\n    \"tenure_months\": \"TenureMonths\",\n    \"region_cd\": \"RegionCode\",\n    \"plan_type\": \"PlanType\",\n    \"contract_end_dt\": \"ContractEndDate\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "LLM correctly resolves rev_30d to RevenueLast30d given SQL context with alias a.rev_30d.",
    "expected_assertions": {
      "column_map_key_rev_30d": "RevenueLast30d",
      "prompt_contains_sql": true,
      "ambiguous_fields_empty": true
    }
  },
  "missing_second_block": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\"\n}\ndf = df.rename(columns=rename_map)\n```",
    "description": "Only one code block returned — no ambiguous_fields block.",
    "expected_assertions": {
      "cleaning_code_non_empty": true,
      "ambiguous_fields_empty": true,
      "ambiguous_fields_not_none": true
    }
  },
  "no_code_blocks": {
    "llm_response": "I cannot determine the column mappings without additional context about the schema.",
    "description": "LLM returns plain prose, no fenced code blocks.",
    "expected_assertions": {
      "cleaning_code_empty": true,
      "ambiguous_fields_empty": true,
      "column_map_empty": true
    }
  },
  "duplicate_values_in_rename_map": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"mobile_no\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Two keys map to same clean name. Known gap: agent does not detect duplicate values.",
    "expected_assertions": {
      "column_map_non_empty": true,
      "no_exception_raised": true,
      "column_map_values_unique": true
    }
  },
  "missing_columns_from_rename_map": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"gender\": \"Gender\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Only 2 of 10 columns covered in rename_map. Known gap: partial coverage silently accepted.",
    "expected_assertions": {
      "column_map_len": 2,
      "column_map_len_equals_df_columns": false
    }
  },
  "ambiguous_fields_missing_keys": {
    "llm_response": "```python\nrename_map = {\n    \"gender\": \"Gender\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = [\n    {\"original_column\": \"gender\"}\n]\n```",
    "description": "ambiguous_fields entry has only 'original_column', missing 'candidates' and 'reason'.",
    "expected_assertions": {
      "ambiguous_fields_non_empty": true,
      "ambiguous_fields_0_candidates_is_none": true,
      "no_exception_raised": true
    }
  },
  "keys_dont_match_df_columns": {
    "llm_response": "```python\nrename_map = {\n    \"subscriber_id\": \"Msisdn\",\n    \"phone_number\": \"AlternatePhone\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "rename_map keys reference columns not present in the test DataFrame (a.msisdn, etc.).",
    "expected_assertions": {
      "column_map_non_empty": true,
      "no_exception_raised": true,
      "all_keys_in_df_columns": false
    }
  },
  "malformed_python": {
    "llm_response": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
    "description": "Syntax error in rename code — missing closing brace on the dict literal.",
    "expected_assertions": {
      "column_map_empty": true,
      "cleaning_code_non_empty": true,
      "no_exception_raised": true
    }
  }
}
```

---

### `tests/fixtures/agent_2/scenarios.json`

```json
{
  "happy_path": {
    "llm_response": "```python\nimport numpy as np\n\n# ── CATEGORICAL CLEANING ──────────────────────────────────────\nif 'Gender' in df.columns:\n    df['Gender'] = df['Gender'].str.strip().str.title()\n    df['Gender'] = df['Gender'].replace({\n        'M'   : 'Male',\n        'Male' : 'Male',\n        'F'   : 'Female',\n        'Female': 'Female',\n        'N/A' : None,\n        'NA'  : None,\n    })\n\nif 'RegionCode' in df.columns:\n    df['RegionCode'] = df['RegionCode'].str.strip().str.title()\n\nif 'PlanType' in df.columns:\n    df['PlanType'] = df['PlanType'].str.strip()\n    df['PlanType'] = df['PlanType'].replace({\n        'PRE-PAID' : 'Prepaid',\n        'prepaid'  : 'Prepaid',\n        'POST-PAID': 'Postpaid',\n        'postpaid' : 'Postpaid',\n    })\n\n# ── NUMERIC-LIKE CLEANING ─────────────────────────────────────\nif 'RevenueLast30d' in df.columns:\n    df['RevenueLast30d'] = df['RevenueLast30d'].astype(str).str.strip()\n    df['RevenueLast30d'] = df['RevenueLast30d'].str.replace(',', '', regex=False)\n    df['RevenueLast30d'] = df['RevenueLast30d'].where(\n        df['RevenueLast30d'].str.match(r'^-?\\d+(\\.\\d+)?$', na=False), None\n    )\n```\n\n```python\nflagged_columns = []\n```",
    "description": "Valid two-block response. Categorical and numeric-like columns all cleaned.",
    "expected_assertions": {
      "cleaning_code_non_empty": true,
      "flagged_columns_empty": true,
      "cleaning_code_contains": "df['Gender']",
      "output_path_set": true
    }
  },
  "metadata_context_injection": {
    "llm_response": "```python\nimport numpy as np\n\nif 'Gender' in df.columns:\n    df['Gender'] = df['Gender'].str.strip().str.title()\n    df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female', 'N/A': None})\n```\n\n```python\nflagged_columns = []\n```",
    "description": "Tests that value_counts_summary and null_summary are injected into prompt. SQL query must NOT be injected.",
    "expected_assertions": {
      "prompt_contains_value_counts": true,
      "prompt_contains_null_summary": true,
      "prompt_does_not_contain_select": true,
      "cleaning_code_non_empty": true
    }
  },
  "missing_second_block": {
    "llm_response": "```python\nimport numpy as np\n\nif 'Gender' in df.columns:\n    df['Gender'] = df['Gender'].str.strip().str.title()\n```",
    "description": "Only one code block returned — no flagged_columns block.",
    "expected_assertions": {
      "cleaning_code_non_empty": true,
      "flagged_columns_empty": true,
      "flagged_columns_not_none": true
    }
  },
  "no_code_blocks": {
    "llm_response": "The data appears to be already clean. No transformations required.",
    "description": "LLM returns plain prose, no fenced code blocks.",
    "expected_assertions": {
      "cleaning_code_empty": true,
      "flagged_columns_empty": true
    }
  },
  "flagged_columns_missing_keys": {
    "llm_response": "```python\nimport numpy as np\n\nif 'Gender' in df.columns:\n    df['Gender'] = df['Gender'].str.strip()\n```\n\n```python\nflagged_columns = [\n    {\"column\": \"RevenueLast30d\"}\n]\n```",
    "description": "flagged_columns entry has only 'column', missing 'reason'.",
    "expected_assertions": {
      "flagged_columns_non_empty": true,
      "flagged_columns_0_reason_is_none": true,
      "no_exception_raised": true
    }
  },
  "cleaning_code_introduces_new_nulls": {
    "llm_response": "```python\nimport numpy as np\n\nif 'Gender' in df.columns:\n    df['Gender'] = df['Gender'].replace({'Male': None, 'Female': None})\n```\n\n```python\nflagged_columns = []\n```",
    "description": "Cleaning code replaces all valid Gender values with None — known gap in null detection. Executor should warn.",
    "expected_assertions": {
      "executor_error_log_non_none": true,
      "executor_error_log_contains_warning": true
    }
  },
  "cleaning_code_drops_columns": {
    "llm_response": "```python\nimport numpy as np\n\ndf = df.drop(columns=['RegionCode'])\n```\n\n```python\nflagged_columns = []\n```",
    "description": "Cleaning code drops a column. Executor's column-drop detection should fire.",
    "expected_assertions": {
      "executor_error_log_non_none": true,
      "executor_error_log_contains_warning": true
    }
  },
  "malformed_python": {
    "llm_response": "```python\nimport numpy as np\n\nif 'Gender' in df.columns\n    df['Gender'] = df['Gender'].str.strip()\n```\n\n```python\nflagged_columns = []\n```",
    "description": "Syntax error in cleaning block — missing colon after if condition.",
    "expected_assertions": {
      "cleaning_code_non_empty": true,
      "no_exception_from_agent": true,
      "executor_error_log_contains_syntax_error": true
    }
  }
}
```

---

## 5. Sample Test CSV — `tests/fixtures/sample_telecom.csv`

**Exact columns and data specification:**

The CSV must have exactly these 10 columns matching the happy_path fixture's rename_map keys (after alias stripping by `preprocess_column_names`):

| Raw column name  | Post-alias-strip | Alias prefix | Data type in CSV | Notes |
|------------------|------------------|--------------|------------------|-------|
| `a.msisdn`       | `msisdn`         | `a.`         | string           | Tests alias-prefix stripping |
| `gender`         | `gender`         | none         | string/object    | Categorical; includes M/Male/F/Female synonyms |
| `rev_30d`        | `rev_30d`        | none         | object           | Numeric-like dirty; includes "N/A" and "—" |
| `mou_out_30d`    | `mou_out_30d`    | none         | float            | Numeric |
| `data_vol_30d`   | `data_vol_30d`   | none         | float            | Numeric |
| `churn_flag`     | `churn_flag`     | none         | int (0/1)        | Target column |
| `tenure_months`  | `tenure_months`  | none         | int              | Numeric |
| `region_cd`      | `region_cd`      | none         | string/object    | Categorical |
| `plan_type`      | `plan_type`      | none         | string/object    | Categorical; PRE-PAID/POST-PAID variants |
| `contract_end_dt`| `contract_end_dt`| none         | int (YYYYMMDD)   | Date-as-integer |

**50 rows.** Sample data values (first 5 rows shown, rest must follow same patterns):

```
a.msisdn,gender,rev_30d,mou_out_30d,data_vol_30d,churn_flag,tenure_months,region_cd,plan_type,contract_end_dt
27831000001,Male,150.5,320.0,512000.0,0,24,North,Prepaid,20261231
27831000002,M,N/A,280.0,256000.0,1,12,South,POST-PAID,20250630
27831000003,Female,200.0,410.0,1024000.0,0,36,East,Prepaid,20271231
27831000004,F,—,150.0,128000.0,0,6,West,PRE-PAID,20260630
27831000005,male,175.5,390.0,768000.0,1,18,North,Postpaid,20251231
27831000006,MALE,N/A,260.0,384000.0,0,30,South,postpaid,20261231
27831000007,Female,195.0,340.0,512000.0,0,48,East,prepaid,20271231
27831000008,female,180.0,290.0,256000.0,1,9,West,PRE-PAID,20250630
27831000009,M,210.0,450.0,1280000.0,0,60,North,Prepaid,20281231
27831000010,F,N/A,200.0,640000.0,1,15,South,POST-PAID,20261231
```

Continue for 50 rows total. Ensure:
- `gender` column contains all of: "Male", "M", "male", "MALE", "Female", "F", "female", "FEMALE" across the 50 rows.
- `rev_30d` column contains numeric values (e.g., "150.5", "200.0") AND dirty non-numeric values: "N/A", "—", "null", "". At least 5 of 50 rows must have dirty values so the column is classified as `numeric_like` by `classify_columns`.
- `plan_type` column contains: "Prepaid", "PRE-PAID", "Pre-Paid", "pre-paid", "Postpaid", "POST-PAID", "Post-Paid".
- `region_cd` column contains 4 values: "North", "South", "East", "West" in varying capitalizations.
- `contract_end_dt` values are all 8-digit integers in range 20250101–20281231.
- All 50 rows of `a.msisdn` follow format `2783100XXXX` (unique).
- No completely empty rows. No duplicate msisdn values.

---

## 6. Execution Order

Implement files in this exact order to avoid import errors during development:

```
Step 1  — source_code/__init__.py               (empty, no deps)
Step 2  — source_code/agents/__init__.py         (empty, no deps)
Step 3  — source_code/config/__init__.py         (empty initially, update after steps 4-6)
Step 4  — source_code/config/llm_config.py       (deps: stdlib only)
Step 5  — source_code/config/llm_factory.py      (deps: langchain_core, llm_config)
Step 6  — source_code/config/loaders.py          (deps: utils.py which exists)
Step 7  — source_code/config/__init__.py         (update to re-export from steps 4-5)
Step 8  — source_code/reclassify.py              (deps: state.py, utils.py — both exist)
Step 9  — source_code/agents/executor.py         (REWRITE: deps: utils.py, state.py)
Step 10 — source_code/agents/agent_1_field_renamer.py  (MODIFY: deps: config from steps 4-5)
Step 11 — source_code/agents/agent_2_field_cleaner.py  (MODIFY: deps: config from steps 4-5)
Step 12 — source_code/graph.py                   (REWRITE: deps: steps 8-11)
Step 13 — main.py                                (MODIFY: deps: step 6, step 12)
          [COMMIT POINT — steps 9-13 must be one atomic commit] [AUDIT H6]
Step 14 — requirements-dev.txt                   (no code deps)
Step 15 — tests/__init__.py                      (empty)
Step 16 — tests/agents/__init__.py               (empty)
Step 17 — tests/mock_llm.py                      (deps: langchain_core)
Step 18 — tests/fixtures/sample_telecom.csv      (data file, no code deps)
Step 19 — tests/fixtures/agent_1/scenarios.json  (data file)
Step 20 — tests/fixtures/agent_2/scenarios.json  (data file)
Step 21 — tests/agents/test_agent_1.py           (deps: mock_llm, config, agent_1)
Step 22 — tests/agents/test_agent_2.py           (deps: mock_llm, config, agent_2)
Step 23 — tests/agents/test_executor.py          (deps: executor, fixtures)
Step 24 — tests/run_tests.py                     (deps: pytest, all test files)
Step 25 — tests/reports/.gitkeep                 (no deps)
```

---

## 7. Acceptance Checklist

All items are binary pass/fail. The Tester and Committer verify each before closing the feature.

### LLMConfig / PipelineConfig [H4]
- [ ] `LLMConfig` has field named `llm_instance` (no underscore prefix)
- [ ] `LLMConfig.llm_instance` has `repr=True`
- [ ] `PipelineConfig.with_mock(mock)` exists and returns a config with `default_llm.llm_instance == mock`
- [ ] `PipelineConfig.from_env()` constructs without any environment variables set (all defaults apply)
- [ ] `PipelineConfig.get_llm_config("agent1")` returns `default_llm` when no override registered
- [ ] `PipelineConfig.get_llm_config("agent1")` returns the override when one is registered for "agent1"

### LLMFactory [M11]
- [ ] `LLMFactory.create(provider="ollama", ...)` imports from `langchain_ollama`, not `langchain_community`
- [ ] `LLMFactory.create(provider="groq", ...)` deferred import works (no ImportError for uninstalled providers)
- [ ] `LLMFactory.create(..., llm_instance=mock)` returns `mock` directly without calling any provider import
- [ ] `LLMFactory.create(provider="unknown", ...)` raises `ValueError`

### MockLLM [H3]
- [ ] `MockLLM` inherits `BaseChatModel`
- [ ] `MockLLM` implements `_generate` (returns `ChatResult` wrapping `AIMessage`)
- [ ] `MockLLM._llm_type` returns `"mock"`
- [ ] `MockLLM.invoke(prompt)` records `prompt` in `mock.last_prompt`
- [ ] `MockLLM.invoke(prompt).content` returns the constructor-supplied string
- [ ] `MockLLM.last_prompt` is `None` before first `.invoke()` call

### Agent 1 Refactor
- [ ] `make_field_renamer_agent` exists in `agent_1_field_renamer.py`
- [ ] `field_renamer_agent` is no longer a module-level function (only returned by factory)
- [ ] Old `ChatGroq` import removed from `agent_1_field_renamer.py`
- [ ] `exec(ambiguous_code, {"__builtins__": __builtins__}, local_ns)` used [AUDIT L12]
- [ ] `exec(map_code, {"__builtins__": __builtins__}, local_ns)` used [AUDIT L12]
- [ ] `make_field_renamer_agent(PipelineConfig.with_mock(mock))` returns a callable without network calls

### Agent 2 Refactor
- [ ] `make_field_cleaner_agent` exists in `agent_2_field_cleaner.py`
- [ ] `field_cleaner_agent` is no longer a module-level function
- [ ] Old `ChatGroq` import removed from `agent_2_field_cleaner.py`
- [ ] `exec(flagged_code, {"__builtins__": __builtins__}, local_ns)` used [AUDIT L12]
- [ ] `make_field_cleaner_agent(PipelineConfig.with_mock(mock))` returns a callable without network calls

### Executor Refactor [C1, C2, L12, L13]
- [ ] `code_executor_agent` no longer exists in `executor.py`
- [ ] `rename_executor_agent` exists in `executor.py`
- [ ] `cleaning_executor_agent` exists in `executor.py`
- [ ] `rename_executor_agent` does NOT call `exec(cleaning_code)` [AUDIT C2]
- [ ] `rename_executor_agent` reads from `state["file_path"]` (original CSV) [AUDIT L13]
- [ ] `rename_executor_agent` returns `{"output_path": "standardized_output_renamed.csv", ...}` [AUDIT C1]
- [ ] `cleaning_executor_agent` reads from `state["output_path"]` (NOT `state["file_path"]`) [AUDIT L13]
- [ ] `exec()` in `cleaning_executor_agent` uses `{"__builtins__": __builtins__, "pd": pd, "np": np}` [AUDIT L12]
- [ ] `_check_safety` appends WARNING strings for dropped columns
- [ ] `_check_safety` appends WARNING strings for new null values

### Graph Refactor [H6]
- [ ] `build_graph(config: PipelineConfig)` exists in `graph.py`
- [ ] Module-level `ds_machine` no longer exists in `graph.py`
- [ ] Graph includes `reclassify_columns` node between `executor1` and `agent2_cleaner`
- [ ] `executor1` node uses `rename_executor_agent`
- [ ] `executor2` node uses `cleaning_executor_agent`
- [ ] `reclassify_columns_node` imported from `source_code.reclassify` (not `source_code.agents.reclassify`) [AUDIT M8]

### reclassify_columns_node [C1, M8]
- [ ] `reclassify_columns_node` lives in `source_code/reclassify.py` (not in `agents/`) [AUDIT M8]
- [ ] Fallback path is `"standardized_output_renamed.csv"` [AUDIT C1]
- [ ] Reads from `state.get("output_path", "standardized_output_renamed.csv")`
- [ ] Returns all 5 metadata keys: `categorical_cols`, `numeric_like_cols`, `true_numeric_cols`, `value_counts_summary`, `null_summary`

### Config Loaders [M10, L15]
- [ ] `load_pipeline_inputs` takes 4 positional params: `data_path, query_path, rules_path, target_column` [AUDIT M10/L15]
- [ ] `_build_initial_input` has `target_column` as explicit required parameter (no empty string default)
- [ ] `load_client_config` has `target_column` as an explicit parameter
- [ ] `load_pipeline_inputs` returns dict with `"target_column": target_column` (non-empty)

### main.py
- [ ] `from source_code.graph import ds_machine` removed
- [ ] `from source_code.graph import build_graph` present
- [ ] `from source_code.config.loaders import load_pipeline_inputs` present
- [ ] `ds_machine = build_graph(PipelineConfig.from_env())` called inside `run_pipeline()`
- [ ] `run_pipeline()` calls `load_pipeline_inputs(..., target_column="ChurnFlag")`
- [ ] `python main.py` runs end-to-end without import errors (with valid env vars set)

### Test Files
- [ ] `tests/agents/test_agent_1.py` has exactly 9 test functions (TC1–TC9)
- [ ] `tests/agents/test_agent_2.py` has exactly 8 test functions (TC10–TC17)
- [ ] `tests/agents/test_executor.py` has exactly 6 test functions (TC18–TC23)
- [ ] All executor tests use `tmp_path` fixture parameter [AUDIT H5]
- [ ] TC5, TC6, TC8 decorated with `@pytest.mark.xfail(strict=False, reason=...)` [AUDIT L14]
- [ ] TC15, TC16, TC21, TC22 decorated with `@pytest.mark.xfail(strict=False, reason=...)` [AUDIT L14]
- [ ] `pytest.ini` or `pyproject.toml` registers `agent1`, `agent2`, `executor` as custom markers (to suppress PytestUnknownMarkWarning)
- [ ] All `expected_assertions` keys in fixture JSON are consumed programmatically by test code

### Fixture Files [M7]
- [ ] `tests/fixtures/agent_1/scenarios.json` uses `expected_assertions` (dict), not `expected_behavior` (string) [AUDIT M7]
- [ ] `tests/fixtures/agent_2/scenarios.json` uses `expected_assertions` (dict) [AUDIT M7]
- [ ] `tests/fixtures/sample_telecom.csv` has column `a.msisdn` (with alias prefix)
- [ ] `tests/fixtures/sample_telecom.csv` has `rev_30d` with at least 5 dirty values ("N/A", "—", etc.)
- [ ] `tests/fixtures/sample_telecom.csv` has exactly 50 rows
- [ ] `tests/fixtures/sample_telecom.csv` has exactly 10 columns matching happy_path rename_map keys

### requirements-dev.txt [M9]
- [ ] `requirements-dev.txt` exists at project root
- [ ] Contains `pytest>=7.0`
- [ ] Contains `pytest-json-report>=1.5`
- [ ] `tests/run_tests.py` checks for `pytest_jsonreport` at startup and prints install instruction if missing [AUDIT M9]

### test runner and report
- [ ] `python tests/run_tests.py` runs without error when all tests pass
- [ ] `python tests/run_tests.py --agent 1` runs only @pytest.mark.agent1 tests
- [ ] `python tests/run_tests.py --agent 2` runs only @pytest.mark.agent2 tests
- [ ] A `.md` file is written to `tests/reports/` after each run
- [ ] Report maps `XFAIL` status → "WARNING" in the generated markdown [AUDIT L14]
- [ ] `tests/reports/.gitkeep` exists (directory tracked by git)

### __init__.py files [L16]
- [ ] `source_code/__init__.py` exists (even if empty)
- [ ] `source_code/agents/__init__.py` exists (even if empty)
- [ ] `source_code/config/__init__.py` exists and re-exports `PipelineConfig`, `LLMConfig`, `LLMFactory`

### Atomic commit [H6]
- [ ] `graph.py`, `main.py`, `agent_1_field_renamer.py`, `agent_2_field_cleaner.py`, `executor.py` appear in the same git commit
- [ ] After that commit, `python main.py` runs without `ImportError` or `AttributeError`

### Existing pipeline (regression)
- [ ] `python main.py` completes successfully end-to-end with GROQ_API_KEY set
- [ ] Output file `standardized_output_renamed.csv` is written after executor1
- [ ] Output file `output_agent2_cleaned.csv` is written after executor2
- [ ] `reclassify_columns_node` logs are visible in console between executor1 and agent2_cleaner
- [ ] No regression in `source_code/utils.py` (no changes made, all existing tests still pass)

---

## 8. Key Decisions Summary (Digest for Code Writer)

1. **C2 resolution**: `rename_executor_agent` applies ONLY the composite map. No `exec(cleaning_code)`. The composite map is: `{orig: column_map[cleaned] for orig, cleaned in preprocess_column_names(df.columns).items() if cleaned in column_map}`.

2. **C1 resolution**: `rename_executor_agent` returns `{"output_path": "standardized_output_renamed.csv"}`. `reclassify_columns_node` default fallback is `"standardized_output_renamed.csv"` (not `"standardized_output.csv"`).

3. **L13 bootstrap**: `rename_executor_agent` reads from `state["file_path"]` (original). `cleaning_executor_agent` reads from `state["output_path"]` (post-rename output).

4. **H3 MockLLM**: inherits `BaseChatModel`, stubs `_generate` and `_llm_type`. Override `invoke()` to record `last_prompt` and return `MockResponse`.

5. **H4 field naming**: `llm_instance` (no underscore). `PipelineConfig.with_mock(mock)` convenience constructor.

6. **M8 file location**: `reclassify.py` lives at `source_code/reclassify.py`, NOT in `source_code/agents/`.

7. **M11 Ollama import**: `from langchain_ollama import ChatOllama` (not `langchain_community`).

8. **L14 WARNING tests**: Use `@pytest.mark.xfail(strict=False)`. Report generator maps XFAIL → WARNING.

9. **H6 atomicity**: Steps 9–13 (executor, agent_1, agent_2, graph, main) must be one git commit.

10. **M7 fixtures**: Use `expected_assertions` dict (machine-readable), not `expected_behavior` string.
