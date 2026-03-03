# Audit — Agent Testing Harness + LLM Abstraction Layer

**Feature:** `agent-testing-harness`
**Author stage:** Auditor (Stage 4)
**Input:** `00_feature_brief.md`, `01_brainstorm.md`, `02_architecture.md`, source files: state.py, graph.py, agent_1_field_renamer.py, agent_2_field_cleaner.py, executor.py, utils.py, main.py
**Date:** 2026-03-02

---

## 0. Audit Methodology

This audit reviews the architecture design document (`02_architecture.md`) against the actual source code for conflicts, implementation risks, design gaps, and internal inconsistencies. Every finding is rated by severity:

- **Critical** — Will cause silent data corruption, pipeline breakage, or test suite invalidity. Must be resolved before implementation begins.
- **High** — Will cause runtime errors or test failures that block the feature from working. Should be resolved before implementation.
- **Medium** — Creates fragility, ambiguity, or likely implementation confusion. Should be resolved or documented.
- **Low** — Minor inconsistencies, style issues, or future risks. Document and address when convenient.

---

## 1. CRITICAL — reclassify_columns_node Reads from Wrong State Key

**Finding:**

The architecture's Bug 2 fix introduces `reclassify_columns_node` (Section 7 of the architecture). The node reads the post-rename CSV using:

```python
output_path = state.get("output_path", "standardized_output.csv")
```

However, the existing Executor (`executor.py`) does NOT write to `state["output_path"]`. It returns:

```python
return {"error_log": None, "cleaned_file_path": output_path}
```

The key returned is `"cleaned_file_path"`, not `"output_path"`. Furthermore, `AgentState` in `state.py` defines the field as `output_path : Optional[str]`, and the existing executor never populates it. The current architecture spec (Section 13, item 3) says the executor will be updated to return `{"output_path": output_path}`, but this is described as a to-do change — the fix is not atomic. If the implementer modifies the executor but forgets to remove the `"cleaned_file_path"` return key, or introduces a rename executor that returns one key but not the other, `reclassify_columns_node` will silently fall back to the hardcoded `"standardized_output.csv"`.

More critically: the rename executor and cleaning executor have different default output paths per Section 13 (rename executor defaults to `"standardized_output_renamed.csv"`, cleaning executor defaults to `"output_agent2_cleaned.csv"`). But `reclassify_columns_node` hardcodes its fallback as `"standardized_output.csv"` — which matches neither default. If `output_path` is not in state at reclassify time, it will attempt to read from a file that does not exist.

**Severity:** Critical

**Resolution:** The architecture must be explicit:
1. `rename_executor_agent` MUST return `{"output_path": "standardized_output_renamed.csv", ...}` (not `"cleaned_file_path"`).
2. `reclassify_columns_node` default fallback must match exactly: `state.get("output_path", "standardized_output_renamed.csv")`.
3. The architecture should add a contract: if `rename_executor_agent` raises before writing, `output_path` is absent, and `reclassify_columns_node` must detect this and either fail fast or propagate the error rather than attempting to read a non-existent file.

---

## 2. CRITICAL — Bug 1 Fix Applies Double Transformation to Columns Already Preprocessed

**Finding:**

The Bug 1 fix in Section 1 of the architecture describes the executor building a composite rename map:

```python
pre_cleaned = preprocess_column_names(list(df.columns))  # {original: cleaned}
composite_map = {
    orig: column_map[cleaned]
    for orig, cleaned in pre_cleaned.items()
    if cleaned in column_map
}
df = df.rename(columns=composite_map)
```

The design intent is: `original_col → cleaned_col (via preprocess) → standardised_col (via column_map)`. This is correct when the DataFrame has original column names with alias prefixes (e.g., `a.msisdn`).

**However**, the Executor ALSO runs `exec(cleaning_code, exec_globals, local_vars)` where `cleaning_code` is Agent 1's generated Python that already contains `df = df.rename(columns=rename_map)`. This means:

- Step 1: `exec()` runs Agent 1's rename code: `df.rename(columns={"msisdn": "Msisdn", ...})`. If columns are `["a.msisdn", "gender", ...]` this silently renames nothing (the bug).
- Step 2: The composite map fix runs and correctly renames everything.

This is fine for the alias-prefix case. But for a DataFrame where columns DO match the cleaned names exactly (no alias prefixes — e.g., `["msisdn", "gender", "rev_30d"]`), the `exec()` step would succeed and rename the columns, AND THEN the composite map step runs again. After `exec()`, `df.columns` would already be `["Msisdn", "Gender", ...]`. When `preprocess_column_names` is called on those already-renamed columns:
- `"Msisdn"` → `"msisdn"` (lowercased)
- `"msisdn"` is in `column_map`, so composite_map gets `{"Msisdn": "Msisdn"}`
- The second rename is idempotent only if the column_map value equals the already-renamed column name.

The spec says in Section 13, item 2: "After `exec()` runs for Agent 1's code, apply the alias-composite rename." and "Alternatively (and more explicitly), use two distinct executor functions". The architecture then decides to use two distinct functions (`rename_executor_agent` and `cleaning_executor_agent`). The description is that `rename_executor_agent` runs `exec()` AND the composite map.

**The double-exec problem is real**: if `exec()` successfully renames some columns (those without alias prefixes) and the composite map then also tries to rename the same columns, the composite map would use the post-exec column names as keys. `preprocess_column_names(["Msisdn", "Gender", ...])` lowercases them to `["msisdn", "gender", ...]` which ARE in `column_map`, so composite_map = `{"Msisdn": "Msisdn", "Gender": "Gender", ...}`. This is a no-op rename — harmless, but wasteful and confusing.

**The true bug**: the architecture does not resolve whether `rename_executor_agent` should run BOTH exec(cleaning_code) AND the composite map, or ONLY the composite map (treating `cleaning_code` as unused for the rename path). If `exec(cleaning_code)` is also run by `rename_executor_agent`, the agent 1 code block must be structurally idempotent when applied on top of the composite map result. This is fragile.

**Severity:** Critical

**Resolution:** The architecture must make a definitive statement: `rename_executor_agent` should NOT call `exec(cleaning_code)` at all. Instead, it should:
1. Load `df` from `file_path`.
2. Build the composite map from `state["column_map"]` and `preprocess_column_names(df.columns)`.
3. Apply `df.rename(columns=composite_map)`.
4. Save the result and return `{"output_path": ..., "error_log": None}`.

The `cleaning_code` field in state (which contains Agent 1's generated `df.rename(...)` snippet) is then NOT executed by the rename executor — the executor reconstructs the rename from `column_map` directly. This avoids the double-exec risk entirely and makes the fix clean. The architecture currently does not make this explicit.

---

## 3. HIGH — MockLLM Duck Typing Breaks Under LangChain Type Checks

**Finding:**

The architecture states (Section 5):

> `MockLLM` does NOT inherit from `BaseChatModel`. Inheriting from `BaseChatModel` requires implementing abstract methods and registering with LangChain's provider registry, which is unnecessary overhead for a test double. Python's duck typing means that as long as `MockLLM` exposes `.invoke()` returning an object with `.content`, it is fully compatible with agent code.

The agent factory functions (Section 6.1) declare the resolved LLM as `llm: BaseChatModel`. The type annotation is not enforced at runtime in Python, so the `MockLLM` instance will be assigned to `llm` without error. However, the concern arises from `LLMFactory.create()`:

```python
@staticmethod
def create(..., _llm_instance=None) -> BaseChatModel:
    if _llm_instance is not None:
        return _llm_instance
```

This return type is annotated as `BaseChatModel` but can return a `MockLLM`. Static type checkers (mypy, pyright) will flag this as an error, which may fail CI if type-checking is added later. More seriously, LangChain callbacks, tracing, and middleware (e.g., `with_config()`, `stream()`, `astream()`) are sometimes applied to the LLM object by the calling code or by LangGraph's node tracing machinery. If any code path calls a method beyond `.invoke()` on the LLM, it will raise `AttributeError` on `MockLLM`.

The agents currently only call `llm.invoke(prompt)`. However, the closure design means that if LangGraph internally wraps or inspects the node's callable, it may introspect the LLM object. This risk is low with the current LangGraph version but is not validated.

**Severity:** High

**Resolution:** `MockLLM` should inherit from `BaseChatModel` with minimal stub implementations of the required abstract methods. This removes the type annotation mismatch, satisfies static analysis, and insulates the test suite from future LangChain version changes. The required abstract methods in `BaseChatModel` are `_generate` and `_llm_type`. Adding these stubs is approximately 10 lines of code and eliminates the risk entirely.

Alternatively, if inheriting is deemed unacceptable, the type annotation on `LLMFactory.create()` should be changed to `Union[BaseChatModel, Any]` and a comment should document the deliberate type bypass.

---

## 4. HIGH — LLMConfig._llm_instance is a Dataclass Private Field Used as Public DI Mechanism

**Finding:**

The architecture injects MockLLM via:

```python
cfg = PipelineConfig(default_llm=LLMConfig(_llm_instance=mock))
```

The field `_llm_instance` uses a leading underscore, which in Python conventionally signals a private or internal member. Dataclass fields with leading underscores create a specific problem: `@dataclass` generates `__init__` with the field name as a keyword argument — including the underscore. The call above works (`LLMConfig(_llm_instance=mock)`) but is surprising and non-idiomatic.

Additionally, the field is typed as `Any` and has `compare=False, repr=False`. This means:
1. Two `LLMConfig` objects with identical parameters but different `_llm_instance` values will compare as equal (`==`). This could cause subtle bugs if configs are deduplicated or cached.
2. The `repr` exclusion means `print(config)` will not show the injected mock, making debugging harder.

The `from_env()` factory (Section 4) constructs `PipelineConfig` without any override mechanism for `_llm_instance`. If a user calls `PipelineConfig.from_env()` and then needs to inject a mock (e.g., for integration testing), they must either reconstruct the config or manually set the field: `cfg.default_llm._llm_instance = mock`. This is an unclean mutation of a "constructed" config object.

**Severity:** High

**Resolution:** Rename the field to `llm_instance` (no underscore) to signal it is a legitimate optional field. Add it to `repr=True` so debugging is easier. Accept that `compare=False` is correct (two configs with identical params but different injected instances are not the same for comparison purposes — the opposite behavior from what exists).

For `from_env()`, document explicitly that injection is done post-construction: `cfg.default_llm.llm_instance = mock`. Alternatively, add a `PipelineConfig.with_mock(mock)` convenience factory method to keep injection ergonomic. The architecture currently has a gap here — it shows test construction with `PipelineConfig(default_llm=LLMConfig(_llm_instance=mock))` but never addresses the `from_env()` injection path.

---

## 5. HIGH — Test Isolation: File System Collisions in Parallel Test Runs

**Finding:**

The Executor writes to file paths like `"standardized_output_renamed.csv"` and `"output_agent2_cleaned.csv"`. These are relative paths — they resolve to the current working directory at test execution time. The test runner uses pytest, which supports parallel execution via `pytest-xdist` (a common pattern). Even without `pytest-xdist`, pytest collects all tests and can run them in rapid succession.

Multiple executor tests (TC18–TC23) and agent tests that invoke the executor (step 8 in the runner flow per Section 9) will all read/write to the same shared file paths. With concurrent or rapid-sequential runs:
- Test A writes `standardized_output_renamed.csv`.
- Test B reads it before Test A's run has completed — getting stale data.
- The assertion in Test B fails non-deterministically.

The architecture does not address this at all. There is no mention of temp directories, unique file paths per test, or test isolation mechanisms.

**Severity:** High

**Resolution:** Executor tests must use `tempfile.mkdtemp()` or `pytest`'s `tmp_path` fixture to create isolated working directories. The executor functions accept `output_path` via state — tests should always supply an explicit unique temp path rather than relying on the default. The architecture should mandate that all test functions that invoke the executor supply a `tmp_path`-based `output_path` in state.

---

## 6. HIGH — Executor Split Breaks Existing graph.py Import Pattern

**Finding:**

The current `graph.py` imports:

```python
from source_code.agents.executor import code_executor_agent
```

And uses the same function for both executor nodes:

```python
workflow.add_node("executor1", code_executor_agent)
workflow.add_node("executor2", code_executor_agent)
```

After the refactor, `executor.py` will expose `rename_executor_agent` and `cleaning_executor_agent`. The old `code_executor_agent` function is removed (or renamed). The new `graph.py` uses `build_graph(config)`.

The breakage issue: `main.py` currently imports `ds_machine` directly from `graph.py`:

```python
from source_code.graph import ds_machine
```

After the refactor, `ds_machine` is no longer a module-level symbol — it requires calling `build_graph(config)`. The architecture (Section 7) acknowledges this and says "Backward-compatible module-level instance for existing `main.py` usage" will be removed once `main.py` is updated.

However, the architecture does not guarantee atomicity — it lists `main.py` as a modified file but does not lock the two changes together. If the implementer updates `graph.py` (removing `ds_machine`) but delays `main.py` update, the application breaks immediately. In a team setting or multi-PR workflow this is a merge risk.

Furthermore, the comment in the architecture says the backward-compat `ds_machine` "is removed once main.py is updated". But the architecture does not specify what the backward-compat version looks like — it cannot be a `build_graph(config)` call since `config` requires a `PipelineConfig` which requires the new config module. The backward-compat stub would need `build_graph(PipelineConfig.from_env())`, which means the new config module must already exist when the backward-compat stub is compiled.

**Severity:** High

**Resolution:** The architecture must specify that `graph.py`, `main.py`, the two agent files, and `executor.py` are refactored in a single atomic commit. There is no safe intermediate state where `graph.py` is updated but `main.py` is not. The implementation plan must enforce this.

---

## 7. MEDIUM — Fixture JSON Schema: `expected_behavior` is a String, Not Machine-Readable

**Finding:**

The fixture schema (Section 10) defines:

```json
{
  "scenario_name": {
    "llm_response": "...",
    "description": "...",
    "expected_behavior": "column_map has 10 entries, ambiguous_fields is [], cleaning_code contains df.rename"
  }
}
```

`expected_behavior` is a free-form English string. In the example fixtures provided:
- `"expected_behavior": "column_map has 10 entries, ambiguous_fields is [], cleaning_code contains df.rename"` — this is documentation, not an assertion specification.
- `"expected_behavior": "DOCUMENTED GAP: column_map will have duplicate values; no error raised"` — these GAP cases cannot be programmatically asserted.

The test code (Section 9) shows tests asserting on `result` fields directly in Python. The `expected_behavior` string in the fixture is never consumed by the test code — it is purely decorative documentation. This means:
1. A test can pass even if the actual behavior completely contradicts what `expected_behavior` says.
2. There is no automated validation that the fixture's documented expectation matches the test's assertions.
3. Maintainers will update one and forget the other, creating stale documentation.

The architecture does not define how `expected_behavior` is used in the test code, which means the implementer will either ignore it (making it dead data) or invent their own interpretation.

**Severity:** Medium

**Resolution:** Either:

Option A (recommended): Change `expected_behavior` to a machine-readable dict with a defined schema:
```json
"expected_assertions": {
  "cleaning_code_non_empty": true,
  "column_map_len": 10,
  "ambiguous_fields_empty": true,
  "cleaning_code_contains": "df.rename"
}
```
Test code reads these assertions and applies them, guaranteeing the fixture and test logic stay in sync.

Option B: Accept that `expected_behavior` is documentation-only and rename it to `expected_behavior_doc` to make this explicit. Add a comment in the test template file stating that all assertions are in the Python test function, not the fixture.

---

## 8. MEDIUM — reclassify_columns_node File Location Inconsistency

**Finding:**

The architecture places `reclassify_columns_node` in `source_code/agents/reclassify.py` (Section 7, file tree). However, this function is not an agent — it makes no LLM call, accepts no `PipelineConfig`, and is a pure data transformation node. Placing it in the `agents/` subdirectory is conceptually incorrect and will confuse maintainers who expect `agents/` to contain LLM-calling components.

Additionally, the import in `graph.py` (Section 7) shows:
```python
from source_code.agents.reclassify import reclassify_columns_node
```

This import will work but signals that the graph assembler treats reclassify as an "agent", which is not accurate.

**Severity:** Medium

**Resolution:** Move `reclassify.py` to `source_code/graph_nodes/reclassify.py` or keep it in `source_code/` as a top-level utility. Given the current project structure only has `source_code/agents/` and no `graph_nodes/` directory, the least disruptive option is to place it at `source_code/reclassify.py` (top-level in `source_code/`). If a `graph_nodes/` pattern is desired, that directory must be created and its `__init__.py` accounted for.

---

## 9. MEDIUM — pytest-json-report is an Undeclared Dependency with No requirements.txt

**Finding:**

The architecture (Section 9) identifies `pytest-json-report` as "the only new test dependency" and specifies it must be `pip install pytest-json-report`. However:
1. The project has no `requirements.txt`, `pyproject.toml`, or `setup.py`. All dependencies are currently implicit.
2. `pytest-json-report` is a third-party pytest plugin, not part of the standard `pytest` distribution.
3. If a developer runs `python tests/run_tests.py` without installing this plugin, pytest will silently ignore the `--json-report` flag in some versions, or raise a plugin-not-found error in others. Either outcome breaks the test runner in a non-obvious way.

**Severity:** Medium

**Resolution:** The implementation must create a `requirements-dev.txt` or `pyproject.toml [test]` section that lists all test dependencies including `pytest`, `pytest-json-report`, and any other packages needed. The architecture should mandate this file as part of the deliverables. Alternatively, `run_tests.py` should add an import guard that checks for `pytest_jsonreport` and prints a clear install instruction if missing.

---

## 10. MEDIUM — _build_initial_input() Hardcodes target_column as Empty String

**Finding:**

The `_build_initial_input()` function in `loaders.py` (Section 8) returns:

```python
return {
    "file_path": data_path,
    "target_column": "",  # Caller sets this
    ...
}
```

The `AgentState` schema defines `target_column: str` as a non-optional required field. Returning an empty string for a required field and noting "Caller sets this" is a fragile contract — it relies on the caller always overriding this before passing to the pipeline, but there is no enforcement.

In the current `main.py`, `target_column` is hardcoded to `"ChurnFlag"`. After the refactor, `main.py` calls `load_pipeline_inputs(...)` and would need to manually add `target_column` to the returned dict — a step that is easy to miss.

**Severity:** Medium

**Resolution:** Add `target_column: str` as an explicit parameter to `load_pipeline_inputs()` and `load_client_config()`. This forces callers to supply it explicitly, eliminating the silent empty-string default. The `_build_initial_input()` function signature becomes:

```python
def _build_initial_input(data_path, query_path, rules_path, client_id, target_column: str) -> dict:
```

---

## 11. MEDIUM — ChatOllama Import from Deprecated Package

**Finding:**

The `LLMFactory` (Section 3) imports `ChatOllama` from:

```python
from langchain_community.chat_models import ChatOllama
```

In recent LangChain versions (0.2+), `ChatOllama` has been moved to the `langchain_ollama` package. The `langchain_community` import still works but emits a deprecation warning. In LangChain 0.3+, it may be removed entirely. The architecture hardcodes the deprecated import path.

**Severity:** Medium

**Resolution:** Change the import to:
```python
from langchain_ollama import ChatOllama
```
This requires `langchain-ollama` as an optional dependency. Add a comment noting the package must be installed separately. The deferred import pattern already in use (inside the `elif p == "ollama"` branch) means `langchain_ollama` is only imported when Ollama is actually selected.

---

## 12. LOW — exec() in Agent Files is Not Fixed by the Architecture

**Finding:**

The architecture states (Section 1, Bug 3):

> "This fix applies to BOTH `exec()` calls in the Executor and to the `exec()` calls in agent files that parse `rename_map` and `ambiguous_fields` from the LLM response."

However, examining the current agent files:

In `agent_1_field_renamer.py`:
```python
exec(ambiguous_code, {}, local_ns)   # line 64
exec(map_code, {}, local_ns)          # line 74
```

In `agent_2_field_cleaner.py`:
```python
exec(flagged_code, {}, local_ns)      # line 90
```

All three use `{}` as globals. If the LLM-generated `ambiguous_fields` or `flagged_columns` parsing code uses any built-in function (e.g., `dict()`, `list()`, `str()`), it would fail. The architecture says this will be fixed but does not provide the exact code change for agent files — only for the executor. The implementer could miss the agent-level fixes.

**Severity:** Low

**Resolution:** Section 13 should be extended to explicitly list ALL exec() call sites that must be updated, including the two in `agent_1_field_renamer.py` and one in `agent_2_field_cleaner.py`. The exact replacement at each site should be specified:

```python
exec(ambiguous_code, {"__builtins__": __builtins__}, local_ns)
```

---

## 13. LOW — No Strategy for Handling executor2 Reading from Wrong file_path

**Finding:**

After the refactor, `cleaning_executor_agent` (executor2) will load the DataFrame from `state.get("output_path", "output_agent2_cleaned.csv")`. But `output_path` in state at that point is set by `rename_executor_agent` (executor1) to `"standardized_output_renamed.csv"`.

So `cleaning_executor_agent` would load from `"standardized_output_renamed.csv"` (the output of executor1/rename), not from the original `file_path`. This is the correct behavior — executor2 should operate on the renamed DataFrame. However, the architecture does not explicitly state this, and a naive implementer might think executor2 should use `state["file_path"]` (the original CSV) because that's how the current shared executor works.

The current code shows: `file_path = state["file_path"]` — both executors use the same original file. The Bug 2 (stale classification) fix works BECAUSE reclassify_columns_node re-reads from `output_path`. But if `cleaning_executor_agent` ALSO reads from `file_path` (original), then Agent 2's cleaning code would be applied to the original (pre-rename) data, and the entire rename would be undone.

**Severity:** Low (but will manifest as a High-severity bug if the architecture is followed naively)

**Resolution:** Section 13 must explicitly state: "Both `rename_executor_agent` and `cleaning_executor_agent` must read from `state.get('output_path', <default>)` — NOT from `state['file_path']`. The `file_path` key is the original input file and must not be used after executor1 has run." An example of the `cleaning_executor_agent` load line should be included: `df = pd.read_csv(state.get("output_path", "output_agent2_cleaned.csv"))`.

Wait — this creates a bootstrap problem: on executor1's first call, `output_path` is not in state yet (it hasn't been set). Executor1 must therefore read from `state["file_path"]` and write to its output path. Only executor2 reads from `output_path`. The architecture must distinguish these two cases explicitly.

---

## 14. LOW — WARNING Status in pytest Has No Standard Mechanism

**Finding:**

The review document format (Section 11) includes a `WARNING` status for tests:

> `WARNING`: assertions pass but a secondary condition indicates a documented gap or degraded behavior. Set programmatically by the test via `pytest.warns()` or a custom marker.

`pytest.warns()` is for Python `warnings.warn()` calls, not for test result status. There is no native `WARNING` status in pytest — only `PASSED`, `FAILED`, `XFAIL`, `XPASS`, `SKIP`, and `ERROR`. The `--json-report` output will not contain a `WARNING` status.

The `run_tests.py` markdown generator would need to convert `XFAIL` or a custom marker to "WARNING" in the report. The architecture does not specify how this conversion works, leaving the implementer to invent a mechanism.

**Severity:** Low

**Resolution:** Clarify the WARNING mechanism. The recommended approach: use `@pytest.mark.xfail(reason="documented gap", strict=False)` for WARNING tests. `strict=False` means: if the test passes, record as `XPASS` (unexpected pass, may indicate the gap was fixed); if it fails, record as `XFAIL`. The report generator maps `XFAIL` → WARNING in the markdown output. This is standard pytest idiom and maps cleanly to `--json-report` output.

---

## 15. LOW — load_pipeline_inputs Signature Does Not Match Digest Description

**Finding:**

The architect's digest (provided to this auditor) states:

> `load_pipeline_inputs(data_path, query_path, rules_path)` — three arguments.

The actual architecture document (Section 8) defines:

```python
def load_pipeline_inputs(
    data_path : str,
    query_path: str,
    rules_path: str,
    client_id : str = "default",
) -> dict:
```

The function has four parameters. The digest omitted `client_id`. This is a minor inconsistency but the digest is the input to downstream agents (Code Planner, Implementer). The Code Planner may generate code using the 3-argument signature and get a wrong-but-valid default behavior.

**Severity:** Low

**Resolution:** The digest should be corrected to include all four parameters. No architecture change needed.

---

## 16. LOW — No __init__.py Listed for source_code/agents/ After Adding reclassify.py

**Finding:**

The architecture places `reclassify.py` in `source_code/agents/`. The existing `source_code/agents/` directory contains `agent_1_field_renamer.py`, `agent_2_field_cleaner.py`, and `executor.py` but no `__init__.py` (not mentioned in any file listing). If `source_code/agents/` has no `__init__.py`, it must be treated as a namespace package for imports to work. This works in Python 3.3+ but is inconsistent with the `tests/` directory which explicitly has `__init__.py` files listed.

The architecture's file tree (Section 2) lists `tests/__init__.py` and `tests/agents/__init__.py` explicitly but does not list `source_code/__init__.py` or `source_code/agents/__init__.py`. If these don't exist, the new `source_code/config/` package requires `source_code/config/__init__.py` (which IS listed) but the parent `source_code/` must also be importable.

**Severity:** Low

**Resolution:** The architecture should explicitly list `source_code/__init__.py` and `source_code/agents/__init__.py` in the file tree (Section 2) as existing files, or confirm they already exist. The implementer should not have to guess.

---

## Summary Table

| # | Issue | Severity | Resolution Required |
|---|-------|----------|---------------------|
| 1 | reclassify_columns_node reads wrong state key (`output_path` vs `cleaned_file_path`); fallback path doesn't match rename executor default | Critical | Fix state key contract and fallback defaults in architecture |
| 2 | Bug 1 fix may double-rename columns; architecture is ambiguous whether rename_executor runs exec() or only the composite map | Critical | Architecture must state: rename_executor skips exec(), applies composite map only |
| 3 | MockLLM duck typing breaks if LangChain or LangGraph accesses any method beyond `.invoke()`; type annotation mismatch | High | MockLLM should inherit BaseChatModel with minimal stubs |
| 4 | `LLMConfig._llm_instance` uses underscore-private naming for public DI mechanism; `from_env()` has no injection path | High | Rename to `llm_instance`; add `PipelineConfig.with_mock()` convenience method |
| 5 | Tests write to relative file paths; parallel/sequential tests clobber each other | High | All executor-touching tests must use `tmp_path` fixture for output paths |
| 6 | graph.py + main.py refactor must be atomic; no safe intermediate state; architecture doesn't enforce this | High | Mandate atomic commit for graph.py + main.py + agent files + executor |
| 7 | `expected_behavior` in fixture JSON is a plain string; test code cannot use it programmatically | Medium | Change to structured `expected_assertions` dict or rename to documentation-only field |
| 8 | `reclassify.py` placed in `agents/` directory despite not being an agent | Medium | Move to `source_code/` top-level or new `source_code/graph_nodes/` |
| 9 | `pytest-json-report` is undeclared dependency; no requirements file exists | Medium | Create `requirements-dev.txt`; add startup guard in `run_tests.py` |
| 10 | `_build_initial_input()` returns `target_column: ""` as hardcoded empty; fragile caller contract | Medium | Add `target_column` as explicit parameter to `load_pipeline_inputs()` |
| 11 | `ChatOllama` imported from deprecated `langchain_community`; will break in LangChain 0.3+ | Medium | Change to `langchain_ollama` package import |
| 12 | Bug 3 exec() fix not specified for agent-level exec() calls in agent_1 and agent_2 files | Low | Add explicit exec() fix specification for agent files in Section 13 |
| 13 | Architecture ambiguous about which file executor2 loads from; naive reading could cause executor2 to clean original pre-rename data | Low | Add explicit statement: executor1 reads `file_path`, executor2 reads `output_path` |
| 14 | `WARNING` test status has no native pytest mechanism; `pytest.warns()` is misidentified as the tool | Low | Clarify: use `@pytest.mark.xfail(strict=False)` mapped to WARNING in report |
| 15 | Digest omits `client_id` from `load_pipeline_inputs` signature | Low | Correct in digest for Code Planner |
| 16 | `source_code/__init__.py` and `source_code/agents/__init__.py` not listed in file tree | Low | Add to Section 2 file tree explicitly |

---

## CONCLUSION

**Approved with Mandatory Changes**

The architecture is structurally sound and the design decisions (factory closures, dedicated reclassify node, two executor functions, LLMFactory pattern) are all correct in principle. However, **two Critical and four High severity issues must be resolved before implementation begins.**

The two Critical issues are not edge cases — they are at the core of the feature's correctness:
- Issue 1 (wrong state key) will cause the reclassify node to silently read from a non-existent file on every run.
- Issue 2 (double exec ambiguity) will cause the rename executor to either double-rename columns or fail to rename them depending on whether exec() is retained.

The four High severity issues will each independently block the feature from shipping correctly (type safety, test isolation, deployment atomicity, mock injection).

**Mandatory pre-implementation changes:**
1. Resolve Issues 1 and 2 in the architecture document with explicit code specifications.
2. Resolve Issues 3, 4, 5, and 6 with updated design decisions.
3. Address Issues 7, 9, and 10 as they will cause implementation confusion.

The Medium and Low issues should be documented as implementation notes and resolved during implementation without requiring a new architecture revision.
