# Brainstorm — Agent Testing Harness + LLM Abstraction Layer

**Feature:** `agent-testing-harness`
**Author stage:** Brainstormer (Stage 2)
**Input:** `00_feature_brief.md`, `docs/system/architecture.md`, source files for Agents 1, 2, Executor, State, Utils, main.py

---

## 1. LLM Abstraction Design

### 1.1 The Core Problem with the Current Code

Both `agent_1_field_renamer.py` and `agent_2_field_cleaner.py` instantiate `ChatGroq` directly at the top of the agent function body. The LLM is not injected, not configurable, not abstracted. This means:

- Switching to OpenAI requires editing two files (will be N files as agents grow).
- There is no hook point for the test harness to swap in a `MockLLM`.
- The `GROQ_API_KEY` environment variable is implicitly required at import time (the `api_key=os.getenv(...)` is in the function body, not at module import, but the model name `llama-3.3-70b-versatile` is hardcoded).

The LLM abstraction must solve all three problems simultaneously.

### 1.2 What Interface Every LLM Wrapper Must Expose

The core interface requirement is minimal but non-negotiable: every LLM wrapper must expose a `.invoke(prompt: str)` method that returns an object with a `.content` attribute of type `str`.

This maps exactly to the LangChain `BaseChatModel` contract. The current `ChatGroq` instance already satisfies this contract. The critical insight is that we do not need to invent a custom protocol — LangChain's existing `BaseChatModel` class is the right base class. Our `LLMFactory` creates instances that are `BaseChatModel`-compatible, and agents receive them as that type.

```
interface LLMWrapper:
    def invoke(prompt: str | list[BaseMessage]) -> AIMessage:
        # AIMessage has a .content attribute (str)
```

The `MockLLM` must satisfy this same interface — it must return an object where `.content` is a string, not the string itself directly.

### 1.3 How LLMFactory Should Work

`LLMFactory` is a single function (or class with a static `create` method) that accepts:
- `provider: str` — e.g. `"groq"`, `"openai"`, `"anthropic"`, `"ollama"`
- `model: str` — e.g. `"llama-3.3-70b-versatile"`, `"gpt-4o"`, `"claude-opus-4-6"`
- `api_key: str | None` — None is valid for Ollama (local)
- `base_url: str | None` — for Ollama; defaults to `http://localhost:11434`
- `temperature: float` — defaults to 0
- `max_tokens: int` — defaults to 8000

The factory returns a `BaseChatModel` instance. It raises `ValueError` with a clear message for unknown providers.

```python
# source_code/llm/factory.py
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

def create_llm(provider: str, model: str, api_key: str | None = None,
               base_url: str | None = None, temperature: float = 0,
               max_tokens: int = 8000):
    p = provider.lower().strip()
    if p == "groq":
        return ChatGroq(model=model, temperature=temperature,
                        api_key=api_key, max_tokens=max_tokens)
    elif p == "openai":
        return ChatOpenAI(model=model, temperature=temperature,
                          api_key=api_key, max_tokens=max_tokens)
    elif p == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature,
                             api_key=api_key, max_tokens=max_tokens)
    elif p == "ollama":
        return ChatOllama(model=model, base_url=base_url or "http://localhost:11434",
                          temperature=temperature)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: groq, openai, anthropic, ollama"
        )
```

**Ollama specifics:** Ollama differs from cloud providers in two ways: (a) no API key required (local binary), (b) requires a `base_url` pointing to the local server. The factory handles both by making `api_key` optional and defaulting `base_url` to localhost. The `ChatOllama` LangChain integration handles the rest.

### 1.4 Where LLM Config Lives — The Key Decision

Three options:

**Option A: Stored in AgentState.**
Pros: passes naturally through the LangGraph graph. Cons: `AgentState` is a `TypedDict` storing data, not infrastructure objects. Putting a live LLM object in state pollutes the schema, breaks LangGraph checkpointing (LLM instances are not serializable to JSON/SQLite), and violates the acceptance criterion that says "AgentState does not grow new fields unnecessarily."

**Option B: Global config object (module-level singleton).**
Pros: simple. Cons: makes testing harder (global state between tests), not thread-safe for multi-client concurrent runs, hard to mock on a per-test basis.

**Option C: Injected into each agent at construction via a config object passed at graph compile time.**
This is the correct option. The config lives in a `PipelineConfig` dataclass (or simple dict) that is constructed once in `main.py` (or the Streamlit app) and passed to agent factory functions. Each agent function is wrapped to close over the config.

```python
# source_code/llm/config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0
    max_tokens: int = 8000

@dataclass
class PipelineConfig:
    default_llm: LLMConfig = field(default_factory=LLMConfig)
    agent_overrides: dict = field(default_factory=dict)  # e.g. {"agent1": LLMConfig(...)}

    def get_llm_config(self, agent_name: str) -> LLMConfig:
        return self.agent_overrides.get(agent_name, self.default_llm)
```

### 1.5 Per-Agent Override Mechanism

`PipelineConfig.agent_overrides` is a dict keyed by agent name string. When `get_llm_config("agent1")` is called, it returns the override if one exists, otherwise falls back to `default_llm`. Agent functions receive the `PipelineConfig` at graph construction time and call `create_llm(config.get_llm_config("agent1"))` at the start of each invocation.

This means the LLM is instantiated fresh per agent call, which is fine (instantiation is cheap; connection is lazy). An alternative is to cache the LLM instance inside the closure, but this is a premature optimisation.

### 1.6 How Agent Functions Change

Currently: agent function takes `state: AgentState` and instantiates LLM internally.
After refactor: agent function factory takes `config: PipelineConfig` and returns a function that takes `state: AgentState`.

```python
# Before
def field_renamer_agent(state: AgentState) -> dict:
    llm = ChatGroq(...)  # hardcoded
    ...

# After
def make_field_renamer_agent(config: PipelineConfig):
    def field_renamer_agent(state: AgentState) -> dict:
        llm_cfg = config.get_llm_config("agent1")
        llm = create_llm(llm_cfg.provider, llm_cfg.model, ...)
        ...
    return field_renamer_agent
```

The `graph.py` then calls `make_field_renamer_agent(pipeline_config)` when constructing the graph. This is the closure injection pattern — clean, testable, no global state.

### 1.7 Environment Variable Handling

`PipelineConfig` should be constructable from environment variables as the default path:

```python
@classmethod
def from_env(cls) -> "PipelineConfig":
    return cls(default_llm=LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "groq"),
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    ))
```

This preserves backward compatibility: if `GROQ_API_KEY` is set and no other vars are set, it works exactly as before. The Streamlit UI will later build `PipelineConfig` from form inputs.

---

## 2. Mock LLM Design

### 2.1 What Exactly Needs to Be Mocked

The agent functions call `llm.invoke(prompt)` where `prompt` is a string. The return value is used only as `response.content` — a string containing the raw LLM output (markdown with fenced code blocks). The `MockLLM` must therefore:

1. Accept `.invoke(prompt)` without crashing.
2. Return an object where `.content` is a pre-determined string.
3. NOT make any network call.
4. Optionally record what prompt it was called with (for assertion in tests).

### 2.2 MockLLM Class Design

```python
# tests/mock_llm.py
class MockResponse:
    def __init__(self, content: str):
        self.content = content

class MockLLM:
    def __init__(self, content: str):
        self._content = content
        self.last_prompt = None  # set after invoke

    def invoke(self, prompt):
        self.last_prompt = prompt
        return MockResponse(self._content)
```

`last_prompt` is important — tests can assert that the prompt contains expected substrings (e.g. "the SQL query was injected into the prompt" test).

### 2.3 Fixture Structure — One JSON Per Agent, Multiple Named Scenarios

Two options:
- **One JSON file per test case**: maximum isolation, easy to read, but proliferates files. With 9 Agent 1 tests + 8 Agent 2 tests + 6 Executor tests you get 23 fixture files immediately, and this grows to hundreds as agents are added.
- **One JSON per agent with named scenario keys**: keeps fixtures co-located, easy to see all scenarios for an agent at once, easy to add new cases.

Recommendation: **one JSON file per agent with named scenario keys**. The file is a flat dict where each key is a scenario name and each value is the raw LLM response string.

```
tests/
  fixtures/
    agent_1_fixtures.json
    agent_2_fixtures.json
```

```json
// agent_1_fixtures.json
{
  "happy_path": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    ...\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
  "missing_second_block": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```",
  "no_code_blocks": "I'm sorry, I cannot determine the correct column mappings without more context.",
  "duplicate_values_in_rename_map": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"mobile_number\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
  "keys_dont_match_df_columns": "```python\nrename_map = {\n    \"nonexistent_col\": \"SomeCleanName\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
  "malformed_python": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
  "missing_columns_from_rename_map": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```",
  "ambiguous_fields_missing_keys": "```python\nrename_map = {\n    \"gndr\": \"ambiguous_gndr\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = [\n    {\"original_column\": \"gndr\"}\n]\n```",
  "sql_context_alias_resolution": "```python\nrename_map = {\n    \"msisdn\": \"Msisdn\",\n    \"rev_30d\": \"RevenueLast30d\"\n}\ndf = df.rename(columns=rename_map)\n```\n\n```python\nambiguous_fields = []\n```"
}
```

### 2.4 Fixture Realism Requirement

The brief says fixtures must be "realistic (valid responses the real LLM would produce)." This means:
- The happy path fixture must use real column names from a known test dataset.
- The rename_map must follow the naming conventions in the prompt (PascalCase, correct abbreviation expansion, etc.).
- Malformed Python fixtures must be realistic — plausible mistakes an LLM might actually make (missing closing brace, extra indentation, etc.) not just garbage strings.

### 2.5 Test Dataset for Fixtures

Tests need a known, small, deterministic CSV so assertions on column names are stable. The solution is a tiny fixture CSV in `tests/fixtures/`:

```
tests/fixtures/sample_telecom.csv
```

With ~10 columns and ~20 rows. Columns chosen to exercise naming rules: an SQL-aliased column (`a.msisdn`), a dirty-numeric column, a categorical column with synonym values, a date column as integer (YYYYMMDD), an ambiguous column.

---

## 3. Test Case Coverage Analysis

### 3.1 Agent 1 Test Cases

**TC1 — Happy Path**
- Fixture: valid two-block response with all columns covered.
- Assertions: `column_map` is a non-empty dict; `cleaning_code` contains `df = df.rename`; `ambiguous_fields` is a list (may be empty); no keys in `column_map` appear in `ambiguous_fields` unless prefixed with `ambiguous_`; all input columns appear as keys in `column_map`.
- Risk: the "all input columns covered" assertion requires knowing what `df_columns` was passed into the state. The test must construct the state explicitly.

**TC2 — SQL Query Context**
- Fixture: response where `rev_30d` (the bare column name after alias stripping) is renamed to `RevenueLast30d`.
- The SQL passed in state contains `SELECT a.msisdn, a.rev_30d FROM customer a`.
- `preprocess_column_names` is called on `df_columns` before the prompt is built — but `df_columns` in the CSV would be bare names (no alias prefix). The alias stripping in `preprocess_column_names` applies to raw df columns, not the SQL.
- Critical subtlety: `df_columns` might come from a DataFrame loaded from the result of the SQL query, so the column names in the CSV are already bare (the alias is a SQL-time concept). The SQL query is provided for *semantic context* only — e.g. to know that `rev_30d` means `Revenue Last 30 Days`.
- Assertion: `column_map["rev_30d"]` == `"RevenueLast30d"` (not `"RevLast30d"` or `"Rev30d"`). This asserts the SQL context influenced the naming decision in the expected direction. The MockLLM fixture just pre-sets this — we can't actually verify the LLM "reasoned" about it, but we can verify the agent correctly injected the SQL into the prompt by checking `mock_llm.last_prompt` contains the SQL string.
- Secondary assertion: `mock_llm.last_prompt` contains the SQL query verbatim (proves injection worked).

**TC3 — Missing Second Code Block**
- Fixture: only one code block returned.
- Assertions: `cleaning_code` is non-empty (the rename code); `ambiguous_fields` is `[]` (not None, not missing from return dict); `column_map` is populated (may be empty if exec of map_code fails — but if fixture rename code is valid, it should succeed).
- Risk: the current code handles this correctly (`ambiguous_code = code_blocks[1] if len(code_blocks) >= 2 else ""`). The test verifies this guard works.

**TC4 — No Code Blocks**
- Fixture: plain prose response, no fenced code blocks.
- Assertions: `cleaning_code` is `""`; `ambiguous_fields` is `[]`; `column_map` is `{}`.
- Current code returns `{**state, "cleaning_code": "", "ambiguous_fields": [], "column_map": {}}` — this is correct.
- Risk: the `{**state, ...}` pattern means the returned dict includes ALL state fields. The test should only assert on the specific fields the agent is responsible for writing, not inadvertently assert on inherited state fields.

**TC5 — Duplicate Values in rename_map**
- Fixture: two keys map to the same clean name (e.g. `"msisdn": "Msisdn"` and `"mobile_no": "Msisdn"`).
- Key question from brief: does Agent 1 detect this, or does it pass through and the executor fails?
- Analysis of current code: Agent 1 does NOT detect duplicate values. It calls `exec(map_code)` to extract `rename_map` then stores it in `column_map`. Pandas `df.rename(columns=rename_map)` with duplicate values does not raise an error — it just renames both columns to the same name, resulting in two columns with identical names. This is a silent data corruption bug.
- Assertion: the test should assert `len(set(column_map.values())) < len(column_map.values())` (i.e. duplicates exist in the returned map) to verify the agent does NOT catch this — and the test should be marked as a "Warning" or "Known Gap" in the review document, flagging that duplicate detection must be added.
- The test proves the bug exists, documents it, and the fix suggestion is "add post-exec validation: assert len(set(rename_map.values())) == len(rename_map)."

**TC6 — Not All df_columns Covered in rename_map**
- Fixture: rename_map only covers 2 of 5 columns.
- Analysis: Agent 1 does not validate coverage. The returned `column_map` has only 2 entries. The executor would run the rename code successfully (pandas silently skips columns not in the map), but columns would retain their raw names.
- Assertion: `len(column_map) < len(state["df_columns"])` — test proves the gap exists.
- Fix suggestion to document: "add post-LLM validation that all df_columns appear in rename_map keys."
- Note: `preprocess_column_names` strips aliases BEFORE passing to LLM as `cleaned_cols`. So the rename_map keys should match the pre-cleaned names, not the raw `df_columns`. The test must account for this — `column_map.keys()` should match `list(preprocess_column_names(df_columns).values())`, not `df_columns` directly. This is a subtle but important distinction.

**TC7 — ambiguous_fields Missing Required Keys**
- Fixture: ambiguous_fields entry has only `original_column` but no `candidates` or `reason`.
- Assertions: `ambiguous_fields` is a non-empty list; entries are dicts; `ambiguous_fields[0].get("candidates")` is `None`; `ambiguous_fields[0].get("reason")` is `None`.
- Secondary assertion: `main.py` accesses `field.get("candidates")` and `field.get("reason")` — these would return `None` silently, not crash. Test verifies graceful degradation.
- Fix suggestion: add schema validation on `ambiguous_fields` entries after extraction.

**TC8 — rename_map Keys Don't Match Actual df Columns**
- Fixture: rename_map references column names not in the actual `df_columns`.
- Analysis: `column_map` would have keys that don't exist in the real dataframe. When executor runs `df.rename(columns=rename_map)`, pandas silently ignores non-existent keys — no error, but the intended rename never happens.
- Assertion: `set(column_map.keys()).isdisjoint(set(preprocess_column_names(state["df_columns"]).values()))` — tests that keys don't match pre-cleaned column names.
- Fix suggestion: add post-LLM validation that column_map keys exist in pre-cleaned df_columns.

**TC9 — Malformed Python in Code Block**
- Fixture: code block with SyntaxError (missing closing brace on rename_map dict).
- Analysis of current code: `exec(map_code, {}, local_ns)` is wrapped in `try/except Exception`. The except block prints a warning and leaves `column_map = {}`. So `column_map` is empty, but `cleaning_code` still contains the malformed code. When the executor runs `exec(cleaning_code, ...)`, it will raise `SyntaxError`, which IS caught by the executor's try/except and written to `error_log`.
- Assertions: Agent 1 — `column_map` is `{}`; `cleaning_code` is non-empty (the malformed string). Executor — `error_log` is non-empty; contains "SyntaxError". This is actually an integration test that spans Agent 1 + Executor.
- Risk: the malformed code test for Agent 1 alone (unit test) is limited because Agent 1 "succeeds" from its own perspective. The real failure surfaces in the Executor. Test should cover both layers.

### 3.2 Agent 2 Test Cases

**TC10 — Happy Path**
- Fixture: valid two-block response with cleaning code and valid `flagged_columns`.
- Assertions: `cleaning_code` is non-empty; `flagged_columns` is a list; each entry has `column` and `reason` keys; `output_path` is set.

**TC11 — SQL Query Context**
- Note: Agent 2 does NOT include `sql_query` in its prompt template (confirmed by reading `agent_2.json` — the template uses `metadata_summary`, `null_summary`, `value_counts_summary`, `categorical_cols`, `numeric_like_cols`, `special_rules`). The SQL query is absent from Agent 2's prompt.
- This is a design gap, not a test gap. The brief asks to "verify agent uses column context from the query" but Agent 2 doesn't receive the SQL query at all.
- Two paths: (a) add `sql_query` injection to Agent 2's prompt before writing the test, or (b) write the test to document this gap (test asserts `sql_query` NOT in mock_llm.last_prompt and marks as Warning).
- Recommendation: document the gap in the brainstorm; the Architect will decide whether Agent 2 needs the SQL context. Most likely it does not — by the time Agent 2 runs, columns are already renamed and SQL is less relevant. The test for "SQL context" in Agent 2 might simply be replaced with "metadata context" — verify the value_counts and null_summary are injected.
- Assertion: `mock_llm.last_prompt` contains the value_counts_summary string verbatim.

**TC12 — Missing Second Code Block (Agent 2)**
- Same structure as TC3 but for Agent 2. `flagged_columns` defaults to `[]`. `output_path` is still set to `"output_agent2_cleaned.csv"`. All covered by existing guards.

**TC13 — No Code Blocks (Agent 2)**
- Returns `cleaning_code: ""`, `flagged_columns: []`, `output_path: "output_agent2_cleaned.csv"`. Current code handles this.

**TC14 — flagged_columns Missing Required Keys**
- Fixture: `flagged_columns` entry missing `reason`.
- `agent_2_field_cleaner.py` accesses `f.get('column')` and `f.get('reason')` — both use `.get()` so no crash, returns `None`. Test asserts graceful degradation.

**TC15 — Cleaning Code Introduces New Nulls**
- This test requires actually EXECUTING the cleaning code against a real (small) dataframe.
- The fixture code should do something like: `df['Gender'] = df['Gender'].str.strip()` followed by a replace that aggressively converts valid values to None.
- Assertion: compare null counts in df before and after execution; `df.isna().sum().sum()` post-exec > pre-exec.
- This is important because the brief explicitly lists it as a case to detect. Currently, the executor does NOT check for new nulls. The test would pass technically (code runs without error) but the safety check does not exist yet. Mark as a gap requiring post-exec validation.

**TC16 — Cleaning Code Drops Columns**
- The brief says "catches destructive operations." Currently, executor does NOT check for dropped columns.
- Fixture cleaning code: `df = df.drop(columns=['Gender'])`.
- Assertion: compare `df.columns` before and after exec; columns after exec is a strict subset.
- Again, this is a gap — executor does not currently check this. Test documents the gap.

**TC17 — Malformed Python (Agent 2)**
- Same as TC9 logic. Agent 2 catches exec error internally, `flagged_columns` is `[]`. `cleaning_code` contains malformed string. Executor will catch the SyntaxError.

### 3.3 Executor Test Cases

**TC18 — Happy Path**
- State: valid cleaning code (simple rename or strip operation); valid file_path pointing to test CSV.
- Assertions: return dict has `error_log: None`; output file exists at returned path; file is a valid CSV (can be read with pd.read_csv without error).
- Note: `code_executor_agent` currently hardcodes `output_path = "standardized_output.csv"` — this is a second hardcoding issue (alongside main.py paths). After Agent 2 sets `output_path` in state, the executor should use `state.get("output_path", "standardized_output.csv")`. Currently it ignores the state's output_path. Test should expose this.

**TC19 — Empty cleaning_code**
- State: `cleaning_code: ""`.
- Assertions: `error_log` is non-empty string; equals "No cleaning code provided by the agent."; no file created.

**TC20 — Code Raises Runtime Exception**
- Fixture code: `df = df['nonexistent_column']` (KeyError).
- Assertions: `error_log` is non-empty; contains "KeyError"; does not re-raise (test completes without exception propagating).

**TC21 — Code Drops a Column**
- Fixture code: `df = df.drop(columns=['Gender'])`.
- Currently executor does not detect this. Assertion documents the gap.
- Post-exec validation needed: compare `set(df_original.columns) - set(df_clean.columns)` and write to `error_log` or a `warning_log`.

**TC22 — Code Introduces New Nulls**
- Fixture code: `df['Gender'] = None`.
- Assertion documents the gap — executor does not currently detect null introduction.

**TC23 — Output File Written Correctly**
- State: valid cleaning code that adds a column.
- Assertions: output file exists; `pd.read_csv(output_path)` succeeds; shape is correct.

---

## 4. SQL Query Context Testing — Detailed Design

### 4.1 The Fundamental Challenge

We cannot inspect what the LLM "thought." We can only inspect two things:
1. What the agent put into the prompt (pre-LLM).
2. What the agent extracted from the response (post-LLM, via fixture).

For the SQL context test, both sides need assertions.

### 4.2 Pre-LLM Assertion: SQL Was Injected Into the Prompt

The `MockLLM.last_prompt` attribute records the exact prompt string passed to `invoke()`. The test should assert:

```python
assert state["sql_query"] in mock_llm.last_prompt, \
    "SQL query was not injected into the prompt"
```

This verifies the agent correctly called `.replace("{sql_query}", state["sql_query"])` in its prompt building. This is a genuine and testable assertion.

### 4.3 Post-LLM Assertion: SQL Context Influenced Correct Output

This is where it gets subtle. We can't make the LLM actually use the SQL. Instead, the fixture pre-programs the "correct" response that a good LLM WOULD return when given SQL context — and the test verifies the agent correctly extracts and passes it through.

The test scenario: SQL is `SELECT a.msisdn, a.rev_30d FROM customer a`. The `df_columns` list is `["msisdn", "rev_30d"]` (already alias-stripped by the time it reaches the DataFrame). The SQL context tells us `rev_30d` = revenue for last 30 days → should map to `RevenueLast30d`.

Fixture response programs `rename_map = {"msisdn": "Msisdn", "rev_30d": "RevenueLast30d"}`.

Assertion:
```python
assert column_map.get("rev_30d") == "RevenueLast30d", \
    f"Expected 'rev_30d' → 'RevenueLast30d' (SQL context should resolve this). Got: {column_map.get('rev_30d')}"
```

The test name and description make clear: "This fixture represents the expected output when the SQL query provides sufficient context to resolve `rev_30d` to `RevenueLast30d`. The test verifies the pipeline correctly processes and exposes this output — not that the LLM inferred it."

### 4.4 The Alias Stripping Nuance

The `preprocess_column_names` utility strips `a.` prefixes. So `df_columns = ["a.msisdn", "a.rev_30d"]` becomes `cleaned_cols = ["msisdn", "rev_30d"]` before being passed to the LLM. This means:

- The rename_map keys in the fixture should be `"msisdn"` and `"rev_30d"` (post-strip), NOT `"a.msisdn"`.
- The SQL query in the test state can contain `a.msisdn` (as the user would provide it) — the stripping happens on df_columns, not on the SQL string.
- The test for "SQL alias resolution" should verify that `a.msisdn` does NOT appear as a key in `column_map` (the stripping worked correctly at the pre-processing stage).

```python
# Verify no alias-prefixed keys made it into the column map
for key in column_map.keys():
    assert not re.match(r'^[a-zA-Z]\.', key), \
        f"Alias-prefixed key '{key}' found in column_map — preprocess_column_names failed"
```

### 4.5 Recommended SQL Test Fixture

```
SQL: SELECT a.msisdn, a.rev_30d, a.mou_out_30d, a.data_vol_30d FROM customer a
df_columns: ["a.msisdn", "a.rev_30d", "a.mou_out_30d", "a.data_vol_30d"]
```

After pre-processing: `["msisdn", "rev_30d", "mou_out_30d", "data_vol_30d"]`

Expected rename_map (in fixture):
```json
{
  "msisdn": "Msisdn",
  "rev_30d": "RevenueLast30d",
  "mou_out_30d": "MouOutgoingLast30d",
  "data_vol_30d": "DataUsageLast30dKb"
}
```

Assertions:
1. No `a.` prefix keys in column_map.
2. `column_map["rev_30d"] == "RevenueLast30d"` (SQL context resolves `rev` to `Revenue` and `30d` to `Last30d`).
3. `mock_llm.last_prompt` contains `"SELECT a.msisdn, a.rev_30d"`.
4. `"a.msisdn"` is NOT in `column_map` keys (alias stripping worked).

---

## 5. Review Document Design

### 5.1 What Makes a Good Test Report

When a test fails, the developer needs:
1. Which test failed and why immediately visible (no scrolling).
2. What was expected vs what was received.
3. The root cause — is this a fixture issue, a parsing issue, or a logic issue in the agent?
4. A specific fix suggestion (not generic "check the code").

What is noise:
- Full stack traces (link to them, don't embed).
- Raw LLM response for passing tests.
- Fixture file content in the report (it's in the repo, link to it).

### 5.2 Proposed Markdown Structure

```markdown
# Test Review — Agent Testing Harness
Generated: 2024-01-15 14:32:01
Command: python tests/run_tests.py

## Summary
| | Count |
|---|---|
| PASSED | 17 |
| FAILED | 4 |
| WARNING | 2 |
| TOTAL | 23 |

Failures: TC5, TC8, TC15, TC21
Warnings: TC6, TC22

---

## Agent 1 — Field Renamer (9 tests)

| Test | Category | Result | Notes |
|---|---|---|---|
| TC1 — Happy Path | Core | PASSED | |
| TC2 — SQL Context | Context Injection | PASSED | |
| TC3 — Missing Second Block | Edge Case | PASSED | |
| ... | | | |

### TC5 — Duplicate Values in rename_map
**Result:** FAILED
**Category:** Edge Case — Data Integrity
**Expected:** `column_map` should contain no duplicate values (or agent should raise/warn)
**Received:** `column_map = {"msisdn": "Msisdn", "mobile_no": "Msisdn"}` — duplicate value "Msisdn"
**Probable Cause:** Agent 1 does not validate uniqueness of rename_map values after extraction. The LLM generated a duplicate and it passed through silently.
**Suggested Fix:** After extracting `column_map` from exec(), add: `if len(set(column_map.values())) != len(column_map): raise ValueError(f"Duplicate values in rename_map: {[v for v, c in Counter(column_map.values()).items() if c > 1]}")`

---

## Agent 2 — Field Cleaner (8 tests)
...

## Executor (6 tests)
...

## Integration Tests (TBD)
...
```

### 5.3 Pass vs Warning vs Fail Distinction

- **PASSED**: assertions all pass, no concerns.
- **WARNING**: assertions pass but a secondary check detected a potential issue (e.g. null count increased but no assertion required it). Also used for "documented gap" tests where the test proves a known limitation exists.
- **FAILED**: one or more assertions failed; pipeline should not be considered reliable until fixed.

### 5.4 Timestamp in Filename

`tests/reports/review_20240115_143201.md` — YYYYMMDD_HHMMSS format. Easy to sort, no colons (Windows-compatible).

### 5.5 What Each Test Block Must Contain on Failure

```
Test ID: TC5
Test Name: Duplicate values in rename_map
Agent: Agent 1
Category: Edge Case — Data Integrity
Status: FAILED

Expected:
  column_map values should be unique (no two raw columns mapped to same clean name)

Received:
  column_map = {"msisdn": "Msisdn", "mobile_no": "Msisdn"}
  Duplicate values: ["Msisdn"]

Probable Cause:
  Agent 1 does not validate rename_map value uniqueness. LLM-generated duplicates
  pass through silently. Pandas .rename() will create duplicate column names in the df.

Suggested Fix:
  In agent_1_field_renamer.py, after extracting column_map, add duplicate value check.
  Raise ValueError or log a warning and remove the duplicate entry.

Fixture Used:
  tests/fixtures/agent_1_fixtures.json → "duplicate_values_in_rename_map"
```

---

## 6. Input Config Refactor

### 6.1 Current State of main.py

`main.py` hardcodes:
- `file_path = r"data/telecom_churn_data.csv"`
- Opens `r"queries/churn_query.sql"` and `r"rules/special_rules.txt"` directly.

This is unsuitable for: (a) multi-client pipelines, (b) test harness (tests need to inject different files), (c) Streamlit UI.

### 6.2 What the Config Module Must Support

Requirements from the brief + architecture doc:
1. Multiple clients — each client has their own file_path, sql_query, special_rules.
2. Replaceable by Streamlit UI later — the config module is the boundary layer.
3. Must not break existing pipeline — backward compatible.
4. LLM config lives here too (combined with LLM abstraction).

### 6.3 Proposed Module Structure

```
source_code/
  config/
    __init__.py
    pipeline_config.py    ← PipelineConfig, LLMConfig dataclasses
    loaders.py            ← load_from_files(), load_from_env(), load_from_dict()
```

`loaders.py` is the key piece — it replaces the hardcoded file reads:

```python
# source_code/config/loaders.py

def load_pipeline_config_from_files(
    data_path: str,
    query_path: str,
    rules_path: str,
    client_id: str = "default",
    llm_config: LLMConfig | None = None
) -> dict:
    """
    Load pipeline inputs from files. Returns initial_input dict
    compatible with AgentState. This is the "file mode" loader —
    later, a Streamlit loader will replace this.
    """
    with open(query_path, "r", encoding="utf-8") as f:
        sql_query = f.read()
    with open(rules_path, "r", encoding="utf-8") as f:
        special_rules = f.read()

    df = pd.read_csv(data_path)
    # ... build metadata, classify columns, etc.

    return {
        "file_path": data_path,
        "sql_query": sql_query,
        "special_rules": special_rules,
        "client_id": client_id,
        # ... rest of initial_input
    }
```

### 6.4 Multi-Client Support Design

For multi-client support, each client gets a directory:

```
clients/
  client_telco_a/
    data.csv
    query.sql
    rules.txt
  client_telco_b/
    data.csv
    query.sql
    rules.txt
```

The config loader takes a `client_id` and maps it to the right directory:

```python
def load_client_config(client_id: str, base_dir: str = "clients/") -> dict:
    client_dir = os.path.join(base_dir, client_id)
    return load_pipeline_config_from_files(
        data_path=os.path.join(client_dir, "data.csv"),
        query_path=os.path.join(client_dir, "query.sql"),
        rules_path=os.path.join(client_dir, "rules.txt"),
        client_id=client_id
    )
```

For the current single-client setup, retain backward compatibility:
```python
# main.py — after refactor
config = load_pipeline_config_from_files(
    data_path="data/telecom_churn_data.csv",
    query_path="queries/churn_query.sql",
    rules_path="rules/special_rules.txt"
)
```

This is identical in effect to current behavior but routes through the config module.

### 6.5 Streamlit Replacement Path

When the Streamlit UI is built, it will construct the `initial_input` dict from form widgets (file upload, text area for rules, dropdown for SQL). The config module exposes a `load_from_dict(user_inputs: dict)` that validates and transforms this into the same shape as `load_from_files`. Main pipeline code never changes — only the loader changes.

### 6.6 AgentState Backward Compatibility

The brief says "AgentState does not grow new fields unnecessarily." `client_id` might be useful for multi-client routing in LangGraph (it maps to `thread_id`). But `client_id` is a LangGraph-level concept, not a data-processing concept. It should live in the LangGraph config dict (`{"configurable": {"thread_id": client_id}}`), not in `AgentState`. No new fields needed in `AgentState` for this refactor.

---

## 7. Edge Cases and Risks

### 7.1 Mock LLM Risks

**Risk 1: Fixture divergence from reality.**
MockLLM fixtures will be written once. As the real LLM prompt evolves (abbreviation_map grows, new rules added), the fixture responses may no longer represent what the real LLM would produce. Over time, tests pass against stale fixtures but the real LLM produces different (possibly worse) output.

Mitigation: add a `fixture_version` field to each fixture file and a `prompt_hash` field that records the hash of the prompt template at fixture creation time. If the prompt template changes (via the JSON config), flag a warning in the test output: "prompt template has changed since fixture was generated — fixture may be stale."

**Risk 2: MockLLM too simple for multi-message prompts.**
Currently agents use string prompts. If future agents use multi-turn conversation history (list of messages), `MockLLM.invoke()` will receive a list, not a string. Design `MockLLM.invoke()` to accept both:
```python
def invoke(self, prompt):
    self.last_prompt = prompt if isinstance(prompt, str) else str(prompt)
    return MockResponse(self._content)
```

**Risk 3: Tests pass when agents fail.**
If fixture response is malformed in a way that triggers the agent's graceful degradation, the test might assert on the degraded output and mark it PASSED when the real behavior is a silent failure. Every test must explicitly assert what it is testing — "TC4 tests graceful degradation and asserts degraded output" is fine if stated.

### 7.2 Agent 1 Gaps Not in the Brief

**Gap 1: Preprocess vs raw column name mismatch in column_map.**
`preprocess_column_names` returns a mapping of `{original: cleaned}`. The rename_map keys from the LLM should match the *cleaned* names (what the agent passes as `cleaned_cols` to the prompt). But `column_map` is supposed to be an audit trail from `original` to `standardized`. Currently, `column_map` contains LLM keys (cleaned names) mapped to LLM values (standardized names). The outer mapping from original to cleaned is lost.

For the executor, this doesn't matter — it runs `df.rename(columns=rename_map)` where rename_map keys are the current column names (which are already pre-cleaned after `preprocess_column_names` output is used... wait, actually NO — the pre-processing only generates the string to PUT IN THE PROMPT, the actual `df.columns` still have the original names with aliases).

This is a genuine bug: if `df.columns` contains `"a.msisdn"` but the rename_map (from LLM) has key `"msisdn"` (because the LLM was given the cleaned version), then `df.rename(columns=rename_map)` will NOT rename `"a.msisdn"` because the key doesn't match.

The executor receives the raw CSV with original column names. The LLM generates keys based on `cleaned_cols` (alias-stripped). There is a mismatch. Either:
- The executor must pre-clean columns before renaming (apply `preprocess_column_names` first), OR
- Agent 1 must pass `column_map` with original keys (reconstruct the mapping: `{original: standardized}` by composing `pre_cleaned` dict + LLM rename_map).

This is a significant architectural issue that should be documented as a bug and addressed in the test harness. Add a test case: "Column with alias prefix — verify executor correctly applies rename to original-named column."

**Gap 2: rename_map keys use cleaned names, not original names.**
As above. Recommend: Agent 1 should reconstruct `column_map` as `{original_name: final_standardized_name}` by composing the two mappings. The `cleaning_code` (what the executor runs) should also apply `preprocess_column_names` first OR should use the original column names as keys.

**Gap 3: No test for `preprocess_column_names` utility directly.**
`preprocess_column_names` is pure Python with no external dependencies. It should have its own unit tests separate from Agent 1 tests. Test cases: single-letter alias (`a.msisdn`), multi-char alias (`cust.name`), no alias, already lowercase, mixed case, leading/trailing spaces.

### 7.3 Agent 2 Gaps Not in the Brief

**Gap 4: Agent 2 output_path hardcoded.**
`agent_2_field_cleaner.py` always returns `output_path = "output_agent2_cleaned.csv"`. This is never used by the executor (which hardcodes `"standardized_output.csv"`). The `output_path` field in `AgentState` is populated but silently ignored. Write a test that verifies `state["output_path"]` is set after Agent 2 and equals `"output_agent2_cleaned.csv"` — then separately document that the executor currently ignores it.

**Gap 5: Agent 2 does not receive the SQL query.**
As noted in TC11 above. Agent 2's prompt template has no `{sql_query}` placeholder. Whether this is intentional or an omission should be clarified. By the time Agent 2 runs, columns are renamed — but knowing the original SQL could help Agent 2 understand which columns are critical (e.g. the revenue column used in the churn definition should not have nulls introduced by aggressive cleaning). A test that asserts `sql_query NOT in mock_llm.last_prompt` documents this design decision.

**Gap 6: `if 'col' in df.columns` guards.**
Agent 2's prompt instructs the LLM to wrap every operation in `if col in df.columns`. But if the LLM omits this guard (which it sometimes does), a `KeyError` or `AttributeError` will occur. A test with a fixture that omits the guard is valuable to verify executor catches this.

### 7.4 Integration Test: Agent 1 → Agent 2 Chain

The most important uncovered scenario is the full chain: Agent 1 renames columns → Executor applies renames → Agent 2 receives renamed columns. Key questions:

1. Agent 2 receives `categorical_cols` and `numeric_like_cols` from `AgentState`. These are classified BEFORE Agent 1 renames the columns (in `main.py`). After Agent 1 renames, the column names in those lists are stale — they still use pre-rename names. Agent 2's cleaning code would use the old names, which no longer exist in the df, causing silent no-ops or KeyErrors.

This is a significant pipeline bug. The `classify_columns` + `build_value_counts_summary` + `build_null_summary` calls in `main.py` run against the raw (un-renamed) DataFrame. But Agent 2 runs after the executor applies Agent 1's renames. So Agent 2 is given column names that no longer match the df.

Test: integration test that runs Agent 1 (with fixture) → Executor → Agent 2 setup, and asserts that the columns Agent 2 is told to clean actually exist in the DataFrame it receives.

Mitigation design: the Executor after Agent 1 should update `categorical_cols`, `numeric_like_cols`, etc. in state using the new column names (via `column_map`). OR: Agent 2 should receive a freshly classified column list based on the post-rename df. Either way, the current code is broken for any dataset with non-trivial column names.

### 7.5 Executor-Specific Risks

**Risk: exec() namespace pollution.**
`exec(cleaning_code, {}, local_vars)` uses an empty global namespace `{}`. If cleaning code does `import numpy as np`, this will fail because the empty globals dict has no builtins. The fix is `exec(cleaning_code, {"__builtins__": __builtins__, "pd": pd}, local_vars)`. Agent 2's prompt explicitly generates `import numpy as np` at the top of cleaning code. This will fail in the executor as currently written.

Write a test: fixture cleaning code includes `import numpy as np` and uses `np.nan`. Assert executor handles this correctly (i.e. `error_log` should be `None`).

**Risk: output file path conflicts.**
Both executor calls (after Agent 1 and after Agent 2) write to different hardcoded paths (`"standardized_output.csv"` for both, since the executor ignores `state["output_path"]`). The second executor call would overwrite the first. Tests should verify output file content is from the correct agent's code.

### 7.6 Test Runner Design

The test runner (`tests/run_tests.py`) must:
1. Discover all test classes/functions.
2. Run each, catching exceptions (a test error must not abort the run).
3. Collect results with metadata.
4. Generate the review document.

Use Python's standard `unittest` or `pytest` for discovery — do not reinvent. Pytest is preferable because:
- Fixtures (pytest fixtures, distinct from our LLM fixtures) handle setup/teardown cleanly.
- Parametrize decorator allows scenario-driven tests without boilerplate.
- The `--agent 1` filter can be implemented as a pytest mark: `@pytest.mark.agent1`.

However, the brief specifies `python tests/run_tests.py` as the entry point (not `pytest`). Solution: `run_tests.py` simply calls `pytest` programmatically via `pytest.main()` and then generates the markdown report from pytest's JSON output (`--json-report` plugin or `--tb=json`).

```python
# tests/run_tests.py
import pytest
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    agent_filter = None
    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        agent_filter = sys.argv[idx + 1]

    args = ["tests/", "-v", "--tb=short", "--json-report",
            f"--json-report-file=tests/.pytest_report.json"]
    if agent_filter:
        args += ["-m", f"agent{agent_filter}"]

    pytest.main(args)
    generate_review_document()

def generate_review_document():
    with open("tests/.pytest_report.json") as f:
        report = json.load(f)
    # ... build markdown from report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(f"tests/reports/review_{ts}.md").write_text(...)
```

### 7.7 File Structure for the Full Feature

```
source_code/
  config/
    __init__.py
    pipeline_config.py
    loaders.py
  llm/
    __init__.py
    factory.py
    mock_llm.py
  agents/
    agent_1_field_renamer.py   ← refactored (injection)
    agent_2_field_cleaner.py   ← refactored (injection)
    executor.py                ← refactored (output_path from state)
  graph.py                     ← refactored (passes config to agent factories)
  state.py                     ← unchanged
  utils.py                     ← unchanged

tests/
  __init__.py
  run_tests.py
  fixtures/
    agent_1_fixtures.json
    agent_2_fixtures.json
    sample_telecom.csv
  unit/
    test_agent_1.py
    test_agent_2.py
    test_executor.py
    test_utils.py
    test_llm_factory.py
  integration/
    test_agent1_executor_chain.py
    test_agent2_executor_chain.py
  reports/
    .gitkeep

main.py                        ← refactored (uses config loader)
```

---

## 8. Summary of Open Questions for the Architect

1. **Agent 1 column_map key mismatch bug**: Should the fix be in Agent 1 (reconstruct map with original keys) or in the Executor (apply preprocess_column_names before exec)? This is a correctness bug that must be addressed before tests are written.

2. **Agent 2 SQL context**: Should Agent 2 receive `sql_query`? If so, which placeholder in the prompt template and at what position in the instructions?

3. **Stale categorical_cols/numeric_like_cols after Agent 1 rename**: Should the Executor reclassify columns post-rename, or should Agent 2 receive a fresh classification? Reclassifying in the Executor is cleaner but requires the Executor to become more intelligent than "just run exec()."

4. **LLM instantiation caching**: Instantiate LLM once per pipeline session and cache in PipelineConfig, or instantiate fresh per agent call? The brief implies session-level: "LLM selection happens at pipeline initialisation." Caching is preferred for latency.

5. **exec() globals namespace**: The executor's `exec(cleaning_code, {}, local_vars)` with empty globals will break `import numpy`. Fix: pass `{"__builtins__": __builtins__, "pd": pd, "np": np}` as globals. Confirm this is in scope for this feature.

6. **Executor output_path**: Should the executor use `state.get("output_path")` or keep its own hardcoded path? The `output_path` field exists in `AgentState` for this purpose but is currently unused by the executor.

---

## Key Decisions Summary

1. `LLMFactory.create(provider, model, api_key, ...)` returns a `BaseChatModel`-compatible instance. One file to change when adding providers.
2. `PipelineConfig` + `LLMConfig` dataclasses; injected via closure into agent factory functions; NOT stored in `AgentState`.
3. `MockLLM` stores `last_prompt` after `.invoke()` — enables both pre-LLM (prompt injection) and post-LLM (output extraction) assertions.
4. Fixtures: one JSON per agent, keyed by scenario name.
5. Test runner: `pytest` under the hood, `run_tests.py` wrapper generates markdown report.
6. Report structure: summary table → per-agent sections → detailed failure blocks.
7. Config refactor: `source_code/config/loaders.py` replaces file reads in main.py; multi-client via `clients/<client_id>/` directories.
8. Three undocumented bugs found: (a) alias-prefix mismatch in column_map keys vs executor df, (b) stale categorical_cols names after Agent 1 renames, (c) exec() empty globals breaks numpy imports.
9. Agent 2 SQL context gap: Agent 2 does not currently receive the SQL query — needs Architect decision.
10. The "duplicate values" and "column drops" tests prove gaps exist — they are documentation tests as much as correctness tests.
