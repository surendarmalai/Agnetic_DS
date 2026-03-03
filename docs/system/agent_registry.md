# Agent Registry — Agnetic DS

All agents in the pipeline. Update this file whenever an agent is added, modified, or removed.

---

## Status Legend
- ✅ Done & tested
- 🔧 Done, untested
- 🗓 Planned
- ❌ Deprecated

---

## Agent 1 — Field Renamer
**Status:** ✅ Done & tested
**File:** `source_code/agents/agent_1_field_renamer.py`
**Prompt:** `source_code/prompts/agent_1.json`
**Public API:** `make_field_renamer_agent(config: PipelineConfig) -> Callable[[AgentState], dict]`

**Purpose:** Standardizes raw column names from the client dataset to a clean, telecom-standard PascalCase naming convention.

**Reads from state:**
- `df_columns` — raw column names
- `sql_query` — for SQL alias resolution context
- `metadata_summary` — dtypes + sample row
- `special_rules` — client-specific overrides

**Writes to state:**
- `cleaning_code` — Python rename map code block (used as audit reference; executor reconstructs rename independently)
- `ambiguous_fields` — list of `{original_column, candidates, reason, sample_values}`
- `column_map` — `{original: standardized}` audit dict consumed by `rename_executor_agent`

**Interrupt:** Yes — surfaces `ambiguous_fields` for human resolution before executor runs. (Not yet wired — interrupt() not implemented.)

**Output contract:** 2 code blocks — rename code, then ambiguous_fields assignment.

**Test coverage:** 9 TCs (TC1–TC9). 6 PASS, 3 xfail (documented gaps: TC5 duplicate map targets, TC6 partial map, TC8 key mismatch).

**Note:** The old module-level `field_renamer_agent` function no longer exists. The agent is only accessible as a closure returned by the factory. `ChatGroq` is no longer imported in this file — LLM is injected via `PipelineConfig`.

---

## Agent 2 — Field Cleaner
**Status:** ✅ Done & tested
**File:** `source_code/agents/agent_2_field_cleaner.py`
**Prompt:** `source_code/prompts/agent_2.json`
**Public API:** `make_field_cleaner_agent(config: PipelineConfig) -> Callable[[AgentState], dict]`

**Purpose:** Cleans dirty field values — fixes encoding issues, standardizes categorical values, converts numeric-like object columns to proper numeric types.

**Reads from state:**
- `metadata_summary`
- `categorical_cols` — genuinely categorical object columns (refreshed by `reclassify_columns_node` post-rename)
- `numeric_like_cols` — object columns that are actually numeric but dirty (refreshed post-rename)
- `value_counts_summary` — top 20 value counts per categorical column (refreshed post-rename)
- `null_summary` — null counts per column (refreshed post-rename)
- `special_rules`

**Writes to state:**
- `cleaning_code` — Python cleaning code executed by `cleaning_executor_agent`
- `flagged_columns` — list of `{column, reason}` for columns that couldn't be safely cleaned

**Interrupt:** Yes — surfaces `flagged_columns` for human review. (Not yet wired — interrupt() not implemented.)

**Output contract:** 2 code blocks — cleaning code, then flagged_columns assignment.

**Test coverage:** 8 TCs (TC10–TC17). 7 PASS (including TC15 xpassed), 1 xfail (TC16: agent does not detect column drops in generated code).

**Note:** The old module-level `field_cleaner_agent` function no longer exists. The agent is only accessible as a closure returned by the factory. `ChatGroq` is no longer imported in this file.

---

## Executor — Rename Executor
**Status:** ✅ Done & tested
**File:** `source_code/agents/executor.py`
**Function:** `rename_executor_agent(state: AgentState) -> dict`

**Purpose:** Applies the composite column rename map to the raw CSV. Does NOT call exec(). Reconstructs the rename from `state["column_map"]` + `preprocess_column_names`.

**Reads from state:**
- `file_path` — original raw CSV (always reads from source)
- `column_map` — `{cleaned_name: standardized_name}` audit dict from Agent 1
- `output_path` — output path (default: `"standardized_output_renamed.csv"`)

**Exec scope:** None. No exec() call in this function.

**Writes to state:**
- `output_path` — path where renamed CSV was saved
- `error_log` — None on success; WARNING strings for dropped columns or new nulls

**Safety checks:** Column count drop detection; new null detection (via inline logic, not `_check_safety`).

**Test coverage:** 6 TCs (TC18–TC23) covering rename executor and cleaning executor combined.

---

## Executor — Cleaning Executor
**Status:** ✅ Done & tested
**File:** `source_code/agents/executor.py`
**Function:** `cleaning_executor_agent(state: AgentState) -> dict`

**Purpose:** Runs Agent 2's generated cleaning code against the post-rename CSV via exec(). Applies safety checks after execution.

**Reads from state:**
- `output_path` — post-rename CSV path (written by `rename_executor_agent`)
- `cleaning_code` — Python cleaning code from Agent 2

**Exec scope available to generated code:** `df`, `pd`, `np`, `__builtins__`

**Writes to state:**
- `output_path` — `"output_agent2_cleaned.csv"` (hardcoded)
- `error_log` — None on success; WARNING strings for dropped columns or new nulls

**Safety checks:** `_check_safety(df_before, df_after)` detects dropped columns (set diff) and new null values (sum diff). Warnings appended to error_log; execution is not aborted.

**Test coverage:** TC21 (column drop detected — xpassed), TC22 (new nulls detected — xpassed).

**Known gaps:** `cleaning_executor_agent` output path is hardcoded to `"output_agent2_cleaned.csv"`. Should be read from a configurable state field in a future improvement.

---

## reclassify_columns_node
**Status:** ✅ Done & tested (via graph topology verification in test suite)
**File:** `source_code/reclassify.py`
**Function:** `reclassify_columns_node(state: AgentState) -> dict`

**Purpose:** Re-classifies columns after Agent 1 rename so Agent 2 receives correct post-rename metadata. Not an agent (no LLM call). Placed in graph between executor1 and agent2_cleaner.

**Reads from state:**
- `output_path` — post-rename CSV (default fallback: `"standardized_output_renamed.csv"`)

**Writes to state:**
- `categorical_cols`, `numeric_like_cols`, `true_numeric_cols`, `value_counts_summary`, `null_summary`

---

## Agent 3 — Query Builder
**Status:** 🗓 Planned

**Purpose:** Given a churn definition and client schema context, writes the SQL query to pull the right dataset from the client's database.

**Reads from state:** churn definition (from human), schema docs (from file upload), DB connection config

**Writes to state:** `sql_query`, optionally asks human to approve before execution

**Interrupt:** Yes — human approves the query before it runs against the client DB.

---

## Agent 4 — Feature Engineer
**Status:** 🗓 Planned

**Purpose:** Builds model-ready features — lag features, ratios, activity flags, behavioural deltas.

**Reads from state:** cleaned df, column_map (to know what columns exist post-rename), special_rules, human suggestions from interrupt

**Writes to state:** feature engineering code, list of engineered features with descriptions

**Interrupt:** Yes — surfaces proposed features for human to approve, add, or remove before execution.

---

## Agent 5 — EDA
**Status:** 🗓 Planned

**Purpose:** Surfaces data quality issues, distributions, correlations, class imbalance, potential leakage columns.

**Writes to state:** EDA report (text + charts), recommended actions

**Interrupt:** Yes — human reviews before proceeding to model building.

---

## Agent 6 — Model Builder
**Status:** 🗓 Planned

**Purpose:** Trains multiple classification algorithms (Logistic Regression, XGBoost, Random Forest, LightGBM minimum).

**Writes to state:** trained model objects, training metadata

---

## Agent 7 — Evaluator
**Status:** 🗓 Planned

**Purpose:** Computes AUC, KS statistic, Gini, precision, recall, F1, lift charts.

**Writes to state:** metrics dict, evaluation report

**Interrupt:** Yes — human reviews metrics and decides: proceed, loop back for more data, or tune.

---

## Agent 8 — Data Augmentor (Conditional)
**Status:** 🗓 Planned

**Purpose:** If evaluation is unsatisfactory, re-queries with broader or different criteria to pull more/better data. Loops back into the pipeline.

---

## Agent 9 — Hyperparameter Tuner
**Status:** 🗓 Planned

**Purpose:** Optimises the best-performing model from evaluation.

---

## Agent 10 — Model Comparator
**Status:** 🗓 Planned

**Purpose:** Compares all candidate models, selects the winner based on agreed metrics.

---

## Agent 11 — Reporter
**Status:** 🗓 Planned

**Purpose:** Generates a model card and performance report for client delivery.

**Interrupt:** Yes — human reviews and approves before final output is produced.
