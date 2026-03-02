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
**Status:** 🔧 Done, untested
**File:** `source_code/agents/agent_1_field_renamer.py`
**Prompt:** `source_code/prompts/agent_1.json`

**Purpose:** Standardizes raw column names from the client dataset to a clean, telecom-standard PascalCase naming convention.

**Reads from state:**
- `df_columns` — raw column names
- `sql_query` — for SQL alias resolution context
- `metadata_summary` — dtypes + sample row
- `special_rules` — client-specific overrides

**Writes to state:**
- `cleaning_code` — Python rename code (`rename_map` + `df.rename()`)
- `ambiguous_fields` — list of `{original_column, candidates, reason, sample_values}`
- `column_map` — `{original: standardized}` audit dict

**Interrupt:** Yes — surfaces `ambiguous_fields` for human resolution before executor runs.

**Output contract:** 2 code blocks — rename code, then ambiguous_fields assignment.

---

## Agent 2 — Field Cleaner
**Status:** 🔧 Done, untested
**File:** `source_code/agents/agent_2_field_cleaner.py`
**Prompt:** `source_code/prompts/agent_2.json`

**Purpose:** Cleans dirty field values — fixes encoding issues, standardizes categorical values, converts numeric-like object columns to proper numeric types.

**Reads from state:**
- `metadata_summary`
- `categorical_cols` — genuinely categorical object columns
- `numeric_like_cols` — object columns that are actually numeric but dirty
- `value_counts_summary` — top 20 value counts per categorical column
- `null_summary` — null counts per column
- `special_rules`

**Writes to state:**
- `cleaning_code` — Python cleaning code
- `flagged_columns` — list of `{column, reason}` for columns that couldn't be safely cleaned
- `output_path` — where executor saves the result

**Interrupt:** Yes — surfaces `flagged_columns` for human review.

**Output contract:** 2 code blocks — cleaning code, then flagged_columns assignment.

---

## Executor (Shared)
**Status:** 🔧 Done, untested
**File:** `source_code/agents/executor.py`

**Purpose:** Runs LLM-generated Python code against the dataset. Shared across all executor nodes in the graph.

**Reads from state:**
- `file_path` — always reloads from source CSV
- `cleaning_code` — code to execute

**Exec scope available to generated code:** `df`, `pd`

**Writes to state:**
- `error_log` — None on success, error message + traceback on failure
- Saves output to `standardized_output.csv` (currently hardcoded)

**Note:** Currently always saves to `standardized_output.csv` regardless of `output_path` in state. This needs to be fixed — executor should read `output_path` from state and chain CSVs between stages.

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
