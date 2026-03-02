# Pipeline Stages — Agnetic DS

Full end-to-end pipeline for the churn product. This is the living reference for what's built, what's planned, and what's in progress.

## Status Legend
- ✅ Done & tested
- 🔧 Done, untested
- 🚧 In progress
- 🗓 Planned
- ❌ Blocked / deprecated

---

## Stage 0 — Project Setup & Human Context Injection
**Status:** 🗓 Planned (part of Streamlit UI)

The user opens the system and provides upfront context before the pipeline starts:
- Client name / project identifier
- Churn definition (e.g. "active in last 3 days, inactive in next 25 days")
- DB connection details
- Upload: data dictionary, schema docs, business rules, previous model reports
- Any known important features or domain rules

This context is loaded into `AgentState` as `special_rules` + uploaded file contents.

---

## Stage 1 — Query Building
**Status:** 🗓 Planned (Agent 3)

- LLM writes SQL query given churn definition + schema context
- **[INTERRUPT]** Human reviews and approves query before it runs
- Human can edit, reject and re-generate, or approve

---

## Stage 2 — Data Retrieval
**Status:** 🗓 Planned

- Executor runs approved SQL against client DB
- Loads result as a pandas DataFrame
- **[INTERRUPT]** Human validates: row count, date range, presence of key columns

---

## Stage 3 — Column Standardization
**Status:** 🔧 Done, untested (Agent 1 + Executor 1)

- Strips SQL aliases, lowercases column names
- LLM maps raw names to PascalCase telecom-standard names
- **[INTERRUPT]** Human resolves ambiguous column mappings
- Executor applies rename

---

## Stage 4 — Field Cleaning
**Status:** 🔧 Done, untested (Agent 2 + Executor 2)

- Classifies columns: categorical / numeric-like (dirty) / true numeric
- LLM writes cleaning code: fixes encoding, standardizes categoricals, converts dirty numerics
- **[INTERRUPT]** Human reviews flagged columns that couldn't be safely cleaned
- Executor applies cleaning

---

## Stage 5 — Feature Engineering
**Status:** 🗓 Planned (Agent 4)

- Builds model-ready features: lag features, ratios, activity flags, behavioural deltas
- LLM proposes feature set based on column_map + domain knowledge
- **[INTERRUPT]** Human adds, removes, or modifies proposed features
- Executor applies feature engineering

---

## Stage 6 — EDA & Data Validation
**Status:** 🗓 Planned (Agent 5)

- Distributions, correlations, class imbalance, potential leakage
- Flags data quality issues
- **[INTERRUPT]** Human reviews findings, approves to proceed or requests fixes

---

## Stage 7 — Model Building
**Status:** 🗓 Planned (Agent 6)

- Trains: Logistic Regression, XGBoost, Random Forest, LightGBM (at minimum)
- Uses cleaned + engineered df
- Outputs: trained model objects + training metadata

---

## Stage 8 — Model Evaluation
**Status:** 🗓 Planned (Agent 7)

- Metrics: AUC, KS statistic, Gini, precision, recall, F1, lift curve
- **[INTERRUPT]** Human reviews metrics, decides:
  - ✅ Proceed to tuning
  - 🔁 Loop back — fetch more data (→ Stage 1 with modified query)
  - 🔁 Loop back — add more features (→ Stage 5)
  - ❌ Abort

---

## Stage 9 — Data Augmentation (Conditional)
**Status:** 🗓 Planned (Agent 8)

- Only runs if human decides more data is needed
- Re-queries with broader or different criteria
- Re-enters pipeline from Stage 2

---

## Stage 10 — Hyperparameter Tuning
**Status:** 🗓 Planned (Agent 9)

- Optimises the best-performing model from evaluation
- Reports tuned vs baseline metrics

---

## Stage 11 — Model Comparison & Selection
**Status:** 🗓 Planned (Agent 10)

- Compares all candidate models
- Selects winner based on agreed primary metric (typically KS or AUC for churn)

---

## Stage 12 — Reporting
**Status:** 🗓 Planned (Agent 11)

- Generates model card + performance report
- **[INTERRUPT]** Human reviews and approves before final delivery

---

## Feedback Loops

```
Evaluation (Stage 8)
  ├── needs more data    → back to Stage 1 (new query)
  ├── needs more features → back to Stage 5
  └── approved          → forward to Stage 10
```
