from typing import TypedDict, Optional


class AgentState(TypedDict):

    # ── Core inputs ─────────────────────────────────────────────────────────────
    file_path             : str
    target_column         : str
    sql_query             : str
    df_columns            : list
    special_rules         : str

    # ── Standard metadata (Agent 1) ──────────────────────────────────────────────
    metadata_summary      : str

    # ── Enriched metadata (Agent 2 cleaning) ─────────────────────────────────────
    categorical_cols      : Optional[list]   # object cols that are genuinely categorical
    numeric_like_cols     : Optional[list]   # object cols that are actually numeric but dirty
    true_numeric_cols     : Optional[list]   # already numeric dtype cols
    value_counts_summary  : Optional[str]    # top 20 value_counts for categorical cols
    null_summary          : Optional[str]    # isna().sum() for all cols

    # ── Agent outputs ─────────────────────────────────────────────────────────────
    cleaning_code         : Optional[str]    # generated Python code from any agent
    ambiguous_fields      : Optional[list]   # fields Agent 1 could not confidently map
    column_map            : Optional[dict]   # {original: standardized} audit trail
    flagged_columns       : Optional[list]   # columns Agent 2/3 could not safely handle

    # ── Pipeline control ──────────────────────────────────────────────────────────
    iteration_count       : Optional[int]
    error_log             : Optional[str]
    output_path           : Optional[str]    # executor saves cleaned df here