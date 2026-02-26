from typing import TypedDict, Optional

class AgentState(TypedDict):
    # ── Input ──────────────────────────────
    file_path       : str
    target_column   : str
    metadata_summary: str
    sql_query       : str
    df_columns      : list
    special_rules   : str

    # ── Agent Outputs ───────────────────────
    cleaning_code   : Optional[str]
    ambiguous_fields: Optional[list]
    column_map      : Optional[dict]

    # ── Pipeline Control ────────────────────
    iteration_count : Optional[int]
    error_log       : Optional[str]