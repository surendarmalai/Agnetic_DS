import pandas as pd
from source_code.state import AgentState
from source_code.utils import classify_columns, build_value_counts_summary, build_null_summary


def reclassify_columns_node(state: AgentState) -> dict:
    """
    Graph node: re-runs column classification after Agent 1 renames columns.

    Reads the post-rename CSV from state["output_path"], applies classify_columns,
    build_value_counts_summary, and build_null_summary, and returns updated
    metadata fields so Agent 2 sees the correctly-named columns.

    Must be placed in graph: executor1 -> reclassify_columns -> agent2_cleaner.

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
    output_path = state.get("output_path", "standardized_output_renamed.csv")
    df = pd.read_csv(output_path)   # raises FileNotFoundError if file missing
    categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    null_summary = build_null_summary(df)
    return {
        "categorical_cols"    : categorical_cols,
        "numeric_like_cols"   : numeric_like_cols,
        "true_numeric_cols"   : true_numeric_cols,
        "value_counts_summary": value_counts_summary,
        "null_summary"        : null_summary,
    }
