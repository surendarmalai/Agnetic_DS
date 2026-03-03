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
    file_path = state["file_path"]                  # [AUDIT L13] reads original file
    column_map = state.get("column_map", {})
    output_path = state.get("output_path", "standardized_output_renamed.csv")

    try:
        df = pd.read_csv(file_path)
        df_before = df.copy()

        if column_map:
            pre_cleaned = preprocess_column_names(list(df.columns))
            composite_map = {
                orig: column_map[cleaned]
                for orig, cleaned in pre_cleaned.items()
                if cleaned in column_map
            }
            df = df.rename(columns=composite_map)

        # For rename executor: check by column count (not name set) since
        # renaming legitimately changes all column names in the set.
        # A count drop indicates a genuine duplicate-key collision in the map.
        warnings = []
        if len(df.columns) < len(df_before.columns):
            dropped_count = len(df_before.columns) - len(df.columns)
            warnings.append(f"WARNING: {dropped_count} column(s) lost after rename (duplicate map targets)")
        new_nulls = df.isna().sum().sum() - df_before.isna().sum().sum()
        if new_nulls > 0:
            warnings.append(f"WARNING: {new_nulls} new null values introduced")

        df.to_csv(output_path, index=False)

        error_log = "\n".join(warnings) if warnings else None
        return {"output_path": output_path, "error_log": error_log}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return {"error_log": error_msg}


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
    input_path = state.get("output_path", "standardized_output_renamed.csv")
    cleaning_code = state.get("cleaning_code", "")
    final_output_path = "output_agent2_cleaned.csv"

    if not cleaning_code:
        return {"error_log": "No cleaning code provided by the agent."}

    try:
        df = pd.read_csv(input_path)        # [AUDIT L13] reads from output_path, not file_path
        df_before = df.copy()

        local_vars = {"df": df, "pd": pd}
        exec_globals = {"__builtins__": __builtins__, "pd": pd, "np": np}  # [AUDIT L12]
        exec(cleaning_code, exec_globals, local_vars)
        df_clean = local_vars["df"]

        warnings = _check_safety(df_before, df_clean)
        df_clean.to_csv(final_output_path, index=False)

        error_log = "\n".join(warnings) if warnings else None
        return {"output_path": final_output_path, "error_log": error_log}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return {"error_log": error_msg}


def _check_safety(df_before: pd.DataFrame, df_after: pd.DataFrame) -> list:
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
    warnings = []
    dropped = set(df_before.columns) - set(df_after.columns)
    if dropped:
        warnings.append(f"WARNING: columns dropped: {sorted(dropped)}")
    new_nulls = df_after.isna().sum().sum() - df_before.isna().sum().sum()
    if new_nulls > 0:
        warnings.append(f"WARNING: {new_nulls} new null values introduced")
    return warnings
