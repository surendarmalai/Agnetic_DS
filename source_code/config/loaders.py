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
    with open(query_path, "r", encoding="utf-8") as f:
        return f.read()


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
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


def load_client_config(
    client_id    : str,
    base_dir     : str = "clients/",
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
    client_dir = os.path.join(base_dir, client_id)
    return _build_initial_input(
        data_path=os.path.join(client_dir, "data.csv"),
        query_path=os.path.join(client_dir, "query.sql"),
        rules_path=os.path.join(client_dir, "rules.txt"),
        client_id=client_id,
        target_column=target_column,
    )


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
    return _build_initial_input(
        data_path=data_path,
        query_path=query_path,
        rules_path=rules_path,
        client_id=client_id,
        target_column=target_column,
    )


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
    sql_query = load_query(query_path)
    special_rules = load_rules(rules_path)
    df = pd.read_csv(data_path)
    dtypes_dict = df.dtypes.astype(str).to_dict()
    sample_row = df.head(1).to_dict(orient="records")[0]
    metadata_summary = (
        f"DATASET PROFILE:\n"
        f"1. COLUMNS & TYPES:\n{dtypes_dict}\n"
        f"2. SAMPLE ROW:\n{sample_row}"
    )
    categorical_cols, numeric_like_cols, true_numeric_cols = classify_columns(df)
    value_counts_summary = build_value_counts_summary(df, categorical_cols, top_n=20)
    null_summary = build_null_summary(df)
    return {
        "file_path"           : data_path,
        "target_column"       : target_column,   # [AUDIT M10] explicit, not ""
        "sql_query"           : sql_query,
        "df_columns"          : list(df.columns),
        "special_rules"       : special_rules,
        "iteration_count"     : 0,
        "metadata_summary"    : metadata_summary,
        "categorical_cols"    : categorical_cols,
        "numeric_like_cols"   : numeric_like_cols,
        "true_numeric_cols"   : true_numeric_cols,
        "value_counts_summary": value_counts_summary,
        "null_summary"        : null_summary,
    }
