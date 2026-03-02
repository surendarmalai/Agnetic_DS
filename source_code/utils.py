import re
import json
import os
import pandas as pd

def load_prompts(path):
    path = os.path.join(path)
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_column_names(columns: list) -> dict:
    """
    Fixes column names by stripping table aliases, lowercasing, and trimming whitespace.
    Returns a mapping of original column name to cleaned version.
    """
    cleaned = {}
    for col in columns:
        c = col.strip()
        c = re.sub(r'^[a-zA-Z]\.', '', c)
        c = re.sub(r'^[a-zA-Z_]+\.', '', c)
        c = c.strip().lower()
        cleaned[col] = c
    return cleaned

def classify_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Classifies DataFrame columns into three groups:

    - categorical_cols   : object dtype with genuinely string/categorical values
    - numeric_like_cols  : object dtype but >80% of values are actually numeric
                           (dirty columns — need cleaning before type conversion)
    - true_numeric_cols  : already numeric dtype (int, float)

    This prevents misclassifying dirty numeric columns as categorical
    just because pandas read them as object due to a few dirty values.

    Returns:
        (categorical_cols, numeric_like_cols, true_numeric_cols)
    """
    true_numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols  = []
    numeric_like_cols = []

    for col in df.select_dtypes(include='object').columns:
        converted        = pd.to_numeric(df[col], errors='coerce')
        non_null_orig    = df[col].notna().sum()
        non_null_conv    = converted.notna().sum()

        # If >80% of non-null values convert successfully → numeric-like
        if non_null_orig > 0 and (non_null_conv / non_null_orig) > 0.8:
            numeric_like_cols.append(col)
        else:
            categorical_cols.append(col)

    return categorical_cols, numeric_like_cols, true_numeric_cols


def build_value_counts_summary(df: pd.DataFrame, categorical_cols: list, top_n: int = 20) -> str:
    """
    Builds a compact value_counts summary string for categorical columns only.
    Includes NaN counts via dropna=False.
    Returns a formatted string ready to inject into a prompt.
    """
    summary = ""
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False).head(top_n).to_string()
        summary += f"\n{col}:\n{vc}\n"
    return summary.strip()


def build_null_summary(df: pd.DataFrame) -> str:
    """
    Builds a null count summary for all columns.
    Returns a formatted string ready to inject into a prompt.
    """
    return df.isna().sum().to_string()