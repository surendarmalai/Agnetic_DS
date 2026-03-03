"""
Agent 1 — Field Renamer: 9 test cases.

All tests use pytest fixtures and the MockLLM test double.
Fixture responses are loaded from tests/fixtures/agent_1/scenarios.json.
Agent under test: make_field_renamer_agent(config) from agent_1_field_renamer.py.
"""
import json
import pytest
from pathlib import Path
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_1_field_renamer import make_field_renamer_agent

# Load all scenarios once at module level.
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "agent_1" / "scenarios.json"
SCENARIOS = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

# Shared AgentState dict used by all Agent 1 tests.
# Columns match sample_telecom.csv with alias prefixes (a.msisdn etc.)
BASE_STATE = {
    "file_path"       : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "target_column"   : "ChurnFlag",
    "sql_query"       : "SELECT a.msisdn, a.rev_30d, a.mou_out_30d FROM churn_table a",
    "df_columns"      : ["a.msisdn", "gender", "rev_30d", "mou_out_30d",
                         "data_vol_30d", "churn_flag", "tenure_months",
                         "region_cd", "plan_type", "contract_end_dt"],
    "metadata_summary": "DATASET PROFILE:\n1. COLUMNS & TYPES:\n{...}\n2. SAMPLE ROW:\n{...}",
    "special_rules"   : "None",
    "iteration_count" : 0,
}


def _make_agent(scenario_key: str):
    """Helper: build agent from fixture scenario key."""
    mock = MockLLM(content=SCENARIOS[scenario_key]["llm_response"])
    cfg = PipelineConfig.with_mock(mock)
    return make_field_renamer_agent(cfg), mock


@pytest.mark.agent1
def test_tc1_happy_path():
    """
    TC1 — Happy Path.
    LLM returns a valid two-block response. All 10 columns covered.

    Assertions (from expected_assertions):
    - result["column_map"] has exactly 10 entries
    - result["ambiguous_fields"] == []
    - result["cleaning_code"] contains "df.rename"
    - mock.last_prompt is not None (prompt was sent)
    """
    agent_fn, mock = _make_agent("happy_path")
    result = agent_fn(BASE_STATE)

    assert len(result["column_map"]) == 10
    assert result["ambiguous_fields"] == []
    assert "df.rename" in result["cleaning_code"]
    assert mock.last_prompt is not None


@pytest.mark.agent1
def test_tc2_sql_context_alias_resolution():
    """
    TC2 — SQL Query Context.
    LLM correctly resolves alias columns using the injected SQL query.

    Assertions (from expected_assertions):
    - result["column_map"].get("rev_30d") == "RevenueLast30d"
    - "SELECT" in str(mock.last_prompt)   (SQL was injected into prompt)
    - result["ambiguous_fields"] == []
    """
    agent_fn, mock = _make_agent("sql_context_alias_resolution")
    result = agent_fn(BASE_STATE)

    assert result["column_map"].get("rev_30d") == "RevenueLast30d"
    assert "SELECT" in str(mock.last_prompt)
    assert result["ambiguous_fields"] == []


@pytest.mark.agent1
def test_tc3_missing_second_block():
    """
    TC3 — Missing Second Code Block.
    LLM returns only one code block (no ambiguous_fields block).

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["ambiguous_fields"] == []   (defaults to empty, not None)
    - result["column_map"] is not None
    """
    agent_fn, mock = _make_agent("missing_second_block")
    result = agent_fn(BASE_STATE)

    assert result["cleaning_code"] != ""
    assert result["ambiguous_fields"] == []
    assert result["column_map"] is not None


@pytest.mark.agent1
def test_tc4_no_code_blocks():
    """
    TC4 — No Code Blocks.
    LLM returns plain prose with no fenced code blocks.

    Assertions (from expected_assertions):
    - result["cleaning_code"] == ""
    - result["ambiguous_fields"] == []
    - result["column_map"] == {}
    """
    agent_fn, mock = _make_agent("no_code_blocks")
    result = agent_fn(BASE_STATE)

    assert result["cleaning_code"] == ""
    assert result["ambiguous_fields"] == []
    assert result["column_map"] == {}


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: duplicate rename_map values not detected")
def test_tc5_duplicate_values_in_rename_map():
    """
    TC5 — Duplicate Values in rename_map.
    [AUDIT L14] marked xfail(strict=False) -> maps to WARNING in report.
    Two source columns map to the same clean name. Known gap: agent does not detect.

    Assertions (from expected_assertions):
    - result["column_map"] has entries  (non-empty)
    - No exception raised during agent execution
    - assert len(set(result["column_map"].values())) == len(result["column_map"].values())
      (this assertion WILL FAIL for the duplicate fixture, producing XFAIL)
    """
    agent_fn, mock = _make_agent("duplicate_values_in_rename_map")
    result = agent_fn(BASE_STATE)

    assert len(result["column_map"]) > 0
    # This assertion will fail (XFAIL): duplicate values are present
    assert len(set(result["column_map"].values())) == len(list(result["column_map"].values()))


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: partial rename_map silently accepted")
def test_tc6_missing_columns_from_rename_map():
    """
    TC6 — Missing Columns from rename_map.
    [AUDIT L14] xfail(strict=False) -> WARNING in report.
    Only 2 of 10 columns returned in rename_map.

    Assertions (from expected_assertions):
    - result["column_map"] has exactly 2 entries
    - assert len(result["column_map"]) == len(BASE_STATE["df_columns"])
      (this assertion WILL FAIL, producing XFAIL)
    """
    agent_fn, mock = _make_agent("missing_columns_from_rename_map")
    result = agent_fn(BASE_STATE)

    assert len(result["column_map"]) == 2
    # This assertion will fail (XFAIL): only 2 columns, not 10
    assert len(result["column_map"]) == len(BASE_STATE["df_columns"])


@pytest.mark.agent1
def test_tc7_ambiguous_fields_missing_keys():
    """
    TC7 — ambiguous_fields Entry Missing Keys.
    LLM returns ambiguous_fields with only "original_column", no "candidates" or "reason".

    Assertions (from expected_assertions):
    - result["ambiguous_fields"] is a non-empty list
    - result["ambiguous_fields"][0].get("candidates") is None   (no crash)
    - result["ambiguous_fields"][0].get("reason") is None       (no crash)
    """
    agent_fn, mock = _make_agent("ambiguous_fields_missing_keys")
    result = agent_fn(BASE_STATE)

    assert len(result["ambiguous_fields"]) > 0
    assert result["ambiguous_fields"][0].get("candidates") is None
    assert result["ambiguous_fields"][0].get("reason") is None


@pytest.mark.agent1
@pytest.mark.xfail(strict=False, reason="documented gap: rename_map keys not validated against df columns")
def test_tc8_keys_dont_match_df_columns():
    """
    TC8 — rename_map Keys Don't Match df Columns.
    [AUDIT L14] xfail(strict=False) -> WARNING in report.
    LLM uses column names not present in the test DataFrame.

    Assertions (from expected_assertions):
    - result["column_map"] is non-empty
    - No exception raised
    - assert all(k in BASE_STATE["df_columns_cleaned"] for k in result["column_map"])
      (this assertion WILL FAIL, producing XFAIL)
    """
    agent_fn, mock = _make_agent("keys_dont_match_df_columns")
    result = agent_fn(BASE_STATE)

    assert len(result["column_map"]) > 0
    # Compute expected cleaned column names (alias-stripped)
    from source_code.utils import preprocess_column_names
    cleaned_cols = list(preprocess_column_names(BASE_STATE["df_columns"]).values())
    # This assertion will fail (XFAIL): keys like "subscriber_id" are not in cleaned_cols
    assert all(k in cleaned_cols for k in result["column_map"])


@pytest.mark.agent1
def test_tc9_malformed_python():
    """
    TC9 — Malformed Python in Code Block.
    LLM returns a syntax error in the rename block (missing closing brace).

    Assertions (from expected_assertions):
    - result["column_map"] == {}   (exec fails, column_map defaults to {})
    - result["cleaning_code"] != ""  (the raw code string is still stored)
    - No unhandled exception propagates from the agent
    """
    agent_fn, mock = _make_agent("malformed_python")
    result = agent_fn(BASE_STATE)

    assert result["column_map"] == {}
    assert result["cleaning_code"] != ""
