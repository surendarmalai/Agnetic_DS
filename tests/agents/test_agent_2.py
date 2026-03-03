"""
Agent 2 — Field Cleaner: 8 test cases.

All tests use pytest fixtures and the MockLLM test double.
Fixture responses are loaded from tests/fixtures/agent_2/scenarios.json.
Agent under test: make_field_cleaner_agent(config) from agent_2_field_cleaner.py.
"""
import json
import pytest
from pathlib import Path
from tests.mock_llm import MockLLM
from source_code.config.llm_config import PipelineConfig
from source_code.agents.agent_2_field_cleaner import make_field_cleaner_agent

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "agent_2" / "scenarios.json"
SCENARIOS = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

# State after rename. Columns are standardised (no alias prefixes).
BASE_STATE = {
    "file_path"           : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "output_path"         : str(Path(__file__).parent.parent / "fixtures" / "sample_telecom.csv"),
    "target_column"       : "ChurnFlag",
    "metadata_summary"    : "DATASET PROFILE:\n1. COLUMNS & TYPES:\n{...}\n2. SAMPLE ROW:\n{...}",
    "categorical_cols"    : ["Gender", "RegionCode", "PlanType"],
    "numeric_like_cols"   : ["RevenueLast30d"],
    "true_numeric_cols"   : ["ChurnFlag", "TenureMonths", "MouOutgoingLast30d",
                             "DataUsageLast30dKb", "ContractEndDate"],
    "value_counts_summary": "Gender:\nMale    20\nM       10\nFemale  15\nF       5\n",
    "null_summary"        : "Gender    3\nRevenueLast30d    5",
    "special_rules"       : "None",
    "iteration_count"     : 0,
    "sql_query"           : "",   # Agent 2 does not use sql_query [architecture §12]
}


def _make_agent(scenario_key: str):
    """Helper: build agent from fixture scenario."""
    mock = MockLLM(content=SCENARIOS[scenario_key]["llm_response"])
    cfg = PipelineConfig.with_mock(mock)
    return make_field_cleaner_agent(cfg), mock


@pytest.mark.agent2
def test_tc10_happy_path():
    """
    TC10 — Happy Path.
    Valid two-block response with cleaning code and empty flagged_columns.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["flagged_columns"] == []
    - result["cleaning_code"] contains "df['Gender']"
    - result["output_path"] == "output_agent2_cleaned.csv"
    """
    agent_fn, mock = _make_agent("happy_path")
    result = agent_fn(BASE_STATE)

    assert result["cleaning_code"] != ""
    assert result["flagged_columns"] == []
    assert "df['Gender']" in result["cleaning_code"]
    assert result["output_path"] == "output_agent2_cleaned.csv"


@pytest.mark.agent2
def test_tc11_metadata_context_injection():
    """
    TC11 — Metadata Context Injection.
    Verifies that value_counts_summary and null_summary appear in the prompt.
    [Architecture §12: Agent 2 does NOT receive sql_query]

    Assertions (from expected_assertions):
    - "Gender" in str(mock.last_prompt)               (value_counts injected)
    - "null" in str(mock.last_prompt).lower()         (null_summary injected)
    - "SELECT" not in str(mock.last_prompt)           (sql_query NOT injected)
    """
    agent_fn, mock = _make_agent("metadata_context_injection")
    result = agent_fn(BASE_STATE)

    assert "Gender" in str(mock.last_prompt)
    assert "null" in str(mock.last_prompt).lower()
    assert "SELECT" not in str(mock.last_prompt)


@pytest.mark.agent2
def test_tc12_missing_second_block():
    """
    TC12 — Missing Second Code Block.
    LLM returns only cleaning code, no flagged_columns block.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""
    - result["flagged_columns"] == []   (defaults to empty, not None)
    """
    agent_fn, mock = _make_agent("missing_second_block")
    result = agent_fn(BASE_STATE)

    assert result["cleaning_code"] != ""
    assert result["flagged_columns"] == []


@pytest.mark.agent2
def test_tc13_no_code_blocks():
    """
    TC13 — No Code Blocks.
    LLM returns plain prose.

    Assertions (from expected_assertions):
    - result["cleaning_code"] == ""
    - result["flagged_columns"] == []
    """
    agent_fn, mock = _make_agent("no_code_blocks")
    result = agent_fn(BASE_STATE)

    assert result["cleaning_code"] == ""
    assert result["flagged_columns"] == []


@pytest.mark.agent2
def test_tc14_flagged_columns_missing_keys():
    """
    TC14 — flagged_columns Entry Missing Keys.
    LLM returns flagged_columns with only "column", no "reason".

    Assertions (from expected_assertions):
    - result["flagged_columns"] is a non-empty list
    - result["flagged_columns"][0].get("reason") is None   (no crash)
    """
    agent_fn, mock = _make_agent("flagged_columns_missing_keys")
    result = agent_fn(BASE_STATE)

    assert len(result["flagged_columns"]) > 0
    assert result["flagged_columns"][0].get("reason") is None


@pytest.mark.agent2
@pytest.mark.xfail(strict=False, reason="documented gap: agent does not detect null introduction")
def test_tc15_cleaning_code_introduces_new_nulls():
    """
    TC15 — Cleaning Code Introduces New Nulls.
    [AUDIT L14] xfail(strict=False) -> WARNING in report.
    The cleaning code replaces valid values with None, increasing null count.
    The executor should detect this via new_nulls check.

    Assertions (from expected_assertions):
    - result["error_log"] is not None
    - "WARNING" in result["error_log"]   (executor's null detection fires)
    This test requires executor integration — call executor after agent.
    """
    from source_code.agents.executor import cleaning_executor_agent

    agent_fn, mock = _make_agent("cleaning_code_introduces_new_nulls")
    agent_result = agent_fn(BASE_STATE)

    exec_state = {**BASE_STATE, **agent_result}
    exec_result = cleaning_executor_agent(exec_state)

    # This assertion WILL FAIL (XFAIL): executor currently warns but error_log
    # may not be checked. Demonstrates the gap in null detection feedback.
    assert exec_result.get("error_log") is not None
    assert "WARNING" in exec_result.get("error_log", "")


@pytest.mark.agent2
@pytest.mark.xfail(strict=False, reason="documented gap: agent does not detect column drops")
def test_tc16_cleaning_code_drops_columns():
    """
    TC16 — Cleaning Code Drops Columns.
    [AUDIT L14] xfail(strict=False) -> WARNING in report.
    The cleaning code contains df.drop().

    Assertions (from expected_assertions):
    - result["error_log"] is not None
    - "WARNING" in result["error_log"]   (executor's column-drop detection fires)
    This test requires executor integration — call executor after agent.
    """
    from source_code.agents.executor import cleaning_executor_agent

    agent_fn, mock = _make_agent("cleaning_code_drops_columns")
    agent_result = agent_fn(BASE_STATE)

    exec_state = {**BASE_STATE, **agent_result}
    exec_result = cleaning_executor_agent(exec_state)

    # This assertion WILL FAIL (XFAIL): executor currently warns but agent
    # does not preemptively detect the drop.
    assert exec_result.get("error_log") is not None
    assert "WARNING" in exec_result.get("error_log", "")


@pytest.mark.agent2
def test_tc17_malformed_python():
    """
    TC17 — Malformed Python.
    LLM returns a syntax error in cleaning code block.

    Assertions (from expected_assertions):
    - result["cleaning_code"] != ""   (raw code stored even if malformed)
    - No unhandled exception from agent
    - When executor runs this code: result["error_log"] contains "SyntaxError"
    """
    from source_code.agents.executor import cleaning_executor_agent

    agent_fn, mock = _make_agent("malformed_python")
    agent_result = agent_fn(BASE_STATE)

    assert agent_result["cleaning_code"] != ""

    exec_state = {**BASE_STATE, **agent_result}
    exec_result = cleaning_executor_agent(exec_state)

    assert exec_result.get("error_log") is not None
    assert "SyntaxError" in exec_result.get("error_log", "")
