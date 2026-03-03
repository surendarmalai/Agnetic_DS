#!/usr/bin/env python
"""
CLI wrapper: runs pytest, reads JSON report, generates markdown review document.

Usage:
    python tests/run_tests.py             # run all tests
    python tests/run_tests.py --agent 1   # run only agent1-marked tests
    python tests/run_tests.py --agent 2   # run only agent2-marked tests
"""
from __future__ import annotations
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def check_dependencies() -> None:
    """
    Startup guard: verify that pytest-json-report is installed.
    [AUDIT M9] Prints clear install instruction and exits(1) if missing.
    """
    try:
        import pytest_jsonreport
    except ImportError:
        print("ERROR: pytest-json-report is not installed.")
        print("Install with: pip install pytest-json-report")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace with attributes:
        agent : Optional[int]  — 1 or 2 if --agent was passed, else None.
    """
    parser = argparse.ArgumentParser(
        description="Run agent tests and generate markdown review document."
    )
    parser.add_argument("--agent", type=int, choices=[1, 2], default=None,
                        help="Filter to run only tests for agent 1 or 2.")
    return parser.parse_args()


def run_pytest(agent_filter) -> tuple:
    """
    Invoke pytest programmatically and write JSON report.

    Parameters
    ----------
    agent_filter : int or None
        If 1 or 2, adds -m "agent1" or -m "agent2" marker filter.
        If None, runs all tests.

    Returns
    -------
    tuple[int, Path]
        (pytest exit code, path to the generated JSON report file)
    """
    report_path = Path("tests/.pytest_report.json")
    args = [
        "tests/agents/",
        "-v",
        "--tb=short",
        "--json-report",
        f"--json-report-file={report_path}",
    ]
    if agent_filter is not None:
        args += ["-m", f"agent{agent_filter}"]
    import pytest
    exit_code = pytest.main(args)
    return exit_code, report_path


def generate_review_document(report_path: Path, command: str) -> Path:
    """
    Read the pytest JSON report and render a structured markdown document.

    Parameters
    ----------
    report_path : Path  — Path to the .pytest_report.json file.
    command     : str   — The CLI command that was run (for the report header).

    Returns
    -------
    Path
        Path to the written markdown file under tests/reports/.
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    tests = data.get("tests", [])

    # Map xfail -> WARNING [AUDIT L14]
    def _normalize_status(t: dict) -> str:
        outcome = t.get("outcome", "")
        if outcome == "xfailed":
            return "WARNING"
        return outcome.upper()

    normalized = []
    for t in tests:
        normalized.append({
            "nodeid"  : t.get("nodeid", ""),
            "name"    : t.get("nodeid", "").split("::")[-1],
            "status"  : _normalize_status(t),
            "outcome" : t.get("outcome", ""),
            "duration": t.get("duration", 0),
            "call"    : t.get("call", {}),
        })

    passed  = sum(1 for t in normalized if t["status"] == "PASSED")
    failed  = sum(1 for t in normalized if t["status"] == "FAILED")
    warning = sum(1 for t in normalized if t["status"] == "WARNING")
    total   = len(normalized)

    counts = {"PASSED": passed, "FAILED": failed, "WARNING": warning, "TOTAL": total}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit_hash = "unknown"

    lines = []
    lines.append(f"# Agent Test Review Report")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Command:** `{command}`")
    lines.append(f"**Git commit:** `{commit_hash}`")
    lines.append(f"")
    lines.append(_render_summary_table(counts))
    lines.append(f"")

    # Group tests by agent
    agent1_tests = [t for t in normalized if "test_agent_1" in t["nodeid"] or "tc1_" in t["name"] or "tc2_" in t["name"] or "tc3_" in t["name"] or "tc4_" in t["name"] or "tc5_" in t["name"] or "tc6_" in t["name"] or "tc7_" in t["name"] or "tc8_" in t["name"] or "tc9_" in t["name"]]
    agent2_tests = [t for t in normalized if "test_agent_2" in t["nodeid"]]
    executor_tests = [t for t in normalized if "test_executor" in t["nodeid"]]

    if agent1_tests:
        lines.append(_render_agent_table(agent1_tests, "Agent 1 — Field Renamer (TC1–TC9)"))
        lines.append("")
    if agent2_tests:
        lines.append(_render_agent_table(agent2_tests, "Agent 2 — Field Cleaner (TC10–TC17)"))
        lines.append("")
    if executor_tests:
        lines.append(_render_agent_table(executor_tests, "Executor (TC18–TC23)"))
        lines.append("")

    # Failure blocks
    failed_tests = [t for t in normalized if t["status"] == "FAILED"]
    if failed_tests:
        lines.append("## Failure Details")
        lines.append("")
        for t in failed_tests:
            lines.append(_render_failure_block(t))
            lines.append("")

    # Warning blocks (xfail)
    warning_tests = [t for t in normalized if t["status"] == "WARNING"]
    if warning_tests:
        lines.append("## Warning Details (Expected Failures)")
        lines.append("")
        for t in warning_tests:
            lines.append(f"### WARNING: {t['name']}")
            lines.append(f"- **Status:** WARNING (xfail — documented gap)")
            lines.append(f"- **Node:** `{t['nodeid']}`")
            lines.append("")

    output_dir = Path("tests/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"review_{timestamp}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Review written to: {out_path}")
    return out_path


def _render_summary_table(counts: dict) -> str:
    """
    Render the markdown summary table.

    Parameters
    ----------
    counts : dict with keys "PASSED", "FAILED", "WARNING", "TOTAL"

    Returns
    -------
    str — Markdown table string.
    """
    lines = [
        "## Summary",
        "",
        "| Status  | Count |",
        "|---------|-------|",
        f"| PASSED  | {counts.get('PASSED', 0)} |",
        f"| FAILED  | {counts.get('FAILED', 0)} |",
        f"| WARNING | {counts.get('WARNING', 0)} |",
        f"| **TOTAL** | **{counts.get('TOTAL', 0)}** |",
    ]
    return "\n".join(lines)


def _render_agent_table(tests: list, agent_label: str) -> str:
    """
    Render per-agent result table.

    Parameters
    ----------
    tests       : list of dicts, each with keys: name, status, nodeid
    agent_label : str — e.g. "Agent 1 — Field Renamer"

    Returns
    -------
    str — Markdown section with H2 header + table.
    """
    lines = [
        f"## {agent_label}",
        "",
        "| Test Name | Status | Duration (s) |",
        "|-----------|--------|-------------|",
    ]
    for t in tests:
        status = t["status"]
        name = t["name"]
        duration = f"{t.get('duration', 0):.3f}"
        lines.append(f"| `{name}` | {status} | {duration} |")
    return "\n".join(lines)


def _render_failure_block(test: dict) -> str:
    """
    Render the full 8-field failure block for a FAILED test.

    Parameters
    ----------
    test : dict with keys: name, status, nodeid, call

    Returns
    -------
    str — Markdown block (H3 + fields).
    """
    call = test.get("call", {})
    longrepr = call.get("longrepr", "No details available.")
    lines = [
        f"### FAILED: {test['name']}",
        f"",
        f"- **Status:** FAILED",
        f"- **Node:** `{test['nodeid']}`",
        f"- **Duration:** {test.get('duration', 0):.3f}s",
        f"",
        f"**Failure output:**",
        f"```",
        str(longrepr),
        f"```",
    ]
    return "\n".join(lines)


def main() -> None:
    """
    Entry point. Orchestrates: check_dependencies -> parse_args -> run_pytest ->
    generate_review_document.
    """
    check_dependencies()
    args = parse_args()
    command = "python tests/run_tests.py" + (f" --agent {args.agent}" if args.agent else "")
    exit_code, report_path = run_pytest(args.agent)
    if report_path.exists():
        out = generate_review_document(report_path, command)
        print(f"Review written to: {out}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
