"""Plugin smoke tests via pytester (runs an isolated in-process pytest)."""

PASSING_SCENARIO = """\
name: greets
input: hello there
expected:
  - contains: help
"""

FAILING_SCENARIO = """\
name: wants_zebra
input: hello there
expected:
  - contains: zebra
"""


def test_plugin_collects_and_passes(pytester):
    scenario_dir = pytester.mkdir("scenarios")
    (scenario_dir / "greets.yaml").write_text(PASSING_SCENARIO, encoding="utf-8")
    result = pytester.runpytest(
        "-p", "agenteval.pytest_plugin",
        "--agenteval-scenarios=scenarios",
        "--agenteval-no-record",
        "scenarios",
    )
    result.assert_outcomes(passed=1)


def test_plugin_reports_failures_with_check_details(pytester):
    scenario_dir = pytester.mkdir("scenarios")
    (scenario_dir / "wants_zebra.yaml").write_text(FAILING_SCENARIO, encoding="utf-8")
    result = pytester.runpytest(
        "-p", "agenteval.pytest_plugin",
        "--agenteval-scenarios=scenarios",
        "--agenteval-no-record",
        "scenarios",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*zebra*"])


def test_plugin_inert_without_option(pytester):
    scenario_dir = pytester.mkdir("scenarios")
    (scenario_dir / "greets.yaml").write_text(PASSING_SCENARIO, encoding="utf-8")
    result = pytester.runpytest("-p", "agenteval.pytest_plugin", "scenarios")
    result.assert_outcomes()  # nothing collected, nothing failed
