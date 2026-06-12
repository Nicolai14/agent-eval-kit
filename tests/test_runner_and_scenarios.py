"""End-to-end: the shipped example scenarios must pass offline against MockAdapter."""

import os

import pytest

from agenteval import (
    MockAdapter,
    RunRecorder,
    load_scenarios,
    run_scenario,
    run_scenarios,
)

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "scenarios")


def test_load_example_scenarios():
    scenarios = load_scenarios(SCENARIO_DIR)
    assert len(scenarios) == 5
    names = {s.name for s in scenarios}
    assert "tool_use_happy_path" in names
    assert "refused_instruction" in names


@pytest.mark.parametrize("scenario", load_scenarios(SCENARIO_DIR),
                         ids=lambda s: s.name)
def test_example_scenarios_pass_against_mock(scenario):
    result = run_scenario(MockAdapter(), scenario)
    assert result.passed, result.summary()


def test_judge_check_is_skipped_offline():
    scenarios = {s.name: s for s in load_scenarios(SCENARIO_DIR)}
    result = run_scenario(MockAdapter(), scenarios["tool_use_happy_path"])
    statuses = {c.name: c.status for c in result.checks}
    assert statuses["judge: helpfulness"] == "skip"


def test_multi_turn_trajectory_contains_all_turns():
    scenarios = {s.name: s for s in load_scenarios(SCENARIO_DIR)}
    result = run_scenario(MockAdapter(), scenarios["multi_turn_context"])
    user_turns = [t for t in result.trajectory.turns if t.role == "user"]
    assistant_turns = result.trajectory.assistant_turns()
    assert len(user_turns) == 2
    assert len(assistant_turns) == 2
    assert "Your name is Ada" in result.trajectory.final_text()


def test_run_scenarios_records(tmp_path):
    scenarios = load_scenarios(SCENARIO_DIR)
    recorder = RunRecorder(root=str(tmp_path))
    results = run_scenarios(MockAdapter(), scenarios, recorder=recorder)
    assert all(r.passed for r in results)
    lines = open(recorder.path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == len(scenarios)


def test_result_to_dict_shape():
    scenarios = load_scenarios(SCENARIO_DIR)
    result = run_scenario(MockAdapter(), scenarios[0])
    data = result.to_dict()
    assert set(data) >= {"scenario", "passed", "checks", "trajectory", "recorded_at"}
    assert isinstance(data["trajectory"]["turns"], list)


def test_failing_scenario_summary(tmp_path):
    scenario_file = tmp_path / "fails.yaml"
    scenario_file.write_text(
        "name: fails\ninput: hello\nexpected:\n  - contains: zebra\n",
        encoding="utf-8",
    )
    scenarios = load_scenarios(str(tmp_path))
    result = run_scenario(MockAdapter(), scenarios[0])
    assert not result.passed
    assert "FAIL" in result.summary()
    assert "zebra" in result.summary()
