"""Run scenarios against an agent and collect check results."""

import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapters import Agent
from .checks import CheckResult, run_checks
from .scenario import Scenario
from .trajectory import RunRecorder, Trajectory, Turn


@dataclass
class ScenarioResult:
    scenario: Scenario
    trajectory: Trajectory
    checks: List[CheckResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        """True when no check failed (skipped checks don't fail a scenario)."""
        return all(c.status != "fail" for c in self.checks)

    def summary(self) -> str:
        lines = ["scenario '{}': {}".format(
            self.scenario.name, "PASS" if self.passed else "FAIL")]
        for check in self.checks:
            line = "  [{}] {}".format(check.status.upper(), check.name)
            if check.detail and check.status != "pass":
                line += " — {}".format(check.detail)
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "description": self.scenario.description,
            "passed": self.passed,
            "duration_s": round(self.duration_s, 3),
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checks": [c.to_dict() for c in self.checks],
            "trajectory": self.trajectory.to_dict(),
        }


def run_scenario(agent: Agent, scenario: Scenario,
                 judge_model: Optional[str] = None) -> ScenarioResult:
    """Run one scenario.

    Multi-turn semantics: each `user` message in the scenario input is sent in
    order; after each one the agent is invoked with the full conversation so
    far (unless the scenario scripts the following assistant turn itself).
    The returned trajectory contains the complete conversation.
    """
    started = time.time()
    full = Trajectory(metadata={"scenario": scenario.name})

    input_messages = scenario.input
    for index, message in enumerate(input_messages):
        full.turns.append(Turn(role=message["role"], content=message["content"]))
        if message["role"] != "user":
            continue  # scripted context turn, no agent invocation
        next_is_scripted_assistant = (
            index + 1 < len(input_messages)
            and input_messages[index + 1].get("role") == "assistant"
        )
        if next_is_scripted_assistant:
            continue
        out = agent.run(full.to_messages(), tools=scenario.tools)
        full.turns.extend(out.turns)
        if out.metadata:
            full.metadata.update(out.metadata)

    checks = run_checks(scenario.checks, full,
                        base_dir=scenario.base_dir, judge_model=judge_model)
    return ScenarioResult(
        scenario=scenario,
        trajectory=full,
        checks=checks,
        duration_s=time.time() - started,
    )


def run_scenarios(
    agent: Agent,
    scenarios: List[Scenario],
    recorder: Optional[RunRecorder] = None,
    judge_model: Optional[str] = None,
) -> List[ScenarioResult]:
    results = []
    for scenario in scenarios:
        result = run_scenario(agent, scenario, judge_model=judge_model)
        if recorder is not None:
            recorder.record(result.to_dict())
        results.append(result)
    return results
