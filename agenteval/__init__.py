"""agenteval — pytest-based eval harness for LLM agents.

Declarative YAML scenarios, trajectory recording, optional LLM judge.
Runs offline against a MockAdapter, locally against your own agent, and in CI.
"""

from .adapters import (
    Agent,
    AnthropicAdapter,
    CallableAdapter,
    MockAdapter,
    load_agent,
)
from .checks import CheckResult, run_checks
from .judge import (
    DEFAULT_JUDGE_MODEL,
    JudgeResult,
    JudgeUnavailable,
    Rubric,
    judge_trajectory,
)
from .runner import ScenarioResult, run_scenario, run_scenarios
from .scenario import Scenario, ScenarioError, load_scenario, load_scenarios
from .trajectory import RunRecorder, ToolCall, Trajectory, Turn

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AnthropicAdapter",
    "CallableAdapter",
    "MockAdapter",
    "load_agent",
    "CheckResult",
    "run_checks",
    "DEFAULT_JUDGE_MODEL",
    "JudgeResult",
    "JudgeUnavailable",
    "Rubric",
    "judge_trajectory",
    "ScenarioResult",
    "run_scenario",
    "run_scenarios",
    "Scenario",
    "ScenarioError",
    "load_scenario",
    "load_scenarios",
    "RunRecorder",
    "ToolCall",
    "Trajectory",
    "Turn",
    "__version__",
]
