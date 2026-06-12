"""pytest plugin: collect YAML scenarios as parametrized tests.

Usage:

    pytest --agenteval-scenarios=./scenarios --agenteval-agent=myapp.agent:create

Every *.yaml file inside the scenario directory becomes one test item. The
scenario directory must be within pytest's collection roots (run pytest from
the project root, or pass the directory as an argument:
``pytest scenarios/ --agenteval-scenarios=scenarios``).

Programmatic alternative (no plugin options needed):

    import pytest
    from agenteval import MockAdapter, load_scenarios, run_scenario

    @pytest.mark.parametrize("scenario", load_scenarios("scenarios"),
                             ids=lambda s: s.name)
    def test_agent(scenario):
        result = run_scenario(MockAdapter(), scenario)
        assert result.passed, result.summary()
"""

from pathlib import Path

import pytest

from .adapters import load_agent
from .runner import ScenarioResult, run_scenario
from .scenario import load_scenario
from .trajectory import RunRecorder


def pytest_addoption(parser):
    group = parser.getgroup("agenteval")
    group.addoption("--agenteval-scenarios", action="store", default=None,
                    help="directory of YAML agent-eval scenarios to collect as tests")
    group.addoption("--agenteval-agent", action="store", default="mock",
                    help="agent spec 'module:factory' or 'mock' (default)")
    group.addoption("--agenteval-no-record", action="store_true", default=False,
                    help="do not persist trajectories under .agenteval/runs/")


def _scenario_dir(config):
    opt = config.getoption("--agenteval-scenarios")
    return Path(opt).resolve() if opt else None


def _get_agent(config):
    if not hasattr(config, "_agenteval_agent"):
        config._agenteval_agent = load_agent(config.getoption("--agenteval-agent"))
    return config._agenteval_agent


def _get_recorder(config):
    if config.getoption("--agenteval-no-record"):
        return None
    if not hasattr(config, "_agenteval_recorder"):
        config._agenteval_recorder = RunRecorder()
    return config._agenteval_recorder


def pytest_collect_file(file_path, parent):
    scenario_dir = _scenario_dir(parent.config)
    if scenario_dir is None:
        return None
    path = Path(file_path)
    if path.suffix not in (".yaml", ".yml"):
        return None
    try:
        path.resolve().relative_to(scenario_dir)
    except ValueError:
        return None
    return ScenarioFile.from_parent(parent, path=path)


class ScenarioFailure(AssertionError):
    def __init__(self, result: ScenarioResult) -> None:
        super().__init__(result.summary())
        self.result = result


class ScenarioFile(pytest.File):
    def collect(self):
        scenario = load_scenario(str(self.path))
        yield ScenarioItem.from_parent(self, name=scenario.name, scenario=scenario)


class ScenarioItem(pytest.Item):
    def __init__(self, *, scenario, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario

    def runtest(self):
        agent = _get_agent(self.config)
        result = run_scenario(agent, self.scenario)
        recorder = _get_recorder(self.config)
        if recorder is not None:
            recorder.record(result.to_dict())
        if not result.passed:
            raise ScenarioFailure(result)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, ScenarioFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, "agenteval scenario: {}".format(self.name)
