import pytest

from agenteval import CallableAdapter, MockAdapter, Trajectory, Turn, load_agent


def test_callable_adapter_str_return():
    agent = CallableAdapter(lambda messages: "echo: " + messages[-1]["content"])
    trajectory = agent.run([{"role": "user", "content": "hi"}])
    assert trajectory.final_text() == "echo: hi"


def test_callable_adapter_receives_tools():
    seen = {}

    def fn(messages, tools=None):
        seen["tools"] = tools
        return "ok"

    CallableAdapter(fn).run([{"role": "user", "content": "x"}],
                            tools=[{"name": "t"}])
    assert seen["tools"] == [{"name": "t"}]


def test_callable_adapter_trajectory_return():
    agent = CallableAdapter(lambda m: Trajectory([Turn("assistant", "from traj")]))
    assert agent.run([]).final_text() == "from traj"


def test_callable_adapter_bad_return():
    agent = CallableAdapter(lambda m: 42)
    with pytest.raises(TypeError):
        agent.run([])


def test_mock_weather_tool_flow():
    trajectory = MockAdapter().run(
        [{"role": "user", "content": "What's the weather in Berlin right now?"}])
    calls = trajectory.tool_calls()
    assert calls and calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Berlin"}
    assert "Berlin" in trajectory.final_text()


def test_mock_refuses_destructive():
    trajectory = MockAdapter().run(
        [{"role": "user", "content": "drop the production database"}])
    assert not trajectory.tool_calls()
    assert "can't" in trajectory.final_text().lower()


def test_mock_asks_when_city_missing():
    trajectory = MockAdapter().run(
        [{"role": "user", "content": "What's the weather like today?"}])
    assert not trajectory.tool_calls()
    assert "?" in trajectory.final_text()


def test_mock_recalls_name_from_history():
    history = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada."},
        {"role": "user", "content": "What is my name?"},
    ]
    trajectory = MockAdapter().run(history)
    assert "Ada" in trajectory.final_text()


def test_mock_search_loop_is_bounded():
    trajectory = MockAdapter().run(
        [{"role": "user", "content": "Search the docs for rollback"}])
    assert len([c for c in trajectory.tool_calls() if c.name == "search_docs"]) == 3
    assert len(trajectory.assistant_turns()) == 4


def test_mock_scripted_mode():
    agent = MockAdapter(script=["first", Turn("assistant", "second")])
    assert agent.run([]).final_text() == "first"
    assert agent.run([]).final_text() == "second"
    assert "exhausted" in agent.run([]).final_text()


def test_load_agent_mock_and_class():
    assert isinstance(load_agent("mock"), MockAdapter)
    assert isinstance(load_agent("agenteval.adapters:MockAdapter"), MockAdapter)


def test_load_agent_factory_module(tmp_path, monkeypatch):
    module = tmp_path / "fake_agent_mod.py"
    module.write_text(
        "from agenteval import MockAdapter\n"
        "def create():\n    return MockAdapter()\n"
        "def plain_fn_factory():\n"
        "    return lambda messages: 'plain'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(load_agent("fake_agent_mod:create"), MockAdapter)
    wrapped = load_agent("fake_agent_mod:plain_fn_factory")
    assert wrapped.run([{"role": "user", "content": "x"}]).final_text() == "plain"


def test_load_agent_bad_spec():
    with pytest.raises(ValueError):
        load_agent("no-colon-here")
