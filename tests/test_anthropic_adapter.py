"""Offline test of the AnthropicAdapter tool-use loop via an injected fake client."""

from types import SimpleNamespace

from agenteval import AnthropicAdapter


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(name, args, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id)


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # snapshot: the adapter mutates the messages list after the call
        snapshot = dict(kwargs)
        snapshot["messages"] = [dict(m) for m in kwargs["messages"]]
        self.calls.append(snapshot)
        return self._responses.pop(0)


def test_tool_use_loop_executes_tools_and_records_turns():
    first = SimpleNamespace(
        content=[_text_block("Let me check."),
                 _tool_block("get_weather", {"city": "Berlin"}, "toolu_1")],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    second = SimpleNamespace(
        content=[_text_block("It's 12°C in Berlin.")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=20, output_tokens=8),
    )
    client = FakeAnthropicClient([first, second])
    executed = []

    def executor(name, args):
        executed.append((name, args))
        return "12°C"

    adapter = AnthropicAdapter(model="claude-opus-4-8", system="be brief",
                               tool_executor=executor, client=client)
    trajectory = adapter.run(
        [{"role": "user", "content": "Weather in Berlin?"}],
        tools=[{"name": "get_weather", "description": "d",
                "input_schema": {"type": "object"}}],
    )

    assert executed == [("get_weather", {"city": "Berlin"})]
    assert trajectory.final_text() == "It's 12°C in Berlin."
    assert [c.name for c in trajectory.tool_calls()] == ["get_weather"]
    # roles: assistant (tool call), tool (result), assistant (final)
    assert [t.role for t in trajectory.turns] == ["assistant", "tool", "assistant"]
    assert trajectory.turns[0].input_tokens == 10

    # second API call must contain the tool_result block
    follow_up = client.calls[1]["messages"]
    assert follow_up[-1]["content"][0]["type"] == "tool_result"
    assert follow_up[-1]["content"][0]["tool_use_id"] == "toolu_1"
    assert client.calls[0]["system"] == "be brief"


def test_loop_terminates_at_max_iterations():
    looping = SimpleNamespace(
        content=[_tool_block("search", {"q": "x"}, "toolu_n")],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = FakeAnthropicClient([looping] * 3)
    adapter = AnthropicAdapter(max_iterations=3, client=client)
    trajectory = adapter.run([{"role": "user", "content": "go"}],
                             tools=[{"name": "search"}])
    assert len(client.calls) == 3
    assert len(trajectory.tool_calls()) == 3
    # without an executor, placeholder results keep the loop moving
    assert "no tool_executor" in trajectory.turns[1].tool_results[0]["content"]
