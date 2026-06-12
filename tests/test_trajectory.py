import json
import os

from agenteval import RunRecorder, ToolCall, Trajectory, Turn


def _sample_trajectory():
    return Trajectory(
        turns=[
            Turn("user", "What's the weather in Berlin?"),
            Turn("assistant", "", tool_calls=[
                ToolCall("get_weather", {"city": "Berlin"}, id="call_1")]),
            Turn("tool", tool_results=[{
                "tool": "get_weather", "tool_use_id": "call_1", "content": "12°C"}]),
            Turn("assistant", "It's 12°C in Berlin.",
                 input_tokens=100, output_tokens=20, latency_ms=421.5),
        ],
        metadata={"scenario": "demo"},
    )


def test_roundtrip():
    trajectory = _sample_trajectory()
    restored = Trajectory.from_dict(trajectory.to_dict())
    assert restored == trajectory


def test_accessors():
    trajectory = _sample_trajectory()
    assert trajectory.final_text() == "It's 12°C in Berlin."
    assert "Berlin" in trajectory.assistant_text()
    assert len(trajectory.assistant_turns()) == 2
    assert [c.name for c in trajectory.tool_calls()] == ["get_weather"]


def test_to_messages_includes_tool_info():
    messages = _sample_trajectory().to_messages()
    assert messages[1]["tool_calls"][0]["name"] == "get_weather"
    assert messages[2]["tool_results"][0]["tool"] == "get_weather"


def test_recorder_writes_jsonl(tmp_path):
    recorder = RunRecorder(root=str(tmp_path))
    path = recorder.record({"scenario": "a", "passed": True})
    recorder.record({"scenario": "b", "passed": False})
    assert os.path.exists(path)
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scenario"] == "a"
    assert recorder.run_id in path
