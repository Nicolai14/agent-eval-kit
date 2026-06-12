import json
from types import SimpleNamespace

import pytest

from agenteval import (
    DEFAULT_JUDGE_MODEL,
    JudgeUnavailable,
    Rubric,
    Trajectory,
    Turn,
    judge_trajectory,
)
from agenteval.judge import render_transcript


class FakeClient:
    """Mimics the minimal anthropic client surface the judge uses."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.last_kwargs = None
        block = SimpleNamespace(type="text", text=response_text)
        self._response = SimpleNamespace(content=[block])
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


@pytest.fixture
def rubric(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text(
        "name: quality\npass_threshold: 0.6\ncriteria:\n"
        "  - name: correct\n    description: answer is correct\n    weight: 3\n"
        "  - name: concise\n    description: answer is concise\n    weight: 1\n",
        encoding="utf-8",
    )
    return Rubric.load(str(path))


def _trajectory():
    return Trajectory([Turn("user", "hi"), Turn("assistant", "hello")])


def test_rubric_load(rubric):
    assert rubric.name == "quality"
    assert rubric.pass_threshold == 0.6
    assert [c.weight for c in rubric.criteria] == [3.0, 1.0]


def test_judge_weighted_score_pass(rubric):
    client = FakeClient(json.dumps(
        {"scores": {"correct": 1.0, "concise": 0.0}, "reasoning": "mostly right"}))
    result = judge_trajectory(_trajectory(), rubric, client=client)
    assert result.score == 0.75  # (3*1 + 1*0) / 4
    assert result.passed is True
    assert result.model == DEFAULT_JUDGE_MODEL
    assert client.last_kwargs["model"] == DEFAULT_JUDGE_MODEL


def test_judge_fail_below_threshold(rubric):
    client = FakeClient('{"scores": {"correct": 0.4, "concise": 0.4}, "reasoning": "weak"}')
    result = judge_trajectory(_trajectory(), rubric, client=client)
    assert result.passed is False


def test_judge_parses_fenced_json_and_clamps(rubric):
    client = FakeClient('```json\n{"scores": {"correct": 7, "concise": -1}}\n```')
    result = judge_trajectory(_trajectory(), rubric, client=client)
    assert result.scores == {"correct": 1.0, "concise": 0.0}


def test_judge_unavailable_without_key(rubric):
    with pytest.raises(JudgeUnavailable):
        judge_trajectory(_trajectory(), rubric)


def test_judge_model_override(rubric, monkeypatch):
    client = FakeClient('{"scores": {"correct": 1, "concise": 1}}')
    judge_trajectory(_trajectory(), rubric, model="claude-haiku-4-5", client=client)
    assert client.last_kwargs["model"] == "claude-haiku-4-5"
    monkeypatch.setenv("AGENTEVAL_JUDGE_MODEL", "env-model")
    judge_trajectory(_trajectory(), rubric, client=client)
    assert client.last_kwargs["model"] == "env-model"


def test_render_transcript_includes_tools():
    from agenteval import ToolCall
    trajectory = Trajectory([
        Turn("user", "weather?"),
        Turn("assistant", "", tool_calls=[ToolCall("get_weather", {"city": "Berlin"})]),
        Turn("tool", tool_results=[{"tool": "get_weather", "content": "12°C"}]),
        Turn("assistant", "12°C in Berlin"),
    ])
    text = render_transcript(trajectory)
    assert "get_weather" in text and "12°C" in text and "user: weather?" in text
