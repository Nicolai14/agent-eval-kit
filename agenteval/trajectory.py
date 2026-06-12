"""Trajectory data structures and JSONL run persistence.

A Trajectory is the full record of one scenario run: every user message,
every assistant turn (with tool calls), and every tool result. Checks and
the LLM judge operate on this structure, and the recorder persists it as
JSONL under ``.agenteval/runs/<run_id>/results.jsonl`` so runs can be
diffed later.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A single tool invocation requested by the agent."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": self.args, "id": self.id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(name=data["name"], args=data.get("args", {}), id=data.get("id"))


@dataclass
class Turn:
    """One turn in a conversation.

    role is one of "user", "assistant", "tool" (tool results), or "system".
    Token and latency fields are optional and filled in by adapters that
    have access to that data (e.g. AnthropicAdapter).
    """

    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_results:
            data["tool_results"] = self.tool_results
        for key in ("input_tokens", "output_tokens", "latency_ms"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls", [])],
            tool_results=data.get("tool_results", []),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            latency_ms=data.get("latency_ms"),
        )


@dataclass
class Trajectory:
    """An ordered list of turns plus free-form metadata."""

    turns: List[Turn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------

    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def assistant_text(self) -> str:
        """All assistant text, joined with newlines."""
        return "\n".join(t.content for t in self.assistant_turns() if t.content)

    def final_text(self) -> str:
        """Content of the last assistant turn that has text."""
        for turn in reversed(self.turns):
            if turn.role == "assistant" and turn.content:
                return turn.content
        return ""

    def tool_calls(self) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    def to_messages(self) -> List[Dict[str, Any]]:
        """Plain dict view of the conversation, suitable for passing to agents."""
        messages = []
        for turn in self.turns:
            msg: Dict[str, Any] = {"role": turn.role, "content": turn.content}
            if turn.tool_calls:
                msg["tool_calls"] = [c.to_dict() for c in turn.tool_calls]
            if turn.tool_results:
                msg["tool_results"] = turn.tool_results
            messages.append(msg)
        return messages

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        return cls(
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
            metadata=data.get("metadata", {}),
        )


class RunRecorder:
    """Append-only JSONL recorder. One directory per run, one line per scenario.

    Layout::

        .agenteval/runs/<run_id>/results.jsonl

    where run_id is a timestamp plus a short random suffix. Each line is the
    dict produced by ScenarioResult.to_dict() and contains the full trajectory,
    so two run directories can be diffed offline.
    """

    def __init__(self, root: str = os.path.join(".agenteval", "runs"),
                 run_id: Optional[str] = None) -> None:
        self.root = root
        self.run_id = run_id or "{}-{}".format(
            time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:6]
        )
        self.run_dir = os.path.join(self.root, self.run_id)
        self.path = os.path.join(self.run_dir, "results.jsonl")

    def record(self, result: Dict[str, Any]) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        return self.path
