"""Agent adapters.

The only contract agenteval needs is the Agent protocol:

    class Agent(Protocol):
        def run(self, messages, tools=None) -> Trajectory: ...

`messages` is the conversation so far (list of {role, content, ...} dicts,
see Trajectory.to_messages()). The adapter returns a Trajectory containing
only the *new* turns produced for the latest user message — the runner
stitches multi-turn conversations together.

Shipped adapters:
  - CallableAdapter   wrap any Python function
  - MockAdapter       deterministic, offline; powers the example scenarios and tests
  - AnthropicAdapter  Messages API with a tool-use loop (optional `anthropic` extra)
"""

import importlib
import inspect
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

try:  # Protocol is in typing from 3.8 on
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

    def runtime_checkable(cls):  # type: ignore
        return cls

from .trajectory import ToolCall, Trajectory, Turn


@runtime_checkable
class Agent(Protocol):
    """Anything with a run(messages, tools=None) -> Trajectory method."""

    def run(self, messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
        ...  # pragma: no cover


def _as_trajectory(result: Any) -> Trajectory:
    if isinstance(result, Trajectory):
        return result
    if isinstance(result, Turn):
        return Trajectory([result])
    if isinstance(result, str):
        return Trajectory([Turn(role="assistant", content=result)])
    if isinstance(result, list) and all(isinstance(t, Turn) for t in result):
        return Trajectory(list(result))
    raise TypeError(
        "agent function must return a str, Turn, list[Turn] or Trajectory, "
        "got {}".format(type(result).__name__)
    )


class CallableAdapter:
    """Wrap an arbitrary Python function as an Agent.

    The function receives the message list (and `tools=` if its signature
    accepts it) and may return a plain string, a Turn, a list of Turns, or a
    full Trajectory.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        try:
            params = inspect.signature(fn).parameters
            self._accepts_tools = "tools" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):  # builtins / C callables
            self._accepts_tools = False

    def run(self, messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
        if self._accepts_tools:
            result = self.fn(messages, tools=tools)
        else:
            result = self.fn(messages)
        return _as_trajectory(result)


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------

_CITY_RE = re.compile(r"\bin\s+([A-Z][A-Za-z-]+)")
_NAME_RE = re.compile(r"my name is\s+([A-Za-z]+)", re.IGNORECASE)


class MockAdapter:
    """Deterministic offline agent for demos and tests.

    Two modes:

    * scripted: ``MockAdapter(script=["first reply", "second reply"])`` plays
      back one script item per run() call (str, Turn, list[Turn] or Trajectory).
    * heuristic (default): keyword-driven canned behavior that exercises the
      shipped example scenarios — tool use, refusal, multi-turn recall,
      clarifying questions instead of hallucinated tool args, and a bounded
      search loop.
    """

    def __init__(self, script: Optional[List[Any]] = None) -> None:
        self._script = list(script) if script is not None else None
        self._call_count = 0

    def run(self, messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
        self._call_count += 1
        if self._script is not None:
            if not self._script:
                return Trajectory([Turn("assistant", "(mock script exhausted)")])
            return _as_trajectory(self._script.pop(0))

        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        low = last_user.lower()

        # 1. refuse destructive instructions
        if ("delete" in low or "drop" in low or "wipe" in low) and (
                "database" in low or "production" in low):
            return Trajectory([Turn(
                "assistant",
                "I can't do that. Dropping or deleting a production database is a "
                "destructive, irreversible action. If this is really needed, go "
                "through your DBA process with a verified backup in place.",
            )])

        # 2. weather: call the tool when a city is given, ask otherwise
        if "weather" in low:
            match = _CITY_RE.search(last_user)
            if match:
                city = match.group(1)
                call = ToolCall("get_weather", {"city": city}, id="call_1")
                return Trajectory([
                    Turn("assistant", "", tool_calls=[call]),
                    Turn("tool", tool_results=[{
                        "tool": "get_weather", "tool_use_id": "call_1",
                        "content": "12°C, overcast, light wind",
                    }]),
                    Turn("assistant",
                         "It is currently 12°C and overcast in {} with light wind."
                         .format(city)),
                ])
            return Trajectory([Turn(
                "assistant",
                "Which city should I check the weather for? I don't want to guess.",
            )])

        # 3. multi-turn recall: answer name questions from history
        if "what" in low and "name" in low:
            history = " ".join(str(m.get("content", "")) for m in messages)
            match = _NAME_RE.search(history)
            if match:
                return Trajectory([Turn("assistant",
                                        "Your name is {}.".format(match.group(1)))])
            return Trajectory([Turn(
                "assistant", "I don't think you've told me your name yet.")])

        if _NAME_RE.search(last_user):
            name = _NAME_RE.search(last_user).group(1)
            return Trajectory([Turn(
                "assistant", "Nice to meet you, {}. How can I help?".format(name))])

        # 4. bounded search loop (3 tool calls, then an answer)
        if "search" in low or "find" in low:
            turns: List[Turn] = []
            queries = [last_user, last_user + " best practices", "rollback procedure"]
            for index, query in enumerate(queries, start=1):
                call_id = "call_{}".format(index)
                turns.append(Turn("assistant", "", tool_calls=[
                    ToolCall("search_docs", {"query": query}, id=call_id)]))
                turns.append(Turn("tool", tool_results=[{
                    "tool": "search_docs", "tool_use_id": call_id,
                    "content": "result {}: see deployment guide, section 'Rollback'"
                               .format(index),
                }]))
            turns.append(Turn(
                "assistant",
                "Found it: the rollback procedure is in the deployment guide under "
                "'Rollback' — redeploy the previous build, then invalidate caches.",
            ))
            return Trajectory(turns)

        # 5. deterministic fallback
        return Trajectory([Turn(
            "assistant",
            "I can help with that. Could you share a bit more detail about what "
            "you need?",
        )])


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """Run an agent on the Anthropic Messages API with a tool-use loop.

    Requires the optional dependency: ``pip install 'agenteval[anthropic]'``
    and an ANTHROPIC_API_KEY in the environment (or pass `client=`).

    `tool_executor` is a callable ``(name, args) -> str`` that executes tool
    calls; without one, tool calls receive a placeholder result so the loop
    still terminates and the trajectory still records what the model wanted
    to do.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        system: Optional[str] = None,
        max_tokens: int = 16000,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        max_iterations: int = 10,
        client: Any = None,
    ) -> None:
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        if client is None:
            try:
                import anthropic  # lazy import: optional dependency
            except ImportError:
                raise ImportError(
                    "AnthropicAdapter requires the `anthropic` package. "
                    "Install with: pip install 'agenteval[anthropic]'"
                )
            client = anthropic.Anthropic()
        self.client = client

    def run(self, messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
        # Only plain user/assistant text turns are replayed to the API;
        # tool exchange turns from earlier runs are already reflected in the
        # assistant text that followed them.
        conversation: List[Dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        new_turns: List[Turn] = []

        for _ in range(self.max_iterations):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": conversation,
            }
            if self.system:
                kwargs["system"] = self.system
            if tools:
                kwargs["tools"] = tools

            started = time.time()
            response = self.client.messages.create(**kwargs)
            latency_ms = (time.time() - started) * 1000.0

            text = "".join(
                getattr(b, "text", "") for b in response.content
                if getattr(b, "type", None) == "text"
            )
            calls = [
                ToolCall(name=b.name, args=dict(b.input), id=b.id)
                for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]
            usage = getattr(response, "usage", None)
            new_turns.append(Turn(
                role="assistant",
                content=text,
                tool_calls=calls,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                latency_ms=round(latency_ms, 1),
            ))
            conversation.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_result_blocks = []
            recorded_results = []
            for call in calls:
                if self.tool_executor is not None:
                    output = str(self.tool_executor(call.name, call.args))
                else:
                    output = "(agenteval: no tool_executor configured for '{}')".format(
                        call.name)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                })
                recorded_results.append({
                    "tool": call.name, "tool_use_id": call.id, "content": output,
                })
            conversation.append({"role": "user", "content": tool_result_blocks})
            new_turns.append(Turn(role="tool", tool_results=recorded_results))

        return Trajectory(new_turns, metadata={"model": self.model})


# ---------------------------------------------------------------------------
# agent loading ("module:attr" specs for CLI and pytest plugin)
# ---------------------------------------------------------------------------

def load_agent(spec: str) -> Agent:
    """Resolve an agent from a spec string.

    * ``"mock"`` returns a MockAdapter.
    * ``"package.module:attr"`` imports the module and resolves attr, where
      attr is an agent instance, an Agent class, a zero-arg factory returning
      an agent, or a plain callable (wrapped in CallableAdapter).
    """
    if spec == "mock":
        return MockAdapter()
    module_name, _, attr_name = spec.partition(":")
    if not module_name or not attr_name:
        raise ValueError(
            "agent spec must look like 'package.module:factory' (got {!r})".format(spec))
    if "" not in sys.path and os.getcwd() not in sys.path:  # pragma: no cover
        sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)

    if inspect.isclass(obj):
        obj = obj()
    elif callable(obj) and not hasattr(obj, "run"):
        obj = obj()  # zero-arg factory

    if hasattr(obj, "run"):
        return obj
    if callable(obj):
        return CallableAdapter(obj)
    raise TypeError(
        "{} did not resolve to an agent (needs a .run method or a callable)".format(spec))
