# agenteval

Eval harness for LLM agents, built on pytest. You describe expected agent behavior in YAML scenarios — tool calls, refusals, output shape, multi-turn memory — and agenteval runs them against your agent, records every trajectory, and fails your build when behavior regresses. No platform, no dashboard, no per-seat pricing: it's a Python package that runs on your machine and in your CI.

```
pip install agenteval
```

Dependencies: `pyyaml` and `pytest`. The `anthropic` SDK is an optional extra, only needed for the bundled `AnthropicAdapter` and the LLM judge. Everything else — including the full test suite — runs offline.

## Quickstart (under 5 minutes)

Clone or install, then run the shipped example scenarios against the deterministic `MockAdapter`:

```
python -m agenteval run ./scenarios
```

```
scenario                check                                   result
----------------------------------------------------------------------
hallucinated_tool_args  tool_not_called: get_weather            PASS
                        contains: which city                    PASS
...
5/5 scenarios passed (16/17 checks passed, 1 skipped)
run recorded: .agenteval/runs/20260611-192201-e65ec4/results.jsonl
```

Now point it at your own agent. Write a factory that returns anything with a `run(messages, tools=None) -> Trajectory` method (or just a plain function — see Adapters below):

```python
# myapp/evalagent.py
from agenteval import CallableAdapter

def create():
    return CallableAdapter(my_agent_function)
```

```
python -m agenteval run ./scenarios --agent myapp.evalagent:create
```

Or run scenarios as pytest tests:

```
pytest --agenteval-scenarios=./scenarios --agenteval-agent=myapp.evalagent:create
```

Every run is persisted as JSONL under `.agenteval/runs/<run_id>/results.jsonl` — full trajectories included — so you can diff runs later.

## Scenario format

One YAML file per scenario:

```yaml
name: tool_use_happy_path          # optional, defaults to the file name
description: Agent answers a weather question via the get_weather tool.

input: "What's the weather in Berlin right now?"
# multi-turn works too:
# input:
#   - role: user
#     content: "My name is Ada."
#   - role: user
#     content: "What is my name?"

tools:                             # optional; passed through to the agent
  - name: get_weather
    description: Get the current weather for a city.
    input_schema:
      type: object
      properties:
        city: {type: string}
      required: [city]

expected:
  - tool_called: get_weather
  - tool_args_match:
      tool: get_weather
      args: {city: Berlin}
  - contains: Berlin
  - max_turns: 3
```

Multi-turn semantics: each `user` message is sent in order, and the agent is invoked with the full conversation so far after each one. You can also script `assistant` turns in `input` to set up context without invoking the agent.

### Checks

| Check | Behavior |
| --- | --- |
| `contains: "text"` | Substring in assistant output, case-insensitive. Also accepts a list (all must match). |
| `not_contains: "text"` | Inverse of `contains`. |
| `regex: "pattern"` | `re.search` over assistant output, case-sensitive. |
| `json_schema: {...}` | Final assistant message must parse as JSON matching the schema (minimal subset: `type`, `properties`, `required`, `items`, `enum`; code fences are stripped). |
| `tool_called: name` | Tool was called at least once. Also `{name: ..., min_times: N}`. |
| `tool_not_called: name` | Tool was never called. |
| `tool_args_match: {tool: name, args: {...}}` | Some call to the tool has args that contain the given keys/values (recursive subset match). |
| `max_turns: N` | At most N assistant turns. Catches loops and cost regressions. |
| `judge: {rubric: path}` | LLM-judged check against a rubric (see below). Optional `model` and `threshold` overrides. |

A scenario passes when no check fails. Skipped checks (e.g. judge without an API key) don't fail a scenario.

## Adapters

agenteval talks to your agent through one protocol:

```python
class Agent(Protocol):
    def run(self, messages, tools=None) -> Trajectory: ...
```

`messages` is a list of `{role, content, ...}` dicts (the conversation so far). The returned `Trajectory` contains the new turns: assistant text, tool calls, tool results, plus optional token/latency fields.

**`CallableAdapter`** — wrap any function. It may return a plain string, a `Turn`, a list of `Turn`s, or a `Trajectory`:

```python
from agenteval import CallableAdapter, Trajectory, Turn, ToolCall

def my_agent(messages, tools=None):
    # call your stack here: LangGraph, your own loop, an HTTP service, ...
    return Trajectory([
        Turn("assistant", "", tool_calls=[ToolCall("get_weather", {"city": "Berlin"})]),
        Turn("tool", tool_results=[{"tool": "get_weather", "content": "12°C"}]),
        Turn("assistant", "It's 12°C in Berlin."),
    ])

agent = CallableAdapter(my_agent)
```

If your function only returns the final string, that works too — you just lose tool-call checks.

**`AnthropicAdapter`** — runs a Messages API tool-use loop (`pip install 'agenteval[anthropic]'`):

```python
from agenteval import AnthropicAdapter

def execute_tool(name, args):
    return my_tools[name](**args)

agent = AnthropicAdapter(
    model="claude-opus-4-8",
    system="You are a support agent.",
    tool_executor=execute_tool,
)
```

**`MockAdapter`** — deterministic and offline. Powers the example scenarios and is useful for testing your scenario files themselves: `MockAdapter(script=["reply 1", "reply 2"])` plays back one item per invocation.

For OpenAI or any other provider, wrap your existing client code in a `CallableAdapter` — agenteval never imports a provider SDK on your behalf.

## LLM judge

For qualities that string checks can't express (helpfulness, tone, groundedness), add a `judge` check pointing at a rubric:

```yaml
# rubrics/helpfulness.yaml
name: helpfulness
pass_threshold: 0.7
criteria:
  - name: addresses_request
    description: The response addresses the user's actual request.
    weight: 3
  - name: actionable
    description: The user can act on the answer without follow-ups.
    weight: 2
```

```yaml
expected:
  - judge:
      rubric: ../rubrics/helpfulness.yaml
```

The judge sends the transcript and rubric to the Anthropic API, scores each criterion 0.0–1.0, and passes when the weighted average reaches `pass_threshold`. The default judge model is `claude-haiku-4-5-20251001` to keep costs low; override per check (`model:`), via `--judge-model`, or with `AGENTEVAL_JUDGE_MODEL`.

Without `ANTHROPIC_API_KEY` (or without the `anthropic` package), judge checks are **skipped with a clear message**, never failed — your offline runs and CI without secrets stay green.

## pytest integration

Plugin mode (registered automatically on install):

```
pytest --agenteval-scenarios=./scenarios --agenteval-agent=myapp.evalagent:create
```

Each YAML file becomes one test item with a readable failure report listing every check. The scenario directory needs to be within pytest's collection roots — run from the project root or pass the directory as an argument (`pytest scenarios/ --agenteval-scenarios=scenarios`). Add `--agenteval-no-record` to skip trajectory persistence.

Programmatic mode, if you prefer plain test code:

```python
import pytest
from agenteval import load_scenarios, run_scenario, load_agent

agent = load_agent("myapp.evalagent:create")

@pytest.mark.parametrize("scenario", load_scenarios("scenarios"), ids=lambda s: s.name)
def test_agent(scenario):
    result = run_scenario(agent, scenario)
    assert result.passed, result.summary()
```

## CI example (GitHub Actions)

```yaml
name: agent-evals
on: pull_request
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install agenteval
      - run: python -m agenteval run ./scenarios --agent myapp.evalagent:create
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}  # optional: enables judge checks
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: agenteval-runs
          path: .agenteval/runs/
```

`--min-pass-rate 0.9` is available if you want a soft gate instead of failing on any single scenario.

## FAQ

**Why not just unit-test my agent?**
You should — for the deterministic parts. Agent behavior (which tool gets called, with which args, whether a refusal happens) is emergent and shifts with every prompt or model change. Scenarios pin that behavior down as data, not code.

**Do I need an API key to use this?**
No. The harness, all check types except `judge`, the MockAdapter, and the full test suite run offline. You need a key only for `AnthropicAdapter` (if you use it) and judge checks.

**Are LLM-judged checks flaky?**
Less than you'd expect with a rubric of independently gradeable criteria and a threshold, but they're still a model's opinion. Use string/tool checks for hard contracts and the judge for soft qualities. Judge checks report their score and reasoning in the run log so you can audit them.

**How does this compare to hosted eval platforms?**
Hosted platforms give you dashboards, hosted datasets, and team workflows — for a monthly fee and your trajectories on their servers. agenteval is a library: your scenarios live in your repo, runs are JSONL files on your disk, and CI is your existing pipeline. If you outgrow it, the JSONL format is trivial to export.

**Python version support?**
3.9+.

**Non-Anthropic models for the judge?**
Currently the judge speaks the Anthropic API. The judge client is injectable (`judge_trajectory(..., client=...)`) if you want to adapt it.

---

## Pro pack

If agenteval saves you time, there's a paid companion pack — [Agent Eval Kit Pro](https://aiactkit.gumroad.com/l/agent-eval-pro) ($39, one-time, single team): 25 battle-tested scenario templates across five failure categories (tool misuse, hallucination, injection resistance, multi-turn memory, regression safety), a library of 8 judge rubrics, ready-made GitHub Actions workflows (PR gate + nightly full run), and an HTML trajectory diff report for comparing two runs. The core stays MIT and complete — the pack is a head start, not a feature unlock.
