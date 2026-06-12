"""Scenario loading.

A scenario is a YAML file:

    name: tool_use_happy_path          # optional, defaults to the file name
    description: ...                   # optional
    input: "single user message"       # or a list of messages (multi-turn)
    tools:                             # optional tool definitions passed to the agent
      - name: get_weather
        description: ...
        input_schema: {...}
    expected:                          # list of declarative checks
      - contains: Berlin
      - tool_called: get_weather

See checks.py for the full list of check types.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class ScenarioError(ValueError):
    """Raised for malformed scenario files."""


@dataclass
class Scenario:
    name: str
    input: List[Dict[str, str]]
    checks: List[Dict[str, Any]]
    description: str = ""
    tools: Optional[List[Dict[str, Any]]] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path)) if self.path else "."


def _normalize_input(raw: Any, path: str) -> List[Dict[str, str]]:
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if isinstance(raw, list):
        messages = []
        for item in raw:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict) and "content" in item:
                messages.append({
                    "role": str(item.get("role", "user")),
                    "content": str(item["content"]),
                })
            else:
                raise ScenarioError(
                    "{}: each input message must be a string or a "
                    "{{role, content}} mapping, got: {!r}".format(path, item)
                )
        if not messages:
            raise ScenarioError("{}: input is empty".format(path))
        return messages
    raise ScenarioError(
        "{}: input must be a string or a list of messages".format(path)
    )


def load_scenario(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ScenarioError("{}: top level must be a mapping".format(path))
    if "input" not in data:
        raise ScenarioError("{}: missing required field 'input'".format(path))

    checks = data.get("expected", [])
    if not isinstance(checks, list):
        raise ScenarioError("{}: 'expected' must be a list of checks".format(path))

    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    return Scenario(
        name=str(name),
        description=str(data.get("description", "")),
        input=_normalize_input(data["input"], path),
        tools=data.get("tools"),
        checks=checks,
        path=os.path.abspath(path),
        metadata=data.get("metadata", {}),
    )


def load_scenarios(directory: str) -> List[Scenario]:
    """Load every *.yaml / *.yml file in a directory (non-recursive), sorted."""
    if os.path.isfile(directory):
        return [load_scenario(directory)]
    if not os.path.isdir(directory):
        raise ScenarioError("scenario path does not exist: {}".format(directory))
    paths = sorted(
        glob.glob(os.path.join(directory, "*.yaml"))
        + glob.glob(os.path.join(directory, "*.yml"))
    )
    if not paths:
        raise ScenarioError("no .yaml scenarios found in {}".format(directory))
    return [load_scenario(p) for p in paths]
