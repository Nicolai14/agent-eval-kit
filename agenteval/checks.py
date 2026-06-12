"""Declarative checks that run against a Trajectory.

Supported check types (each scenario lists them under `expected:`):

    - contains: "text"                  # substring in assistant output (case-insensitive);
                                        # also accepts a list — all must match
    - not_contains: "text"              # inverse of contains
    - regex: "pattern"                  # re.search over assistant output (case-sensitive)
    - json_schema: {...}                # final assistant message parses as JSON
                                        # matching a minimal JSON-schema subset
    - tool_called: name                 # or {name: ..., min_times: N}
    - tool_not_called: name
    - tool_args_match: {tool: name, args: {...}}   # subset match on any call's args
    - max_turns: N                      # at most N assistant turns
    - judge: {rubric: path, model: ..., threshold: ...}   # LLM judge (skips w/o API key)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .judge import JudgeUnavailable, Rubric, judge_trajectory
from .trajectory import Trajectory

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _result(name: str, ok: bool, detail_fail: str = "", detail_pass: str = "") -> CheckResult:
    return CheckResult(name, PASS if ok else FAIL, detail_pass if ok else detail_fail)


# ---------------------------------------------------------------------------
# text checks
# ---------------------------------------------------------------------------

def _check_contains(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    needles = value if isinstance(value, list) else [value]
    text = trajectory.assistant_text().lower()
    missing = [str(n) for n in needles if str(n).lower() not in text]
    name = "contains: {}".format(", ".join(str(n) for n in needles))
    return _result(name, not missing,
                   "missing substring(s): {}".format(", ".join(repr(m) for m in missing)))


def _check_not_contains(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    needles = value if isinstance(value, list) else [value]
    text = trajectory.assistant_text().lower()
    found = [str(n) for n in needles if str(n).lower() in text]
    name = "not_contains: {}".format(", ".join(str(n) for n in needles))
    return _result(name, not found,
                   "forbidden substring(s) present: {}".format(", ".join(repr(f) for f in found)))


def _check_regex(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    pattern = str(value)
    name = "regex: {}".format(pattern)
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as exc:
        return CheckResult(name, FAIL, "invalid regex: {}".format(exc))
    ok = bool(compiled.search(trajectory.assistant_text()))
    return _result(name, ok, "pattern not found in assistant output")


# ---------------------------------------------------------------------------
# json_schema check (minimal validator — no extra dependency)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _validate_schema(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    errors: List[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("{}: {!r} not in enum {!r}".format(path, instance, schema["enum"]))

    expected_type = schema.get("type")
    if expected_type:
        if expected_type == "integer":
            ok = isinstance(instance, int) and not isinstance(instance, bool)
        elif expected_type == "number":
            ok = isinstance(instance, (int, float)) and not isinstance(instance, bool)
        else:
            py_type = _TYPE_MAP.get(expected_type)
            ok = py_type is not None and isinstance(instance, py_type)
        if not ok:
            errors.append("{}: expected type {}, got {}".format(
                path, expected_type, type(instance).__name__))
            return errors

    if isinstance(instance, dict):
        for required_key in schema.get("required", []):
            if required_key not in instance:
                errors.append("{}: missing required property {!r}".format(path, required_key))
        for key, sub_schema in schema.get("properties", {}).items():
            if key in instance and isinstance(sub_schema, dict):
                errors.extend(_validate_schema(instance[key], sub_schema,
                                               "{}.{}".format(path, key)))
    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(instance):
            errors.extend(_validate_schema(item, schema["items"],
                                           "{}[{}]".format(path, index)))
    return errors


def _check_json_schema(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    schema = value.get("schema", value) if isinstance(value, dict) else value
    name = "json_schema"
    if not isinstance(schema, dict):
        return CheckResult(name, FAIL, "schema must be a mapping")
    raw = trajectory.final_text().strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        instance = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CheckResult(name, FAIL,
                           "final assistant message is not valid JSON: {}".format(exc))
    errors = _validate_schema(instance, schema)
    return _result(name, not errors, "; ".join(errors))


# ---------------------------------------------------------------------------
# tool checks
# ---------------------------------------------------------------------------

def _tool_name_and_min(value: Any) -> Tuple[str, int]:
    if isinstance(value, dict):
        return str(value.get("name", "")), int(value.get("min_times", 1))
    return str(value), 1


def _check_tool_called(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    tool_name, min_times = _tool_name_and_min(value)
    count = sum(1 for c in trajectory.tool_calls() if c.name == tool_name)
    name = "tool_called: {}".format(tool_name)
    detail = "called {} time(s), expected >= {}".format(count, min_times)
    return _result(name, count >= min_times, detail)


def _check_tool_not_called(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    tool_name = str(value)
    count = sum(1 for c in trajectory.tool_calls() if c.name == tool_name)
    name = "tool_not_called: {}".format(tool_name)
    return _result(name, count == 0, "tool was called {} time(s)".format(count))


def _args_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _args_subset(v, actual[k]) for k, v in expected.items())
    return expected == actual


def _check_tool_args_match(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    if not isinstance(value, dict) or "tool" not in value:
        return CheckResult("tool_args_match", FAIL,
                           "expected {tool: name, args: {...}}")
    tool_name = str(value["tool"])
    expected_args = value.get("args", {})
    name = "tool_args_match: {}".format(tool_name)
    calls = [c for c in trajectory.tool_calls() if c.name == tool_name]
    if not calls:
        return CheckResult(name, FAIL, "tool was never called")
    if any(_args_subset(expected_args, c.args) for c in calls):
        return CheckResult(name, PASS)
    observed = "; ".join(json.dumps(c.args, ensure_ascii=False, default=str) for c in calls)
    return CheckResult(name, FAIL,
                       "no call matched {} — observed args: {}".format(
                           json.dumps(expected_args, ensure_ascii=False), observed))


def _check_max_turns(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    limit = int(value)
    count = len(trajectory.assistant_turns())
    name = "max_turns: {}".format(limit)
    return _result(name, count <= limit,
                   "{} assistant turns, limit is {}".format(count, limit))


# ---------------------------------------------------------------------------
# judge check
# ---------------------------------------------------------------------------

def _check_judge(value: Any, trajectory: Trajectory, ctx: Dict[str, Any]) -> CheckResult:
    if isinstance(value, str):
        value = {"rubric": value}
    if not isinstance(value, dict) or "rubric" not in value:
        return CheckResult("judge", FAIL, "expected {rubric: path, ...}")

    rubric_path = str(value["rubric"])
    if not os.path.isabs(rubric_path):
        rubric_path = os.path.normpath(os.path.join(ctx.get("base_dir", "."), rubric_path))
    try:
        rubric = Rubric.load(rubric_path)
    except (OSError, ValueError) as exc:
        return CheckResult("judge", FAIL, "cannot load rubric {}: {}".format(rubric_path, exc))

    name = "judge: {}".format(rubric.name)
    try:
        result = judge_trajectory(
            trajectory,
            rubric,
            model=value.get("model") or ctx.get("judge_model"),
            threshold=value.get("threshold"),
        )
    except JudgeUnavailable as exc:
        return CheckResult(name, SKIP, str(exc))
    except Exception as exc:  # API/network/parse errors are real failures
        return CheckResult(name, FAIL, "judge error: {}".format(exc))

    detail = "score {:.2f} (threshold {:.2f}, model {}) — {}".format(
        result.score, result.threshold, result.model, result.reasoning)
    return CheckResult(name, PASS if result.passed else FAIL, detail)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

HANDLERS = {
    "contains": _check_contains,
    "not_contains": _check_not_contains,
    "regex": _check_regex,
    "json_schema": _check_json_schema,
    "tool_called": _check_tool_called,
    "tool_not_called": _check_tool_not_called,
    "tool_args_match": _check_tool_args_match,
    "max_turns": _check_max_turns,
    "judge": _check_judge,
}


def run_checks(
    checks: List[Dict[str, Any]],
    trajectory: Trajectory,
    base_dir: str = ".",
    judge_model: Optional[str] = None,
) -> List[CheckResult]:
    ctx = {"base_dir": base_dir, "judge_model": judge_model}
    results: List[CheckResult] = []
    for spec in checks:
        if not isinstance(spec, dict) or len(spec) != 1:
            results.append(CheckResult(repr(spec), FAIL,
                                       "each check must be a single-key mapping"))
            continue
        check_type, value = next(iter(spec.items()))
        handler = HANDLERS.get(check_type)
        if handler is None:
            results.append(CheckResult(str(check_type), FAIL,
                                       "unknown check type (known: {})".format(
                                           ", ".join(sorted(HANDLERS)))))
            continue
        try:
            results.append(handler(value, trajectory, ctx))
        except Exception as exc:
            results.append(CheckResult(str(check_type), FAIL,
                                       "check raised {}: {}".format(type(exc).__name__, exc)))
    return results
