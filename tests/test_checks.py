from agenteval import ToolCall, Trajectory, Turn
from agenteval.checks import run_checks


def _trajectory(final="The answer is 42.", tool=None, assistant_turns=1):
    turns = [Turn("user", "question")]
    for _ in range(assistant_turns - 1):
        turns.append(Turn("assistant", "thinking out loud"))
    calls = [ToolCall(tool["name"], tool.get("args", {}), id="c1")] if tool else []
    if calls:
        turns.append(Turn("assistant", "", tool_calls=calls))
        turns.append(Turn("tool", tool_results=[{"tool": tool["name"], "content": "ok"}]))
    turns.append(Turn("assistant", final))
    return Trajectory(turns)


def _one(check, trajectory):
    results = run_checks([check], trajectory)
    assert len(results) == 1
    return results[0]


def test_contains_pass_and_fail():
    t = _trajectory("Berlin is sunny.")
    assert _one({"contains": "berlin"}, t).status == "pass"  # case-insensitive
    r = _one({"contains": "Hamburg"}, t)
    assert r.status == "fail"
    assert "Hamburg" in r.detail


def test_contains_list_all_must_match():
    t = _trajectory("Berlin is sunny.")
    assert _one({"contains": ["Berlin", "sunny"]}, t).status == "pass"
    assert _one({"contains": ["Berlin", "rainy"]}, t).status == "fail"


def test_not_contains():
    t = _trajectory("All good.")
    assert _one({"not_contains": "error"}, t).status == "pass"
    assert _one({"not_contains": "good"}, t).status == "fail"


def test_regex():
    t = _trajectory("Order #12345 confirmed.")
    assert _one({"regex": r"#\d{5}"}, t).status == "pass"
    assert _one({"regex": r"#\d{9}"}, t).status == "fail"
    assert _one({"regex": "("}, t).status == "fail"  # invalid pattern


def test_tool_called_and_not_called():
    t = _trajectory(tool={"name": "search", "args": {"q": "x"}})
    assert _one({"tool_called": "search"}, t).status == "pass"
    assert _one({"tool_called": "send_email"}, t).status == "fail"
    assert _one({"tool_not_called": "send_email"}, t).status == "pass"
    assert _one({"tool_not_called": "search"}, t).status == "fail"


def test_tool_called_min_times():
    t = _trajectory(tool={"name": "search"})
    assert _one({"tool_called": {"name": "search", "min_times": 2}}, t).status == "fail"
    assert _one({"tool_called": {"name": "search", "min_times": 1}}, t).status == "pass"


def test_tool_args_match_subset_and_nested():
    t = _trajectory(tool={"name": "search",
                          "args": {"q": "x", "filters": {"lang": "de", "limit": 5}}})
    ok = {"tool_args_match": {"tool": "search", "args": {"filters": {"lang": "de"}}}}
    bad = {"tool_args_match": {"tool": "search", "args": {"q": "y"}}}
    missing = {"tool_args_match": {"tool": "nope", "args": {}}}
    assert _one(ok, t).status == "pass"
    assert _one(bad, t).status == "fail"
    assert _one(missing, t).detail == "tool was never called"


def test_max_turns():
    t = _trajectory(assistant_turns=3)
    assert _one({"max_turns": 3}, t).status == "pass"
    assert _one({"max_turns": 2}, t).status == "fail"


def test_json_schema():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    }
    good = _trajectory('{"name": "Ada", "age": 36, "tags": ["dev"]}')
    assert _one({"json_schema": schema}, good).status == "pass"

    fenced = _trajectory('```json\n{"name": "Ada", "age": 36}\n```')
    assert _one({"json_schema": schema}, fenced).status == "pass"

    wrong_type = _trajectory('{"name": "Ada", "age": "old"}')
    r = _one({"json_schema": schema}, wrong_type)
    assert r.status == "fail" and "age" in r.detail

    not_json = _trajectory("hello there")
    assert _one({"json_schema": schema}, not_json).status == "fail"

    missing_required = _trajectory('{"name": "Ada"}')
    r = _one({"json_schema": schema}, missing_required)
    assert r.status == "fail" and "age" in r.detail


def test_json_schema_enum_and_bool_not_integer():
    t = _trajectory('{"status": "ok", "flag": true}')
    schema = {"type": "object",
              "properties": {"status": {"enum": ["ok", "error"]},
                             "flag": {"type": "boolean"}}}
    assert _one({"json_schema": schema}, t).status == "pass"
    int_schema = {"type": "object", "properties": {"flag": {"type": "integer"}}}
    assert _one({"json_schema": int_schema}, t).status == "fail"


def test_unknown_and_malformed_checks_fail():
    t = _trajectory()
    assert _one({"frobnicate": 1}, t).status == "fail"
    results = run_checks(["not a dict"], t)
    assert results[0].status == "fail"


def test_judge_skips_without_api_key(tmp_path):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "name: demo\npass_threshold: 0.5\ncriteria:\n"
        "  - name: quality\n    description: any\n    weight: 1\n",
        encoding="utf-8",
    )
    t = _trajectory()
    result = run_checks([{"judge": {"rubric": str(rubric)}}], t)[0]
    assert result.status == "skip"
    assert "skipped" in result.detail.lower()


def test_judge_missing_rubric_fails():
    t = _trajectory()
    result = run_checks([{"judge": {"rubric": "does/not/exist.yaml"}}], t)[0]
    assert result.status == "fail"
