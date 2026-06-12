import os

from agenteval.cli import main

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "scenarios")


def test_cli_run_examples_passes(capsys):
    exit_code = main(["run", SCENARIO_DIR, "--no-record"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "5/5 scenarios passed" in out
    assert "PASS" in out
    assert "SKIP" in out  # the offline judge check


def test_cli_records_run(tmp_path, capsys):
    exit_code = main(["run", SCENARIO_DIR, "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "run recorded:" in out
    run_dirs = os.listdir(str(tmp_path))
    assert len(run_dirs) == 1
    assert os.path.exists(os.path.join(str(tmp_path), run_dirs[0], "results.jsonl"))


def test_cli_failure_exit_code(tmp_path, capsys):
    scenario = tmp_path / "fail.yaml"
    scenario.write_text("input: hello\nexpected:\n  - contains: zebra\n",
                        encoding="utf-8")
    exit_code = main(["run", str(tmp_path), "--no-record"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL" in out
    assert "0/1 scenarios passed" in out


def test_cli_min_pass_rate(tmp_path, capsys):
    (tmp_path / "ok.yaml").write_text(
        "input: hello\nexpected:\n  - contains: help\n", encoding="utf-8")
    (tmp_path / "bad.yaml").write_text(
        "input: hello\nexpected:\n  - contains: zebra\n", encoding="utf-8")
    # 1 of 2 passes -> rate 0.5
    assert main(["run", str(tmp_path), "--no-record", "--min-pass-rate", "0.4"]) == 0
    capsys.readouterr()
    assert main(["run", str(tmp_path), "--no-record", "--min-pass-rate", "0.9"]) == 1


def test_cli_bad_paths(capsys):
    assert main(["run", "/nonexistent/dir", "--no-record"]) == 2
    assert main(["run", SCENARIO_DIR, "--agent", "nope.module:create",
                 "--no-record"]) == 2
