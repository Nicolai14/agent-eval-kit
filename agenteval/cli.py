"""Command line interface.

    python -m agenteval run ./scenarios
    python -m agenteval run ./scenarios --agent myapp.agent:create
"""

import argparse
import os
import sys
from typing import List, Optional

from .adapters import load_agent
from .runner import run_scenarios
from .scenario import ScenarioError, load_scenarios
from .trajectory import RunRecorder

_STATUS_LABEL = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agenteval",
        description="pytest-based eval harness for LLM agents",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="run a scenario directory against an agent")
    run_parser.add_argument("scenarios", help="directory with *.yaml scenarios (or one file)")
    run_parser.add_argument("--agent", default="mock",
                            help="agent spec 'module:factory', or 'mock' (default)")
    run_parser.add_argument("--judge-model", default=None,
                            help="override the LLM judge model")
    run_parser.add_argument("--no-record", action="store_true",
                            help="do not persist the run under .agenteval/runs/")
    run_parser.add_argument("--runs-dir", default=os.path.join(".agenteval", "runs"),
                            help="where to persist runs (default: .agenteval/runs)")
    run_parser.add_argument("--min-pass-rate", type=float, default=None, metavar="0.0-1.0",
                            help="exit non-zero if fewer scenarios pass than this rate "
                                 "(default: any failure is non-zero)")
    run_parser.add_argument("-v", "--verbose", action="store_true",
                            help="print check details also for passing checks")
    return parser


def _print_table(results, verbose: bool) -> None:
    rows = []
    for result in results:
        for check in result.checks:
            rows.append((result.scenario.name, check.name,
                         _STATUS_LABEL.get(check.status, check.status.upper()),
                         check.detail))
    if not rows:
        print("no checks defined in any scenario")
        return
    scenario_width = max(len(r[0]) for r in rows)
    check_width = min(max(len(r[1]) for r in rows), 48)
    header = "{:<{sw}}  {:<{cw}}  {}".format(
        "scenario", "check", "result", sw=scenario_width, cw=check_width)
    print(header)
    print("-" * len(header))
    previous_scenario = None
    for scenario_name, check_name, status, detail in rows:
        shown = scenario_name if scenario_name != previous_scenario else ""
        previous_scenario = scenario_name
        print("{:<{sw}}  {:<{cw}}  {}".format(
            shown, check_name[:check_width], status,
            sw=scenario_width, cw=check_width))
        if detail and (status != "PASS" or verbose):
            print("{:<{sw}}    -> {}".format("", detail, sw=scenario_width))
    print("-" * len(header))


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 2

    try:
        scenarios = load_scenarios(args.scenarios)
    except ScenarioError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    try:
        agent = load_agent(args.agent)
    except Exception as exc:
        print("error loading agent {!r}: {}".format(args.agent, exc), file=sys.stderr)
        return 2

    recorder = None if args.no_record else RunRecorder(root=args.runs_dir)
    results = run_scenarios(agent, scenarios, recorder=recorder,
                            judge_model=args.judge_model)

    _print_table(results, verbose=args.verbose)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    all_checks = [c for r in results for c in r.checks]
    checks_passed = sum(1 for c in all_checks if c.status == "pass")
    checks_skipped = sum(1 for c in all_checks if c.status == "skip")
    print("{}/{} scenarios passed ({}/{} checks passed, {} skipped)".format(
        passed, total, checks_passed, len(all_checks), checks_skipped))
    if recorder is not None:
        print("run recorded: {}".format(recorder.path))

    if args.min_pass_rate is not None:
        rate = passed / total if total else 0.0
        if rate < args.min_pass_rate:
            print("pass rate {:.2f} below required {:.2f}".format(
                rate, args.min_pass_rate), file=sys.stderr)
            return 1
        return 0
    return 0 if passed == total else 1


def entry() -> None:  # console_scripts entry point
    sys.exit(main())
