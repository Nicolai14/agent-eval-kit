"""LLM judge: scores a trajectory against a YAML rubric via the Anthropic API.

A rubric file looks like:

    name: helpfulness
    description: Did the agent actually help the user?
    pass_threshold: 0.7
    criteria:
      - name: addresses_request
        description: The response addresses the user's actual request.
        weight: 3
      - name: actionable
        description: The user can act on the answer without follow-up questions.
        weight: 2

The judge asks the model to score each criterion between 0.0 and 1.0 and
computes the weighted average. If the `anthropic` package is missing or
ANTHROPIC_API_KEY is not set, judge checks are *skipped* (not failed) with a
clear message, so the offline test suite stays green.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .trajectory import Trajectory

#: Default judge model. Haiku keeps judge costs low; override per-check with
#: `judge: {model: ...}` or globally via the AGENTEVAL_JUDGE_MODEL env var.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


class JudgeUnavailable(Exception):
    """Judge cannot run in this environment (missing key or package)."""


class JudgeParseError(Exception):
    """The judge model returned something we could not parse."""


@dataclass
class Criterion:
    name: str
    description: str
    weight: float = 1.0


@dataclass
class Rubric:
    name: str
    criteria: List[Criterion]
    description: str = ""
    pass_threshold: float = 0.7

    @classmethod
    def load(cls, path: str) -> "Rubric":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or "criteria" not in data:
            raise ValueError("{}: rubric must define a 'criteria' list".format(path))
        criteria = [
            Criterion(
                name=str(c["name"]),
                description=str(c.get("description", "")),
                weight=float(c.get("weight", 1.0)),
            )
            for c in data["criteria"]
        ]
        if not criteria:
            raise ValueError("{}: rubric has no criteria".format(path))
        return cls(
            name=str(data.get("name") or os.path.splitext(os.path.basename(path))[0]),
            description=str(data.get("description", "")),
            pass_threshold=float(data.get("pass_threshold", 0.7)),
            criteria=criteria,
        )


@dataclass
class JudgeResult:
    score: float
    passed: bool
    threshold: float
    scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    model: str = ""


def render_transcript(trajectory: Trajectory) -> str:
    """Human-readable transcript for the judge prompt."""
    lines = []
    for turn in trajectory.turns:
        if turn.role == "tool":
            for result in turn.tool_results:
                lines.append("[tool result: {}] {}".format(
                    result.get("tool", "?"), result.get("content", "")))
            continue
        if turn.content:
            lines.append("{}: {}".format(turn.role, turn.content))
        for call in turn.tool_calls:
            lines.append("{} -> tool call: {}({})".format(
                turn.role, call.name, json.dumps(call.args, ensure_ascii=False)))
    return "\n".join(lines)


def _build_prompt(trajectory: Trajectory, rubric: Rubric) -> str:
    criteria_lines = []
    for c in rubric.criteria:
        criteria_lines.append("- {} (weight {}): {}".format(c.name, c.weight, c.description))
    return (
        "Evaluate the following agent conversation transcript against a rubric.\n\n"
        "Rubric: {name}\n{description}\n\nCriteria:\n{criteria}\n\n"
        "Transcript:\n---\n{transcript}\n---\n\n"
        "Score each criterion between 0.0 (not met at all) and 1.0 (fully met).\n"
        "Respond with JSON only, exactly in this shape:\n"
        '{{"scores": {{{score_keys}}}, "reasoning": "<one or two sentences>"}}'
    ).format(
        name=rubric.name,
        description=rubric.description,
        criteria="\n".join(criteria_lines),
        transcript=render_transcript(trajectory),
        score_keys=", ".join('"{}": <0.0-1.0>'.format(c.name) for c in rubric.criteria),
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of the model response."""
    text = text.strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeParseError("no JSON object in judge response: {!r}".format(text[:200]))
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError("invalid JSON in judge response: {}".format(exc))


def judge_trajectory(
    trajectory: Trajectory,
    rubric: Rubric,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
    client: Any = None,
) -> JudgeResult:
    """Score a trajectory against a rubric.

    `client` is injectable for testing; when None, an anthropic.Anthropic()
    client is created lazily. Raises JudgeUnavailable when no API key is set
    or the anthropic package is not installed — callers turn this into a
    skipped check.
    """
    model = model or os.environ.get("AGENTEVAL_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    effective_threshold = rubric.pass_threshold if threshold is None else float(threshold)

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise JudgeUnavailable(
                "ANTHROPIC_API_KEY is not set — judge check skipped. "
                "Export the key to enable LLM-judged checks."
            )
        try:
            import anthropic  # lazy import: optional dependency
        except ImportError:
            raise JudgeUnavailable(
                "the `anthropic` package is not installed — judge check skipped. "
                "Install with: pip install 'agenteval[anthropic]'"
            )
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "You are a strict evaluation judge for AI agent transcripts. "
            "You respond with JSON only — no prose outside the JSON object."
        ),
        messages=[{"role": "user", "content": _build_prompt(trajectory, rubric)}],
    )
    text = "".join(
        getattr(block, "text", "") for block in response.content
        if getattr(block, "type", None) == "text"
    )
    data = _extract_json(text)
    raw_scores = data.get("scores", {})

    scores: Dict[str, float] = {}
    total_weight = 0.0
    weighted = 0.0
    for criterion in rubric.criteria:
        try:
            value = float(raw_scores.get(criterion.name, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(1.0, value))
        scores[criterion.name] = value
        total_weight += criterion.weight
        weighted += criterion.weight * value

    score = weighted / total_weight if total_weight else 0.0
    return JudgeResult(
        score=round(score, 4),
        passed=score >= effective_threshold,
        threshold=effective_threshold,
        scores=scores,
        reasoning=str(data.get("reasoning", "")),
        model=model,
    )
