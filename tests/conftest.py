import pytest


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests must be deterministic and offline: never let a real key leak in."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AGENTEVAL_JUDGE_MODEL", raising=False)
