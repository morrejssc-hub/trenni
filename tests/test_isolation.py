import os

from trenni.isolation import _build_job_env


def test_build_job_env_forwards_explicit_extra_env_keys(monkeypatch):
    monkeypatch.setenv("PASLOE_API_KEY", "pasloe-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    env = _build_job_env(
        "PASLOE_API_KEY",
        extra_env_keys=["OPENAI_API_KEY", "GITHUB_TOKEN"],
    )

    assert env["PASLOE_API_KEY"] == "pasloe-secret"
    assert env["OPENAI_API_KEY"] == "openai-secret"
    assert env["GITHUB_TOKEN"] == "github-secret"
