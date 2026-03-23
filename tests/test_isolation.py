import os
from pathlib import Path

from trenni.isolation import (
    _build_git_credential_env,
    _build_job_env,
    prepare_workspace,
    JobWorkspace,
)


def test_build_job_env_forwards_explicit_env_keys(monkeypatch):
    monkeypatch.setenv("PASLOE_API_KEY", "pasloe-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    env = _build_job_env(
        "PASLOE_API_KEY",
        env_keys=["OPENAI_API_KEY", "GITHUB_TOKEN"],
    )

    assert env["PASLOE_API_KEY"] == "pasloe-secret"
    assert env["OPENAI_API_KEY"] == "openai-secret"
    assert env["GITHUB_TOKEN"] == "github-secret"


def test_build_job_env_no_hardcoded_keys(monkeypatch):
    """Verify no hardcoded ANTHROPIC_API_KEY or GIT_TOKEN leaks through."""
    monkeypatch.setenv("PASLOE_API_KEY", "pasloe-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-appear")
    monkeypatch.setenv("GIT_TOKEN", "should-not-appear")

    env = _build_job_env("PASLOE_API_KEY", env_keys=[])

    assert "ANTHROPIC_API_KEY" not in env
    assert "GIT_TOKEN" not in env


def test_build_git_credential_env(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "test-token-123")
    result = _build_git_credential_env("MY_TOKEN")

    assert result["GIT_CONFIG_COUNT"] == "2"
    assert "AUTHORIZATION" in result["GIT_CONFIG_VALUE_1"]


def test_build_git_credential_env_empty_when_no_token(monkeypatch):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    result = _build_git_credential_env("MY_TOKEN")
    assert result == {}


def test_build_git_credential_env_empty_when_no_env_name():
    result = _build_git_credential_env("")
    assert result == {}


def test_prepare_workspace_creates_dir_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PASLOE_API_KEY", "test-key")

    evo_dir = tmp_path / "evo-repo"
    evo_dir.mkdir()

    ws = prepare_workspace(
        job_id="test-job-1",
        work_dir=tmp_path / "work",
        evo_repo_path=str(evo_dir),
        config={"job_id": "test-job-1", "task": "test"},
        eventstore_api_key_env="PASLOE_API_KEY",
        env_keys=[],
    )

    assert isinstance(ws, JobWorkspace)
    assert ws.job_dir.exists()
    assert ws.config_path.exists()
    assert (ws.job_dir / "evo").is_symlink()
    assert ws.env["PASLOE_API_KEY"] == "test-key"


def test_prepare_workspace_injects_git_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("PASLOE_API_KEY", "test-key")
    monkeypatch.setenv("MY_GIT_TOKEN", "git-secret")

    evo_dir = tmp_path / "evo-repo"
    evo_dir.mkdir()

    ws = prepare_workspace(
        job_id="test-job-2",
        work_dir=tmp_path / "work",
        evo_repo_path=str(evo_dir),
        config={"job_id": "test-job-2"},
        eventstore_api_key_env="PASLOE_API_KEY",
        env_keys=[],
        git_token_env="MY_GIT_TOKEN",
    )

    assert ws.env.get("GIT_CONFIG_COUNT") == "2"
    assert "AUTHORIZATION" in ws.env.get("GIT_CONFIG_VALUE_1", "")
