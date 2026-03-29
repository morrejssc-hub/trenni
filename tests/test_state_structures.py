"""Tests for SpawnedJob/SpawnDefaults schema per ADR-0007.

ADR-0007 Decision 2: Spawn payload is a task-semantics document.
It does not carry execution config overrides (llm, workspace, publication).
"""

import pytest


def test_spawned_job_no_execution_overrides():
    """SpawnedJob must not carry execution config overrides."""
    from trenni.state import SpawnedJob

    job = SpawnedJob(
        job_id="test-123",
        source_event_id="evt-1",
        task="Do something",
        role="implementer",
        repo="https://github.com/org/repo",
        init_branch="main",
        evo_sha="abc123",
        task_id="task-1",
        team="default",
    )
    # These fields should not exist or should be empty dicts
    # After ADR-0007, they are removed entirely
    assert not hasattr(job, "llm_overrides")
    assert not hasattr(job, "workspace_overrides")
    assert not hasattr(job, "publication_overrides")


def test_spawn_defaults_no_execution_overrides():
    """SpawnDefaults must not carry execution config overrides."""
    from trenni.state import SpawnDefaults

    defaults = SpawnDefaults(
        repo="https://github.com/org/repo",
        init_branch="main",
        role="implementer",
        evo_sha="abc123",
        team="default",
    )
    assert not hasattr(defaults, "llm_overrides")
    assert not hasattr(defaults, "workspace_overrides")
    assert not hasattr(defaults, "publication_overrides")


def test_spawned_job_role_params_only_for_internal_flags():
    """role_params exists but should only contain role-internal flags."""
    from trenni.state import SpawnedJob

    job = SpawnedJob(
        job_id="test-123",
        source_event_id="evt-1",
        task="Do something",
        role="planner",
        repo="https://github.com/org/repo",
        init_branch="main",
        evo_sha="abc123",
        role_params={"mode": "join"},  # role-internal flag, not task content
    )
    assert job.role_params == {"mode": "join"}
    # goal is NOT in role_params per ADR-0007
    assert "goal" not in job.role_params


def test_spawn_defaults_role_params_only_for_internal_flags():
    """SpawnDefaults role_params should only contain role-internal flags."""
    from trenni.state import SpawnDefaults

    defaults = SpawnDefaults(
        repo="https://github.com/org/repo",
        init_branch="main",
        role="planner",
        evo_sha="abc123",
        role_params={"mode": "join"},  # role-internal flag
        team="default",
    )
    assert defaults.role_params == {"mode": "join"}