"""Tests for SpawnedJob/SpawnDefaults schema per ADR-0007.

ADR-0007 Decision 2: Spawn payload is a task-semantics document.
It does not carry execution config overrides (llm, workspace, publication).
"""

import pytest
import tempfile
from pathlib import Path


@pytest.mark.skip(reason="Budget validation disabled per Bundle MVP")
def test_spawn_handler_rejects_budget_above_max_cost():
    """Spawn is rejected if budget exceeds role's max_cost (ADR-0004 D1a).
    
    Note: Budget validation is disabled per Bundle MVP.
    """
    from trenni.spawn_handler import SpawnHandler
    from trenni.state import SupervisorState, SpawnedJob
    from yoitsu_contracts.role_metadata import RoleMetadataReader
    from yoitsu_contracts.events import SpawnRequestData

    # Create a mock evo directory with a role that has max_cost=0.50
    with tempfile.TemporaryDirectory() as tmpdir:
        evo_path = Path(tmpdir)
        roles_dir = evo_path / "roles"
        roles_dir.mkdir()
        (roles_dir / "planner.py").write_text('''
from palimpsest.runtime import role

@role(
    name="planner",
    description="Test planner",
    min_cost=0.10,
    max_cost=0.50,  # Low max_cost
)
def planner_role(**params):
    pass
''')
        
        state = SupervisorState()
        role_reader = RoleMetadataReader(evo_path)
        handler = SpawnHandler(state, role_reader)
        
        # Add a parent job to the state
        parent_job = SpawnedJob(
            job_id="parent-123",
            source_event_id="evt-1",
            goal="Parent task",
            role="planner",
            repo="https://github.com/org/repo",
            init_branch="main",
            bundle_sha="abc123",
            budget=1.0,
            task_id="parent-task",
            bundle="default",
        )
        state.jobs_by_id[parent_job.job_id] = parent_job
        state.spawn_defaults_by_job[parent_job.job_id] = parent_job
        
        # Try to spawn with budget=10.0 which exceeds max_cost=0.50
        payload = SpawnRequestData(
            job_id="parent-123",
            tasks=[{
                "goal": "Test task",
                "role": "planner",
                "budget": 10.0,  # Exceeds max_cost=0.50
            }]
        )
        
        # Create a proper event object
        from types import SimpleNamespace
        event = SimpleNamespace(id="test-event", data=payload.model_dump())
        
        with pytest.raises(ValueError, match="max_cost"):
            handler.expand(event)


def test_spawn_handler_accepts_budget_within_max_cost():
    """Spawn is accepted if budget is within role's max_cost (ADR-0004 D1a)."""
    from trenni.spawn_handler import SpawnHandler
    from trenni.state import SupervisorState, SpawnedJob
    from yoitsu_contracts.role_metadata import RoleMetadataReader
    from yoitsu_contracts.events import SpawnRequestData

    with tempfile.TemporaryDirectory() as tmpdir:
        evo_path = Path(tmpdir)
        roles_dir = evo_path / "roles"
        roles_dir.mkdir()
        (roles_dir / "implementer.py").write_text('''
from palimpsest.runtime import role

@role(
    name="implementer",
    description="Test implementer",
    min_cost=0.10,
    max_cost=2.00,
)
def implementer_role(**params):
    pass
''')
        
        state = SupervisorState()
        role_reader = RoleMetadataReader(evo_path)
        handler = SpawnHandler(state, role_reader)
        
        parent_job = SpawnedJob(
            job_id="parent-123",
            source_event_id="evt-1",
            goal="Parent task",
            role="implementer",
            repo="https://github.com/org/repo",
            init_branch="main",
            bundle_sha="abc123",
            budget=1.0,
            task_id="parent-task",
            bundle="default",
        )
        state.jobs_by_id[parent_job.job_id] = parent_job
        state.spawn_defaults_by_job[parent_job.job_id] = parent_job
        
        # Spawn with budget=1.5 which is within max_cost=2.00
        payload = SpawnRequestData(
            job_id="parent-123",
            tasks=[{
                "goal": "Test task",
                "role": "implementer",
                "budget": 1.5,  # Within max_cost=2.00
            }]
        )
        
        from types import SimpleNamespace
        event = SimpleNamespace(id="test-event", data=payload.model_dump())
        
        plan = handler.expand(event)
        assert len(plan.jobs) > 0


def test_spawned_job_no_execution_overrides():
    """SpawnedJob must not carry execution config overrides."""
    from trenni.state import SpawnedJob

    job = SpawnedJob(
        job_id="test-123",
        source_event_id="evt-1",
        goal="Do something",
        role="implementer",
        repo="https://github.com/org/repo",
        init_branch="main",
        bundle_sha="abc123",
        task_id="task-1",
        bundle="default",
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
        bundle_sha="abc123",
        bundle="default",
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
        goal="Do something",
        role="planner",
        repo="https://github.com/org/repo",
        init_branch="main",
        bundle_sha="abc123",
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
        bundle_sha="abc123",
        role_params={"mode": "join"},  # role-internal flag
        bundle="default",
    )
    assert defaults.role_params == {"mode": "join"}


def test_spawn_requires_explicit_role():
    """Spawn without role is rejected at validation (canonical contract)."""
    from yoitsu_contracts.events import SpawnRequestData
    import pydantic

    # Spawn without role should fail validation
    with pytest.raises(pydantic.ValidationError, match="role"):
        SpawnRequestData(
            job_id="parent-123",
            tasks=[{
                "goal": "Test task without role",
                # No role specified - should fail
            }]
        )