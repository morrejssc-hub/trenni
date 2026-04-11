"""Tests for Scheduler BundleLaunchCondition integration.

ADR-0011 D5: Scheduler checks BundleLaunchCondition before launching jobs.
"""

import pytest


def test_scheduler_accepts_teams_config():
    """Scheduler accepts teams configuration in constructor."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {
        "factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2)),
        "default": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=0)),  # unlimited
    }

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    assert scheduler.bundles == teams


def test_scheduler_teams_defaults_to_empty_dict():
    """Scheduler defaults teams to empty dict if not provided."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState

    state = SupervisorState()
    scheduler = Scheduler(state, max_workers=10)

    assert scheduler.bundles == {}


def test_scheduler_has_bundle_capacity_for_unlimited_team():
    """Scheduler allows launch when team has max_concurrent_jobs=0 (unlimited)."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=0))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # Team has 5 running jobs but max_concurrent=0 (unlimited)
    for _ in range(5):
        state.increment_bundle_running("factorio")

    assert scheduler.has_bundle_capacity("factorio") is True


def test_scheduler_has_bundle_capacity_when_below_limit():
    """Scheduler allows launch when team running count is below max."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=3))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # 2 running, max 3 -> has capacity
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")

    assert scheduler.has_bundle_capacity("factorio") is True


def test_scheduler_has_bundle_capacity_when_at_limit():
    """Scheduler denies launch when team running count equals max."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # 2 running, max 2 -> no capacity
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")

    assert scheduler.has_bundle_capacity("factorio") is False


def test_scheduler_has_bundle_capacity_for_unknown_team():
    """Scheduler allows launch for team not in config (no limit set)."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState

    state = SupervisorState()
    scheduler = Scheduler(state, max_workers=10, bundles={})

    # Unknown team has no limit
    assert scheduler.has_bundle_capacity("unknown-team") is True


def test_scheduler_has_bundle_capacity_independent_per_team():
    """Team capacity checks are independent per team."""
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {
        "factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1)),
        "default": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2)),
    }

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # Both teams at capacity
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("default")
    state.increment_bundle_running("default")

    assert scheduler.has_bundle_capacity("factorio") is False  # 1/1
    assert scheduler.has_bundle_capacity("default") is False  # 2/2
    assert scheduler.has_bundle_capacity("other") is True  # not configured


def test_scheduler_enqueue_checks_bundle_capacity():
    """enqueue() checks team capacity before moving to ready queue."""
    import asyncio
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState, SpawnedJob
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # First job should go to ready queue
    job1 = SpawnedJob(
        job_id="job-1",
        source_event_id="event-1",
        goal="Task 1",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="factorio",
    )

    cancelled = asyncio.run(scheduler.enqueue(job1))
    assert len(cancelled) == 0
    assert state.ready_queue.qsize() == 1

    # Simulate first job running
    state.running_jobs["job-1"] = None  # type: ignore
    state.increment_bundle_running("factorio")

    # Second job for same team should stay in pending (not ready)
    job2 = SpawnedJob(
        job_id="job-2",
        source_event_id="event-2",
        goal="Task 2",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="factorio",
    )

    cancelled = asyncio.run(scheduler.enqueue(job2))
    assert len(cancelled) == 0
    assert state.ready_queue.qsize() == 1  # Still just job-1
    assert "job-2" in state.pending_jobs  # job-2 is pending, not ready


def test_scheduler_enqueue_different_bundles_independent():
    """enqueue() treats different teams independently for capacity."""
    import asyncio
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState, SpawnedJob
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {
        "factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1)),
        "default": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1)),
    }

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # Launch factorio job
    job1 = SpawnedJob(
        job_id="job-1",
        source_event_id="event-1",
        goal="Task 1",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="factorio",
    )
    asyncio.run(scheduler.enqueue(job1))
    state.running_jobs["job-1"] = None  # type: ignore
    state.increment_bundle_running("factorio")

    # Default team job should still be ready (different team)
    job2 = SpawnedJob(
        job_id="job-2",
        source_event_id="event-2",
        goal="Task 2",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="default",
    )

    cancelled = asyncio.run(scheduler.enqueue(job2))
    assert len(cancelled) == 0
    assert state.ready_queue.qsize() == 2  # Both jobs ready


def test_scheduler_resolve_pending_respects_bundle_capacity():
    """_resolve_pending checks team capacity when promoting jobs."""
    import asyncio
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState, SpawnedJob
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # Put job in pending (simulating condition that was previously unsatisfied)
    job = SpawnedJob(
        job_id="job-1",
        source_event_id="event-1",
        goal="Task 1",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="factorio",
    )
    state.pending_jobs["job-1"] = job

    # Team is at capacity
    state.increment_bundle_running("factorio")

    # Resolve pending - job should NOT move to ready
    ready, cancelled = asyncio.run(scheduler._resolve_pending())

    assert len(ready) == 0
    assert len(cancelled) == 0
    assert "job-1" in state.pending_jobs  # Still pending


def test_scheduler_resolve_pending_promotes_when_bundle_has_capacity():
    """_resolve_pending promotes jobs when team has capacity."""
    import asyncio
    from trenni.scheduler import Scheduler
    from trenni.state import SupervisorState, SpawnedJob
    from trenni.config import BundleConfig, BundleSchedulingConfig

    state = SupervisorState()
    teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2))}

    scheduler = Scheduler(state, max_workers=10, bundles=teams)

    # Put job in pending
    job = SpawnedJob(
        job_id="job-1",
        source_event_id="event-1",
        goal="Task 1",
        role="worker",
        repo="https://example.com/repo",
        init_branch="main",
        bundle_sha=None,
        bundle="factorio",
    )
    state.pending_jobs["job-1"] = job

    # Team has capacity (1 running, max 2)
    state.increment_bundle_running("factorio")

    # Resolve pending - job should move to ready
    ready, cancelled = asyncio.run(scheduler._resolve_pending())

    assert len(ready) == 1
    assert ready[0].job_id == "job-1"
    assert "job-1" not in state.pending_jobs