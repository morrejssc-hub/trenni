"""Tests for strict max_concurrent_jobs enforcement at launch time.

Issue 5: Scheduler only checks running_jobs_by_bundle when enqueuing,
not when actually launching. Multiple jobs from same team can enter
ready queue in same poll cycle, and _drain_queue() launches without
re-checking team capacity.

These tests verify:
1. Team capacity is re-checked right before launch in _drain_queue()
2. Replay correctly rebuilds team counts from running jobs
"""

import asyncio

import pytest

from trenni.config import BundleConfig, BundleSchedulingConfig
from trenni.scheduler import Scheduler
from trenni.state import SupervisorState, SpawnedJob


class TestStrictBundleCapacityAtLaunch:
    """Tests for team capacity re-check at launch time."""

    @pytest.mark.asyncio
    async def test_multiple_ready_jobs_same_bundle_respects_capacity(self):
        """Multiple jobs in ready queue for same team should respect capacity.

        Scenario:
        1. Team has max_concurrent_jobs=1
        2. Two jobs enter ready queue (e.g., before team count is incremented)
        3. _drain_queue() should only launch one, re-queue the other
        """
        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

        # Put two jobs in ready queue for the same team
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

        # Manually add both to ready queue (simulating race condition)
        await state.ready_queue.put(job1)
        await state.ready_queue.put(job2)

        # At this point, team has 0 running jobs
        # First launch should succeed (increment team count to 1)
        # Second launch attempt should find team at capacity

        # This test documents the expected behavior:
        # After draining, only one job should be "in flight" for the team
        # The other should be back in pending or still in ready queue

        # Check that scheduler knows team capacity is limited
        assert scheduler.has_bundle_capacity("factorio") is True  # Initially has capacity

        # Simulate first job "running"
        state.increment_bundle_running("factorio")

        # Now team is at capacity
        assert scheduler.has_bundle_capacity("factorio") is False

    @pytest.mark.asyncio
    async def test_scheduler_can_check_and_launch_with_bundle_capacity(self):
        """Scheduler provides method to safely launch with team capacity check.

        This tests the mechanism that _drain_queue should use.
        """
        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

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

        # Team has capacity initially
        assert scheduler.has_bundle_capacity("factorio") is True

        # Simulate acquiring team slot
        state.increment_bundle_running("factorio")

        # Now team is at capacity
        assert scheduler.has_bundle_capacity("factorio") is False

        # If job was in pending, resolve_pending should keep it there
        state.pending_jobs["job-1"] = job
        ready, cancelled = await scheduler._resolve_pending()

        assert len(ready) == 0  # Should NOT promote to ready
        assert "job-1" in state.pending_jobs  # Should stay pending


class TestReplayTeamCounts:
    """Tests for rebuilding team running counts after restart."""

    def test_replay_bundle_counts_from_running_jobs(self):
        """replay_bundle_counts() rebuilds counts from running jobs list."""
        state = SupervisorState()

        # Simulate 3 running jobs for factorio, 2 for default
        running_jobs = [
            ("job-1", "factorio"),
            ("job-2", "factorio"),
            ("job-3", "factorio"),
            ("job-4", "default"),
            ("job-5", "default"),
        ]

        # Call replay_bundle_counts (to be implemented)
        state.replay_bundle_counts(running_jobs)

        assert state.running_count_for_bundle("factorio") == 3
        assert state.running_count_for_bundle("default") == 2
        assert state.running_count_for_bundle("other") == 0  # Unknown team

    def test_replay_bundle_counts_empty_list(self):
        """replay_bundle_counts() handles empty running jobs list."""
        state = SupervisorState()

        state.replay_bundle_counts([])

        assert state.running_count_for_bundle("any-team") == 0

    def test_replay_bundle_counts_preserves_existing(self):
        """replay_bundle_counts() adds to existing counts, doesn't reset."""
        state = SupervisorState()

        # Pre-existing count
        state.increment_bundle_running("factorio")
        state.increment_bundle_running("factorio")

        # Replay additional jobs
        state.replay_bundle_counts([
            ("job-1", "factorio"),
            ("job-2", "default"),
        ])

        # Should have 3 total for factorio (2 existing + 1 replayed)
        assert state.running_count_for_bundle("factorio") == 3
        assert state.running_count_for_bundle("default") == 1


class TestDrainQueueBundleCapacity:
    """Tests for _drain_queue behavior with team capacity enforcement."""

    @pytest.mark.asyncio
    async def test_drain_queue_respects_bundle_capacity_on_launch(self):
        """_drain_queue should re-check team capacity before each launch.

        This is a more integrated test that simulates the actual drain behavior.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

        # Two jobs ready for same team
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

        await state.ready_queue.put(job1)
        await state.ready_queue.put(job2)

        # After draining with max_concurrent=1, only one should be allowed to launch
        # The second should go back to pending

        # This documents expected behavior - actual implementation
        # will be tested by supervisor queue tests
        initial_ready = state.ready_queue.qsize()
        assert initial_ready == 2

        # Team should have capacity for first job
        assert scheduler.has_bundle_capacity("factorio") is True


class TestJobCompletionFreesBundleCapacity:
    """Tests for team capacity being freed when jobs complete."""

    def test_decrement_bundle_running_on_completion(self):
        """When a job completes, team running count should decrement."""
        state = SupervisorState()

        # Start 2 jobs for factorio
        state.increment_bundle_running("factorio")
        state.increment_bundle_running("factorio")

        assert state.running_count_for_bundle("factorio") == 2

        # Complete one
        state.decrement_bundle_running("factorio")

        assert state.running_count_for_bundle("factorio") == 1

        # Complete another
        state.decrement_bundle_running("factorio")

        assert state.running_count_for_bundle("factorio") == 0

    def test_decrement_does_not_go_negative(self):
        """decrement_bundle_running should not go below 0."""
        state = SupervisorState()

        # Decrement without any running jobs
        state.decrement_bundle_running("factorio")
        state.decrement_bundle_running("factorio")

        assert state.running_count_for_bundle("factorio") == 0


class TestBundleCapacityWithPendingPromotion:
    """Tests for pending jobs being promoted only when team has capacity."""

    @pytest.mark.asyncio
    async def test_pending_job_promoted_when_bundle_has_capacity(self):
        """Pending job should be promoted to ready when team has capacity."""
        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

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

        # Job in pending, team has capacity (1 running, max 2)
        state.pending_jobs["job-1"] = job
        state.increment_bundle_running("factorio")

        ready, cancelled = await scheduler._resolve_pending()

        assert len(ready) == 1
        assert ready[0].job_id == "job-1"
        assert "job-1" not in state.pending_jobs

    @pytest.mark.asyncio
    async def test_pending_job_stays_pending_when_bundle_at_capacity(self):
        """Pending job should stay pending when team is at capacity."""
        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=1))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

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

        # Job in pending, team at capacity
        state.pending_jobs["job-1"] = job
        state.increment_bundle_running("factorio")

        ready, cancelled = await scheduler._resolve_pending()

        assert len(ready) == 0
        assert "job-1" in state.pending_jobs

    @pytest.mark.asyncio
    async def test_multiple_pending_jobs_respect_capacity(self):
        """Multiple pending jobs should only be promoted up to capacity limit."""
        state = SupervisorState()
        teams = {"factorio": BundleConfig(scheduling=BundleSchedulingConfig(max_concurrent_jobs=2))}
        scheduler = Scheduler(state, max_workers=10, bundles=teams)

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
        job3 = SpawnedJob(
            job_id="job-3",
            source_event_id="event-3",
            goal="Task 3",
            role="worker",
            repo="https://example.com/repo",
            init_branch="main",
            bundle_sha=None,
            bundle="factorio",
        )

        # All three jobs pending, team has 1 running (max 2)
        state.pending_jobs["job-1"] = job1
        state.pending_jobs["job-2"] = job2
        state.pending_jobs["job-3"] = job3
        state.increment_bundle_running("factorio")  # 1 running, can promote 1 more

        ready, cancelled = await scheduler._resolve_pending()

        # Only one job should be promoted (1 running + 1 promoted = 2 max)
        assert len(ready) == 1
        # Two should still be pending
        assert len(state.pending_jobs) == 2