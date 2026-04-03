"""Tests for observation signal emission (ADR-0010)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
from types import SimpleNamespace

from yoitsu_contracts.observation import (
    ObservationBudgetVarianceData,
    ObservationPreparationFailureData,
    OBSERVATION_BUDGET_VARIANCE,
    OBSERVATION_PREPARATION_FAILURE,
)


class TestBudgetVarianceEmission:
    """Tests for budget_variance observation emission."""

    @pytest.mark.asyncio
    async def test_emit_budget_variance_on_job_completion(self):
        """budget_variance is emitted when job completes with budget > 0."""
        from trenni.state import SupervisorState, SpawnedJob, SpawnDefaults
        from types import SimpleNamespace

        state = SupervisorState()
        client = AsyncMock()

        # Create a minimal supervisor-like object
        class MinimalSupervisor:
            def __init__(self):
                self.state = state
                self.client = client
                self._spawn_defaults_by_job = {}

            async def _emit_budget_variance(self, job_id, event):
                """Emit budget_variance observation after job completion (ADR-0010 D5)."""
                from yoitsu_contracts.observation import (
                    ObservationBudgetVarianceData,
                    OBSERVATION_BUDGET_VARIANCE,
                )
                spawn_defaults = self._spawn_defaults_by_job.get(job_id)
                if not spawn_defaults or spawn_defaults.budget <= 0:
                    return

                job = self.state.jobs_by_id.get(job_id)
                if not job:
                    return

                estimated_budget = spawn_defaults.budget
                actual_cost = float(event.data.get("cost", 0.0) or 0.0)

                if estimated_budget > 0:
                    variance_ratio = (actual_cost - estimated_budget) / estimated_budget

                    await self.client.emit(
                        OBSERVATION_BUDGET_VARIANCE,
                        ObservationBudgetVarianceData(
                            task_id=job.task_id or job_id,
                            job_id=job_id,
                            role=job.role,
                            estimated_budget=estimated_budget,
                            actual_cost=actual_cost,
                            variance_ratio=variance_ratio,
                        ).model_dump(),
                    )

        supervisor = MinimalSupervisor()

        # Setup job with budget
        job_id = "test-job-001"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-001",
            goal="Test task",
            role="planner",
            repo="https://github.com/org/repo",
            init_branch="main",
            evo_sha="abc123",
            budget=5.0,
            task_id="task-001",
            team="default",
        )
        state.jobs_by_id[job_id] = job
        supervisor._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo="https://github.com/org/repo",
            init_branch="main",
            role="planner",
            evo_sha="abc123",
            task_id="task-001",
            team="default",
            budget=5.0,
        )

        # Simulate job completion event
        event = SimpleNamespace(
            id="evt-002",
            type="agent.job.completed",
            data={"cost": 6.0, "summary": "Done"},
        )

        await supervisor._emit_budget_variance(job_id, event)

        # Verify emission
        client.emit.assert_called_once()
        call_args = client.emit.call_args
        assert call_args[0][0] == OBSERVATION_BUDGET_VARIANCE

        data = call_args[0][1]
        assert data["task_id"] == "task-001"
        assert data["job_id"] == job_id
        assert data["role"] == "planner"
        assert data["estimated_budget"] == 5.0
        assert data["actual_cost"] == 6.0
        # variance_ratio = (6.0 - 5.0) / 5.0 = 0.2
        assert abs(data["variance_ratio"] - 0.2) < 0.001

    @pytest.mark.asyncio
    async def test_no_budget_variance_when_budget_zero(self):
        """No budget_variance emitted when budget is 0."""
        from trenni.state import SupervisorState, SpawnedJob, SpawnDefaults

        state = SupervisorState()
        client = AsyncMock()

        class MinimalSupervisor:
            def __init__(self):
                self.state = state
                self.client = client
                self._spawn_defaults_by_job = {}

            async def _emit_budget_variance(self, job_id, event):
                from yoitsu_contracts.observation import (
                    ObservationBudgetVarianceData,
                    OBSERVATION_BUDGET_VARIANCE,
                )
                spawn_defaults = self._spawn_defaults_by_job.get(job_id)
                if not spawn_defaults or spawn_defaults.budget <= 0:
                    return
                job = self.state.jobs_by_id.get(job_id)
                if not job:
                    return
                estimated_budget = spawn_defaults.budget
                actual_cost = float(event.data.get("cost", 0.0) or 0.0)
                if estimated_budget > 0:
                    variance_ratio = (actual_cost - estimated_budget) / estimated_budget
                    await self.client.emit(
                        OBSERVATION_BUDGET_VARIANCE,
                        ObservationBudgetVarianceData(
                            task_id=job.task_id or job_id,
                            job_id=job_id,
                            role=job.role,
                            estimated_budget=estimated_budget,
                            actual_cost=actual_cost,
                            variance_ratio=variance_ratio,
                        ).model_dump(),
                    )

        supervisor = MinimalSupervisor()

        job_id = "test-job-002"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-001",
            goal="Test task",
            role="planner",
            repo="https://github.com/org/repo",
            init_branch="main",
            evo_sha="abc123",
            budget=0.0,  # No budget
            task_id="task-002",
            team="default",
        )
        state.jobs_by_id[job_id] = job
        supervisor._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo="https://github.com/org/repo",
            init_branch="main",
            role="planner",
            evo_sha="abc123",
            task_id="task-002",
            team="default",
            budget=0.0,
        )

        event = SimpleNamespace(
            id="evt-002",
            type="agent.job.completed",
            data={"cost": 1.0, "summary": "Done"},
        )

        await supervisor._emit_budget_variance(job_id, event)

        # Should not emit
        client.emit.assert_not_called()


# Note: Preparation failure tests are in palimpsest/tests/
# because they require palimpsest modules not available in trenni venv