"""Tests for optimizer output handling and ReviewProposal closure (ADR-0010)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from datetime import datetime, timezone

from yoitsu_contracts.review_proposal import (
    ReviewProposal,
    ProblemClassification,
    ExecutableProposal,
    TaskTemplate,
    ProblemCategory,
    SeverityLevel,
    ActionType,
    EvidenceEvent,
)
from yoitsu_contracts.events import TriggerData
from yoitsu_contracts.external_events import review_proposal_to_trigger
from trenni.state import SupervisorState, SpawnedJob


class TestOptimizerOutputHandling:
    """Test _handle_optimizer_output method."""

    @pytest.mark.asyncio
    async def test_optimizer_output_parsed_and_spawned(self):
        """Optimizer job output is parsed and spawns optimization task."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        # Create supervisor with mocked client
        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Create optimizer job
        job_id = "optimizer-001"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-threshold",
            goal="Analyze budget variance",
            role="optimizer",
            repo="",
            init_branch="main",
            evo_sha="abc123",
            budget=0.5,
            task_id="task-opt",
            team="default",
        )
        supervisor.state.jobs_by_id[job_id] = job

        # Create proposal JSON
        proposal = ReviewProposal(
            problem_classification=ProblemClassification(
                category=ProblemCategory.BUDGET_ACCURACY,
                severity=SeverityLevel.HIGH,
                summary="Budget variance exceeds threshold",
            ),
            executable_proposal=ExecutableProposal(
                action_type=ActionType.ADJUST_BUDGET,
                description="Increase planner budget by 20%",
                estimated_impact="Reduce variance by 15%",
            ),
            task_template=TaskTemplate(
                goal="Adjust planner budget estimation",
                role="implementer",
                budget=0.3,
            ),
        )
        summary = proposal.model_dump_json()

        # Create completion event
        event = SimpleNamespace(
            id="evt-complete",
            type="agent.job.completed",
            data={"summary": summary, "cost": 0.45},
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output
        await supervisor._handle_optimizer_output(job_id, job, event)

        # Verify task was spawned (client.emit called for task.created)
        # At minimum, supervisor.task.created should be emitted
        assert supervisor.client.emit.called

    @pytest.mark.asyncio
    async def test_non_optimizer_job_not_processed(self):
        """Non-optimizer jobs are not processed for proposal output."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Create planner job (not optimizer)
        job_id = "planner-001"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-001",
            goal="Plan something",
            role="planner",
            repo="https://github.com/org/repo",
            init_branch="main",
            evo_sha="abc123",
            budget=0.5,
            task_id="task-001",
            team="default",
        )
        supervisor.state.jobs_by_id[job_id] = job

        # Create completion event with JSON-like summary
        event = SimpleNamespace(
            id="evt-complete",
            type="agent.job.completed",
            data={"summary": '{"some": "json"}', "cost": 0.3},
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output
        await supervisor._handle_optimizer_output(job_id, job, event)

        # Should not process - no emit calls for task spawning
        # Only planner job, not optimizer
        assert not supervisor.client.emit.called

    @pytest.mark.asyncio
    async def test_invalid_proposal_json_logged_not_crashed(self):
        """Invalid proposal JSON is logged but doesn't crash."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Create optimizer job
        job_id = "optimizer-002"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-threshold",
            goal="Analyze budget variance",
            role="optimizer",
            repo="",
            init_branch="main",
            evo_sha="abc123",
            budget=0.5,
            task_id="task-opt",
            team="default",
        )
        supervisor.state.jobs_by_id[job_id] = job

        # Create completion event with invalid JSON
        event = SimpleNamespace(
            id="evt-complete",
            type="agent.job.completed",
            data={"summary": "not valid json at all", "cost": 0.45},
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output - should not crash
        await supervisor._handle_optimizer_output(job_id, job, event)

        # Should not spawn any task
        assert not supervisor.client.emit.called

    @pytest.mark.asyncio
    async def test_missing_summary_logged_not_crashed(self):
        """Missing summary is logged but doesn't crash."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Create optimizer job
        job_id = "optimizer-003"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-threshold",
            goal="Analyze budget variance",
            role="optimizer",
            repo="",
            init_branch="main",
            evo_sha="abc123",
            budget=0.5,
            task_id="task-opt",
            team="default",
        )
        supervisor.state.jobs_by_id[job_id] = job

        # Create completion event without summary
        event = SimpleNamespace(
            id="evt-complete",
            type="agent.job.completed",
            data={"cost": 0.45},  # No summary
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output - should not crash
        await supervisor._handle_optimizer_output(job_id, job, event)

        # Should not spawn any task
        assert not supervisor.client.emit.called

    @pytest.mark.asyncio
    async def test_proposal_in_markdown_code_block_parsed(self):
        """Proposal embedded in markdown code block is parsed."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Create optimizer job
        job_id = "optimizer-004"
        job = SpawnedJob(
            job_id=job_id,
            source_event_id="evt-threshold",
            goal="Analyze budget variance",
            role="optimizer",
            repo="",
            init_branch="main",
            evo_sha="abc123",
            budget=0.5,
            task_id="task-opt",
            team="default",
        )
        supervisor.state.jobs_by_id[job_id] = job

        # Create proposal in markdown code block (realistic optimizer output)
        summary = """Based on my analysis, the planner role shows budget variance...

Here is my proposal:

```json
{
    "problem_classification": {
        "category": "budget_accuracy",
        "severity": "high",
        "summary": "Planner budget underestimation"
    },
    "executable_proposal": {
        "action_type": "adjust_budget",
        "description": "Increase planner budget by 20%",
        "estimated_impact": "Reduce variance by 15%"
    },
    "task_template": {
        "goal": "Adjust planner budget defaults",
        "role": "implementer",
        "budget": 0.3
    }
}
```
"""

        event = SimpleNamespace(
            id="evt-complete",
            type="agent.job.completed",
            data={"summary": summary, "cost": 0.45},
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output
        await supervisor._handle_optimizer_output(job_id, job, event)

        # Should spawn task
        assert supervisor.client.emit.called


class TestReviewProposalTriggerConversion:
    """Test review_proposal_to_trigger produces valid TriggerData."""

    def test_full_proposal_converts_to_valid_trigger(self):
        """Full proposal with task_template converts to valid TriggerData."""
        proposal = ReviewProposal(
            problem_classification=ProblemClassification(
                category=ProblemCategory.BUDGET_ACCURACY,
                severity=SeverityLevel.HIGH,
                summary="Budget variance issue",
            ),
            evidence_events=[
                EvidenceEvent(
                    event_type="observation.budget_variance",
                    task_id="task-123",
                    job_id="job-456",
                    role="planner",
                    key_metric="variance_ratio=0.35",
                ),
            ],
            executable_proposal=ExecutableProposal(
                action_type=ActionType.ADJUST_BUDGET,
                description="Increase planner budget",
                estimated_impact="Reduce variance",
            ),
            task_template=TaskTemplate(
                goal="Fix planner budget estimation",
                role="implementer",
                budget=0.5,
                repo="https://github.com/org/yoitsu",
                team="backend",
            ),
        )
        trigger_data = review_proposal_to_trigger(proposal)
        data = TriggerData.model_validate(trigger_data)

        assert data.goal == "Fix planner budget estimation"
        assert data.role == "implementer"
        assert data.budget == 0.5
        assert data.repo == "https://github.com/org/yoitsu"
        assert data.team == "backend"
        assert data.params.get("source_review") is True
        assert len(data.params.get("evidence_summary", [])) == 1


class TestEndToEndOptimizationLoop:
    """End-to-end smoke test for autonomous optimization loop."""

    @pytest.mark.asyncio
    async def test_threshold_to_optimizer_to_optimization_task(self):
        """Complete loop: observation_threshold -> optimizer -> proposal -> optimization task.

        This test simulates the full autonomous review loop:
        1. observation_threshold event triggers optimizer task
        2. optimizer job completes with ReviewProposal JSON
        3. proposal parsed and converted to optimization trigger
        4. optimization task spawned
        """
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig
        from yoitsu_contracts.external_events import (
            ObservationThresholdEvent,
            observation_threshold_to_trigger,
        )

        config = TrenniConfig()
        supervisor = Supervisor(config)
        supervisor.client = AsyncMock()

        # Step 1: Observation threshold event
        threshold_event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
            role="planner",
            team="default",
            budget=0.5,
            window_hours=24,
        )

        threshold_trigger = observation_threshold_to_trigger(threshold_event)

        # Create synthetic threshold event
        event = SimpleNamespace(
            id="evt-threshold-001",
            source_id=config.source_id,
            type="external.event",
            data={
                "event_type": "observation_threshold",
                **threshold_event.model_dump(),
            },
            ts=datetime.now(timezone.utc),
        )

        # Process threshold trigger
        trigger_data = TriggerData.model_validate(threshold_trigger)
        await supervisor._process_trigger(event, trigger_data, replay=False)

        # Verify optimizer task was spawned
        # Find the optimizer job in state
        optimizer_jobs = [
            j for j in supervisor.state.jobs_by_id.values()
            if j.role == "optimizer"
        ]
        assert len(optimizer_jobs) >= 1, "Optimizer task should be spawned"

        optimizer_job = optimizer_jobs[0]

        # Step 2: Simulate optimizer job completion with proposal
        proposal = ReviewProposal(
            problem_classification=ProblemClassification(
                category=ProblemCategory.BUDGET_ACCURACY,
                severity=SeverityLevel.HIGH,
                summary=f"Budget variance {threshold_event.current_value} exceeds threshold {threshold_event.threshold}",
            ),
            executable_proposal=ExecutableProposal(
                action_type=ActionType.ADJUST_BUDGET,
                description="Increase planner budget by 20%",
                estimated_impact="Reduce variance to below threshold",
            ),
            task_template=TaskTemplate(
                goal="Adjust planner budget estimation parameters",
                role="implementer",
                budget=0.3,
                team="default",
            ),
        )

        completion_event = SimpleNamespace(
            id="evt-optimizer-complete",
            type="agent.job.completed",
            data={
                "summary": proposal.model_dump_json(),
                "cost": 0.45,
            },
            ts=datetime.now(timezone.utc),
        )

        # Handle optimizer output
        await supervisor._handle_optimizer_output(
            optimizer_job.job_id,
            optimizer_job,
            completion_event,
        )

        # Step 3: Verify optimization task was spawned from proposal
        # Find implementer jobs spawned after optimizer
        implementer_jobs = [
            j for j in supervisor.state.jobs_by_id.values()
            if j.role == "implementer" and "-proposal" in j.source_event_id
        ]
        assert len(implementer_jobs) >= 1, "Optimization implementer task should be spawned"

        # Verify the implementer task has correct goal
        implementer_job = implementer_jobs[0]
        assert "budget" in implementer_job.goal.lower()
        assert implementer_job.team == "default"