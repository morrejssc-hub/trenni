"""Tests for observation threshold event handling (ADR-0010)."""
from __future__ import annotations

import pytest

from yoitsu_contracts.external_events import (
    ObservationThresholdEvent,
    observation_threshold_to_trigger,
)
from yoitsu_contracts.events import TriggerData
from yoitsu_contracts.review_proposal import (
    ReviewProposal,
    ProblemClassification,
    ExecutableProposal,
    ProblemCategory,
    SeverityLevel,
    ActionType,
)


class TestObservationThresholdEventHandling:
    """Test observation_threshold event conversion and processing."""

    def test_event_model_validation(self):
        """ObservationThresholdEvent validates correctly."""
        event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
            role="planner",
            team="default",
            budget=0.5,
            window_hours=24,
        )
        assert event.metric_type == "budget_variance"
        assert event.threshold == 0.3
        assert event.current_value == 0.45
        assert event.role == "planner"
        assert event.source == "observation_threshold"
        assert event.event_type == "observation_threshold"

    def test_trigger_conversion_creates_optimizer_task(self):
        """Observation threshold triggers optimizer role."""
        event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
            role="planner",
        )
        trigger = observation_threshold_to_trigger(event)
        assert trigger["role"] == "optimizer"
        assert trigger["trigger_type"] == "observation_threshold"
        assert "budget_variance" in trigger["goal"]

    def test_trigger_data_validation(self):
        """Converted trigger passes TriggerData validation."""
        event = ObservationThresholdEvent(
            metric_type="preparation_failure",
            threshold=0.1,
            current_value=0.15,
            team="backend",
            budget=0.4,
        )
        trigger = observation_threshold_to_trigger(event)
        # Validate as TriggerData
        data = TriggerData.model_validate(trigger)
        assert data.role == "optimizer"
        assert data.team == "backend"
        assert data.budget == 0.4
        assert "preparation_failure" in data.goal

    def test_no_role_in_goal(self):
        """Trigger without role has generic goal."""
        event = ObservationThresholdEvent(
            metric_type="tool_retry",
            threshold=0.2,
            current_value=0.25,
        )
        trigger = observation_threshold_to_trigger(event)
        assert "role" not in trigger["goal"]
        assert trigger["params"]["trigger_role"] is None

    def test_window_hours_preserved(self):
        """Window hours passed to params."""
        event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
            window_hours=12,
        )
        trigger = observation_threshold_to_trigger(event)
        assert trigger["params"]["window_hours"] == 12

    def test_metric_values_in_params(self):
        """Threshold and current value in params."""
        event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
        )
        trigger = observation_threshold_to_trigger(event)
        assert trigger["params"]["threshold"] == 0.3
        assert trigger["params"]["current_value"] == 0.45
        assert trigger["params"]["metric_type"] == "budget_variance"


class TestReviewProposalIntegration:
    """Test ReviewProposal integration with trigger flow."""

    def test_proposal_to_trigger_flow(self):
        """ReviewProposal can be converted to trigger for optimization."""
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
        )
        trigger = proposal.to_trigger_data()
        # Validate as TriggerData
        data = TriggerData.model_validate(trigger)
        assert data.role == "implementer"
        assert data.params.get("source_review") is True
        assert data.params.get("problem_category") == "budget_accuracy"

    def test_threshold_then_proposal_flow(self):
        """Complete flow: threshold -> optimizer -> proposal -> optimization."""
        # 1. Threshold exceeded
        threshold_event = ObservationThresholdEvent(
            metric_type="budget_variance",
            threshold=0.3,
            current_value=0.45,
            role="planner",
            budget=0.5,
        )

        # 2. Convert to trigger for optimizer
        optimizer_trigger = observation_threshold_to_trigger(threshold_event)
        optimizer_data = TriggerData.model_validate(optimizer_trigger)
        assert optimizer_data.role == "optimizer"

        # 3. Optimizer produces proposal (simulated)
        proposal = ReviewProposal(
            problem_classification=ProblemClassification(
                category=ProblemCategory.BUDGET_ACCURACY,
                severity=SeverityLevel.HIGH,
                summary=f"Budget variance {threshold_event.current_value} exceeds threshold {threshold_event.threshold}",
            ),
            executable_proposal=ExecutableProposal(
                action_type=ActionType.ADJUST_BUDGET,
                description="Adjust planner budget estimation",
                estimated_impact="Reduce variance to below threshold",
            ),
        )

        # 4. Proposal converts to optimization trigger
        optimization_trigger = proposal.to_trigger_data()
        optimization_data = TriggerData.model_validate(optimization_trigger)
        assert optimization_data.role == "implementer"