"""Tests for trigger evaluator scaffold (ADR-0008 D2, D3)."""

from trenni.trigger_evaluator import TriggerEvaluator, TriggerRule


def test_trigger_rule_parsing():
    """TriggerRule parses YAML configuration."""
    rule = TriggerRule(
        name="ci_failure",
        match={"type": "github.ci.completed", "data.conclusion": "failure"},
        accumulate=1,
        spawn_goal="investigate CI failure: {{data.summary}}",
    )
    assert rule.name == "ci_failure"
    assert rule.accumulate == 1


def test_trigger_evaluator_scaffold():
    """TriggerEvaluator exists as scaffold for Phase 2."""
    evaluator = TriggerEvaluator([])
    assert evaluator.rules == []


def test_trigger_evaluator_with_rules():
    """TriggerEvaluator can be initialized with rules."""
    rules = [
        TriggerRule(name="ci_failure", accumulate=20),
        TriggerRule(name="optimization_review", accumulate=10),
    ]
    evaluator = TriggerEvaluator(rules)
    assert len(evaluator.rules) == 2


def test_trigger_evaluator_evaluate_returns_none():
    """TriggerEvaluator.evaluate returns None (scaffold behavior)."""
    evaluator = TriggerEvaluator([
        TriggerRule(name="test", match={"type": "test.event"})
    ])
    # Phase 2: scaffold always returns None
    assert evaluator.evaluate({"type": "test.event"}) is None


def test_trigger_evaluator_reset():
    """TriggerEvaluator.reset clears accumulation counters."""
    evaluator = TriggerEvaluator([])
    evaluator._accumulated["test_rule"] = 5

    evaluator.reset("test_rule")
    assert "test_rule" not in evaluator._accumulated

    evaluator._accumulated["rule1"] = 1
    evaluator._accumulated["rule2"] = 2
    evaluator.reset()
    assert len(evaluator._accumulated) == 0