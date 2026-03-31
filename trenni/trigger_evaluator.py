"""Trigger evaluator for external event → task translation (ADR-0008).

Phase 2 scaffold. Full implementation deferred until external event sources
are integrated.

Per ADR-0008:
- Triggers are declarative rules in Trenni's configuration
- When a trigger matches, Trenni creates a normal task
- No special code path needed
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TriggerRule:
    """Declarative trigger rule configuration.

    Per ADR-0008 Issue 3: accumulate uses simple batch semantics.
    Trigger fires after N matching events, then resets counter.
    """

    name: str
    match: dict[str, str] = field(default_factory=dict)
    accumulate: int = 1  # Fire after N matching events
    spawn_goal: str = ""  # Template for task goal


class TriggerEvaluator:
    """Evaluates external events against trigger rules.

    Scaffold for Phase 2. Currently only parses rules, does not evaluate.
    Full implementation will:
    1. Poll new events from Pasloe (existing behavior)
    2. Route task/job events to supervisor state machine (existing behavior)
    3. Evaluate remaining events against trigger rules (new behavior - Phase 2)
    4. Matched triggers create tasks through normal task creation path
    """

    def __init__(self, rules: list[TriggerRule]) -> None:
        self.rules = list(rules)
        self._accumulated: dict[str, int] = {}  # rule_name -> count

    def evaluate(self, event: dict) -> str | None:
        """Check if event matches any trigger rule.

        Returns spawn goal template if matched, None otherwise.

        NOTE: Full implementation deferred to Phase 2.
        This scaffold only validates rule structure.
        """
        # Phase 2: implement pattern matching and accumulation
        return None

    def reset(self, rule_name: str | None = None) -> None:
        """Reset accumulation counters.

        Args:
            rule_name: Specific rule to reset, or None for all rules.
        """
        if rule_name:
            self._accumulated.pop(rule_name, None)
        else:
            self._accumulated.clear()