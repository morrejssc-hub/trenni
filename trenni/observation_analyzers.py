"""Observation analyzers for post-job analysis (ADR-0017).

Analyzers examine job events and produce observation data.
Trenni calls analyzers after job reaches terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

# Event is dict-like in supervisor context
# from yoitsu_contracts.events import Event  -- Event is from pasloe_client, not exported

# Use Any for Event type (it's a SimpleNamespace or dict)

@runtime_checkable
class ObservationAnalyzer(Protocol):
    """Analyzer protocol per ADR-0017.
    
    Analyzers examine job events and return observation data.
    Trenni emits observation.* events from the returned data.
    """
    name: str  # Analyzer identifier (e.g., "tool_repetition", "budget_variance")
    
    def analyze(self, job_events: list[Event], job_id: str, task_id: str, role: str, bundle: str) -> list[dict[str, Any]]:
        """Analyze job events and return observation data.
        
        Args:
            job_events: List of events from this job
            job_id: Job identifier
            task_id: Task identifier  
            role: Role type
            bundle: Bundle name
            
        Returns:
            List of observation data dicts (will be emitted as observation.* events)
        """
        ...


class ToolRepetitionAnalyzer:
    """Analyze tool call patterns for repetition.
    
    Detects when a tool is called multiple times with similar arguments.
    This indicates an abstraction opportunity for bundle evolution.
    """
    name = "tool_repetition"
    
    def analyze(self, job_events: list[dict]) -> list[dict[str, Any]]:
        """Analyze tool call history from job events.
        
        Args:
            job_events: List of events for this job (from pasloe query)
                - agent.tool.exec: {tool_name, arguments_preview, job_id}
                - agent.job.completed: {summary, status}
        
        Returns:
            List of observation data dicts (will be emitted as observation.* events)
        """
        # Extract tool.exec events
        tool_execs = [
            evt for evt in job_events
            if evt.get("type") == "agent.tool.exec"
        ]
        
        if not tool_execs:
            return []
        
        # Group by tool_name, count calls
        tool_counts: dict[str, int] = {}
        for evt in tool_execs:
            tool_name = evt.get("data", {}).get("tool_name", "")
            if tool_name:
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        # Build results for tools called >= 5 times
        results = []
        for tool_name, count in tool_counts.items():
            if count >= 5:
                results.append({
                    "tool_name": tool_name,
                    "call_count": count,
                    "arg_pattern": "",  # Builtin analyzer doesn't analyze args
                    "similarity": 0.0,
                })
        
        return results


class BudgetVarianceAnalyzer:
    """Analyze budget prediction accuracy.
    
    Compares estimated budget with actual cost.
    """
    name = "budget_variance"
    
    def analyze(self, job_events: list[Event], job_id: str, task_id: str, role: str, bundle: str) -> list[dict[str, Any]]:
        """Analyze budget variance from spawn_defaults and job.completed.
        
        Note: This analyzer needs access to spawn_defaults which is in Supervisor.
        In practice, budget_variance is handled directly in supervisor for now.
        
        Returns:
            Budget variance observation (if applicable)
        """
        # Budget variance requires spawn_defaults (estimated budget)
        # which is not in job_events. This analyzer is informational only.
        # Real budget_variance emission stays in supervisor._emit_budget_variance
        return []


# === Analyzer Registry ===

BUILTIN_ANALYZERS: dict[str, ObservationAnalyzer] = {
    "tool_repetition": ToolRepetitionAnalyzer(),
    "budget_variance": BudgetVarianceAnalyzer(),  # Placeholder
}


def get_analyzer(name: str) -> ObservationAnalyzer | None:
    """Get an analyzer by name from registry."""
    return BUILTIN_ANALYZERS.get(name)