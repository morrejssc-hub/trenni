from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from yoitsu_contracts.conditions import Condition, condition_from_data
from yoitsu_contracts.config import JobContextConfig
from yoitsu_contracts.events import EvalSpec, TaskResult

from .runtime_types import JobHandle


def _queue_factory() -> asyncio.Queue["SpawnedJob"]:
    return asyncio.Queue()


@dataclass
class SpawnedJob:
    """Job specification passed from spawn_handler to runtime_builder.

    Contains only task semantics and runtime identity.
    Execution config (llm, workspace, publication) is derived from role definition.

    Fields:
        job_id: Unique identifier assigned by Trenni.
        source_event_id: The spawn event that created this job.
        goal: The goal text (authoritative task description).
        role: Role type to execute this task.
        repo: Repository URL.
        init_branch: Starting branch.
        evo_sha: Git SHA for evo version.
        budget: Maximum cost for this task (task semantics field).
        role_params: Role-internal behavior flags only (e.g. mode="join").
        depends_on: Job IDs this job waits for.
        task_id: Reference to the task this job belongs to.
        condition: Optional condition for conditional execution.
        job_context: Join/eval context configuration.
        parent_job_id: Job ID of the spawning parent.
        team: Team that owns this task (inherited, not overridable).
    """

    job_id: str
    source_event_id: str
    goal: str
    role: str
    repo: str
    init_branch: str
    evo_sha: str | None
    budget: float = 0.0
    role_params: dict[str, Any] = field(default_factory=dict)
    depends_on: frozenset[str] = field(default_factory=frozenset)
    task_id: str = ""
    condition: Condition | None = None
    job_context: JobContextConfig = field(default_factory=JobContextConfig)
    parent_job_id: str = ""
    team: str = "default"

    def to_enqueued_data(self, queue_state: str, condition_data: dict | None) -> dict[str, Any]:
        """Generate SupervisorJobEnqueuedData dict from this job."""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id or self.job_id,
            "source_event_id": self.source_event_id,
            "goal": self.goal,
            "role": self.role,
            "role_params": dict(self.role_params),
            "team": self.team,
            "repo": self.repo,
            "init_branch": self.init_branch,
            "evo_sha": self.evo_sha or "",
            "budget": self.budget,
            "parent_job_id": self.parent_job_id,
            "condition": condition_data,
            "job_context": self.job_context.model_dump(mode="json"),
            "queue_state": queue_state,
        }

    def to_launched_data(self, runtime_kind: str, container_id: str, container_name: str, condition_data: dict | None) -> dict[str, Any]:
        """Generate SupervisorJobLaunchedData dict from this job."""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id or self.job_id,
            "source_event_id": self.source_event_id,
            "goal": self.goal,
            "role": self.role,
            "role_params": dict(self.role_params),
            "team": self.team,
            "repo": self.repo,
            "init_branch": self.init_branch,
            "evo_sha": self.evo_sha or "",
            "budget": self.budget,
            "runtime_kind": runtime_kind,
            "container_id": container_id,
            "container_name": container_name,
            "parent_job_id": self.parent_job_id,
            "condition": condition_data,
        }

    @classmethod
    def from_enqueued_data(cls, data: dict[str, Any]) -> "SpawnedJob":
        """Reconstruct SpawnedJob from SupervisorJobEnqueuedData dict."""
        return cls(
            job_id=data["job_id"],
            source_event_id=data["source_event_id"],
            goal=data["goal"],
            role=data["role"],
            role_params=dict(data.get("role_params", {})),
            team=data.get("team", "default"),
            repo=data["repo"],
            init_branch=data["init_branch"],
            evo_sha=data.get("evo_sha") or None,
            budget=data.get("budget", 0.0),
            task_id=data.get("task_id") or data["job_id"],
            condition=condition_from_data(data.get("condition")),
            parent_job_id=data.get("parent_job_id", ""),
            job_context=JobContextConfig.model_validate(data.get("job_context", {})),
        )


@dataclass
class SpawnDefaults:
    """Default values inherited by spawned jobs.

    Per ADR-0007, this contains only task semantics defaults.
    Execution config defaults come from TrenniConfig and role definition.

    Removed per ADR-0007:
        llm_overrides, workspace_overrides, publication_overrides
    """

    repo: str
    init_branch: str
    role: str
    evo_sha: str | None
    role_params: dict[str, Any] = field(default_factory=dict)  # only role-internal flags
    task_id: str = ""
    team: str = "default"
    budget: float = 0.0  # ADR-0010: for budget_variance observation


@dataclass
class TeamLaunchCondition:
    """Launch condition checking team concurrent job limit.

    Per ADR-0011 D5: per-team launch conditions, not scheduling policy.
    The scheduler evaluates this condition alongside existing depends_on and
    structural conditions. The scheduler itself has no team awareness.
    """
    team: str
    max_concurrent: int

    def is_satisfied(self, state: SupervisorState) -> bool:
        """Check if team has room for another concurrent job.

        Returns True if:
        - max_concurrent <= 0 (no limit)
        - running_count < max_concurrent
        """
        if self.max_concurrent <= 0:
            return True  # No limit
        return state.running_count_for_team(self.team) < self.max_concurrent


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    state: str = "pending"
    terminal: bool = False
    terminal_state: str = ""
    source_event_id: str = ""
    spec: dict = field(default_factory=dict)
    team: str = "default"
    eval_spec: EvalSpec | None = None
    eval_spawned: bool = False
    eval_job_id: str = ""
    result: TaskResult | None = None
    job_order: list[str] = field(default_factory=list)

@dataclass
class SupervisorState:
    event_cursor: str | None = None
    running_jobs: dict[str, JobHandle] = field(default_factory=dict)
    ready_queue: asyncio.Queue[SpawnedJob] = field(default_factory=_queue_factory)
    pending_jobs: dict[str, SpawnedJob] = field(default_factory=dict)
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    jobs_by_id: dict[str, SpawnedJob] = field(default_factory=dict)
    completed_jobs: set[str] = field(default_factory=set)
    failed_jobs: set[str] = field(default_factory=set)
    cancelled_jobs: set[str] = field(default_factory=set)
    job_summaries: dict[str, str] = field(default_factory=dict)
    job_git_refs: dict[str, str] = field(default_factory=dict)
    job_completion_codes: dict[str, str] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)
    launched_event_ids: set[str] = field(default_factory=set)
    spawn_defaults_by_job: dict[str, SpawnDefaults] = field(default_factory=dict)
    running_jobs_by_team: dict[str, int] = field(default_factory=dict)

    def increment_team_running(self, team: str) -> None:
        """Increment running job count for a team."""
        self.running_jobs_by_team[team] = self.running_jobs_by_team.get(team, 0) + 1

    def decrement_team_running(self, team: str) -> None:
        """Decrement running job count for a team (does not go below 0)."""
        current = self.running_jobs_by_team.get(team, 0)
        if current > 0:
            self.running_jobs_by_team[team] = current - 1

    def running_count_for_team(self, team: str) -> int:
        """Get running job count for a team (returns 0 if team not present)."""
        return self.running_jobs_by_team.get(team, 0)

    def replay_team_counts(self, running_jobs: list[tuple[str, str]]) -> None:
        """Rebuild team running counts from running jobs after restart.

        Args:
            running_jobs: List of (job_id, team) tuples for jobs that were
                          running before restart.

        This is called during replay to restore accurate team capacity
        tracking without double-counting jobs that are already running.
        """
        for job_id, team in running_jobs:
            self.increment_team_running(team)

    def task_states(self) -> dict[str, str]:
        return {
            task_id: (record.terminal_state if record.terminal else (record.state or "pending"))
            for task_id, record in self.tasks.items()
        }

    def snapshot(self) -> dict:
        return {
            "cursor": self.event_cursor,
            "running_jobs": list(self.running_jobs.keys()),
            "pending_jobs": list(self.pending_jobs.keys()),
            "ready_queue_size": self.ready_queue.qsize(),
            "tasks": self.task_states(),
        }

    def remove_pending_job(self, job_id: str) -> SpawnedJob | None:
        return self.pending_jobs.pop(job_id, None)

    def ready_queue_snapshot(self) -> list[SpawnedJob]:
        items: list[SpawnedJob] = []
        while True:
            try:
                items.append(self.ready_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for item in items:
            self.ready_queue.put_nowait(item)
        return items

    def has_ready_job(self, job_id: str) -> bool:
        return any(job.job_id == job_id for job in self.ready_queue_snapshot())

    def drop_from_ready_queue(self, job_id: str) -> None:
        survivors: list[SpawnedJob] = []
        while True:
            try:
                job = self.ready_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job.job_id != job_id:
                survivors.append(job)

        for job in survivors:
            self.ready_queue.put_nowait(job)

    def discard_scheduled_by_source_event(self, source_event_id: str, *, keep_job_id: str = "") -> None:
        if not source_event_id:
            return

        for job_id, job in list(self.pending_jobs.items()):
            if job.source_event_id == source_event_id and job_id != keep_job_id:
                del self.pending_jobs[job_id]

        survivors: list[SpawnedJob] = []
        while True:
            try:
                job = self.ready_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job.source_event_id == source_event_id and job.job_id != keep_job_id:
                continue
            survivors.append(job)

        for job in survivors:
            self.ready_queue.put_nowait(job)

        for job_id, job in list(self.jobs_by_id.items()):
            if job.source_event_id == source_event_id and job_id != keep_job_id and job_id not in self.running_jobs:
                del self.jobs_by_id[job_id]
