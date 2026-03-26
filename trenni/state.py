from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from yoitsu_contracts.conditions import Condition
from yoitsu_contracts.config import JobContextConfig
from yoitsu_contracts.events import EvalSpec, TaskResult

from .runtime_types import JobHandle


def _queue_factory() -> asyncio.Queue["SpawnedJob"]:
    return asyncio.Queue()


@dataclass
class SpawnedJob:
    job_id: str
    source_event_id: str
    task: str
    role: str
    repo: str
    init_branch: str
    evo_sha: str | None
    role_params: dict[str, Any] = field(default_factory=dict)
    llm_overrides: dict[str, Any] = field(default_factory=dict)
    workspace_overrides: dict[str, Any] = field(default_factory=dict)
    publication_overrides: dict[str, Any] = field(default_factory=dict)
    depends_on: frozenset[str] = field(default_factory=frozenset)
    task_id: str = ""
    condition: Condition | None = None
    job_context: JobContextConfig = field(default_factory=JobContextConfig)
    parent_job_id: str = ""
    team: str = "default"


@dataclass
class SpawnDefaults:
    repo: str
    init_branch: str
    role: str
    evo_sha: str | None
    role_params: dict[str, Any] = field(default_factory=dict)
    llm_overrides: dict[str, Any] = field(default_factory=dict)
    workspace_overrides: dict[str, Any] = field(default_factory=dict)
    publication_overrides: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    team: str = "default"


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
