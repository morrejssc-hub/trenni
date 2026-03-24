from __future__ import annotations

from dataclasses import replace

from yoitsu_contracts.conditions import evaluate_condition

from .state import SpawnedJob, SupervisorState, TaskRecord

_TASK_TERMINAL_STATES = {"complete", "failed", "cancelled"}


class Scheduler:
    def __init__(self, state: SupervisorState, *, max_workers: int) -> None:
        self.state = state
        self.max_workers = max_workers

    def has_capacity(self) -> bool:
        return len(self.state.running_jobs) < self.max_workers

    def evaluate_job(self, job: SpawnedJob) -> bool | None:
        if job.condition is not None:
            return evaluate_condition(job.condition, self.state.task_states())

        if job.depends_on:
            failed_deps = job.depends_on & self.state.failed_jobs
            if failed_deps:
                return False
            unsatisfied = job.depends_on - self.state.completed_jobs
            return True if not unsatisfied else None

        return True

    async def enqueue(self, job: SpawnedJob) -> list[SpawnedJob]:
        task_id = job.task_id or job.job_id
        if not job.task_id:
            job = replace(job, task_id=task_id)
        self.state.jobs_by_id[job.job_id] = job

        outcome = self.evaluate_job(job)
        if outcome is True:
            await self.state.ready_queue.put(job)
            return []
        if outcome is False:
            return [job]

        self.state.pending_jobs[job.job_id] = job
        return []

    def record_task_submission(
        self,
        *,
        task_id: str,
        goal: str,
        source_event_id: str,
        spec: dict,
    ) -> None:
        self.state.tasks[task_id] = TaskRecord(
            task_id=task_id,
            goal=goal,
            source_event_id=source_event_id,
            spec=spec,
        )

    async def mark_task_terminal(
        self,
        *,
        task_id: str,
        state: str,
    ) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        record = self.state.tasks.get(task_id)
        if record is None:
            return [], []
        
        if record.terminal:
            return [], []

        record.terminal = True
        record.terminal_state = state

        return await self._resolve_pending()

    async def record_job_terminal(
        self,
        *,
        job_id: str,
        summary: str = "",
        failed: bool = False,
        cancelled: bool = False,
    ) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        self.state.completed_jobs.add(job_id)
        if summary:
            self.state.job_summaries[job_id] = summary

        if failed:
            self.state.failed_jobs.add(job_id)
        if cancelled:
            self.state.cancelled_jobs.add(job_id)

        return await self._resolve_pending()



    def status_snapshot(self, *, runtime_kind: str, running: bool, paused: bool) -> dict:
        return {
            "running": running,
            "paused": paused,
            "running_jobs": len(self.state.running_jobs),
            "max_workers": self.max_workers,
            "pending_jobs": len(self.state.pending_jobs),
            "ready_queue_size": self.state.ready_queue.qsize(),
            "runtime_kind": runtime_kind,
            "tasks": self.state.task_states(),
        }

    async def _resolve_pending(self) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        ready: list[SpawnedJob] = []
        cancelled: list[SpawnedJob] = []

        for job_id, job in list(self.state.pending_jobs.items()):
            outcome = self.evaluate_job(job)
            if outcome is True:
                del self.state.pending_jobs[job_id]
                await self.state.ready_queue.put(job)
                ready.append(job)
            elif outcome is False:
                del self.state.pending_jobs[job_id]
                cancelled.append(job)

        return ready, cancelled
