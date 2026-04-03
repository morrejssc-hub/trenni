"""Core supervisor: event routing + scheduler + isolation backend."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yoitsu_contracts.observation import (
    ObservationBudgetVarianceData,
    OBSERVATION_BUDGET_VARIANCE,
)
from yoitsu_contracts.conditions import condition_from_data, condition_to_data
from yoitsu_contracts.config import EvalContextConfig, JobContextConfig, JoinContextConfig
from yoitsu_contracts.events import (
    EvalSpec,
    JobCancelledData,
    JobCompletedData,
    JobFailedData,
    SemanticVerdict,
    SpawnRequestData,
    StructuralVerdict,
    SupervisorCheckpointData,
    SupervisorJobEnqueuedData,
    SupervisorJobLaunchedData,
    TaskCancelledData,
    TaskCompletedData,
    TaskCreatedData,
    TaskEvalFailedData,
    TaskEvaluatingData,
    TaskPartialData,
    TaskFailedData,
    TaskResult,
    TaskTraceEntry,
    TriggerData,
)
from yoitsu_contracts.role_metadata import RoleMetadataReader

from .checkpoint import mark_exited_jobs, reap_timed_out_jobs
from .config import TrenniConfig
from .pasloe_client import Event, PasloeClient
from .podman_backend import PodmanBackend
from .replay import rebuild_state
from .runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from .runtime_types import ContainerState, JobHandle
from .scheduler import Scheduler
from .spawn_handler import SpawnHandler
from .state import SpawnDefaults, SpawnedJob, SupervisorState, TaskRecord

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_CYCLES = 30
_CONTROL_EVENT_TIMEOUT_S = 1.0
_DEFAULT_TEAM_DEFINITION = SimpleNamespace(
    name="default",
    description="Default planning and execution team",
    roles=["default"],
    planner_role="planner",
    eval_role="evaluator",
)


class Supervisor:
    def __init__(self, config: TrenniConfig) -> None:
        self.config = config
        self.client = PasloeClient(
            base_url=config.pasloe_url,
            api_key_env=config.pasloe_api_key_env,
            source_id=config.source_id,
        )
        self.running = False

        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()

        self.runtime_defaults = build_runtime_defaults(config)
        self.runtime_builder = RuntimeSpecBuilder(config, self.runtime_defaults)
        self.backend = PodmanBackend(self.runtime_defaults)

        self.state = SupervisorState()
        self.jobs = self.state.running_jobs
        self._ready_queue = self.state.ready_queue
        self._pending = self.state.pending_jobs
        self._completed_jobs = self.state.completed_jobs
        self._failed_jobs = self.state.failed_jobs
        self._job_summaries = self.state.job_summaries
        self._job_git_refs = self.state.job_git_refs
        self._job_completion_codes = self.state.job_completion_codes
        self._processed_event_ids = self.state.processed_event_ids
        self._processing_event_ids: set[str] = set()
        self._event_processing_lock = asyncio.Lock()
        self._launched_event_ids = self.state.launched_event_ids
        self._spawn_defaults_by_job = self.state.spawn_defaults_by_job

        self.scheduler = Scheduler(self.state, max_workers=config.max_workers, teams=config.teams)
        self.spawn_handler = SpawnHandler(self.state)

        self._checkpoint_cycles = _DEFAULT_CHECKPOINT_CYCLES
        self._reap_timeout = self.runtime_defaults.cleanup_timeout_seconds

        self._webhook_id: str | None = None
        self._webhook_active = False
        self._webhook_poll_not_before: float = 0.0
        
        # Role catalog with SHA-based invalidation per ADR-0007
        self._role_catalog_cache: dict[str, dict[str, Any]] | None = None
        self._role_metadata_reader: RoleMetadataReader | None = None
        self._cached_evo_sha: str | None = None

    @property
    def event_cursor(self) -> str | None:
        return self.state.event_cursor

    @event_cursor.setter
    def event_cursor(self, value: str | None) -> None:
        self.state.event_cursor = value

    @property
    def paused(self) -> bool:
        return not self._resume_event.is_set()

    async def start(self) -> None:
        logger.info(
            "Supervisor starting (max_workers=%d, runtime=%s)",
            self.config.max_workers,
            self.runtime_defaults.kind,
        )
        self.running = True
        drain_task: asyncio.Task | None = None
        try:
            await self.client.register_source()
            self._validate_role_catalog()
            await self._replay_unfinished_tasks()
            await self._try_register_webhook()

            drain_task = asyncio.create_task(self._drain_queue())
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Supervisor loop cancelled")
        finally:
            self.running = False
            if drain_task is not None:
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass
            await self.client.close()
            await self.backend.close()

    async def stop(self, force: bool = False) -> None:
        logger.info("Supervisor stopping (force=%s)", force)
        self.running = False
        if self._webhook_id:
            try:
                await self.client.delete_webhook(self._webhook_id)
            except Exception as exc:
                logger.warning("Could not deregister webhook: %s", exc)

        for handle in list(self.jobs.values()):
            try:
                if force:
                    await self.backend.remove(handle, force=True)
                else:
                    await self.backend.stop(handle, self.runtime_defaults.stop_grace_seconds)
                    await self.backend.remove(handle)
            except Exception as exc:
                logger.warning("Could not stop job %s cleanly: %s", handle.job_id, exc)
        self.jobs.clear()

    async def pause(self) -> None:
        self._resume_event.clear()
        await self._emit_control_event("supervisor.paused")

    async def resume(self) -> None:
        self._resume_event.set()
        await self._emit_control_event("supervisor.resumed")

    async def _try_register_webhook(self) -> None:
        url = self.config.trenni_webhook_url
        if not url:
            return
        try:
            self._webhook_id = await self.client.register_webhook(
                url=url,
                secret=self.config.webhook_secret,
                event_types=[
                    "trigger.*",
                    "agent.job.spawn_request",
                    "agent.job.completed",
                    "agent.job.failed",
                    "agent.job.cancelled",
                    "agent.job.started",
                ],
            )
            self._webhook_active = True
            self._reset_webhook_poll_deadline()
        except Exception as exc:
            logger.warning("Webhook registration failed; falling back to polling: %s", exc)

    async def _run_loop(self) -> None:
        polls_since_checkpoint = 0
        while self.running:
            if self._poll_due_now():
                try:
                    await self._poll_and_handle()
                except Exception:
                    logger.exception("Error in poll cycle")

            try:
                await self._mark_exited_jobs()
            except Exception:
                logger.exception("Error while inspecting job containers")

            polls_since_checkpoint += 1
            if polls_since_checkpoint >= self._checkpoint_cycles:
                await self._checkpoint()
                polls_since_checkpoint = 0

            await asyncio.sleep(self.config.poll_interval)

    async def _drain_queue(self) -> None:
        while True:
            await self._resume_event.wait()
            job = await self._ready_queue.get()

            evaluation = self.scheduler.evaluate_job(job)
            if evaluation is False:
                await self._emit_cancellations([job], reason="Condition became impossible before launch")
                continue
            if evaluation is None:
                self._pending[job.job_id] = job
                continue

            # Re-check team capacity right before launch (Issue 5 fix)
            if not self.scheduler.has_team_capacity(job.team):
                # Team at capacity - put back in pending
                self._pending[job.job_id] = job
                logger.debug("Job %s team %s at capacity, returning to pending", job.job_id, job.team)
                continue

            while not self.scheduler.has_capacity() or not self._resume_event.is_set():
                await asyncio.sleep(1.0)
                evaluation = self.scheduler.evaluate_job(job)
                if evaluation is False:
                    await self._emit_cancellations([job], reason="Condition became impossible before launch")
                    break
                if evaluation is None:
                    self._pending[job.job_id] = job
                    break
                # Re-check team capacity during wait loop
                if not self.scheduler.has_team_capacity(job.team):
                    self._pending[job.job_id] = job
                    logger.debug("Job %s team %s at capacity during wait, returning to pending", job.job_id, job.team)
                    break
            else:
                try:
                    await self._launch_from_spawned(job)
                except Exception:
                    logger.exception("Failed to launch queued job %s", job.job_id)
                continue

    async def _launch_from_spawned(self, job: SpawnedJob) -> None:
        """Launch a SpawnedJob using its task semantics fields.

        Per ADR-0007: execution config from role defaults, budget from SpawnedJob.budget.
        """
        await self._launch(
            job_id=job.job_id,
            task_id=job.task_id or job.job_id,
            goal=job.goal,
            role=job.role,
            role_params=job.role_params,
            team=job.team,
            repo=job.repo,
            init_branch=job.init_branch,
            evo_sha=job.evo_sha,
            budget=job.budget,
            source_event_id=job.source_event_id,
            job_context=job.job_context,
            parent_job_id=job.parent_job_id,
            condition=job.condition,
        )

    async def _poll_and_handle(self) -> None:
        events, _ = await self.client.poll(cursor=self.event_cursor, limit=100)
        for event in events:
            await self._handle_event(event)
            self._advance_cursor_from_event(event)

    async def _handle_event(self, event: Event, *, realtime: bool = False, replay: bool = False) -> None:
        if event.id:
            async with self._event_processing_lock:
                if (
                    event.id in self._processed_event_ids
                    or event.id in self._processing_event_ids
                ):
                    return
                self._processing_event_ids.add(event.id)

        try:
            if event.type.startswith("trigger."):
                await self._handle_trigger(event, replay=replay)
            else:
                match event.type:
                    case "agent.job.spawn_request":
                        await self._handle_spawn(event, replay=replay)
                    case "supervisor.job.enqueued":
                        await self._handle_job_enqueued(event, replay=replay)
                    case "agent.job.completed" | "agent.job.failed" | "agent.job.cancelled":
                        await self._handle_job_done(event, replay=replay)
                    case "agent.job.started":
                        await self._handle_job_started(event, replay=replay)
                    case "supervisor.job.launched":
                        self._register_replayed_launch(event)
                    case "supervisor.task.created" | "supervisor.task.evaluating" | "supervisor.task.completed" | "supervisor.task.failed" | "supervisor.task.partial" | "supervisor.task.cancelled" | "supervisor.task.eval_failed":
                        pass  # State rebuilt naturally via replay if needed, but handled directly in Trigger/Done for realtime
        except Exception:
            if event.id:
                async with self._event_processing_lock:
                    self._processing_event_ids.discard(event.id)
            raise
        else:
            if event.id:
                async with self._event_processing_lock:
                    self._processing_event_ids.discard(event.id)
                    self._processed_event_ids.add(event.id)
            if realtime:
                self._advance_cursor_from_event(event)
                self._reset_webhook_poll_deadline()

    async def _handle_trigger(self, event: Event, *, replay: bool = False) -> None:
        if event.id in self._launched_event_ids and not replay:
            return

        try:
            data = TriggerData.model_validate(event.data)
        except Exception as e:
            logger.warning("Invalid trigger event %s: %s", event.id, e)
            return

        # goal is now required by pydantic (min_length=1), no need to check

        task_id = self._root_task_id(event.id)
        root_job_id = f"{task_id}-root"
        team = str(data.team or "default").strip() or "default"

        self.scheduler.record_task_submission(
            task_id=task_id,
            goal=data.goal,
            source_event_id=event.id,
            spec={"team": team, "budget": data.budget, "role": data.role},
        )
        if task_id in self.state.tasks:
            self.state.tasks[task_id].team = team

        if not replay:
            await self.client.emit(
                "supervisor.task.created",
                TaskCreatedData(
                    task_id=task_id,
                    team=team,
                    goal=data.goal,
                    source_trigger_id=event.id,
                ).model_dump(mode="json"),
                idempotency_key=self._event_idempotency_key(
                    source_event_id=event.id,
                    event_type="supervisor.task.created",
                    entity_id=task_id,
                ),
            )

        # Use canonical fields from TriggerData
        team_def = self._resolve_team_definition(team)
        role = data.role or team_def.planner_role or "planner"

        # role_params contains only role-internal flags
        role_params = dict(data.params)
        if role == team_def.planner_role:
            role_params.setdefault("mode", "initial")

        repo = data.repo
        init_branch = data.init_branch or "main"
        evo_sha = data.sha

        root_job = SpawnedJob(
            job_id=root_job_id,
            source_event_id=event.id,
            goal=data.goal,
            role=role,
            role_params=role_params,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            budget=data.budget,
            task_id=task_id,
            team=team,
        )
        budget_error = self._validate_spawned_job_budget(root_job)
        if budget_error:
            if not replay:
                await self._emit_task_terminal(
                    task_id=task_id,
                    state="failed",
                    result=TaskResult(),
                    reason=budget_error,
                )
            return

        cancelled = await self._enqueue(root_job, replay=replay)
        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Initial condition is impossible")
        self._launched_event_ids.add(event.id)

    async def _handle_spawn(self, event: Event, *, replay: bool = False) -> None:
        payload = SpawnRequestData.model_validate(event.data)
        if not payload.tasks:
            return

        # ADR-0007 D9: ensure role catalog reflects current evo/ state before
        # any role resolution or budget validation in this spawn expansion.
        self._load_role_catalog()

        plan = self.spawn_handler.expand(event)
        for task in plan.child_tasks:
            self.state.tasks[task.task_id] = task
            if not replay:
                parent_id = task.task_id.rsplit('/', 1)[0]
                await self.client.emit(
                    "supervisor.task.created",
                    TaskCreatedData(
                        task_id=task.task_id,
                        parent_task_id=parent_id,
                        team=task.team,
                        goal=task.goal,
                        source_trigger_id=task.source_event_id,
                        eval_spec=task.eval_spec,
                    ).model_dump(mode="json"),
                    idempotency_key=self._event_idempotency_key(
                        source_event_id=event.id,
                        event_type="supervisor.task.created",
                        entity_id=task.task_id,
                    ),
                )

        cancelled: list[SpawnedJob] = []
        for job in plan.jobs:
            budget_error = self._validate_spawned_job_budget(job)
            if budget_error:
                if not replay:
                    await self._emit_task_terminal(
                        task_id=job.task_id or job.job_id,
                        state="failed",
                        result=TaskResult(),
                        reason=budget_error,
                    )
                continue
            cancelled.extend(await self._enqueue(job, replay=replay))

        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Spawn condition is already impossible")

    async def _enqueue(self, job: SpawnedJob, *, replay: bool = False) -> list[SpawnedJob]:
        cancelled = await self.scheduler.enqueue(job)
        queue_state = "cancelled" if cancelled else "pending" if job.job_id in self._pending else "ready"
        task = self.state.tasks.get(job.task_id or "")
        if task is not None:
            if not self._is_eval_job(job) and job.job_id not in task.job_order:
                task.job_order.append(job.job_id)
            if not task.terminal and not task.eval_spawned and queue_state in {"pending", "ready"}:
                task.state = "pending" if queue_state == "pending" else "running"

        if not replay:
            await self.client.emit(
                "supervisor.job.enqueued",
                job.to_enqueued_data(queue_state, condition_to_data(job.condition)),
                idempotency_key=self._event_idempotency_key(
                    source_event_id=job.source_event_id,
                    event_type="supervisor.job.enqueued",
                    entity_id=job.job_id,
                ),
            )

        if job.job_id in self._pending:
            logger.debug("Pending job %s waiting on condition", job.job_id)
        elif not cancelled:
            logger.debug("Queued job %s", job.job_id)
        return [] if replay else cancelled

    async def _handle_job_enqueued(self, event: Event, *, replay: bool = False) -> None:
        if not replay:
            return

        data = SupervisorJobEnqueuedData.model_validate(event.data)
        job = SpawnedJob.from_enqueued_data(data.model_dump(mode="json"))

        if data.queue_state == "cancelled":
            await self.scheduler.record_job_terminal(
                job_id=job.job_id,
                summary="Condition became impossible before launch",
                cancelled=True,
            )
            return

        await self._enqueue(job, replay=True)

    async def _handle_job_done(self, event: Event, *, replay: bool = False) -> None:
        job_id = event.data.get("job_id", "")
        if not job_id:
            return

        is_failure = event.type == "agent.job.failed"
        is_cancelled = event.type == "agent.job.cancelled"
        summary = (
            event.data.get("summary")
            or event.data.get("error")
            or event.data.get("reason")
            or ""
        )
        git_ref = str(event.data.get("git_ref") or "")
        if git_ref:
            self._job_git_refs[job_id] = git_ref
        completion_code = str(event.data.get("code") or "")
        if completion_code:
            self._job_completion_codes[job_id] = completion_code

        handle = self.jobs.pop(job_id, None)
        job_record = self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None))
        task_id = job_record.task_id
        team = job_record.team
        
        # ADR-0011 D5: Track running jobs per team for launch conditions
        if team:
            self.state.decrement_team_running(team)
        
        _, cancelled = await self.scheduler.record_job_terminal(
            job_id=job_id,
            summary=summary,
            failed=is_failure,
            cancelled=is_cancelled,
        )

        if handle is not None and not replay:
            await self._cleanup_handle(handle, failed=is_failure or is_cancelled)

        # ADR-0010 D5: Emit budget_variance observation for completed jobs
        if not replay and not is_failure and not is_cancelled:
            await self._emit_budget_variance(job_id, event)

        if not replay:
            await self._evaluate_task_termination(job_id=job_id, task_id=task_id, event=event)

        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Condition became impossible")

    async def _evaluate_task_termination(self, job_id: str, task_id: str, event: Event) -> None:
        if not task_id:
            return

        task = self.state.tasks.get(task_id)
        if task is None or task.terminal:
            return

        job = self.state.jobs_by_id.get(job_id)
        if self._is_eval_job(job):
            await self._settle_eval_job(task=task, event=event)
            return

        if self._has_remaining_productive_jobs(task_id):
            return

        structural = self._build_structural_verdict(task_id)
        trace = self._build_task_trace(task_id)
        result = TaskResult(
            structural=structural,
            semantic=SemanticVerdict(verdict="unknown"),
            trace=trace,
        )
        task.result = result

        if task.eval_spec and not task.eval_spawned:
            eval_job_id = self._eval_job_id(task_id)
            task.eval_spawned = True
            task.eval_job_id = eval_job_id
            task.state = "evaluating"
            await self.client.emit(
                "supervisor.task.evaluating",
                TaskEvaluatingData(task_id=task_id, eval_job_id=eval_job_id, result=result).model_dump(mode="json"),
            )
            await self._enqueue(self._build_eval_job(task, eval_job_id), replay=False)
            return

        final_state = self._structural_terminal_state(structural)
        reason = (
            event.data.get("error")
            or event.data.get("reason")
            or event.data.get("summary")
            or ""
        )
        await self._emit_task_terminal(task_id=task_id, state=final_state, result=result, reason=reason)

    async def _settle_eval_job(self, *, task: TaskRecord, event: Event) -> None:
        result = task.result or TaskResult(
            structural=self._build_structural_verdict(task.task_id),
            semantic=SemanticVerdict(verdict="unknown"),
            trace=self._build_task_trace(task.task_id),
        )

        if event.type == "agent.job.completed":
            result.semantic = self._semantic_from_eval_event(event.data)
            task.result = result
            final_state = self._semantic_terminal_state(result.semantic, result.structural)
            await self._emit_task_terminal(
                task_id=task.task_id,
                state=final_state,
                result=result,
                reason=result.semantic.summary,
            )
            return

        reason = event.data.get("error") or event.data.get("reason") or "Eval job failed"
        result.semantic = SemanticVerdict(verdict="unknown", summary=reason)
        task.result = result
        await self.scheduler.mark_task_terminal(task_id=task.task_id, state="eval_failed")
        task.state = "eval_failed"
        await self.client.emit(
            "supervisor.task.eval_failed",
            TaskEvalFailedData(task_id=task.task_id, reason=reason, result=result).model_dump(mode="json"),
        )

    async def _emit_task_terminal(self, *, task_id: str, state: str, result: TaskResult, reason: str = "") -> None:
        await self.scheduler.mark_task_terminal(task_id=task_id, state=state)
        task = self.state.tasks.get(task_id)
        if task is not None:
            task.state = state
            task.result = result

        if state == "completed":
            summary = (
                result.semantic.summary
                or next((entry.summary for entry in reversed(result.trace) if entry.summary), "")
            )
            await self.client.emit(
                "supervisor.task.completed",
                TaskCompletedData(task_id=task_id, summary=summary, result=result).model_dump(mode="json"),
            )
            return

        if state == "failed":
            fail_reason = reason or result.semantic.summary or "Task failed"
            await self.client.emit(
                "supervisor.task.failed",
                TaskFailedData(task_id=task_id, reason=fail_reason, result=result).model_dump(mode="json"),
            )
            await self._cascade_cancel(task_id, reason=f"Parent or sibling failed: {fail_reason}")
            return

        if state == "partial":
            partial_reason = reason or result.semantic.summary or "Task budget exhausted before completion"
            await self.client.emit(
                "supervisor.task.partial",
                TaskPartialData(task_id=task_id, reason=partial_reason, result=result).model_dump(mode="json"),
            )
            return

        if state == "cancelled":
            cancel_reason = reason or "Task cancelled"
            await self.client.emit(
                "supervisor.task.cancelled",
                TaskCancelledData(task_id=task_id, reason=cancel_reason, result=result).model_dump(mode="json"),
            )
            await self._cascade_cancel(task_id, reason=f"Parent or sibling cancelled: {cancel_reason}")
            return

        await self.client.emit(
            "supervisor.task.eval_failed",
            TaskEvalFailedData(task_id=task_id, reason=reason or "Eval failed", result=result).model_dump(mode="json"),
        )

    def _has_remaining_productive_jobs(self, task_id: str) -> bool:
        for job_id, job in self.state.jobs_by_id.items():
            if job.task_id != task_id or self._is_eval_job(job):
                continue
            if job_id not in self.state.completed_jobs:
                return True
        for job_id, job in self.state.pending_jobs.items():
            if job.task_id == task_id and not self._is_eval_job(job):
                return True
        for job in self.state.ready_queue_snapshot():
            if job.task_id == task_id and not self._is_eval_job(job):
                return True
        return False

    @staticmethod
    def _is_eval_job(job: SpawnedJob | None) -> bool:
        if job is None:
            return False
        return bool(job.job_context and job.job_context.eval is not None)

    def _build_structural_verdict(self, task_id: str) -> StructuralVerdict:
        verdict = StructuralVerdict()
        for job_id in self._task_trace_job_ids(task_id):
            if job_id in self.state.failed_jobs:
                verdict.failed += 1
            elif job_id in self.state.cancelled_jobs:
                verdict.cancelled += 1
            elif self.state.job_completion_codes.get(job_id) == "budget_exhausted":
                verdict.partial += 1
            elif job_id in self.state.completed_jobs:
                verdict.success += 1
            else:
                verdict.unknown += 1
        return verdict

    def _build_task_trace(self, task_id: str) -> list[TaskTraceEntry]:
        trace: list[TaskTraceEntry] = []
        for job_id in self._task_trace_job_ids(task_id):
            job = self.state.jobs_by_id.get(job_id)
            if job is None:
                continue
            if job_id in self.state.failed_jobs:
                outcome = "failed"
            elif job_id in self.state.cancelled_jobs:
                outcome = "cancelled"
            elif self.state.job_completion_codes.get(job_id) == "budget_exhausted":
                outcome = "partial"
            elif job_id in self.state.completed_jobs:
                outcome = "success"
            else:
                outcome = "unknown"
            trace.append(
                TaskTraceEntry(
                    job_id=job_id,
                    role=job.role,
                    outcome=outcome,
                    summary=self.state.job_summaries.get(job_id, ""),
                    git_ref=self.state.job_git_refs.get(job_id, ""),
                )
            )
        return trace

    def _task_trace_job_ids(self, task_id: str) -> list[str]:
        record = self.state.tasks.get(task_id)
        ordered = list(record.job_order) if record else []
        seen = set(ordered)
        for job_id, job in self.state.jobs_by_id.items():
            if job.task_id != task_id or self._is_eval_job(job) or job_id in seen:
                continue
            ordered.append(job_id)
            seen.add(job_id)
        return ordered

    @staticmethod
    def _structural_terminal_state(structural: StructuralVerdict) -> str:
        if structural.failed > 0 or structural.unknown > 0:
            return "failed"
        if structural.partial > 0:
            return "partial"
        if structural.cancelled > 0:
            return "cancelled"
        return "completed"

    @staticmethod
    def _semantic_terminal_state(semantic: SemanticVerdict, structural: StructuralVerdict) -> str:
        if structural.partial > 0:
            return "partial"
        if semantic.verdict == "pass":
            return "completed"
        if semantic.verdict == "fail":
            return "failed"
        return Supervisor._structural_terminal_state(structural)

    @staticmethod
    def _semantic_from_eval_event(data: dict[str, Any]) -> SemanticVerdict:
        raw = (data.get("summary") or "").strip()
        if not raw:
            return SemanticVerdict(verdict="unknown", summary="")

        try:
            parsed = json.loads(raw)
        except Exception:
            status = str(data.get("status") or "").strip().lower()
            if status in {"failed", "fail"}:
                verdict = "fail"
            elif status in {"complete", "completed", "success", "pass"}:
                verdict = "pass"
            else:
                verdict = "unknown"
            return SemanticVerdict(verdict=verdict, summary=raw)

        verdict = str(parsed.get("verdict", "unknown")).strip().lower()
        if verdict not in {"pass", "fail", "unknown"}:
            verdict = "unknown"
        summary = str(parsed.get("summary", "")).strip()
        criteria_results = parsed.get("criteria_results")
        if not isinstance(criteria_results, list):
            criteria_results = []
        return SemanticVerdict(
            verdict=verdict,
            summary=summary,
            criteria_results=[item for item in criteria_results if isinstance(item, dict)],
        )

    async def _cascade_cancel(self, task_id: str, reason: str) -> None:
        prefix = task_id + "/"
        jobs_to_cancel: list[SpawnedJob] = []

        for tid, record in list(self.state.tasks.items()):
            if tid.startswith(prefix) and not record.terminal:
                await self.scheduler.mark_task_terminal(task_id=tid, state="cancelled")
                record.state = "cancelled"
                await self.client.emit(
                    "supervisor.task.cancelled",
                    TaskCancelledData(
                        task_id=tid,
                        reason=reason,
                        result=record.result or TaskResult(),
                    ).model_dump(mode="json")
                )
        
        for job_id, job in list(self.state.jobs_by_id.items()):
            if job.task_id and job.task_id.startswith(prefix) and job_id not in self.state.completed_jobs:
                jobs_to_cancel.append(job)

        if jobs_to_cancel:
            await self._emit_cancellations(jobs_to_cancel, reason=reason)

    async def _mark_exited_jobs(self) -> None:
        await mark_exited_jobs(self.jobs, self.backend)

    async def _checkpoint(self) -> None:
        reaped = await reap_timed_out_jobs(
            self.jobs,
            backend=self.backend,
            reap_timeout=self._reap_timeout,
        )

        for handle, logs in reaped:
            self.jobs.pop(handle.job_id, None)
            job_record = self.state.jobs_by_id.get(handle.job_id, SpawnedJob("", "", "", "", "", "", None))
            task_id = job_record.task_id if handle.job_id in self.state.jobs_by_id else handle.job_id
            team = job_record.team
            
            # ADR-0011 D5: Track running jobs per team for launch conditions
            if team:
                self.state.decrement_team_running(team)
            
            await self.scheduler.record_job_terminal(
                job_id=handle.job_id,
                summary=f"Container exited without terminal event (exit_code={handle.exit_code})",
                failed=True,
            )
            event_data = {
                "job_id": handle.job_id,
                "task_id": task_id,
                "error": f"Container exited without emitting terminal event (exit_code={handle.exit_code})",
                "code": "runtime_lost",
                "logs_tail": logs[-4000:],
            }
            event = SimpleNamespace(id="", source_id="", type="agent.job.failed", data=event_data)
            await self.client.emit("agent.job.failed", event_data)
            await self._cleanup_handle(handle, failed=True)
            await self._evaluate_task_termination(job_id=handle.job_id, task_id=task_id, event=event)

        snapshot = SupervisorCheckpointData.model_validate(self.state.snapshot())
        await self.client.emit("supervisor.checkpoint", snapshot.model_dump(mode="json"))

    async def _fetch_all(self, type_: str, source: str | None = None) -> list[Event]:
        results: list[Event] = []
        cursor = None
        while True:
            events, next_cursor = await self.client.poll(
                cursor=cursor,
                source=source,
                type_=type_,
                limit=100,
            )
            results.extend(events)
            if not next_cursor:
                return results
            cursor = next_cursor

    async def _replay_unfinished_tasks(self) -> None:
        await rebuild_state(self)

    async def _handle_job_started(self, event: Event, *, replay: bool = False) -> None:
        job_id = event.data.get("job_id", "")
        if not job_id:
            return
        job = self.state.jobs_by_id.get(job_id)
        if job is None:
            return
        task = self.state.tasks.get(job.task_id)
        if task is None or task.terminal or task.state == "evaluating":
            return
        task.state = "running"

    async def _launch(
        self,
        job_id: str,
        goal: str,
        role: str,
        repo: str,
        init_branch: str,
        evo_sha: str | None,
        budget: float | None = None,
        source_event_id: str = "",
        task_id: str = "",
        job_context=None,
        parent_job_id: str = "",
        condition=None,
        team: str = "default",
        role_params: dict[str, Any] | None = None,
    ) -> None:
        """Launch a job in the isolation backend."""
        spec = self.runtime_builder.build(
            job_id=job_id,
            task_id=task_id or job_id,
            source_event_id=source_event_id,
            goal=goal,
            role=role,
            role_params=role_params,
            team=team,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            budget=budget,
            job_context=job_context,
        )

        # Validate runtime environment before container creation
        await self.backend.ensure_ready(spec)

        handle = await self.backend.prepare(spec)
        try:
            await self.backend.start(handle)
        except Exception:
            await self.backend.remove(handle, force=True)
            raise

        self.jobs[job_id] = handle
        self.state.jobs_by_id[job_id] = SpawnedJob(
            job_id=job_id,
            source_event_id=source_event_id,
            goal=goal,
            role=role,
            role_params=dict(role_params or {}),
            team=team,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            budget=budget or 0.0,
            task_id=task_id or job_id,
            condition=condition,
            job_context=job_context or self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None)).job_context,
            parent_job_id=parent_job_id,
        )
        
        # ADR-0011 D5: Track running jobs per team for launch conditions
        self.state.increment_team_running(team)
        
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=repo,
            init_branch=init_branch,
            role=role,
            role_params=dict(role_params or {}),
            team=team,
            evo_sha=evo_sha,
            task_id=task_id or job_id,
            budget=budget or 0.0,  # ADR-0010: for budget_variance observation
        )

        # Get llm config from spec for observability event
        # (fully-resolved config from runtime_builder, not override deltas)
        spec_llm = spec.config_payload_b64  # We'll extract from the built spec

        # Use the SpawnedJob just stored to emit the launched event
        launched_job = self.state.jobs_by_id[job_id]
        await self.client.emit(
            "supervisor.job.launched",
            launched_job.to_launched_data(
                self.runtime_defaults.kind,
                handle.container_id,
                handle.container_name,
                condition_to_data(condition),
            ),
        )

    async def _cleanup_handle(self, handle: JobHandle, *, failed: bool) -> None:
        if failed and self.runtime_defaults.retain_on_failure:
            return
        try:
            await self.backend.stop(handle, self.runtime_defaults.stop_grace_seconds)
        except Exception:
            pass
        try:
            await self.backend.remove(handle, force=failed)
        except Exception as exc:
            logger.warning("Could not remove container for job %s: %s", handle.job_id, exc)

    async def _emit_control_event(self, event_type: str) -> None:
        try:
            await self.client.emit(event_type, {}, timeout=_CONTROL_EVENT_TIMEOUT_S)
        except Exception:
            logger.warning("Could not emit %s", event_type)

    async def _emit_cancellations(self, jobs: list[SpawnedJob], *, reason: str) -> None:
        for job in jobs:
            task_id = job.task_id or job.job_id
            await self.client.emit(
                "agent.job.cancelled",
                JobCancelledData(job_id=job.job_id, task_id=task_id, reason=reason).model_dump(mode="json"),
            )
            await self.scheduler.record_job_terminal(job_id=job.job_id, summary=reason, cancelled=True)
            
            await self._evaluate_task_termination(
                job_id=job.job_id, 
                task_id=task_id, 
                event=SimpleNamespace(id="", type="agent.job.cancelled", source_id="", data={"reason": reason})
            )

    def _poll_due_now(self) -> bool:
        if not self._webhook_active:
            return True
        return asyncio.get_running_loop().time() >= self._webhook_poll_not_before

    def _reset_webhook_poll_deadline(self) -> None:
        if self._webhook_active:
            self._webhook_poll_not_before = asyncio.get_running_loop().time() + self.config.webhook_poll_interval

    def _advance_cursor_from_event(self, event: Event) -> None:
        current = self._cursor_key(self.event_cursor)
        candidate = (self._normalize_cursor_ts(event.ts), event.id)
        if current is None or candidate > current:
            self.event_cursor = f"{event.ts.isoformat()}|{event.id}"

    @staticmethod
    def _cursor_key(cursor: str | None) -> tuple[datetime, str] | None:
        if not cursor:
            return None
        try:
            ts_raw, event_id = cursor.split("|", 1)
            return Supervisor._normalize_cursor_ts(datetime.fromisoformat(ts_raw)), event_id
        except ValueError:
            return None

    @staticmethod
    def _normalize_cursor_ts(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts
        return ts.astimezone(timezone.utc).replace(tzinfo=None)

    async def _inspect_replay_state(self, container_id: str, container_name: str) -> ContainerState:
        ref = container_id or container_name
        if not ref:
            return ContainerState(exists=False)
        return await self.backend.inspect(
            JobHandle(job_id="", container_id=container_id or ref, container_name=container_name or ref)
        )

    def _handle_from_replay(self, job_id: str, container_id: str, container_name: str) -> JobHandle:
        ref = container_id or container_name or f"yoitsu-job-{job_id}"
        return JobHandle(job_id=job_id, container_id=container_id or ref, container_name=container_name or ref)

    def _team_root(self) -> Path:
        if self.config.evo_root:
            return Path(self.config.evo_root)
        return Path(__file__).resolve().parents[2] / "palimpsest" / "evo"

    def _read_evo_sha(self) -> str:
        """Read current HEAD SHA from evo root for cache invalidation."""
        evo_root = self._team_root()
        try:
            result = subprocess.run(
                ["git", "-C", str(evo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _load_role_catalog(self) -> dict[str, dict[str, Any]]:
        """Load role catalog with SHA-based cache invalidation.

        Per ADR-0007: cache is invalidated when evo SHA changes.
        Per ADR-0011 D2/D3: Team membership is determined by directory location:
        - evo/roles/<name>.py → available to all teams (teams = ["*"])
        - evo/teams/<team>/roles/<name>.py → available only to <team> (teams = [team_name])
        Uses RoleMetadataReader from yoitsu-contracts for AST parsing.

        Returns a dict keyed by role name, where each value is a dict containing:
        - "global": the global role definition (if exists), or None
        - "teams": {team_name: team_specific_definition, ...}

        This structure preserves global roles when team-specific roles have the same name.
        """
        current_sha = self._read_evo_sha()
        
        # Invalidate cache if SHA changed
        if current_sha != self._cached_evo_sha:
            self._role_catalog_cache = None
            if self._role_metadata_reader:
                self._role_metadata_reader.invalidate_cache()
            self._cached_evo_sha = current_sha

        if self._role_catalog_cache is not None:
            return self._role_catalog_cache

        catalog: dict[str, dict[str, Any]] = {}
        evo_root = self._team_root()

        # Scan global roles (available to all teams)
        global_roles_dir = evo_root / "roles"
        if global_roles_dir.exists():
            reader = RoleMetadataReader(evo_root)
            for meta in reader.list_definitions():
                catalog[meta.name] = {
                    "global": {
                        "name": meta.name,
                        "description": meta.description,
                        "role_type": meta.role_type,
                        "teams": ["*"],  # Available to all teams (ADR-0011 D2)
                        "min_cost": meta.min_cost,
                        "recommended_cost": meta.recommended_cost,
                        "max_cost": meta.max_cost,
                        "min_capability": meta.min_capability,
                        "source_team": None,  # Global role
                    },
                    "teams": {},  # No team-specific override yet
                }

        # Scan team-specific roles (ADR-0011 D3)
        teams_dir = evo_root / "teams"
        if teams_dir.exists():
            for team_dir in teams_dir.iterdir():
                if not team_dir.is_dir():
                    continue
                team_name = team_dir.name
                team_roles_dir = team_dir / "roles"
                if not team_roles_dir.exists():
                    continue
                # Create a reader scoped to this team's roles directory
                for py_path in sorted(team_roles_dir.glob("*.py")):  
                    if py_path.name.startswith("_"):
                        continue
                    meta = self._read_role_file(py_path, evo_root)
                    if meta is None:
                        continue
                    # Add to existing entry (global role exists with same name) or create new
                    if meta.name not in catalog:
                        catalog[meta.name] = {
                            "global": None,  # No global version
                            "teams": {},
                        }
                    catalog[meta.name]["teams"][team_name] = {
                        "name": meta.name,
                        "description": meta.description,
                        "role_type": meta.role_type,
                        "teams": [team_name],  # Only this team (ADR-0011 D3)
                        "min_cost": meta.min_cost,
                        "recommended_cost": meta.recommended_cost,
                        "max_cost": meta.max_cost,
                        "min_capability": meta.min_capability,
                        "source_team": team_name,  # Team-specific role
                    }

        self._role_catalog_cache = catalog
        return catalog

    def _get_role_for_team(self, role_name: str, team: str) -> dict[str, Any] | None:
        """Get role definition for a specific team, preferring team-specific over global.

        Args:
            role_name: Name of the role to look up
            team: Team name to resolve the role for

        Returns:
            Role definition dict, preferring team-specific version if available,
            otherwise falling back to global version. Returns None if role not found.
        """
        catalog = self._load_role_catalog()
        entry = catalog.get(role_name)
        if not entry:
            return None

        # Prefer team-specific version
        team_roles = entry.get("teams", {})
        if team in team_roles:
            return team_roles[team]

        # Fallback to global version
        return entry.get("global")

    def _read_role_file(self, py_path: Path, evo_root: Path) -> "RoleMetadata | None":
        """Read role metadata from a single Python file using AST parsing.

        This is a lightweight version of RoleMetadataReader._read_role_file
        for scanning team-specific role directories.
        """
        import ast
        from dataclasses import dataclass, field
        from typing import Any

        @dataclass
        class RoleMetadata:
            name: str
            description: str
            teams: list[str] = field(default_factory=list)
            role_type: str = "worker"
            min_cost: float = 0.0
            recommended_cost: float = 0.0
            max_cost: float = 10.0
            min_capability: str = ""

        source = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if self._is_role_decorator_ast(decorator):
                            return self._extract_metadata_ast(decorator, py_path, RoleMetadata)
        return None

    def _is_role_decorator_ast(self, decorator: "ast.Call") -> bool:
        """Check if decorator call is @role(...)."""
        import ast
        if isinstance(decorator.func, ast.Name):
            return decorator.func.id == "role"
        if isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == "role":
                return True
        return False

    def _extract_metadata_ast(self, decorator: "ast.Call", py_path: Path, RoleMetadata: type) -> "RoleMetadata":
        """Extract RoleMetadata from @role(...) decorator keywords."""
        import ast
        from typing import Any

        kwargs: dict[str, Any] = {}
        for keyword in decorator.keywords:
            try:
                value = ast.literal_eval(keyword.value)
            except ValueError as e:
                raise ValueError(
                    f"Role decorator argument '{keyword.arg}' in {py_path} "
                    f"must be a literal expression. Got non-literal: {ast.unparse(keyword.value)}. "
                    f"Error: {e}"
                ) from e
            kwargs[keyword.arg] = value

        return RoleMetadata(
            name=str(kwargs.get("name", "")),
            description=str(kwargs.get("description", "")),
            teams=list(kwargs.get("teams", [])),  # Ignored per ADR-0011 D3
            role_type=str(kwargs.get("role_type", "worker")),
            min_cost=float(kwargs.get("min_cost", 0.0)),
            recommended_cost=float(kwargs.get("recommended_cost", 0.0)),
            max_cost=float(kwargs.get("max_cost", 10.0)),
            min_capability=str(kwargs.get("min_capability", "")),
        )

    def _validate_role_catalog(self) -> None:
        catalog = self._load_role_catalog()
        # Collect all team names from the catalog structure
        team_names = set()
        for entry in catalog.values():
            if entry.get("global"):
                # Global roles are available to all teams
                team_names.add("default")
            team_names.update(entry.get("teams", {}).keys())
        team_names = sorted(team_names)
        if "default" not in team_names:
            team_names.insert(0, "default")
        for team_name in team_names:
            self._resolve_team_definition(team_name)

    def _resolve_role_metadata(self, role_name: str, team: str | None = None) -> dict[str, Any]:
        """Resolve role metadata, optionally for a specific team.

        Args:
            role_name: Name of the role to look up
            team: Optional team name. If provided, prefers team-specific version.
                 If None, returns global version if available.

        Returns:
            Role definition dict.

        Raises:
            FileNotFoundError: If role not found for the given context.
        """
        catalog = self._load_role_catalog()
        if role_name not in catalog:
            raise FileNotFoundError(f"Role definition not found: {role_name!r}")

        entry = catalog[role_name]

        # If team specified, prefer team-specific version
        if team:
            team_roles = entry.get("teams", {})
            if team in team_roles:
                return team_roles[team]

        # Fallback to global version
        global_role = entry.get("global")
        if global_role:
            return global_role

        # No global version exists, but team-specific versions do
        # This means the role only exists for specific teams
        available_teams = list(entry.get("teams", {}).keys())
        raise FileNotFoundError(
            f"Role {role_name!r} not available for team {team!r}. "
            f"Available teams: {available_teams}"
        )

    def _allocated_job_budget(self, job: SpawnedJob) -> float:
        """Get the allocated budget for a job.

        Per ADR-0007: budget is read from SpawnedJob.budget (task semantics field).
        Single channel, not role_params or llm_overrides.
        """
        # Primary: budget field on SpawnedJob (ADR-0007)
        if job.budget and job.budget > 0:
            return float(job.budget)
        
        # Fallback: default from TrenniConfig
        default_budget = self.config.default_llm.get("max_total_cost")
        if isinstance(default_budget, (int, float)):
            return float(default_budget)
        return 0.0

    def _validate_spawned_job_budget(self, job: SpawnedJob) -> str | None:
        try:
            meta = self._resolve_role_metadata(job.role, team=job.team)
        except FileNotFoundError as exc:
            return str(exc)
        min_cost = float(meta.get("min_cost", 0.0) or 0.0)
        allocated = self._allocated_job_budget(job)
        if min_cost > 0 and allocated < min_cost:
            return (
                f"Allocated budget ${allocated:.4f} is below role {job.role!r} "
                f"minimum cost ${min_cost:.4f}"
            )
        return None

    def _resolve_team_definition(self, name: str):
        team_name = (name or "default").strip() or "default"
        catalog = self._load_role_catalog()

        # Per ADR-0011 D2: Build deduplicated member list with shadowing.
        # Team-specific roles shadow global roles of the same name.
        # For each role name, prefer team-specific over global.
        seen_names: set[str] = set()
        members: list[dict[str, Any]] = []

        for role_name, entry in catalog.items():
            # Prefer team-specific over global (shadowing semantics per ADR-0011 D2)
            team_roles = entry.get("teams", {})
            if team_name in team_roles:
                # Team-specific version exists - use it, skip global
                members.append(team_roles[team_name])
                seen_names.add(role_name)
            elif entry.get("global"):
                # No team-specific version, use global
                members.append(entry["global"])
                seen_names.add(role_name)

        if not members:
            if team_name == "default":
                return _DEFAULT_TEAM_DEFINITION
            raise FileNotFoundError(f"No roles found for team {team_name!r}")

        # Categorize by role_type (no duplicates possible now)
        planners = [m for m in members if m["role_type"] == "planner"]
        evaluators = [m for m in members if m["role_type"] == "evaluator"]
        workers = [m for m in members if m["role_type"] == "worker"]

        planner_names = [m["name"] for m in planners]
        evaluator_names = [m["name"] for m in evaluators]
        worker_names = [m["name"] for m in workers]

        if len(planner_names) != 1:
            raise ValueError(f"Team {team_name!r} must have exactly one planner role")
        if len(evaluator_names) > 1:
            raise ValueError(f"Team {team_name!r} must have at most one evaluator role")
        if not worker_names:
            raise ValueError(f"Team {team_name!r} must have at least one worker role")

        return SimpleNamespace(
            name=team_name,
            description=f"Derived team {team_name}",
            roles=[m["name"] for m in members],
            planner_role=planner_names[0],
            eval_role=evaluator_names[0] if evaluator_names else _DEFAULT_TEAM_DEFINITION.eval_role,
            worker_roles=worker_names,
        )

    def _register_replayed_launch(self, event: Event) -> None:
        """Register a job from a replayed supervisor.job.launched event.

        Per ADR-0007: SpawnedJob and SpawnDefaults no longer carry execution config overrides.
        """
        data = event.data
        job_id = data.get("job_id", "")
        if not job_id:
            return

        source_event_id = data.get("source_event_id", "")

        existing = self.state.jobs_by_id.get(job_id)
        self.state.jobs_by_id[job_id] = existing or SpawnedJob(
            job_id=job_id,
            source_event_id=source_event_id,
            goal=data.get("goal", ""),
            role=data.get("role", "default"),
            team=data.get("team", "default"),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            evo_sha=data.get("evo_sha") or None,
            budget=0.0,
            task_id=data.get("task_id", "") or job_id,
            condition=condition_from_data(data.get("condition")),
            parent_job_id=data.get("parent_job_id", ""),
        )
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            role=data.get("role", "default"),
            evo_sha=data.get("evo_sha") or None,
            task_id=data.get("task_id", "") or job_id,
            team=data.get("team", "default"),
            budget=0.0,  # budget not available in launched event
        )
        if source_event_id:
            self._launched_event_ids.add(source_event_id)
        self.state.remove_pending_job(job_id)
        self.state.drop_from_ready_queue(job_id)


    @staticmethod
    def _root_task_id(source_event_id: str) -> str:
        hex_only = "".join(ch for ch in (source_event_id or "").lower() if ch in "0123456789abcdef")
        if len(hex_only) >= 16:
            return hex_only[:16]
        digest = hashlib.sha256((source_event_id or "").encode("utf-8")).hexdigest()
        return digest[:16]

    @staticmethod
    def _eval_job_id(task_id: str) -> str:
        return f"{task_id}-eval"

    def _direct_child_task_ids(self, parent_task_id: str) -> list[str]:
        prefix = parent_task_id + "/"
        parent_depth = parent_task_id.count("/")
        children: list[str] = []
        for task_id in self.state.tasks:
            if not task_id.startswith(prefix):
                continue
            if task_id.count("/") == parent_depth + 1:
                children.append(task_id)
        return sorted(children)

    def _build_eval_job(self, task: TaskRecord, eval_job_id: str) -> SpawnedJob:
        """Build an eval job for a task.

        Per ADR-0007: eval jobs use role-internal params for eval-specific behavior.
        workspace.new_branch=False and publication.strategy="skip" are handled by
        role definition or role_params.
        """
        eval_spec = task.eval_spec or EvalSpec()
        team_def = self._resolve_team_definition(task.team)
        role = eval_spec.role or team_def.eval_role or "evaluator"
        base_job = next(
            (
                self.state.jobs_by_id[job_id]
                for job_id in reversed(task.job_order)
                if job_id in self.state.jobs_by_id and not self._is_eval_job(self.state.jobs_by_id[job_id])
            ),
            None,
        )
        latest_git_ref = next(
            (
                self.state.job_git_refs.get(job_id, "")
                for job_id in reversed(task.job_order)
                if self.state.job_git_refs.get(job_id, "")
            ),
            "",
        )
        eval_branch = self._branch_from_git_ref(latest_git_ref) or (base_job.init_branch if base_job else "main")
        child_task_ids = self._direct_child_task_ids(task.task_id)
        
        # Per ADR-0007: role_params for eval-specific flags
        eval_role_params = {
            "eval_mode": True,  # role-internal flag
        }
        
        return SpawnedJob(
            job_id=eval_job_id,
            source_event_id=task.source_event_id,
            goal=self._eval_prompt(task.goal, eval_spec),
            role=role,
            role_params=eval_role_params,
            team=task.team,
            repo=base_job.repo if base_job else "",
            init_branch=eval_branch,
            evo_sha=base_job.evo_sha if base_job else None,
            task_id=task.task_id,
            parent_job_id=base_job.parent_job_id if base_job else "",
            job_context=JobContextConfig(
                join=(
                    None
                    if not child_task_ids
                    else JoinContextConfig(
                        parent_job_id=base_job.job_id if base_job else "",
                        parent_task_id=task.task_id,
                        parent_summary=task.goal,
                        child_task_ids=child_task_ids,
                    )
                ),
                eval=EvalContextConfig(
                    task_id=task.task_id,
                    goal=task.goal,
                    deliverables=list(eval_spec.deliverables),
                    criteria=list(eval_spec.criteria),
                    structural=(task.result.structural.model_dump(mode="json") if task.result else {}),
                    child_task_ids=child_task_ids,
                ),
            ),
        )

    @staticmethod
    def _eval_prompt(goal: str, spec: EvalSpec) -> str:
        deliverables = "\n".join(f"- {item}" for item in spec.deliverables) or "- (not provided)"
        criteria = "\n".join(f"- {item}" for item in spec.criteria) or "- (not provided)"
        return (
            "You are the evaluator for task semantic quality.\n"
            "Assess whether the task goal is truly met using the provided context.\n"
            "Return a single JSON object in your final response with fields:\n"
            '{"verdict":"pass|fail|unknown","summary":"...","criteria_results":[{"criterion":"...","result":"pass|fail|unknown","evidence":"..."}]}\n'
            "Do not perform rework and do not mutate workflow state.\n\n"
            f"Original goal:\n{goal}\n\n"
            f"Expected deliverables:\n{deliverables}\n\n"
            f"Verification criteria:\n{criteria}\n"
        )

    @staticmethod
    def _branch_from_git_ref(git_ref: str) -> str:
        if ":" not in git_ref:
            return ""
        return git_ref.split(":", 1)[0].strip()

    async def _emit_budget_variance(self, job_id: str, event: Event) -> None:
        """Emit budget_variance observation after job completion (ADR-0010 D5).
        
        Uses spawn_defaults for estimated_budget and event.data for actual cost.
        """
        spawn_defaults = self._spawn_defaults_by_job.get(job_id)
        if not spawn_defaults or spawn_defaults.budget <= 0:
            return  # No estimated budget, skip variance calculation
        
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

    @staticmethod
    def _event_idempotency_key(*, source_event_id: str, event_type: str, entity_id: str) -> str | None:
        if not source_event_id:
            return None
        return f"trenni:{source_event_id}:{event_type}:{entity_id}"


    def _generate_job_id(self) -> str:
        import uuid_utils

        return str(uuid_utils.uuid7())

    @property
    def status(self) -> dict:
        return self.scheduler.status_snapshot(
            runtime_kind=self.runtime_defaults.kind,
            running=self.running,
            paused=self.paused,
        )
