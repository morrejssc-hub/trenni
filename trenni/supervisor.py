"""Core supervisor: event routing + scheduler + isolation backend."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from yoitsu_contracts.conditions import condition_from_data, condition_to_data
from yoitsu_contracts.events import (
    JobCancelledData,
    JobCompletedData,
    JobFailedData,
    SpawnRequestData,
    SupervisorCheckpointData,
    SupervisorJobLaunchedData,
    TriggerData,
    TaskCreatedData,
    TaskCompletedData,
    TaskFailedData,
    TaskCancelledData,
)

from .checkpoint import mark_exited_jobs, reap_timed_out_jobs
from .config import TrenniConfig
from .pasloe_client import Event, PasloeClient
from .podman_backend import PodmanBackend
from .replay import rebuild_state
from .runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from .runtime_types import ContainerState, JobHandle
from .scheduler import Scheduler
from .spawn_handler import SpawnHandler
from .state import SpawnDefaults, SpawnedJob, SupervisorState

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_CYCLES = 30
_CONTROL_EVENT_TIMEOUT_S = 1.0


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
        self._processed_event_ids = self.state.processed_event_ids
        self._launched_event_ids = self.state.launched_event_ids
        self._spawn_defaults_by_job = self.state.spawn_defaults_by_job

        self.scheduler = Scheduler(self.state, max_workers=config.max_workers)
        self.spawn_handler = SpawnHandler(self.state)

        self._checkpoint_cycles = _DEFAULT_CHECKPOINT_CYCLES
        self._reap_timeout = self.runtime_defaults.cleanup_timeout_seconds

        self._webhook_id: str | None = None
        self._webhook_active = False
        self._webhook_poll_not_before: float = 0.0

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
            await self.backend.ensure_ready()
            await self.client.register_source()
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
                    "job.spawn.request",
                    "job.completed",
                    "job.failed",
                    "job.cancelled",
                    "job.started",
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

            while not self.scheduler.has_capacity() or not self._resume_event.is_set():
                await asyncio.sleep(1.0)
                evaluation = self.scheduler.evaluate_job(job)
                if evaluation is False:
                    await self._emit_cancellations([job], reason="Condition became impossible before launch")
                    break
                if evaluation is None:
                    self._pending[job.job_id] = job
                    break
            else:
                try:
                    await self._launch_from_spawned(job)
                except Exception:
                    logger.exception("Failed to launch queued job %s", job.job_id)
                continue

    async def _launch_from_spawned(self, job: SpawnedJob) -> None:
        await self._launch(
            job_id=job.job_id,
            task_id=job.task_id or job.job_id,
            task=job.task,
            role=job.role,
            repo=job.repo,
            init_branch=job.init_branch,
            evo_sha=job.evo_sha,
            llm_overrides=job.llm_overrides,
            workspace_overrides=job.workspace_overrides,
            publication_overrides=job.publication_overrides,
            source_event_id=job.source_event_id,
            job_context=job.job_context,
            parent_job_id=job.parent_job_id,
            condition=job.condition,
        )

    async def _poll_and_handle(self) -> None:
        events, next_cursor = await self.client.poll(cursor=self.event_cursor, limit=100)
        if next_cursor:
            self.event_cursor = next_cursor
        elif events:
            last = events[-1]
            self.event_cursor = f"{last.ts.isoformat()}|{last.id}"

        for event in events:
            await self._handle_event(event)

    async def _handle_event(self, event: Event, *, realtime: bool = False, replay: bool = False) -> None:
        if realtime:
            self._advance_cursor_from_event(event)
            self._reset_webhook_poll_deadline()

        if event.id in self._processed_event_ids:
            return
        self._processed_event_ids.add(event.id)

        if event.type.startswith("trigger."):
            await self._handle_trigger(event, replay=replay)
            return

        match event.type:
            case "job.spawn.request":
                await self._handle_spawn(event, replay=replay)
            case "job.completed" | "job.failed" | "job.cancelled":
                await self._handle_job_done(event, replay=replay)
            case "job.started":
                await self._handle_job_started(event, replay=replay)
            case "supervisor.job.launched":
                self._register_replayed_launch(event)
            case "task.created" | "task.completed" | "task.failed" | "task.cancelled":
                pass  # State rebuilt naturally via replay if needed, but handled directly in Trigger/Done for realtime
            case "job.spawn.request":
                await self._handle_spawn(event, replay=replay)
            case "job.completed" | "job.failed" | "job.cancelled":
                await self._handle_job_done(event, replay=replay)
            case "job.started":
                await self._handle_job_started(event, replay=replay)
            case "supervisor.job.launched":
                self._register_replayed_launch(event)

    async def _handle_trigger(self, event: Event, *, replay: bool = False) -> None:
        if event.id in self._launched_event_ids and not replay:
            return

        data = TriggerData.model_validate(event.data)
        if not data.goal:
            logger.warning("Ignoring trigger event %s with no goal", event.id)
            return

        task_id = self._generate_job_id()  # Use UUID7 for root task ID

        self.scheduler.record_task_submission(
            task_id=task_id,
            goal=data.goal,
            source_event_id=event.id,
            spec=data.context,
        )

        if not replay:
            await self.client.emit(
                "task.created",
                TaskCreatedData(
                    task_id=task_id,
                    goal=data.goal,
                    source_trigger_id=event.id,
                ).model_dump(mode="json"),
            )

        self._launched_event_ids.add(event.id)

        # Context might define execution details, otherwise use defaults
        role = data.context.get("role", "default")
        repo = data.context.get("repo", "")
        init_branch = data.context.get("init_branch", "main")
        evo_sha = data.context.get("evo_sha")

        cancelled = await self._enqueue(
            SpawnedJob(
                job_id=self._generate_job_id(),
                source_event_id=event.id,
                task=data.goal,
                role=role,
                repo=repo,
                init_branch=init_branch,
                evo_sha=evo_sha,
                llm_overrides=dict(data.context.get("llm", {})),
                workspace_overrides=dict(data.context.get("workspace", {})),
                publication_overrides=dict(data.context.get("publication", {})),
                task_id=task_id,
            ),
            replay=replay,
        )
        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Initial condition is impossible")

    async def _handle_spawn(self, event: Event, *, replay: bool = False) -> None:
        payload = SpawnRequestData.model_validate(event.data)
        if not payload.tasks:
            return

        plan = self.spawn_handler.expand(event)
        for task in plan.child_tasks:
            self.state.tasks[task.task_id] = task
            if not replay:
                parent_id = task.task_id.rsplit('/', 1)[0]
                await self.client.emit(
                    "task.created",
                    TaskCreatedData(
                        task_id=task.task_id,
                        parent_task_id=parent_id,
                        goal=task.goal,
                        source_trigger_id=task.source_event_id,
                    ).model_dump(mode="json"),
                )

        cancelled: list[SpawnedJob] = []
        for job in plan.jobs:
            cancelled.extend(await self._enqueue(job, replay=replay))

        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Spawn condition is already impossible")

    async def _enqueue(self, job: SpawnedJob, *, replay: bool = False) -> list[SpawnedJob]:
        cancelled = await self.scheduler.enqueue(job)
        if job.job_id in self._pending:
            logger.debug("Pending job %s waiting on condition", job.job_id)
        elif not cancelled:
            logger.debug("Queued job %s", job.job_id)
        return [] if replay else cancelled

    async def _handle_job_done(self, event: Event, *, replay: bool = False) -> None:
        job_id = event.data.get("job_id", "")
        if not job_id:
            return

        is_failure = event.type == "job.failed"
        is_cancelled = event.type == "job.cancelled"
        summary = (
            event.data.get("summary")
            or event.data.get("error")
            or event.data.get("reason")
            or ""
        )

        handle = self.jobs.pop(job_id, None)
        task_id = self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None)).task_id
        
        _, cancelled = await self.scheduler.record_job_terminal(
            job_id=job_id,
            summary=summary,
            failed=is_failure,
            cancelled=is_cancelled,
        )

        if handle is not None and not replay:
            await self._cleanup_handle(handle, failed=is_failure or is_cancelled)
            await self._evaluate_task_termination(job_id=job_id, task_id=task_id, event=event)

        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Condition became impossible")

    async def _evaluate_task_termination(self, job_id: str, task_id: str, event: Event) -> None:
        if not task_id:
            return
        
        task = self.state.tasks.get(task_id)
        if task is None or task.terminal:
            return

        if not self._has_remaining_jobs(task_id):
            if event.type == "job.failed":
                state = "failed"
                await self.scheduler.mark_task_terminal(task_id=task_id, state="failed")
                await self.client.emit(
                    "task.failed",
                    TaskFailedData(task_id=task_id, reason=event.data.get("error", "Job failed")).model_dump(mode="json")
                )
                await self._cascade_cancel(task_id, reason=f"Parent or sibling failed: {event.data.get('error', '')}")
            elif event.type == "job.cancelled":
                state = "cancelled"
                await self.scheduler.mark_task_terminal(task_id=task_id, state="cancelled")
                await self.client.emit(
                    "task.cancelled",
                    TaskCancelledData(task_id=task_id, reason=event.data.get("reason", "Job cancelled")).model_dump(mode="json")
                )
                await self._cascade_cancel(task_id, reason=f"Parent or sibling cancelled: {event.data.get('reason', '')}")
            else:
                state = "completed"
                await self.scheduler.mark_task_terminal(task_id=task_id, state="completed")
                await self.client.emit(
                    "task.completed",
                    TaskCompletedData(task_id=task_id, summary=event.data.get("summary", "")).model_dump(mode="json")
                )

    def _has_remaining_jobs(self, task_id: str) -> bool:
        for job_id, job in self.state.jobs_by_id.items():
            if job.task_id == task_id and job_id not in self.state.completed_jobs:
                return True
        for job_id, job in self.state.pending_jobs.items():
            if job.task_id == task_id:
                return True
        for job in list(self.state.ready_queue._queue):
            if job.task_id == task_id:
                return True
        return False

    async def _cascade_cancel(self, task_id: str, reason: str) -> None:
        prefix = task_id + "/"
        jobs_to_cancel: list[SpawnedJob] = []

        for tid, record in list(self.state.tasks.items()):
            if tid.startswith(prefix) and not record.terminal:
                await self.scheduler.mark_task_terminal(task_id=tid, state="cancelled")
                await self.client.emit(
                    "task.cancelled",
                    TaskCancelledData(task_id=tid, reason=reason).model_dump(mode="json")
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
            await self.scheduler.record_job_terminal(
                job_id=handle.job_id,
                summary=f"Container exited without terminal event (exit_code={handle.exit_code})",
                failed=True,
            )
            await self.client.emit(
                "job.failed",
                {
                    "job_id": handle.job_id,
                    "task_id": self.state.jobs_by_id.get(handle.job_id, SpawnedJob("", "", "", "", "", "", None)).task_id,
                    "error": f"Container exited without emitting terminal event (exit_code={handle.exit_code})",
                    "code": "runtime_lost",
                    "logs_tail": logs[-4000:],
                },
            )
            await self._cleanup_handle(handle, failed=True)

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
        pass  # We no longer redundantly update task states to in_progress based on job.started

    async def _launch(
        self,
        job_id: str,
        task: str,
        role: str,
        repo: str,
        init_branch: str,
        evo_sha: str | None,
        llm_overrides: dict[str, Any] | None = None,
        workspace_overrides: dict[str, Any] | None = None,
        publication_overrides: dict[str, Any] | None = None,
        source_event_id: str = "",
        task_id: str = "",
        job_context=None,
        parent_job_id: str = "",
        condition=None,
    ) -> None:
        spec = self.runtime_builder.build(
            job_id=job_id,
            task_id=task_id or job_id,
            source_event_id=source_event_id,
            task=task,
            role=role,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            llm_overrides=llm_overrides,
            workspace_overrides=workspace_overrides,
            publication_overrides=publication_overrides,
            job_context=job_context,
        )
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
            task=task,
            role=role,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            llm_overrides=dict(llm_overrides or {}),
            workspace_overrides=dict(workspace_overrides or {}),
            publication_overrides=dict(publication_overrides or {}),
            task_id=task_id or job_id,
            condition=condition,
            job_context=job_context or self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None)).job_context,
            parent_job_id=parent_job_id,
        )
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=repo,
            init_branch=init_branch,
            role=role,
            evo_sha=evo_sha,
            llm_overrides=dict(llm_overrides or {}),
            workspace_overrides=dict(workspace_overrides or {}),
            publication_overrides=dict(publication_overrides or {}),
            task_id=task_id or job_id,
        )

        launch = SupervisorJobLaunchedData(
            job_id=job_id,
            task_id=task_id or job_id,
            source_event_id=source_event_id,
            task=task,
            role=role,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha or "",
            llm=dict(llm_overrides or {}),
            workspace=dict(workspace_overrides or {}),
            publication=dict(publication_overrides or {}),
            runtime_kind=self.runtime_defaults.kind,
            container_id=handle.container_id,
            container_name=handle.container_name,
            parent_job_id=parent_job_id,
            condition=condition_to_data(condition),
        )
        await self.client.emit("supervisor.job.launched", launch.model_dump(mode="json"))

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
                "job.cancelled",
                JobCancelledData(job_id=job.job_id, task_id=task_id, reason=reason).model_dump(mode="json"),
            )
            await self.scheduler.record_job_terminal(job_id=job.job_id, summary=reason, cancelled=True)
            
            # Since these jobs are now definitively terminated, evaluate if their tasks are also terminal
            await self._evaluate_task_termination(
                job_id=job.job_id, 
                task_id=task_id, 
                event=Event(id="", type="job.cancelled", source_id="", ts=datetime.now(timezone.utc), data={"reason": reason})
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

    def _register_replayed_launch(self, event: Event) -> None:
        data = event.data
        job_id = data.get("job_id", "")
        if not job_id:
            return

        source_event_id = data.get("source_event_id", "")
        self.state.discard_scheduled_by_source_event(source_event_id, keep_job_id=job_id)

        existing = self.state.jobs_by_id.get(job_id)
        self.state.jobs_by_id[job_id] = existing or SpawnedJob(
            job_id=job_id,
            source_event_id=source_event_id,
            task=data.get("task", ""),
            role=data.get("role", "default"),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            evo_sha=data.get("evo_sha") or None,
            llm_overrides=dict(data.get("llm") or {}),
            workspace_overrides=dict(data.get("workspace") or {}),
            publication_overrides=dict(data.get("publication") or {}),
            task_id=data.get("task_id", "") or job_id,
            condition=condition_from_data(data.get("condition")),
            parent_job_id=data.get("parent_job_id", ""),
        )
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            role=data.get("role", "default"),
            evo_sha=data.get("evo_sha") or None,
            llm_overrides=dict(data.get("llm") or {}),
            workspace_overrides=dict(data.get("workspace") or {}),
            publication_overrides=dict(data.get("publication") or {}),
            task_id=data.get("task_id", "") or job_id,
        )
        if source_event_id:
            self._launched_event_ids.add(source_event_id)
        self.state.remove_pending_job(job_id)
        self.state.drop_from_ready_queue(job_id)



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
