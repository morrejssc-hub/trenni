"""Core supervisor: event routing + scheduler + isolation backend."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from yoitsu_contracts.conditions import condition_from_data, condition_to_data
from yoitsu_contracts.events import (
    JobCancelledData,
    JobCompletedData,
    JobFailedData,
    SpawnRequestData,
    SupervisorCheckpointData,
    SupervisorJobLaunchedData,
    TaskSubmitData,
    TaskUpdatedData,
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
                    "task.submit",
                    "task.updated",
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

        match event.type:
            case "task.submit":
                await self._handle_task_submit(event, replay=replay)
            case "task.updated":
                await self._handle_task_updated(event, replay=replay)
            case "job.spawn.request":
                await self._handle_spawn(event, replay=replay)
            case "job.completed" | "job.failed" | "job.cancelled":
                await self._handle_job_done(event, replay=replay)
            case "job.started":
                await self._handle_job_started(event, replay=replay)
            case "supervisor.job.launched":
                self._register_replayed_launch(event)

    async def _handle_task_submit(self, event: Event, *, replay: bool = False) -> None:
        if event.id in self._launched_event_ids and not replay:
            return

        data = TaskSubmitData.model_validate(event.data)
        if not data.task:
            logger.warning("Ignoring empty task.submit %s", event.id)
            return

        task_id = data.task_id or event.id
        self.scheduler.record_task_submission(
            task_id=task_id,
            task=data.task,
            source_event_id=event.id,
            role=data.role,
            repo=data.repo,
            init_branch=data.init_branch,
            evo_sha=data.evo_sha,
        )
        self._launched_event_ids.add(event.id)

        cancelled = await self._enqueue(
            SpawnedJob(
                job_id=self._generate_job_id(),
                source_event_id=event.id,
                task=data.task,
                role=data.role,
                repo=data.repo,
                init_branch=data.init_branch,
                evo_sha=data.evo_sha,
                llm_overrides=dict(data.llm),
                workspace_overrides=dict(data.workspace),
                publication_overrides=dict(data.publication),
                task_id=task_id,
            ),
            replay=replay,
        )
        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Initial condition is impossible")

    async def _handle_task_updated(self, event: Event, *, replay: bool = False) -> None:
        payload = TaskUpdatedData.model_validate(event.data)
        _, cancelled = await self.scheduler.update_task_state(
            task_id=payload.task_id,
            status=payload.status,
            summary=payload.summary,
        )
        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Dependent task state made condition impossible")

    async def _handle_spawn(self, event: Event, *, replay: bool = False) -> None:
        payload = SpawnRequestData.model_validate(event.data)
        if not payload.tasks:
            return

        plan = self.spawn_handler.expand(event)
        parent_task_id = payload.task_id or self.state.jobs_by_id.get(payload.job_id, SpawnedJob("", "", "", "", "", "", None)).task_id

        if parent_task_id:
            _, cancelled = await self.scheduler.update_task_state(
                task_id=parent_task_id,
                status="in_progress",
                summary=f"Spawned {len(plan.child_tasks)} child task(s)",
            )
            if cancelled and not replay:
                await self._emit_cancellations(cancelled, reason="Parent task update made condition impossible")

        for task in plan.child_tasks:
            self.state.tasks[task.task_id] = task

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
        _, cancelled = await self.scheduler.record_job_terminal(
            job_id=job_id,
            summary=summary,
            failed=is_failure,
            cancelled=is_cancelled,
        )

        if handle is not None and not replay:
            await self._cleanup_handle(handle, failed=is_failure or is_cancelled)

        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Condition became impossible")

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
        task_id = event.data.get("task_id", "")
        if task_id:
            _, cancelled = await self.scheduler.update_task_state(
                task_id=task_id,
                status="in_progress",
                summary=self.scheduler.task_summary(task_id),
            )
            if cancelled and not replay:
                await self._emit_cancellations(cancelled, reason="Condition became impossible")

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
        await self.scheduler.update_task_state(
            task_id=task_id or job_id,
            status="in_progress",
            summary=self.scheduler.task_summary(task_id or job_id),
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
            await self.scheduler.update_task_state(task_id=task_id, status="cancelled", summary=reason)
            await self.client.emit(
                "task.updated",
                TaskUpdatedData(task_id=task_id, status="cancelled", summary=reason).model_dump(mode="json"),
            )
            await self.client.emit(
                "job.cancelled",
                JobCancelledData(job_id=job.job_id, task_id=task_id, reason=reason).model_dump(mode="json"),
            )
            await self.scheduler.record_job_terminal(job_id=job.job_id, summary=reason, cancelled=True)

    def _poll_due_now(self) -> bool:
        if not self._webhook_active:
            return True
        return asyncio.get_running_loop().time() >= self._webhook_poll_not_before

    def _reset_webhook_poll_deadline(self) -> None:
        if self._webhook_active:
            self._webhook_poll_not_before = asyncio.get_running_loop().time() + self.config.webhook_poll_interval

    def _advance_cursor_from_event(self, event: Event) -> None:
        current = self._cursor_key(self.event_cursor)
        candidate = (event.ts, event.id)
        if current is None or candidate > current:
            self.event_cursor = f"{event.ts.isoformat()}|{event.id}"

    @staticmethod
    def _cursor_key(cursor: str | None) -> tuple[datetime, str] | None:
        if not cursor:
            return None
        try:
            ts_raw, event_id = cursor.split("|", 1)
            return datetime.fromisoformat(ts_raw), event_id
        except ValueError:
            return None

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

    def _spawned_job_from_event(self, event: Event) -> SpawnedJob:
        data = TaskSubmitData.model_validate(event.data)
        task_id = data.task_id or event.id
        return SpawnedJob(
            source_event_id=event.id,
            job_id=self._generate_job_id(),
            task=data.task,
            role=data.role,
            repo=data.repo,
            init_branch=data.init_branch,
            evo_sha=data.evo_sha,
            llm_overrides=dict(data.llm),
            workspace_overrides=dict(data.workspace),
            publication_overrides=dict(data.publication),
            task_id=task_id,
        )

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
