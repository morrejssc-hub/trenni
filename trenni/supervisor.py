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

from yoitsu_contracts.artifact import ArtifactBinding
from yoitsu_contracts.observation import (
    ObservationBudgetVarianceData,
    OBSERVATION_BUDGET_VARIANCE,
)
from yoitsu_contracts.conditions import condition_from_data, condition_to_data
from yoitsu_contracts.config import EvalContextConfig, JobContextConfig, JoinContextConfig, AnalyzerVersion
from yoitsu_contracts.events import (
    EvalSpec,
    JobCancelledData,
    JobCompletedData,
    SemanticVerdict,
    SpawnRequestData,
    StructuralVerdict,
    SupervisorCheckpointData,
    SupervisorJobEnqueuedData,
    SupervisorJobFailedData,
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
from .workspace_manager import WorkspaceManager

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
        self._job_git_refs = self.state.job_git_refs
        self._job_completion_codes = self.state.job_completion_codes
        self._processed_event_ids = self.state.processed_event_ids
        self._processing_event_ids: set[str] = set()
        self._event_processing_lock = asyncio.Lock()
        self._launched_event_ids = self.state.launched_event_ids
        self._started_job_ids: set[str] = set()
        self._spawn_defaults_by_job = self.state.spawn_defaults_by_job
        self._job_temp_dirs: dict[str, list[Path]] = {}  # ADR-0015: workspace cleanup

        self.scheduler = Scheduler(self.state, max_workers=config.max_workers, bundles=config.bundles)
        self.spawn_handler = SpawnHandler(self.state)
        self.workspace_manager = WorkspaceManager(config)  # ADR-0015: clone bundle/target

        self._checkpoint_cycles = _DEFAULT_CHECKPOINT_CYCLES
        self._reap_timeout = self.runtime_defaults.cleanup_timeout_seconds

        self._webhook_id: str | None = None
        self._webhook_active = False
        self._webhook_poll_not_before: float = 0.0
        
        # Role catalog with SHA-based invalidation per ADR-0007
        self._role_catalog_cache: dict[str, dict[str, Any]] | None = None
        self._role_metadata_reader: RoleMetadataReader | None = None
        self._cached_bundle_sha: str | None = None
        
        # Observation aggregation state (ADR-0010 extension)
        self._last_aggregation: float = 0.0
        # Ordered FIFO tracking for processed observation events
        # Using list + set to maintain insertion order while allowing fast lookup
        self._processed_observation_ids_order: list[str] = []
        self._processed_observation_ids_set: set[str] = set()
        
        # ADR-0017: Read analyzer version SHAs from environment variables
        # These are constant for the supervisor's lifetime (not dynamic)
        import os
        from yoitsu_contracts import AnalyzerVersion
        
        trenni_sha = os.environ.get(config.trenni_sha_env, "")[:12]
        palimpsest_sha = os.environ.get(config.palimpsest_sha_env, "")[:12]
        bundle_sha = os.environ.get(config.bundle_sha_env, "")[:12]  # Global fallback
        
        if trenni_sha and palimpsest_sha and bundle_sha:
            self._analyzer_version = AnalyzerVersion(
                bundle_sha=bundle_sha,
                trenni_sha=trenni_sha,
                palimpsest_sha=palimpsest_sha,
            )
            logger.info(
                "Analyzer version initialized: bundle=%s, trenni=%s, palimpsest=%s",
                bundle_sha, trenni_sha, palimpsest_sha
            )
        else:
            self._analyzer_version = None
            logger.warning(
                "Missing analyzer version SHAs: bundle=%s, trenni=%s, palimpsest=%s. "
                "Set YOITSU_BUNDLE_SHA, YOITSU_TRENNI_SHA, YOITSU_PALIMPSEST_SHA env vars. "
                "Observation events will have empty analyzer_version.",
                bundle_sha or "<missing>",
                trenni_sha or "<missing>",
                palimpsest_sha or "<missing>"
            )
        self._max_processed_observation_ids: int = 1000  # FIFO prune limit

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
            # Per Bundle MVP: no role catalog validation needed
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
        import time
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

            # 定时聚合 observation (ADR-0010 extension)
            now = time.time()
            if now - self._last_aggregation >= self.config.observation_aggregation_interval:
                try:
                    await self._aggregate_and_spawn_optimizer()
                except Exception:
                    logger.exception("Error in observation aggregation")
                self._last_aggregation = now

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

            # Re-check bundle capacity right before launch (Issue 5 fix)
            if not self.scheduler.has_bundle_capacity(job.bundle):
                # Team at capacity - put back in pending
                self._pending[job.job_id] = job
                logger.debug("Job %s bundle %s at capacity, returning to pending", job.job_id, job.bundle)
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
                # Re-check bundle capacity during wait loop
                if not self.scheduler.has_bundle_capacity(job.bundle):
                    self._pending[job.job_id] = job
                    logger.debug("Job %s bundle %s at capacity during wait, returning to pending", job.job_id, job.bundle)
                    break
            else:
                try:
                    await self._launch_from_spawned(job)
                except Exception as exc:
                    logger.exception("Failed to launch queued job %s", job.job_id)
                    await self._fail_job_before_launch(
                        job_id=job.job_id,
                        error=f"Runtime launch failed for job {job.job_id}: {exc}",
                        code="runtime_launch_failed",
                    )
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
            bundle=job.bundle,
            repo=job.repo,
            init_branch=job.init_branch,
            bundle_sha=job.bundle_sha,
            budget=job.budget,
            source_event_id=job.source_event_id,
            job_context=job.job_context,
            parent_job_id=job.parent_job_id,
            condition=job.condition,
            input_artifacts=job.input_artifacts,  # ADR-0013
            analyzer_version=job.analyzer_version,  # ADR-0017
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
            elif event.type == "external.event":
                await self._handle_external_event(event, replay=replay)
            else:
                match event.type:
                    case "agent.job.spawn_request":
                        await self._handle_spawn(event, replay=replay)
                    case "supervisor.job.enqueued":
                        await self._handle_job_enqueued(event, replay=replay)
                    case "agent.job.completed" | "agent.job.failed" | "agent.job.cancelled" | "supervisor.job.failed":
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

        await self._process_trigger(event, data, replay=replay)

    async def _handle_external_event(self, event: Event, *, replay: bool = False) -> None:
        """Handle external events (CI failure, labeled issues/PRs, observation thresholds).

        Converts external events to TriggerData and processes them.
        """
        from yoitsu_contracts.external_events import (
            CIFailureEvent,
            IssueLabeledEvent,
            PRLabeledEvent,
            ObservationThresholdEvent,
            ci_failure_to_trigger,
            issue_labeled_to_trigger,
            pr_labeled_to_trigger,
            observation_threshold_to_trigger,
        )

        event_type = event.data.get("event_type", "")
        trigger_data = None

        try:
            if event_type == "ci_failure":
                external = CIFailureEvent.model_validate(event.data)
                trigger_data = ci_failure_to_trigger(external)
            elif event_type == "issue_labeled":
                external = IssueLabeledEvent.model_validate(event.data)
                trigger_data = issue_labeled_to_trigger(external)
            elif event_type == "pr_labeled":
                external = PRLabeledEvent.model_validate(event.data)
                trigger_data = pr_labeled_to_trigger(external)
            elif event_type == "observation_threshold":
                external = ObservationThresholdEvent.model_validate(event.data)
                trigger_data = observation_threshold_to_trigger(external)
            else:
                logger.warning("Unknown external event type: %s", event_type)
                return
        except Exception as e:
            logger.warning("Invalid external event %s: %s", event.id, e)
            return

        if trigger_data is None:
            logger.info("External event %s not mapped to trigger", event.id)
            return

        # Convert to TriggerData and process
        try:
            data = TriggerData.model_validate(trigger_data)
        except Exception as e:
            logger.warning("Invalid trigger data from external event %s: %s", event.id, e)
            return

        await self._process_trigger(event, data, replay=replay)

    async def _process_trigger(self, event: Event, data: TriggerData, *, replay: bool = False) -> None:
        """Process a trigger event after validation.

        Shared logic for both direct triggers and external events.
        Per Bundle MVP: bundle is required; role defaults to bundle's default_role.
        """
        # goal is now required by pydantic (min_length=1), no need to check

        task_id = self._root_task_id(event.id)
        root_job_id = f"{task_id}-root"
        bundle = str(data.bundle or "").strip()
        role = str(data.role or "").strip()

        if not bundle:
            await self._reject_task_submission(
                task_id=task_id,
                bundle="",
                goal=data.goal,
                source_trigger_id=event.id,
                reason="Trigger missing bundle field",
                role=role,
                budget=data.budget,
                replay=replay,
            )
            return

        # Resolve role: explicit > bundle default > reject
        if not role:
            bundle_config = self.config.bundles.get(bundle)
            if bundle_config and bundle_config.default_role:
                role = bundle_config.default_role
                logger.info(f"Using default_role '{role}' for bundle '{bundle}'")
            else:
                await self._reject_task_submission(
                    task_id=task_id,
                    bundle=bundle,
                    goal=data.goal,
                    source_trigger_id=event.id,
                    reason=f"Trigger missing role field and bundle '{bundle}' has no default_role configured",
                    role="",
                    budget=data.budget,
                    replay=replay,
                )
                return

        self.scheduler.record_task_submission(
            task_id=task_id,
            goal=data.goal,
            source_event_id=event.id,
            spec={"bundle": bundle, "budget": data.budget, "role": role},
            eval_spec=data.eval_spec,
        )
        if task_id in self.state.tasks:
            self.state.tasks[task_id].bundle = bundle

        if not replay:
            await self.client.emit(
                "supervisor.task.created",
                TaskCreatedData(
                    task_id=task_id,
                    bundle=bundle,
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
        role_params = dict(data.params)

        repo = data.repo
        init_branch = data.init_branch or "main"
        bundle_sha = data.sha

        root_job = SpawnedJob(
            job_id=root_job_id,
            source_event_id=event.id,
            goal=data.goal,
            role=role,
            role_params=role_params,
            repo=repo,
            init_branch=init_branch,
            bundle_sha=bundle_sha,
            budget=data.budget,
            task_id=task_id,
            bundle=bundle,
            input_artifacts=list(data.input_artifacts),  # ADR-0013
            analyzer_version=self._analyzer_version,  # ADR-0017
        )
        validation_error = self._validate_spawned_job(root_job)
        if validation_error:
            if not replay:
                await self._emit_task_terminal(
                    task_id=task_id,
                    state="failed",
                    result=TaskResult(),
                    reason=validation_error,
                )
            return

        cancelled = await self._enqueue(root_job, replay=replay)
        if cancelled and not replay:
            await self._emit_cancellations(cancelled, reason="Initial condition is impossible")
        self._launched_event_ids.add(event.id)

    async def _reject_task_submission(
        self,
        *,
        task_id: str,
        bundle: str,
        goal: str,
        source_trigger_id: str,
        reason: str,
        role: str,
        budget: float,
        replay: bool,
    ) -> None:
        self.scheduler.record_task_submission(
            task_id=task_id,
            goal=goal,
            source_event_id=source_trigger_id,
            spec={"bundle": bundle, "budget": budget, "role": role},
        )
        task = self.state.tasks.get(task_id)
        if task is not None:
            task.bundle = bundle

        if replay:
            return

        await self.client.emit(
            "supervisor.task.created",
            TaskCreatedData(
                task_id=task_id,
                bundle=bundle,
                goal=goal,
                source_trigger_id=source_trigger_id,
            ).model_dump(mode="json"),
            idempotency_key=self._event_idempotency_key(
                source_event_id=source_trigger_id,
                event_type="supervisor.task.created",
                entity_id=task_id,
            ),
        )
        await self._emit_task_terminal(
            task_id=task_id,
            state="failed",
            result=TaskResult(),
            reason=reason,
        )

    async def _handle_spawn(self, event: Event, *, replay: bool = False) -> None:
        payload = SpawnRequestData.model_validate(event.data)
        if not payload.tasks:
            return

        # Per Bundle MVP: no role catalog preloading needed
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
                        bundle=task.bundle,
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
            validation_error = self._validate_spawned_job(job)
            if validation_error:
                if not replay:
                    await self._emit_task_terminal(
                        task_id=job.task_id or job.job_id,
                        state="failed",
                        result=TaskResult(),
                        reason=validation_error,
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

        is_failure = event.type in {"agent.job.failed", "supervisor.job.failed"}
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
        self._started_job_ids.discard(job_id)

        handle = self.jobs.pop(job_id, None)
        job_record = self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None))
        task_id = job_record.task_id
        bundle = job_record.bundle
        
        # ADR-0011 D5: Track running jobs per bundle for launch conditions
        if bundle:
            self.state.decrement_bundle_running(bundle)
        
        # ADR-0015: Cleanup workspaces after job terminal
        temp_dirs = self._job_temp_dirs.pop(job_id, [])
        if temp_dirs:
            self.workspace_manager.cleanup(temp_dirs)
        
        _, cancelled = await self.scheduler.record_job_terminal(
            job_id=job_id,
            summary=summary,
            failed=is_failure,
            cancelled=is_cancelled,
        )

        if handle is not None and not replay:
            await self._cleanup_handle(handle, failed=is_failure or is_cancelled)

        # ADR-0010 D5: Emit budget_variance observation for completed jobs
        # Skip optimizer/implementer roles to prevent self-optimization loop
        if not replay and not is_failure and not is_cancelled and job_record and job_record.role not in ("optimizer", "implementer"):
            await self._emit_budget_variance(job_id, event)
        
        # ADR-0017: Run observation analyzers after job terminal
        # Skip optimizer/implementer roles to prevent cascade
        if not replay and job_record and job_record.role not in ("optimizer", "implementer"):
            await self._run_observation_analyzers(job_id, event, job_record)

        # ADR-0010: Handle optimizer output - parse ReviewProposal and spawn optimization task
        if not replay and not is_failure and not is_cancelled:
            await self._handle_optimizer_output(job_id, job_record, event)

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

    async def _fail_job_before_launch(
        self,
        *,
        job_id: str,
        error: str,
        code: str,
    ) -> None:
        self.jobs.pop(job_id, None)
        self._started_job_ids.discard(job_id)

        job_record = self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None))
        task_id = job_record.task_id or job_id

        temp_dirs = self._job_temp_dirs.pop(job_id, [])
        if temp_dirs:
            self.workspace_manager.cleanup(temp_dirs)

        self._job_completion_codes[job_id] = code
        await self.scheduler.record_job_terminal(
            job_id=job_id,
            summary=error,
            failed=True,
        )

        payload = SupervisorJobFailedData(
            job_id=job_id,
            task_id=task_id,
            error=error,
            code=code,
        ).model_dump(mode="json", exclude_none=True)
        emitted_id = await self.client.emit(
            "supervisor.job.failed",
            payload,
            idempotency_key=self._event_idempotency_key(
                source_event_id=job_record.source_event_id or job_id,
                event_type="supervisor.job.failed",
                entity_id=job_id,
            ),
        )
        if emitted_id:
            self._processed_event_ids.add(emitted_id)

        await self._evaluate_task_termination(
            job_id=job_id,
            task_id=task_id,
            event=SimpleNamespace(
                id=emitted_id or "",
                source_id=self.config.source_id,
                type="supervisor.job.failed",
                data=payload,
            ),
        )

    async def _close_runtime_failed_job(
        self,
        handle: JobHandle,
        *,
        error: str,
        code: str,
        logs_tail: str = "",
        cleanup_handle: bool = True,
    ) -> None:
        self.jobs.pop(handle.job_id, None)
        self._started_job_ids.discard(handle.job_id)

        job_record = self.state.jobs_by_id.get(handle.job_id, SpawnedJob("", "", "", "", "", "", None))
        task_id = job_record.task_id or handle.job_id
        bundle = job_record.bundle

        if bundle:
            self.state.decrement_bundle_running(bundle)

        temp_dirs = self._job_temp_dirs.pop(handle.job_id, [])
        if temp_dirs:
            self.workspace_manager.cleanup(temp_dirs)

        self._job_completion_codes[handle.job_id] = code
        await self.scheduler.record_job_terminal(
            job_id=handle.job_id,
            summary=error,
            failed=True,
        )

        payload = SupervisorJobFailedData(
            job_id=handle.job_id,
            task_id=task_id,
            error=error,
            code=code,
            container_id=handle.container_id,
            container_name=handle.container_name,
            exit_code=handle.exit_code,
            logs_tail=logs_tail[-4000:],
        ).model_dump(mode="json", exclude_none=True)
        emitted_id = await self.client.emit(
            "supervisor.job.failed",
            payload,
            idempotency_key=self._event_idempotency_key(
                source_event_id=job_record.source_event_id or handle.job_id,
                event_type="supervisor.job.failed",
                entity_id=handle.job_id,
            ),
        )
        if emitted_id:
            self._processed_event_ids.add(emitted_id)

        if cleanup_handle:
            await self._cleanup_handle(handle, failed=True)

        await self._evaluate_task_termination(
            job_id=handle.job_id,
            task_id=task_id,
            event=SimpleNamespace(
                id=emitted_id or "",
                source_id=self.config.source_id,
                type="supervisor.job.failed",
                data=payload,
            ),
        )

    async def _mark_exited_jobs(self) -> None:
        previous_exit_times = {
            job_id: handle.exited_at
            for job_id, handle in self.jobs.items()
        }
        await mark_exited_jobs(self.jobs, self.backend)
        for job_id, handle in list(self.jobs.items()):
            if handle.exited_at is None:
                continue
            if previous_exit_times.get(job_id) is not None:
                continue
            if job_id in self._started_job_ids:
                continue
            if job_id not in self.state.jobs_by_id:
                continue
            try:
                logs = await self.backend.logs(handle)
            except Exception:
                logs = ""
            await self._close_runtime_failed_job(
                handle,
                error=f"Container exited before agent.job.started (exit_code={handle.exit_code})",
                code="runtime_exit_before_start",
                logs_tail=logs,
            )

    async def _checkpoint(self) -> None:
        reaped = await reap_timed_out_jobs(
            self.jobs,
            backend=self.backend,
            reap_timeout=self._reap_timeout,
        )

        for handle, logs in reaped:
            await self._close_runtime_failed_job(
                handle,
                error=f"Container exited without emitting terminal event (exit_code={handle.exit_code})",
                code="runtime_lost",
                logs_tail=logs,
            )

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
        self._started_job_ids.add(job_id)
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
        bundle_sha: str | None,
        budget: float | None = None,
        source_event_id: str = "",
        task_id: str = "",
        job_context=None,
        parent_job_id: str = "",
        condition=None,
        bundle: str = "",
        role_params: dict[str, Any] | None = None,
        input_artifacts: list | None = None,  # ADR-0013
        analyzer_version=None,  # ADR-0017: AnalyzerVersion
    ) -> None:
        """Launch a job in the isolation backend."""
        
        # ADR-0015: Prepare bundle and target workspaces
        prepared = self.workspace_manager.prepare(
            job_id=job_id,
            bundle=bundle,
            repo=repo,
            init_branch=init_branch,
            bundle_sha=bundle_sha,  # Bundle SHA from trigger/spawn
        )
        
        # Track temp dirs for cleanup
        self._job_temp_dirs[job_id] = prepared.temp_dirs

        if bundle and not prepared.bundle_source:
            await self._fail_job_before_launch(
                job_id=job_id,
                error=f"Bundle workspace preparation failed for bundle {bundle!r}",
                code="bundle_workspace_prepare_failed",
            )
            return

        if repo and not prepared.target_source:
            await self._fail_job_before_launch(
                job_id=job_id,
                error=f"Target workspace preparation failed for repo {repo!r}",
                code="target_workspace_prepare_failed",
            )
            return
        
        spec = self.runtime_builder.build(
            job_id=job_id,
            task_id=task_id or job_id,
            source_event_id=source_event_id,
            goal=goal,
            role=role,
            role_params=role_params,
            bundle=bundle,
            repo=repo,
            init_branch=init_branch,
            bundle_sha=bundle_sha,
            budget=budget,
            job_context=job_context,
            input_artifacts=input_artifacts,  # ADR-0013
            analyzer_version=analyzer_version,  # ADR-0017
            bundle_source=prepared.bundle_source,  # ADR-0015
            target_source=prepared.target_source,  # ADR-0015
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
            bundle=bundle,
            repo=repo,
            init_branch=init_branch,
            bundle_sha=bundle_sha,
            budget=budget or 0.0,
            task_id=task_id or job_id,
            condition=condition,
            job_context=job_context or self.state.jobs_by_id.get(job_id, SpawnedJob("", "", "", "", "", "", None)).job_context,
            parent_job_id=parent_job_id,
            analyzer_version=analyzer_version,  # ADR-0017
        )
        
        # ADR-0011 D5: Track running jobs per bundle for launch conditions
        self.state.increment_bundle_running(bundle)
        
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=repo,
            init_branch=init_branch,
            role=role,
            role_params=dict(role_params or {}),
            bundle=bundle,
            bundle_sha=bundle_sha,
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

    def _bundle_root(self) -> Path:
        """Get the bundle root directory."""
        if self.config.bundle_root:
            return Path(self.config.bundle_root)
        return Path(__file__).resolve().parents[2] / "palimpsest" / "bundle"

    def _read_bundle_sha(self) -> str:
        """Read current HEAD SHA from bundle root for cache invalidation."""
        bundle_root = self._bundle_root()
        try:
            result = subprocess.run(
                ["git", "-C", str(bundle_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _allocated_job_budget(self, job: SpawnedJob) -> float:
        """Get the allocated budget for a job.

        Per Bundle MVP: budget is read from SpawnedJob.budget.
        Fallback to TrenniConfig.default_llm.max_total_cost if not set.
        """
        # Primary: budget field on SpawnedJob
        if job.budget and job.budget > 0:
            return float(job.budget)
        
        # Fallback: default from TrenniConfig
        default_budget = self.config.default_llm.get("max_total_cost")
        if isinstance(default_budget, (int, float)) and default_budget > 0:
            return float(default_budget)
        return 0.0

    def _validate_spawned_job_budget(self, job: SpawnedJob) -> str | None:
        """Validate job budget. Per Bundle MVP, always returns None (no role metadata validation)."""
        return None

    def _validate_spawned_job(self, job: SpawnedJob) -> str | None:
        if not job.bundle:
            return "Job missing bundle field"
        bundle_config = self.config.bundles.get(job.bundle)
        if bundle_config is None:
            return f"Unknown bundle {job.bundle!r}"
        if not bundle_config.source.url:
            return f"Bundle {job.bundle!r} has no configured source"
        return self._validate_spawned_job_budget(job)

    def _register_replayed_launch(self, event: Event) -> None:
        """Register a job from a replayed supervisor.job.launched event.

        Per ADR-0007: SpawnedJob and SpawnDefaults no longer carry execution config overrides.
        """
        data = event.data
        job_id = data.get("job_id", "")
        if not job_id:
            return

        source_event_id = data.get("source_event_id", "")

        # ADR-0013: restore input_artifacts from event data
        input_artifacts_data = data.get("input_artifacts", [])
        input_artifacts = [
            ArtifactBinding.model_validate(b) for b in input_artifacts_data
        ] if input_artifacts_data else []

        existing = self.state.jobs_by_id.get(job_id)
        self.state.jobs_by_id[job_id] = existing or SpawnedJob(
            job_id=job_id,
            source_event_id=source_event_id,
            goal=data.get("goal", ""),
            role=data.get("role", "default"),
            bundle=data.get("bundle", ""),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            bundle_sha=data.get("bundle_sha") or None,
            budget=data.get("budget", 0.0),
            task_id=data.get("task_id", "") or job_id,
            condition=condition_from_data(data.get("condition")),
            parent_job_id=data.get("parent_job_id", ""),
            input_artifacts=input_artifacts,  # ADR-0013
            analyzer_version=AnalyzerVersion.model_validate(data.get("analyzer_version")) if data.get("analyzer_version") else None,  # ADR-0017
        )
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            role=data.get("role", "default"),
            bundle_sha=data.get("bundle_sha") or None,
            task_id=data.get("task_id", "") or job_id,
            bundle=data.get("bundle", ""),
            budget=data.get("budget", 0.0),
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
        # Per Bundle MVP: role must be specified in eval_spec or default to 'evaluator'
        role = eval_spec.role or "evaluator"
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
            bundle=task.bundle,
            repo=base_job.repo if base_job else "",
            init_branch=eval_branch,
            bundle_sha=base_job.bundle_sha if base_job else None,
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
        Uses job.analyzer_version for three-party SHA (ADR-0017).
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
            
            # Build analyzer_version dict (ADR-0017)
            analyzer_version_dict = {}
            if job.analyzer_version:
                analyzer_version_dict = job.analyzer_version.model_dump(mode="json")
            
            await self.client.emit(
                OBSERVATION_BUDGET_VARIANCE,
                ObservationBudgetVarianceData(
                    task_id=job.task_id or job_id,
                    job_id=job_id,
                    role=job.role,
                    bundle=job.bundle,
                    estimated_budget=estimated_budget,
                    actual_cost=actual_cost,
                    variance_ratio=variance_ratio,
                    analyzer_version=analyzer_version_dict,
                ).model_dump(),
            )

    async def _run_observation_analyzers(
        self,
        job_id: str,
        event: Event,
        job: SpawnedJob | None,
    ) -> None:
        """Run observation analyzers after job terminal (ADR-0017).
        
        Per ADR-0017: Trenni loads bundle-provided analyzers, queries pasloe for
        job events, and emits observation.* events from analyzer results.
        """
        from yoitsu_contracts.observation import ObservationToolRepetitionEvent, OBSERVATION_TOOL_REPETITION
        
        if job is None:
            return
        
        # Skip optimizer/implementer to prevent cascade
        if job.role in ("optimizer", "implementer"):
            return
        
        task_id = job.task_id or job_id
        role = job.role
        bundle = job.bundle
        
        # Build analyzer_version from job
        analyzer_version_dict = {}
        if job.analyzer_version:
            analyzer_version_dict = job.analyzer_version.model_dump(mode="json")
        
        # Query pasloe for job events (tool.exec, tool.result)
        job_events = await self._fetch_job_events(job_id)
        
        # Load bundle analyzer
        analyzer = self._load_bundle_analyzer(bundle, "tool_repetition")
        if analyzer is None:
            # Fallback to builtin (for non-bundle jobs)
            from .observation_analyzers import get_analyzer
            analyzer = get_analyzer("tool_repetition")
        
        if analyzer:
            try:
                observations = analyzer.analyze(job_events)
                
                # Emit observation events
                for obs in observations:
                    await self.client.emit(
                        OBSERVATION_TOOL_REPETITION,
                        ObservationToolRepetitionEvent(
                            job_id=job_id,
                            task_id=task_id,
                            role=role,
                            bundle=bundle,
                            tool_name=obs.get("tool_name", ""),
                            call_count=obs.get("call_count", 0),
                            arg_pattern=obs.get("arg_pattern", ""),
                            similarity=obs.get("similarity", 0.0),
                            analyzer_version_bundle_sha=analyzer_version_dict.get("bundle_sha", ""),
                            analyzer_version_trenni_sha=analyzer_version_dict.get("trenni_sha", ""),
                            analyzer_version_palimpsest_sha=analyzer_version_dict.get("palimpsest_sha", ""),
                        ).model_dump(),
                    )
            except Exception as e:
                logger.warning("Observation analyzer failed for job %s: %s", job_id, e)

    async def _fetch_job_events(self, job_id: str) -> list[dict]:
        """Fetch all events for a job from pasloe.
        
        Queries agent.tool.exec and agent.tool.result events.
        """
        import os
        events = []
        api_key = os.environ.get(self.config.pasloe_api_key_env, "")
        
        # Query tool.exec events via HTTP
        import httpx
        headers = {"X-API-Key": api_key} if api_key else {}
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config.pasloe_url}/events",
                    params={"type": "agent.tool.exec", "limit": 100, "order": "desc"},
                    headers=headers,
                )
                resp.raise_for_status()
                for evt in resp.json():
                    if evt.get("data", {}).get("job_id") == job_id:
                        events.append({
                            "type": evt.get("type", ""),
                            "ts": evt.get("ts", ""),
                            "data": evt.get("data", {}),
                        })
        except Exception as e:
            logger.warning("Failed to fetch tool.exec events: %s", e)
        
        return events

    def _load_bundle_analyzer(self, bundle: str, analyzer_name: str):
        """Load analyzer from bundle workspace.
        
        Args:
            bundle: Bundle name
            analyzer_name: Analyzer name (e.g., "tool_repetition")
        
        Returns:
            Analyzer module with analyze() function, or None
        """
        import importlib.util
        
        bundle_config = self.config.bundles.get(bundle)
        if not bundle_config:
            return None
        
        # Find analyzer in bundle workspace
        # TODO: Use bundle_sha to find the right workspace
        # For now, look in any available bundle workspace
        for job_record in self.state.jobs_by_id.values():
            if job_record.bundle == bundle and job_record.bundle_sha:
                # Try to find workspace from prepared record
                ws_key = f"{job_record.job_id}-bundle"
                ws_path = self.workspace_manager._base_dir / ws_key
                if ws_path.exists():
                    analyzer_path = ws_path / "scripts" / "analyzers" / f"{analyzer_name}.py"
                    if analyzer_path.exists():
                        try:
                            spec = importlib.util.spec_from_file_location(
                                f"bundle_analyzer_{analyzer_name}",
                                analyzer_path,
                            )
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            return module
                        except Exception as e:
                            logger.warning("Failed to load bundle analyzer %s: %s", analyzer_path, e)
        
        return None

    async def _handle_optimizer_output(
        self,
        job_id: str,
        job: SpawnedJob | None,
        event: Event,
    ) -> None:
        """Handle optimizer role output - parse ReviewProposal and spawn optimization task.

        Per ADR-0010: The optimizer role outputs structured proposals that can be
        converted into optimization tasks. This method:
        1. Detects if the completed job was an optimizer role
        2. Parses the summary field as ReviewProposal JSON
        3. Converts the proposal to a TriggerData
        4. Spawns the optimization task via _process_trigger

        Parsing failures are logged but do not interrupt normal flow.
        """
        from yoitsu_contracts.review_proposal import ReviewProposal
        from yoitsu_contracts.external_events import review_proposal_to_trigger

        # Only process optimizer role
        if job is None or job.role != "optimizer":
            return

        # Extract summary (contains the ReviewProposal JSON)
        summary = event.data.get("summary", "")
        if not summary:
            logger.warning("Optimizer job %s completed without summary", job_id)
            return

        # Parse ReviewProposal from summary
        proposal = ReviewProposal.from_json_str(summary)
        if proposal is None:
            logger.warning(
                "Optimizer job %s summary could not be parsed as ReviewProposal",
                job_id,
            )
            return

        # Convert proposal to trigger data
        trigger_data = review_proposal_to_trigger(proposal)
        
        # Bundle MVP: inherit bundle from parent job if not specified in proposal
        if not trigger_data.get("bundle") and job.bundle:
            trigger_data["bundle"] = job.bundle
        
        # Create synthetic event for _process_trigger
        proposal_source_event_id = f"{event.id}-proposal"
        synthetic_event = SimpleNamespace(
            id=proposal_source_event_id,
            source_id=self.config.source_id,
            type="trigger.review_proposal",
            data=trigger_data,
            ts=datetime.now(timezone.utc),
        )

        # Validate as TriggerData
        try:
            data = TriggerData.model_validate(trigger_data)
        except Exception as e:
            logger.warning(
                "Optimizer proposal from job %s failed TriggerData validation: %s",
                job_id,
                e,
            )
            return

        # Process the trigger to spawn optimization task
        logger.info(
            "Spawning optimization task from optimizer job %s: goal=%s",
            job_id,
            data.goal[:50] if data.goal else "",
        )
        await self._process_trigger(synthetic_event, data, replay=False)

    async def _aggregate_and_spawn_optimizer(self) -> None:
        """Aggregate observation events and spawn optimizer when thresholds exceeded.
        
        Per ADR-0010 extension + ADR-0017 §2h:
        
        Flow: create Review Task first -> emit observation.consumed -> mark processed
        
        Atomicity guarantee:
        - new_ids are only marked processed AFTER Task creation succeeds
        - If _process_trigger or emit fails, these IDs remain "new" for next round
        - Each metric gets its own batch_members (not global new_ids)
        
        Idempotency: triggered_by is used as the key. On replay:
        - If Task exists: only emit observation.consumed (catch-up)
        - If Task doesn't exist: recreate Task + emit consumed
        """
        import hashlib
        import os
        from .observation_aggregator import aggregate_observations, AggregationResult
        from yoitsu_contracts import OBSERVATION_CONSUMED, ObservationConsumedData
        
        # Get API key from environment
        api_key = os.environ.get(self.config.pasloe_api_key_env, "")
        
        results, metric_new_ids = await aggregate_observations(
            self.config.pasloe_url,
            self.config.observation_window_hours,
            self.config.observation_thresholds,
            api_key=api_key,
            processed_ids=self._processed_observation_ids_set,
        )
        
        # DO NOT mark processed here - wait until Task creation succeeds
        # This ensures atomicity: failed spawn doesn't "eat" observations
        
        # Track which IDs were successfully processed in this round
        successfully_processed_ids: set[str] = set()
        
        for r in results:
            if not r.exceeded:
                continue  # Threshold not exceeded
            
            # Per ADR-0017: per-metric batch, not global new_ids
            # Get observation events for THIS specific metric only
            batch_ids = metric_new_ids.get(r.metric_type, [])
            if not batch_ids:
                continue  # No new events for this metric
            
            logger.info(
                f"Observation threshold exceeded: {r.metric_type} "
                f"(window_total={r.count} >= threshold={r.threshold}, new_events={len(batch_ids)})"
            )
            
            # Per-metric batch hash for idempotency
            batch_hash = hashlib.md5(json.dumps(sorted(batch_ids)).encode()).hexdigest()[:8]
            
            # Resolve target bundle from evidence
            target_bundle = _resolve_bundle_for_observations(r.evidence)
            
            # Construct TriggerData for optimizer
            trigger_data = {
                "goal": (
                    f"Analyze {r.metric_type} pattern in bundle '{target_bundle}' "
                    f"({r.count} occurrences in {self.config.observation_window_hours}h window). "
                    "Output a ReviewProposal JSON in your summary."
                ),
                "role": "optimizer",
                "bundle": target_bundle,
                "budget": 0.5,
                "params": {
                    "metric_type": r.metric_type,
                    "observation_count": r.count,
                    "window_hours": self.config.observation_window_hours,
                    "evidence": r.evidence,
                    "triggered_by": list(batch_ids),  # Per-metric batch
                },
            }
            
            # Create synthetic event
            synthetic_event = SimpleNamespace(
                id=f"obs-agg-{r.metric_type}-{batch_hash}",
                source_id="observation_aggregator",
                type="trigger",
                ts=datetime.now(timezone.utc),
                data=trigger_data,
            )
            
            # Compute task_id for consumed event
            task_id = self._root_task_id(synthetic_event.id)
            
            # Validate and process trigger (creates Review Task)
            try:
                data = TriggerData.model_validate(trigger_data)
            except Exception as e:
                logger.warning(
                    "Aggregated observation trigger failed TriggerData validation: %s",
                    e,
                )
                continue  # Skip this metric, IDs remain "new" for next round
            
            try:
                await self._process_trigger(synthetic_event, data, replay=False)
                
                # ADR-0017 §2h: emit observation.consumed AFTER Task creation
                await self.client.emit(
                    OBSERVATION_CONSUMED,
                    ObservationConsumedData(
                        batch_members=list(batch_ids),  # Per-metric
                        trigger_task_id=task_id,
                        bundle=target_bundle,
                        metric_type=r.metric_type,
                    ).model_dump(mode="json"),
                    idempotency_key=f"obs-consumed-{r.metric_type}-{batch_hash}",  # Per-metric key
                )
                
                # Only mark processed after both succeed
                successfully_processed_ids.update(batch_ids)
            
            except Exception as e:
                logger.exception(f"Failed to spawn optimizer for {r.metric_type}: {e}")
                # IDs remain "new" for next round - atomicity preserved
                continue
        
        # Now mark all successfully processed IDs
        for id in successfully_processed_ids:
            if id not in self._processed_observation_ids_set:
                self._processed_observation_ids_order.append(id)
                self._processed_observation_ids_set.add(id)
        
        # Prune old IDs using true FIFO
        while len(self._processed_observation_ids_order) > self._max_processed_observation_ids:
            old_id = self._processed_observation_ids_order.pop(0)
            self._processed_observation_ids_set.discard(old_id)

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


def _resolve_bundle_for_observations(evidence: list[dict]) -> str:
    """Resolve target bundle from observation evidence.
    
    Per plan Task 1: use the bundle field from observation events to route
    optimizer spawn to the correct bundle. Takes the most common bundle
    from evidence (majority vote).
    
    Args:
        evidence: List of observation event payloads with 'bundle' field.
        
    Returns:
        Bundle name to use for optimizer spawn, or 'default' if no evidence.
    """
    from collections import Counter
    
    bundles = [e.get("bundle", "") for e in evidence if e.get("bundle")]
    if not bundles:
        logger.warning("Observation evidence missing bundle field; falling back to 'default'")
        return "default"
    return Counter(bundles).most_common(1)[0][0]
