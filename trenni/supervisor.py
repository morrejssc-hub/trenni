"""Core supervisor: event polling loop + job launcher.

Unified spawn model: all jobs are SpawnedJob instances, differing only
in their ``depends_on`` set.  Jobs with no dependencies go straight to
the ready queue; jobs with dependencies wait in ``_pending`` until all
prerequisites complete.  This replaces the separate ForkJoin + TaskItem
structures.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import TrenniConfig
from .isolation import JobProcess, create_backend, launch_job
from .pasloe_client import Event, PasloeClient

logger = logging.getLogger(__name__)


@dataclass
class SpawnedJob:
    """A job recognised by the supervisor, possibly waiting on dependencies."""
    job_id: str
    source_event_id: str
    task: str
    role: str
    repo: str
    init_branch: str
    evo_sha: str | None
    depends_on: frozenset[str] = field(default_factory=frozenset)


# Number of poll cycles between checkpoints (checkpoint_interval * poll_interval seconds).
_DEFAULT_CHECKPOINT_CYCLES = 30          # ~60 s at default 2 s poll
_DEFAULT_REAP_TIMEOUT_S = 120.0


class Supervisor:
    def __init__(self, config: TrenniConfig) -> None:
        self.config = config
        self.client = PasloeClient(
            base_url=config.pasloe_url,
            api_key_env=config.pasloe_api_key_env,
            source_id=config.source_id,
        )
        self.running: bool = False

        # Pause/resume control
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # initially running (not paused)

        # Isolation backend
        backend_kwargs = {}
        if config.isolation_backend == "bubblewrap":
            backend_kwargs["unshare_net"] = config.isolation_unshare_net
        self.backend = create_backend(config.isolation_backend, **backend_kwargs)

        # In-memory state
        self.event_cursor: str | None = None
        self.jobs: dict[str, JobProcess] = {}           # job_id → running process

        # Unified spawn state
        self._ready_queue: asyncio.Queue[SpawnedJob] = asyncio.Queue()
        self._pending: dict[str, SpawnedJob] = {}       # job_id → waiting for deps
        self._completed_jobs: set[str] = set()          # terminal job_ids (includes failed)
        self._failed_jobs: set[str] = set()             # subset of terminal: only failed
        self._job_summaries: dict[str, str] = {}        # job_id → summary text

        # Dedup guard for source events (task.submit / spawn request)
        self._launched_event_ids: set[str] = set()

        # Checkpoint config
        self._checkpoint_cycles = _DEFAULT_CHECKPOINT_CYCLES
        self._reap_timeout = _DEFAULT_REAP_TIMEOUT_S

        # Webhook state
        self._webhook_id: str | None = None
        self._webhook_active: bool = False

    @property
    def paused(self) -> bool:
        return not self._resume_event.is_set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info(
            "Supervisor starting (max_workers=%d, isolation=%s)",
            self.config.max_workers, self.config.isolation_backend,
        )
        self.running = True
        work_dir = Path(self.config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        await self.client.register_source()
        logger.info("Registered source '%s' with Pasloe", self.config.source_id)

        await self._replay_unfinished_tasks()
        await self._try_register_webhook()

        drain_task = asyncio.create_task(self._drain_queue())
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Supervisor loop cancelled")
        finally:
            self.running = False
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            await self.client.close()

    async def stop(self, force: bool = False) -> None:
        logger.info("Supervisor stopping (force=%s)", force)
        self.running = False
        if self._webhook_id:
            try:
                await self.client.delete_webhook(self._webhook_id)
                logger.info("Deregistered webhook %s", self._webhook_id)
            except Exception as exc:
                logger.warning("Could not deregister webhook: %s", exc)
        if force:
            for job_id, jp in list(self.jobs.items()):
                logger.warning("Force-killing job %s (pid=%d)", job_id, jp.proc.pid)
                try:
                    jp.proc.kill()
                except ProcessLookupError:
                    pass

    async def pause(self) -> None:
        logger.info("Supervisor pausing")
        self._resume_event.clear()
        try:
            await self.client.emit("supervisor.paused", {})
        except Exception:
            logger.warning("Could not emit supervisor.paused event")

    async def resume(self) -> None:
        logger.info("Supervisor resuming")
        self._resume_event.set()
        try:
            await self.client.emit("supervisor.resumed", {})
        except Exception:
            logger.warning("Could not emit supervisor.resumed event")

    async def _try_register_webhook(self) -> None:
        """Best-effort webhook registration. Falls back to polling on failure."""
        url = self.config.trenni_webhook_url
        if not url:
            return
        try:
            self._webhook_id = await self.client.register_webhook(
                url=url,
                secret=self.config.webhook_secret,
                event_types=["task.submit", "job.spawn.request",
                             "job.completed", "job.failed", "job.started"],
            )
            self._webhook_active = True
            logger.info(
                "Registered webhook (id=%s) → %s (poll fallback every %.0fs)",
                self._webhook_id, url, self.config.webhook_poll_interval,
            )
        except Exception as exc:
            logger.warning(
                "Could not register webhook with Pasloe (%s) — falling back to pure polling",
                exc,
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        polls_since_checkpoint = 0
        while self.running:
            try:
                await self._poll_and_handle()
            except Exception:
                logger.exception("Error in poll cycle")

            self._mark_exited_processes()

            polls_since_checkpoint += 1
            if polls_since_checkpoint >= self._checkpoint_cycles:
                await self._checkpoint()
                polls_since_checkpoint = 0

            interval = (
                self.config.webhook_poll_interval
                if self._webhook_active
                else self.config.poll_interval
            )
            await asyncio.sleep(interval)

    async def _drain_queue(self) -> None:
        """Background coroutine: dequeue SpawnedJobs and launch when capacity allows."""
        while True:
            # Block here while paused; resumes instantly when resume_event is set
            await self._resume_event.wait()
            job = await self._ready_queue.get()
            # Wait for capacity AND not paused
            while not self._has_capacity() or not self._resume_event.is_set():
                await asyncio.sleep(1.0)
            try:
                await self._launch_from_spawned(job)
            except Exception:
                logger.exception(
                    "Failed to launch queued job %s, dropping (recoverable on restart)",
                    job.job_id,
                )

    async def _launch_from_spawned(self, job: SpawnedJob) -> None:
        await self._launch(
            job_id=job.job_id,
            task=job.task,
            role=job.role,
            repo=job.repo,
            init_branch=job.init_branch,
            evo_sha=job.evo_sha,
            source_event_id=job.source_event_id,
        )

    async def _poll_and_handle(self) -> None:
        events, next_cursor = await self.client.poll(
            cursor=self.event_cursor,
            limit=100,
        )
        if next_cursor:
            self.event_cursor = next_cursor
        elif events:
            last = events[-1]
            self.event_cursor = f"{last.ts.isoformat()}|{last.id}"

        for event in events:
            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("Error handling event %s (type=%s)", event.id, event.type)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        match event.type:
            case "task.submit":
                await self._handle_task_submit(event)
            case "job.spawn.request":
                await self._handle_spawn(event)
            case "job.completed" | "job.failed":
                await self._handle_job_done(event)
            case "job.started":
                self._handle_job_started(event)

    # ------------------------------------------------------------------
    # task.submit → enqueue
    # ------------------------------------------------------------------

    async def _handle_task_submit(self, event: Event) -> None:
        if event.id in self._launched_event_ids:
            logger.debug("Skipping already-processed task.submit %s", event.id)
            return

        data = event.data
        task = data.get("task", "")
        if not task:
            logger.warning("Ignoring task.submit with empty task (event=%s)", event.id)
            return

        self._launched_event_ids.add(event.id)

        job = SpawnedJob(
            job_id=self._generate_job_id(),
            source_event_id=event.id,
            task=task,
            role=data.get("role", "default"),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", "main"),
            evo_sha=data.get("evo_sha"),
        )
        await self._enqueue(job)
        logger.info(
            "Queued task %s (job_id=%s, queue_size=%d)",
            event.id, job.job_id, self._ready_queue.qsize(),
        )

    # ------------------------------------------------------------------
    # job.spawn.request → fan-out children (no continuation for now)
    # ------------------------------------------------------------------

    async def _handle_spawn(self, event: Event) -> None:
        data = event.data
        parent_job_id = data.get("job_id", "")
        tasks = data.get("tasks", [])

        if not tasks:
            logger.warning("Empty spawn request from job %s", parent_job_id)
            return

        evo_sha: str | None = None
        child_ids: list[str] = []

        for i, child_def in enumerate(tasks):
            child_id = f"{parent_job_id}-c{i}"
            child_ids.append(child_id)

            evo_sha = child_def.get("role_sha", evo_sha)
            role_file = child_def.get("role_file", "")
            role = role_file.replace("roles/", "").replace(".py", "") if role_file else "default"

            child = SpawnedJob(
                job_id=child_id,
                source_event_id=event.id,
                task=child_def.get("task", ""),
                role=role,
                repo=child_def.get("repo", ""),
                init_branch=child_def.get("branch", "main"),
                evo_sha=evo_sha,
            )
            await self._enqueue(child)

        logger.info(
            "Spawn: parent=%s children=%s (fan-out only, no continuation)",
            parent_job_id, child_ids,
        )

    # ------------------------------------------------------------------
    # Unified enqueue
    # ------------------------------------------------------------------

    async def _enqueue(self, job: SpawnedJob) -> None:
        """Route a job to ready queue or pending table based on dependencies."""
        unsatisfied = job.depends_on - self._completed_jobs
        if not unsatisfied:
            await self._ready_queue.put(job)
        else:
            self._pending[job.job_id] = job

    # ------------------------------------------------------------------
    # job.completed / job.failed → resolve dependencies
    # ------------------------------------------------------------------

    async def _handle_job_done(self, event: Event) -> None:
        job_id = event.data.get("job_id", "")
        is_failure = event.type == "job.failed"
        logger.info("Job %s %s", job_id, "failed" if is_failure else "completed")

        self.jobs.pop(job_id, None)
        self._completed_jobs.add(job_id)
        if is_failure:
            self._failed_jobs.add(job_id)
        self._job_summaries[job_id] = event.data.get("summary", "")

        # Scan pending jobs whose dependencies may now be satisfied
        newly_ready: list[str] = []
        propagate_failed: list[str] = []

        for pending_id, pending_job in list(self._pending.items()):
            # If any dependency failed, propagate failure
            failed_deps = pending_job.depends_on & self._failed_jobs
            if failed_deps:
                propagate_failed.append(pending_id)
                continue

            unsatisfied = pending_job.depends_on - self._completed_jobs
            if not unsatisfied:
                newly_ready.append(pending_id)
                await self._ready_queue.put(pending_job)

        for jid in newly_ready:
            del self._pending[jid]

        # Propagate failure: emit failure event for pending jobs with failed deps
        for jid in propagate_failed:
            pending_job = self._pending.pop(jid)
            failed_deps = pending_job.depends_on & self._failed_jobs
            logger.warning(
                "Propagating failure to job %s (failed deps: %s)",
                jid, sorted(failed_deps),
            )
            self._completed_jobs.add(jid)
            self._failed_jobs.add(jid)
            await self.client.emit("job.failed", {
                "job_id": jid,
                "error": f"Dependency failed: {', '.join(sorted(failed_deps))}",
                "code": "dependency_failed",
            })

    # ------------------------------------------------------------------
    # Checkpoint + reap
    # ------------------------------------------------------------------

    def _mark_exited_processes(self) -> None:
        """Record the first-observed exit time for finished processes."""
        now = time.monotonic()
        for jp in self.jobs.values():
            if jp.proc.returncode is not None and jp.exited_at is None:
                jp.exited_at = now
                logger.info("Process for job %s exited (rc=%d)", jp.job_id, jp.proc.returncode)

    async def _checkpoint(self) -> None:
        """Periodic maintenance: reap dead slots, emit checkpoint anchor."""
        now = time.monotonic()
        reaped: list[str] = []
        for job_id, jp in list(self.jobs.items()):
            if jp.exited_at is not None and (now - jp.exited_at) > self._reap_timeout:
                logger.warning(
                    "Job %s process exited %ds ago without terminal event, "
                    "emitting compensating failure",
                    job_id, int(now - jp.exited_at),
                )
                await self.client.emit("job.failed", {
                    "job_id": job_id,
                    "error": f"Process exited (rc={jp.proc.returncode}) without emitting terminal event",
                    "code": "process_lost",
                })
                reaped.append(job_id)

        for job_id in reaped:
            del self.jobs[job_id]

        await self.client.emit("supervisor.checkpoint", {
            "cursor": self.event_cursor,
            "running_jobs": list(self.jobs.keys()),
            "pending_jobs": list(self._pending.keys()),
            "ready_queue_size": self._ready_queue.qsize(),
            "completed_count": len(self._completed_jobs),
        })

    # ------------------------------------------------------------------
    # Replay on startup
    # ------------------------------------------------------------------

    async def _fetch_all(
        self,
        type_: str,
        source: str | None = None,
    ) -> list[Event]:
        """Fetch all Pasloe events of a given type, paginating until exhausted."""
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
                break
            cursor = next_cursor
        return results

    async def _replay_unfinished_tasks(self) -> None:
        """On startup: replay Pasloe history to recover unfinished tasks."""
        logger.info("Replaying unfinished tasks from Pasloe...")

        # Try to resume from latest checkpoint
        checkpoints = await self._fetch_all(
            "supervisor.checkpoint", source=self.config.source_id
        )
        replay_cursor: str | None = None
        if checkpoints:
            latest_cp = checkpoints[-1]
            replay_cursor = latest_cp.data.get("cursor")
            logger.info("Found checkpoint, replay from cursor=%s", replay_cursor)

        # Fetch relevant event sets
        launched_events = await self._fetch_all(
            "supervisor.job.launched", source=self.config.source_id
        )
        started_events = await self._fetch_all("job.started")
        completed_events = await self._fetch_all("job.completed")
        failed_events = await self._fetch_all("job.failed")
        submit_events = await self._fetch_all("task.submit")

        # Build lookup maps
        launched_map: dict[str, str] = {
            e.data["source_event_id"]: e.data["job_id"]
            for e in launched_events
            if e.data.get("source_event_id")
        }
        started_job_ids: set[str] = {
            e.data["job_id"] for e in started_events if e.data.get("job_id")
        }
        finished_job_ids: set[str] = {
            e.data["job_id"]
            for e in (completed_events + failed_events)
            if e.data.get("job_id")
        }

        # Populate _completed_jobs and _job_summaries from history
        for e in completed_events:
            jid = e.data.get("job_id", "")
            if jid:
                self._completed_jobs.add(jid)
                self._job_summaries[jid] = e.data.get("summary", "")
        for e in failed_events:
            jid = e.data.get("job_id", "")
            if jid:
                self._completed_jobs.add(jid)
                self._failed_jobs.add(jid)

        # Initialise cursor to latest event
        all_events = launched_events + started_events + completed_events + failed_events + submit_events
        if replay_cursor:
            self.event_cursor = replay_cursor
        else:
            latest = max(all_events, key=lambda e: (e.ts, e.id), default=None)
            if latest:
                self.event_cursor = f"{latest.ts.isoformat()}|{latest.id}"
        logger.info("Replay cursor set to %s", self.event_cursor)

        # Classify each task.submit
        enqueued = skipped = orphans = 0
        for event in submit_events:
            data = event.data
            task = data.get("task", "")
            if not task:
                continue

            job_id_from_launch = launched_map.get(event.id)

            if job_id_from_launch:
                job_started = job_id_from_launch in started_job_ids
                job_finished = job_id_from_launch in finished_job_ids

                if job_started and job_finished:
                    self._launched_event_ids.add(event.id)
                    skipped += 1
                    continue

                if not job_started and not job_finished:
                    if job_id_from_launch in self.jobs:
                        self._launched_event_ids.add(event.id)
                        skipped += 1
                        continue

                if job_started and not job_finished:
                    if job_id_from_launch in self.jobs:
                        self._launched_event_ids.add(event.id)
                        skipped += 1
                        continue
                    logger.warning(
                        "Orphaned job %s (task.submit=%s): started but no terminal event.",
                        job_id_from_launch, event.id,
                    )
                    orphans += 1
                    continue

            # Not launched, or launched-not-started with dead process → re-enqueue
            new_job_id = self._generate_job_id()
            self._launched_event_ids.add(event.id)
            job = SpawnedJob(
                source_event_id=event.id,
                job_id=new_job_id,
                task=task,
                role=data.get("role", "default"),
                repo=data.get("repo", ""),
                init_branch=data.get("init_branch", data.get("branch", "main")),
                evo_sha=data.get("evo_sha"),
            )
            await self._enqueue(job)
            enqueued += 1

        logger.info(
            "Replay complete: %d enqueued, %d skipped, %d orphans",
            enqueued, skipped, orphans,
        )

    # ------------------------------------------------------------------
    # job.started → evo hard gate (just log for now)
    # ------------------------------------------------------------------

    def _handle_job_started(self, event: Event) -> None:
        job_id = event.data.get("job_id", "")
        evo_sha = event.data.get("evo_sha", "")
        logger.info("Job %s started (evo_sha=%s)", job_id, evo_sha or "unknown")

    # ------------------------------------------------------------------
    # Job launch helper
    # ------------------------------------------------------------------

    async def _launch(
        self, job_id: str, task: str, role: str,
        repo: str, init_branch: str, evo_sha: str | None,
        source_event_id: str = "",
    ) -> None:
        logger.info("Launching job %s (role=%s, source=%s)", job_id, role, source_event_id or "?")

        jp = await launch_job(
            backend=self.backend,
            job_id=job_id,
            task=task,
            role=role,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            evo_repo_path=self.config.evo_repo_path,
            palimpsest_command=self.config.palimpsest_command,
            work_dir=Path(self.config.work_dir),
            eventstore_url=self.config.eventstore_url,
            eventstore_api_key_env=self.config.pasloe_api_key_env,
            eventstore_source=self.config.default_eventstore_source,
            llm_defaults=self.config.default_llm,
            workspace_defaults=self.config.default_workspace,
            publication_defaults=self.config.default_publication,
        )

        self.jobs[job_id] = jp

        await self.client.emit("supervisor.job.launched", {
            "job_id": job_id,
            "source_event_id": source_event_id,
            "task": task,
            "role": role,
            "evo_sha": evo_sha or "",
            "pid": jp.proc.pid,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_capacity(self) -> bool:
        return len(self.jobs) < self.config.max_workers

    def _generate_job_id(self) -> str:
        import uuid_utils
        return str(uuid_utils.uuid7())

    @property
    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "running_jobs": len(self.jobs),
            "max_workers": self.config.max_workers,
            "pending_jobs": len(self._pending),
            "ready_queue_size": self._ready_queue.qsize(),
        }
