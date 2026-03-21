"""Core supervisor: event polling loop + job launcher."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import TrenniConfig
from .isolation import JobProcess, create_backend, launch_job
from .pasloe_client import Event, PasloeClient

logger = logging.getLogger(__name__)


@dataclass
class ForkJoin:
    parent_job_id: str
    spawn_event: Event
    child_ids: list[str]
    completed: set[str] = field(default_factory=set)


class Supervisor:
    def __init__(self, config: TrenniConfig) -> None:
        self.config = config
        self.client = PasloeClient(
            base_url=config.pasloe_url,
            api_key_env=config.pasloe_api_key_env,
            source_id=config.source_id,
        )
        self.running: bool = False

        # Isolation backend
        backend_kwargs = {}
        if config.isolation_backend == "bubblewrap":
            backend_kwargs["unshare_net"] = config.isolation_unshare_net
        self.backend = create_backend(config.isolation_backend, **backend_kwargs)

        # In-memory state
        self.event_cursor: str | None = None
        self.jobs: dict[str, JobProcess] = {}       # job_id -> process
        self.fork_joins: dict[str, ForkJoin] = {}   # parent_job_id -> ForkJoin

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

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Supervisor loop cancelled")
        finally:
            self.running = False
            await self.client.close()

    async def stop(self, force: bool = False) -> None:
        logger.info("Supervisor stopping (force=%s)", force)
        self.running = False
        if force:
            for job_id, jp in list(self.jobs.items()):
                logger.warning("Force-killing job %s (pid=%d)", job_id, jp.proc.pid)
                try:
                    jp.proc.kill()
                except ProcessLookupError:
                    pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self.running:
            try:
                await self._poll_and_handle()
            except Exception:
                logger.exception("Error in poll cycle")

            self._reap_processes()

            await asyncio.sleep(self.config.poll_interval)

    async def _poll_and_handle(self) -> None:
        events, next_cursor = await self.client.poll(
            cursor=self.event_cursor,
            limit=100,
        )
        if next_cursor:
            self.event_cursor = next_cursor

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
    # task.submit -> launch job
    # ------------------------------------------------------------------

    async def _handle_task_submit(self, event: Event) -> None:
        data = event.data
        task = data.get("task", "")
        role = data.get("role", "default")
        repo = data.get("repo", "")
        branch = data.get("branch", "main")
        evo_sha = data.get("evo_sha")

        if not task:
            logger.warning("Ignoring task.submit with empty task (event=%s)", event.id)
            return

        job_id = self._generate_job_id()

        if not self._has_capacity():
            logger.info("At capacity (%d/%d), deferring job %s",
                        len(self.jobs), self.config.max_workers, job_id)
            # TODO: queue for later. For now, log and skip.
            return

        await self._launch(job_id, task, role, repo, branch, evo_sha)

    # ------------------------------------------------------------------
    # job.spawn.request -> fork-join
    # ------------------------------------------------------------------

    async def _handle_spawn(self, event: Event) -> None:
        data = event.data
        parent_job_id = data.get("job_id", "")
        tasks = data.get("tasks", [])
        evo_sha = None

        if not tasks:
            logger.warning("Empty spawn request from job %s", parent_job_id)
            return

        child_ids: list[str] = []
        for i, child_task in enumerate(tasks):
            child_id = f"{parent_job_id}-c{i}"
            child_ids.append(child_id)

            # Use role_sha from spawn if present, else latest
            evo_sha = child_task.get("role_sha", evo_sha)
            role_file = child_task.get("role_file", "")
            # Extract role name from "roles/foo.py" -> "foo"
            role = role_file.replace("roles/", "").replace(".py", "") if role_file else "default"

            child_task_str = child_task.get("task", "")
            repo = child_task.get("repo", "")
            branch = child_task.get("branch", "main")

            if self._has_capacity():
                await self._launch(child_id, child_task_str, role, repo, branch, evo_sha)
            else:
                logger.info("At capacity, cannot launch child %s", child_id)

        self.fork_joins[parent_job_id] = ForkJoin(
            parent_job_id=parent_job_id,
            spawn_event=event,
            child_ids=child_ids,
        )

        logger.info("Fork-join created: parent=%s children=%s", parent_job_id, child_ids)

    # ------------------------------------------------------------------
    # job.completed / job.failed -> check fork-join
    # ------------------------------------------------------------------

    async def _handle_job_done(self, event: Event) -> None:
        job_id = event.data.get("job_id", "")
        status = "completed" if event.type == "job.completed" else "failed"
        logger.info("Job %s %s", job_id, status)

        # Remove from running processes
        self.jobs.pop(job_id, None)

        # Check fork-join membership
        for parent_id, fj in list(self.fork_joins.items()):
            if job_id in fj.child_ids:
                fj.completed.add(job_id)
                logger.info(
                    "Fork-join %s: %d/%d children done",
                    parent_id, len(fj.completed), len(fj.child_ids),
                )
                if fj.completed >= set(fj.child_ids):
                    await self._resolve_fork_join(parent_id, fj)
                    del self.fork_joins[parent_id]
                break

    async def _resolve_fork_join(self, parent_id: str, fj: ForkJoin) -> None:
        logger.info("Fork-join resolved for parent %s, launching continuation", parent_id)

        # Query child completion events for context
        child_summaries = []
        for child_id in fj.child_ids:
            events, _ = await self.client.poll(
                source=self.config.default_eventstore_source,
            )
            # Find completion events for this child
            for ev in events:
                if ev.data.get("job_id") == child_id and ev.type == "job.completed":
                    child_summaries.append(
                        f"- {child_id}: {ev.data.get('summary', '(no summary)')}"
                    )
                    break
            else:
                child_summaries.append(f"- {child_id}: (completed, no summary found)")

        # Build continuation task
        original_data = fj.spawn_event.data
        continuation_task = (
            f"Continue after child tasks completed.\n\n"
            f"Child results:\n" + "\n".join(child_summaries)
        )

        continuation_id = f"{parent_id}-r1"
        repo = original_data.get("repo", "")
        branch = original_data.get("branch", "main")

        if self._has_capacity():
            await self._launch(continuation_id, continuation_task, "default", repo, branch, None)
        else:
            logger.info("At capacity, cannot launch continuation %s", continuation_id)

    # ------------------------------------------------------------------
    # job.started -> evo hard gate (just log for now)
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
        repo: str, branch: str, evo_sha: str | None,
    ) -> None:
        logger.info("Launching job %s (role=%s)", job_id, role)

        jp = await launch_job(
            backend=self.backend,
            job_id=job_id,
            task=task,
            role=role,
            repo=repo,
            branch=branch,
            evo_sha=evo_sha,
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

    def _reap_processes(self) -> None:
        for job_id, jp in list(self.jobs.items()):
            if jp.proc.returncode is not None:
                logger.info(
                    "Process for job %s exited (rc=%d)",
                    job_id, jp.proc.returncode,
                )
                # Don't remove from self.jobs here — wait for the
                # authoritative job.completed/job.failed event from Pasloe.
                # But if the process exited without emitting events,
                # we'll eventually notice and clean up.

    def _generate_job_id(self) -> str:
        import uuid
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"trenni-{ts}-{short}"

    @property
    def status(self) -> dict:
        return {
            "running": self.running,
            "running_jobs": len(self.jobs),
            "max_workers": self.config.max_workers,
            "fork_joins_active": len(self.fork_joins),
        }
