from __future__ import annotations

import logging

from .checkpoint import ACTIVE_CONTAINER_STATES

logger = logging.getLogger(__name__)


async def rebuild_state(supervisor) -> None:
    logger.info("Replaying unfinished tasks from Pasloe...")

    checkpoints = await supervisor._fetch_all("supervisor.checkpoint", source=supervisor.config.source_id)
    replay_cursor = checkpoints[-1].data.get("cursor") if checkpoints else None
    if replay_cursor:
        supervisor.event_cursor = replay_cursor
        logger.info("Found checkpoint, replay from cursor=%s", replay_cursor)

    fetch_plan = [
        ("trigger.*", None),
        ("job.spawn.request", None),
        ("supervisor.job.launched", supervisor.config.source_id),
        ("job.started", None),
        ("job.completed", None),
        ("job.failed", None),
        ("job.cancelled", None),
        ("task.created", None),
        ("task.completed", None),
        ("task.failed", None),
        ("task.cancelled", None),
    ]

    all_events = []
    for event_type, source in fetch_plan:
        all_events.extend(await supervisor._fetch_all(event_type, source=source))
    all_events.sort(key=lambda event: (event.ts, event.id))

    launched_by_job: dict[str, object] = {}
    started_job_ids: set[str] = set()
    finished_job_ids: set[str] = set()

    for event in all_events:
        if event.type == "supervisor.job.launched":
            launched_by_job[event.data.get("job_id", "")] = event
        elif event.type == "job.started" and event.data.get("job_id"):
            started_job_ids.add(event.data["job_id"])
        elif event.type in {"job.completed", "job.failed", "job.cancelled"} and event.data.get("job_id"):
            finished_job_ids.add(event.data["job_id"])
        elif event.type == "task.created":
            task_id = event.data.get("task_id", "")
            if task_id and task_id not in supervisor.state.tasks:
                supervisor.scheduler.record_task_submission(
                    task_id=task_id,
                    goal=event.data.get("goal", ""),
                    source_event_id=event.data.get("source_trigger_id", ""),
                    spec={},
                )
        elif event.type in {"task.completed", "task.failed", "task.cancelled"}:
            task_id = event.data.get("task_id", "")
            if task_id:
                state = event.type.split(".")[1]
                await supervisor.scheduler.mark_task_terminal(task_id=task_id, state=state)

        await supervisor._handle_event(event, replay=True)

    if not supervisor.event_cursor and all_events:
        last = all_events[-1]
        supervisor.event_cursor = f"{last.ts.isoformat()}|{last.id}"

    for job_id, launched in launched_by_job.items():
        if not job_id or job_id in finished_job_ids:
            continue

        data = launched.data
        container_id = data.get("container_id", "")
        container_name = data.get("container_name", "")
        state = await supervisor._inspect_replay_state(container_id, container_name)
        handle = supervisor._handle_from_replay(job_id, container_id, container_name)

        if state.exists and (state.running or state.status in ACTIVE_CONTAINER_STATES):
            supervisor.jobs[job_id] = handle
            continue

        job = supervisor.state.jobs_by_id.get(job_id)
        if job is None:
            continue

        if job_id in started_job_ids:
            await supervisor.client.emit(
                "job.failed",
                {
                    "job_id": job_id,
                    "task_id": job.task_id,
                    "error": "Container disappeared before a terminal event was emitted",
                    "code": "runtime_lost",
                },
            )
            await supervisor.scheduler.record_job_terminal(
                job_id=job_id,
                summary="Container disappeared before a terminal event was emitted",
                failed=True,
            )
            continue

        cancelled = await supervisor.scheduler.enqueue(job)
        if cancelled:
            logger.info("Replay cancelled %s because its condition is already impossible", job_id)

    logger.info(
        "Replay complete: running=%d pending=%d ready=%d tasks=%d",
        len(supervisor.jobs),
        len(supervisor._pending),
        supervisor._ready_queue.qsize(),
        len(supervisor.state.tasks),
    )
