"""FastAPI control plane for Trenni supervisor.

Exposes:
  GET  /control/status    — supervisor state
  GET  /control/tasks     — live task list
  GET  /control/tasks/{task_id} — live task detail
  GET  /control/jobs      — live job list
  GET  /control/jobs/{job_id} — live job detail
  POST /control/pause     — stop dispatching new jobs
  POST /control/resume    — re-enable dispatch
  POST /control/stop      — graceful shutdown
  POST /hooks/events      — Pasloe webhook delivery intake
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from .supervisor import Supervisor

logger = logging.getLogger(__name__)


class EventPayload(BaseModel):
    id: str
    source_id: str
    type: str
    ts: datetime
    data: dict = {}


def _task_state(record) -> str:
    return record.terminal_state if record.terminal else (record.state or "pending")


def _job_queue_state(supervisor: "Supervisor", job_id: str) -> str:
    if job_id in supervisor.jobs:
        return "running"
    if job_id in supervisor.state.pending_jobs:
        return "pending"
    if supervisor.state.has_ready_job(job_id):
        return "ready"
    if job_id in supervisor.state.failed_jobs:
        return "failed"
    if job_id in supervisor.state.cancelled_jobs:
        return "cancelled"
    if job_id in supervisor.state.completed_jobs:
        return "completed"
    return "unknown"


def build_control_app(supervisor: "Supervisor") -> FastAPI:
    app = FastAPI(title="Trenni Control API", docs_url=None, redoc_url=None)

    @app.get("/control/status")
    async def status():
        return supervisor.status

    @app.get("/control/tasks")
    async def list_tasks(state: str | None = None, team: str | None = None):
        tasks = []
        for task_id, record in sorted(supervisor.state.tasks.items()):
            item = {
                "task_id": task_id,
                "goal": record.goal,
                "state": _task_state(record),
                "team": record.team,
            }
            if state and item["state"] != state:
                continue
            if team and item["team"] != team:
                continue
            tasks.append(item)
        return tasks

    @app.get("/control/tasks/{task_id:path}")
    async def task_detail(task_id: str):
        record = supervisor.state.tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            "task_id": task_id,
            "goal": record.goal,
            "state": _task_state(record),
            "team": record.team,
            "eval_spawned": record.eval_spawned,
            "eval_job_id": record.eval_job_id,
            "job_order": list(record.job_order),
            "result": record.result.model_dump(mode="json") if record.result else None,
        }

    @app.get("/control/jobs")
    async def list_jobs(task_id: str | None = None, role: str | None = None, state: str | None = None):
        jobs = []
        for job_id, record in sorted(supervisor.state.jobs_by_id.items()):
            item = {
                "job_id": job_id,
                "task_id": record.task_id,
                "role": record.role,
                "team": record.team,
                "state": _job_queue_state(supervisor, job_id),
            }
            if task_id and item["task_id"] != task_id:
                continue
            if role and item["role"] != role:
                continue
            if state and item["state"] != state:
                continue
            jobs.append(item)
        return jobs

    @app.get("/control/jobs/{job_id}")
    async def job_detail(job_id: str):
        record = supervisor.state.jobs_by_id.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "task_id": record.task_id,
            "role": record.role,
            "team": record.team,
            "state": _job_queue_state(supervisor, job_id),
            "parent_job_id": record.parent_job_id,
            "condition": None if record.condition is None else str(record.condition),
            "job_context": record.job_context.model_dump(mode="json", exclude_none=True),
        }

    @app.post("/control/pause")
    async def pause():
        await supervisor.pause()
        return {"ok": True, "paused": True}

    @app.post("/control/resume")
    async def resume():
        await supervisor.resume()
        return {"ok": True, "paused": False}

    @app.post("/control/stop")
    async def stop():
        await supervisor.stop()
        return {"ok": True}

    @app.post("/hooks/events")
    async def receive_event(request: Request):
        body = await request.body()
        secret = getattr(supervisor.config, "webhook_secret", "")
        if secret:
            sig_header = request.headers.get("X-Pasloe-Signature", "")
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, sig_header):
                raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            payload = EventPayload.model_validate_json(body)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid payload")

        from .pasloe_client import Event
        event = Event(id=payload.id, source_id=payload.source_id, type=payload.type, ts=payload.ts, data=payload.data)
        try:
            await supervisor._handle_event(event, realtime=True)
        except Exception:
            logger.exception("Error handling webhook event %s", payload.id)
        return {"ok": True}

    return app
