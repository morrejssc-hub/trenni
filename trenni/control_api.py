"""FastAPI control plane for Trenni supervisor.

Exposes:
  GET  /control/status    — supervisor state
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


def build_control_app(supervisor: "Supervisor") -> FastAPI:
    app = FastAPI(title="Trenni Control API", docs_url=None, redoc_url=None)

    @app.get("/control/status")
    async def status():
        return supervisor.status

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
