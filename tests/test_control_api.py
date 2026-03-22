"""Tests for Trenni control API."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from trenni.control_api import build_control_app
from trenni.supervisor import Supervisor
from trenni.config import TrenniConfig


@pytest.fixture
def supervisor():
    return Supervisor(TrenniConfig())


@pytest.fixture
def app(supervisor):
    return build_control_app(supervisor)


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestStatus:
    async def test_status_returns_running_state(self, client, supervisor):
        r = await client.get("/control/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data
        assert "paused" in data
        assert "running_jobs" in data


class TestPauseResume:
    async def test_pause(self, client, supervisor):
        r = await client.post("/control/pause")
        assert r.status_code == 200
        assert supervisor.paused is True

    async def test_resume(self, client, supervisor):
        supervisor._resume_event.clear()  # simulate paused
        r = await client.post("/control/resume")
        assert r.status_code == 200
        assert supervisor.paused is False

    async def test_pause_then_resume(self, client, supervisor):
        await client.post("/control/pause")
        assert supervisor.paused is True
        await client.post("/control/resume")
        assert supervisor.paused is False


class TestWebhookIntake:
    async def test_post_event_calls_handle_event(self, client, supervisor):
        supervisor._handle_event = AsyncMock()
        r = await client.post("/hooks/events", json={
            "id": "e1", "source_id": "pasloe", "type": "task.submit",
            "ts": datetime.utcnow().isoformat(), "data": {"task": "do something"},
        })
        assert r.status_code == 200
        supervisor._handle_event.assert_called_once()

    async def test_post_event_invalid_json(self, client):
        r = await client.post("/hooks/events", content=b"not json",
                              headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    async def test_unsigned_event_accepted_when_no_secret(self, client, supervisor):
        """No signature check when webhook_secret is empty (default)."""
        supervisor._handle_event = AsyncMock()
        r = await client.post("/hooks/events", json={
            "id": "e2", "source_id": "s", "type": "job.completed",
            "ts": datetime.utcnow().isoformat(), "data": {},
        })
        assert r.status_code == 200


class TestWebhookSignature:
    async def test_valid_signature_accepted(self, client, supervisor):
        import hashlib, hmac as _hmac, json
        supervisor.config.webhook_secret = "test-secret"
        payload = {"id": "e2", "source_id": "s", "type": "job.completed",
                   "ts": datetime.utcnow().isoformat(), "data": {}}
        body = json.dumps(payload).encode()
        sig = "sha256=" + _hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
        supervisor._handle_event = AsyncMock()
        r = await client.post("/hooks/events", content=body,
                              headers={"Content-Type": "application/json",
                                       "X-Pasloe-Signature": sig})
        assert r.status_code == 200

    async def test_invalid_signature_rejected(self, client, supervisor):
        supervisor.config.webhook_secret = "test-secret"
        r = await client.post("/hooks/events",
                              json={"id": "e3", "source_id": "s", "type": "job.failed",
                                    "ts": datetime.utcnow().isoformat(), "data": {}},
                              headers={"X-Pasloe-Signature": "sha256=badhex"})
        assert r.status_code == 401
