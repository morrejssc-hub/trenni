"""HTTP client for Pasloe event store — poll and emit."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import httpx


@dataclass
class Event:
    id: str
    source_id: str
    type: str
    ts: datetime
    data: dict


@dataclass
class PasloeClient:
    base_url: str
    api_key_env: str = "PASLOE_API_KEY"
    source_id: str = "trenni-supervisor"
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get(self.api_key_env, "")
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    async def register_source(self) -> None:
        await self._client.post("/sources", json={"id": self.source_id})

    async def emit(self, event_type: str, data: dict) -> str:
        resp = await self._client.post(
            "/events",
            json={
                "source_id": self.source_id,
                "type": event_type,
                "data": data,
            },
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def poll(
        self,
        cursor: str | None = None,
        source: str | None = None,
        type_: str | None = None,
        limit: int = 100,
    ) -> tuple[list[Event], str | None]:
        params: dict = {"limit": limit, "order": "asc"}
        if cursor:
            params["cursor"] = cursor
        if source:
            params["source"] = source
        if type_:
            params["type"] = type_

        resp = await self._client.get("/events", params=params)
        resp.raise_for_status()

        next_cursor = resp.headers.get("X-Next-Cursor")
        events = [
            Event(
                id=e["id"],
                source_id=e["source_id"],
                type=e["type"],
                ts=datetime.fromisoformat(e["ts"]),
                data=e.get("data", {}),
            )
            for e in resp.json()
        ]
        return events, next_cursor

    async def close(self) -> None:
        await self._client.aclose()
