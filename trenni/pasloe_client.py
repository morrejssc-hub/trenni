from __future__ import annotations

from yoitsu_contracts.client import AsyncPasloeClient, PasloeEvent

Event = PasloeEvent


class PasloeClient(AsyncPasloeClient):
    def __init__(self, base_url: str, api_key_env: str = "PASLOE_API_KEY", source_id: str = "trenni-supervisor") -> None:
        super().__init__(base_url=base_url, api_key_env=api_key_env, source_id=source_id)
