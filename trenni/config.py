from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrenniConfig:
    pasloe_url: str = "http://localhost:8000"
    pasloe_api_key_env: str = "PASLOE_API_KEY"
    source_id: str = "trenni-supervisor"

    palimpsest_command: str = "palimpsest"
    evo_repo_path: str = "./palimpsest-evo"
    work_dir: str = "./trenni-work"

    max_workers: int = 4
    poll_interval: float = 2.0

    # Defaults injected into every JobConfig
    default_eventstore_url: str = ""  # defaults to pasloe_url if empty
    default_eventstore_source: str = "palimpsest-agent"
    default_llm: dict = field(default_factory=dict)
    default_workspace: dict = field(default_factory=dict)
    default_publication: dict = field(default_factory=dict)

    # Isolation backend: "subprocess" or "bubblewrap"
    isolation_backend: str = "subprocess"
    # Bubblewrap-specific: isolate network (blocks LLM/eventstore access)
    isolation_unshare_net: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8100

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrenniConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def eventstore_url(self) -> str:
        return self.default_eventstore_url or self.pasloe_url
