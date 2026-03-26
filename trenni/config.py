from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PodmanRuntimeConfig:
    socket_uri: str = ""
    pod_name: str = "yoitsu-dev"
    image: str = "localhost/yoitsu-palimpsest-job:dev"
    pull_policy: str = "never"
    stop_grace_seconds: int = 10
    cleanup_timeout_seconds: int = 120
    retain_on_failure: bool = False
    git_token_env: str = "GITHUB_TOKEN"
    env_allowlist: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PodmanRuntimeConfig":
        return cls(**(data or {}))


@dataclass
class RuntimeConfig:
    kind: str = "podman"
    podman: PodmanRuntimeConfig = field(default_factory=PodmanRuntimeConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RuntimeConfig":
        payload = data or {}
        return cls(
            kind=payload.get("kind", "podman"),
            podman=PodmanRuntimeConfig.from_dict(payload.get("podman")),
        )


_LEGACY_RUNTIME_FIELDS = {
    "palimpsest_command",
    "evo_repo_path",
    "work_dir",
    "isolation_backend",
    "isolation_unshare_net",
}


@dataclass
class TrenniConfig:
    pasloe_url: str = "http://localhost:8000"
    pasloe_api_key_env: str = "PASLOE_API_KEY"
    source_id: str = "trenni-supervisor"
    evo_root: str = ""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    max_workers: int = 4
    poll_interval: float = 2.0

    # Defaults injected into every JobConfig
    default_eventstore_url: str = ""  # defaults to pasloe_url if empty
    default_eventstore_source: str = "palimpsest-agent"
    default_llm: dict = field(default_factory=dict)
    default_workspace: dict = field(default_factory=dict)
    default_publication: dict = field(default_factory=dict)

    api_host: str = "127.0.0.1"
    api_port: int = 8100

    # Webhook config — Trenni registers itself with Pasloe on startup
    webhook_secret: str = ""          # HMAC secret; empty = unsigned (OK on localhost)
    trenni_public_url: str = ""       # Override if not reachable at api_host:api_port
    webhook_poll_interval: float = 30.0  # Fallback poll interval when webhook active

    @property
    def trenni_webhook_url(self) -> str:
        base = self.trenni_public_url or f"http://{self.api_host}:{self.api_port}"
        return f"{base}/hooks/events"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrenniConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        legacy = sorted(_LEGACY_RUNTIME_FIELDS & set(data))
        if legacy:
            joined = ", ".join(legacy)
            raise ValueError(
                f"Legacy runtime fields are no longer supported: {joined}. "
                "Use the runtime.podman block instead."
            )

        payload = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__ and k != "runtime"
        }
        payload["runtime"] = RuntimeConfig.from_dict(data.get("runtime"))
        return cls(**payload)

    @property
    def eventstore_url(self) -> str:
        return self.default_eventstore_url or self.pasloe_url
