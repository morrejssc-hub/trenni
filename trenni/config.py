from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml


# Sentinel value to distinguish "unset" from "explicit null" for pod_name
_UNSET: Final[object] = object()


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


def _is_unset(value: object) -> bool:
    """Check if a value is the unset sentinel."""
    return value is _UNSET


@dataclass
class BundleRuntimeConfig:
    image: str | None = None
    # pod_name uses sentinel to distinguish "unset" (inherit default) from "explicit null" (no pod)
    pod_name: str | None | object = _UNSET  # type: ignore[assignment]
    env_allowlist: list[str] = field(default_factory=list)
    extra_networks: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BundleRuntimeConfig":
        payload = data or {}
        # Use sentinel for missing key to distinguish from explicit null/None
        pod_name = payload.get("pod_name", _UNSET)
        return cls(
            image=payload.get("image"),
            pod_name=pod_name,
            env_allowlist=list(payload.get("env_allowlist", [])),
            extra_networks=list(payload.get("extra_networks", [])),
        )


@dataclass
class BundleSourceConfig:
    """Bundle source configuration in Trenni registry.

    Declares where the bundle repo is and which branch/tag to track.
    Trenni resolves selector to a specific commit SHA at job dispatch time.

    Attributes:
        url: Bundle repo URI (git+file://, git+ssh://, git+https://, or direct git remote)
        selector: Branch or tag name to track (e.g., "evolve", "main", "v1.2.3")
    """
    url: str = ""
    selector: str = "main"

    @classmethod
    def from_dict(cls, data: dict | None) -> "BundleSourceConfig":
        payload = data or {}
        return cls(
            url=payload.get("url", ""),
            selector=payload.get("selector", "main"),
        )


@dataclass
class BundleSchedulingConfig:
    max_concurrent_jobs: int = 0  # 0 = unlimited

    @classmethod
    def from_dict(cls, data: dict | None) -> "BundleSchedulingConfig":
        payload = data or {}
        return cls(max_concurrent_jobs=int(payload.get("max_concurrent_jobs", 0)))


@dataclass
class BundleConfig:
    """Complete bundle configuration in Trenni registry.

    Attributes:
        source: Where the bundle repo is and which ref to track
        runtime: Container runtime settings for jobs using this bundle
        scheduling: Concurrency and scheduling settings
        default_role: Entry role for tasks without explicit role (e.g., "planner")
    """
    source: BundleSourceConfig = field(default_factory=BundleSourceConfig)
    runtime: BundleRuntimeConfig = field(default_factory=BundleRuntimeConfig)
    scheduling: BundleSchedulingConfig = field(default_factory=BundleSchedulingConfig)
    default_role: str = ""  # Entry role for external triggers without role field

    @classmethod
    def from_dict(cls, data: dict | None) -> "BundleConfig":
        payload = data or {}
        return cls(
            source=BundleSourceConfig.from_dict(payload.get("source")),
            runtime=BundleRuntimeConfig.from_dict(payload.get("runtime")),
            scheduling=BundleSchedulingConfig.from_dict(payload.get("scheduling")),
            default_role=payload.get("default_role", ""),
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
    workspace_root: str = "/tmp/yoitsu-workspaces"
    bundle_root: str = ""
    bundle_root_host: str = ""  # Host path for volume mounts (when running in container)

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    bundles: dict[str, BundleConfig] = field(default_factory=dict)

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

    # Observation aggregation config (ADR-0010 extension for autonomous optimization)
    observation_aggregation_interval: float = 300.0  # 5 minutes
    observation_window_hours: int = 24
    observation_thresholds: dict[str, float] = field(default_factory=lambda: {
        "budget_variance": 0.3,
        "preparation_failure": 0.1,
        "tool_retry": 0.2,
        "tool_repetition": 5.0,  # Absolute count, not ratio
        "context_late_lookup": 3.0,
    })
    
    # ADR-0017: Analyzer version env vars
    trenni_sha_env: str = "YOITSU_TRENNI_SHA"
    palimpsest_sha_env: str = "YOITSU_PALIMPSEST_SHA"
    bundle_sha_env: str = "YOITSU_BUNDLE_SHA"  # Global fallback (per-bundle resolved_ref preferred)

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
            if k in cls.__dataclass_fields__ and k not in ("runtime", "bundles")
        }
        payload["runtime"] = RuntimeConfig.from_dict(data.get("runtime"))
        payload["bundles"] = {
            name: BundleConfig.from_dict(bundle_data)
            for name, bundle_data in (data.get("bundles") or {}).items()
        }
        return cls(**payload)

    @property
    def eventstore_url(self) -> str:
        return self.default_eventstore_url or self.pasloe_url
