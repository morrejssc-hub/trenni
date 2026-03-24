from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class RuntimeDefaults:
    kind: Literal["podman"]
    socket_uri: str
    pod_name: str
    image: str
    pull_policy: Literal["always", "missing", "newer", "never"]
    stop_grace_seconds: int
    cleanup_timeout_seconds: int
    retain_on_failure: bool
    labels: Mapping[str, str]
    env_allowlist: tuple[str, ...]
    git_token_env: str


@dataclass(frozen=True)
class JobRuntimeSpec:
    job_id: str
    source_event_id: str
    container_name: str
    image: str
    pod_name: str
    labels: Mapping[str, str]
    env: Mapping[str, str]
    command: tuple[str, ...]
    config_payload_b64: str


@dataclass
class JobHandle:
    job_id: str
    container_id: str
    container_name: str
    exit_code: int | None = None
    exited_at: float | None = None


@dataclass(frozen=True)
class ContainerState:
    exists: bool
    status: str = "missing"
    running: bool = False
    exit_code: int | None = None


@dataclass(frozen=True)
class ContainerExit:
    status_code: int | None
