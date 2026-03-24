from __future__ import annotations

from typing import Protocol

from yoitsu_contracts.env import build_git_auth_env

from .podman_backend import PodmanBackend
from .runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from .runtime_types import ContainerExit, ContainerState, JobHandle, JobRuntimeSpec, RuntimeDefaults


class IsolationBackend(Protocol):
    async def prepare(self, spec: JobRuntimeSpec) -> JobHandle: ...
    async def start(self, handle: JobHandle) -> None: ...
    async def inspect(self, handle: JobHandle) -> ContainerState: ...
    async def stop(self, handle: JobHandle, timeout_s: int) -> None: ...
    async def remove(self, handle: JobHandle, *, force: bool = False) -> None: ...
    async def logs(self, handle: JobHandle) -> str: ...


__all__ = [
    "ContainerExit",
    "ContainerState",
    "IsolationBackend",
    "JobHandle",
    "JobRuntimeSpec",
    "PodmanBackend",
    "RuntimeDefaults",
    "RuntimeSpecBuilder",
    "build_git_auth_env",
    "build_runtime_defaults",
]
