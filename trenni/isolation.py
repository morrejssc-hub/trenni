"""Process isolation for Palimpsest job subprocesses.

Pluggable backends: subprocess (default) and bubblewrap (bwrap).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class JobProcess:
    job_id: str
    proc: asyncio.subprocess.Process
    work_dir: Path
    config_path: Path


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class IsolationBackend(Protocol):
    """Interface for job isolation strategies."""

    async def launch(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> asyncio.subprocess.Process:
        """Launch a command with isolation, return the process handle."""
        ...


# ---------------------------------------------------------------------------
# Subprocess backend (default — no extra dependencies)
# ---------------------------------------------------------------------------

class SubprocessBackend:
    """Plain subprocess with process-group isolation."""

    async def launch(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )


# ---------------------------------------------------------------------------
# Bubblewrap backend (Linux namespace isolation)
# ---------------------------------------------------------------------------

@dataclass
class BubblewrapBackend:
    """Bubblewrap (bwrap) sandbox with Linux namespace isolation.

    Security features:
    - Read-only bind of host filesystem
    - Isolated /tmp
    - PID namespace isolation
    - Process dies with parent
    - Optional network isolation (disabled by default since jobs
      need to reach LLM APIs and the eventstore)

    Requires: ``bwrap`` binary on PATH.
    """

    unshare_net: bool = False
    extra_ro_binds: list[str] = field(default_factory=list)
    extra_rw_binds: list[str] = field(default_factory=list)

    async def launch(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> asyncio.subprocess.Process:
        bwrap_cmd = [
            "bwrap",
            # Read-only bind the entire host filesystem
            "--ro-bind", "/", "/",
            # Writable /tmp for scratch
            "--tmpfs", "/tmp",
            # Writable work directory
            "--bind", str(cwd), str(cwd),
            # PID namespace — child sees itself as PID 1
            "--unshare-pid",
            # New session — clean signal handling
            "--new-session",
            # Die when parent exits — no orphan processes
            "--die-with-parent",
            # Mount a fresh /proc for the PID namespace
            "--proc", "/proc",
            # Writable /dev/shm for shared memory
            "--dev", "/dev",
        ]

        if self.unshare_net:
            bwrap_cmd.append("--unshare-net")

        for path in self.extra_ro_binds:
            bwrap_cmd += ["--ro-bind", path, path]

        for path in self.extra_rw_binds:
            bwrap_cmd += ["--bind", path, path]

        # HOME may need to be writable for git/pip caches
        home = env.get("HOME", "/tmp")
        if home != "/tmp":
            bwrap_cmd += ["--bind", home, home]

        bwrap_cmd += command

        return await asyncio.create_subprocess_exec(
            *bwrap_cmd,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {
    "subprocess": SubprocessBackend,
    "bubblewrap": BubblewrapBackend,
}


def create_backend(name: str, **kwargs) -> IsolationBackend:
    """Instantiate an isolation backend by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown isolation backend {name!r}. "
            f"Available: {', '.join(_BACKENDS)}"
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Job launcher (uses a backend)
# ---------------------------------------------------------------------------

def _build_job_env(eventstore_api_key_env: str) -> dict[str, str]:
    """Build minimal environment for a job subprocess."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        eventstore_api_key_env: os.environ.get(eventstore_api_key_env, ""),
    }
    for key in ("ANTHROPIC_API_KEY", "GIT_TOKEN"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


async def launch_job(
    *,
    backend: IsolationBackend,
    job_id: str,
    task: str,
    role: str,
    repo: str,
    branch: str,
    evo_sha: str | None,
    palimpsest_command: str,
    work_dir: Path,
    eventstore_url: str,
    eventstore_api_key_env: str,
    eventstore_source: str,
    llm_defaults: dict,
    workspace_defaults: dict,
    publication_defaults: dict,
) -> JobProcess:
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "job_id": job_id,
        "task": task,
        "role": role,
        "workspace": {
            "repo": repo,
            "branch": branch,
            **workspace_defaults,
        },
        "llm": {**llm_defaults} if llm_defaults else {},
        "publication": {**publication_defaults} if publication_defaults else {},
        "eventstore": {
            "url": eventstore_url,
            "api_key_env": eventstore_api_key_env,
            "source_id": eventstore_source,
        },
    }

    config_path = job_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    env = _build_job_env(eventstore_api_key_env)
    command = [palimpsest_command, "run", str(config_path)]

    proc = await backend.launch(command, cwd=job_dir, env=env)

    return JobProcess(
        job_id=job_id,
        proc=proc,
        work_dir=job_dir,
        config_path=config_path,
    )
