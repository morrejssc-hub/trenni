"""Process isolation for Palimpsest job subprocesses.

Three-layer architecture:
  Layer 1 (Supervisor) — decides which jobs to launch
  Layer 2 (This module) — prepares workspace & environment, launches in backend
  Layer 3 (Palimpsest) — agent logic, assumes local git credentials are ready

Pluggable backends: subprocess (default) and bubblewrap (bwrap).
"""
from __future__ import annotations

import asyncio
import base64
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
    exited_at: float | None = None


@dataclass
class JobWorkspace:
    """Prepared workspace for a job subprocess."""
    job_dir: Path
    config_path: Path
    env: dict[str, str]


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
# Layer 2: Workspace preparation (environment setup)
# ---------------------------------------------------------------------------

def _build_git_credential_env(git_token_env: str) -> dict[str, str]:
    """Build GIT_CONFIG_* env vars for authenticated HTTPS operations.

    This sets up git credentials at the environment level so that
    Palimpsest (Layer 3) can use plain git clone/push without
    handling authentication itself.
    """
    token = os.environ.get(git_token_env, "") if git_token_env else ""
    if not token:
        return {}

    auth_str = f"x-access-token:{token}"
    b64_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
    return {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "http.extraHeader",
        "GIT_CONFIG_VALUE_1": f"AUTHORIZATION: basic {b64_auth}",
    }


def _build_job_env(
    eventstore_api_key_env: str,
    env_keys: list[str],
) -> dict[str, str]:
    """Build minimal environment for a job subprocess.

    Only forwards explicitly requested environment variables.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        eventstore_api_key_env: os.environ.get(eventstore_api_key_env, ""),
    }
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def prepare_workspace(
    *,
    job_id: str,
    work_dir: Path,
    evo_repo_path: str,
    config: dict,
    eventstore_api_key_env: str,
    env_keys: list[str],
    git_token_env: str = "",
) -> JobWorkspace:
    """Prepare job working directory, config file, and environment.

    This is the Layer 2 responsibility: set up everything the Palimpsest
    process needs to run, including git credentials and evo symlink.
    """
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Palimpsest looks for evo/ relative to cwd — symlink it in.
    evo_link = job_dir / "evo"
    if not evo_link.exists():
        evo_link.symlink_to(Path(evo_repo_path).resolve())

    # Write Palimpsest job config
    config_path = job_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    # Build environment: base + forwarded keys + git credentials
    env = _build_job_env(eventstore_api_key_env, env_keys)
    git_env = _build_git_credential_env(git_token_env)
    env.update(git_env)

    return JobWorkspace(job_dir=job_dir, config_path=config_path, env=env)


async def launch_in_backend(
    backend: IsolationBackend,
    workspace: JobWorkspace,
    command: list[str],
    job_id: str,
) -> JobProcess:
    """Launch a prepared job in the isolation backend."""
    proc = await backend.launch(command, cwd=workspace.job_dir, env=workspace.env)
    return JobProcess(
        job_id=job_id,
        proc=proc,
        work_dir=workspace.job_dir,
        config_path=workspace.config_path,
    )


# ---------------------------------------------------------------------------
# Top-level launch interface (called by Supervisor / Layer 1)
# ---------------------------------------------------------------------------

async def launch_job(
    *,
    backend: IsolationBackend,
    job_id: str,
    task: str,
    role: str,
    repo: str,
    init_branch: str,
    evo_sha: str | None,
    evo_repo_path: str,
    palimpsest_command: str,
    work_dir: Path,
    eventstore_url: str,
    eventstore_api_key_env: str,
    eventstore_source: str,
    llm_defaults: dict,
    workspace_defaults: dict,
    publication_defaults: dict,
) -> JobProcess:
    """Supervisor-facing interface: prepare workspace + launch in backend."""

    # Build Palimpsest job config
    config = {
        "job_id": job_id,
        "task": task,
        "role": role,
        "workspace": {
            "repo": repo,
            "init_branch": init_branch,
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

    # Collect env var keys to forward
    env_keys: list[str] = []
    llm_key_env = llm_defaults.get("api_key_env") if llm_defaults else None
    if llm_key_env:
        env_keys.append(llm_key_env)

    git_token_env = workspace_defaults.get("git_token_env", "") if workspace_defaults else ""

    ws = prepare_workspace(
        job_id=job_id,
        work_dir=work_dir,
        evo_repo_path=evo_repo_path,
        config=config,
        eventstore_api_key_env=eventstore_api_key_env,
        env_keys=env_keys,
        git_token_env=git_token_env,
    )

    command = [palimpsest_command, "run", str(ws.config_path)]
    return await launch_in_backend(backend, ws, command, job_id)
