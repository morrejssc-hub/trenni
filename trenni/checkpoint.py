from __future__ import annotations

import time

from .runtime_types import JobHandle

ACTIVE_CONTAINER_STATES = {"created", "configured", "running", "paused"}


async def mark_exited_jobs(jobs: dict[str, JobHandle], backend) -> None:
    now = time.monotonic()
    for handle in jobs.values():
        state = await backend.inspect(handle)
        if not state.exists:
            if handle.exited_at is None:
                handle.exited_at = now
            continue

        if state.running or state.status in ACTIVE_CONTAINER_STATES:
            continue

        if handle.exited_at is None:
            handle.exited_at = now
            handle.exit_code = state.exit_code


async def reap_timed_out_jobs(
    jobs: dict[str, JobHandle],
    *,
    backend,
    reap_timeout: float,
) -> list[tuple[JobHandle, str]]:
    now = time.monotonic()
    reaped: list[tuple[JobHandle, str]] = []

    for handle in list(jobs.values()):
        if handle.exited_at is None or (now - handle.exited_at) <= reap_timeout:
            continue
        logs = await backend.logs(handle)
        reaped.append((handle, logs))

    return reaped
